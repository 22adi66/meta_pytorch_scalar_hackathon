"""
trainer_daemon.py

Runs GRPO training in a background thread inside the FastAPI container,
using the A10G GPU allocated to the HF Space.

Stack: HuggingFace transformers + PEFT + TRL (no unsloth, no bitsandbytes —
avoids CUDA 12/13 library conflicts). Qwen2.5-3B in bf16 fits comfortably in
the A10G's 24 GB with full room for GRPO rollouts.

Usage (via API):
  POST /train/start   — launch training
  GET  /train/status  — live progress
  POST /train/stop    — graceful stop signal
"""

import os
import re
import time
import threading
import traceback
from typing import Any, Dict, List, Optional

# ── Shared state ──────────────────────────────────────────────────────────────

_lock          = threading.Lock()
_stop_flag     = threading.Event()
_training_thread: Optional[threading.Thread] = None

_state: Dict[str, Any] = {
    "status":          "idle",
    "step":            0,
    "max_steps":       0,
    "current_reward":  0.0,
    "best_reward":     0.0,
    "reward_history":  [],
    "log":             [],
    "model_repo":      None,
    "error":           None,
    "start_time":      None,
    "elapsed_seconds": 0,
    "gpu_name":        None,
}


# ── Public helpers ────────────────────────────────────────────────────────────

def get_status() -> Dict[str, Any]:
    with _lock:
        return dict(_state)

def request_stop():
    _stop_flag.set()

def is_running() -> bool:
    with _lock:
        return _state["status"] in ("starting", "running")


# ── Private helpers ───────────────────────────────────────────────────────────

def _log(msg: str):
    print(f"[trainer] {msg}", flush=True)
    with _lock:
        _state["log"].append(msg)
        if len(_state["log"]) > 200:
            _state["log"] = _state["log"][-200:]

def _set(**kwargs):
    with _lock:
        _state.update(kwargs)


# ── Training thread ───────────────────────────────────────────────────────────

def _run_training(
    max_steps:  int,
    model_name: str,
    hf_token:   str,
    hf_repo:    str,
    workspace:  str,
    phase:      int,
):
    _stop_flag.clear()
    _set(status="starting", start_time=time.time(), step=0,
         reward_history=[], current_reward=0.0, best_reward=0.0,
         error=None, model_repo=None, max_steps=max_steps)

    try:
        import torch

        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            _log(f"GPU: {gpu}  |  VRAM: {vram:.1f} GB")
        else:
            gpu = "CPU"
            _log("WARNING: No GPU detected — training will be slow!")
        _set(gpu_name=gpu)

        # ── 1. Load model in bf16 (no bitsandbytes needed on A10G) ────────
        _log(f"Loading {model_name} in bf16...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype   = dtype,
            device_map    = "auto",
            trust_remote_code = True,
        )
        model.enable_input_require_grads()

        lora_cfg = LoraConfig(
            r              = 16,
            lora_alpha     = 32,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_dropout   = 0.05,
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        _log("Model + LoRA adapters ready.")

        # ── 2. Build dataset ───────────────────────────────────────────────
        from datasets import Dataset
        from .env import MigrationEnv

        SYSTEM_PROMPT = (
            "You are an expert Rust systems programmer specialising in C-to-Rust migration. "
            "Translate the given C source to memory-safe, idiomatic Rust. "
            "NEVER use `unsafe`. Keep external crate imports below 3. "
            "Output ONLY the complete .rs file — no markdown, no explanations."
        )

        def build_prompt(obs: Dict) -> str:
            constraints = "\n".join(f"  - {c}" for c in obs.get("constraints", []))
            dep_ctx     = obs.get("dependency_context", {})
            dep_str     = "\n".join(f"// {f}:\n{s}" for f, s in dep_ctx.items()) or "// (none yet)"
            errors      = obs.get("recent_errors", [])
            err_str     = "\n".join(
                f'  [{e.get("level","?")}] {e.get("message","?")}'
                for e in errors
            ) or "  None"
            user_msg = (
                f"## Constraints\n{constraints}\n\n"
                f"## C Source: {obs.get('current_target','?')}\n```c\n{obs.get('c_source_code','')}\n```\n\n"
                f"## Already-Translated Context\n{dep_str}\n\n"
                f"## Recent Compiler Errors\n{err_str}\n\n"
                f"Provide the complete Rust source file:"
            )
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )

        constraint_sets = [
            ["Do not use the unsafe keyword", "Maintain a CBO score below 3"],
            ["Do not use the unsafe keyword"],
            ["Maintain a CBO score below 3"],
            ["Do not use the unsafe keyword", "Maintain a CBO score below 3"],
        ] * 3  # 12 prompts

        env_builder = MigrationEnv(workspace_dir=workspace)
        prompts: List[str] = []
        for c_set in constraint_sets:
            try:
                obs = env_builder.reset(constraints=c_set, phase=phase)
                if obs.get("current_target"):
                    prompts.append(build_prompt(obs))
            except Exception:
                pass

        if not prompts:
            raise RuntimeError("Could not build any training prompts from the environment.")

        dataset = Dataset.from_dict({"prompt": prompts})
        _log(f"Dataset: {len(dataset)} prompts  |  Phase: {phase}")

        # ── 3. Reward function ─────────────────────────────────────────────
        def reward_func(prompts=None, completions=None, **kwargs):
            rewards: List[float] = []
            reward_env = MigrationEnv(workspace_dir=workspace)
            for completion in completions:
                if _stop_flag.is_set():
                    rewards.append(0.0)
                    continue
                try:
                    obs2    = reward_env.reset(phase=1)
                    target  = obs2.get("current_target", "math_ops.c")
                    rs_path = "src/" + re.sub(r"\.c$", ".rs", os.path.basename(target))
                    result  = reward_env.step({"file_path": rs_path, "code_content": completion})
                    rewards.append(result["reward"])
                except Exception:
                    rewards.append(0.01)

            avg = sum(rewards) / max(len(rewards), 1)
            with _lock:
                _state["current_reward"] = round(avg, 4)
                _state["reward_history"].append(round(avg, 4))
                if avg > _state["best_reward"]:
                    _state["best_reward"] = round(avg, 4)
                _state["elapsed_seconds"] = round(time.time() - _state["start_time"], 1)
            return rewards

        # ── 4. Progress callback ───────────────────────────────────────────
        from transformers import TrainerCallback

        class _ProgressCB(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                with _lock:
                    _state["step"]            = state.global_step
                    _state["status"]          = "running"
                    _state["elapsed_seconds"] = round(time.time() - _state["start_time"], 1)
                if state.global_step % 10 == 0:
                    _log(
                        f"Step {state.global_step}/{max_steps}  "
                        f"reward={_state['current_reward']:.4f}  "
                        f"best={_state['best_reward']:.4f}"
                    )
                if _stop_flag.is_set():
                    control.should_training_stop = True

        # ── 5. GRPOConfig + Trainer ────────────────────────────────────────
        from trl import GRPOTrainer, GRPOConfig

        IS_A10G = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 20e9
        batch   = 2 if IS_A10G else 1
        _log(f"GRPO config: batch={batch}, steps={max_steps}, generations=4, dtype={'bf16' if dtype==torch.bfloat16 else 'fp16'}")

        training_args = GRPOConfig(
            output_dir                  = "/app/crust_outputs",
            learning_rate               = 2e-5,
            per_device_train_batch_size = batch,
            gradient_accumulation_steps = 4,
            max_steps                   = max_steps,
            logging_steps               = 5,
            save_steps                  = 50,
            warmup_ratio                = 0.05,
            lr_scheduler_type           = "cosine",
            num_generations             = 4,
            max_completion_length       = 512,
            report_to                   = "none",
            seed                        = 42,
            bf16                        = (dtype == torch.bfloat16),
            fp16                        = (dtype == torch.float16),
        )

        _set(status="running")
        _log(f"Training started on {gpu}!")

        trainer = GRPOTrainer(
            model            = model,
            processing_class = tokenizer,
            reward_funcs     = reward_func,
            args             = training_args,
            train_dataset    = dataset,
            callbacks        = [_ProgressCB()],
        )

        trainer.train()

        # ── 6. Save + push ─────────────────────────────────────────────────
        _log("Saving LoRA adapters to /app/crust_lora ...")
        model.save_pretrained("/app/crust_lora")
        tokenizer.save_pretrained("/app/crust_lora")

        if hf_token and hf_token not in ("", "YOUR_HF_TOKEN_HERE"):
            from huggingface_hub import login as hf_login
            hf_login(token=hf_token, add_to_git_credential=False)
            _log(f"Pushing to https://huggingface.co/{hf_repo} ...")
            model.push_to_hub(hf_repo, token=hf_token)
            tokenizer.push_to_hub(hf_repo, token=hf_token)
            _set(model_repo=f"https://huggingface.co/{hf_repo}")
            _log(f"Model live at https://huggingface.co/{hf_repo}")

        _set(status="complete",
             elapsed_seconds=round(time.time() - _state["start_time"], 1))
        _log("Training complete!")

    except Exception:
        err = traceback.format_exc()
        _log(f"FATAL ERROR:\n{err}")
        _set(status="error", error=err,
             elapsed_seconds=round(time.time() - (_state.get("start_time") or time.time()), 1))


# ── Entry point ───────────────────────────────────────────────────────────────

def start_training(
    max_steps:  int  = 100,
    model_name: str  = "Qwen/Qwen2.5-3B-Instruct",
    hf_token:   str  = "",
    hf_repo:    str  = "crust-grpo-qwen25-3b",
    workspace:  str  = "",
    phase:      int  = 1,
) -> tuple:
    global _training_thread

    if is_running():
        return False, "Training already in progress. POST /train/stop first."

    if not workspace:
        workspace = os.getenv("CRUST_WORKSPACE",
                              os.path.join(os.path.dirname(__file__), "dummy_workspace"))

    _training_thread = threading.Thread(
        target = _run_training,
        args   = (max_steps, model_name, hf_token, hf_repo, workspace, phase),
        daemon = True,
        name   = "crust-grpo-trainer",
    )
    _training_thread.start()
    return True, "Training thread launched."
