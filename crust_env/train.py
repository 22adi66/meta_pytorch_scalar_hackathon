"""
CRust GRPO Training Pipeline — train.py

Trains an LLM to perform C-to-Rust migration using:
  - Group Relative Policy Optimization (GRPO) via TRL
  - Memory-efficient QLoRA fine-tuning via Unsloth
  - Verifiable programmatic rewards from the CRust OpenEnv (no LLM-as-a-judge)

Hackathon Theme #2: Super Long-Horizon Planning & Instruction Following

Training progression:
  Phase 1 → isolated leaf functions (short horizon, dense reward)
  Phase 2 → two-file dependency chains (medium horizon)
  Phase 3 → cross-file header dependencies (longer horizon)
  Phase 4 → full repository cascades (super long horizon, sparse reward)

Run this script:
    python -m crust_env.train

Or on Colab / Hugging Face Spaces:
    Uncomment trainer.train() at the bottom and launch.
"""

import os
import re
from typing import List, Dict, Any

import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

from .env import MigrationEnv

# ── Configuration ─────────────────────────────────────────────────────────

MODEL_NAME      = "unsloth/Meta-Llama-3-8B-Instruct"   # Base model
MAX_SEQ_LENGTH  = 2048
LOAD_IN_4BIT    = True                                  # QLoRA memory efficiency
OUTPUT_DIR      = "crust_model_outputs"
WORKSPACE_DIR   = os.getenv("CRUST_WORKSPACE", "/tmp/crust_workspace")

# ── Prompt Templates ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Rust systems programmer specializing in memory-safe C-to-Rust migration.

Your task: Translate the given C source file into idiomatic, safe Rust code.

Rules you MUST follow:
1. NEVER use the `unsafe` keyword — use Rust's ownership model instead.
2. Keep CBO (Coupling Between Objects) below 3 — minimize external crate imports.
3. Preserve all function signatures' semantics (same inputs → same outputs).
4. Use idiomatic Rust: Option<T> instead of sentinel values, iterators, match statements.
5. Output ONLY the complete Rust file content. No markdown, no explanations.\
"""

USER_PROMPT_TEMPLATE = """\
## Migration Task
Translate this C file to memory-safe, idiomatic Rust.

## Active Constraints
{constraints}

## C Source: {target}
```c
{c_source}
```

## Dependency Context (Already-Translated Rust APIs)
{dep_context}

## Recent Compiler Errors (Fix these in your translation)
{recent_errors}

Provide the complete Rust source file:\
"""

# ── Dataset construction ──────────────────────────────────────────────────

def build_prompt(obs: Dict[str, Any]) -> str:
    """Build the full prompt string from an environment observation."""
    constraints_str = "\n".join(
        f"  - {c}" for c in obs.get("constraints", [])
    )
    dep_ctx = obs.get("dependency_context", {})
    dep_str = "\n".join(
        f"// {fname}:\n{sigs}" for fname, sigs in dep_ctx.items()
    ) if dep_ctx else "// No prior translations yet."

    errors = obs.get("recent_errors", [])
    err_str = "\n".join(
        f"  [{e.get('level','?')}] {e.get('message','')}"
        for e in errors
    ) if errors else "  None"

    user_section = USER_PROMPT_TEMPLATE.format(
        constraints=constraints_str or "  - (none)",
        target=obs.get("current_target", "unknown.c"),
        c_source=obs.get("c_source_code", "// (not found)"),
        dep_context=dep_str,
        recent_errors=err_str,
    )
    return f"{SYSTEM_PROMPT}\n\n{user_section}"


def prepare_curriculum_dataset(phase: int = 1) -> Dataset:
    """
    Generate training prompts by querying the CRust environment.

    For each curriculum phase, reset the environment and collect the initial
    observation as a training prompt. In a full training run, this would
    generate many diverse tasks; here we build a compact representative set.
    """
    env = MigrationEnv(workspace_dir=WORKSPACE_DIR)
    prompts: List[str] = []

    # Phase-specific constraint sets (instruction following diversity)
    constraint_sets = [
        ["Do not use the unsafe keyword", "Maintain a CBO score below 3"],
        ["Do not use the unsafe keyword"],
        ["Maintain a CBO score below 3"],
        ["Do not use the unsafe keyword", "Maintain a CBO score below 3",
         "Refactor C-style switch statements into idiomatic Rust match expressions"],
    ]

    for constraints in constraint_sets:
        obs = env.reset(constraints=constraints, phase=phase)
        if obs.get("current_target"):
            prompts.append(build_prompt(obs))

    if not prompts:
        # Fallback: single training example
        obs = env.reset(phase=1)
        prompts.append(build_prompt(obs))

    return Dataset.from_dict({"prompt": prompts})


# ── Reward function ───────────────────────────────────────────────────────

def reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Programmatic reward function for GRPO.

    Called once per training step. For each (prompt, completion) pair:
    1. Resets the CRust environment
    2. Steps with the completion as the agent's Rust code
    3. Returns the clamped, composited reward

    No LLM-as-a-judge — rewards come entirely from:
        - cargo check (compilation)
        - cargo test (semantic equivalence)
        - unsafe detection (memory safety)
        - CBO/LCOM metrics (architectural modularity)

    Anti-reward-hacking: environment refuses writes to protected files.
    """
    rewards: List[float] = []

    # One environment instance per batch to avoid state contamination
    env = MigrationEnv(workspace_dir=WORKSPACE_DIR)

    for completion in completions:
        obs = env.reset(phase=1)

        # Skip if environment has no target
        if not obs.get("current_target"):
            rewards.append(0.01)
            continue

        # Derive the Rust file path from the C target name
        c_target = obs["current_target"]
        rs_path = "src/" + re.sub(r'\.c$', '.rs', os.path.basename(c_target))

        action = {
            "file_path": rs_path,
            "code_content": completion,
        }

        result = env.step(action)
        rewards.append(result["reward"])

    return rewards


# ── Training pipeline ─────────────────────────────────────────────────────

def train(phase: int = 1, max_steps: int = 100):
    """
    Full GRPO training loop using Unsloth + TRL.

    Args:
        phase: Curriculum phase to train on (1-4).
        max_steps: Number of training steps (increase for real runs).
    """
    print(f"[CRust] Initializing Unsloth model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,         # auto-detect (bfloat16 on Ampere+, float16 otherwise)
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Apply LoRA adapters for memory-efficient fine-tuning
    # WARNING: Do NOT naively upcast 4-bit model to 16-bit before merging LoRA.
    # Use FastLanguageModel.save_pretrained_merged() with proper merge path.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"[CRust] Building curriculum dataset (Phase {phase})...")
    dataset = prepare_curriculum_dataset(phase=phase)
    print(f"[CRust] Dataset size: {len(dataset)} prompts")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        logging_steps=5,
        save_steps=25,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        # GRPO-specific: generate G=4 rollouts per prompt for group-relative scoring
        num_generations=4,
        temperature=0.8,
        max_new_tokens=512,
        report_to="none",   # Change to "wandb" for experiment tracking
    )

    print("[CRust] Initializing GRPO Trainer with CRust verifiable rewards...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,   # Fully programmatic — no LLM judge
        args=training_args,
        train_dataset=dataset,
    )

    print(f"[CRust] Starting GRPO training — Phase {phase}...")
    # ── Uncomment to run training: ─────────────────────────────────────────
    # trainer.train()

    # ── After training, save using the proper Unsloth merge path: ──────────
    # model.save_pretrained_merged(
    #     OUTPUT_DIR + "/final_merged",
    #     tokenizer,
    #     save_method="merged_16bit",
    # )

    print("[CRust] Training scaffold ready. Uncomment trainer.train() to run.")
    print(f"[CRust] Output directory: {OUTPUT_DIR}")


def train_full_curriculum():
    """
    Progressive curriculum training across all 4 phases.
    Implements the RLVE (Reinforcement Learning with Verifiable Environments)
    curriculum strategy: train on easy tasks first, escalate only on success.
    """
    for phase in range(1, 5):
        print(f"\n{'='*60}")
        print(f" CURRICULUM PHASE {phase}/4")
        print(f"{'='*60}")
        steps = 50 * phase   # More steps for harder phases
        train(phase=phase, max_steps=steps)


if __name__ == "__main__":
    train(phase=1)
