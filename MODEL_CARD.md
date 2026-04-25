---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- reinforcement-learning
- grpo
- code-generation
- rust
- c-to-rust
- openenv
- meta-hackathon
pipeline_tag: text-generation
---

# CRust GRPO — C-to-Rust Migration RL Agent

Fine-tuned **Qwen/Qwen2.5-3B-Instruct** via **GRPO** (Group Relative Policy Optimization) to migrate legacy C code to memory-safe, idiomatic Rust.

Trained as part of the **Meta OpenEnv Hackathon — Theme #2: Long-Horizon Planning**.

## Training

| Setting | Value |
|---|---|
| Algorithm | GRPO (TRL) — fully programmatic reward, no LLM judge |
| Hardware | NVIDIA A10G 24 GB on Hugging Face Spaces |
| Steps | 100 steps, batch=2, 4 generations/prompt |
| Best reward | **0.70 / 1.00** |
| LoRA rank | r=16, all projection layers, bf16 |

## Reward breakdown at convergence

| Component | Weight | Result |
|---|---|---|
| Compilation (`cargo check`) | 0.30 | Full marks |
| Memory safety (zero `unsafe`) | 0.20 | Full marks |
| CBO < 3 (coupling) | 0.10 | Full marks |
| LCOM cohesion | 0.10 | Full marks |
| Semantic tests (`cargo test`) | 0.30 | Phase 2 target |

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "Qwen/Qwen2.5-3B-Instruct"
LORA = "Adithyakommuri/crust-grpo-qwen25-3b"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, LORA)
model.eval()

SYSTEM = (
    "You are an expert Rust programmer. "
    "Translate the given C source to memory-safe, idiomatic Rust. "
    "NEVER use unsafe. Output ONLY the .rs file."
)
C_CODE = """
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }
int divide(int a, int b) { return b == 0 ? -1 : a / b; }
int clamp(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
"""

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": f"Translate this C to Rust:\n\n```c\n{C_CODE}\n```"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=400, temperature=0.7, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
rust_code = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(rust_code)
```

## Score generated code against the live OpenEnv

```bash
# Get a task
curl -X POST https://adithyakommuri-meta-hackathon-final.hf.space/reset \
  -H "Content-Type: application/json" -d '{"phase":1}'

# Submit generated Rust and receive reward (0-1)
curl -X POST https://adithyakommuri-meta-hackathon-final.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"file_path":"src/math_ops.rs","code_content":"<your rust here>"}'
```

Live Swagger UI: **https://adithyakommuri-meta-hackathon-final.hf.space/docs**
