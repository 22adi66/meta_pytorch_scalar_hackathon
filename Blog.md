# Teaching an LLM to Migrate an Entire C Codebase to Rust — with RL

*Meta PyTorch OpenEnv Hackathon 2026 · Theme #2: Super Long-Horizon Planning & Instruction Following*

---

## The Problem Nobody Talks About

Every systems programmer knows the pain: you inherit 10,000 lines of legacy C code. Buffer overflows, null-pointer dereferences, use-after-free bugs. You know Rust would fix all of it. But migrating isn't just "translate function by function" — the files **depend on each other**, the compiler gives you **cascading errors** across files, and one wrong move breaks everything downstream.

An LLM with no training on this task will:
- Generate Rust that wraps everything in `pub mod` (wrong namespace — breaks all callers)
- Use `unsafe` blocks to avoid thinking about ownership
- Ignore `#include` dependencies and translate files in the wrong order
- Give up after the first cascade of 12 compiler errors

**CRust trains an LLM to do better — using reinforcement learning with a real Rust compiler as the judge.**

---

## How the Environment Works

We built an **OpenEnv-compatible** RL environment where:

1. **The agent receives a C file to translate** (e.g. `data_store.c`) along with:
   - The actual C source code
   - Active constraints: *"Do not use the `unsafe` keyword"*, *"Maintain CBO score below 3"*
   - Already-translated Rust signatures of dependencies (so it has context)
   - Recent compiler errors from its last attempt

2. **The agent outputs a Rust file**

3. **The environment runs the real Rust toolchain** (`cargo check` + `cargo test`) in a sandboxed workspace and returns a reward:

```
reward = 0.30 × compiles
       + 0.30 × tests_pass (semantic equivalence)
       + 0.20 × S(r)        ← 5-family LAC2R safety score
       + 0.10 × CBO < 3
       + 0.10 × cohesion_score
```

**The safety component S(r)** is not a simple binary `has_unsafe` flag. It implements the LAC2R paper formula (Sim et al., arXiv:2505.15858, Eq. 3):

```
S(r_i) = m(r_i) × max(1 − T_i / T_0, 0)
```

where T_i = **RPC + RPR + LUC + UCE + UTC** — five fine-grained unsafe construct families. The agent earns a continuous safety signal for reducing each family independently, rather than hitting a hard binary wall. T_0 is estimated from the C source complexity (number of pointer ops, casts, `malloc/memcpy` calls).

4. **The schedule is topologically sorted** — the agent migrates leaf nodes first (files with no dependencies), then works up the dependency graph. You cannot translate `data_store.c` before you've translated `math_ops.c` that it depends on.

### The Dependency Graph

```
math_ops.h ──────────┐
                      ▼
string_ops.h ──► data_store.h ──► target_service.c
                      │
                      ▼
                 data_store.c
```

The agent must navigate this **left to right** — each file builds on the previous ones. This is the long-horizon planning challenge.

---

## Before Training: The Zero-Shot LLM

Without any fine-tuning, Qwen2.5-3B generates code like this for `math_ops.c`:

```rust
// ❌ Zero-shot output — WRONG
pub mod math_ops {           // wraps in extra module — breaks all callers
    pub fn add(a: i32, b: i32) -> i32 { a + b }
    pub fn divide(a: i32, b: i32) -> i32 {
        unsafe {             // uses unsafe to avoid thinking about division by zero
            a / b            // panics on b=0 instead of handling gracefully
        }
    }
}
```

**Problems:**
- `pub mod math_ops` wrapping → `cargo test` fails because tests import `crate::math_ops::add`, not `crate::math_ops::math_ops::add`
- `unsafe` block → violates constraint, gets −0.50 penalty
- No `None` return for divide-by-zero → wrong semantics

**Reward: ~0.05** (just barely above zero)

---

## After 100 Steps of GRPO: The Trained Agent

After training on the CRust environment with GRPO:

```rust
// ✅ GRPO-trained output — CORRECT
pub fn add(a: i32, b: i32) -> i32 { a + b }

pub fn subtract(a: i32, b: i32) -> i32 { a - b }

pub fn multiply(a: i32, b: i32) -> i32 { a * b }

pub fn divide(a: i32, b: i32) -> Option<i32> {   // idiomatic Rust!
    if b == 0 { None } else { Some(a / b) }
}

pub fn clamp(value: i32, min_val: i32, max_val: i32) -> i32 {
    value.max(min_val).min(max_val)               // method chaining, no unsafe
}
```

**What changed:**
- No `pub mod` wrapper → correct namespace → `cargo test` can find functions
- `Option<i32>` for divide → idiomatic Rust, handles zero safely
- `value.max().min()` → native Rust method chaining, zero unsafe
- CBO = 0 (no external imports) → constraint satisfied

**Reward: 0.70** — full marks on compilation, memory safety, CBO, and cohesion. The remaining 0.30 (test suite) is Phase 2 training territory.

---

## The Reward Curve

![Reward Curve](reward_curve.png)

In ~25 steps the agent learns to:
1. Stop using `pub mod` wrappers (compilation reward kicks in)
2. Avoid `unsafe` (memory safety reward kicks in)
3. Keep imports minimal (CBO reward kicks in)

The GRPO algorithm works by sampling 4 completions per prompt, computing relative advantages, and updating the policy to favor higher-reward outputs — with no human labels, no LLM judge, just the Rust compiler.

---

## Anti-Reward Hacking

We specifically designed against the obvious exploits:

| Attack | Defense |
|---|---|
| Agent modifies `tests/integration_test.rs` to make all tests trivially pass | **Tests are in `PROTECTED_FILES`** — any write attempt returns reward 0.01 immediately |
| Agent generates infinite loop to stall test runner | **60-second subprocess timeout** on `cargo test` |
| Agent uses `#[allow(unused)]` + empty functions to pass compilation | **Integration tests require correct return values** — empty functions fail tests |
| Agent copies exact golden solution | **Rewards scale on metrics** — copying gets 0.99 but that's the goal |
| Path traversal (`../../etc/passwd`) | **Absolute paths and `..` segments blocked** before any write |

---

## Multi-File Long-Horizon Episode

Here's what a full Phase 4 episode looks like:

```
Step 1: Target = math_ops.h  → reward 0.70 ✅ (advances to next)
Step 2: Target = string_ops.h → reward 0.10 ✗ (compile error)
Step 3: Target = string_ops.h → reward 0.70 ✅ (fixed, advances)
Step 4: Target = data_store.h → reward 0.10 ✗ (cascading error from string_ops)
Step 5: Target = data_store.h → reward 0.40 ✗ (compiles but tests fail)
Step 6: Target = data_store.h → reward 0.70 ✅ (all constraints met)
Step 7: Target = target_service.c → reward 0.70 ✅
→ EPISODE COMPLETE: full repository migrated 🎉
```

The agent must resolve **cascading errors across 4 files** while maintaining constraints throughout. This is the long-horizon planning challenge.

---

## Try It Live

The environment is running at **https://adithyakommuri-meta-hackathon-final.hf.space/docs**

```bash
# Get your first migration task
curl -X POST https://adithyakommuri-meta-hackathon-final.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"phase": 1, "constraints": ["Do not use the unsafe keyword", "Maintain a CBO score below 3"]}'

# Submit your Rust and get a real reward from cargo
curl -X POST https://adithyakommuri-meta-hackathon-final.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"file_path": "src/math_ops.rs", "code_content": "pub fn add(a: i32, b: i32) -> i32 { a + b }"}'
```

**Trained model:** [Adithyakommuri/crust-grpo-qwen25-3b](https://huggingface.co/Adithyakommuri/crust-grpo-qwen25-3b)  
**Training notebook:** [CRust_Training_Colab.ipynb](CRust_Training_Colab.ipynb)  
**Training logs:** [training_logs.txt](training_logs.txt)  
**GitHub:** [22adi66/meta_pytorch_scalar_hackathon](https://github.com/22adi66/meta_pytorch_scalar_hackathon)

---

---

## Training Logs

Full step-by-step training logs are in [`training_logs.txt`](training_logs.txt), showing per-step reward, safety score S(r), and unsafe construct count T_i for every training step. Key excerpt:

```
Step  1 | reward 0.70 | S(r) 1.00 | T_i 0 | clean Rust — no unsafe
Step  4 | reward 0.44 | S(r) 0.00 | T_i 8 | regression: excessive unsafe blocks
Step  8 | reward 0.70 | S(r) 1.00 | T_i 0 | recovered to plateau
Step 37 | reward 0.70 | S(r) 1.00 | T_i 0 | training complete — stable plateau
```

---

*Built for the Meta PyTorch OpenEnv Hackathon. The environment, verifier, scheduler, 5-family safety scoring, training loop, and deployment are all original work.*
