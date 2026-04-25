---
title: CRust Migration OpenEnv
emoji: 🦀
colorFrom: orange
colorTo: red
sdk: docker
app_port: 8000
pinned: true
tags:
  - reinforcement-learning
  - openenv
  - rust
  - llm
  - code-migration
---

# CRust — C-to-Rust Repository Migration RL Environment

> **Meta OpenEnv Hackathon | Theme #2: Super Long-Horizon Planning & Instruction Following**

An OpenEnv-compatible RL environment that trains LLMs to **migrate legacy C codebases to memory-safe, modular Rust** through dependency-aware topological scheduling, cascading error resolution, and multi-objective verifiable rewards.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit Rust translation, get reward |
| GET | `/state` | Full internal state |
| GET | `/observation` | Agent-visible partial observation |
| GET | `/health` | Health check |
| GET | `/info` | Environment metadata |
| GET | `/docs` | Interactive Swagger UI |

## Quick Start

```python
import requests

BASE = "https://adithyakommuri-meta-hackathon-final.hf.space"

# Reset environment (Phase 1: leaf node)
obs = requests.post(f"{BASE}/reset", json={"phase": 1, "constraints": ["Do not use the unsafe keyword"]}).json()
print(obs["current_target"])   # math_ops.c
print(obs["c_source_code"])    # C code to translate

# Submit Rust translation
result = requests.post(f"{BASE}/step", json={
    "file_path": "src/math_ops.rs",
    "code_content": "pub fn add(a: i32, b: i32) -> i32 { a + b }"
}).json()
print(result["reward"])  # 0.01–0.99
```
