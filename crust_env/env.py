"""
CRust Migration Environment — env.py

Full Markov Decision Process (MDP) implementation for the OpenEnv hackathon.
Trains an LLM agent to migrate legacy C codebases to memory-safe, modular Rust
through dependency-aware topological scheduling and multi-objective verifiable rewards.

Aligned with Hackathon Theme #2: Super Long-Horizon Planning & Instruction Following.
"""

from typing import Dict, Any, List, Optional, Tuple
import uuid
import os
import re

from .verifier import CRustVerifier
from .scheduler import CDependencyGraph
from .metrics import ModularityMetrics
from .unsafe_constructs import compute_safety_score, estimate_baseline_T_from_c

# ── OpenEnv base class (pip install openenv-core) ─────────────────────────────
# We inherit from the official openenv.core Environment base class so that
# the framework's tooling (openenv validate, openenv push, client discovery)
# can identify this as a compliant OpenEnv environment.
try:
    from openenv.core.env_server.interfaces import Environment as _OpenEnvBase
except ImportError:
    # Fallback if openenv-core is not installed (e.g. local dev without pip install)
    class _OpenEnvBase:  # type: ignore
        """Minimal shim matching the openenv.core.env_server.interfaces.Environment API."""
        def reset(self, **kwargs): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        @property
        def state(self): raise NotImplementedError


class MigrationEnv(_OpenEnvBase):
    """
    CRust: C-to-Rust Repository Migration RL Environment.

    Core innovations implemented here:
    1. Dependency-aware topological scheduling (leaf-first translation order)
    2. Multi-objective verifiable rewards (compilation + tests + safety + modularity)
    3. Process supervision rewards (partial credit per compiler error cleared)
    4. Curriculum learning phases (Phase 1 → 4 escalating complexity)
    5. Multi-constraint instruction following (injected at reset())
    6. Anti-reward-hacking guardrails (unsafe penalty, CBO penalty, step limit)
    """

    DEFAULT_CONSTRAINTS: List[str] = [
        "Do not use the unsafe keyword",
        "Maintain a CBO score below 3",
    ]

    # ── Reward weights (sum to 1.0 at maximum) ────────────────────────────
    W_COMPILATION   = 0.30   # cargo check passes
    W_TESTS         = 0.30   # cargo test passes (semantic equivalence)
    W_MEMORY_SAFE   = 0.20   # no unsafe blocks (memory safety)
    W_CBO           = 0.10   # coupling between objects < 3
    W_COHESION      = 0.10   # LCOM cohesion score

    # ── Penalties ─────────────────────────────────────────────────────────
    P_UNSAFE_USED   = 0.50   # hard penalty for violating unsafe constraint
    P_HIGH_CBO      = 0.20   # penalty for violating CBO constraint

    # ── Process supervision ───────────────────────────────────────────────
    PROCESS_REWARD_PER_ERROR_CLEARED = 0.02   # reward each cleared compiler error

    def __init__(self, workspace_dir: str, legacy_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir
        self.legacy_dir = legacy_dir or os.path.normpath(
            os.path.join(workspace_dir, "..", "legacy_c")
        )
        self.session_id = str(uuid.uuid4())
        self.verifier = CRustVerifier(workspace_dir)

        # Internal state — uninitialized until reset()
        self._current_state: Dict[str, Any] = {"status": "uninitialized"}
        self._constraints: List[str] = []
        self._step_count: int = 0
        self._max_steps: int = 200
        self._phase: int = 1

        self._schedule: List[str] = []        # topologically sorted C file names
        self._current_idx: int = 0            # pointer into schedule
        self._translated: Dict[str, str] = {} # fname → rust code (already done)
        self._error_history: List[Dict] = []  # recent compiler diagnostics
        self._prev_error_count: int = 0       # for process supervision delta

    # ── OpenEnv Interface ─────────────────────────────────────────────────

    def reset(
        self,
        constraints: Optional[List[str]] = None,
        phase: int = 1,
    ) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.

        Args:
            constraints: Multi-constraint directives injected at episode start.
                         If None, uses defaults (no unsafe, CBO < 3).
            phase: Curriculum phase 1-4.
                   Phase 1 = isolated leaf node (simplest).
                   Phase 2 = two-file chain.
                   Phase 3 = cross-file with headers.
                   Phase 4 = full repository cascade (long-horizon).
        """
        self._step_count = 0
        self._phase = max(1, min(4, phase))
        self._constraints = list(constraints) if constraints else list(self.DEFAULT_CONSTRAINTS)
        self._translated = {}
        self._error_history = []
        self._prev_error_count = 0

        # Build dependency-aware topological schedule from legacy C source
        scheduler = CDependencyGraph(self.legacy_dir)
        full_schedule = scheduler.get_topological_schedule()

        # Curriculum gating: expose a controlled subset based on phase
        if self._phase == 1:
            self._schedule = full_schedule[:1]    # single leaf node
        elif self._phase == 2:
            self._schedule = full_schedule[:2]    # leaf + one dependent
        elif self._phase == 3:
            self._schedule = full_schedule[:3]    # partial DAG
        else:
            self._schedule = full_schedule        # full repository

        self._current_idx = 0

        self._current_state = {
            "status": "ready",
            "phase": self._phase,
            "schedule": self._schedule,
            "current_idx": self._current_idx,
            "files_total": len(self._schedule),
            "files_done": 0,
            "translated_files": [],
            "metrics": {"cbo": 0, "lcom": 0},
            "constraints": self._constraints,
            "error_history": [],
            "step_count": 0,
        }
        return self.observation()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply one agent action to the environment.

        Action format:
            {
                "file_path": "src/math_ops.rs",     # target Rust file path
                "code_content": "pub fn add(...)"   # generated Rust code
            }

        Returns:
            {
                "observation": {...},   # next observation
                "reward": float,        # clamped [0.01, 0.99]
                "done": bool,           # True when episode ends
                "info": {...}           # diagnostics, breakdown, metrics
            }
        """
        self._step_count += 1
        self._current_state["step_count"] = self._step_count

        # ── Hard episode termination ───────────────────────────────────────
        if self._step_count >= self._max_steps:
            return self._format_response(
                done=True, reward=0.01,
                info={"reason": "max_steps_exceeded", "step": self._step_count}
            )

        file_path = (action.get("file_path") or "").strip()
        code_content = (action.get("code_content") or "").strip()

        if not file_path or not code_content:
            return self._format_response(
                done=False, reward=0.01,
                info={"error": "Invalid action: file_path and code_content are required."}
            )

        # ── Step 1: Sandboxed compilation + test verification ─────────────
        verification = self.verifier.verify(action)

        # ── Step 2: Modularity metrics + 5-family safety score ────────────
        current_target = self._get_current_target()
        c_source = self._read_c_source(current_target) if current_target else ""
        compilable = verification.get("stage") in ("testing", "complete")
        metrics = ModularityMetrics.evaluate(code_content, c_source=c_source, compilable=compilable)
        self._current_state["metrics"] = metrics

        # ── Step 3: Compute multi-objective reward ─────────────────────────
        reward, breakdown = self._compute_reward(code_content, verification, metrics, c_source)

        # ── Step 4: Process supervision — reward clearing compiler errors ──
        current_errors = [
            d for d in verification.get("diagnostics", [])
            if d.get("level") == "error"
        ]
        errors_cleared = max(0, self._prev_error_count - len(current_errors))
        process_reward = errors_cleared * self.PROCESS_REWARD_PER_ERROR_CLEARED
        reward = min(0.99, reward + process_reward)
        self._prev_error_count = len(current_errors)
        self._error_history = verification.get("diagnostics", [])[-10:]

        # ── Step 5: Advance schedule on success ────────────────────────────
        success = verification.get("success", False)
        episode_done = False

        if success:
            self._translated[file_path] = code_content
            self._current_idx += 1
            self._current_state["files_done"] = self._current_idx
            self._current_state["translated_files"] = list(self._translated.keys())

            if self._current_idx >= len(self._schedule):
                episode_done = True   # 🎉 entire repository migrated

        self._current_state.update({
            "current_idx": self._current_idx,
            "error_history": self._error_history,
        })

        info = {
            "step": self._step_count,
            "verification": verification,
            "metrics": metrics,
            "reward_breakdown": breakdown,
            "process_reward": round(process_reward, 4),
            "errors_cleared": errors_cleared,
            "files_done": self._current_idx,
            "files_total": len(self._schedule),
        }

        return self._format_response(episode_done, reward, info)

    @property
    def state(self) -> Dict[str, Any]:
        """Full internal state (used by /state endpoint for debugging)."""
        return {
            **self._current_state,
            "session_id": self.session_id,
            "workspace_dir": self.workspace_dir,
        }

    def observation(self) -> Dict[str, Any]:
        """
        Partial observation returned to the agent.
        Includes: current C source, active constraints, compiler errors, dependency context.
        This is deliberately partial — the agent cannot see global state.
        """
        target = self._get_current_target()
        c_source = self._read_c_source(target) if target else ""
        dep_context = self._get_dependency_context()

        return {
            "current_target": target,
            "c_source_code": c_source,
            "constraints": self._constraints,
            "recent_errors": self._error_history[-5:],
            "dependency_context": dep_context,
            "phase": self._phase,
            "files_remaining": max(0, len(self._schedule) - self._current_idx),
            "step": self._step_count,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_current_target(self) -> Optional[str]:
        if self._current_idx < len(self._schedule):
            return self._schedule[self._current_idx]
        return None

    def _read_c_source(self, filename: str) -> str:
        """Locate and read the C source file from the legacy directory."""
        for root, _, files in os.walk(self.legacy_dir):
            if filename in files:
                try:
                    with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                        return f.read()
                except Exception:
                    pass
        return f"// Source not found: {filename}"

    def _get_dependency_context(self) -> Dict[str, str]:
        """
        Return public function signatures of already-translated Rust modules.
        Injects foundational type/API context into the next translation step.
        """
        context: Dict[str, str] = {}
        for fname, code in self._translated.items():
            sigs = re.findall(r'pub fn\s+\w+[^{]+', code)
            pub_types = re.findall(r'pub (?:struct|enum|type)\s+\w+[^{;]*', code)
            context[fname] = "\n".join(sigs + pub_types)
        return context

    def _compute_reward(
        self, code: str, verification: Dict, metrics: Dict, c_source: str = ""
    ) -> Tuple[float, Dict]:
        """
        Multi-objective reward computation.

        Components (all independently verifiable — no LLM judge):
        1. Compilation success (cargo check)
        2. Test pass rate (cargo test — semantic equivalence)
        3. Memory safety — 5-family LAC2R S(r) score (RPC+RPR+LUC+UCE+UTC)
        4. Low coupling (CBO < 3)
        5. High cohesion (LCOM → 0)

        The safety component now uses the LAC2R paper's Equation 3:
          S(r_i) = m(r_i) * max(1 - T_i / T_0, 0)
        where T_i = sum of all 5 unsafe construct families and T_0 is estimated
        from the complexity of the C source being translated.

        This is strictly superior to a binary has/doesn't-have-unsafe check:
          - Agent gets credit for reducing unsafe constructs progressively
          - Each unsafe family (RPC, RPR, LUC, UCE, UTC) contributes individually
          - Compile-gate (m) prevents gaming via compilable-but-unsafe submissions
        """
        reward = 0.0
        breakdown: Dict[str, float] = {}

        stage = verification.get("stage", "")
        success = verification.get("success", False)
        compiled = stage in ("testing", "complete")
        diag = verification.get("diagnostics") or []

        # ── 1. Compilation ─────────────────────────────────────────────────
        if compiled:
            reward += self.W_COMPILATION
            breakdown["compilation"] = self.W_COMPILATION
        else:
            error_count = sum(1 for d in diag if d.get("level") == "error")
            # Partial credit: attempt compiles with warnings only (no errors)
            if error_count == 0 and diag:
                partial = self.W_COMPILATION * 0.5
                reward += partial
                breakdown["compilation"] = partial
            else:
                breakdown["compilation"] = 0.0

        # ── 2. Tests (semantic equivalence) ───────────────────────────────
        if success:
            reward += self.W_TESTS
            breakdown["tests"] = self.W_TESTS
        elif stage == "complete":
            breakdown["tests"] = self.W_TESTS
        else:
            breakdown["tests"] = 0.0

        # ── 3. Memory safety — 5-family LAC2R S(r) (paper Eq. 3) ──────────
        # Compute baseline T_0 from the C source complexity
        baseline_T = estimate_baseline_T_from_c(c_source) if c_source else 1
        snapshot = compute_safety_score(code, baseline_T, compilable=compiled)
        S = snapshot.S   # [0, 1]

        unsafe_violated = any("unsafe" in c.lower() for c in self._constraints)
        has_any_unsafe = snapshot.total_unsafe > 0

        if has_any_unsafe and unsafe_violated:
            # Hard penalty: constraint explicitly forbids unsafe AND agent used it
            reward -= self.P_UNSAFE_USED
            breakdown["unsafe_penalty"] = -self.P_UNSAFE_USED
            breakdown["memory_safety"] = 0.0
        else:
            # Continuous safety reward: S ∈ [0, 1] scales the weight
            mem_reward = round(self.W_MEMORY_SAFE * S, 4)
            reward += mem_reward
            breakdown["memory_safety"] = mem_reward

        # Expose per-family counts in breakdown for observability
        breakdown["safety_S"] = round(S, 4)
        breakdown["unsafe_rpc"] = snapshot.counts.rpc
        breakdown["unsafe_rpr"] = snapshot.counts.rpr
        breakdown["unsafe_luc"] = snapshot.counts.luc
        breakdown["unsafe_uce"] = snapshot.counts.uce
        breakdown["unsafe_utc"] = snapshot.counts.utc
        breakdown["total_unsafe"] = snapshot.total_unsafe

        # ── 4. Coupling (CBO) ──────────────────────────────────────────────
        cbo = metrics.get("cbo", 0)
        cbo_constraint = any("cbo" in c.lower() for c in self._constraints)

        if cbo_constraint and cbo >= 3:
            reward -= self.P_HIGH_CBO
            breakdown["cbo_penalty"] = -self.P_HIGH_CBO
            breakdown["cbo"] = 0.0
        else:
            cbo_reward = self.W_CBO if cbo < 3 else max(0.0, self.W_CBO * (1 - (cbo - 2) / 5))
            reward += cbo_reward
            breakdown["cbo"] = round(cbo_reward, 4)

        # ── 5. Cohesion (LCOM) ─────────────────────────────────────────────
        lcom = metrics.get("lcom", 0)
        cohesion_score = max(0.0, 1.0 - lcom / 5.0)
        cohesion_reward = round(self.W_COHESION * cohesion_score, 4)
        reward += cohesion_reward
        breakdown["cohesion"] = cohesion_reward

        clamped = round(max(0.01, min(0.99, reward)), 4)
        breakdown["total"] = clamped
        return clamped, breakdown

    def _format_response(
        self, done: bool, reward: float, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        clamped = round(max(0.01, min(0.99, float(reward))), 4)
        return {
            "observation": self.observation(),
            "reward": clamped,
            "done": done,
            "info": info,
        }
