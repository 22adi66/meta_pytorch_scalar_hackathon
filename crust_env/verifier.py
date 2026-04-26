"""
CRust Verifier — verifier.py

Sandboxed compilation, test execution, and static analysis for the CRust environment.
Provides deterministic, programmatic rewards — no LLM-as-a-judge.

Security measures:
  - Path traversal protection on all file writes
  - Strict subprocess timeouts (cargo check: 30s, cargo test: 60s)
  - Test suite is READ-ONLY — agent cannot modify it (anti-reward-hacking)
  - 5-family unsafe construct analysis (RPC, RPR, LUC, UCE, UTC) for rich safety signal
"""

import subprocess
import json
import os
import re
import shutil
from typing import Dict, Any, List, Tuple

from .unsafe_constructs import count_unsafe_constructs, UnsafeConstructCounts


class VerifierFailedException(Exception):
    pass


class CRustVerifier:
    """
    Handles sandboxed Rust compilation, test execution, and static analysis.

    Verification pipeline:
        1. write_code_to_sandbox()  — secure file write
        2. check_syntax()           — cargo check --message-format=json
        3. run_tests()              — cargo test
        4. count_unsafe_blocks()    — static analysis

    Each stage returns partial rewards, implementing process supervision
    (agent gets credit for intermediate progress, not just final success).
    """

    # Protected files the agent must never overwrite
    PROTECTED_FILES: List[str] = [
        "tests/integration_test.rs",
        "Cargo.toml",
    ]

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def verify(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full verification pipeline. Called once per env.step().

        Returns a dict with keys:
          success, stage, reward, diagnostics, unsafe_count, memory_safety_ratio
        """
        file_path: str = (action.get("file_path") or "").strip()
        code_content: str = (action.get("code_content") or "").strip()

        if not file_path or not code_content:
            return self._fail("missing_input", "Missing file_path or code_content.", reward=0.01)

        # ── Anti-hacking: block writes to protected files ──────────────────
        norm = os.path.normpath(file_path).replace("\\", "/")
        for protected in self.PROTECTED_FILES:
            if protected in norm:
                return self._fail(
                    "security_violation",
                    f"Attempted write to protected file: {file_path}",
                    reward=0.01
                )

        # ── Write code to sandbox ──────────────────────────────────────────
        try:
            self.write_code_to_sandbox(file_path, code_content)
        except VerifierFailedException as e:
            return self._fail("sandbox_write_failed", str(e), reward=0.01)

        # ── Stage 1: Syntax / cargo check ─────────────────────────────────
        syntax_result = self.check_syntax()
        compiled = syntax_result.get("success", False)
        unsafe_info = self.count_unsafe_constructs(code_content, compilable=compiled)

        if not compiled:
            return {
                "success": False,
                "stage": "compilation",
                "reward": 0.10,   # Process supervision: minimal reward for attempt
                "diagnostics": syntax_result.get("diagnostics", []),
                "stderr": syntax_result.get("stderr", ""),
                **unsafe_info,
            }

        # ── Stage 2: Semantic equivalence via unit tests ───────────────────
        test_result = self.run_tests()

        if not test_result.get("success"):
            return {
                "success": False,
                "stage": "testing",
                "reward": 0.40,   # Compiled but tests failed
                "diagnostics": [],
                "test_output": test_result.get("output", ""),
                **unsafe_info,
            }

        # ── All stages passed ──────────────────────────────────────────────
        return {
            "success": True,
            "stage": "complete",
            "reward": 0.99,
            "diagnostics": [],
            "test_output": test_result.get("output", ""),
            **unsafe_info,
        }

    def write_code_to_sandbox(self, file_path: str, code_content: str) -> None:
        """
        Writes agent code into the workspace with path traversal protection.
        Raises VerifierFailedException on invalid paths.
        """
        # Reject absolute paths and traversal attempts
        if os.path.isabs(file_path):
            raise VerifierFailedException(f"Absolute paths not allowed: {file_path}")
        if ".." in file_path.replace("\\", "/").split("/"):
            raise VerifierFailedException(f"Path traversal detected: {file_path}")

        full_path = os.path.normpath(os.path.join(self.workspace_dir, file_path))

        # Double-check the resolved path is still inside workspace
        if not full_path.startswith(os.path.normpath(self.workspace_dir)):
            raise VerifierFailedException(f"Path escape detected: {file_path}")

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code_content)

    def check_syntax(self) -> Dict[str, Any]:
        """
        Runs `cargo check --message-format=json` and parses structured diagnostics.
        Returns structured compiler messages with error codes, levels, and messages.
        """
        try:
            result = subprocess.run(
                ["cargo", "check", "--message-format=json"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            diagnostics: List[Dict] = []
            for line in result.stdout.splitlines():
                try:
                    msg = json.loads(line)
                    if msg.get("reason") == "compiler-message":
                        m = msg.get("message", {})
                        diagnostics.append({
                            "message": m.get("message", ""),
                            "level": m.get("level", ""),
                            "code": m.get("code"),
                            "spans": [
                                {
                                    "file": s.get("file_name"),
                                    "line_start": s.get("line_start"),
                                    "line_end": s.get("line_end"),
                                }
                                for s in (m.get("spans") or [])
                            ],
                        })
                except json.JSONDecodeError:
                    continue

            return {
                "success": result.returncode == 0,
                "diagnostics": diagnostics,
                "stderr": result.stderr[:2000],  # cap stderr length
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "diagnostics": [],
                "stderr": "Compilation timed out (30s limit).",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "diagnostics": [],
                "stderr": "cargo not found. Please install Rust: https://rustup.rs/",
            }

    def run_tests(self) -> Dict[str, Any]:
        """
        Runs `cargo test` inside the sandboxed workspace.

        Sandboxing strategy:
          - subprocess with strict timeout (60s) prevents infinite loops
          - test suite files are protected against agent writes
          - resource isolation via timeout prevents macro-stall attacks
        """
        try:
            result = subprocess.run(
                ["cargo", "test", "--", "--test-output", "immediate"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout[:3000],
                "stderr": result.stderr[:1000],
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "stderr": "Test execution timed out (60s limit — possible infinite loop).",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "stderr": "cargo not found.",
            }

    def count_unsafe_constructs(self, code: str, *, compilable: bool = True) -> Dict[str, Any]:
        """
        5-family unsafe construct analysis (LAC2R paper §2.3, Eq. 3).

        Replaces the old binary count_unsafe_blocks() with a rich, per-family
        breakdown that gives the RL reward function a much more informative signal.

        Families:
          RPC — Raw Pointer Creations  (*const T, *mut T)
          RPR — Raw Pointer References (deref ops, ptr arithmetic)
          LUC — Lines in Unsafe Constructs (LOC inside unsafe { } blocks)
          UCE — Unsafe Calls / Extern C (FFI, extern "C")
          UTC — Unsafe Transmutes/Casts (mem::transmute, as *const/mut)

        Returns dict compatible with the old interface (unsafe_count, unsafe_lines,
        memory_safety_ratio) PLUS the full 5-family breakdown.
        """
        counts: UnsafeConstructCounts = count_unsafe_constructs(code)
        total = counts.total()

        # Legacy-compatible fields (kept so nothing else breaks)
        lines = code.splitlines()
        total_lines = max(1, len(lines))
        unsafe_lines = counts.luc   # LUC = lines inside unsafe blocks
        memory_safety_ratio = round(1.0 - unsafe_lines / total_lines, 4)

        return {
            # Legacy fields
            "unsafe_count": total,
            "unsafe_lines": unsafe_lines,
            "memory_safety_ratio": memory_safety_ratio,
            # 5-family breakdown (new)
            "unsafe_rpc": counts.rpc,
            "unsafe_rpr": counts.rpr,
            "unsafe_luc": counts.luc,
            "unsafe_uce": counts.uce,
            "unsafe_utc": counts.utc,
            "total_unsafe_constructs": total,
            "compilable": compilable,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _fail(self, stage: str, message: str, reward: float = 0.01) -> Dict[str, Any]:
        return {
            "success": False,
            "stage": stage,
            "reward": reward,
            "diagnostics": [{"message": message, "level": "error", "code": None, "spans": []}],
            # Legacy fields
            "unsafe_count": 0,
            "unsafe_lines": 0,
            "memory_safety_ratio": 1.0,
            # 5-family breakdown
            "unsafe_rpc": 0,
            "unsafe_rpr": 0,
            "unsafe_luc": 0,
            "unsafe_uce": 0,
            "unsafe_utc": 0,
            "total_unsafe_constructs": 0,
            "compilable": False,
        }
