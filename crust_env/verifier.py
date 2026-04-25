"""
CRust Verifier — verifier.py

Sandboxed compilation, test execution, and static analysis for the CRust environment.
Provides deterministic, programmatic rewards — no LLM-as-a-judge.

Security measures:
  - Path traversal protection on all file writes
  - Strict subprocess timeouts (cargo check: 30s, cargo test: 60s)
  - Test suite is READ-ONLY — agent cannot modify it (anti-reward-hacking)
  - unsafe block counting for memory safety metric
"""

import subprocess
import json
import os
import re
import shutil
from typing import Dict, Any, List, Tuple


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
        unsafe_info = self.count_unsafe_blocks(code_content)

        if not syntax_result.get("success"):
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

    def count_unsafe_blocks(self, code: str) -> Dict[str, Any]:
        """
        Static analysis: count unsafe keyword occurrences and compute memory safety ratio.

        Memory safety ratio = 1 - (unsafe_lines / total_lines)
        A ratio of 1.0 means perfectly safe; 0.0 means entirely unsafe.
        """
        lines = code.splitlines()
        total_lines = max(1, len(lines))
        unsafe_lines = sum(1 for line in lines if re.search(r'\bunsafe\b', line))
        unsafe_blocks = len(re.findall(r'\bunsafe\s*\{', code))
        memory_safety_ratio = round(1.0 - unsafe_lines / total_lines, 4)

        return {
            "unsafe_count": unsafe_blocks,
            "unsafe_lines": unsafe_lines,
            "memory_safety_ratio": memory_safety_ratio,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _fail(self, stage: str, message: str, reward: float = 0.01) -> Dict[str, Any]:
        return {
            "success": False,
            "stage": stage,
            "reward": reward,
            "diagnostics": [{"message": message, "level": "error", "code": None, "spans": []}],
            "unsafe_count": 0,
            "unsafe_lines": 0,
            "memory_safety_ratio": 1.0,
        }
