"""
CRust Modularity Metrics — metrics.py

Computes software quality metrics on generated Rust code to enforce
architectural modularity — the core differentiator from naive 1:1 transpilation.

Metrics:
  CBO  (Coupling Between Objects)   — count of external, non-std dependencies
  LCOM (Lack of Cohesion in Methods) — measures field sharing across methods

  5-Family Safety Score (LAC2R paper, Eq. 3):
    RPC  — Raw Pointer Creations      (*const T, *mut T)
    RPR  — Raw Pointer References     (deref, ptr arithmetic, std::ptr::*)
    LUC  — Lines in Unsafe Constructs (LOC inside unsafe { } blocks)
    UCE  — Unsafe Calls / Extern C    (FFI, extern "C")
    UTC  — Unsafe Transmutes/Casts    (mem::transmute, as *const, as *mut)

    S(r) = m * max(1 - T_i/T_0, 0) where T_i = RPC+RPR+LUC+UCE+UTC

A well-migrated module should have:
  CBO  < 3   (loosely coupled, reusable component)
  LCOM = 0   (highly cohesive — all methods touch the same fields)
  S    = 1   (zero unsafe constructs, code compiles)
"""

import re
from typing import List, Dict, Tuple, Optional

from .unsafe_constructs import (
    count_unsafe_constructs,
    compute_safety_score,
    estimate_baseline_T_from_c,
    UnsafeConstructCounts,
    SafetySnapshot,
)


class ModularityMetrics:
    """
    Static analysis of Rust source code for modularity quality metrics.
    Used in the CRust RL reward function to penalize monolithic God-objects.
    """

    @staticmethod
    def calculate_cbo(code: str) -> int:
        """
        Coupling Between Objects (CBO).

        Approximation: count the number of external `use` import paths that are:
          - Not from std::
          - Not from core::
          - Not from alloc::
          - Not empty/super/crate self-references

        Each unique external crate imported contributes +1 to CBO.
        CBO >= 3 triggers a reward penalty per the hackathon constraints.
        """
        use_statements = re.findall(
            r'^\s*use\s+([A-Za-z_][A-Za-z0-9_:*{}, ]+);',
            code,
            re.MULTILINE
        )

        STDLIB_PREFIXES = ("std::", "core::", "alloc::", "super::", "crate::", "self::")

        external_crates = set()
        for stmt in use_statements:
            # Extract the top-level crate name
            top = stmt.strip().split("::")[0].strip().lstrip("{").strip()
            if top and not any(top + "::" in p or top == p.rstrip("::") for p in STDLIB_PREFIXES):
                external_crates.add(top)

        return len(external_crates)

    @staticmethod
    def calculate_lcom(code: str) -> float:
        """
        Lack of Cohesion in Methods (LCOM).

        Algorithm:
          1. Find the first struct definition and extract its field names.
          2. Find the corresponding impl block.
          3. For each method body, count how many fields it references via `self.field`.
          4. LCOM = |fields| - average(fields used per method).

        Interpretation:
          LCOM = 0  → perfect cohesion (every method uses every field)
          LCOM > 0  → lower cohesion (methods operate on disjoint field subsets)
        """
        # Find struct fields
        struct_match = re.search(
            r'(?:pub\s+)?struct\s+\w+\s*(?:<[^>]*>)?\s*\{([^}]*)\}',
            code,
            re.DOTALL
        )
        if not struct_match:
            return 0.0   # Pure functions / no struct → LCOM undefined, treat as 0

        fields_str = struct_match.group(1)
        # Extract field names (name: Type pattern)
        fields: List[str] = re.findall(
            r'(?:pub\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*:(?!:)',
            fields_str
        )
        # Filter out common false positives
        fields = [f for f in fields if f not in ("", "pub", "mut", "ref")]

        if not fields:
            return 0.0

        # Find impl block
        impl_match = re.search(
            r'impl\s+\w+[^{]*\{(.*)\}',
            code,
            re.DOTALL
        )
        if not impl_match:
            return 0.0   # No methods → cohesion undefined

        impl_body = impl_match.group(1)

        # Extract individual method bodies
        method_bodies: List[str] = re.findall(
            r'fn\s+\w+\s*\([^)]*\)[^{]*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}',
            impl_body,
            re.DOTALL
        )

        if not method_bodies:
            return 0.0

        # Count field usage per method
        usage_per_method = [
            sum(1 for field in fields if f"self.{field}" in body)
            for body in method_bodies
        ]

        avg_usage = sum(usage_per_method) / len(method_bodies)
        lcom = max(0.0, len(fields) - avg_usage)
        return round(lcom, 4)

    @staticmethod
    def count_pub_functions(code: str) -> int:
        """Count public functions — a simple measure of module surface area."""
        return len(re.findall(r'^\s*pub\s+fn\s+\w+', code, re.MULTILINE))

    @staticmethod
    def count_trait_implementations(code: str) -> int:
        """Count `impl Trait for Type` blocks — measures use of Rust idioms."""
        return len(re.findall(r'\bimpl\s+\w+\s+for\s+\w+', code))

    @staticmethod
    def has_unsafe(code: str) -> bool:
        """Binary check — kept for backward compatibility. Prefer safety_score_S for RL rewards."""
        return bool(re.search(r'\bunsafe\b', code))

    @staticmethod
    def get_unsafe_breakdown(code: str) -> Dict[str, int]:
        """
        5-family unsafe construct breakdown (LAC2R paper §2.3).
        Returns per-family counts as a flat dict.
        """
        counts: UnsafeConstructCounts = count_unsafe_constructs(code)
        return counts.as_dict()

    @staticmethod
    def safety_score_S(
        rust_code: str,
        baseline_T: int,
        *,
        compilable: bool,
    ) -> SafetySnapshot:
        """
        Compute the LAC2R safety score S(r) ∈ [0, 1] for generated Rust code.

        S(r_i) = m(r_i) * max(1 - T_i / T_0, 0)
          T_i = total unsafe constructs (RPC+RPR+LUC+UCE+UTC)
          T_0 = baseline_T estimated from C source
          m   = 1 if compilable, else 0
        """
        return compute_safety_score(rust_code, baseline_T, compilable=compilable)

    @staticmethod
    def evaluate(
        code: str,
        c_source: Optional[str] = None,
        compilable: bool = True,
    ) -> Dict[str, object]:
        """
        Full evaluation suite. Returns all metrics used in the reward function.

        Args:
            code:       Generated Rust source.
            c_source:   Original C source (used to estimate safety baseline T_0).
                        If None, T_0 defaults to 1 (conservative — any unsafe = penalty).
            compilable: Whether cargo check passed (gates the safety score).
        """
        cbo = ModularityMetrics.calculate_cbo(code)
        lcom = ModularityMetrics.calculate_lcom(code)
        pub_fns = ModularityMetrics.count_pub_functions(code)
        trait_impls = ModularityMetrics.count_trait_implementations(code)
        unsafe_present = ModularityMetrics.has_unsafe(code)

        # 5-family safety score (replaces binary has_unsafe in the reward)
        baseline_T = estimate_baseline_T_from_c(c_source) if c_source else 1
        snapshot: SafetySnapshot = compute_safety_score(
            code, baseline_T, compilable=compilable
        )

        return {
            "cbo": cbo,
            "lcom": lcom,
            "pub_functions": pub_fns,
            "trait_implementations": trait_impls,
            "has_unsafe": unsafe_present,
            # 5-family safety (LAC2R Eq. 3)
            "safety_S": snapshot.S,
            "unsafe_counts": snapshot.counts.as_dict(),
            "total_unsafe": snapshot.total_unsafe,
            "baseline_T": baseline_T,
            # Human-readable quality summary
            "quality": (
                "good"
                if cbo < 3 and lcom < 1 and snapshot.S >= 0.8
                else "needs_improvement"
            ),
        }
