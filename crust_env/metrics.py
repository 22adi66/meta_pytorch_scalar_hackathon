"""
CRust Modularity Metrics — metrics.py

Computes software quality metrics on generated Rust code to enforce
architectural modularity — the core differentiator from naive 1:1 transpilation.

Metrics:
  CBO  (Coupling Between Objects)   — count of external, non-std dependencies
  LCOM (Lack of Cohesion in Methods) — measures field sharing across methods

A well-migrated module should have:
  CBO  < 3   (loosely coupled, reusable component)
  LCOM = 0   (highly cohesive — all methods touch the same fields)

These metrics are computed via regex heuristics (AST-level analysis would require
a Rust parser; regex provides a fast, MVP-viable approximation for the RL reward loop).
"""

import re
from typing import List, Dict, Tuple


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
        return bool(re.search(r'\bunsafe\b', code))

    @staticmethod
    def evaluate(code: str) -> Dict[str, object]:
        """
        Full evaluation suite.
        Returns all metrics used in the reward function.
        """
        cbo = ModularityMetrics.calculate_cbo(code)
        lcom = ModularityMetrics.calculate_lcom(code)
        pub_fns = ModularityMetrics.count_pub_functions(code)
        trait_impls = ModularityMetrics.count_trait_implementations(code)
        unsafe_present = ModularityMetrics.has_unsafe(code)

        return {
            "cbo": cbo,
            "lcom": lcom,
            "pub_functions": pub_fns,
            "trait_implementations": trait_impls,
            "has_unsafe": unsafe_present,
            # Human-readable quality summary
            "quality": "good" if cbo < 3 and lcom < 1 and not unsafe_present else "needs_improvement",
        }
