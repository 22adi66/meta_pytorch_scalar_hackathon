"""
CRust — 5-Family Unsafe Construct Analysis (unsafe_constructs.py)

Implements the five unsafe-construct families from the LAC2R paper:
  "Search-Based Multi-Trajectory Refinement for Safe C-to-Rust Translation
   with Large Language Models" (Sim et al., arXiv:2505.15858), Equation 3.

The five families:
  RPC  — Raw Pointer Creations      (*const T, *mut T type declarations)
  RPR  — Raw Pointer References     (deref ops, ptr arithmetic, std::ptr::*)
  LUC  — Lines in Unsafe Constructs (LOC inside unsafe { } blocks)
  UCE  — Unsafe Calls / Extern C    (extern "C" blocks, FFI fn calls)
  UTC  — Unsafe Transmutes/Casts    (mem::transmute, as *const, as *mut, mem::forget)

Safety score S(r_i) = m(r_i) * max(1 - T_i / T_0, 0)
  where T_i  = RPC + RPR + LUC + UCE + UTC for the generated Rust
        T_0  = baseline total from C source complexity estimate
        m    = 1 if code compiles, 0 otherwise (compile gate)

This gives a continuous [0, 1] safety signal that rewards progressively safer
Rust (fewer unsafe patterns) rather than a binary has/doesn't-have-unsafe flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields as dc_fields
from typing import List, Tuple


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class UnsafeConstructCounts:
    """Per-file counts for each of the 5 unsafe construct families."""
    rpc: int = 0   # Raw Pointer Creations
    rpr: int = 0   # Raw Pointer References
    luc: int = 0   # Lines in Unsafe Constructs
    uce: int = 0   # Unsafe Calls / Extern C
    utc: int = 0   # Unsafe Transmutes / Casts

    def total(self) -> int:
        return self.rpc + self.rpr + self.luc + self.uce + self.utc

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


@dataclass
class SafetySnapshot:
    """Full safety evaluation for one Rust source file."""
    S: float                    # Safety score [0, 1]
    m: int                      # Compile gate (1 = compiles, 0 = doesn't)
    counts: UnsafeConstructCounts
    total_unsafe: int
    baseline_T: int             # T_0 used for ratio

    def as_dict(self) -> dict:
        return {
            "S": round(self.S, 4),
            "m": self.m,
            "total_unsafe": self.total_unsafe,
            "baseline_T": self.baseline_T,
            **self.counts.as_dict(),
        }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _strip_comments(src: str) -> str:
    """Remove // and /* */ comments before pattern matching."""
    s = re.sub(r"//[^\n]*", " ", src)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    return s


def _unsafe_block_spans(text: str) -> List[Tuple[int, int]]:
    """
    Find character spans of all top-level `unsafe { ... }` blocks.
    Uses brace-counting to handle nested blocks correctly.
    """
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(text):
        m = re.search(r"\bunsafe\s*\{", text[i:])
        if not m:
            break
        start = i + m.start()
        j = i + m.end()
        depth = 1
        k = j
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        if depth == 0:
            spans.append((start, k))
        i = j
    return spans


def _count_lines_in_spans(text: str, spans: List[Tuple[int, int]]) -> int:
    """Count how many source lines overlap with at least one unsafe block span."""
    if not spans:
        return 0
    lines_counted = 0
    pos = 0
    for line in text.splitlines():
        line_start = pos
        line_end = pos + len(line)
        if any(s < line_end and e > line_start for s, e in spans):
            lines_counted += 1
        pos = line_end + 1   # +1 for the newline
    return lines_counted


# ── Core 5-family counter ──────────────────────────────────────────────────────

def count_unsafe_constructs(code: str) -> UnsafeConstructCounts:
    """
    Count all five unsafe construct families in a Rust source file.

    Returns UnsafeConstructCounts(rpc, rpr, luc, uce, utc).
    """
    c = _strip_comments(code)
    out = UnsafeConstructCounts()

    # ── RPC: Raw Pointer Creations ─────────────────────────────────────────
    # *const T and *mut T appearing in type position (declarations)
    out.rpc = (
        len(re.findall(r"\*const\s+\w", c))
        + len(re.findall(r"\*mut\s+\w", c))
    )

    # ── RPR: Raw Pointer References ────────────────────────────────────────
    # Dereferencing raw pointers, pointer arithmetic, std::ptr helpers
    rpr_patterns = [
        r"\*\s*\w+\s*(?:\.|\[)",          # *ptr.field or *ptr[idx]
        r"\.as_ptr\s*\(\)",               # slice.as_ptr()
        r"\.as_mut_ptr\s*\(\)",           # vec.as_mut_ptr()
        r"std\s*::\s*ptr\s*::\s*\w+",     # std::ptr::read / write / copy
        r"core\s*::\s*ptr\s*::\s*\w+",   # core::ptr::*
        r"\.offset\s*\(",                  # ptr.offset(n)
        r"\.add\s*\(\s*\d",               # ptr.add(n)
        r"\.sub\s*\(\s*\d",               # ptr.sub(n)
        r"ptr\s*::\s*read\b",             # ptr::read
        r"ptr\s*::\s*write\b",            # ptr::write
        r"ptr\s*::\s*copy\b",             # ptr::copy
        r"ptr\s*::\s*null\b",             # ptr::null / null_mut
    ]
    out.rpr = sum(len(re.findall(p, c)) for p in rpr_patterns)

    # ── LUC: Lines in Unsafe Constructs ───────────────────────────────────
    spans = _unsafe_block_spans(c)
    out.luc = _count_lines_in_spans(c, spans)

    # ── UCE: Unsafe Calls / Extern C ──────────────────────────────────────
    # extern "C" / "system" blocks, and direct FFI function calls
    uce_patterns = [
        r'extern\s+"C"',                   # extern "C" { ... }
        r'extern\s+"system"',              # extern "system" { ... }
        r'extern\s+"cdecl"',              # extern "cdecl" { ... }
        r'#\s*\[\s*no_mangle\s*\]',        # #[no_mangle] (FFI export)
        r'libc\s*::\s*\w+\s*\(',          # libc::malloc(, libc::free(
    ]
    out.uce = sum(len(re.findall(p, c)) for p in uce_patterns)

    # ── UTC: Unsafe Transmutes / Casts ─────────────────────────────────────
    # mem::transmute, mem::forget, mem::zeroed, dangerous as-casts
    utc_patterns = [
        r"mem\s*::\s*transmute\b",         # std::mem::transmute
        r"std\s*::\s*mem\s*::\s*transmute\b",
        r"core\s*::\s*mem\s*::\s*transmute\b",
        r"mem\s*::\s*forget\b",            # mem::forget (leaks memory)
        r"mem\s*::\s*zeroed\b",            # mem::zeroed (uninitialized)
        r"mem\s*::\s*uninitialized\b",     # deprecated but dangerous
        r"as\s+\*\s*const\s+\w",          # expr as *const T
        r"as\s+\*\s*mut\s+\w",            # expr as *mut T
        r"MaybeUninit\s*::\s*assume_init", # MaybeUninit::assume_init()
    ]
    out.utc = sum(len(re.findall(p, c)) for p in utc_patterns)

    return out


# ── Baseline estimation from C source ─────────────────────────────────────────

def estimate_baseline_T_from_c(c_source: str) -> int:
    """
    Estimate T_0 (baseline unsafe construct count) from C source code.

    Logic: C patterns that typically map to unsafe Rust when translated naively:
      - Pointer dereferences (*ptr, ptr->field)  → RPC + RPR in Rust
      - Explicit casts ((type *)expr)             → UTC in Rust
      - malloc/free/memcpy/memset calls           → UCE + RPR in Rust
      - Array subscript with pointer arithmetic   → RPR in Rust

    Returns max(1, count) so T_0 is never zero (avoids division by zero).
    """
    c = _strip_comments(c_source)
    count = 0

    # Pointer dereference and arrow operator
    count += len(re.findall(r"\*\s*[a-zA-Z_]\w*", c))
    count += len(re.findall(r"[a-zA-Z_]\w*\s*->", c))

    # Explicit pointer casts: (type *), (const type *)
    count += len(re.findall(r"\(\s*(?:const\s+)?\w+\s*\*+\s*\)", c))

    # Unsafe stdlib calls
    unsafe_calls = [
        r"\bmalloc\s*\(", r"\bcalloc\s*\(", r"\brealloc\s*\(", r"\bfree\s*\(",
        r"\bmemcpy\s*\(", r"\bmemmove\s*\(", r"\bmemset\s*\(",
        r"\bstrcpy\s*\(", r"\bstrcat\s*\(", r"\bsprintf\s*\(",
    ]
    count += sum(len(re.findall(p, c)) for p in unsafe_calls)

    # Pointer arithmetic (ptr + n, ptr - n)
    count += len(re.findall(r"[a-zA-Z_]\w*\s*[+\-]\s*\d+\s*[;,\)]", c))

    return max(1, count)


# ── High-level safety score computation ───────────────────────────────────────

def compute_safety_score(
    rust_code: str,
    baseline_T: int,
    *,
    compilable: bool,
) -> SafetySnapshot:
    """
    Compute the full safety snapshot for a generated Rust file.

    Implements paper Equation 3:
      S(r_i) = m(r_i) * max(1 - T_i / T_0, 0)

    Args:
        rust_code:   The generated Rust source string.
        baseline_T:  T_0, estimated from C source complexity.
        compilable:  Whether the code passed `cargo check`.

    Returns:
        SafetySnapshot with S ∈ [0, 1], compile gate m, and per-family counts.
    """
    counts = count_unsafe_constructs(rust_code)
    T_i = counts.total()
    T_0 = max(1, int(baseline_T))
    m = 1 if compilable else 0
    ratio = max(0.0, 1.0 - (T_i / float(T_0)))
    S = float(m) * ratio

    return SafetySnapshot(
        S=round(S, 4),
        m=m,
        counts=counts,
        total_unsafe=T_i,
        baseline_T=T_0,
    )
