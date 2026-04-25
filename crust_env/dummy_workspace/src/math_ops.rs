//! math_ops — Basic arithmetic operations (leaf node, no dependencies).
//!
//! Rust translation target for: legacy_c/src/math_ops.c
//!
//! Agent must implement all public functions below.
//! Constraints: no `unsafe`, CBO < 3 (no external crate imports).

/// Add two integers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Subtract b from a.
pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

/// Multiply two integers.
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// Integer division. Returns None on divide-by-zero (idiomatic Rust vs C sentinel -1).
pub fn divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

/// Clamp a value to the inclusive range [min_val, max_val].
pub fn clamp(value: i32, min_val: i32, max_val: i32) -> i32 {
    value.max(min_val).min(max_val)
}
