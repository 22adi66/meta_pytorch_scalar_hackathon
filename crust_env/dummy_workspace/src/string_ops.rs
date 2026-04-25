//! string_ops — String utility operations (leaf node, no dependencies).
//!
//! Rust translation target for: legacy_c/src/string_ops.c
//!
//! Agent must implement idiomatic Rust equivalents.
//! Note: Rust strings are UTF-8, but C legacy code is ASCII-only.
//! Constraints: no `unsafe`, CBO < 3.

/// Returns the number of bytes in the string (UTF-8 length).
pub fn str_len(s: &str) -> usize {
    s.len()
}

/// Returns a new uppercase String (ASCII-only, matching C behavior).
pub fn to_upper(s: &str) -> String {
    s.chars()
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// Returns a new lowercase String (ASCII-only, matching C behavior).
pub fn to_lower(s: &str) -> String {
    s.chars()
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Returns true if both strings are equal.
pub fn str_equals(a: &str, b: &str) -> bool {
    a == b
}
