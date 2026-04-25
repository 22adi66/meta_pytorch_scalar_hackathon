//! Integration tests — semantic equivalence ground truth.
//!
//! PROTECTED FILE: The RL agent must NOT modify this file.
//! (CRustVerifier enforces this via PROTECTED_FILES list.)
//!
//! These tests verify that the agent's Rust translations are semantically
//! equivalent to the original C source — the core of "semantic equivalence"
//! reward in the CRust RL environment.
//!
//! Test strategy:
//!   - Each test covers one C function's behavior exactly
//!   - Edge cases (zero, negative, boundary) match C semantics
//!   - Tests are immutable to prevent reward hacking

use crust_workspace::math_ops;
use crust_workspace::string_ops;
use crust_workspace::data_store::DataStore;
use crust_workspace::target_service::TargetService;

// ── math_ops tests ────────────────────────────────────────────────────────

#[test]
fn test_add_basic() {
    assert_eq!(math_ops::add(2, 3), 5);
}

#[test]
fn test_add_negatives() {
    assert_eq!(math_ops::add(-5, 3), -2);
    assert_eq!(math_ops::add(-1, -1), -2);
}

#[test]
fn test_add_zero() {
    assert_eq!(math_ops::add(0, 0), 0);
    assert_eq!(math_ops::add(42, 0), 42);
}

#[test]
fn test_subtract() {
    assert_eq!(math_ops::subtract(10, 3), 7);
    assert_eq!(math_ops::subtract(0, 5), -5);
    assert_eq!(math_ops::subtract(-3, -3), 0);
}

#[test]
fn test_multiply() {
    assert_eq!(math_ops::multiply(3, 4), 12);
    assert_eq!(math_ops::multiply(-2, 5), -10);
    assert_eq!(math_ops::multiply(0, 999), 0);
}

#[test]
fn test_divide_normal() {
    assert_eq!(math_ops::divide(10, 2), Some(5));
    assert_eq!(math_ops::divide(7, 2), Some(3));  // integer division
}

#[test]
fn test_divide_by_zero() {
    // C returns -1; Rust returns None (idiomatic Rust equivalent)
    assert_eq!(math_ops::divide(5, 0), None);
}

#[test]
fn test_clamp_within_range() {
    assert_eq!(math_ops::clamp(5, 0, 10), 5);
}

#[test]
fn test_clamp_below_min() {
    assert_eq!(math_ops::clamp(-5, 0, 10), 0);
}

#[test]
fn test_clamp_above_max() {
    assert_eq!(math_ops::clamp(100, 0, 10), 10);
}

// ── string_ops tests ──────────────────────────────────────────────────────

#[test]
fn test_str_len_basic() {
    assert_eq!(string_ops::str_len("hello"), 5);
}

#[test]
fn test_str_len_empty() {
    assert_eq!(string_ops::str_len(""), 0);
}

#[test]
fn test_str_len_spaces() {
    assert_eq!(string_ops::str_len("a b"), 3);
}

#[test]
fn test_to_upper() {
    assert_eq!(string_ops::to_upper("hello"), "HELLO");
    assert_eq!(string_ops::to_upper("Hello World"), "HELLO WORLD");
}

#[test]
fn test_to_upper_already_upper() {
    assert_eq!(string_ops::to_upper("ABC"), "ABC");
}

#[test]
fn test_to_lower() {
    assert_eq!(string_ops::to_lower("WORLD"), "world");
    assert_eq!(string_ops::to_lower("Mixed"), "mixed");
}

#[test]
fn test_str_equals_same() {
    assert!(string_ops::str_equals("abc", "abc"));
}

#[test]
fn test_str_equals_different() {
    assert!(!string_ops::str_equals("abc", "xyz"));
    assert!(!string_ops::str_equals("abc", "abcd"));
}

#[test]
fn test_str_equals_empty() {
    assert!(string_ops::str_equals("", ""));
    assert!(!string_ops::str_equals("", "a"));
}

// ── data_store tests ──────────────────────────────────────────────────────

#[test]
fn test_store_set_and_get() {
    let mut store = DataStore::new();
    assert!(store.set("key1", 42));
    assert_eq!(store.get("key1"), Some(42));
}

#[test]
fn test_store_get_missing() {
    let store = DataStore::new();
    assert_eq!(store.get("nonexistent"), None);
}

#[test]
fn test_store_update_existing() {
    let mut store = DataStore::new();
    store.set("x", 1);
    store.set("x", 2);
    assert_eq!(store.get("x"), Some(2));
    assert_eq!(store.count(), 1);  // still just one entry
}

#[test]
fn test_store_delete() {
    let mut store = DataStore::new();
    store.set("temp", 99);
    assert!(store.delete("temp"));
    assert_eq!(store.get("temp"), None);
}

#[test]
fn test_store_delete_missing() {
    let mut store = DataStore::new();
    assert!(!store.delete("ghost"));
}

#[test]
fn test_store_count() {
    let mut store = DataStore::new();
    assert_eq!(store.count(), 0);
    store.set("a", 1);
    store.set("b", 2);
    assert_eq!(store.count(), 2);
    store.delete("a");
    assert_eq!(store.count(), 1);
}

#[test]
fn test_store_value_clamped() {
    let mut store = DataStore::new();
    store.set("big", 9_999_999);
    // value should be clamped to 1_000_000
    assert_eq!(store.get("big"), Some(1_000_000));
}

#[test]
fn test_store_capacity() {
    let mut store = DataStore::new();
    for i in 0..64 {
        assert!(store.set(&format!("key{}", i), i));
    }
    // 65th insert should fail (at capacity, no existing key)
    assert!(!store.set("overflow", 0));
}

// ── target_service tests ──────────────────────────────────────────────────

#[test]
fn test_service_set_and_get() {
    let mut svc = TargetService::new();
    assert!(svc.set("metric", 100));
    assert_eq!(svc.get("metric"), Some(100));
}

#[test]
fn test_service_get_missing_increments_errors() {
    let mut svc = TargetService::new();
    assert_eq!(svc.get("ghost"), None);
    // request_count should be 1
    assert_eq!(svc.request_count(), 1);
}

#[test]
fn test_service_delete() {
    let mut svc = TargetService::new();
    svc.set("tmp", 5);
    assert!(svc.delete("tmp"));
    assert_eq!(svc.get("tmp"), None);
}

#[test]
fn test_service_entry_count() {
    let mut svc = TargetService::new();
    svc.set("a", 1);
    svc.set("b", 2);
    assert_eq!(svc.entry_count(), 2);
}

#[test]
fn test_service_error_rate_zero() {
    let mut svc = TargetService::new();
    svc.set("k", 1);
    let _ = svc.get("k");   // success, no error
    assert_eq!(svc.error_rate(), 0);
}
