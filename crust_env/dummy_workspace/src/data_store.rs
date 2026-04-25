//! data_store — Fixed-capacity key-value store (intermediate node).
//!
//! Rust translation target for: legacy_c/src/data_store.c
//! Depends on: math_ops (add, subtract, clamp), string_ops (str_equals)
//!
//! The agent must refactor the C struct+array pattern into idiomatic Rust
//! using HashMap, eliminating raw pointer aliasing and manual bounds checks.
//! Constraints: no `unsafe`, CBO < 3.

use std::collections::HashMap;

use crate::math_ops::clamp;

/// Maximum number of entries in the store (matches C MAX_ENTRIES).
pub const MAX_ENTRIES: usize = 64;
/// Maximum value clamped on insert (matches C store_set behavior).
const VALUE_CLAMP: i32 = 1_000_000;

/// Idiomatic Rust key-value store replacing the C DataStore struct.
pub struct DataStore {
    entries: HashMap<String, i32>,
}

impl DataStore {
    /// Create an empty store.
    pub fn new() -> Self {
        DataStore {
            entries: HashMap::new(),
        }
    }

    /// Insert or update a key. Returns false if at capacity.
    pub fn set(&mut self, key: &str, value: i32) -> bool {
        if !self.entries.contains_key(key) && self.entries.len() >= MAX_ENTRIES {
            return false;
        }
        let clamped = clamp(value, -VALUE_CLAMP, VALUE_CLAMP);
        self.entries.insert(key.to_string(), clamped);
        true
    }

    /// Retrieve value by key. Returns None if not found.
    pub fn get(&self, key: &str) -> Option<i32> {
        self.entries.get(key).copied()
    }

    /// Delete a key. Returns true if the key existed.
    pub fn delete(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Return the number of entries currently in the store.
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for DataStore {
    fn default() -> Self {
        Self::new()
    }
}
