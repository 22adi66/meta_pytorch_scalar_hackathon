//! target_service — Top-level service wrapping the data store (root node).
//!
//! Rust translation target for: legacy_c/src/target_service.c
//! Depends on: data_store, math_ops
//!
//! This is the "God object" the agent must decompose into a clean, modular Rust struct.
//! The RL reward penalizes high CBO — agent must NOT import unnecessary external crates.
//! Constraints: no `unsafe`, CBO < 3.

use crate::data_store::DataStore;
use crate::math_ops::{add, clamp, divide, multiply};

/// Service wrapping a DataStore with request/error tracking.
pub struct TargetService {
    store:         DataStore,
    request_count: i32,
    error_count:   i32,
}

impl TargetService {
    /// Create a new, empty service.
    pub fn new() -> Self {
        TargetService {
            store:         DataStore::new(),
            request_count: 0,
            error_count:   0,
        }
    }

    /// Set a key-value pair. Returns true on success.
    pub fn set(&mut self, key: &str, value: i32) -> bool {
        self.request_count = add(self.request_count, 1);
        let ok = self.store.set(key, value);
        if !ok {
            self.error_count = add(self.error_count, 1);
        }
        ok
    }

    /// Get a value by key. Returns None if not found.
    pub fn get(&mut self, key: &str) -> Option<i32> {
        self.request_count = add(self.request_count, 1);
        let result = self.store.get(key);
        if result.is_none() {
            self.error_count = add(self.error_count, 1);
        }
        result
    }

    /// Delete a key. Returns true if deleted.
    pub fn delete(&mut self, key: &str) -> bool {
        self.request_count = add(self.request_count, 1);
        self.store.delete(key)
    }

    /// Return the number of active store entries.
    pub fn entry_count(&self) -> usize {
        self.store.count()
    }

    /// Return total requests processed.
    pub fn request_count(&self) -> i32 {
        self.request_count
    }

    /// Error rate as a percentage [0, 100].
    pub fn error_rate(&self) -> i32 {
        if self.request_count == 0 {
            return 0;
        }
        let rate = multiply(
            divide(self.error_count, self.request_count).unwrap_or(0),
            100,
        );
        clamp(rate, 0, 100)
    }
}

impl Default for TargetService {
    fn default() -> Self {
        Self::new()
    }
}
