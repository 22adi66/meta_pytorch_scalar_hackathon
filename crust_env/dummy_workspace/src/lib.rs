//! CRust workspace — Rust migrations of the legacy C modules.
//!
//! Module structure mirrors the C dependency graph:
//!   math_ops   (leaf)  ← no dependencies
//!   string_ops (leaf)  ← no dependencies
//!   data_store         ← depends on math_ops, string_ops
//!   target_service     ← depends on data_store
//!
//! The RL agent populates these modules one at a time, bottom-up.
//! Tests in tests/integration_test.rs verify semantic equivalence with the C source.

pub mod math_ops;
pub mod string_ops;
pub mod data_store;
pub mod target_service;
