//! Adapters layer: Concrete implementations of ports.
//!
//! These modules contain the actual integration with external libraries:
//! - `tfhe`: tfhe-rs for FHE operations
//! - `opendp`: OpenDP for differential privacy
//! - `sqlite`: SQLite for local storage
//! - `sanitize`: PII filtering for logs

pub mod opendp;
pub mod sanitize;
pub mod sqlite;
pub mod tfhe;

// Re-export storage error for lib.rs
pub use sqlite::StorageError;

