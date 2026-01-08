//! Ports layer: Trait definitions for external operations.
//!
//! Following Hexagonal Architecture, these traits define the boundaries
//! between the application and external systems (FHE library, storage, etc.).

mod fhe_engine;
mod privacy;
mod storage;

pub use fhe_engine::FheEngine;
pub use privacy::{DpError, DifferentialPrivacy, PrivateStatistics};
pub use storage::{DiagnosisPage, Storage};

