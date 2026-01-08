//! # Pulsecure
#![allow(non_snake_case)]
//!
//! Privacy-Preserving Medical ML Pipeline using Fully Homomorphic Encryption.
//!
//! This crate provides:
//! - FHE-based encrypted inference for medical diagnostics
//! - Differential privacy for aggregated statistics
//! - Terminal UI for local-only deployment
//!
//! ## Architecture
//!
//! The crate follows Hexagonal Architecture:
//! - `domain`: Core business types (Patient, Diagnosis, Crypto keys)
//! - `ports`: Trait definitions for external operations
//! - `adapters`: Concrete implementations (tfhe-rs, OpenDP, SQLite)
//! - `application`: Use cases orchestrating domain and ports
//! - `tui`: Terminal user interface

pub mod adapters;
pub mod application;
pub mod domain;
pub mod ports;
pub mod tui;

pub use domain::{Diagnosis, PatientData, RiskLevel};

/// Result type for Pulsecure operations
pub type Result<T> = std::result::Result<T, PulsecureError>;

/// Main error type for Pulsecure
#[derive(Debug, thiserror::Error)]
pub enum PulsecureError {
    #[error("Cryptographic operation failed: {0}")]
    Crypto(#[from] domain::CryptoError),

    #[error("Storage operation failed: {0}")]
    Storage(#[from] adapters::StorageError),

    #[error("Invalid patient data: {0}")]
    Validation(String),

    #[error("FHE noise budget exhausted")]
    NoiseBudgetExhausted,

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Privacy error: {0}")]
    Privacy(String),
}
