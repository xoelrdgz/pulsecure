#![allow(non_snake_case)]

pub mod adapters;
pub mod application;
pub mod domain;
pub mod ports;
pub mod web;

pub use domain::{Diagnosis, PatientData, RiskLevel};

pub type Result<T> = std::result::Result<T, PulsecureError>;

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
