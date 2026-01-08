//! Domain layer: Core business types and logic.
//!
//! This module contains pure Rust types with no external dependencies.
//! All types are serializable and implement strict validation.

mod crypto;
mod diagnosis;
pub mod kdf;
mod patient;

pub use crypto::{ClientKey, CryptoError, EncryptedDiagnosis, EncryptedPatientData, KeyPair, ServerKey};
pub use diagnosis::{Diagnosis, DiagnosisResult, RiskLevel};
pub use patient::{PatientData, PatientFeatures};
