//! Inference service: Orchestrates encrypted ML inference.
//!
//! This service coordinates:
//! - Key generation/loading
//! - Data encryption
//! - Blind computation
//! - Result decryption
//! - Storage persistence

use std::sync::Arc;

use crate::domain::{Diagnosis, KeyPair, PatientData};
use crate::ports::{FheEngine, Storage};
use crate::PulsecureError;

/// Service for running encrypted ML inference.
///
/// # Key Memory Security
///
/// This service avoids holding decrypted private keys in long-lived fields.
/// Keys are loaded from storage only when needed (just-in-time) and dropped
/// immediately after the operation completes.
///
/// The `KeyPair` implements `Zeroize` and `ZeroizeOnDrop`, ensuring key material
/// is securely erased when dropped.
pub struct InferenceService<F, S>
where
    F: FheEngine,
    S: Storage,
{
    fhe: Arc<F>,
    storage: Arc<S>,
}

impl<F, S> InferenceService<F, S>
where
    F: FheEngine,
    S: Storage,
    S::Error: Into<crate::adapters::StorageError>,
{
    /// Create a new inference service.
    pub fn new(fhe: Arc<F>, storage: Arc<S>) -> Self {
        Self { fhe, storage }
    }

    /// Initialize the service by loading or generating keys.
    ///
    /// # Errors
    /// Returns error if key operations fail.
    pub fn initialize(&mut self) -> Result<(), PulsecureError> {
        tracing::info!("Initializing inference service...");

        let allow_clear_on_key_decryption_failure = std::env::var(
            "PULSECURE_CLEAR_STORAGE_ON_KEY_DECRYPTION_FAILURE",
        )
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);

        // Ensure keys exist. Do not keep them resident after initialization.
        match self.storage.has_keys() {
            Ok(true) => {
                // Validate that keys are actually loadable/decryptable.
                match self.storage.load_keys() {
                    Ok(Some(_)) => tracing::info!("Loaded existing keys from storage"),
                    Ok(None) => {
                        tracing::info!("Keys missing, generating new keys...");
                        self.generate_new_keys()?;
                    }
                    Err(e) => {
                        let se: crate::adapters::StorageError = e.into();

                        // If keys exist but cannot be decrypted (wrong password / corrupted), we can
                        // only recover safely when there is no historical data that would become
                        // undecryptable.
                        if matches!(se, crate::adapters::StorageError::KeyDecryption) {
                            let count = self
                                .storage
                                .count_diagnoses()
                                .map_err(|e| PulsecureError::Storage(e.into()))?;
                            if count == 0 {
                                tracing::warn!(
                                    "Stored keys cannot be decrypted; no diagnoses present, regenerating keys"
                                );
                                let _ = self.storage.delete_keys();
                                self.generate_new_keys()?;
                                return Ok(());
                            }

                            if allow_clear_on_key_decryption_failure {
                                tracing::warn!(
                                    "Stored keys cannot be decrypted; clearing storage due to explicit override"
                                );
                                self.storage
                                    .clear_all()
                                    .map_err(|e| PulsecureError::Storage(e.into()))?;
                                self.generate_new_keys()?;
                                return Ok(());
                            }
                        }

                        return Err(PulsecureError::Storage(se));
                    }
                }
            }
            Ok(false) => {
                tracing::info!("No existing keys found, generating new keys...");
                self.generate_new_keys()?;
            }
            Err(e) => {
                tracing::warn!("Failed to check keys: {:?}, attempting to load/generate", e);
                // Best-effort fallback: try to load; if missing, generate.
                match self.storage.load_keys() {
                    Ok(Some(_)) => tracing::info!("Loaded existing keys from storage"),
                    Ok(None) => self.generate_new_keys()?,
                    Err(_) => self.generate_new_keys()?,
                }
            }
        }

        Ok(())
    }

    /// Generate new FHE keys.
    fn generate_new_keys(&mut self) -> Result<(), PulsecureError> {
        let keys = self.fhe.generate_keys()?;

        // Save to storage
        self.storage
            .save_keys(&keys)
            .map_err(|e| PulsecureError::Storage(e.into()))?;
        Ok(())
    }

    /// Check if the service is initialized with keys.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.storage.has_keys().unwrap_or(false)
    }

    /// Get the current key fingerprints.
    #[must_use]
    pub fn key_fingerprints(&self) -> Option<(String, String)> {
        match self.storage.load_keys() {
            Ok(Some(k)) => Some((k.client.fingerprint.clone(), k.server.fingerprint.clone())),
            _ => None,
        }
    }

    fn load_keys_or_err(&self) -> Result<KeyPair, PulsecureError> {
        self.storage
            .load_keys()
            .map_err(|e| PulsecureError::Storage(e.into()))?
            .ok_or_else(|| PulsecureError::ModelNotLoaded("Keys not initialized".to_string()))
    }

    /// Run encrypted inference on patient data.
    ///
    /// Performs the full pipeline:
    /// 1. Encrypt patient data
    /// 2. Run blind computation
    /// 3. Decrypt result
    /// 4. Save diagnosis to storage
    ///
    /// # Errors
    /// Returns error if any step fails.
    pub fn run_inference(&self, patient: PatientData) -> Result<Diagnosis, PulsecureError> {
        let keys = self.load_keys_or_err()?;

        tracing::info!("Starting encrypted inference pipeline...");

        // Step 1: Encrypt patient data
        tracing::debug!("Step 1: Encrypting patient data...");
        let encrypted = self.fhe.encrypt(&patient, &keys.client)?;
        tracing::debug!(
            "Encrypted {} features, ciphertext size: {} bytes",
            encrypted.num_features,
            encrypted.size_bytes()
        );

        // Step 2: Blind computation
        tracing::debug!("Step 2: Running homomorphic computation...");
        let encrypted_result = self.fhe.compute(&encrypted, &keys.server)?;

        // Step 3: Decrypt result
        tracing::debug!("Step 3: Decrypting diagnosis...");
        let mut diagnosis = self.fhe.decrypt(&encrypted_result, &keys.client)?;

        // Associate with patient if ID provided
        if let Some(patient_id) = &patient.id {
            diagnosis.patient_id = Some(patient_id.clone());
        }

        // Step 4: Save to storage
        tracing::debug!("Step 4: Saving diagnosis to storage...");
        if let Err(e) = self.storage.save_diagnosis(&diagnosis) {
            tracing::warn!("Failed to save diagnosis: {:?}", e);
        }

        tracing::info!(
            "Inference complete: prediction={}, confidence={:.2}%, risk={}",
            diagnosis.result.prediction,
            diagnosis.result.confidence * 100.0,
            diagnosis.risk_level
        );

        Ok(diagnosis)
    }

    /// Get recent diagnoses from storage.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    pub fn get_recent_diagnoses(&self, limit: usize) -> Result<Vec<Diagnosis>, PulsecureError> {
        self.storage
            .load_recent_diagnoses(limit)
            .map_err(|e| PulsecureError::Storage(e.into()))
    }

    /// Get total diagnosis count.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    pub fn get_diagnosis_count(&self) -> Result<usize, PulsecureError> {
        self.storage
            .count_diagnoses()
            .map_err(|e| PulsecureError::Storage(e.into()))
    }

    /// Regenerate keys (destroys existing keys).
    ///
    /// # Errors
    /// Returns error if key operations fail.
    pub fn regenerate_keys(&mut self) -> Result<(), PulsecureError> {
        tracing::warn!("Regenerating FHE keys...");

        // Prevent orphaning historical encrypted data by default.
        // If diagnoses exist, regenerating keys can make previously stored encrypted
        // artifacts undecryptable. Require explicit opt-in.
        let existing = self
            .storage
            .count_diagnoses()
            .map_err(|e| PulsecureError::Storage(e.into()))?;
        if existing > 0 {
            let allow = std::env::var("PULSECURE_ALLOW_KEY_REGEN_WITH_EXISTING_DATA")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(false);
            if !allow {
                return Err(PulsecureError::Validation(
                    "Refusing to regenerate keys while diagnoses exist. Set PULSECURE_ALLOW_KEY_REGEN_WITH_EXISTING_DATA=true to force (this may orphan historical encrypted data).".to_string(),
                ));
            }
        }

        // Generate and save new keys WITHOUT deleting old keys first.
        // This avoids a window where a crash would leave the system without any keys.
        self.generate_new_keys()?;

        tracing::info!("Keys regenerated successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::sqlite::SqliteStorage;
    use crate::adapters::tfhe::TfheAdapter;
    use crate::domain::PatientFeatures;
    use std::path::Path;
    use std::sync::Once;

    fn allow_unsigned_models_for_tests() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            std::env::set_var("PULSECURE_ALLOW_UNSIGNED_MODELS", "true");
            // Required for SqliteStorage key encryption during tests.
            std::env::set_var(
                "PULSECURE_KEY_PASSWORD",
                "test_password_for_ci_only_32chars!",
            );
        });
    }

    fn create_test_service() -> InferenceService<TfheAdapter, SqliteStorage> {
        allow_unsigned_models_for_tests();

        let mut fhe = TfheAdapter::new();
        fhe.load_model(Path::new("models"))
            .expect("Model should load for tests");
        let fhe = Arc::new(fhe);
        let storage = Arc::new(SqliteStorage::in_memory().expect("Should create db"));
        InferenceService::new(fhe, storage)
    }

    #[test]
    fn test_initialization() {
        let mut service = create_test_service();
        assert!(!service.is_initialized());

        service.initialize().expect("Should initialize");
        assert!(service.is_initialized());
        assert!(service.key_fingerprints().is_some());
    }

    #[test]
    fn test_inference_pipeline() {
        let mut service = create_test_service();
        service.initialize().expect("Should initialize");

        let patient = PatientData::new(PatientFeatures {
            age: 55.0,
            hypertension: 1.0,
            sys_bp: 142.0,
            smoking: 1.0,
            hdl_chol: 45.0,
            creatinine: 1.1,
            waist_circ: 102.0,
            diabetes: 0.0,
            hba1c: 5.9,
        });

        let diagnosis = service.run_inference(patient).expect("Should run inference");

        assert!(diagnosis.result.probability >= 0.0);
        assert!(diagnosis.result.probability <= 1.0);
        assert!(diagnosis.encrypted_computation);
    }
}
