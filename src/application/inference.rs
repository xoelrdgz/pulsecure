use std::sync::Arc;

use crate::domain::{Diagnosis, KeyPair, PatientData};
use crate::ports::{FheEngine, Storage};
use crate::PulsecureError;

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
    pub fn new(fhe: Arc<F>, storage: Arc<S>) -> Self {
        Self { fhe, storage }
    }

    pub fn initialize(&mut self) -> Result<(), PulsecureError> {
        tracing::info!("Initializing inference service...");

        let allow_clear_on_key_decryption_failure =
            std::env::var("PULSECURE_CLEAR_STORAGE_ON_KEY_DECRYPTION_FAILURE")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(false);

        match self.storage.has_keys() {
            Ok(true) => match self.storage.load_keys() {
                Ok(Some(_)) => tracing::info!("Loaded existing keys from storage"),
                Ok(None) => {
                    tracing::info!("Keys missing, generating new keys...");
                    self.generate_new_keys()?;
                }
                Err(e) => {
                    let se: crate::adapters::StorageError = e.into();

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
            },
            Ok(false) => {
                tracing::info!("No existing keys found, generating new keys...");
                self.generate_new_keys()?;
            }
            Err(e) => {
                tracing::warn!("Failed to check keys: {:?}, attempting to load/generate", e);

                match self.storage.load_keys() {
                    Ok(Some(_)) => tracing::info!("Loaded existing keys from storage"),
                    Ok(None) => self.generate_new_keys()?,
                    Err(_) => self.generate_new_keys()?,
                }
            }
        }

        Ok(())
    }

    fn generate_new_keys(&mut self) -> Result<(), PulsecureError> {
        let keys = self.fhe.generate_keys()?;

        self.storage
            .save_keys(&keys)
            .map_err(|e| PulsecureError::Storage(e.into()))?;
        Ok(())
    }

    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.storage.has_keys().unwrap_or(false)
    }

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

    pub fn run_inference(&self, patient: PatientData) -> Result<Diagnosis, PulsecureError> {
        let keys = self.load_keys_or_err()?;

        tracing::info!("Starting encrypted inference pipeline...");

        tracing::debug!("Step 1: Encrypting patient data...");
        let encrypted = self.fhe.encrypt(&patient, &keys.client)?;
        tracing::debug!(
            "Encrypted {} features, ciphertext size: {} bytes",
            encrypted.num_features,
            encrypted.size_bytes()
        );

        tracing::debug!("Step 2: Running homomorphic computation...");
        let encrypted_result = self.fhe.compute(&encrypted, &keys.server)?;

        tracing::debug!("Step 3: Decrypting diagnosis...");
        let mut diagnosis = self.fhe.decrypt(&encrypted_result, &keys.client)?;

        if let Some(patient_id) = &patient.id {
            diagnosis.patient_id = Some(patient_id.clone());
        }

        tracing::debug!("Step 4: Saving diagnosis to storage...");
        if let Err(e) = self.storage.save_diagnosis(&diagnosis) {
            tracing::warn!("Failed to save diagnosis: {:?}", e);
        }

        tracing::info!(
            "Inference complete: screening_positive={}, calibrated_probability={:.4}, threshold={:.4}, risk={}",
            diagnosis.result.screening_positive,
            diagnosis.result.probability,
            diagnosis.result.threshold_used,
            diagnosis.risk_level
        );

        Ok(diagnosis)
    }

    pub fn get_recent_diagnoses(&self, limit: usize) -> Result<Vec<Diagnosis>, PulsecureError> {
        self.storage
            .load_recent_diagnoses(limit)
            .map_err(|e| PulsecureError::Storage(e.into()))
    }

    pub fn get_diagnosis_count(&self) -> Result<usize, PulsecureError> {
        self.storage
            .count_diagnoses()
            .map_err(|e| PulsecureError::Storage(e.into()))
    }

    pub fn regenerate_keys(&mut self) -> Result<(), PulsecureError> {
        tracing::warn!("Regenerating FHE keys...");

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
    #[ignore = "initializes real FHE keys; run manually on high-memory machines"]
    fn test_initialization() {
        let mut service = create_test_service();
        assert!(!service.is_initialized());

        service.initialize().expect("Should initialize");
        assert!(service.is_initialized());
        assert!(service.key_fingerprints().is_some());
    }

    #[test]
    #[ignore = "runs full real FHE inference; run manually on high-memory machines"]
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
            ..Default::default()
        });

        let diagnosis = service
            .run_inference(patient)
            .expect("Should run inference");

        assert!(diagnosis.result.probability >= 0.0);
        assert!(diagnosis.result.probability <= 1.0);
        assert!(diagnosis.encrypted_computation);
    }
}
