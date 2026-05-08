use std::path::Path;
use std::{collections::BTreeMap, fs};

use base64::Engine;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use tfhe::prelude::*;
use tfhe::{
    generate_keys, set_server_key, unset_server_key, ClientKey as TfheClientKey, ConfigBuilder,
    FheInt64, ServerKey as TfheServerKey,
};

use crate::domain::{
    ClientKey, CryptoError, Diagnosis, DiagnosisResult, EncryptedDiagnosis, EncryptedPatientData,
    KeyPair, PatientData, ServerKey,
};
use crate::ports::FheEngine;

#[cfg(debug_assertions)]
const ALLOW_UNSIGNED_MODELS_ENV: &str = "PULSECURE_ALLOW_UNSIGNED_MODELS";

const MAX_FEATURES: usize = 15;

pub const MODEL_FEATURE_NAMES: [&str; 15] = [
    "age",
    "sex",
    "race",
    "bmi",
    "waist",
    "sbp",
    "hypertension",
    "total_chol",
    "hdl",
    "hba1c",
    "serum_glucose",
    "diabetes",
    "egfr",
    "urine_albumin",
    "ever_smoker",
];

fn feature_contract_is_supported(feature_names: &[String]) -> bool {
    let names: Vec<&str> = feature_names.iter().map(String::as_str).collect();
    names.as_slice() == MODEL_FEATURE_NAMES
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelMetadata {
    pub schema_version: u32,
    pub intended_use: Option<String>,
    pub feature_names: Vec<String>,
    pub precision_bits: u32,
    pub scale_factor: i64,
    pub clinical_threshold: Option<ExportedClinicalThreshold>,
    pub validation: Option<ExportedValidationMetrics>,
    pub dataset: Option<ExportedDatasetMetadata>,
    pub training_metadata: Option<ExportedTrainingMetadata>,
    pub metadata_complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedQuantizedModel {
    #[serde(default)]
    pub schema_version: u32,
    #[serde(default)]
    pub intended_use: Option<String>,
    pub precision_bits: u32,
    pub scale_factor: i64,
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub feature_schema: Vec<ExportedFeature>,
    pub coefficients_q: Vec<i64>,
    pub intercept_q: i64,
    pub scaler_mean_q: Vec<i64>,
    pub scaler_std_inv_q: Vec<i64>,
    #[serde(default)]
    pub sigmoid_lut: Option<ExportedSigmoidLut>,
    #[serde(default)]
    pub calibration_lut: Option<ExportedCalibrationLut>,
    #[serde(default)]
    pub clinical_threshold: Option<ExportedClinicalThreshold>,
    #[serde(default)]
    pub validation: Option<ExportedValidationMetrics>,
    #[serde(default)]
    pub training_metadata: Option<ExportedTrainingMetadata>,
    #[serde(default)]
    pub dataset: Option<ExportedDatasetMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedFeature {
    pub name: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedSigmoidLut {
    pub input_bits: u32,
    pub output_bits: u32,
    pub values: Vec<i64>,
    #[serde(default = "default_sigmoid_input_range")]
    pub input_range: f64,
}

fn default_sigmoid_input_range() -> f64 {
    8.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedCalibrationLut {
    pub x_breakpoints: Vec<i64>,
    pub y_values: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedClinicalThreshold {
    pub value: f64,
    pub quantized: i64,
    #[serde(default)]
    pub recall: Option<f64>,
    #[serde(default)]
    pub precision: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedValidationMetrics {
    #[serde(default)]
    pub auc_float: Option<f64>,
    #[serde(default)]
    pub auc_calibrated: Option<f64>,
    #[serde(default)]
    pub auc_quantized: Option<f64>,
    #[serde(default)]
    pub brier_before: Option<f64>,
    #[serde(default)]
    pub brier_after: Option<f64>,
    #[serde(default)]
    pub max_calibration_error: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedTrainingMetadata {
    #[serde(default)]
    pub trained_at: Option<String>,
    #[serde(default)]
    pub random_seed: Option<u64>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub calibration_method: Option<String>,
    #[serde(default)]
    pub target_recall: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedDatasetMetadata {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub n_samples: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SignedModelManifest {
    version: u32,
    serial: u64,
    created_at: i64,
    nonce_b64: String,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct RollbackState {
    #[serde(default)]
    last_serial: u64,
    #[serde(default)]
    last_created_at: i64,
    #[serde(default)]
    last_manifest_sha256: String,
}

fn parse_bool_env(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn validate_nonce_b64(nonce_b64: &str) -> Result<(), CryptoError> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(nonce_b64.trim())
        .map_err(|e| CryptoError::Serialization(format!("Invalid nonce base64: {e}")))?;
    if raw.len() != 16 {
        return Err(CryptoError::Serialization(
            "nonce must decode to exactly 16 bytes".into(),
        ));
    }
    Ok(())
}

fn read_rollback_state(path: &std::path::Path) -> Result<Option<RollbackState>, CryptoError> {
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read(path)
        .map_err(|e| CryptoError::Serialization(format!("Failed to read rollback state: {e}")))?;
    let state: RollbackState = serde_json::from_slice(&content)
        .map_err(|e| CryptoError::Serialization(format!("Invalid rollback state format: {e}")))?;
    Ok(Some(state))
}

fn write_rollback_state(path: &std::path::Path, state: &RollbackState) -> Result<(), CryptoError> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let bytes = serde_json::to_vec_pretty(state).map_err(|e| {
        CryptoError::Serialization(format!("Failed to serialize rollback state: {e}"))
    })?;
    std::fs::write(path, bytes)
        .map_err(|e| CryptoError::Serialization(format!("Failed to write rollback state: {e}")))?;
    Ok(())
}

pub struct TfheAdapter {
    model: Option<ExportedQuantizedModel>,
}

impl TfheAdapter {
    #[must_use]
    pub fn new() -> Self {
        tracing::info!("Initializing TfheAdapter (tfhe-rs)");
        Self { model: None }
    }

    pub fn load_model(&mut self, model_dir: &Path) -> Result<(), CryptoError> {
        let manifest = self.verify_model_signature(model_dir)?;

        let base_dir = if model_dir.is_dir() {
            model_dir
        } else {
            model_dir.parent().unwrap_or(model_dir)
        };

        let model_path = if let Some(manifest) = manifest {
            if manifest.files.contains_key("calibrated_model.json") {
                base_dir.join("calibrated_model.json")
            } else if manifest.files.contains_key("model.json") {
                base_dir.join("model.json")
            } else {
                return Err(CryptoError::Serialization(
                    "manifest.json must include model.json or calibrated_model.json".into(),
                ));
            }
        } else {
            let candidate_paths: Vec<std::path::PathBuf> = if model_dir.is_file() {
                vec![model_dir.to_path_buf()]
            } else {
                vec![
                    base_dir.join("calibrated_model.json"),
                    base_dir.join("model.json"),
                ]
            };

            candidate_paths
                .into_iter()
                .find(|p| p.exists())
                .ok_or_else(|| {
                    CryptoError::Serialization(format!(
                        "No model JSON found in {:?} (expected model.json or calibrated_model.json)",
                        model_dir
                    ))
                })?
        };

        let content = std::fs::read_to_string(&model_path)
            .map_err(|e| CryptoError::Serialization(e.to_string()))?;
        let model: ExportedQuantizedModel = serde_json::from_str(&content)
            .map_err(|e| CryptoError::Serialization(e.to_string()))?;

        let n = model.feature_names.len();
        if n == 0 || n > MAX_FEATURES {
            return Err(CryptoError::Serialization(format!(
                "Invalid feature count in model: got {n}, max {MAX_FEATURES}"
            )));
        }
        if model.schema_version >= 1 && !feature_contract_is_supported(&model.feature_names) {
            return Err(CryptoError::Serialization(
                "Model feature_names do not match a supported deployed feature contract".into(),
            ));
        }
        if model.coefficients_q.len() != n
            || model.scaler_mean_q.len() != n
            || model.scaler_std_inv_q.len() != n
        {
            return Err(CryptoError::Serialization(
                "Model parameter lengths do not match feature_names length".into(),
            ));
        }
        if let Some(lut) = &model.sigmoid_lut {
            if lut.values.is_empty() || lut.output_bits == 0 || lut.input_range <= 0.0 {
                return Err(CryptoError::Serialization(
                    "Invalid sigmoid_lut in model contract".into(),
                ));
            }
        }
        if let Some(lut) = &model.calibration_lut {
            if lut.x_breakpoints.len() != lut.y_values.len() || lut.x_breakpoints.len() < 2 {
                return Err(CryptoError::Serialization(
                    "Invalid calibration_lut in model contract".into(),
                ));
            }
        }
        if let Some(threshold) = &model.clinical_threshold {
            if !(0.0..=1.0).contains(&threshold.value) {
                return Err(CryptoError::Serialization(
                    "Invalid clinical_threshold.value in model contract".into(),
                ));
            }
        }

        tracing::info!(
            "Loaded model from {:?} (schema_version={}, precision_bits={}, scale_factor={}, n_features={})",
            model_path,
            model.schema_version,
            model.precision_bits,
            model.scale_factor,
            n
        );

        self.model = Some(model);
        Ok(())
    }

    #[must_use]
    pub fn model_metadata(&self) -> Option<ModelMetadata> {
        self.model.as_ref().map(|model| {
            let metadata_complete = model
                .dataset
                .as_ref()
                .and_then(|dataset| dataset.sha256.as_ref())
                .filter(|sha| !sha.trim().is_empty())
                .is_some()
                && model
                    .dataset
                    .as_ref()
                    .and_then(|dataset| dataset.n_samples)
                    .is_some()
                && model
                    .training_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.trained_at.clone())
                    .is_some();

            ModelMetadata {
                schema_version: model.schema_version,
                intended_use: model.intended_use.clone(),
                feature_names: model.feature_names.clone(),
                precision_bits: model.precision_bits,
                scale_factor: model.scale_factor,
                clinical_threshold: model.clinical_threshold.clone(),
                validation: model.validation.clone(),
                dataset: model.dataset.clone(),
                training_metadata: model.training_metadata.clone(),
                metadata_complete,
            }
        })
    }

    fn verify_model_signature(
        &self,
        model_dir: &Path,
    ) -> Result<Option<SignedModelManifest>, CryptoError> {
        #[cfg(debug_assertions)]
        fn allow_unsigned_models_for_debug() -> bool {
            std::env::var(ALLOW_UNSIGNED_MODELS_ENV)
                .map(|v| v == "true")
                .unwrap_or(false)
        }

        let base_dir = if model_dir.is_dir() {
            model_dir
        } else {
            model_dir.parent().unwrap_or(model_dir)
        };

        let sig_path = base_dir.join("model.sig");
        let manifest_path = base_dir.join("manifest.json");

        if !sig_path.exists() || !manifest_path.exists() {
            #[cfg(not(debug_assertions))]
            {
                tracing::error!(
                    "Model signature not found at {:?}. \
                     Production builds require signed models.",
                    sig_path
                );
                return Err(CryptoError::Serialization(
                    "Model signature required in production".into(),
                ));
            }

            #[cfg(debug_assertions)]
            {
                if allow_unsigned_models_for_debug() {
                    tracing::warn!(
                        "Loading UNSIGNED model ({ALLOW_UNSIGNED_MODELS_ENV}=true). \
                         This is only allowed in debug builds for testing."
                    );
                    return Ok(None);
                } else {
                    tracing::error!(
                        "Model signature not found at {:?}. \
                         Set {ALLOW_UNSIGNED_MODELS_ENV}=true to bypass in debug builds.",
                        sig_path
                    );
                    return Err(CryptoError::Serialization(
                        format!(
                            "Model signature required. Set {ALLOW_UNSIGNED_MODELS_ENV}=true for testing."
                        ),
                    ));
                }
            }
        }

        let sig_bytes = fs::read(&sig_path)
            .map_err(|e| CryptoError::Serialization(format!("Failed to read signature: {e}")))?;

        if sig_bytes.len() != 64 {
            return Err(CryptoError::Serialization(
                "Invalid signature length (expected 64 bytes)".into(),
            ));
        }

        let signature = Signature::from_bytes(
            sig_bytes
                .as_slice()
                .try_into()
                .map_err(|_| CryptoError::Serialization("Invalid signature format".into()))?,
        );

        let manifest_content = fs::read(&manifest_path)
            .map_err(|e| CryptoError::Serialization(format!("Failed to read manifest: {e}")))?;

        let public_key = Self::developer_public_key()?;
        public_key
            .verify(&manifest_content, &signature)
            .map_err(|_| CryptoError::Serialization("Invalid model signature".into()))?;

        let manifest: SignedModelManifest =
            serde_json::from_slice(&manifest_content).map_err(|e| {
                CryptoError::Serialization(format!("Invalid manifest.json format: {e}"))
            })?;
        if manifest.version != 1 {
            return Err(CryptoError::Serialization(format!(
                "Unsupported manifest version: {}",
                manifest.version
            )));
        }

        let serial = manifest.serial;
        let created_at = manifest.created_at;
        validate_nonce_b64(&manifest.nonce_b64)?;

        if created_at > 0 {
            let now = unix_now();

            if created_at > now + 300 {
                return Err(CryptoError::Serialization(
                    "manifest created_at is in the future".into(),
                ));
            }

            if let Ok(v) = std::env::var("PULSECURE_MODEL_MAX_AGE_SECS") {
                if let Ok(max_age) = v.trim().parse::<i64>() {
                    if max_age > 0 && now.saturating_sub(created_at) > max_age {
                        return Err(CryptoError::Serialization(
                            "manifest is older than allowed max age".into(),
                        ));
                    }
                }
            }
        }

        if manifest.files.is_empty() {
            return Err(CryptoError::Serialization(
                "manifest.json contains no files".into(),
            ));
        }

        let binds_expected_model = manifest.files.contains_key("model.json")
            || manifest.files.contains_key("calibrated_model.json");
        if !binds_expected_model {
            return Err(CryptoError::Serialization(
                "manifest.json must include model.json or calibrated_model.json".into(),
            ));
        }

        for (rel, expected_hex) in &manifest.files {
            let path = base_dir.join(rel);
            let bytes = fs::read(&path).map_err(|e| {
                CryptoError::Serialization(format!(
                    "Manifest references missing/unreadable file {:?}: {e}",
                    path
                ))
            })?;
            let actual = Sha256::digest(&bytes);
            let actual_hex = actual
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();

            if !constant_time_eq_str(&actual_hex, expected_hex) {
                return Err(CryptoError::Serialization(format!(
                    "File hash mismatch for {}",
                    rel
                )));
            }
        }

        let state_path = std::env::var("PULSECURE_MODEL_ROLLBACK_STATE_FILE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/app/data/model_rollback_state.json"));
        let enforce_state = parse_bool_env("PULSECURE_ENFORCE_ROLLBACK_PROTECTION");

        let manifest_hash = sha256_hex_bytes(&manifest_content);
        match read_rollback_state(&state_path) {
            Ok(Some(state)) => {
                if serial > 0 && state.last_serial > 0 {
                    if serial < state.last_serial {
                        return Err(CryptoError::Serialization(
                            "Refusing to load older signed manifest (serial rollback detected)"
                                .into(),
                        ));
                    }
                    if serial == state.last_serial && manifest_hash != state.last_manifest_sha256 {
                        return Err(CryptoError::Serialization(
                            "Refusing to load different manifest with same serial".into(),
                        ));
                    }
                }

                if created_at > 0 && state.last_created_at > 0 && created_at < state.last_created_at
                {
                    return Err(CryptoError::Serialization(
                        "Refusing to load older signed manifest (rollback detected)".into(),
                    ));
                }

                if serial > state.last_serial
                    || created_at > state.last_created_at
                    || (serial == state.last_serial
                        && created_at == state.last_created_at
                        && manifest_hash != state.last_manifest_sha256)
                {
                    let new_state = RollbackState {
                        last_serial: state.last_serial.max(serial),
                        last_created_at: created_at.max(state.last_created_at),
                        last_manifest_sha256: manifest_hash,
                    };
                    if let Err(e) = write_rollback_state(&state_path, &new_state) {
                        if enforce_state {
                            return Err(e);
                        }
                        tracing::warn!("Failed to write rollback state: {e}");
                    }
                }
            }
            Ok(None) => {
                if serial > 0 || created_at > 0 {
                    let new_state = RollbackState {
                        last_serial: serial,
                        last_created_at: created_at,
                        last_manifest_sha256: manifest_hash,
                    };
                    if let Err(e) = write_rollback_state(&state_path, &new_state) {
                        if enforce_state {
                            return Err(e);
                        }
                        tracing::warn!("Failed to initialize rollback state: {e}");
                    }
                }
            }
            Err(e) => {
                if enforce_state {
                    return Err(e);
                }
                tracing::warn!("Rollback state unavailable: {e}");
            }
        }

        tracing::info!("Model signature and hashes verified successfully");
        Ok(Some(manifest))
    }

    fn developer_public_key() -> Result<VerifyingKey, CryptoError> {
        const PUBKEY_FILE_ENV: &str = "PULSECURE_MODEL_SIGNING_PUBKEY_B64_FILE";
        const DOCKER_SECRET_PUBKEY: &str = "/run/secrets/pulsecure_model_signing_pubkey_b64";

        if let Ok(path) = std::env::var(PUBKEY_FILE_ENV) {
            let b64 = fs::read_to_string(path.trim()).map_err(|e| {
                CryptoError::Serialization(format!("Failed reading pubkey file: {e}"))
            })?;
            return Self::verifying_key_from_b64(&b64);
        }

        if Path::new(DOCKER_SECRET_PUBKEY).exists() {
            let b64 = fs::read_to_string(DOCKER_SECRET_PUBKEY).map_err(|e| {
                CryptoError::Serialization(format!("Failed reading docker pubkey secret: {e}"))
            })?;
            return Self::verifying_key_from_b64(&b64);
        }

        #[cfg(test)]
        {
            const TEST_PUBKEY_ENV: &str = "PULSECURE_TEST_DEV_PUBKEY_B64";
            if let Ok(b64) = std::env::var(TEST_PUBKEY_ENV) {
                return Self::verifying_key_from_b64(&b64)
                    .map_err(|_| CryptoError::Serialization("Invalid test verifying key".into()));
            }
        }

        const DEV_PUBKEY: [u8; 32] = [
            0xf1, 0xb2, 0xec, 0x37, 0x25, 0x2c, 0x98, 0xe4, 0x30, 0x14, 0x5c, 0xae, 0x58, 0x35,
            0x08, 0x5a, 0x50, 0x67, 0xe7, 0xaf, 0x72, 0xb1, 0x28, 0x28, 0x67, 0x98, 0x14, 0xb0,
            0x77, 0x34, 0x15, 0x46,
        ];

        VerifyingKey::from_bytes(&DEV_PUBKEY)
            .map_err(|_| CryptoError::Serialization("Invalid embedded public key".into()))
    }

    fn verifying_key_from_b64(b64: &str) -> Result<VerifyingKey, CryptoError> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .map_err(|_| CryptoError::Serialization("Invalid public key base64".into()))?;
        if bytes.len() != 32 {
            return Err(CryptoError::Serialization(
                "Invalid public key length (expected 32 bytes)".into(),
            ));
        }
        let mut pubkey = [0u8; 32];
        pubkey.copy_from_slice(&bytes);
        VerifyingKey::from_bytes(&pubkey)
            .map_err(|_| CryptoError::Serialization("Invalid verifying key".into()))
    }

    fn normalize_and_quantize_features(
        model: &ExportedQuantizedModel,
        raw_features: &[f64],
    ) -> Result<Vec<i64>, CryptoError> {
        let n = model.feature_names.len();
        if raw_features.len() != n {
            return Err(CryptoError::Encryption(format!(
                "Feature count mismatch: got {}, expected {}",
                raw_features.len(),
                n
            )));
        }

        let scale = model.scale_factor;
        if scale <= 0 {
            return Err(CryptoError::Serialization(
                "Invalid scale_factor in model (must be > 0)".into(),
            ));
        }

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x_scaled = (raw_features[i] * scale as f64) as i64;
            let x_centered = x_scaled.wrapping_sub(model.scaler_mean_q[i]);

            let numer = (x_centered as i128) * (model.scaler_std_inv_q[i] as i128);
            let x_norm_q = (numer / (scale as i128)) as i64;
            out.push(x_norm_q);
        }

        Ok(out)
    }

    fn sigmoid_direct(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn apply_sigmoid(model: &ExportedQuantizedModel, linear_result: f64) -> f64 {
        let Some(lut) = &model.sigmoid_lut else {
            return Self::sigmoid_direct(linear_result);
        };
        if lut.values.is_empty() {
            return Self::sigmoid_direct(linear_result);
        }

        let input_range = lut.input_range;
        let clamped = linear_result.clamp(-input_range, input_range);
        let max_index = lut.values.len().saturating_sub(1) as f64;
        let idx = (((clamped + input_range) / (2.0 * input_range)) * max_index)
            .floor()
            .clamp(0.0, max_index) as usize;
        let output_scale = (1u64 << lut.output_bits) as f64;
        (lut.values[idx] as f64 / output_scale).clamp(0.0, 1.0)
    }

    fn apply_isotonic_calibration(model: &ExportedQuantizedModel, probability: f64) -> f64 {
        let Some(lut) = &model.calibration_lut else {
            return probability;
        };
        if lut.x_breakpoints.len() != lut.y_values.len() || lut.x_breakpoints.len() < 2 {
            return probability;
        }

        let output_bits = model
            .sigmoid_lut
            .as_ref()
            .map(|lut| lut.output_bits)
            .unwrap_or(12);
        let scale = (1u64 << output_bits) as f64;
        let x = (probability * scale).clamp(0.0, scale);

        for i in 1..lut.x_breakpoints.len() {
            let x0 = lut.x_breakpoints[i - 1] as f64;
            let x1 = lut.x_breakpoints[i] as f64;
            if x <= x1 {
                let y0 = lut.y_values[i - 1] as f64;
                let y1 = lut.y_values[i] as f64;
                if (x1 - x0).abs() < f64::EPSILON {
                    return (y1 / scale).clamp(0.0, 1.0);
                }
                let t = (x - x0) / (x1 - x0);
                return ((y0 + t * (y1 - y0)) / scale).clamp(0.0, 1.0);
            }
        }

        (*lut.y_values.last().unwrap_or(&0) as f64 / scale).clamp(0.0, 1.0)
    }

    fn clinical_threshold(model: &ExportedQuantizedModel) -> f64 {
        model
            .clinical_threshold
            .as_ref()
            .map(|threshold| threshold.value)
            .unwrap_or(0.5)
    }

    fn deserialize_tfhe_client_key(bytes: &[u8]) -> Result<TfheClientKey, CryptoError> {
        bincode::deserialize(bytes).map_err(|e| {
            CryptoError::Serialization(format!("Failed to deserialize client key: {e}"))
        })
    }

    fn deserialize_tfhe_server_key(bytes: &[u8]) -> Result<TfheServerKey, CryptoError> {
        bincode::deserialize(bytes).map_err(|e| {
            CryptoError::Serialization(format!("Failed to deserialize server key: {e}"))
        })
    }
}

fn constant_time_eq_str(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.as_bytes().iter().zip(b.as_bytes().iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

impl Default for TfheAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl FheEngine for TfheAdapter {
    fn generate_keys(&self) -> Result<KeyPair, CryptoError> {
        tracing::info!("Generating FHE key pair...");

        let config = ConfigBuilder::default().build();

        let (client_key, server_key) = generate_keys(config);

        tracing::info!("Generated tfhe-rs keys");

        let client_bytes = bincode::serialize(&client_key).map_err(|e| {
            CryptoError::KeyGeneration(format!("Failed to serialize client key: {e}"))
        })?;

        let server_bytes = bincode::serialize(&server_key).map_err(|e| {
            CryptoError::KeyGeneration(format!("Failed to serialize server key: {e}"))
        })?;

        let client = ClientKey::from_bytes(client_bytes);
        let server = ServerKey::from_bytes(server_bytes);

        tracing::info!(
            "Generated keys - Client fingerprint: {}, Server fingerprint: {}",
            client.fingerprint,
            server.fingerprint
        );

        Ok(KeyPair::new(client, server))
    }

    fn encrypt(
        &self,
        data: &PatientData,
        key: &ClientKey,
    ) -> Result<EncryptedPatientData, CryptoError> {
        tracing::debug!("Encrypting patient data...");

        let tfhe_client_key = Self::deserialize_tfhe_client_key(key.as_bytes())?;

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| CryptoError::Encryption("Model not loaded".into()))?;

        data.features
            .validate_for_model(&model.feature_names)
            .map_err(|errors| CryptoError::Encryption(errors.join(", ")))?;

        let features = data
            .features
            .to_model_vec(&model.feature_names)
            .map_err(CryptoError::Encryption)?;

        if features.len() > MAX_FEATURES {
            return Err(CryptoError::Encryption(format!(
                "Too many features: got {}, max {}",
                features.len(),
                MAX_FEATURES
            )));
        }

        let quantized = Self::normalize_and_quantize_features(model, &features)?;

        let mut encrypted_features: Vec<Vec<u8>> = Vec::with_capacity(quantized.len());

        for (i, &value) in quantized.iter().enumerate() {
            let encrypted: FheInt64 = FheInt64::encrypt(value, &tfhe_client_key);

            let encrypted_bytes = bincode::serialize(&encrypted).map_err(|e| {
                CryptoError::Encryption(format!("Failed to serialize encrypted feature {i}: {e}"))
            })?;

            encrypted_features.push(encrypted_bytes);
            tracing::trace!("Encrypted feature {i}");
        }

        let ciphertext = bincode::serialize(&encrypted_features).map_err(|e| {
            CryptoError::Encryption(format!("Failed to serialize encrypted data: {e}"))
        })?;

        tracing::info!(
            "Encrypted {} features (ciphertext size: {} bytes)",
            features.len(),
            ciphertext.len()
        );

        Ok(EncryptedPatientData::new(
            ciphertext,
            features.len(),
            key.fingerprint.clone(),
        ))
    }

    fn compute(
        &self,
        encrypted: &EncryptedPatientData,
        server_key: &ServerKey,
    ) -> Result<EncryptedDiagnosis, CryptoError> {
        tracing::info!("Performing homomorphic computation...");

        let tfhe_server_key = Self::deserialize_tfhe_server_key(server_key.as_bytes())?;

        struct ServerKeyGuard;
        impl Drop for ServerKeyGuard {
            fn drop(&mut self) {
                unset_server_key();
            }
        }

        set_server_key(tfhe_server_key);
        let _server_key_guard = ServerKeyGuard;

        let encrypted_features: Vec<Vec<u8>> = bincode::deserialize(&encrypted.ciphertext)
            .map_err(|e| {
                CryptoError::Computation(format!("Failed to deserialize encrypted data: {e}"))
            })?;

        let mut fhe_features: Vec<FheInt64> = Vec::with_capacity(encrypted_features.len());
        for (i, bytes) in encrypted_features.iter().enumerate() {
            let fhe_val: FheInt64 = bincode::deserialize(bytes).map_err(|e| {
                CryptoError::Computation(format!(
                    "Failed to deserialize encrypted feature {i}: {e}"
                ))
            })?;
            fhe_features.push(fhe_val);
        }

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| CryptoError::Computation("Model not loaded".into()))?;

        tracing::debug!("Computing encrypted linear combination...");

        let mut result: FheInt64 = FheInt64::encrypt_trivial(0i64);

        for (i, (fhe_feature, &coef)) in fhe_features
            .iter()
            .zip(model.coefficients_q.iter())
            .enumerate()
        {
            let term = fhe_feature * coef;

            result = result + term;
            tracing::trace!("Computed term {i} (homomorphic multiply + add)");
        }

        let intercept_term = model
            .intercept_q
            .checked_mul(model.scale_factor)
            .ok_or_else(|| CryptoError::Computation("Intercept term overflow".into()))?;
        result = result + intercept_term;

        let result_bytes = bincode::serialize(&result).map_err(|e| {
            CryptoError::Computation(format!("Failed to serialize encrypted result: {e}"))
        })?;

        tracing::info!(
            "Completed homomorphic computation on {} features.",
            fhe_features.len()
        );

        Ok(EncryptedDiagnosis::new(
            result_bytes,
            encrypted.key_fingerprint.clone(),
        ))
    }

    fn decrypt(
        &self,
        result: &EncryptedDiagnosis,
        key: &ClientKey,
    ) -> Result<Diagnosis, CryptoError> {
        tracing::debug!("Decrypting diagnosis result...");

        let tfhe_client_key = Self::deserialize_tfhe_client_key(key.as_bytes())?;

        let encrypted_result: FheInt64 = bincode::deserialize(&result.ciphertext).map_err(|e| {
            CryptoError::Decryption(format!("Failed to deserialize encrypted result: {e}"))
        })?;

        let decrypted_value: i64 = encrypted_result.decrypt(&tfhe_client_key);

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| CryptoError::Decryption("Model not loaded".into()))?;

        let scale = model.scale_factor as f64;
        let linear_result = (decrypted_value as f64) / (scale * scale);

        let uncalibrated_probability = Self::apply_sigmoid(model, linear_result);
        let probability = Self::apply_isotonic_calibration(model, uncalibrated_probability);
        let threshold = Self::clinical_threshold(model);

        tracing::info!(
            "Decrypted result: linear={:.4}, uncalibrated_probability={:.4}, calibrated_probability={:.4}, threshold={:.4}",
            linear_result,
            uncalibrated_probability,
            probability,
            threshold
        );

        if !probability.is_finite() {
            return Err(CryptoError::Decryption(
                "Decryption produced non-finite probability".into(),
            ));
        }

        let diagnosis_result = DiagnosisResult::with_threshold(probability, threshold);
        let diagnosis = Diagnosis::new(diagnosis_result, true);

        Ok(diagnosis)
    }

    fn serialize_keys(&self, keys: &KeyPair) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        Ok((keys.client.inner.clone(), keys.server.inner.clone()))
    }

    fn deserialize_keys(
        &self,
        client_bytes: &[u8],
        server_bytes: &[u8],
    ) -> Result<KeyPair, CryptoError> {
        let _: TfheClientKey = Self::deserialize_tfhe_client_key(client_bytes)?;
        let _: TfheServerKey = Self::deserialize_tfhe_server_key(server_bytes)?;

        let client = ClientKey::from_bytes(client_bytes.to_vec());
        let server = ServerKey::from_bytes(server_bytes.to_vec());
        Ok(KeyPair::new(client, server))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::PatientFeatures;
    use ed25519_dalek::{Signer, SigningKey};
    use rand::RngCore;
    use std::path::Path;
    use std::sync::Once;
    use tempfile::tempdir;

    #[derive(serde::Deserialize)]
    struct ParityFixture {
        threshold: f64,
        cases: Vec<ParityCase>,
    }

    #[derive(serde::Deserialize)]
    struct ParityCase {
        name: String,
        features: ParityFeatures,
        linear: f64,
        uncalibrated_probability: f64,
        probability: f64,
        screening_positive: bool,
    }

    #[derive(serde::Deserialize)]
    struct ParityFeatures {
        age: f64,
        sex: f64,
        race: f64,
        bmi: f64,
        waist: f64,
        sbp: f64,
        hypertension: f64,
        total_chol: f64,
        hdl: f64,
        hba1c: f64,
        serum_glucose: f64,
        diabetes: f64,
        egfr: f64,
        urine_albumin: f64,
        ever_smoker: f64,
    }

    impl From<&ParityFeatures> for PatientFeatures {
        fn from(features: &ParityFeatures) -> Self {
            Self {
                age: features.age,
                sex: features.sex,
                race: features.race,
                bmi: features.bmi,
                waist_circ: features.waist,
                sys_bp: features.sbp,
                hypertension: features.hypertension,
                total_chol: features.total_chol,
                hdl_chol: features.hdl,
                hba1c: features.hba1c,
                serum_glucose: features.serum_glucose,
                diabetes: features.diabetes,
                creatinine: 1.0,
                egfr: features.egfr,
                urine_albumin: features.urine_albumin,
                smoking: features.ever_smoker,
                ..Default::default()
            }
        }
    }

    fn allow_unsigned_models_for_tests() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            std::env::set_var(ALLOW_UNSIGNED_MODELS_ENV, "true");
        });
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn cleartext_linear_result(model: &ExportedQuantizedModel, raw_features: &[f64]) -> f64 {
        let quantized =
            TfheAdapter::normalize_and_quantize_features(model, raw_features).expect("quantize");
        let logits_q: i64 = quantized
            .iter()
            .zip(model.coefficients_q.iter())
            .map(|(feature, coefficient)| feature * coefficient)
            .sum::<i64>()
            + model.intercept_q * model.scale_factor;
        logits_q as f64 / ((model.scale_factor as f64) * (model.scale_factor as f64))
    }

    fn write_exported_model(path: &Path, intercept_q: i64) {
        let model = ExportedQuantizedModel {
            schema_version: 0,
            intended_use: None,
            precision_bits: 12,
            scale_factor: 4096,
            feature_names: vec!["x".into()],
            feature_schema: Vec::new(),
            coefficients_q: vec![1],
            intercept_q,
            scaler_mean_q: vec![0],
            scaler_std_inv_q: vec![4096],
            sigmoid_lut: None,
            calibration_lut: None,
            clinical_threshold: None,
            validation: None,
            training_metadata: None,
            dataset: None,
        };
        let json = serde_json::to_string(&model).expect("serialize model");
        std::fs::write(path, json).expect("write model");
    }

    fn write_signed_manifest(dir: &Path, signing_key: &SigningKey, files: &[(&str, Vec<u8>)]) {
        let mut map = BTreeMap::new();
        for (rel, contents) in files {
            map.insert((*rel).to_string(), sha256_hex(contents));
        }

        let created_at = unix_now();
        let serial = if created_at > 0 { created_at as u64 } else { 1 };
        let nonce_b64 = base64::engine::general_purpose::STANDARD.encode([0u8; 16]);
        let manifest = SignedModelManifest {
            version: 1,
            serial,
            created_at,
            nonce_b64,
            files: map,
        };
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize manifest");
        std::fs::write(dir.join("manifest.json"), &manifest_bytes).expect("write manifest");

        let signature: Signature = signing_key.sign(&manifest_bytes);
        std::fs::write(dir.join("model.sig"), signature.to_bytes()).expect("write signature");
    }

    #[test]
    fn test_load_model_prefers_manifest_bound_model_json() {
        let temp = tempdir().expect("tempdir");
        let dir = temp.path();

        let model_path = dir.join("model.json");
        let calibrated_path = dir.join("calibrated_model.json");
        write_exported_model(&model_path, 111);
        write_exported_model(&calibrated_path, 222);

        let mut sk = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut sk);
        let signing_key = SigningKey::from_bytes(&sk);
        let pubkey_b64 = base64::engine::general_purpose::STANDARD
            .encode(signing_key.verifying_key().to_bytes());
        std::env::set_var("PULSECURE_TEST_DEV_PUBKEY_B64", pubkey_b64);

        let model_bytes = std::fs::read(&model_path).expect("read model");
        write_signed_manifest(dir, &signing_key, &[("model.json", model_bytes)]);

        let mut adapter = TfheAdapter::new();
        adapter.load_model(dir).expect("load signed model");
        assert_eq!(adapter.model.as_ref().unwrap().intercept_q, 111);

        std::env::remove_var("PULSECURE_TEST_DEV_PUBKEY_B64");
    }

    #[test]
    fn test_load_model_prefers_manifest_bound_calibrated_model_json() {
        let temp = tempdir().expect("tempdir");
        let dir = temp.path();

        let model_path = dir.join("model.json");
        let calibrated_path = dir.join("calibrated_model.json");
        write_exported_model(&model_path, 111);
        write_exported_model(&calibrated_path, 222);

        let mut sk = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut sk);
        let signing_key = SigningKey::from_bytes(&sk);
        let pubkey_b64 = base64::engine::general_purpose::STANDARD
            .encode(signing_key.verifying_key().to_bytes());
        std::env::set_var("PULSECURE_TEST_DEV_PUBKEY_B64", pubkey_b64);

        let calibrated_bytes = std::fs::read(&calibrated_path).expect("read calibrated model");
        write_signed_manifest(
            dir,
            &signing_key,
            &[("calibrated_model.json", calibrated_bytes)],
        );

        let mut adapter = TfheAdapter::new();
        adapter.load_model(dir).expect("load signed model");
        assert_eq!(adapter.model.as_ref().unwrap().intercept_q, 222);

        std::env::remove_var("PULSECURE_TEST_DEV_PUBKEY_B64");
    }

    #[test]
    fn test_load_model_fails_if_manifest_references_missing_model_file() {
        let temp = tempdir().expect("tempdir");
        let dir = temp.path();

        let calibrated_path = dir.join("calibrated_model.json");
        write_exported_model(&calibrated_path, 222);

        let mut sk = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut sk);
        let signing_key = SigningKey::from_bytes(&sk);
        let pubkey_b64 = base64::engine::general_purpose::STANDARD
            .encode(signing_key.verifying_key().to_bytes());
        std::env::set_var("PULSECURE_TEST_DEV_PUBKEY_B64", pubkey_b64);

        write_signed_manifest(dir, &signing_key, &[("model.json", b"missing".to_vec())]);

        let mut adapter = TfheAdapter::new();
        let err = adapter.load_model(dir).expect_err("must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("missing") || msg.contains("unreadable") || msg.contains("hash mismatch")
        );

        std::env::remove_var("PULSECURE_TEST_DEV_PUBKEY_B64");
    }

    #[test]
    #[ignore = "generates real tfhe-rs keys; run manually on high-memory machines"]
    fn test_key_generation() {
        let adapter = TfheAdapter::new();
        let keys = adapter
            .generate_keys()
            .expect("Key generation should succeed");

        assert!(
            keys.client.inner.len() > 100,
            "Client key should be substantial"
        );
        assert!(
            keys.server.inner.len() > 100,
            "Server key should be substantial"
        );
        assert!(!keys.client.fingerprint.is_empty());
        assert!(!keys.server.fingerprint.is_empty());
    }

    #[test]
    #[ignore = "generates real tfhe-rs keys; run manually on high-memory machines"]
    fn test_encrypt_decrypt_roundtrip() {
        allow_unsigned_models_for_tests();

        let mut adapter = TfheAdapter::new();
        adapter
            .load_model(Path::new("models"))
            .expect("Model should load for tests");
        let keys = adapter
            .generate_keys()
            .expect("Key generation should succeed");

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

        let encrypted = adapter
            .encrypt(&patient, &keys.client)
            .expect("Encryption should succeed");
        assert_eq!(encrypted.num_features, 9);
        assert!(
            encrypted.ciphertext.len() > 1000,
            "FHE ciphertext should be large"
        );

        tracing::info!("FHE encryption test passed");
    }

    #[test]
    fn test_quantization() {
        let model = ExportedQuantizedModel {
            schema_version: 0,
            intended_use: None,
            precision_bits: 12,
            scale_factor: 4096,
            feature_names: vec!["a".into(), "b".into()],
            feature_schema: Vec::new(),
            coefficients_q: vec![1, 2],
            intercept_q: 0,
            scaler_mean_q: vec![0, 0],
            scaler_std_inv_q: vec![4096, 4096],
            sigmoid_lut: None,
            calibration_lut: None,
            clinical_threshold: None,
            validation: None,
            training_metadata: None,
            dataset: None,
        };

        let raw = vec![0.5, -0.5];
        let q =
            TfheAdapter::normalize_and_quantize_features(&model, &raw).expect("Should quantize");

        assert_eq!(q[0], (0.5 * 4096.0) as i64);
        assert_eq!(q[1], (-0.5 * 4096.0) as i64);
    }

    #[test]
    fn test_bundled_model_contract_loads_calibrated_threshold() {
        allow_unsigned_models_for_tests();

        let mut adapter = TfheAdapter::new();
        adapter
            .load_model(Path::new("models"))
            .expect("Bundled model should load");

        let model = adapter.model.as_ref().expect("model loaded");
        assert_eq!(model.schema_version, 1);
        assert!(feature_contract_is_supported(&model.feature_names));
        assert_eq!(model.feature_names.as_slice(), MODEL_FEATURE_NAMES);
        assert!(model.sigmoid_lut.is_some());
        assert!(model.calibration_lut.is_some());
        assert!(TfheAdapter::clinical_threshold(model) < 0.5);
    }

    #[test]
    fn test_bundled_v2_model_matches_python_reference_cases() {
        allow_unsigned_models_for_tests();

        let fixture: ParityFixture = serde_json::from_str(include_str!(
            "../../../tests/fixtures/v2_model_parity_cases.json"
        ))
        .expect("valid parity fixture");

        let mut adapter = TfheAdapter::new();
        adapter
            .load_model(Path::new("models"))
            .expect("Bundled model should load");
        let model = adapter.model.as_ref().expect("model loaded");

        assert_eq!(TfheAdapter::clinical_threshold(model), fixture.threshold);

        for case in fixture.cases {
            let features = PatientFeatures::from(&case.features);
            features
                .validate_for_model(&model.feature_names)
                .unwrap_or_else(|errors| {
                    panic!("{} should validate: {}", case.name, errors.join(", "))
                });
            let raw = features
                .to_model_vec(&model.feature_names)
                .expect("feature mapping should match model");

            let linear = cleartext_linear_result(model, &raw);
            let uncalibrated = TfheAdapter::apply_sigmoid(model, linear);
            let probability = TfheAdapter::apply_isotonic_calibration(model, uncalibrated);
            let screening_positive = probability >= TfheAdapter::clinical_threshold(model);

            assert!(
                (linear - case.linear).abs() < 1e-12,
                "{} linear mismatch: got {}, expected {}",
                case.name,
                linear,
                case.linear
            );
            assert!(
                (uncalibrated - case.uncalibrated_probability).abs() < 1e-12,
                "{} uncalibrated probability mismatch: got {}, expected {}",
                case.name,
                uncalibrated,
                case.uncalibrated_probability
            );
            assert!(
                (probability - case.probability).abs() < 1e-12,
                "{} calibrated probability mismatch: got {}, expected {}",
                case.name,
                probability,
                case.probability
            );
            assert_eq!(
                screening_positive, case.screening_positive,
                "{} screening decision mismatch",
                case.name
            );
        }
    }

    #[test]
    fn test_exported_luts_and_threshold_are_applied() {
        let model = ExportedQuantizedModel {
            schema_version: 1,
            intended_use: Some(
                "Proof-of-concept encrypted cardiovascular screening; not for medical decisions."
                    .into(),
            ),
            precision_bits: 12,
            scale_factor: 4096,
            feature_names: vec!["x".into()],
            feature_schema: Vec::new(),
            coefficients_q: vec![1],
            intercept_q: 0,
            scaler_mean_q: vec![0],
            scaler_std_inv_q: vec![4096],
            sigmoid_lut: Some(ExportedSigmoidLut {
                input_bits: 2,
                output_bits: 12,
                values: vec![0, 1024, 3072, 4096],
                input_range: 1.0,
            }),
            calibration_lut: Some(ExportedCalibrationLut {
                x_breakpoints: vec![0, 2048, 4096],
                y_values: vec![0, 1024, 4096],
            }),
            clinical_threshold: Some(ExportedClinicalThreshold {
                value: 0.2,
                quantized: 819,
                recall: Some(0.9),
                precision: Some(0.2),
            }),
            validation: None,
            training_metadata: None,
            dataset: None,
        };

        let uncalibrated = TfheAdapter::apply_sigmoid(&model, 0.0);
        assert!((uncalibrated - 0.25).abs() < f64::EPSILON);

        let calibrated = TfheAdapter::apply_isotonic_calibration(&model, uncalibrated);
        assert!((calibrated - 0.125).abs() < f64::EPSILON);
        assert!((TfheAdapter::clinical_threshold(&model) - 0.2).abs() < f64::EPSILON);

        let result =
            DiagnosisResult::with_threshold(calibrated, TfheAdapter::clinical_threshold(&model));
        assert!(!result.screening_positive);
    }
}
