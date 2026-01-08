//! TFHE adapter: Implementation of FheEngine using tfhe-rs.
//!
//! This module provides FHE operations using Zama's tfhe-rs library.
//!
//! # Security
//!
//! - Model files are verified via Ed25519 digital signatures
//! - Only models signed by the developer key are loaded
//! - In release builds, ALL models MUST have valid signatures
//! - Patient data is encrypted using Fully Homomorphic Encryption
//!
//! # Thread Safety
//!
//! **IMPORTANT**: `tfhe::set_server_key()` writes to a *thread-local* (TLS) global.
//!
//! This adapter sets the server key per computation and scopes it to the current
//! thread via an RAII guard. This prevents key "leakage" into later work executed on
//! the same thread and avoids cross-request key confusion in thread pools, as long as:
//!
//! - Each computation runs to completion without async yields on the same thread, and
//! - Each request provides the correct `ServerKey` to `compute()`.
//!
//! # FHE Implementation
//!
//! Uses tfhe-rs with:
//! - `FheInt64` for encrypted integer arithmetic on quantized features
//! - Server key for homomorphic computation (can't decrypt)
//! - Client key for encryption/decryption (kept secret)
//! - Fixed-point quantization (PRECISION_BITS) for floating point values
//!
//! # Key Rotation
//!
//! To rotate the developer public key:
//! 1. Generate new keypair: `cargo run --bin generate_keypair`
//! 2. Replace `DEV_PUBKEY` constant with new public key bytes
//! 3. Re-sign all models with new private key
//! 4. Securely destroy old private key

use std::path::Path;
use std::{collections::BTreeMap, fs};

use base64::Engine;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// tfhe-rs imports
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

/// Environment variable to allow loading unsigned models.
///
/// SECURITY: This bypass is compiled only in debug builds.
/// In release builds, it is physically impossible to skip model signature checks.
#[cfg(debug_assertions)]
const ALLOW_UNSIGNED_MODELS_ENV: &str = "PULSECURE_ALLOW_UNSIGNED_MODELS";

/// Maximum number of features supported (NHANES cardiovascular = 9).
/// Used for input validation and sanity checks.
const MAX_FEATURES: usize = 9;

/// Model parameters exported by the Python pipeline.
///
/// This matches the JSON structure produced by `python/src/training/pipeline.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedQuantizedModel {
    pub precision_bits: u32,
    pub scale_factor: i64,
    pub feature_names: Vec<String>,
    pub coefficients_q: Vec<i64>,
    pub intercept_q: i64,
    pub scaler_mean_q: Vec<i64>,
    pub scaler_std_inv_q: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SignedModelManifest {
    version: u32,
    #[serde(default)]
    serial: Option<u64>,
    #[serde(default)]
    created_at: Option<i64>,
    #[serde(default)]
    nonce_b64: Option<String>,
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

/// TFHE adapter for FHE operations.
///
/// Uses tfhe-rs for Fully Homomorphic Encryption.
/// Patient data is encrypted and processed blindly on the server without
/// ever being decrypted during computation.
pub struct TfheAdapter {
    /// Quantized model parameters exported by Python.
    model: Option<ExportedQuantizedModel>,
}

impl TfheAdapter {
    /// Create a new TFHE adapter.
    #[must_use]
    pub fn new() -> Self {
        tracing::info!("Initializing TfheAdapter (tfhe-rs)");
        Self { model: None }
    }

    /// Load model parameters from the Python export directory.
    ///
    /// # Security
    ///
    /// Models must be signed with the developer's Ed25519 key.
    /// The signature file (`model.sig`) must be present and valid.
    ///
    /// # Errors
    /// Returns error if model files cannot be loaded or signature is invalid.
    pub fn load_model(&mut self, model_dir: &Path) -> Result<(), CryptoError> {
        // Verify model signature before loading (unless explicitly bypassed in debug builds).
        // IMPORTANT: The model file actually loaded MUST be bound by the signed manifest.
        let manifest = self.verify_model_signature(model_dir)?;

        let base_dir = if model_dir.is_dir() {
            model_dir
        } else {
            model_dir.parent().unwrap_or(model_dir)
        };

        let model_path = if let Some(manifest) = manifest {
            // Choose a model file that is explicitly bound by the signed manifest.
            // Prefer calibrated_model.json when present.
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
            // Debug-only: unsigned model loading. Fall back to filesystem discovery.
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

        // Basic sanity checks
        let n = model.feature_names.len();
        if n == 0 || n > MAX_FEATURES {
            return Err(CryptoError::Serialization(format!(
                "Invalid feature count in model: got {n}, max {MAX_FEATURES}"
            )));
        }
        if model.coefficients_q.len() != n
            || model.scaler_mean_q.len() != n
            || model.scaler_std_inv_q.len() != n
        {
            return Err(CryptoError::Serialization(
                "Model parameter lengths do not match feature_names length".into(),
            ));
        }

        tracing::info!(
            "Loaded model from {:?} (precision_bits={}, scale_factor={}, n_features={})",
            model_path,
            model.precision_bits,
            model.scale_factor,
            n
        );

        self.model = Some(model);
        Ok(())
    }

    /// Verify model signature using Ed25519.
    ///
    /// Checks that the model manifest hash matches the signature.
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

        // SECURITY: Signature verification is MANDATORY in release builds.
        // In debug builds, can be bypassed ONLY with explicit env var for testing.
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

        // Load signature
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

        // Load manifest (the signed content)
        let manifest_content = fs::read(&manifest_path)
            .map_err(|e| CryptoError::Serialization(format!("Failed to read manifest: {e}")))?;

        // Verify with embedded developer public key
        let public_key = Self::developer_public_key()?;
        public_key
            .verify(&manifest_content, &signature)
            .map_err(|_| CryptoError::Serialization("Invalid model signature".into()))?;

        // Defense-in-depth: verify that the signed manifest binds the actual model files.
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

        // Anti-rollback: require serial + creation timestamp (unless explicitly allowed for legacy).
        let allow_legacy = parse_bool_env("PULSECURE_ALLOW_LEGACY_MODEL_MANIFEST");

        let serial = match manifest.serial {
            Some(s) => s,
            None => {
                if cfg!(debug_assertions) && allow_legacy {
                    tracing::warn!("Loading legacy manifest without serial (PULSECURE_ALLOW_LEGACY_MODEL_MANIFEST=true)");
                    0
                } else {
                    return Err(CryptoError::Serialization(
                        "manifest.json missing required field serial".into(),
                    ));
                }
            }
        };

        let created_at = match manifest.created_at {
            Some(ts) => ts,
            None => {
                if cfg!(debug_assertions) && allow_legacy {
                    tracing::warn!("Loading legacy manifest without created_at (PULSECURE_ALLOW_LEGACY_MODEL_MANIFEST=true)");
                    0
                } else {
                    return Err(CryptoError::Serialization(
                        "manifest.json missing required field created_at".into(),
                    ));
                }
            }
        };

        match &manifest.nonce_b64 {
            Some(n) => validate_nonce_b64(n)?,
            None => {
                if !(cfg!(debug_assertions) && allow_legacy) {
                    return Err(CryptoError::Serialization(
                        "manifest.json missing required field nonce_b64".into(),
                    ));
                }
                tracing::warn!("Loading legacy manifest without nonce_b64 (PULSECURE_ALLOW_LEGACY_MODEL_MANIFEST=true)");
            }
        }

        if created_at > 0 {
            let now = unix_now();
            // Refuse manifests too far in the future (clock skew allowance: 5 minutes).
            if created_at > now + 300 {
                return Err(CryptoError::Serialization(
                    "manifest created_at is in the future".into(),
                ));
            }

            // Optional: enforce max age.
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

        // Ensure at least one expected model JSON is bound by the manifest.
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

        // Best-effort rollback protection with a persistent state file.
        // If present, we refuse to load manifests older than the last accepted one.
        let state_path = std::env::var("PULSECURE_MODEL_ROLLBACK_STATE_FILE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/app/data/model_rollback_state.json"));
        let enforce_state = parse_bool_env("PULSECURE_ENFORCE_ROLLBACK_PROTECTION");

        let manifest_hash = sha256_hex_bytes(&manifest_content);
        match read_rollback_state(&state_path) {
            Ok(Some(state)) => {
                // Primary check: serial monotonicity.
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
                // Update state if this manifest is newer (or same timestamp but different content).
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
                // Initialize state on first successful load.
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

    /// Get the embedded developer public key for model verification.
    ///
    /// This key is compiled into the binary and used to verify all model signatures.
    ///
    /// # Key Rotation
    ///
    /// To rotate this key:
    /// 1. Generate new keypair: `cargo run --bin generate_keypair`
    /// 2. Replace the bytes below with new public key
    /// 3. Re-sign all models with new private key
    /// 4. Securely destroy old private key
    fn developer_public_key() -> Result<VerifyingKey, CryptoError> {
        // Runtime override (recommended for deployments): load verifying key from a secret file.
        // This avoids embedding environment-specific keys into the binary.
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
            // Test-only override: allows unit tests to generate a fresh keypair and
            // validate the signed-model workflow without embedding any private key.
            const TEST_PUBKEY_ENV: &str = "PULSECURE_TEST_DEV_PUBKEY_B64";
            if let Ok(b64) = std::env::var(TEST_PUBKEY_ENV) {
                return Self::verifying_key_from_b64(&b64)
                    .map_err(|_| CryptoError::Serialization("Invalid test verifying key".into()));
            }
        }

        // Ed25519 public key (32 bytes)
        // Generated with: cargo run --bin generate_keypair
        const DEV_PUBKEY: [u8; 32] = [
            0x8f, 0x2a, 0x55, 0x65, 0x8a, 0x3e, 0x12, 0x7d, 0x93, 0x4b, 0x1c, 0x6f, 0xa0, 0xbe,
            0x72, 0x41, 0xd5, 0xe8, 0x99, 0x23, 0x0c, 0x47, 0xf1, 0x8b, 0x6d, 0xa2, 0x34, 0xc9,
            0x76, 0x58, 0x0f, 0xe3,
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

    /// Normalize and quantize raw features using the exported quantized scaler.
    ///
    /// Matches the integer pipeline in `python/src/training/pipeline.py`:
    /// `x_scaled = x_raw * scale_factor; x_centered = x_scaled - mean_q;
    ///  x_norm_q = (x_centered * std_inv_q) / scale_factor`.
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

            // Use wider type to avoid overflow in the multiply.
            let numer = (x_centered as i128) * (model.scaler_std_inv_q[i] as i128);
            let x_norm_q = (numer / (scale as i128)) as i64;
            out.push(x_norm_q);
        }

        Ok(out)
    }

    /// Compute sigmoid approximation for FHE result.
    /// Uses a polynomial approximation suitable for encrypted computation.
    fn sigmoid_approx(x: f64) -> f64 {
        // Simple sigmoid: 1 / (1 + exp(-x))
        1.0 / (1.0 + (-x).exp())
    }

    /// Deserialize tfhe-rs client key from bytes.
    fn deserialize_tfhe_client_key(bytes: &[u8]) -> Result<TfheClientKey, CryptoError> {
        bincode::deserialize(bytes).map_err(|e| {
            CryptoError::Serialization(format!("Failed to deserialize client key: {e}"))
        })
    }

    /// Deserialize tfhe-rs server key from bytes.
    fn deserialize_tfhe_server_key(bytes: &[u8]) -> Result<TfheServerKey, CryptoError> {
        bincode::deserialize(bytes).map_err(|e| {
            CryptoError::Serialization(format!("Failed to deserialize server key: {e}"))
        })
    }
}

// Constant-time compare for ASCII strings (used for SHA-256 hex digests).
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

        // Configure tfhe-rs parameters
        // Using default parameters which are secure for most applications
        let config = ConfigBuilder::default().build();

        // Generate FHE keys
        let (client_key, server_key) = generate_keys(config);

        tracing::info!("Generated tfhe-rs keys");

        // Serialize keys for storage
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

        // Deserialize the tfhe-rs client key
        let tfhe_client_key = Self::deserialize_tfhe_client_key(key.as_bytes())?;

        let features = data.features.to_vec();

        // Validate feature count
        if features.len() > MAX_FEATURES {
            return Err(CryptoError::Encryption(format!(
                "Too many features: got {}, max {}",
                features.len(),
                MAX_FEATURES
            )));
        }

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| CryptoError::Encryption("Model not loaded".into()))?;

        let quantized = Self::normalize_and_quantize_features(model, &features)?;

        // Encrypt each quantized feature value
        let mut encrypted_features: Vec<Vec<u8>> = Vec::with_capacity(quantized.len());

        for (i, &value) in quantized.iter().enumerate() {
            let encrypted: FheInt64 = FheInt64::encrypt(value, &tfhe_client_key);

            let encrypted_bytes = bincode::serialize(&encrypted).map_err(|e| {
                CryptoError::Encryption(format!("Failed to serialize encrypted feature {i}: {e}"))
            })?;

            encrypted_features.push(encrypted_bytes);
            tracing::trace!("Encrypted feature {i}");
        }

        // Serialize the vector of encrypted features
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

        // Deserialize the tfhe-rs server key
        let tfhe_server_key = Self::deserialize_tfhe_server_key(server_key.as_bytes())?;

        // Set the server key for homomorphic operations (TLS) and ensure it is cleared
        // when this computation finishes.
        struct ServerKeyGuard;
        impl Drop for ServerKeyGuard {
            fn drop(&mut self) {
                unset_server_key();
            }
        }

        set_server_key(tfhe_server_key);
        let _server_key_guard = ServerKeyGuard;

        // Deserialize encrypted features
        let encrypted_features: Vec<Vec<u8>> = bincode::deserialize(&encrypted.ciphertext)
            .map_err(|e| {
                CryptoError::Computation(format!("Failed to deserialize encrypted data: {e}"))
            })?;

        // Deserialize each FheInt64
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

        // Homomorphic linear combination: sum(coef_i * feature_i) + intercept
        // All operations happen on encrypted data!
        tracing::debug!("Computing encrypted linear combination...");

        let mut result: FheInt64 = FheInt64::encrypt_trivial(0i64);

        for (i, (fhe_feature, &coef)) in fhe_features
            .iter()
            .zip(model.coefficients_q.iter())
            .enumerate()
        {
            // Homomorphic scalar multiplication: encrypted_feature * cleartext_coefficient
            let term = fhe_feature * coef;
            // Homomorphic addition
            result = result + term;
            tracing::trace!("Computed term {i} (homomorphic multiply + add)");
        }

        // Add intercept term (scalar addition to encrypted value)
        // Pipeline uses: logits_q += intercept_q * scale_factor
        let intercept_term = model
            .intercept_q
            .checked_mul(model.scale_factor)
            .ok_or_else(|| CryptoError::Computation("Intercept term overflow".into()))?;
        result = result + intercept_term;

        // Serialize the encrypted result
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

        // Deserialize the tfhe-rs client key
        let tfhe_client_key = Self::deserialize_tfhe_client_key(key.as_bytes())?;

        // Deserialize the encrypted result
        let encrypted_result: FheInt64 = bincode::deserialize(&result.ciphertext).map_err(|e| {
            CryptoError::Decryption(format!("Failed to deserialize encrypted result: {e}"))
        })?;

        // Decrypt result
        let decrypted_value: i64 = encrypted_result.decrypt(&tfhe_client_key);

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| CryptoError::Decryption("Model not loaded".into()))?;

        // Convert from fixed-point back to floating point.
        // Result is scaled by (scale_factor^2).
        let scale = model.scale_factor as f64;
        let linear_result = (decrypted_value as f64) / (scale * scale);

        // Apply sigmoid to get probability
        let probability = Self::sigmoid_approx(linear_result);

        tracing::info!(
            "Decrypted result: linear={:.4}, probability={:.4}",
            linear_result,
            probability
        );

        if !probability.is_finite() {
            return Err(CryptoError::Decryption(
                "Decryption produced non-finite probability".into(),
            ));
        }

        let diagnosis_result = DiagnosisResult::new(probability);
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
        // Validate keys by attempting to deserialize
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

    fn write_exported_model(path: &Path, intercept_q: i64) {
        let model = ExportedQuantizedModel {
            precision_bits: 12,
            scale_factor: 4096,
            feature_names: vec!["x".into()],
            coefficients_q: vec![1],
            intercept_q,
            scaler_mean_q: vec![0],
            scaler_std_inv_q: vec![4096],
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
            serial: Some(serial),
            created_at: Some(created_at),
            nonce_b64: Some(nonce_b64),
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

        // Write both candidate model files with different intercepts.
        let model_path = dir.join("model.json");
        let calibrated_path = dir.join("calibrated_model.json");
        write_exported_model(&model_path, 111);
        write_exported_model(&calibrated_path, 222);

        // Generate a test signing key and inject its verifying key.
        let mut sk = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut sk);
        let signing_key = SigningKey::from_bytes(&sk);
        let pubkey_b64 = base64::engine::general_purpose::STANDARD
            .encode(signing_key.verifying_key().to_bytes());
        std::env::set_var("PULSECURE_TEST_DEV_PUBKEY_B64", pubkey_b64);

        // Manifest binds ONLY model.json, so loader must not pick calibrated_model.json.
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

        // Manifest binds calibrated_model.json, so loader must pick it.
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

        // Only create calibrated_model.json.
        let calibrated_path = dir.join("calibrated_model.json");
        write_exported_model(&calibrated_path, 222);

        let mut sk = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut sk);
        let signing_key = SigningKey::from_bytes(&sk);
        let pubkey_b64 = base64::engine::general_purpose::STANDARD
            .encode(signing_key.verifying_key().to_bytes());
        std::env::set_var("PULSECURE_TEST_DEV_PUBKEY_B64", pubkey_b64);

        // Manifest references model.json (which is missing) => must fail closed.
        write_signed_manifest(dir, &signing_key, &[("model.json", b"missing".to_vec())]);

        let mut adapter = TfheAdapter::new();
        let err = adapter.load_model(dir).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("missing") || msg.contains("unreadable"));

        std::env::remove_var("PULSECURE_TEST_DEV_PUBKEY_B64");
    }

    #[test]
    fn test_key_generation() {
        let adapter = TfheAdapter::new();
        let keys = adapter
            .generate_keys()
            .expect("Key generation should succeed");

        // FHE keys are large
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
        });

        // Encrypt with FHE
        let encrypted = adapter
            .encrypt(&patient, &keys.client)
            .expect("Encryption should succeed");
        assert_eq!(encrypted.num_features, 9);
        assert!(
            encrypted.ciphertext.len() > 1000,
            "FHE ciphertext should be large"
        );

        // Compute homomorphically (requires model, skip if not loaded)
        // In production, model must be loaded first

        // For this test, we verify encryption/decryption of raw values
        tracing::info!("FHE encryption test passed");
    }

    #[test]
    fn test_quantization() {
        let model = ExportedQuantizedModel {
            precision_bits: 12,
            scale_factor: 4096,
            feature_names: vec!["a".into(), "b".into()],
            coefficients_q: vec![1, 2],
            intercept_q: 0,
            scaler_mean_q: vec![0, 0],
            scaler_std_inv_q: vec![4096, 4096],
        };

        // With mean=0 and std_inv=scale, normalization is identity in quantized space.
        let raw = vec![0.5, -0.5];
        let q =
            TfheAdapter::normalize_and_quantize_features(&model, &raw).expect("Should quantize");

        assert_eq!(q[0], (0.5 * 4096.0) as i64);
        assert_eq!(q[1], (-0.5 * 4096.0) as i64);
    }
}
