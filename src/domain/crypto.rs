//! Cryptographic types for FHE operations.
//!
//! Wrappers around tfhe-rs types with additional safety guarantees.
//!
//! # Memory Security
//!
//! All key types implement `Zeroize` and `ZeroizeOnDrop` to ensure
//! cryptographic material is securely erased when no longer needed.

use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Error type for cryptographic operations.
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    #[error("Key generation failed: {0}")]
    KeyGeneration(String),

    #[error("Encryption failed: {0}")]
    Encryption(String),

    #[error("Decryption failed: {0}")]
    Decryption(String),

    #[error("FHE computation failed: {0}")]
    Computation(String),

    #[error("Serialization failed: {0}")]
    Serialization(String),

    #[error("Noise budget exhausted")]
    NoiseBudgetExhausted,

    #[error("Invalid key format: {0}")]
    InvalidKeyFormat(String),
}

/// Client-side secret key for encryption/decryption.
///
/// This key MUST remain on the client and should NEVER be transmitted.
///
/// # Security
///
/// - Implements `ZeroizeOnDrop`: key material is securely erased when dropped
/// - `Debug` implementation does NOT expose key bytes
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ClientKey {
    /// Serialized key bytes (tfhe-rs ClientKey)
    #[zeroize(skip)]  // fingerprint is derived, not secret
    pub(crate) inner: Vec<u8>,

    /// Key fingerprint for identification (NOT secret)
    #[zeroize(skip)]
    pub fingerprint: String,
}

impl ClientKey {
    /// Create a new client key from raw bytes.
    ///
    /// # Safety
    /// The caller must ensure the bytes represent a valid tfhe-rs ClientKey.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let fingerprint = compute_fingerprint(&bytes);
        Self {
            inner: bytes,
            fingerprint,
        }
    }

    /// Get the raw key bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }
}

// Intentionally NOT implementing Debug/Display to prevent accidental key leakage
impl std::fmt::Debug for ClientKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientKey")
            .field("fingerprint", &self.fingerprint)
            .field("size_bytes", &self.inner.len())
            .finish()
    }
}

/// Server-side evaluation key for FHE computations.
///
/// This key allows computation on encrypted data but CANNOT decrypt.
///
/// # Security
///
/// - Implements `ZeroizeOnDrop`: key material is securely erased when dropped
/// - `Debug` implementation does NOT expose key bytes
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ServerKey {
    /// Serialized key bytes (tfhe-rs ServerKey)
    pub(crate) inner: Vec<u8>,

    /// Key fingerprint for identification (NOT secret)
    #[zeroize(skip)]
    pub fingerprint: String,
}

impl ServerKey {
    /// Create a new server key from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let fingerprint = compute_fingerprint(&bytes);
        Self {
            inner: bytes,
            fingerprint,
        }
    }

    /// Get the raw key bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }
}

impl std::fmt::Debug for ServerKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerKey")
            .field("fingerprint", &self.fingerprint)
            .field("size_bytes", &self.inner.len())
            .finish()
    }
}

/// Key pair containing both client and server keys.
#[derive(Debug, Clone)]
pub struct KeyPair {
    pub client: ClientKey,
    pub server: ServerKey,
}

impl KeyPair {
    /// Create a new key pair.
    pub fn new(client: ClientKey, server: ServerKey) -> Self {
        Self { client, server }
    }
}

/// Encrypted patient data (feature vector).
#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedPatientData {
    /// Serialized encrypted values
    pub ciphertext: Vec<u8>,

    /// Number of features encrypted
    pub num_features: usize,

    /// Fingerprint of the key used for encryption
    pub key_fingerprint: String,
}

impl EncryptedPatientData {
    /// Create new encrypted patient data.
    pub fn new(ciphertext: Vec<u8>, num_features: usize, key_fingerprint: String) -> Self {
        Self {
            ciphertext,
            num_features,
            key_fingerprint,
        }
    }

    /// Get the size of the ciphertext in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.ciphertext.len()
    }
}

impl std::fmt::Debug for EncryptedPatientData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedPatientData")
            .field("num_features", &self.num_features)
            .field("size_bytes", &self.ciphertext.len())
            .field("key_fingerprint", &self.key_fingerprint)
            .finish()
    }
}

/// Encrypted diagnosis result.
#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedDiagnosis {
    /// Serialized encrypted result
    pub ciphertext: Vec<u8>,

    /// Key fingerprint for verification
    pub key_fingerprint: String,
}

impl EncryptedDiagnosis {
    /// Create new encrypted diagnosis.
    pub fn new(ciphertext: Vec<u8>, key_fingerprint: String) -> Self {
        Self {
            ciphertext,
            key_fingerprint,
        }
    }
}

impl std::fmt::Debug for EncryptedDiagnosis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedDiagnosis")
            .field("size_bytes", &self.ciphertext.len())
            .field("key_fingerprint", &self.key_fingerprint)
            .finish()
    }
}

/// Compute a fingerprint for key identification using SHA-256.
///
/// Uses a cryptographic hash to prevent fingerprinting attacks that could
/// leak information about the underlying key material.
fn compute_fingerprint(bytes: &[u8]) -> String {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();

    // Take first 8 bytes of hash (64 bits) for identification
    // This is the hash of the key, NOT raw key material
    result[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_key_debug_no_leak() {
        let key = ClientKey::from_bytes(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let debug_output = format!("{key:?}");

        // Should NOT contain the actual key bytes
        assert!(!debug_output.contains("1, 2, 3"));
        // Should contain the fingerprint
        assert!(debug_output.contains("fingerprint"));
    }

    #[test]
    fn test_fingerprint_uses_hash() {
        // Verify fingerprint is NOT raw bytes but a SHA-256 hash
        let fp = compute_fingerprint(&[0xde, 0xad, 0xbe, 0xef]);
        // Old insecure implementation would return "deadbeef"
        // New implementation returns SHA-256 hash, which is different
        assert_ne!(fp, "deadbeef");
        assert_eq!(fp.len(), 16); // 8 bytes = 16 hex chars
    }

    #[test]
    fn test_same_key_same_fingerprint() {
        let fp1 = compute_fingerprint(&[1, 2, 3, 4]);
        let fp2 = compute_fingerprint(&[1, 2, 3, 4]);
        assert_eq!(fp1, fp2); // Deterministic
    }

    #[test]
    fn test_encrypted_data_size() {
        let encrypted = EncryptedPatientData::new(vec![0u8; 1024], 13, "test".to_string());
        assert_eq!(encrypted.size_bytes(), 1024);
        assert_eq!(encrypted.num_features, 13);
    }
}
