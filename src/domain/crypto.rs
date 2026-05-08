use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

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

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ClientKey {
    #[zeroize(skip)]
    pub(crate) inner: Vec<u8>,

    #[zeroize(skip)]
    pub fingerprint: String,
}

impl ClientKey {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let fingerprint = compute_fingerprint(&bytes);
        Self {
            inner: bytes,
            fingerprint,
        }
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }
}

impl std::fmt::Debug for ClientKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientKey")
            .field("fingerprint", &self.fingerprint)
            .field("size_bytes", &self.inner.len())
            .finish()
    }
}

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ServerKey {
    pub(crate) inner: Vec<u8>,

    #[zeroize(skip)]
    pub fingerprint: String,
}

impl ServerKey {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let fingerprint = compute_fingerprint(&bytes);
        Self {
            inner: bytes,
            fingerprint,
        }
    }

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

#[derive(Debug, Clone)]
pub struct KeyPair {
    pub client: ClientKey,
    pub server: ServerKey,
}

impl KeyPair {
    pub fn new(client: ClientKey, server: ServerKey) -> Self {
        Self { client, server }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedPatientData {
    pub ciphertext: Vec<u8>,

    pub num_features: usize,

    pub key_fingerprint: String,
}

impl EncryptedPatientData {
    pub fn new(ciphertext: Vec<u8>, num_features: usize, key_fingerprint: String) -> Self {
        Self {
            ciphertext,
            num_features,
            key_fingerprint,
        }
    }

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

#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedDiagnosis {
    pub ciphertext: Vec<u8>,

    pub key_fingerprint: String,
}

impl EncryptedDiagnosis {
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

fn compute_fingerprint(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();

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

        assert!(!debug_output.contains("1, 2, 3"));

        assert!(debug_output.contains("fingerprint"));
    }

    #[test]
    fn test_fingerprint_uses_hash() {
        let fp = compute_fingerprint(&[0xde, 0xad, 0xbe, 0xef]);

        assert_ne!(fp, "deadbeef");
        assert_eq!(fp.len(), 16);
    }

    #[test]
    fn test_same_key_same_fingerprint() {
        let fp1 = compute_fingerprint(&[1, 2, 3, 4]);
        let fp2 = compute_fingerprint(&[1, 2, 3, 4]);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_encrypted_data_size() {
        let encrypted = EncryptedPatientData::new(vec![0u8; 1024], 13, "test".to_string());
        assert_eq!(encrypted.size_bytes(), 1024);
        assert_eq!(encrypted.num_features, 13);
    }
}
