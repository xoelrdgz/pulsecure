//! FHE Engine port: Trait for Fully Homomorphic Encryption operations.
//!
//! This trait abstracts the FHE library (tfhe-rs) from the application logic.

use crate::domain::{
    ClientKey, CryptoError, Diagnosis, EncryptedDiagnosis, EncryptedPatientData, KeyPair,
    PatientData, ServerKey,
};

/// Trait for FHE operations.
///
/// Implementations provide:
/// - Key generation with CSPRNG
/// - Encryption of patient data
/// - Homomorphic computation (blind inference)
/// - Decryption of results
pub trait FheEngine: Send + Sync {
    /// Generate a new key pair for FHE operations.
    ///
    /// Uses a cryptographically secure random number generator.
    ///
    /// # Errors
    /// Returns `CryptoError::KeyGeneration` if key generation fails.
    fn generate_keys(&self) -> Result<KeyPair, CryptoError>;

    /// Encrypt patient data using the client key.
    ///
    /// # Arguments
    /// * `data` - Patient data to encrypt
    /// * `key` - Client key for encryption
    ///
    /// # Errors
    /// Returns `CryptoError::Encryption` if encryption fails.
    fn encrypt(
        &self,
        data: &PatientData,
        key: &ClientKey,
    ) -> Result<EncryptedPatientData, CryptoError>;

    /// Perform homomorphic computation on encrypted data.
    ///
    /// This is the "blind compute" operation that runs the ML model
    /// on encrypted data without decrypting it.
    ///
    /// # Arguments
    /// * `encrypted` - Encrypted patient data
    /// * `server_key` - Server key for computation
    ///
    /// # Errors
    /// Returns `CryptoError::Computation` if computation fails.
    /// Returns `CryptoError::NoiseBudgetExhausted` if noise budget is depleted.
    fn compute(
        &self,
        encrypted: &EncryptedPatientData,
        server_key: &ServerKey,
    ) -> Result<EncryptedDiagnosis, CryptoError>;

    /// Decrypt the diagnosis result using the client key.
    ///
    /// # Arguments
    /// * `result` - Encrypted diagnosis
    /// * `key` - Client key for decryption
    ///
    /// # Errors
    /// Returns `CryptoError::Decryption` if decryption fails.
    fn decrypt(
        &self,
        result: &EncryptedDiagnosis,
        key: &ClientKey,
    ) -> Result<Diagnosis, CryptoError>;

    /// Serialize a key pair for storage.
    ///
    /// # Errors
    /// Returns `CryptoError::Serialization` if serialization fails.
    fn serialize_keys(&self, keys: &KeyPair) -> Result<(Vec<u8>, Vec<u8>), CryptoError>;

    /// Deserialize a key pair from storage.
    ///
    /// # Errors
    /// Returns `CryptoError::InvalidKeyFormat` if deserialization fails.
    fn deserialize_keys(
        &self,
        client_bytes: &[u8],
        server_bytes: &[u8],
    ) -> Result<KeyPair, CryptoError>;
}
