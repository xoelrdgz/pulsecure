use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use argon2::{password_hash::SaltString, Algorithm, Argon2, Params, PasswordHasher, Version};
use rand::RngCore;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KdfError {
    #[error("Key derivation failed: {0}")]
    Derivation(String),

    #[error("Encryption failed: {0}")]
    Encryption(String),

    #[error("Decryption failed: authentication tag mismatch")]
    Decryption,

    #[error("Invalid encrypted key format")]
    InvalidFormat,
}

#[derive(Debug, Clone)]
pub struct EncryptedKey {
    pub ciphertext: Vec<u8>,

    pub salt: String,

    pub nonce: [u8; 12],
}

impl EncryptedKey {
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let salt_bytes = self.salt.as_bytes();
        let salt_len = salt_bytes.len() as u32;

        let mut result = Vec::with_capacity(4 + salt_bytes.len() + 12 + self.ciphertext.len());
        result.extend_from_slice(&salt_len.to_le_bytes());
        result.extend_from_slice(salt_bytes);
        result.extend_from_slice(&self.nonce);
        result.extend_from_slice(&self.ciphertext);
        result
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, KdfError> {
        if bytes.len() < 4 {
            return Err(KdfError::InvalidFormat);
        }

        let salt_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;

        if bytes.len() < 4 + salt_len + 12 {
            return Err(KdfError::InvalidFormat);
        }

        let salt = std::str::from_utf8(&bytes[4..4 + salt_len])
            .map_err(|_| KdfError::InvalidFormat)?
            .to_string();

        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&bytes[4 + salt_len..4 + salt_len + 12]);

        let ciphertext = bytes[4 + salt_len + 12..].to_vec();

        Ok(Self {
            ciphertext,
            salt,
            nonce,
        })
    }
}

fn derive_key(password: &str, salt: &SaltString) -> Result<[u8; 32], KdfError> {
    let params = Params::new(47104, 1, 1, Some(32))
        .map_err(|e| KdfError::Derivation(format!("Invalid Argon2 params: {e}")))?;
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let hash = argon2
        .hash_password(password.as_bytes(), salt)
        .map_err(|e| KdfError::Derivation(e.to_string()))?;

    let hash_bytes = hash
        .hash
        .ok_or_else(|| KdfError::Derivation("Hash output missing".to_string()))?;

    let bytes = hash_bytes.as_bytes();
    if bytes.len() < 32 {
        return Err(KdfError::Derivation("Hash too short".to_string()));
    }

    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes[..32]);
    Ok(key)
}

#[must_use]
pub fn generate_salt() -> SaltString {
    SaltString::generate(&mut OsRng)
}

pub fn encrypt_key(plaintext: &[u8], password: &str) -> Result<EncryptedKey, KdfError> {
    let salt = generate_salt();
    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let key = derive_key(password, &salt)?;
    let cipher =
        Aes256Gcm::new_from_slice(&key).map_err(|e| KdfError::Encryption(e.to_string()))?;

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| KdfError::Encryption(e.to_string()))?;

    Ok(EncryptedKey {
        ciphertext,
        salt: salt.to_string(),
        nonce: nonce_bytes,
    })
}

pub fn decrypt_key(encrypted: &EncryptedKey, password: &str) -> Result<Vec<u8>, KdfError> {
    let salt = SaltString::from_b64(&encrypted.salt).map_err(|_| KdfError::InvalidFormat)?;

    let key = derive_key(password, &salt)?;
    let cipher =
        Aes256Gcm::new_from_slice(&key).map_err(|e| KdfError::Derivation(e.to_string()))?;

    let nonce = Nonce::from_slice(&encrypted.nonce);

    cipher
        .decrypt(nonce, encrypted.ciphertext.as_ref())
        .map_err(|_| KdfError::Decryption)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let plaintext = b"super_secret_fhe_key_material_here";
        let password = "correct-horse-battery-staple";

        let encrypted = encrypt_key(plaintext, password).expect("Encryption should succeed");
        let decrypted = decrypt_key(&encrypted, password).expect("Decryption should succeed");

        assert_eq!(plaintext.to_vec(), decrypted);
    }

    #[test]
    fn test_wrong_password_fails() {
        let plaintext = b"super_secret_fhe_key_material_here";
        let correct_password = "correct-horse-battery-staple";
        let wrong_password = "wrong-password";

        let encrypted =
            encrypt_key(plaintext, correct_password).expect("Encryption should succeed");
        let result = decrypt_key(&encrypted, wrong_password);

        assert!(matches!(result, Err(KdfError::Decryption)));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let plaintext = b"test_key";
        let password = "test";

        let encrypted = encrypt_key(plaintext, password).expect("Encryption should succeed");
        let bytes = encrypted.to_bytes();
        let restored = EncryptedKey::from_bytes(&bytes).expect("Deserialization should succeed");

        let decrypted = decrypt_key(&restored, password).expect("Decryption should succeed");
        assert_eq!(plaintext.to_vec(), decrypted);
    }

    #[test]
    fn test_different_salts_produce_different_ciphertexts() {
        let plaintext = b"same_key";
        let password = "same_password";

        let encrypted1 = encrypt_key(plaintext, password).expect("Encryption should succeed");
        let encrypted2 = encrypt_key(plaintext, password).expect("Encryption should succeed");

        assert_ne!(encrypted1.ciphertext, encrypted2.ciphertext);
    }
}
