//! SQLite adapter: Implementation of Storage.
//!
//! Provides local persistence for keys and diagnoses.
//!
//! # Security
//!
//! FHE keys are encrypted at rest using envelope encryption:
//! - Argon2id for key derivation from password
//! - AES-256-GCM for authenticated encryption
//! - Password sourced from a file descriptor or file (e.g., Docker secret)
//!
//! # Mutex Behavior
//!
//! Database connection is protected by `Mutex`. A poisoned mutex (from panic
//! in another thread) will cause panic. This fail-fast behavior is intentional
//! for data integrity in healthcare applications.
use std::path::Path;
use std::sync::Mutex;
#[cfg(unix)]
use std::{
    fs,
    io::Read,
    os::unix::io::FromRawFd,
};

use rusqlite::{params, Connection};

use crate::domain::{
    kdf::{self, EncryptedKey},
    ClientKey, Diagnosis, DiagnosisResult, KeyPair, RiskLevel, ServerKey,
};
use crate::ports::{DiagnosisPage, Storage};

use zeroize::Zeroizing;

/// Secure sources for key encryption password.
///
/// Precedence (highest first):
/// - `PULSECURE_KEY_PASSWORD_FD` (read from an already-open FD, then close it)
/// - `PULSECURE_KEY_PASSWORD_FILE` (read from a file path)
/// - `/run/secrets/pulsecure_key_password` (Docker/Compose secret default)
///
/// In release builds, reading secrets from environment variables is refused.
const KEY_PASSWORD_FD_ENV: &str = "PULSECURE_KEY_PASSWORD_FD";
const KEY_PASSWORD_FILE_ENV: &str = "PULSECURE_KEY_PASSWORD_FILE";
const KEY_PASSWORD_DOCKER_SECRET_PATH: &str = "/run/secrets/pulsecure_key_password";

// Dev-only escape hatch for local runs and tests.
const KEY_PASSWORD_ENV_DEV: &str = "PULSECURE_KEY_PASSWORD";

/// Error type for storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Key decryption failed: wrong password or corrupted data")]
    KeyDecryption,

    #[error(
        "Missing key password: provide {KEY_PASSWORD_FD_ENV} or {KEY_PASSWORD_FILE_ENV} (or mount {KEY_PASSWORD_DOCKER_SECRET_PATH})"
    )]
    MissingPassword,
}

/// SQLite storage adapter.
pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

impl SqliteStorage {
    /// Create a new SQLite storage with the given database path.
    ///
    /// # Errors
    /// Returns error if database cannot be opened or initialized.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let conn = Connection::open(path)?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.init_schema()?;
        Ok(storage)
    }

    /// Create an in-memory SQLite database (for testing).
    ///
    /// # Errors
    /// Returns error if database cannot be created.
    pub fn in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory()?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.init_schema()?;
        Ok(storage)
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> Result<(), StorageError> {
        let conn = self.conn.lock().expect("Lock failed");

        conn.execute_batch(
            r"
            CREATE TABLE IF NOT EXISTS keys (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                client_key_encrypted BLOB NOT NULL,
                server_key_encrypted BLOB NOT NULL,
                client_fingerprint TEXT NOT NULL,
                server_fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS diagnoses (
                id TEXT PRIMARY KEY,
                patient_id TEXT,
                probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                encrypted_computation INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_diagnoses_created 
                ON diagnoses(created_at DESC);
            ",
        )?;

        Ok(())
    }

    /// Get the key encryption password from a secure source.
    ///
    /// In release builds, environment variables are not accepted for secrets.
    fn get_key_password() -> Result<Zeroizing<String>, StorageError> {
        // 1) Read from an already-open FD (recommended for systemd/K8s sidecars)
        #[cfg(unix)]
        if let Ok(fd_str) = std::env::var(KEY_PASSWORD_FD_ENV) {
            let fd: i32 = fd_str.trim().parse().map_err(|_| StorageError::MissingPassword)?;
            if fd <= 2 {
                // Refuse stdio FDs to avoid interfering with the TUI.
                return Err(StorageError::MissingPassword);
            }

            // SAFETY: We take ownership of the FD for one-time secret read and close it.
            let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
            let mut buf = String::new();
            file.read_to_string(&mut buf)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;

            let secret = buf.trim_end_matches(['\n', '\r']).to_string();
            if secret.is_empty() {
                return Err(StorageError::MissingPassword);
            }
            return Ok(Zeroizing::new(secret));
        }

        // 2) Read from an explicit file path
        #[cfg(unix)]
        if let Ok(path) = std::env::var(KEY_PASSWORD_FILE_ENV) {
            let content = fs::read_to_string(path.trim())
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            let secret = content.trim_end_matches(['\n', '\r']).to_string();
            if secret.is_empty() {
                return Err(StorageError::MissingPassword);
            }
            return Ok(Zeroizing::new(secret));
        }

        // 3) Docker secrets default path
        #[cfg(unix)]
        if Path::new(KEY_PASSWORD_DOCKER_SECRET_PATH).exists() {
            let content = fs::read_to_string(KEY_PASSWORD_DOCKER_SECRET_PATH)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            let secret = content.trim_end_matches(['\n', '\r']).to_string();
            if secret.is_empty() {
                return Err(StorageError::MissingPassword);
            }
            return Ok(Zeroizing::new(secret));
        }

        // 4) Dev-only env var (refused in release builds)
        if cfg!(debug_assertions) {
            if let Ok(v) = std::env::var(KEY_PASSWORD_ENV_DEV) {
                // Dev-only: accept env var for convenience.
                // NOTE: We intentionally do NOT remove it after reading, because
                // the application may need to load keys multiple times per process.
                let secret = v.trim_end_matches(['\n', '\r']).to_string();
                if secret.is_empty() {
                    return Err(StorageError::MissingPassword);
                }
                return Ok(Zeroizing::new(secret));
            }
        }

        Err(StorageError::MissingPassword)
    }

    /// Convert RiskLevel to string for storage.
    fn risk_level_to_string(level: RiskLevel) -> &'static str {
        match level {
            RiskLevel::Low => "low",
            RiskLevel::Moderate => "moderate",
            RiskLevel::High => "high",
        }
    }

    /// Convert string to RiskLevel.
    fn string_to_risk_level(s: &str) -> RiskLevel {
        match s.to_lowercase().as_str() {
            "low" => RiskLevel::Low,
            "moderate" => RiskLevel::Moderate,
            "high" => RiskLevel::High,
            _ => RiskLevel::Moderate,
        }
    }
}

impl Storage for SqliteStorage {
    type Error = StorageError;

    fn save_keys(&self, keys: &KeyPair) -> Result<(), Self::Error> {
        let password = Self::get_key_password()?;
        let conn = self.conn.lock().expect("Lock failed");
        let now = chrono::Utc::now().to_rfc3339();

        // Encrypt keys with envelope encryption
        let client_encrypted = kdf::encrypt_key(keys.client.as_bytes(), password.as_str())
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        let server_encrypted = kdf::encrypt_key(keys.server.as_bytes(), password.as_str())
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        conn.execute(
            r"
            INSERT OR REPLACE INTO keys (
                id, client_key_encrypted, server_key_encrypted, 
                client_fingerprint, server_fingerprint, created_at
            ) VALUES (1, ?1, ?2, ?3, ?4, ?5)
            ",
            params![
                client_encrypted.to_bytes(),
                server_encrypted.to_bytes(),
                keys.client.fingerprint,
                keys.server.fingerprint,
                now,
            ],
        )?;

        tracing::info!("Saved encrypted keys to storage");
        Ok(())
    }

    fn load_keys(&self) -> Result<Option<KeyPair>, Self::Error> {
        let password = Self::get_key_password()?;
        let conn = self.conn.lock().expect("Lock failed");

        let mut stmt = conn.prepare(
            "SELECT client_key_encrypted, server_key_encrypted FROM keys WHERE id = 1",
        )?;

        let result = stmt.query_row([], |row| {
            let client_encrypted_bytes: Vec<u8> = row.get(0)?;
            let server_encrypted_bytes: Vec<u8> = row.get(1)?;
            Ok((client_encrypted_bytes, server_encrypted_bytes))
        });

        match result {
            Ok((client_encrypted_bytes, server_encrypted_bytes)) => {
                // Decrypt keys with envelope encryption
                let client_encrypted = EncryptedKey::from_bytes(&client_encrypted_bytes)
                    .map_err(|_| StorageError::KeyDecryption)?;
                let server_encrypted = EncryptedKey::from_bytes(&server_encrypted_bytes)
                    .map_err(|_| StorageError::KeyDecryption)?;

                let client_bytes = kdf::decrypt_key(&client_encrypted, password.as_str())
                    .map_err(|_| StorageError::KeyDecryption)?;
                let server_bytes = kdf::decrypt_key(&server_encrypted, password.as_str())
                    .map_err(|_| StorageError::KeyDecryption)?;

                let client = ClientKey::from_bytes(client_bytes);
                let server = ServerKey::from_bytes(server_bytes);
                Ok(Some(KeyPair::new(client, server)))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn has_keys(&self) -> Result<bool, Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM keys WHERE id = 1",
            [],
            |row| row.get(0),
        )?;

        Ok(count > 0)
    }

    fn delete_keys(&self) -> Result<(), Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");
        conn.execute("DELETE FROM keys WHERE id = 1", [])?;
        tracing::info!("Deleted keys from storage");
        Ok(())
    }

    fn save_diagnosis(&self, diagnosis: &Diagnosis) -> Result<(), Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");

        conn.execute(
            r"
            INSERT INTO diagnoses (
                id, patient_id, probability, prediction, confidence,
                risk_level, encrypted_computation, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                diagnosis.id,
                diagnosis.patient_id,
                diagnosis.result.probability,
                diagnosis.result.prediction as i64,
                diagnosis.result.confidence,
                Self::risk_level_to_string(diagnosis.risk_level),
                diagnosis.encrypted_computation as i64,
                diagnosis.created_at.to_rfc3339(),
            ],
        )?;

        tracing::debug!("Saved diagnosis {} to storage", diagnosis.id);
        Ok(())
    }

    fn load_diagnoses(&self) -> Result<Vec<Diagnosis>, Self::Error> {
        // NOTE: This loads up to 1000 diagnoses into memory.
        // For larger datasets, use load_diagnoses_paginated() instead.
        self.load_recent_diagnoses(1000)
    }

    fn load_recent_diagnoses(&self, limit: usize) -> Result<Vec<Diagnosis>, Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");

        let mut stmt = conn.prepare(
            r"
            SELECT id, patient_id, probability, prediction, confidence,
                   risk_level, encrypted_computation, created_at
            FROM diagnoses
            ORDER BY created_at DESC
            LIMIT ?1
            ",
        )?;

        let diagnoses = stmt
            .query_map(params![limit as i64], |row| {
                let id: String = row.get(0)?;
                let patient_id: Option<String> = row.get(1)?;
                let probability: f64 = row.get(2)?;
                let prediction: i64 = row.get(3)?;
                let confidence: f64 = row.get(4)?;
                let risk_level_str: String = row.get(5)?;
                let encrypted: i64 = row.get(6)?;
                let created_at_str: String = row.get(7)?;

                let result = DiagnosisResult {
                    probability,
                    prediction: prediction as u8,
                    confidence,
                };

                let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now());

                Ok(Diagnosis {
                    id,
                    patient_id,
                    result,
                    risk_level: Self::string_to_risk_level(&risk_level_str),
                    encrypted_computation: encrypted != 0,
                    created_at,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(diagnoses)
    }

    fn load_diagnoses_paginated(&self, offset: usize, limit: usize) -> Result<DiagnosisPage, Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");

        // Get total count
        let total_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM diagnoses",
            [],
            |row| row.get(0),
        )?;

        // Fetch page with OFFSET/LIMIT
        let mut stmt = conn.prepare(
            r"
            SELECT id, patient_id, probability, prediction, confidence,
                   risk_level, encrypted_computation, created_at
            FROM diagnoses
            ORDER BY created_at DESC
            LIMIT ?1 OFFSET ?2
            ",
        )?;

        let diagnoses = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                let id: String = row.get(0)?;
                let patient_id: Option<String> = row.get(1)?;
                let probability: f64 = row.get(2)?;
                let prediction: i64 = row.get(3)?;
                let confidence: f64 = row.get(4)?;
                let risk_level_str: String = row.get(5)?;
                let encrypted: i64 = row.get(6)?;
                let created_at_str: String = row.get(7)?;

                let result = DiagnosisResult {
                    probability,
                    prediction: prediction as u8,
                    confidence,
                };

                let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now());

                Ok(Diagnosis {
                    id,
                    patient_id,
                    result,
                    risk_level: Self::string_to_risk_level(&risk_level_str),
                    encrypted_computation: encrypted != 0,
                    created_at,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(DiagnosisPage::new(diagnoses, total_count as usize, offset, limit))
    }

    fn count_diagnoses(&self) -> Result<usize, Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM diagnoses",
            [],
            |row| row.get(0),
        )?;

        Ok(count as usize)
    }

    fn delete_diagnosis(&self, id: &str) -> Result<(), Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");
        conn.execute("DELETE FROM diagnoses WHERE id = ?1", params![id])?;
        Ok(())
    }

    fn clear_all(&self) -> Result<(), Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");
        conn.execute_batch(
            "DELETE FROM keys; DELETE FROM diagnoses;",
        )?;
        tracing::warn!("Cleared all data from storage");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test setup: ensure Pulsecure_KEY_PASSWORD is set for key encryption tests.
    fn setup_test_password() {
        // Prefer the new, consistent env var name.
        std::env::set_var(
            "PULSECURE_KEY_PASSWORD",
            "test_password_for_ci_only_32chars!",
        );
    }

    #[test]
    fn test_keys_roundtrip() {
        // SECURITY: Set test password (do NOT use in production)
        setup_test_password();
        
        let storage = SqliteStorage::in_memory().expect("Should create db");

        // No keys initially
        assert!(!storage.has_keys().expect("Should check"));
        assert!(storage.load_keys().expect("Should load").is_none());

        // Save keys
        let client = ClientKey::from_bytes(vec![1, 2, 3, 4]);
        let server = ServerKey::from_bytes(vec![5, 6, 7, 8]);
        let keys = KeyPair::new(client, server);

        storage.save_keys(&keys).expect("Should save");
        assert!(storage.has_keys().expect("Should check"));

        // Load keys
        let loaded = storage.load_keys().expect("Should load").expect("Should exist");
        assert_eq!(loaded.client.fingerprint, keys.client.fingerprint);

        // Delete keys
        storage.delete_keys().expect("Should delete");
        assert!(!storage.has_keys().expect("Should check"));
    }

    #[test]
    fn test_diagnosis_crud() {
        let storage = SqliteStorage::in_memory().expect("Should create db");

        // No diagnoses initially
        assert_eq!(storage.count_diagnoses().expect("Should count"), 0);

        // Save diagnosis
        let result = DiagnosisResult::new(0.75);
        let diagnosis = Diagnosis::new(result, true);
        let id = diagnosis.id.clone();

        storage.save_diagnosis(&diagnosis).expect("Should save");
        assert_eq!(storage.count_diagnoses().expect("Should count"), 1);

        // Load diagnoses
        let loaded = storage.load_diagnoses().expect("Should load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, id);

        // Delete diagnosis
        storage.delete_diagnosis(&id).expect("Should delete");
        assert_eq!(storage.count_diagnoses().expect("Should count"), 0);
    }
}
