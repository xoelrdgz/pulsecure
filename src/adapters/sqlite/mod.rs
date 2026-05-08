use std::path::Path;
use std::sync::Mutex;
#[cfg(unix)]
use std::{fs, io::Read, os::unix::io::FromRawFd};

use rusqlite::types::Type;
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};

use crate::domain::{
    kdf::{self, EncryptedKey},
    ClientKey, Diagnosis, DiagnosisResult, KeyPair, RiskLevel, ServerKey,
};
use crate::ports::{DiagnosisPage, Storage};

use zeroize::Zeroizing;

const KEY_PASSWORD_FD_ENV: &str = "PULSECURE_KEY_PASSWORD_FD";
const KEY_PASSWORD_FILE_ENV: &str = "PULSECURE_KEY_PASSWORD_FILE";
const KEY_PASSWORD_DOCKER_SECRET_PATH: &str = "/run/secrets/pulsecure_key_password";

const KEY_PASSWORD_ENV_DEV: &str = "PULSECURE_KEY_PASSWORD";

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

pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditEvent {
    pub id: String,
    pub event_type: String,
    pub subject_id: Option<String>,
    pub details_json: String,
    pub created_at: String,
}

impl SqliteStorage {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let conn = Connection::open(path)?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.init_schema()?;
        Ok(storage)
    }

    pub fn in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory()?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.init_schema()?;
        Ok(storage)
    }

    pub fn load_recent_audit_events(&self, limit: usize) -> Result<Vec<AuditEvent>, StorageError> {
        let conn = self.conn.lock().expect("Lock failed");
        let mut stmt = conn.prepare(
            r"
            SELECT id, event_type, subject_id, details_json, created_at
            FROM audit_events
            ORDER BY created_at DESC
            LIMIT ?1
            ",
        )?;

        let events = stmt
            .query_map(params![limit as i64], |row| {
                Ok(AuditEvent {
                    id: row.get(0)?,
                    event_type: row.get(1)?,
                    subject_id: row.get(2)?,
                    details_json: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(events)
    }

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
                screening_positive INTEGER NOT NULL DEFAULT 0,
                threshold_used REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                encrypted_computation INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                diagnosis_encrypted BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_diagnoses_created
                ON diagnoses(created_at DESC);

            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                subject_id TEXT,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_audit_events_created
                ON audit_events(created_at DESC);
            ",
        )?;

        let _ = conn.execute(
            "ALTER TABLE diagnoses ADD COLUMN screening_positive INTEGER NOT NULL DEFAULT 0",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE diagnoses ADD COLUMN threshold_used REAL NOT NULL DEFAULT 0.5",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE diagnoses ADD COLUMN diagnosis_encrypted BLOB",
            [],
        );

        Ok(())
    }

    fn get_key_password() -> Result<Zeroizing<String>, StorageError> {
        #[cfg(unix)]
        if let Ok(fd_str) = std::env::var(KEY_PASSWORD_FD_ENV) {
            let fd: i32 = fd_str
                .trim()
                .parse()
                .map_err(|_| StorageError::MissingPassword)?;
            if fd <= 2 {
                return Err(StorageError::MissingPassword);
            }

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

        if cfg!(debug_assertions) {
            if let Ok(v) = std::env::var(KEY_PASSWORD_ENV_DEV) {
                let secret = v.trim_end_matches(['\n', '\r']).to_string();
                if secret.is_empty() {
                    return Err(StorageError::MissingPassword);
                }
                return Ok(Zeroizing::new(secret));
            }
        }

        Err(StorageError::MissingPassword)
    }

    fn string_to_risk_level(s: &str) -> RiskLevel {
        match s.to_lowercase().as_str() {
            "low" => RiskLevel::Low,
            "moderate" => RiskLevel::Moderate,
            "high" => RiskLevel::High,
            _ => RiskLevel::Moderate,
        }
    }

    fn pseudonymize_patient_id(patient_id: &str) -> Result<String, StorageError> {
        let password = Self::get_key_password()?;
        let mut hasher = Sha256::new();
        hasher.update(b"pulsecure-patient-id-v2");
        hasher.update(password.as_bytes());
        hasher.update(patient_id.as_bytes());
        let digest = hasher.finalize();
        let suffix = digest
            .iter()
            .take(16)
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Ok(format!("pid:v2:{suffix}"))
    }

    fn encrypt_diagnosis(diagnosis: &Diagnosis) -> Result<Vec<u8>, StorageError> {
        let password = Self::get_key_password()?;
        let plaintext = serde_json::to_vec(diagnosis)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        let encrypted = kdf::encrypt_key(&plaintext, password.as_str())
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        Ok(encrypted.to_bytes())
    }

    fn decrypt_diagnosis(bytes: &[u8]) -> Result<Diagnosis, StorageError> {
        let password = Self::get_key_password()?;
        let encrypted = EncryptedKey::from_bytes(bytes).map_err(|_| StorageError::KeyDecryption)?;
        let plaintext = kdf::decrypt_key(&encrypted, password.as_str())
            .map_err(|_| StorageError::KeyDecryption)?;
        serde_json::from_slice(&plaintext).map_err(|e| StorageError::Serialization(e.to_string()))
    }

    fn row_conversion_error(idx: usize, error: StorageError) -> rusqlite::Error {
        rusqlite::Error::FromSqlConversionFailure(idx, Type::Blob, Box::new(error))
    }

    fn save_audit_event(
        conn: &Connection,
        event_type: &str,
        subject_id: Option<&str>,
        details_json: &str,
    ) -> Result<(), StorageError> {
        let now = chrono::Utc::now();
        let mut hasher = Sha256::new();
        hasher.update(event_type.as_bytes());
        hasher.update(subject_id.unwrap_or_default().as_bytes());
        hasher.update(now.to_rfc3339().as_bytes());
        let digest = hasher.finalize();
        let event_id = format!(
            "audit:{}",
            digest
                .iter()
                .take(16)
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );

        conn.execute(
            r"
            INSERT INTO audit_events (id, event_type, subject_id, details_json, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ",
            params![
                event_id,
                event_type,
                subject_id,
                details_json,
                now.to_rfc3339()
            ],
        )?;

        Ok(())
    }
}

impl Storage for SqliteStorage {
    type Error = StorageError;

    fn save_keys(&self, keys: &KeyPair) -> Result<(), Self::Error> {
        let password = Self::get_key_password()?;
        let conn = self.conn.lock().expect("Lock failed");
        let now = chrono::Utc::now().to_rfc3339();

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

        let mut stmt = conn
            .prepare("SELECT client_key_encrypted, server_key_encrypted FROM keys WHERE id = 1")?;

        let result = stmt.query_row([], |row| {
            let client_encrypted_bytes: Vec<u8> = row.get(0)?;
            let server_encrypted_bytes: Vec<u8> = row.get(1)?;
            Ok((client_encrypted_bytes, server_encrypted_bytes))
        });

        match result {
            Ok((client_encrypted_bytes, server_encrypted_bytes)) => {
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

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM keys WHERE id = 1", [], |row| {
            row.get(0)
        })?;

        Ok(count > 0)
    }

    fn delete_keys(&self) -> Result<(), Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");
        conn.execute("DELETE FROM keys WHERE id = 1", [])?;
        tracing::info!("Deleted keys from storage");
        Ok(())
    }

    fn save_diagnosis(&self, diagnosis: &Diagnosis) -> Result<(), Self::Error> {
        let mut conn = self.conn.lock().expect("Lock failed");
        let patient_id = diagnosis
            .patient_id
            .as_deref()
            .map(Self::pseudonymize_patient_id)
            .transpose()?;
        let mut stored_diagnosis = diagnosis.clone();
        stored_diagnosis.patient_id = patient_id.clone();
        let diagnosis_encrypted = Self::encrypt_diagnosis(&stored_diagnosis)?;

        let tx = conn.transaction()?;
        tx.execute(
            r"
            INSERT INTO diagnoses (
                id, patient_id, probability, prediction, confidence,
                screening_positive, threshold_used, risk_level, encrypted_computation, created_at,
                diagnosis_encrypted
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            ",
            params![
                diagnosis.id,
                patient_id,
                0.0_f64,
                0_i64,
                0.0_f64,
                0_i64,
                0.5_f64,
                "encrypted",
                diagnosis.encrypted_computation as i64,
                diagnosis.created_at.to_rfc3339(),
                diagnosis_encrypted,
            ],
        )?;

        Self::save_audit_event(
            &tx,
            "screening.saved",
            Some(&diagnosis.id),
            r#"{"phi_safe":true,"stored_result":"encrypted"}"#,
        )?;
        tx.commit()?;

        tracing::debug!("Saved diagnosis {} to storage", diagnosis.id);
        Ok(())
    }

    fn load_diagnoses(&self) -> Result<Vec<Diagnosis>, Self::Error> {
        self.load_recent_diagnoses(1000)
    }

    fn load_recent_diagnoses(&self, limit: usize) -> Result<Vec<Diagnosis>, Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");

        let mut stmt = conn.prepare(
            r"
            SELECT id, patient_id, probability, prediction, confidence,
                   screening_positive, threshold_used, risk_level, encrypted_computation, created_at,
                   diagnosis_encrypted
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
                let screening_positive: i64 = row.get(5)?;
                let threshold_used: f64 = row.get(6)?;
                let risk_level_str: String = row.get(7)?;
                let encrypted: i64 = row.get(8)?;
                let created_at_str: String = row.get(9)?;
                let diagnosis_encrypted: Option<Vec<u8>> = row.get(10)?;

                if let Some(bytes) = diagnosis_encrypted {
                    return Self::decrypt_diagnosis(&bytes)
                        .map_err(|e| Self::row_conversion_error(10, e));
                }

                let result = DiagnosisResult {
                    probability,
                    prediction: prediction as u8,
                    screening_positive: screening_positive != 0,
                    threshold_used,
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

    fn load_diagnoses_paginated(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<DiagnosisPage, Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");

        let total_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM diagnoses", [], |row| row.get(0))?;

        let mut stmt = conn.prepare(
            r"
            SELECT id, patient_id, probability, prediction, confidence,
                   screening_positive, threshold_used, risk_level, encrypted_computation, created_at,
                   diagnosis_encrypted
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
                let screening_positive: i64 = row.get(5)?;
                let threshold_used: f64 = row.get(6)?;
                let risk_level_str: String = row.get(7)?;
                let encrypted: i64 = row.get(8)?;
                let created_at_str: String = row.get(9)?;
                let diagnosis_encrypted: Option<Vec<u8>> = row.get(10)?;

                if let Some(bytes) = diagnosis_encrypted {
                    return Self::decrypt_diagnosis(&bytes)
                        .map_err(|e| Self::row_conversion_error(10, e));
                }

                let result = DiagnosisResult {
                    probability,
                    prediction: prediction as u8,
                    screening_positive: screening_positive != 0,
                    threshold_used,
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

        Ok(DiagnosisPage::new(
            diagnoses,
            total_count as usize,
            offset,
            limit,
        ))
    }

    fn count_diagnoses(&self) -> Result<usize, Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM diagnoses", [], |row| row.get(0))?;

        Ok(count as usize)
    }

    fn delete_diagnosis(&self, id: &str) -> Result<(), Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");
        conn.execute("DELETE FROM diagnoses WHERE id = ?1", params![id])?;
        Ok(())
    }

    fn clear_all(&self) -> Result<(), Self::Error> {
        let conn = self.conn.lock().expect("Lock failed");
        conn.execute_batch("DELETE FROM keys; DELETE FROM diagnoses;")?;
        tracing::warn!("Cleared all data from storage");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_password() {
        std::env::set_var(
            "PULSECURE_KEY_PASSWORD",
            "test_password_for_ci_only_32chars!",
        );
    }

    #[test]
    fn test_keys_roundtrip() {
        setup_test_password();

        let storage = SqliteStorage::in_memory().expect("Should create db");

        assert!(!storage.has_keys().expect("Should check"));
        assert!(storage.load_keys().expect("Should load").is_none());

        let client = ClientKey::from_bytes(vec![1, 2, 3, 4]);
        let server = ServerKey::from_bytes(vec![5, 6, 7, 8]);
        let keys = KeyPair::new(client, server);

        storage.save_keys(&keys).expect("Should save");
        assert!(storage.has_keys().expect("Should check"));

        let loaded = storage
            .load_keys()
            .expect("Should load")
            .expect("Should exist");
        assert_eq!(loaded.client.fingerprint, keys.client.fingerprint);

        storage.delete_keys().expect("Should delete");
        assert!(!storage.has_keys().expect("Should check"));
    }

    #[test]
    fn test_diagnosis_crud() {
        setup_test_password();
        let storage = SqliteStorage::in_memory().expect("Should create db");

        assert_eq!(storage.count_diagnoses().expect("Should count"), 0);

        let result = DiagnosisResult::new(0.75);
        let diagnosis = Diagnosis::new(result, true);
        let id = diagnosis.id.clone();

        storage.save_diagnosis(&diagnosis).expect("Should save");
        assert_eq!(storage.count_diagnoses().expect("Should count"), 1);

        let loaded = storage.load_diagnoses().expect("Should load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, id);

        storage.delete_diagnosis(&id).expect("Should delete");
        assert_eq!(storage.count_diagnoses().expect("Should count"), 0);
    }

    #[test]
    fn test_patient_id_is_pseudonymized_at_rest() {
        setup_test_password();
        let storage = SqliteStorage::in_memory().expect("Should create db");

        let result = DiagnosisResult::new(0.75);
        let diagnosis = Diagnosis::with_patient(result, "MRN-123456", true);
        storage.save_diagnosis(&diagnosis).expect("Should save");

        let loaded = storage.load_diagnoses().expect("Should load");
        let stored_id = loaded[0].patient_id.as_ref().expect("patient pseudonym");
        assert_ne!(stored_id, "MRN-123456");
        assert!(stored_id.starts_with("pid:v2:"));
    }

    #[test]
    fn test_diagnosis_clinical_result_is_encrypted_at_rest() {
        setup_test_password();
        let storage = SqliteStorage::in_memory().expect("Should create db");

        let result = DiagnosisResult::with_threshold(0.75, 0.2);
        let diagnosis = Diagnosis::with_patient(result, "MRN-654321", true);
        let id = diagnosis.id.clone();
        storage.save_diagnosis(&diagnosis).expect("Should save");

        let conn = storage.conn.lock().expect("Lock failed");
        let (probability, prediction, risk_level, encrypted): (f64, i64, String, Vec<u8>) = conn
            .query_row(
                "SELECT probability, prediction, risk_level, diagnosis_encrypted FROM diagnoses WHERE id = ?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .expect("row exists");

        assert_eq!(probability, 0.0);
        assert_eq!(prediction, 0);
        assert_eq!(risk_level, "encrypted");
        assert!(!encrypted.is_empty());
        assert!(!String::from_utf8_lossy(&encrypted).contains("MRN-654321"));
        drop(conn);

        let loaded = storage.load_diagnoses().expect("Should load");
        assert_eq!(loaded[0].result.probability, 0.75);
        assert!(loaded[0].result.screening_positive);
        assert!(loaded[0]
            .patient_id
            .as_deref()
            .unwrap()
            .starts_with("pid:v2:"));
    }

    #[test]
    fn test_phi_safe_audit_event_created_for_screening() {
        setup_test_password();
        let storage = SqliteStorage::in_memory().expect("Should create db");

        let diagnosis = Diagnosis::with_patient(
            DiagnosisResult::with_threshold(0.75, 0.2),
            "MRN-999999",
            true,
        );
        let id = diagnosis.id.clone();
        storage.save_diagnosis(&diagnosis).expect("Should save");

        let conn = storage.conn.lock().expect("Lock failed");
        let (event_type, subject_id, details): (String, String, String) = conn
            .query_row(
                "SELECT event_type, subject_id, details_json FROM audit_events LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("audit row exists");

        assert_eq!(event_type, "screening.saved");
        assert_eq!(subject_id, id);
        assert!(details.contains("phi_safe"));
        assert!(!details.contains("MRN-999999"));
        assert!(!details.contains("0.75"));
    }
}
