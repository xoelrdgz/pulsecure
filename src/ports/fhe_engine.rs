use crate::domain::{
    ClientKey, CryptoError, Diagnosis, EncryptedDiagnosis, EncryptedPatientData, KeyPair,
    PatientData, ServerKey,
};

pub trait FheEngine: Send + Sync {
    fn generate_keys(&self) -> Result<KeyPair, CryptoError>;

    fn encrypt(
        &self,
        data: &PatientData,
        key: &ClientKey,
    ) -> Result<EncryptedPatientData, CryptoError>;

    fn compute(
        &self,
        encrypted: &EncryptedPatientData,
        server_key: &ServerKey,
    ) -> Result<EncryptedDiagnosis, CryptoError>;

    fn decrypt(
        &self,
        result: &EncryptedDiagnosis,
        key: &ClientKey,
    ) -> Result<Diagnosis, CryptoError>;

    fn serialize_keys(&self, keys: &KeyPair) -> Result<(Vec<u8>, Vec<u8>), CryptoError>;

    fn deserialize_keys(
        &self,
        client_bytes: &[u8],
        server_bytes: &[u8],
    ) -> Result<KeyPair, CryptoError>;
}
