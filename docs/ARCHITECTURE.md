# Pulsecure Architecture

## Overview

Pulsecure follows Hexagonal Architecture (Ports and Adapters) to ensure testability, flexibility, and clear separation of concerns.

## System Architecture

```
+------------------+
|    TUI Layer     |  Ratatui terminal interface
+------------------+
         |
+------------------+
| Application Layer|  Use cases, orchestration
+------------------+
         |
+------------------+
|   Domain Layer   |  Core types (Patient, Diagnosis, Crypto)
+------------------+
         |
+------------------+
|   Ports Layer    |  Trait definitions (FheEngine, Storage)
+------------------+
         |
+------------------+
|  Adapters Layer  |  Implementations (tfhe-rs, SQLite)
+------------------+
```

## Data Flow

### Encrypted Inference Pipeline

1. User inputs patient data via TUI
2. Application loads or generates FHE keys
3. Patient data encrypted with ClientKey
4. Model computes on encrypted data using ServerKey
5. Result decrypted with ClientKey
6. Diagnosis displayed to user

The ServerKey enables computation but cannot decrypt. The ClientKey never leaves the local machine.

## Layer Responsibilities

### Domain Layer (src/domain/)

| Module | Responsibility |
|--------|----------------|
| patient.rs | Patient data structures, validation |
| diagnosis.rs | Diagnosis results, risk levels |
| crypto.rs | FHE key wrappers, encrypted data types |
| kdf.rs | Key derivation (Argon2id) and envelope encryption (AES-256-GCM) |

Rules:

- No external dependencies (pure Rust)
- No I/O operations
- All types implement serde traits

### Ports Layer (src/ports/)

Defines traits that the application depends on:

```rust
pub trait FheEngine {
    fn generate_keys(&self) -> Result<(ClientKey, ServerKey), Error>;
    fn encrypt(&self, data: &PatientData, key: &ClientKey) -> Result<Encrypted, Error>;
    fn compute(&self, encrypted: &Encrypted, key: &ServerKey) -> Result<Encrypted, Error>;
    fn decrypt(&self, result: &Encrypted, key: &ClientKey) -> Result<Diagnosis, Error>;
}

pub trait Storage {
    fn save_keys(&self, keys: &KeyPair) -> Result<(), Error>;
    fn load_keys(&self) -> Result<Option<KeyPair>, Error>;
    fn save_diagnosis(&self, diagnosis: &Diagnosis) -> Result<(), Error>;
}
```

### Adapters Layer (src/adapters/)

| Adapter | Port | Dependency |
|---------|------|------------|
| tfhe/ | FheEngine | tfhe-rs |
| sqlite/ | Storage | rusqlite |
| opendp/ | DifferentialPrivacy | opendp |

Rules:

- Contain all external library details
- Handle library-specific errors
- Never exposed directly to application layer

### Application Layer (src/application/)

Orchestrates use cases using domain types and ports:

```rust
pub struct InferenceService<F: FheEngine, S: Storage> {
    fhe: F,
    storage: S,
}

impl<F: FheEngine, S: Storage> InferenceService<F, S> {
    pub fn run_inference(&self, patient: PatientData) -> Result<Diagnosis, Error> {
        let keys = self.storage.load_keys()?
            .unwrap_or_else(|| self.fhe.generate_keys()?);
        
        let encrypted = self.fhe.encrypt(&patient, &keys.client)?;
        let result = self.fhe.compute(&encrypted, &keys.server)?;
        let diagnosis = self.fhe.decrypt(&result, &keys.client)?;
        
        self.storage.save_diagnosis(&diagnosis)?;
        Ok(diagnosis)
    }
}
```

### TUI Layer (src/tui/)

| Module | Responsibility |
|--------|----------------|
| app.rs | Application state machine |
| styles.rs | Color theme |
| worker.rs | Async inference worker |
| ui/ | View components |

## Python Pipeline

```
python/src/
├── utils.py              # load_nhanes_data, quantize_value, generate_sigmoid_lut
├── data/
│   └── data_loader.py    # NHANES loading, preprocessing, SMOTE
├── training/
│   └── pipeline.py       # Train, calibrate, quantize, export
├── analysis/
│   ├── evaluate_model.py # Model evaluation with calibration
│   └── feature_analysis.py
└── export/
    └── exporter.py       # Export to JSON/Rust
```

The Python pipeline produces:

- models/model.json: Quantized parameters
- models/model.rs: Rust constants for tfhe-rs

## Error Handling

No unwrap() or panic(). All errors are handled explicitly:

```rust
#[derive(Debug, thiserror::Error)]
pub enum PulsecureError {
    #[error("Cryptographic operation failed: {0}")]
    Crypto(#[from] CryptoError),
    
    #[error("Storage operation failed: {0}")]
    Storage(#[from] StorageError),
    
    #[error("Invalid patient data: {0}")]
    Validation(String),
}
```

## Dependency Injection

Adapters are injected at startup:

```rust
fn main() -> Result<()> {
    let fhe = TfheAdapter::new()?;
    let storage = SqliteStorage::new("pulsecure.db")?;
    
    let inference = InferenceService::new(fhe, storage);
    
    App::new(inference).run()
}
```

This allows testing with mock implementations.
