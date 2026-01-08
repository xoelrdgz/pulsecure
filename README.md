# Pulsecure

[![Rust](https://img.shields.io/badge/Rust-1.92+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)

Privacy-preserving cardiovascular disease risk prediction using Fully Homomorphic Encryption (FHE).

## Overview

Pulsecure enables machine learning inference on encrypted medical data. The server processes patient information without ever seeing it in plaintext, providing mathematical privacy guarantees rather than policy-based ones.

```
Patient Data -> Encrypt -> [FHE Inference] -> Decrypt -> Risk Assessment
                               |
                     Server sees only
                     encrypted values
```

## Model Performance

The model uses the NHANES dataset from CDC with isotonic calibration for accurate probability estimates.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.85 |
| Brier Score (calibrated) | 0.06 |
| Sensitivity (Recall) | 97.6% |
| Clinical Threshold | 0.033 |
| NPV (Negative Predictive Value) | 99.4% |

The model is optimized for high sensitivity to minimize missed cardiovascular cases. Low precision is acceptable in screening contexts where follow-up testing is standard practice.

> **WARNING**: This is a proof-of-concept. The model tends toward false positives due to its high-sensitivity tuning. It should not be used for actual medical decisions. Always consult a healthcare professional.

## Quick Start

### Docker (Recommended)

```bash
# Clone and enter directory
git clone https://github.com/xoelrdgz/pulsecure.git
cd pulsecure

# Build and run (generates secrets automatically)
make build
make run
```

The `make run` command:
- Creates `./secrets/pulsecure_key_password` if missing
- Generates model signing keypair if missing
- Builds the Docker image
- Attaches to the TUI

### Other Commands

```bash
make stop      # Stop container
make down      # Stop and remove container
make logs      # View logs
```

### Production

Production builds require a signed model:

```bash
make build-prod
make run-prod
```

### Manual Installation

Prerequisites: Rust 1.92+, Python 3.11+, SQLite

```bash
# Train Python model
cd python && pip install -e . && python -m src.training.pipeline && cd ..

# Run
export PULSECURE_KEY_PASSWORD="your-secure-password"
cargo build --release
./target/release/pulsecure
```

## Architecture

```
python/src/
├── utils.py              # Shared utilities (data loading, quantization)
├── data/
│   └── data_loader.py    # NHANES dataset loading with SMOTE
├── training/
│   └── pipeline.py       # Train, calibrate, quantize, export
├── analysis/
│   ├── evaluate_model.py # Model evaluation with calibration
│   └── feature_analysis.py
└── export/
    └── exporter.py       # Export to JSON/Rust

src/                      # Rust application
├── domain/               # Core types (Patient, Diagnosis, Crypto)
├── ports/                # Trait definitions (FheEngine, Storage)
├── adapters/             # Implementations (tfhe-rs, SQLite)
├── application/          # Use cases (InferenceService)
└── tui/                  # Terminal interface
```

## ML Pipeline

### Features

| NHANES Code | Description |
|-------------|-------------|
| RIDAGEYR | Age in years |
| BPQ020 | Hypertension diagnosed |
| BPXSY1 | Systolic blood pressure (mmHg) |
| SMQ020 | Smoked 100+ cigarettes ever |
| LBDHDD | HDL cholesterol (mg/dL) |
| LBXSCR | Serum creatinine (mg/dL) |
| BMXWAIST | Waist circumference (cm) |
| DIQ010 | Diabetes diagnosed |
| LBXGH | HbA1c (%) |

### Training Pipeline

1. **Data Loading**: NHANES dataset from CDC via Kaggle
2. **Class Balancing**: SMOTE oversampling for minority class
3. **Model Training**: Logistic Regression with Elastic Net (C=0.1, l1_ratio=0.3)
4. **Calibration**: Isotonic Regression for accurate probabilities
5. **Threshold Optimization**: Target recall of 90% or higher
6. **Quantization**: 12-bit fixed-point for FHE compatibility
7. **Export**: JSON parameters and Rust constants for tfhe-rs

### Docker Commands

```bash
# Train and export model
docker compose -f docker-compose.analysis.yml run --rm train

# Evaluate model performance
docker compose -f docker-compose.analysis.yml run --rm evaluate

# Feature importance analysis
docker compose -f docker-compose.analysis.yml run --rm features

# Interactive Python shell
docker compose -f docker-compose.analysis.yml run --rm shell
```

## FHE Quantization

The model is quantized to 12-bit fixed-point integers for FHE inference:

- Scale factor: 4096 (2^12)
- Sigmoid: 256-entry lookup table
- Calibration: 64-entry isotonic LUT
- Accumulator: 64-bit integers (no overflow)

Quantization fidelity: AUC difference < 0.001 compared to float model.

## Security

### Cryptographic Design

| Component | Algorithm |
|-----------|-----------|
| FHE Scheme | TFHE (tfhe-rs) |
| Key Encryption | Argon2id + AES-256-GCM |
| Key Fingerprint | SHA-256 (truncated) |
| Randomness | ChaCha20 CSPRNG |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| PULSECURE_KEY_PASSWORD_FD | No | Read key password from a file descriptor (recommended for production) |
| PULSECURE_KEY_PASSWORD_FILE | No | Read key password from a file path (e.g., Docker secret at `/run/secrets/pulsecure_key_password`) |
| PULSECURE_KEY_PASSWORD | Dev-only | Env var password (refused in release builds) |
| PULSECURE_DB_PATH | No | SQLite database path |
| PULSECURE_ALLOW_UNSIGNED_MODELS | No (dev-only) | Allow unsigned models (development only) |
| PULSECURE_DP_MAX_EPSILON | No | DP max epsilon budget (default: 1.0) |
| PULSECURE_DP_DEFAULT_EPSILON | No | DP default epsilon per query (default: 0.1) |
| PULSECURE_DP_AGG_EPS_WEIGHTS | No | DP aggregation epsilon split weights as `count,rate,confidence` (default: `1,1,1`) |
| RUST_LOG | No | Log level (default: info) |

### Operational Security

- Run with `network_mode: none` for air-gapped operation
- Generate passwords with `openssl rand -base64 32`
- Store passwords in a secrets manager
- Backup encrypted database regularly

## Performance

| Operation | Latency |
|-----------|---------|
| Key Generation | 2-5s |
| Encryption | 100ms |
| FHE Inference | 1-5s |
| Decryption | 50ms |

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [FHE Concepts](docs/FHE_GUIDE.md)
- [Privacy Model](docs/PRIVACY.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Threat Model](docs/THREAT_MODEL.md)

## License

GPLv3. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Zama](https://zama.ai/) for tfhe-rs
- [CDC NHANES](https://www.cdc.gov/nchs/nhanes/) for the dataset
- [Ratatui](https://ratatui.rs/) for terminal UI
