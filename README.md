# Pulsecure

Pulsecure is a local proof of concept for privacy-preserving cardiovascular screening. It combines a Rust web API, a React/Vite interface, SQLite storage, and a signed 15-feature NHANES-based model with encrypted inference through `tfhe-rs`.

The goal is not to ship a medical product. The goal is to demonstrate how sensitive health features can move through a small, auditable ML system where privacy, model integrity, and operational boundaries are treated as first-class engineering concerns.

This is not a medical device and must not be used for real clinical decisions.

## What It Shows

- Encrypted screening workflow using `tfhe-rs`.
- Rust API built with Axum and Tokio.
- React/Vite web UI served by the Rust binary.
- SQLite persistence for local diagnosis history and audit events.
- Signed model loading with an explicit 15-feature contract.
- Sanitized logging and security headers for the local web surface.
- Python training/export pipeline for regenerating the model artifact.

## Quick Start

```bash
make run
```

Open `http://127.0.0.1:8080`.

Equivalent explicit path:

```bash
make secret
docker compose up --build pulsecure
```

Useful commands:

```bash
make run
make stop
make down
make logs
make check
make fhe-check
```

`make check` runs Rust compile checks, model contract validation, fast contract/storage/API tests, and the web build. The heavy FHE path is kept separate as `make fhe-check`.

## Architecture

```text
Browser
  -> Rust web API (Axum)
  -> Application service
  -> TFHE adapter
  -> SQLite adapter
  -> signed model artifact
```

| Area | Responsibility |
| --- | --- |
| `src/web` | HTTP routes, static web serving, request validation, security headers |
| `src/application` | Inference orchestration and diagnosis history |
| `src/domain` | Patient features, diagnosis records, key material |
| `src/adapters` | TFHE, SQLite, differential privacy helpers, log sanitization |
| `src/ports` | Internal interfaces for storage, privacy, and FHE |
| `web` | React/Vite user interface |
| `python` | Model training, evaluation, and export tooling |
| `models` | Signed model artifact and manifest |

## Model Contract

Pulsecure currently accepts one 15-feature cardiovascular screening contract:

```text
age
sex
race
bmi
waist
sbp
hypertension
total_chol
hdl
hba1c
serum_glucose
diabetes
egfr
urine_albumin
ever_smoker
```

Bundled artifact metrics:

| Metric | Value |
| --- | ---: |
| Calibrated ROC-AUC | 0.9185 |
| Quantized ROC-AUC | 0.9186 |
| Recall at threshold | 95.56% |
| Precision at threshold | 13.58% |
| Clinical threshold | 0.0728 |

The model is tuned for screening sensitivity, which means false positives are expected. A positive result means "review first" in a research workflow, not a diagnosis.

## API Surface

The Rust binary serves the UI and exposes a small local API:

- `GET /api/health`
- `GET /api/model`
- `POST /api/screenings`
- `GET /api/screenings`
- `GET /api/audit`

The service limits FHE concurrency with `PULSECURE_FHE_CONCURRENCY` because encrypted computation is intentionally heavier than normal plaintext inference.

## Local Development

Run the Rust server:

```bash
cargo run
```

Build the web UI:

```bash
cd web
npm install
npm run build
```

Regenerate the model artifact:

```bash
cd python
pip install -e .
cd ..
python -m python.src.training.pipeline
```

## Configuration

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `PULSECURE_WEB_ADDR` | Bind address, default `127.0.0.1:8080` |
| `PULSECURE_DB_PATH` | SQLite database path |
| `PULSECURE_MODEL_PATH` | Model directory, default `models` |
| `PULSECURE_WEB_DIST` | Built web UI directory, default `web/dist` |
| `PULSECURE_FHE_CONCURRENCY` | Maximum concurrent FHE jobs, default `1` |
| `PULSECURE_LOG_MODE` | `stdout` or `file` |

## Security Boundaries

Pulsecure is intentionally local-first. It does not provide clinical validation, multi-tenant authorization, production key management, or healthcare compliance controls. Those are outside the scope of this proof of concept.

The project focuses on the engineering pieces that are still useful to evaluate: cryptographic workflow isolation, signed model loading, explicit input contracts, auditability, and clear failure behavior.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
