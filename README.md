# Pulsecure

Web PoC for encrypted cardiovascular screening with tfhe-rs and a 15-feature NHANES-based ML model.

This is not a medical product and must not be used for real clinical decisions.

## Run

```bash
make run
```

Open `http://127.0.0.1:8080`.

Equivalent explicit path:

```bash
make secret
docker compose up --build pulsecure
```

## Commands

```bash
make run
make stop
make down
make logs
make check
```

`make check` runs Rust compile checks, model contract validation, fast contract/storage/API tests, and the web build. The heavy FHE test remains separate as `make fhe-check`.

## Stack

- Rust API with Axum
- React/Vite web app served by the Rust binary
- Signed model in `models/calibrated_model.json`
- Static web assets in `web`
- Runtime data in `app-data`

## Model

The active contract is one 15-feature model.

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

- Calibrated ROC-AUC: 0.9185
- Quantized ROC-AUC: 0.9186
- Recall at clinical threshold: 95.56%
- Precision at clinical threshold: 13.58%
- Threshold: 0.0728

## Local Development

```bash
cargo run
```

In another terminal:

```bash
cd web
npm install
npm run build
```

To regenerate the model:

```bash
cd python
pip install -e .
cd ..
python -m python.src.training.pipeline
```

## Layout

```text
src/web
src/application
src/domain
src/adapters
src/ports
web
python
models
tests
```
