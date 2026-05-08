# Architecture

Pulsecure PoC is now local web, Rust API, SQLite storage, and one signed model.

```text
Browser
Rust web API
Application service
TFHE adapter
SQLite adapter
models/calibrated_model.json
```

## Layers

- `src/web`: HTTP API and static web serving
- `src/application`: inference orchestration
- `src/domain`: patient data, diagnosis, and keys
- `src/adapters`: tfhe-rs, SQLite, log sanitization, and differential privacy
- `src/ports`: internal interfaces
- `web`: React/Vite
- `python`: model training and export

## Runtime

The Rust binary always starts the web server. It serves the compiled UI from `/app/web/dist` and exposes `/api/health`, `/api/model`, `/api/screenings`, and `/api/audit`.

## Model

Only the active 15-feature contract is accepted. The loader rejects old contracts and manifests without a valid signature.
