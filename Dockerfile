# syntax=docker/dockerfile:1.6

# Multi-stage build for Pulsecure
# Stage 1: Python model training
# Stage 2: Rust compilation
# Stage 3: Minimal runtime

# ============================================
# Stage 1: Python - Train and export FHE model
# ============================================
FROM python:3.11-slim AS python-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY python/pyproject.toml ./
RUN pip install --no-cache-dir .

COPY python/src ./src

RUN mkdir -p /build/data /build/models

# Train calibrated model and export for Rust
RUN python -c "\
    from pathlib import Path; \
    from src.training import train_calibrated_model, export_to_rust, export_to_json; \
    model = train_calibrated_model(precision_bits=12, min_recall=0.90); \
    export_to_json(model, Path('/build/models/model.json')); \
    export_to_rust(model, Path('/build/models/model.rs')); \
    print('Model exported successfully')" || echo "Model export skipped"

# ============================================
# Stage 2: Rust - Compile the application
# ============================================
FROM rust:1.92-slim-bookworm AS rust-builder

WORKDIR /build

# Compilation parallelism
ARG CARGO_BUILD_JOBS=2

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}

COPY Cargo.toml Cargo.lock ./

# Cache dependencies
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn lib() {}" > src/lib.rs && \
    cargo build --release && \
    rm -rf src

COPY src ./src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    touch src/main.rs src/lib.rs && \
    cargo build --release --bins && \
    mkdir -p /build/out && \
    (test -f target/release/pulsecure && cp -f target/release/pulsecure /build/out/pulsecure || cp -f target/release/Pulsecure /build/out/pulsecure) && \
    cp -f target/release/sign_model /build/out/sign_model && \
    cp -f target/release/generate_keypair /build/out/generate_keypair

# If a signing key is provided at build time, sign the exported model.
# This keeps production images "fail-closed" (signed models required) while
# allowing dev images to skip signing and use docker-compose.dev.yml.
COPY --from=python-builder /build/models /build/models
RUN --mount=type=secret,id=pulsecure_model_signing_key_b64,required=false \
    if [ -f /run/secrets/pulsecure_model_signing_key_b64 ]; then \
    /build/out/sign_model /build/models; \
    else \
    echo "No model signing key secret provided; skipping model signing"; \
    fi

# ============================================
# Stage 3: Runtime - Minimal image
# ============================================
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y \
    libsqlite3-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 pulsecure

WORKDIR /app

COPY --from=rust-builder /build/models ./models/
COPY --from=rust-builder /build/out/pulsecure ./

RUN mkdir -p /app/data && chown -R pulsecure:pulsecure /app

USER pulsecure

ENV RUST_LOG=info
ENV PULSECURE_DB_PATH=/app/data/pulsecure.db

ENTRYPOINT ["/app/pulsecure"]
