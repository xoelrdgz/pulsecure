# Deployment Guide

## Overview

Pulsecure is designed for local deployment. No cloud infrastructure required.

## Requirements

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM (FHE operations are memory-intensive)
- 4+ CPU cores recommended

## Quick Start

```bash
git clone https://github.com/xoelrdgz/pulsecure.git
cd pulsecure

# Build and run (generates secrets automatically)
make build
make run
```

The Makefile handles secret generation, Docker builds, and TUI attachment.

## Docker Architecture

```
+---------------------------------------+
|          Docker Container             |
|  +-------------------------------+    |
|  |       Pulsecure TUI           |    |
|  |   (Ratatui Terminal UI)       |    |
|  +-------------------------------+    |
|  +-------------------------------+    |
|  |    FHE Inference Engine       |    |
|  |        (tfhe-rs)              |    |
|  +-------------------------------+    |
|  +-------------------------------+    |
|  |    Local SQLite Storage       |    |
|  |    (mounted volume)           |    |
|  +-------------------------------+    |
+---------------------------------------+
         |
         v Volume Mount
+---------------------------------------+
|    Host: ./app-data/                  |
|    - pulsecure.db (encrypted keys)    |
+---------------------------------------+
```

## Build Process

Multi-stage Dockerfile:

1. Python stage: Train and export calibrated model
2. Rust stage: Compile application with tfhe-rs
3. Runtime stage: Minimal Debian image

## Volume Mounts

| Path | Purpose |
|------|---------|
| ./app-data/pulsecure.db | Encrypted keys and diagnoses |
| ./secrets/ | Key password and signing keys |

## Security Configuration

### Network

- No ports exposed by default
- Container runs with network_mode: none
- True air-gapped operation

### Filesystem

- read_only: true (container filesystem)
- tmpfs for /tmp
- no-new-privileges security option

### Resources

```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: "4"
    reservations:
      memory: 2G
      cpus: "1"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| PULSECURE_KEY_PASSWORD_FILE | Yes | Path to key password file (Docker secret) |
| PULSECURE_DB_PATH | No | Database location (default: /app/data/pulsecure.db) |
| PULSECURE_ALLOW_UNSIGNED_MODELS | No | Allow unsigned models (dev only) |
| RUST_LOG | No | Log level (default: info) |

## Makefile Commands

```bash
make build       # Build Docker image
make run         # Run with TUI attached
make build-prod  # Build production image (signed model)
make run-prod    # Run production image
make stop        # Stop container
make down        # Stop and remove container
make logs        # View logs
```

## Troubleshooting

### Out of memory during FHE operations

FHE key generation requires significant RAM. Increase memory limit:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

### Slow performance

- First run is slower due to key generation
- Subsequent runs use cached keys
- Ensure release build (default in Docker)

### TUI not displaying

Ensure terminal supports:
- 256-color mode
- UTF-8 encoding
- Minimum size 80x24

```bash
make run
```

## Production Checklist

- Volume backup strategy configured
- Resource limits set appropriately
- Logging configured for audit
- Container image scanned for vulnerabilities
- Network isolation verified
- Key password stored in secrets manager
