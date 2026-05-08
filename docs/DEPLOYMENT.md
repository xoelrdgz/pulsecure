# Deployment

Pulsecure has one PoC deployment path: local web through Docker Compose.

```bash
make run
```

Open `http://127.0.0.1:8080`.

## Flow

`make run` creates `secrets/pulsecure_key_password` when missing and runs:

```bash
docker compose up --build pulsecure
```

## Variables

| Variable | Default |
| --- | --- |
| RUST_LOG | info |
| PULSECURE_WEB_ADDR | 0.0.0.0:8080 |
| PULSECURE_DB_PATH | /app/data/pulsecure.db |
| PULSECURE_FHE_CONCURRENCY | 1 |
| PULSECURE_KEY_PASSWORD_FILE | /run/secrets/pulsecure_key_password |

## Check

```bash
make check
```

The heavy FHE test is separate:

```bash
make fhe-check
```
