.PHONY: secret build run stop down logs check model-check fhe-check

COMPOSE ?= docker compose

secret:
	@mkdir -p secrets
	@if [ ! -f secrets/pulsecure_key_password ]; then \
		umask 077; \
		openssl rand -base64 32 > secrets/pulsecure_key_password; \
		echo "Created secrets/pulsecure_key_password"; \
	fi

build:
	@$(COMPOSE) build pulsecure

run: secret
	@$(COMPOSE) up --build pulsecure

stop:
	@$(COMPOSE) stop pulsecure

down:
	@$(COMPOSE) down

logs:
	@$(COMPOSE) logs -f --no-log-prefix pulsecure

check:
	@bash scripts/production-check.sh

model-check:
	@python3 scripts/validate-model-contract.py --strict-metadata models/calibrated_model.json

fhe-check:
	@cargo test -- --ignored
	@cargo bench --bench fhe_pipeline
