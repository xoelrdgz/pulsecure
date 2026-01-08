
.PHONY: secret build run build-prod run-prod tui up stop down logs

COMPOSE ?= docker compose -f docker-compose.yml -f docker-compose.dev.yml
COMPOSE_PROD := docker compose -f docker-compose.yml

secret:
	@mkdir -p secrets
	@if [ ! -f secrets/pulsecure_key_password ]; then \
		umask 077; \
		openssl rand -base64 32 > secrets/pulsecure_key_password; \
		echo "Created secrets/pulsecure_key_password"; \
	fi
	@if [ ! -f secrets/pulsecure_model_signing_key_b64 ] || [ ! -f secrets/pulsecure_model_signing_pubkey_b64 ]; then \
		umask 077; \
		cargo run --quiet --bin generate_keypair -- \
			--out-seed secrets/pulsecure_model_signing_key_b64 \
			--out-pub secrets/pulsecure_model_signing_pubkey_b64; \
		echo "Created secrets/pulsecure_model_signing_key_b64 and secrets/pulsecure_model_signing_pubkey_b64"; \
	fi

build:
	@$(COMPOSE) build pulsecure

build-prod:
	@$(COMPOSE_PROD) build pulsecure

run: secret
	@COMPOSE="$(COMPOSE)" bash scripts/pulsecure-tui.sh

run-prod: secret
	@COMPOSE="$(COMPOSE_PROD)" bash scripts/pulsecure-tui.sh

tui: run

up:
	@$(COMPOSE) up --build -d pulsecure

stop:
	@$(COMPOSE) stop pulsecure

down:
	@$(COMPOSE) down

logs:
	@$(COMPOSE) logs -f --no-log-prefix pulsecure
