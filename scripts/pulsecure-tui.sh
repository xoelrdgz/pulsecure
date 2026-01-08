#!/usr/bin/env bash
set -euo pipefail

SERVICE="pulsecure"
CONTAINER="pulsecure"

# Allow overriding compose invocation (e.g., COMPOSE='docker compose -f docker-compose.yml -f docker-compose.dev.yml')
COMPOSE=${COMPOSE:-"docker compose"}

# Ensure the container is running. Using `compose up --detach` ensures stdin/TTY is
# wired correctly for an interactive TUI; `compose up --attach` does not reliably
# forward stdin across Compose versions.
running=$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || echo "false")
if [[ "$running" != "true" ]]; then
  ${COMPOSE} up --build -d "${SERVICE}"
fi

# Attach directly to the container's TTY for a clean TUI.
exec docker attach "${CONTAINER}"
