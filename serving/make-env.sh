#!/usr/bin/env bash
# Create serving/.env with a random VLLM_API_KEY (once). Never prints the key.
# Clients (homelab-services LLM_API_KEY) must use the same value.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then
  echo ".env already exists; leaving it alone"
  exit 0
fi
umask 077
printf 'VLLM_API_KEY=%s\n' "$(openssl rand -hex 24)" > .env
echo "wrote $(pwd)/.env (mode $(stat -c %a .env))"
