#!/usr/bin/env bash
# Create a gateway API key for one client. The key is shown once and never stored;
# GATEWAY_KEYS holds only its hash. Usage: ./make-key.sh <client-name>
set -euo pipefail
name=${1:-}
if [[ ! "$name" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "usage: $0 <client-name> (letters, digits, - and _)" >&2
  exit 1
fi
key=$(openssl rand -hex 32)
hash=$(printf '%s' "$key" | openssl dgst -sha256 -r | cut -d' ' -f1)
echo "API key for $name (give this to the client, it is not saved):"
echo "  $key"
echo "GATEWAY_KEYS entry (comma-separate multiple clients):"
echo "  $name=$hash"
