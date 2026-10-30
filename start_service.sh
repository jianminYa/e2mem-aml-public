#!/usr/bin/env bash
set -euo pipefail

# E²-Mem Evidence Anchor Production Startup Script for AML
# Reads configuration from environment variables or .env file

if [ -f .env ]; then
    echo "Loading environment configuration from .env..."
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
fi

PORT="${PORT:-8000}"
E2MEM_HOST="${E2MEM_HOST:-0.0.0.0}"
E2MEM_DB="${E2MEM_DB:-data/memory.sqlite3}"
E2MEM_MODE="${E2MEM_MODE:-competition}"

mkdir -p "$(dirname "$E2MEM_DB")"

echo "=========================================================="
echo " Starting E²-Mem Evidence Anchor (AML Service v2.6.3)"
echo " Host:     ${E2MEM_HOST}:${PORT}"
echo " Database: ${E2MEM_DB}"
echo " Mode:     ${E2MEM_MODE}"
echo "=========================================================="

exec python3 -m service.api
