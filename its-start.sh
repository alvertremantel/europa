#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  printf 'Usage: %s <checkpoint-path>\n\n' "$0"
  printf 'Start the Europa Interpretability Suite for a checkpoint.\n'
  printf 'Run this from the repository root.\n'
  printf 'Requires: uv sync, npm install --prefix eur_is/frontend\n'
  exit 0
fi

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <checkpoint-path>\n' "$0" >&2
  exit 1
fi

CHECKPOINT_PATH="$1"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  printf 'Checkpoint not found: %s\n' "$CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ ! -d "eur_is/frontend/node_modules" ]]; then
  printf 'Frontend dependencies not found. Run: npm install --prefix eur_is/frontend\n' >&2
  exit 1
fi

for port in 8000 5173; do
  if ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
    printf 'Port %s is already in use.\n' "$port" >&2
    exit 1
  fi
done

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

export EUR_IS_CHECKPOINT_PATH="$CHECKPOINT_PATH"

uv run uvicorn eur_is.backend.main:app --reload &
BACKEND_PID=$!

for _ in {1..20}; do
  if uv run python - <<'PY'
import json
from urllib.request import urlopen

try:
    with urlopen("http://127.0.0.1:8000/api/health", timeout=1) as response:
        payload = json.loads(response.read().decode("utf-8"))
        raise SystemExit(0 if payload.get("status") == "ok" else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 0.5
done

if ! uv run python - <<'PY'
import json
from urllib.request import urlopen

try:
    with urlopen("http://127.0.0.1:8000/api/health", timeout=1) as response:
        payload = json.loads(response.read().decode("utf-8"))
        raise SystemExit(0 if payload.get("status") == "ok" else 1)
except Exception:
    raise SystemExit(1)
PY
then
  printf 'Backend failed to start. Check the checkpoint path and port 8000 logs.\n' >&2
  exit 1
fi

printf 'Started backend with checkpoint: %s\n' "$EUR_IS_CHECKPOINT_PATH"
printf 'Frontend: http://localhost:5173\n'
printf 'Backend health: http://localhost:8000/api/health\n'

npm run dev --prefix eur_is/frontend
