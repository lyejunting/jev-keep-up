#!/usr/bin/env bash
set -euo pipefail

if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof is required to find processes using ports 8000 and 5173." >&2
  exit 1
fi

listeners() {
  lsof -nP -t -iTCP:8000 -iTCP:5173 -sTCP:LISTEN 2>/dev/null || true
}

pids=$(listeners | sort -u)
if [[ -z "$pids" ]]; then
  echo "Ports 8000 and 5173 are already free."
  exit 0
fi

echo "Stopping processes listening on ports 8000 and 5173: $pids"
# lsof -t returns numeric PIDs, one per line; split them into kill arguments.
kill -TERM $pids

for ((attempt = 0; attempt < 10; attempt++)); do
  if [[ -z "$(listeners)" ]]; then
    echo "Ports 8000 and 5173 are free."
    exit 0
  fi
  sleep 1
done

echo "Ports are still in use after 10 seconds. Stop the remaining processes before retrying." >&2
exit 1
