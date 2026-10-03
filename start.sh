#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -x .venv/bin/python || ! -f frontend/node_modules/vite/bin/vite.js ]]; then
  echo "Dependencies are missing. Follow the one-time setup in README.md first." >&2
  exit 1
fi

# A terminal's version manager may select an older Node than the IDE uses.
node_supported() {
  "$1" -e '
    const [major, minor] = process.versions.node.split(".").map(Number);
    const crypto = require("node:crypto");
    const supported = (major === 20 && minor >= 19) ||
      (major === 22 && minor >= 12) || major > 22;
    process.exit(supported && typeof crypto.getRandomValues === "function" &&
      typeof crypto.hash === "function" ? 0 : 1);
  ' >/dev/null 2>&1
}

node_bin=${NODE_BIN:-node}
if ! node_supported "$node_bin"; then
  if [[ -z "${NODE_BIN:-}" ]]; then
    for candidate in /opt/homebrew/bin/node /usr/local/bin/node; do
      if node_supported "$candidate"; then
        node_bin=$candidate
        break
      fi
    done
  fi
  if ! node_supported "$node_bin"; then
    echo "Vite requires Node.js 20.19+ or 22.12+ with the Node crypto APIs." >&2
    echo "Select a compatible Node version, or run: NODE_BIN=/path/to/node bash start.sh" >&2
    exit 1
  fi
fi
# Resolve before changing into frontend, including relative NODE_BIN overrides.
node_bin=$("$node_bin" -p 'process.execPath')
echo "Using Node $("$node_bin" --version) at $node_bin"

bash stop.sh

backend_pid=
frontend_pid=
cleanup() {
  trap - EXIT INT TERM
  for pid in "$backend_pid" "$frontend_pid"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
backend_pid=$!
(
  cd frontend
  exec "$node_bin" node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort
) &
frontend_pid=$!

echo "Starting JEV Keep Up at http://127.0.0.1:5173/ — press Ctrl+C to stop both services."

# macOS ships Bash 3, which does not support wait -n.
while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
  sleep 1
done

service_exit_code=0
if ! kill -0 "$backend_pid" 2>/dev/null; then
  wait "$backend_pid" || service_exit_code=$?
else
  wait "$frontend_pid" || service_exit_code=$?
fi
exit "$service_exit_code"
