#!/usr/bin/env bash
#
# ANVIL — start the local server.
# Uses the project-local venv; never touches a global Python.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$SCRIPT_DIR/venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "Local environment not found. Run this first:"
  echo "  ./essential_tools/setup.sh"
  exit 1
fi

cd "$PROJECT_ROOT"

HOST="$(grep -E '^ANVIL_HOST=' .env 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
PORT="$(grep -E '^ANVIL_PORT=' .env 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

echo ""
echo "  ANVIL starting on http://${HOST}:${PORT}"
echo "  Ctrl-C to stop."
echo ""

exec "$VENV_PY" -m server.app "$@"
