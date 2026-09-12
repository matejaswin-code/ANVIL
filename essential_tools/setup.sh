#!/usr/bin/env bash
#
# ANVIL — local environment setup (Linux / macOS)
#
# Creates a project-local Python venv and a project-local node_modules.
# Nothing is installed globally. The only things you need already on your
# machine are Python and Node.js themselves.
#
#   Usage:  ./essential_tools/setup.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; NC=$'\033[0m'

say()  { printf "%s\n" "$*"; }
step() { printf "\n%s==>%s %s%s%s\n" "$RED" "$NC" "$BOLD" "$*" "$NC"; }
ok()   { printf "  %s✓%s %s\n" "$GREEN" "$NC" "$*"; }
warn() { printf "  %s!%s %s\n" "$YELLOW" "$NC" "$*"; }
die()  { printf "\n  %s✗ %s%s\n\n" "$RED" "$*" "$NC" >&2; exit 1; }

cat <<'BANNER'

   █████╗ ███╗   ██╗██╗   ██╗██╗██╗
  ██╔══██╗████╗  ██║██║   ██║██║██║
  ███████║██╔██╗ ██║██║   ██║██║██║
  ██╔══██║██║╚██╗██║╚██╗ ██╔╝██║██║
  ██║  ██║██║ ╚████║ ╚████╔╝ ██║███████╗
  ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝╚══════╝
  local-first AI 3D asset pipeline

BANNER

# ---------------------------------------------------------------------------
step "Checking prerequisites"
# ---------------------------------------------------------------------------

# Python: need 3.10+
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  die "Python 3.10 or newer is required but wasn't found.
      Install it from https://www.python.org/downloads/ (or your package manager)
      and re-run this script. This is the one dependency the script can't install
      for you, since it's the thing that runs everything else."
fi
ok "Python: $($PYTHON_BIN --version 2>&1) ($(command -v "$PYTHON_BIN"))"

# venv module must be present (Debian/Ubuntu split it into python3-venv)
if ! "$PYTHON_BIN" -c "import venv" >/dev/null 2>&1; then
  die "Python is installed but the 'venv' module is missing.
      On Debian/Ubuntu:  sudo apt install python3-venv
      Then re-run this script."
fi

# Node: optional, only powers the in-browser 3D preview
NODE_OK=0
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
  if [ "$NODE_MAJOR" -ge 18 ]; then
    NODE_OK=1
    ok "Node.js: $(node --version)"
  else
    warn "Node.js $(node --version) found, but 18+ is needed for the 3D preview. Skipping it."
  fi
else
  warn "Node.js not found — the browser 3D preview will be disabled."
  warn "Everything else works. Install Node 18+ and re-run to enable it."
fi

# ---------------------------------------------------------------------------
step "Creating the local Python environment"
# ---------------------------------------------------------------------------

if [ -d "$VENV_DIR" ]; then
  ok "Reusing existing venv at essential_tools/venv"
else
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  ok "Created essential_tools/venv"
fi

VENV_PY="$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || die "venv was created but $VENV_PY isn't executable. Delete essential_tools/venv and re-run."

"$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel
ok "pip upgraded inside the venv"

# ---------------------------------------------------------------------------
step "Installing Python dependencies (local to the venv)"
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"
"$VENV_PY" -m pip install --quiet -r requirements.txt
ok "Installed from requirements.txt"
say "  ${DIM}Nothing was installed outside essential_tools/venv.${NC}"

# ---------------------------------------------------------------------------
step "Installing frontend vendor files (local to node_modules)"
# ---------------------------------------------------------------------------

if [ "$NODE_OK" -eq 1 ]; then
  # --prefix keeps everything inside the project; no -g, ever.
  npm install --prefix "$PROJECT_ROOT" --silent --no-audit --no-fund
  node "$SCRIPT_DIR/vendor.js"
  ok "three.js vendored into frontend/vendor/"
else
  mkdir -p "$PROJECT_ROOT/frontend/vendor"
  warn "Skipped — the app runs fine, you just won't get the in-browser mesh preview."
fi

# ---------------------------------------------------------------------------
step "Preparing configuration"
# ---------------------------------------------------------------------------

if [ -f "$PROJECT_ROOT/.env" ]; then
  ok ".env already exists — leaving it alone"
else
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
  ok "Created .env from .env.example"
  warn "Open .env and fill in your model paths / API keys before real generation."
fi

mkdir -p "$PROJECT_ROOT/sessions" "$PROJECT_ROOT/assets" "$PROJECT_ROOT/logs"
ok "Created sessions/, assets/, logs/"

# ---------------------------------------------------------------------------
step "Verifying the install"
# ---------------------------------------------------------------------------

if "$VENV_PY" "$SCRIPT_DIR/verify.py"; then
  ok "Verification passed"
else
  warn "Verification reported problems — see above. The install itself completed."
fi

# ---------------------------------------------------------------------------
cat <<EOF

${GREEN}${BOLD}Setup complete.${NC}

  Start ANVIL:      ${BOLD}./essential_tools/run.sh${NC}
  Then open:        ${BOLD}http://127.0.0.1:8765${NC}

  Everything lives inside this project directory:
    ${DIM}essential_tools/venv/   Python environment
    node_modules/           frontend packages
    sessions/               generated assets + checkpoints${NC}

  Out of the box ANVIL runs on ${BOLD}mock backends${NC} — the full 10-step pipeline
  works on CPU with no model weights, so you can confirm everything is wired up
  before downloading anything. Switch to real backends in .env or the Settings
  panel when you're ready.

EOF
