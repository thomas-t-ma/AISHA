#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORE="$REPO_ROOT/services/aisha-core"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 was not found. Install it first (Homebrew is fine), then rerun."
  exit 1
fi

cd "$CORE"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

if [ ! -f .env ]; then
  cp "$REPO_ROOT/.env.example" .env
fi

echo
echo "AISHA Core environment is ready."
echo "Next:"
echo "  1) Install Ollama separately if desired."
echo "  2) ollama pull qwen3:4b"
echo "  3) python scripts/doctor.py"
echo "  4) uvicorn aisha.main:app --reload"
