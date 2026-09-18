#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORE="$REPO_ROOT/services/aisha-core"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 was not found."
  echo "Install it with: brew install python@3.12"
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
echo "Run the mock backend first:"
echo "  AISHA_PROFILE=mock pytest -q"
echo
echo "For this M2 Max / 96 GB Mac, pull the local model separately:"
echo "  ollama pull qwen3.5:35b-mlx"
echo "Then set AISHA_PROFILE=mac-m2max-96gb in services/aisha-core/.env"
echo "and run: python scripts/doctor.py"
