#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

echo "[install_python_deps] environment ready; run stages with uv run python"
