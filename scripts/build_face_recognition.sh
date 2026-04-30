#!/usr/bin/env bash
# build_face_recognition.sh — Install Orion for the CryptoFace submission.
#
# Usage: bash scripts/build_face_recognition.sh
#
# Installs Orion (pinned to a specific commit) if not already importable.
# All other dependencies (torch, insightface, opencv-python, pyyaml, numpy)
# must be installed separately, e.g. via: pip install -r requirements.txt

set -euo pipefail

ORION_REPO="https://github.com/vboddeti/orion"
ORION_COMMIT="ff56c8237743f6b38308f058f0bcabab4e0cabf8"

echo "[build] CryptoFace submission — installing Orion"

# Install Orion from GitHub if not already importable
if ! python3 -c "import orion" 2>/dev/null; then
    echo "[build] Orion not found — installing ${ORION_REPO}@${ORION_COMMIT} ..."
    python3 -m pip install "git+${ORION_REPO}@${ORION_COMMIT}"
else
    echo "[build] Orion already installed."
fi

echo "[build] Done."
