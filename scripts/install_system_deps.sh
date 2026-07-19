#!/usr/bin/env bash
#
# install_system_deps.sh — install the OS-level packages required to build and
# run the face-verification harness and the CryptoFace reference submission.
#
# Python dependencies are handled separately via requirements.txt.
#
# Usage:  bash scripts/install_system_deps.sh
set -euo pipefail

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
    SUDO="sudo"
fi

${SUDO} apt-get update
${SUDO} apt-get install -y \
    build-essential \
    python3-dev \
    golang-go \
    libgraphviz-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git-lfs

git lfs install

echo "[install_system_deps] done. Next: bash scripts/install_python_deps.sh"
