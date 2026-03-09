#!/usr/bin/env bash
# build_face_recognition.sh — Build PyInstaller binaries for CryptoFace submission.
#
# Usage: bash scripts/build_face_recognition.sh
#
# Builds 7 ELF binaries in submission/face/build/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/submission/face/src"
BUILD_DIR="$REPO_ROOT/submission/face/build"
WORK_DIR="$REPO_ROOT/submission/face/_pyinstaller_work"

ORION_REPO="https://github.com/vboddeti/orion"

echo "[build] CryptoFace submission — building PyInstaller binaries"
echo "[build] Repo root  : $REPO_ROOT"
echo "[build] Source dir : $SRC_DIR"
echo "[build] Output dir : $BUILD_DIR"
echo ""

mkdir -p "$BUILD_DIR"

# Install Orion from GitHub if not already importable
if ! python3 -c "import orion" 2>/dev/null; then
    echo "[build] Orion not found — installing from $ORION_REPO ..."
    pip install "git+${ORION_REPO}"
else
    echo "[build] Orion already installed."
fi
ORION_SO="$(python3 -c "import orion, os; print(os.path.dirname(orion.__file__))")"
echo "[build] Orion lib dir: $ORION_SO"
echo ""

STAGES=(
    client_key_generation
    server_preprocess_model
    client_preprocess_input
    client_encode_encrypt_input
    server_encrypted_compute
    client_decrypt_decode
    client_postprocess
)

for stage in "${STAGES[@]}"; do
    src="$SRC_DIR/${stage}.py"
    if [ ! -f "$src" ]; then
        echo "[build] ERROR: source not found: $src"
        exit 1
    fi
    echo "[build] Building $stage ..."
    pyinstaller \
        --onefile \
        --distpath "$BUILD_DIR" \
        --workpath "$WORK_DIR/$stage" \
        --specpath "$WORK_DIR/$stage" \
        --add-binary "$ORION_SO:orion_lib" \
        --hidden-import orion \
        --name "$stage" \
        "$src"
    echo "[build] Built: $BUILD_DIR/$stage"
    echo ""
done

echo "[build] All binaries built:"
ls -lh "$BUILD_DIR/"
echo ""
echo "[build] Done."
