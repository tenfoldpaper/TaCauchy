#!/usr/bin/env bash
# Download TacEx sensor/robot USD assets into tacex_assets/data via Git LFS.
set -euo pipefail

TACEX_ASSETS_REPO="${TACEX_ASSETS_REPO:-https://github.com/DH-Ng/TacEx.git}"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/source/tacex_assets/tacex_assets/data"
CACHE_DIR="${TMPDIR:-/tmp}/tacex_assets_cache_$$"

echo "[TaCauchy] Target asset directory: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

if command -v git-lfs >/dev/null 2>&1; then
    git lfs install --local >/dev/null 2>&1 || true
elif git lfs version >/dev/null 2>&1; then
    git lfs install --local >/dev/null 2>&1 || true
else
    echo "[ERROR] git-lfs is required. Install with: sudo apt install git-lfs && git lfs install"
    exit 1
fi

echo "[TaCauchy] Cloning TacEx assets (sparse checkout) ..."
rm -rf "${CACHE_DIR}"
git clone --depth 1 --filter=blob:none --sparse "${TACEX_ASSETS_REPO}" "${CACHE_DIR}"
cd "${CACHE_DIR}"
git sparse-checkout set source/tacex_assets/tacex_assets/data
git lfs pull

SRC="${CACHE_DIR}/source/tacex_assets/tacex_assets/data"
if [[ ! -d "${SRC}" ]]; then
    echo "[ERROR] Asset path not found in TacEx repo: ${SRC}"
    exit 1
fi

echo "[TaCauchy] Copying assets into ${TARGET_DIR} ..."
rsync -a --info=progress2 "${SRC}/" "${TARGET_DIR}/"
rm -rf "${CACHE_DIR}"

echo "[TaCauchy] Asset setup complete."
echo "  Sensors : ${TARGET_DIR}/Sensors"
echo "  Robots  : ${TARGET_DIR}/Robots"
echo "  Props   : ${TARGET_DIR}/Props"
