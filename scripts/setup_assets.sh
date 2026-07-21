#!/usr/bin/env bash
# Pull and verify TaCauchy runtime assets tracked by Git LFS.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${ROOT_DIR}/source/tacex_assets/tacex_assets/data"

required_assets=(
    "Sensors/GelSight_Mini/Sensor.usd"
    "Sensors/GelSight_Mini/Case.usd"
    "Sensors/GelSight_Mini/calibs/640x480/dataPack.npz"
    "Sensors/digit/Sensor.usd"
    "Sensors/digit/Case.usd"
    "Sensors/digit/calibs/640x480/dataPack.npz"
    "Sensors/9dtact/Sensor.usd"
    "Sensors/9dtact/case_9DTact.usd"
    "Sensors/9dtact/gelpad_9DTact.usd"
    "Sensors/9dtact/calibs/640x480/dataPack.npz"
    "Robots/Franka/GelSight_Mini/Single_Adapter/uipc_gelpad.usd"
    "Robots/Franka/digit/Single_Adapter/digit_uipc_gelpad.usd"
    "Robots/Franka/9dtact/Single_Adapter/9dtact_uipc_gelpad.usd"
    "Props/plate.usd"
    "Props/ball_wood.usd"
    "Props/tactile_test_shapes/sphere2.usd"
)

if ! git lfs version >/dev/null 2>&1; then
    echo "[ERROR] git-lfs is required. Install it first, then run: git lfs install"
    exit 1
fi

echo "[TaCauchy] Ensuring Git LFS is enabled locally..."
git -C "${ROOT_DIR}" lfs install --local >/dev/null 2>&1 || git lfs install --local >/dev/null 2>&1

if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[TaCauchy] Pulling LFS assets under source/tacex_assets/tacex_assets/data/..."
    git -C "${ROOT_DIR}" lfs pull --include="source/tacex_assets/tacex_assets/data/**"
fi

missing=()
pointer_files=()

for rel_path in "${required_assets[@]}"; do
    file_path="${ASSET_DIR}/${rel_path}"
    if [[ ! -f "${file_path}" ]]; then
        missing+=("${rel_path}")
        continue
    fi
    if head -n 1 "${file_path}" 2>/dev/null | grep -q "^version https://git-lfs.github.com/spec/v1"; then
        pointer_files+=("${rel_path}")
    fi
done

if (( ${#missing[@]} > 0 )); then
    echo "[ERROR] Missing required assets:"
    printf '  - %s\n' "${missing[@]}"
    exit 1
fi

if (( ${#pointer_files[@]} > 0 )); then
    echo "[ERROR] These assets are still Git LFS pointer files; run 'git lfs pull' again:"
    printf '  - %s\n' "${pointer_files[@]}"
    exit 1
fi

echo "[TaCauchy] Asset check passed: ${ASSET_DIR}"
echo "  Sensors : ${ASSET_DIR}/Sensors"
echo "  Robots  : ${ASSET_DIR}/Robots"
echo "  Props   : ${ASSET_DIR}/Props"
