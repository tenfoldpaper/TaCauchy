#!/usr/bin/env bash
# Apply TaCauchy's local libuipc patches to the pinned submodule checkout.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBUIPC_DIR="${ROOT_DIR}/source/tacex_uipc/libuipc"
MUDA_DIR="${LIBUIPC_DIR}/external/muda"

apply_patch_file() {
    local repo_dir="$1"
    local patch_file="$2"
    local label="$3"

    if git -C "${repo_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        echo "[TaCauchy] ${label} already applied."
        return 0
    fi

    if git -C "${repo_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
        git -C "${repo_dir}" apply "${patch_file}"
        echo "[TaCauchy] Applied ${label}."
        return 0
    fi

    echo "[ERROR] Cannot apply ${label}. The submodule may be on the wrong commit or already partially modified." >&2
    echo "        Try: git submodule update --init --recursive --force" >&2
    exit 1
}

if [[ ! -d "${LIBUIPC_DIR}/.git" && ! -f "${LIBUIPC_DIR}/.git" ]]; then
    echo "[TaCauchy] Initializing submodules..."
    git -C "${ROOT_DIR}" submodule update --init --recursive
fi

if [[ ! -d "${MUDA_DIR}/.git" && ! -f "${MUDA_DIR}/.git" ]]; then
    echo "[TaCauchy] Initializing nested libuipc submodules..."
    git -C "${LIBUIPC_DIR}" submodule update --init --recursive
fi

apply_patch_file "${LIBUIPC_DIR}" "${ROOT_DIR}/patches/libuipc-tacauchy.patch" "libuipc patch"
apply_patch_file "${MUDA_DIR}" "${ROOT_DIR}/patches/muda-tacauchy.patch" "muda patch"

echo "[TaCauchy] libuipc patches are ready. You can now build with: pip install -e source/tacex_uipc -v"
