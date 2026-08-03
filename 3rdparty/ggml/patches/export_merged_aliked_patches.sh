#!/usr/bin/env bash
# Export the complete ALIKED Vulkan delta from a dirty ggml v0.17.0 reference
# tree, then prove that applying it to a clean tree recreates the same diff.
#
# Produces: aliked_merged/0001-vulkan-aliked.patch
# Usage: export_merged_aliked_patches.sh [reference_ggml_git_dir]
set -euo pipefail

PATCHES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PATCHES_DIR}/../../.." && pwd)"
REF_GGML="${1:-${LIGHTGLUE_GGML_GGML:-${REPO_ROOT}/../dl/LightGlue-GGML/third_party/ggml}}"
OUT_DIR="${PATCHES_DIR}/aliked_merged"
OUT_PATCH="${OUT_DIR}/0001-vulkan-aliked.patch"

if [[ ! -d "${REF_GGML}/.git" ]]; then
    echo "error: reference ggml git tree required at ${REF_GGML}" >&2
    exit 1
fi
if [[ "$(git -C "${REF_GGML}" describe --tags --exact-match HEAD 2>/dev/null || true)" != "v0.17.0" ]]; then
    echo "error: reference ggml HEAD must be tag v0.17.0" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

mkdir -p "${OUT_DIR}"
cp -a "${REF_GGML}/." "${WORKDIR}/reference"
git -C "${WORKDIR}/reference" add -N .
git -C "${WORKDIR}/reference" diff --binary HEAD > "${OUT_PATCH}"
if [[ ! -s "${OUT_PATCH}" ]]; then
    echo "error: reference tree has no ALIKED changes" >&2
    exit 1
fi

git clone --shared "${REF_GGML}" "${WORKDIR}/verify" --quiet
git -C "${WORKDIR}/verify" reset --hard HEAD --quiet
git -C "${WORKDIR}/verify" clean -fdx --quiet
git -C "${WORKDIR}/verify" apply --whitespace=nowarn "${OUT_PATCH}"
git -C "${WORKDIR}/verify" add -N .

REF_DIFF="$(git -C "${WORKDIR}/reference" diff --binary HEAD | sha256sum | awk '{print $1}')"
VERIFY_DIFF="$(git -C "${WORKDIR}/verify" diff --binary HEAD | sha256sum | awk '{print $1}')"
if [[ "${REF_DIFF}" != "${VERIFY_DIFF}" ]]; then
    echo "error: exported patch does not recreate the reference tree" >&2
    exit 1
fi

echo "wrote ${OUT_PATCH} ($(wc -l < "${OUT_PATCH}") lines)"
echo "OK: clean v0.17.0 + merged patch matches reference (${REF_DIFF})"
