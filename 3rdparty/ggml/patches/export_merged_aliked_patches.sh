#!/usr/bin/env bash
# Export the complete ALIKED Vulkan delta from a patched ggml v0.18.1 source
# tree, then prove that applying it to a clean tree recreates the same diff.
#
# Produces: aliked_merged/0001-vulkan-aliked.patch
# Usage: export_merged_aliked_patches.sh [patched_ggml_source_dir]
set -euo pipefail

PATCHES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PATCHES_DIR}/../../.." && pwd)"
REF_GGML="${1:-${GGML_PATCHED_SOURCE:-${REPO_ROOT}/build_app/ggml/src/ext_ggml}}"
BASE_ARCHIVE="${GGML_V0181_ARCHIVE:-${REPO_ROOT}/3rdparty_downloads/ggml/v0.18.1.tar.gz}"
OUT_DIR="${PATCHES_DIR}/aliked_merged"
OUT_PATCH="${OUT_DIR}/0001-vulkan-aliked.patch"

if [[ ! -d "${REF_GGML}" ]]; then
    echo "error: patched ggml source tree required at ${REF_GGML}" >&2
    exit 1
fi
if [[ ! -f "${BASE_ARCHIVE}" ]]; then
    echo "error: ggml v0.18.1 archive required at ${BASE_ARCHIVE}" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

mkdir -p "${OUT_DIR}" "${WORKDIR}/reference" "${WORKDIR}/verify"
tar -xzf "${BASE_ARCHIVE}" -C "${WORKDIR}/reference" --strip-components=1
tar -xzf "${BASE_ARCHIVE}" -C "${WORKDIR}/verify" --strip-components=1

git -C "${WORKDIR}/reference" init --quiet
git -C "${WORKDIR}/reference" config user.name ggml-patch-export
git -C "${WORKDIR}/reference" config user.email ggml-patch-export@localhost
git -C "${WORKDIR}/reference" add .
git -C "${WORKDIR}/reference" commit --quiet -m base

# The ExternalProject tree also contains independent CPU-variant and CUDA
# patches. Export only files owned by the ALIKED Vulkan feature.
ALIKED_PATHS=(
    CMakeLists.txt
    include/ggml-vulkan-aliked.h
    src/ggml-vulkan/ggml-vulkan-aliked.inc.cpp
    src/ggml-vulkan/ggml-vulkan.cpp
    src/ggml-vulkan/vulkan-shaders/aliked_block_topk.comp
    src/ggml-vulkan/vulkan-shaders/aliked_clamp.comp
    src/ggml-vulkan/vulkan-shaders/aliked_deform_conv.comp
    src/ggml-vulkan/vulkan-shaders/aliked_dense_copy.comp
    src/ggml-vulkan/vulkan-shaders/aliked_dkd_elem.comp
    src/ggml-vulkan/vulkan-shaders/aliked_dkd_refine.comp
    src/ggml-vulkan/vulkan-shaders/aliked_l2norm.comp
    src/ggml-vulkan/vulkan-shaders/aliked_layout_convert.comp
    src/ggml-vulkan/vulkan-shaders/aliked_max_pool.comp
    src/ggml-vulkan/vulkan-shaders/aliked_sddh.comp
    src/ggml-vulkan/vulkan-shaders/aliked_upsample_bilinear.comp
    src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
)
for path in "${ALIKED_PATHS[@]}"; do
    if [[ ! -f "${REF_GGML}/${path}" ]]; then
        echo "error: ALIKED source missing from patched tree: ${path}" >&2
        exit 1
    fi
    mkdir -p "${WORKDIR}/reference/$(dirname "${path}")"
    cp -a "${REF_GGML}/${path}" "${WORKDIR}/reference/${path}"
done

# Normalize whitespace only in ALIKED-owned new files. This keeps repeated
# exports stable without rewriting unrelated upstream source.
sed -i 's/[[:space:]]\+$//' \
    "${WORKDIR}/reference/include/ggml-vulkan-aliked.h" \
    "${WORKDIR}/reference/src/ggml-vulkan/ggml-vulkan-aliked.inc.cpp" \
    "${WORKDIR}/reference/src/ggml-vulkan/vulkan-shaders/aliked_"*.comp

git -C "${WORKDIR}/reference" add -A
git -C "${WORKDIR}/reference" diff --cached --binary HEAD > "${OUT_PATCH}"
# A blank added line is represented by a bare '+'. Some upstream files carry
# whitespace-only blank lines; normalize those in the patch without changing
# any source content.
sed -i 's/^+ $/+/' "${OUT_PATCH}"
if [[ ! -s "${OUT_PATCH}" ]]; then
    echo "error: reference tree has no ALIKED changes" >&2
    exit 1
fi

git -C "${WORKDIR}/verify" apply --whitespace=nowarn "${OUT_PATCH}"

if ! diff -qr --exclude='.git' "${WORKDIR}/reference" "${WORKDIR}/verify" >/dev/null; then
    echo "error: exported patch does not recreate the patched source tree" >&2
    exit 1
fi

echo "wrote ${OUT_PATCH} ($(wc -l < "${OUT_PATCH}") lines)"
echo "OK: clean v0.18.1 + merged patch matches selected ALIKED sources in ${REF_GGML}"
