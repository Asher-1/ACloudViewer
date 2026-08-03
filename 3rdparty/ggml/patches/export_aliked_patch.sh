#!/usr/bin/env bash
# Regenerate aliked/0001-vulkan-aliked-custom-compute.patch from a ggml git tree
# with ALIKED Vulkan changes applied (typically LightGlue-GGML/third_party/ggml).
set -euo pipefail

PATCHES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PATCHES_DIR}/../../.." && pwd)"
PATCH="${PATCHES_DIR}/aliked/0001-vulkan-aliked-custom-compute.patch"
GGML_SRC="${1:-${LIGHTGLUE_GGML_GGML:-${REPO_ROOT}/third_party/ggml}}"

if [[ ! -d "${GGML_SRC}/.git" ]]; then
  echo "error: ggml git tree not found at ${GGML_SRC}" >&2
  exit 1
fi

cd "${GGML_SRC}"
git add -N include/ggml-vulkan-aliked.h \
  src/ggml-vulkan/ggml-vulkan-aliked.inc.cpp \
  src/ggml-vulkan/vulkan-shaders/aliked_*.comp 2>/dev/null || true
git diff HEAD -- . > "${PATCH}"
echo "wrote ${PATCH} ($(wc -l < "${PATCH}") lines)"

if command -v python3 >/dev/null; then
  python3 "${PATCHES_DIR}/apply_ggml_patches.py" --help >/dev/null 2>&1 || true
fi
