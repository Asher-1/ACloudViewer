#!/usr/bin/env bash
# Merge sequential ALIKED ggml patches into functional modules:
#   0001-vulkan-aliked-core.patch   (infra + shaders + C API)
#   0002-vulkan-aliked-sddh.patch   (SDDH parity / device-lost fixes)
#   0003-vulkan-aliked-dkd.patch    (DKD NMS / score idle)
#
# Requires a clean ggml git tree at the ACloudViewer pin (v0.17.0).
# Usage:
#   merge_aliked_patches.sh [ggml_git_dir] [patches_dir]
set -euo pipefail

PATCHES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PATCHES_DIR}/../../.." && pwd)"
GGML_SRC="${1:-${LIGHTGLUE_GGML_GGML:-${REPO_ROOT}/../dl/LightGlue-GGML/third_party/ggml}}"
LEGACY_DIR="${2:-${PATCHES_DIR}/aliked}"

if [[ ! -d "${GGML_SRC}/.git" ]]; then
  echo "error: ggml git tree required at ${GGML_SRC}" >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

cp -a "${GGML_SRC}/." "${WORKDIR}/ggml"
cd "${WORKDIR}/ggml"
git reset --hard HEAD
git clean -fdx
BASE="$(git rev-parse HEAD)"

apply_range() {
  local from="$1" to="$2"
  local i
  for ((i = from; i <= to; ++i)); do
    local p
    p="$(printf "${LEGACY_DIR}/%04d-" "${i}")"*
    local file
    file=( ${p} )
    if [[ ! -f "${file[0]}" ]]; then
      echo "error: missing patch index ${i} in ${LEGACY_DIR}" >&2
      exit 1
    fi
    echo "  applying $(basename "${file[0]}")"
    patch -p1 -N -i "${file[0]}"
  done
}

OUT_DIR="${PATCHES_DIR}/aliked_merged"
mkdir -p "${OUT_DIR}"

echo "==> core (legacy 0001-0003)"
apply_range 1 3
git diff "${BASE}" > "${OUT_DIR}/0001-vulkan-aliked-core.patch"
CORE_HEAD="$(git rev-parse HEAD)"

echo "==> sddh (legacy 0004-0012)"
apply_range 4 12
git diff "${CORE_HEAD}" > "${OUT_DIR}/0002-vulkan-aliked-sddh.patch"
SDDH_HEAD="$(git rev-parse HEAD)"

echo "==> dkd (legacy 0013-0014)"
apply_range 13 14
git diff "${SDDH_HEAD}" > "${OUT_DIR}/0003-vulkan-aliked-dkd.patch"

echo "==> verify full diff matches legacy apply"
cd "${WORKDIR}"
cp -a "${GGML_SRC}/." ggml_verify
cd ggml_verify
git reset --hard HEAD
git clean -fdx
apply_range 1 14
FULL_LEGACY="$(git diff "${BASE}" | sha256sum | awk '{print $1}')"
cd "${WORKDIR}/ggml"
FULL_MERGED="$(git diff "${BASE}" | sha256sum | awk '{print $1}')"
if [[ "${FULL_LEGACY}" != "${FULL_MERGED}" ]]; then
  echo "error: merged patches diverge from legacy 0001-0014 apply" >&2
  exit 1
fi

cat > "${OUT_DIR}/manifest_snippet.yaml" <<EOF
# Replace manifest.yaml aliked entries with:
patches:
  - file: aliked/0001-vulkan-aliked-core.patch
    note: ALIKED Vulkan core (shaders, dense-copy, DCN, C API, proc address)
  - file: aliked/0002-vulkan-aliked-sddh.patch
    note: ALIKED Vulkan SDDH (single/chunk, readback, workspace, barriers)
  - file: aliked/0003-vulkan-aliked-dkd.patch
    note: ALIKED Vulkan DKD (NMS/topk, score-map queue idle)
EOF

echo "wrote ${OUT_DIR}/0001-vulkan-aliked-core.patch ($(wc -l < "${OUT_DIR}/0001-vulkan-aliked-core.patch") lines)"
echo "wrote ${OUT_DIR}/0002-vulkan-aliked-sddh.patch ($(wc -l < "${OUT_DIR}/0002-vulkan-aliked-sddh.patch") lines)"
echo "wrote ${OUT_DIR}/0003-vulkan-aliked-dkd.patch ($(wc -l < "${OUT_DIR}/0003-vulkan-aliked-dkd.patch") lines)"
echo "OK: merged tree matches legacy 0001-0014"
