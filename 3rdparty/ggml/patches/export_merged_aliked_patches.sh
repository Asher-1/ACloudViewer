#!/usr/bin/env bash
# Export ALIKED ggml patches merged by functional module from a reference tree
# (LightGlue-GGML/third_party/ggml with full ALIKED Vulkan changes).
#
# Legacy 0007+ incremental patches have malformed multi-hunk lines; this script
# uses git apply for 0001-0003 and copies the reference tree for the follow-up.
#
# Produces:
#   aliked_merged/0001-vulkan-aliked-core.patch
#   aliked_merged/0002-vulkan-aliked-gpu-followup.patch  (SDDH + DKD parity/speed)
#
# Usage: export_merged_aliked_patches.sh [reference_ggml_git_dir]
set -euo pipefail

PATCHES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PATCHES_DIR}/../../.." && pwd)"
REF_GGML="${1:-${LIGHTGLUE_GGML_GGML:-${REPO_ROOT}/../dl/LightGlue-GGML/third_party/ggml}}"
LEGACY_DIR="${PATCHES_DIR}/aliked"
OUT_DIR="${PATCHES_DIR}/aliked_merged"

if [[ ! -d "${REF_GGML}/.git" ]]; then
  echo "error: reference ggml git tree required at ${REF_GGML}" >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

cp -a "${REF_GGML}/." "${WORKDIR}/ref"
cp -a "${REF_GGML}/." "${WORKDIR}/build"

cd "${WORKDIR}/build"
git reset --hard HEAD
git clean -fdx
BASE="$(git rev-parse HEAD)"

apply_legacy() {
  local from="$1" to="$2"
  local i
  for ((i = from; i <= to; ++i)); do
    local p
    p="$(printf "${LEGACY_DIR}/%04d-" "${i}")"*
    local file=( ${p} )
    if [[ ! -f "${file[0]}" ]]; then
      echo "error: missing legacy patch index ${i}" >&2
      exit 1
    fi
    echo "  applying $(basename "${file[0]}")"
    git apply --whitespace=nowarn "${file[0]}"
  done
}

mkdir -p "${OUT_DIR}"

echo "==> core (legacy 0001-0003)"
apply_legacy 1 3
git add -A
git commit -m "aliked core (0001-0003)" --no-gpg-sign --quiet
git diff "${BASE}" HEAD > "${OUT_DIR}/0001-vulkan-aliked-core.patch"
CORE_HEAD="$(git rev-parse HEAD)"

echo "==> gpu followup (reference tree delta: SDDH + DKD)"
FOLLOWUP_FILES=(
  include/ggml-vulkan-aliked.h
  src/ggml-vulkan/ggml-vulkan-aliked.inc.cpp
  src/ggml-vulkan/ggml-vulkan.cpp
  src/ggml-vulkan/vulkan-shaders/aliked_block_topk.comp
  src/ggml-vulkan/vulkan-shaders/aliked_dkd_elem.comp
  src/ggml-vulkan/vulkan-shaders/aliked_dkd_refine.comp
  src/ggml-vulkan/vulkan-shaders/aliked_sddh.comp
  src/ggml-vulkan/vulkan-shaders/aliked_upsample_bilinear.comp
)
for f in "${FOLLOWUP_FILES[@]}"; do
  mkdir -p "$(dirname "${f}")"
  cp -a "${WORKDIR}/ref/${f}" "${f}"
done
git add -A
git diff "${CORE_HEAD}" > "${OUT_DIR}/0002-vulkan-aliked-gpu-followup.patch"

echo "==> verify merged apply matches reference"
cd "${WORKDIR}"
cp -a "${REF_GGML}/." verify
cd verify
git reset --hard HEAD
git clean -fdx
git apply --whitespace=nowarn "${OUT_DIR}/0001-vulkan-aliked-core.patch"
git apply --whitespace=nowarn "${OUT_DIR}/0002-vulkan-aliked-gpu-followup.patch"

# Reference tree may omit CMake install header (ACloudViewer adds it in core patch).
REF_CHANGED="$(cd "${WORKDIR}/ref" && git diff HEAD --name-only)"
VERIFY_OK=1
while IFS= read -r f; do
  [[ -z "${f}" ]] && continue
  if ! diff -q "${WORKDIR}/ref/${f}" "${f}" >/dev/null 2>&1; then
    echo "  mismatch: ${f}" >&2
    VERIFY_OK=0
  fi
done <<< "${REF_CHANGED}"
if [[ "${VERIFY_OK}" -ne 1 ]]; then
  echo "error: merged patches diverge from reference tree on changed files" >&2
  exit 1
fi

cat > "${OUT_DIR}/manifest_snippet.yaml" <<'EOF'
# Replace manifest.yaml aliked entries with:
patches:
  - file: aliked_merged/0001-vulkan-aliked-core.patch
    note: ALIKED Vulkan core (shaders, dense-copy, DCN, C API, proc address)
  - file: aliked_merged/0002-vulkan-aliked-gpu-followup.patch
    note: ALIKED Vulkan SDDH+DKD parity (queue idle, readback, NMS/topk, barriers)
EOF

echo "wrote ${OUT_DIR}/0001-vulkan-aliked-core.patch ($(wc -l < "${OUT_DIR}/0001-vulkan-aliked-core.patch") lines)"
echo "wrote ${OUT_DIR}/0002-vulkan-aliked-gpu-followup.patch ($(wc -l < "${OUT_DIR}/0002-vulkan-aliked-gpu-followup.patch") lines)"
echo "OK: merged tree matches reference"
