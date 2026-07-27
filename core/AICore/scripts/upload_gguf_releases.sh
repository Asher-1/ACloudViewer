#!/usr/bin/env bash
# Upload ELoFTR / DeepLSD GGUF variants to cloudViewer_downloads releases.
# Requires: gh auth login OR GITHUB_TOKEN with repo scope.
set -euo pipefail

REPO="${CLOUDVIEWER_DOWNLOADS_REPO:-Asher-1/cloudViewer_downloads}"
RELEASE_DIR="$(cd "$(dirname "$0")/../models/release" && pwd)"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install gh or upload manually:" >&2
  echo "  ELoFTR tag: https://github.com/$REPO/releases/tag/ELoFTR" >&2
  echo "  DeepLSD tag: https://github.com/$REPO/releases/tag/DeepLSD" >&2
  ls -lh "$RELEASE_DIR/ELoFTR" "$RELEASE_DIR/DeepLSD"
  exit 1
fi

upload_tag() {
  local tag="$1"
  local dir="$2"
  local note="$3"
  if ! gh release view "$tag" -R "$REPO" >/dev/null 2>&1; then
    gh release create "$tag" -R "$REPO" --title "$tag" --notes "$note"
  fi
  gh release upload "$tag" "$dir"/*.gguf -R "$REPO" --clobber
  echo "Uploaded $tag assets from $dir"
}

upload_tag ELoFTR "$RELEASE_DIR/ELoFTR" \
  "EfficientLoFTR outdoor RepVGG GGUF: eloftr_outdoor F32/F16/Q8_0. Verified CPU/CUDA/Vulkan — see EfficientLoFTR cpp/BENCHMARK.md. (No public indoor checkpoint.)"

upload_tag DeepLSD "$RELEASE_DIR/DeepLSD" \
  "DeepLSD GGUF: wireframe + MegaDepth (md), each F32/F16/Q8_0. Verified CPU/CUDA/Vulkan (F32/F16) — see DeepLSD cpp/BENCHMARK.md."
