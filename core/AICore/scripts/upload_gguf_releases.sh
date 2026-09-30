#!/usr/bin/env bash
# Upload DeepLSD GGUF variants to cloudViewer_downloads releases.
# Requires: gh auth login OR GITHUB_TOKEN with repo scope.
set -euo pipefail

REPO="${CLOUDVIEWER_DOWNLOADS_REPO:-Asher-1/cloudViewer_downloads}"
RELEASE_DIR="$(cd "$(dirname "$0")/../models/release" && pwd)"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install gh or upload manually:" >&2
  echo "  DeepLSD tag: https://github.com/$REPO/releases/tag/DeepLSD" >&2
  ls -lh "$RELEASE_DIR/DeepLSD"
  exit 1
fi

upload_tag() {
  local tag="$1"
  local dir="$2"
  local note="$3"
  if [ ! -d "$dir" ]; then
    echo "skip $tag: $dir has no local release assets"
    return 0
  fi
  if ! gh release view "$tag" -R "$REPO" >/dev/null 2>&1; then
    gh release create "$tag" -R "$REPO" --title "$tag" --notes "$note"
  fi
  gh release upload "$tag" "$dir"/*.gguf -R "$REPO" --clobber
  echo "Uploaded $tag assets from $dir"
}

upload_tag DeepLSD "$RELEASE_DIR/DeepLSD" \
  "DeepLSD GGUF: wireframe + MegaDepth (md), each F32/F16/Q8_0. Verified CPU/CUDA/Vulkan (F32/F16) — see DeepLSD cpp/BENCHMARK.md."

# qAnyLabeling OCR/RAM data plane: first-party Paddle->GGUF (PP-OCR det/rec)
# and Recognize-Anything (Swin-L tagger) containers, three quant tiers each.
upload_tag PP-OCR "$RELEASE_DIR/PP-OCR" \
  "PP-OCRv4 GGUF (first-party Paddle inference -> ggml conversion): Chinese det (DB/LCNetV3) + Chinese/English rec (SVTR). Each F32/F16/Q8_0; rec files embed their recognition dictionary as ocr.dictionary. Typed results: aicore_ocr_box quadrilaterals with UTF-8 text + score."
upload_tag RAM "$RELEASE_DIR/RAM" \
  "Recognize Anything Model GGUF (Swin-L image encoder + tag attention decoder, first-party torch -> ggml conversion). Each F32/F16/Q8_0; the 4574-tag vocabulary is embedded as ram.tag_list. Image-level tagging consumed through the AICore classify ABI — never object boxes."
