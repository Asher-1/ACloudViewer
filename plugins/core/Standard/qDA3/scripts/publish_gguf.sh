#!/usr/bin/env bash
# Publish the Depth Anything 3 GGUF weights to a Hugging Face repo so the LocalAI
# gallery can fetch them via huggingface://<repo>/<file>. The gallery itself lives
# in the LocalAI repo (backend/go/depth-anything-cpp/); the authoritative checksums
# are in models/SHA256SUMS here.
#
# This is an OUTWARD-FACING action: it uploads files to a public service under
# YOUR Hugging Face account. Run it yourself with your own credentials — it is
# intentionally not invoked by any build/test step.
#
# Prerequisites:
#   - huggingface_hub CLI:  pip install -U "huggingface_hub[cli]"
#   - authenticate:         hf auth login   (or set HF_TOKEN)
#
# Usage:
#   scripts/publish_gguf.sh [HF_REPO] [MODELS_DIR]
#
#   HF_REPO     target repo (default: mudler/depth-anything.cpp-gguf), must match
#               the `uri:` fields in the LocalAI gallery entry.
#   MODELS_DIR  directory holding the .gguf files (default: ./models).
#
# The script verifies every file against models/SHA256SUMS BEFORE uploading, so a
# corrupt/rebuilt artifact whose hash no longer matches the gallery is caught
# rather than silently published.
set -euo pipefail

# Xet transfer can hang on large GGUFs; the classic LFS multipart path is reliable.
export HF_HUB_DISABLE_XET=1

HF_REPO="${1:-mudler/depth-anything.cpp-gguf}"
MODELS_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)/models}"
SUMS="${MODELS_DIR}/SHA256SUMS"

command -v hf >/dev/null 2>&1 || { echo "error: 'hf' CLI not found — pip install -U 'huggingface_hub[cli]'"; exit 1; }
[ -f "$SUMS" ] || { echo "error: $SUMS not found"; exit 1; }

echo ">> Verifying checksums in $MODELS_DIR against SHA256SUMS ..."
( cd "$MODELS_DIR" && sha256sum -c SHA256SUMS )

# Files published to the GGUF repo. Edit this list to publish more/fewer.
FILES=(
  depth-anything-base-q4_k.gguf
  depth-anything-base-q8_0.gguf
  depth-anything-base-f16.gguf
  depth-anything-base-f32.gguf
  depth-anything-small-f32.gguf
  depth-anything-large-f32.gguf
  depth-anything-giant-f32.gguf
  depth-anything-mono-large-f32.gguf
  depth-anything-metric-large-f32.gguf
  depth-anything-nested-anyview.gguf
  depth-anything-nested-metric.gguf

  # Depth Anything V2 — relative (f32 + f16 + q8_0 + q6_k + q5_k + q4_k)
  depth-anything2-small-f32.gguf
  depth-anything2-small-f16.gguf
  depth-anything2-small-q8_0.gguf
  depth-anything2-small-q6_k.gguf
  depth-anything2-small-q5_k.gguf
  depth-anything2-small-q4_k.gguf
  depth-anything2-base-f32.gguf
  depth-anything2-base-f16.gguf
  depth-anything2-base-q8_0.gguf
  depth-anything2-base-q6_k.gguf
  depth-anything2-base-q5_k.gguf
  depth-anything2-base-q4_k.gguf
  depth-anything2-large-f32.gguf
  depth-anything2-large-f16.gguf
  depth-anything2-large-q8_0.gguf
  depth-anything2-large-q6_k.gguf
  depth-anything2-large-q5_k.gguf
  depth-anything2-large-q4_k.gguf

  # Depth Anything V2 — metric Hypersim (indoor, max_depth=20)
  depth-anything2-metric-hypersim-small-f32.gguf
  depth-anything2-metric-hypersim-small-f16.gguf
  depth-anything2-metric-hypersim-small-q8_0.gguf
  depth-anything2-metric-hypersim-small-q6_k.gguf
  depth-anything2-metric-hypersim-small-q5_k.gguf
  depth-anything2-metric-hypersim-small-q4_k.gguf
  depth-anything2-metric-hypersim-base-f32.gguf
  depth-anything2-metric-hypersim-base-f16.gguf
  depth-anything2-metric-hypersim-base-q8_0.gguf
  depth-anything2-metric-hypersim-base-q6_k.gguf
  depth-anything2-metric-hypersim-base-q5_k.gguf
  depth-anything2-metric-hypersim-base-q4_k.gguf
  depth-anything2-metric-hypersim-large-f32.gguf
  depth-anything2-metric-hypersim-large-f16.gguf
  depth-anything2-metric-hypersim-large-q8_0.gguf
  depth-anything2-metric-hypersim-large-q6_k.gguf
  depth-anything2-metric-hypersim-large-q5_k.gguf
  depth-anything2-metric-hypersim-large-q4_k.gguf

  # Depth Anything V2 — metric VKITTI (outdoor, max_depth=80)
  depth-anything2-metric-vkitti-small-f32.gguf
  depth-anything2-metric-vkitti-small-f16.gguf
  depth-anything2-metric-vkitti-small-q8_0.gguf
  depth-anything2-metric-vkitti-small-q6_k.gguf
  depth-anything2-metric-vkitti-small-q5_k.gguf
  depth-anything2-metric-vkitti-small-q4_k.gguf
  depth-anything2-metric-vkitti-base-f32.gguf
  depth-anything2-metric-vkitti-base-f16.gguf
  depth-anything2-metric-vkitti-base-q8_0.gguf
  depth-anything2-metric-vkitti-base-q6_k.gguf
  depth-anything2-metric-vkitti-base-q5_k.gguf
  depth-anything2-metric-vkitti-base-q4_k.gguf
  depth-anything2-metric-vkitti-large-f32.gguf
  depth-anything2-metric-vkitti-large-f16.gguf
  depth-anything2-metric-vkitti-large-q8_0.gguf
  depth-anything2-metric-vkitti-large-q6_k.gguf
  depth-anything2-metric-vkitti-large-q5_k.gguf
  depth-anything2-metric-vkitti-large-q4_k.gguf
)

echo ">> Ensuring repo $HF_REPO exists ..."
hf repo create "$HF_REPO" --repo-type model -y >/dev/null 2>&1 || true

# Upload the model card (MODEL_CARD.md -> README.md on the repo) if present.
CARD="${MODELS_DIR}/MODEL_CARD.md"
if [ -f "$CARD" ]; then
  echo ">> Uploading model card -> $HF_REPO/README.md ..."
  hf upload "$HF_REPO" "$CARD" README.md --repo-type model
fi

for f in "${FILES[@]}"; do
  path="${MODELS_DIR}/${f}"
  [ -f "$path" ] || { echo "error: missing $path"; exit 1; }
  echo ">> Uploading $f -> $HF_REPO ..."
  hf upload "$HF_REPO" "$path" "$f" --repo-type model
done

echo ">> Done. Verify the LocalAI gallery sha256 values match models/SHA256SUMS."
echo ">> Users can then: local-ai run depth-anything-3-base"
