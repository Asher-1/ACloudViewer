#!/usr/bin/env bash
# Download the FreeSplatter example scenes and build a .splat for each into web/,
# matching web/scenes.json (so the demo's scene selector is populated).
#   nix develop -c scripts/make_scenes.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GGUF=${FREE_SPLATTER_GGUF:-.cache/freesplatter-scene-f32.gguf}
DEVICE=${FREE_SPLATTER_DEVICE:-vulkan}
CLI=${FREE_SPLATTER_CLI:-build/vulkan/free_splatter-cli}
MAX_SPLATS=${MAX_SPLATS:-300000}
B=https://huggingface.co/spaces/TencentARC/FreeSplatter/resolve/main/examples/views_to_scene

# scene  out.splat  img1 img2
scenes=(
  "blendedmvs_1 blendedmvs_1 001.png 002.png"
  "re10k_2      re10k_2      000064.png 000157.png"
  "co3d_1       co3d_1       000.jpg 001.jpg"
  "dtu_1        dtu_1        000000.png 000002.png"
  "scannetpp_1  scannetpp_1  001.jpg 002.jpg"
  "re10k_1      re10k_1      000008.png 000071.png"
  "wild_1       wild_1       000.jpg 001.jpg"
  "scannetpp_2  scannetpp_2  000.jpg 001.jpg"
)
for e in "${scenes[@]}"; do
  set -- $e; scene=$1; out=$2; f1=$3; f2=$4
  mkdir -p ".cache/scenes/$scene"
  [ -f ".cache/scenes/$scene/$f1" ] || curl -sL -o ".cache/scenes/$scene/$f1" "$B/$scene/$f1"
  [ -f ".cache/scenes/$scene/$f2" ] || curl -sL -o ".cache/scenes/$scene/$f2" "$B/$scene/$f2"
  "$CLI" --device "$DEVICE" --max-splats "$MAX_SPLATS" --splat "web/$out.splat" \
         "$GGUF" ".cache/scenes/$scene/$f1" ".cache/scenes/$scene/$f2" | grep wrote
done
echo "done -> web/*.splat (serve web/ and open index.html)"
