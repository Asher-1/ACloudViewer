#!/usr/bin/env bash
# End-to-end demo asset: scene images -> .splat for the WebGL viewer, entirely
# via the C++ CLI (decode + preprocess + reconstruct + export; no Python).
#   nix develop -c scripts/make_demo.sh [out.splat] img1 img2 [img3 ...]
# Env: FREE_SPLATTER_GGUF, FREE_SPLATTER_CLI, FREE_SPLATTER_DEVICE, MAX_SPLATS.
set -euo pipefail
cd "$(dirname "$0")/.."

GGUF=${FREE_SPLATTER_GGUF:-.cache/freesplatter-scene-f32.gguf}
DEVICE=${FREE_SPLATTER_DEVICE:-vulkan}
CLI=${FREE_SPLATTER_CLI:-build/vulkan/free_splatter-cli}
MAX_SPLATS=${MAX_SPLATS:-300000}

out="${1:-web/scene.splat}"; shift || true
[ "$#" -ge 2 ] || { echo "usage: make_demo.sh [out.splat] img1 img2 [...]" >&2; exit 2; }

"$CLI" --device "$DEVICE" --max-splats "$MAX_SPLATS" --splat "$out" "$GGUF" "$@"
echo "demo asset -> $out  (serve web/ and open index.html)"
