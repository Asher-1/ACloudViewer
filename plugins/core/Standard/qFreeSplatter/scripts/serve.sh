#!/usr/bin/env bash
# Build the Vulkan shared library if needed, then run the free-splatter web app
# (Go HTTP server -> purego -> the C API -> Vulkan). Drop photos in the browser
# to reconstruct a gaussian splat.
#
#   nix develop -c scripts/serve.sh           # http://localhost:8080
#   ADDR=:9000 FREE_SPLATTER_DEVICE=vulkan:1 nix develop -c scripts/serve.sh
#
# Env: FREE_SPLATTER_GGUF (model path), FREE_SPLATTER_DEVICE (cpu|vulkan|vulkan:N), ADDR.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

GGUF=${FREE_SPLATTER_GGUF:-$ROOT/.cache/freesplatter-scene-f16.gguf}
OBJ=${FREE_SPLATTER_OBJECT_GGUF:-$ROOT/.cache/freesplatter-object-f16.gguf}
DEVICE=${FREE_SPLATTER_DEVICE:-vulkan}
ADDR=${ADDR:-:8080}
LIB=$ROOT/build/vulkan/libfree_splatter.so

if [ ! -f "$LIB" ]; then
  echo "building $LIB (Vulkan shared lib) ..."
  cmake -S "$ROOT" -B "$ROOT/build/vulkan" -DFREE_SPLATTER_VULKAN=ON -DFREE_SPLATTER_BUILD_SHARED=ON
  cmake --build "$ROOT/build/vulkan" --target free_splatter -j
fi
if [ ! -f "$GGUF" ]; then
  echo "model not found: $GGUF" >&2
  echo "set FREE_SPLATTER_GGUF, or convert one with scripts/convert.py (see CLAUDE.md)." >&2
  exit 1
fi

# purego needs no cgo; CGO_ENABLED=0 also gives net/http the pure-Go resolver so
# the server builds without a C toolchain on PATH.
# scene is always loaded; the object model is added when its GGUF is present
# (convert one with: scripts/convert.py ckpt.safetensors out.gguf --variant object).
MODELS="scene=$GGUF"
[ -f "$OBJ" ] && MODELS="$MODELS,object=$OBJ"

ARGS=(-addr "$ADDR" -lib "$LIB" -models "$MODELS" -device "$DEVICE")

# always mount the demo-video dir; the storyboard page (/demo.html) lights up once
# scripts/demo/bake.sh has populated it — no server restart needed.
DEMO_DIR=${FREE_SPLATTER_DEMO_DIR:-$ROOT/.cache/demo}
mkdir -p "$DEMO_DIR"
ARGS+=(-demo-dir "$DEMO_DIR")

# auto-enable object background removal if the rembg venv exists
# (set up once with scripts/setup_bgremove.sh).
BGREMOVE_CMD=${FREE_SPLATTER_BGREMOVE_CMD:-}
if [ -z "$BGREMOVE_CMD" ] && [ -x "$ROOT/.venv-bgremove/bin/rembg" ]; then
  BGREMOVE_CMD="$ROOT/.venv-bgremove/bin/rembg p -m u2netp {in} {out}"
fi
[ -n "$BGREMOVE_CMD" ] && ARGS+=(-bgremove-cmd "$BGREMOVE_CMD")

cd "$ROOT/server"
exec env CGO_ENABLED=0 go run . "${ARGS[@]}"
