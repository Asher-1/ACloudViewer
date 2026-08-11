#!/usr/bin/env bash
# One-command demo video: ensure the server is up with baked assets, then render
# the storyboard page to an MP4. Two render paths:
#
#   * GPU (recommended, fast): open the demo page in a normal browser on the
#     machine with the GPU and press "Render". This uses the browser's WebGL/GPU
#     directly and is far faster than software rendering.
#
#   * Headless (this script, offline/CI): drives chromium with the SwiftShader
#     software rasteriser, streams frames to the server, and polls for the
#     encoded MP4. Correct but CPU-bound — keep RES/FRAMES modest.
#
#   RES=1280x720 FRAMES=120 scripts/demo/make.sh        # quick offline smoke
#   RES=1920x1080 scripts/demo/make.sh                  # full (slow on CPU)
#
# Assumes a server is already running on PORT with -demo-dir set (scripts/serve.sh
# wires that up). Env: PORT (4001), RES (1280x720), FRAMES (all), SESSION.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

PORT=${PORT:-4001}
RES=${RES:-1280x720}
W=${RES%x*}; H=${RES#*x}
FRAMES=${FRAMES:-}
SESSION=${SESSION:-demo_${RES}}
SERVER=http://127.0.0.1:$PORT

curl -fsS "$SERVER/api/models" >/dev/null 2>&1 || {
  echo "no server on $SERVER — start one with -demo-dir, e.g.:" >&2
  echo "  ADDR=:$PORT scripts/serve.sh   (after editing it to pass -demo-dir .cache/demo)" >&2
  exit 1
}
if ! curl -fsS "$SERVER/demo-assets/manifest.json" >/dev/null 2>&1; then
  echo "no baked assets — running bake first ..."
  SERVER=$SERVER scripts/demo/bake.sh
fi

command -v bun >/dev/null || { echo "bun required for headless render (scripts/demo/render.js)" >&2; exit 1; }
CHROME=$(command -v chromium || command -v chromium-browser || command -v google-chrome-stable)
[ -n "$CHROME" ] || { echo "no chromium/chrome found" >&2; exit 1; }

# render.js drives chromium over CDP (a plain `--headless URL` never runs the page;
# `--screenshot` exits after one frame — neither can drive a capture loop). It
# renders + POSTs frames to the server, which encodes with ffmpeg. GPU=1 uses the
# real GPU (fast); default is SwiftShader (slow — keep RES/FRAMES modest).
echo "rendering ${RES} (session $SESSION) via CDP driver ${GPU:+(GPU)} ..."
PORT="$PORT" RES="$RES" SESSION="$SESSION" FRAMES="$FRAMES" GPU="${GPU:-}" \
  CHROME="$CHROME" bun "$ROOT/scripts/demo/render.js"

# render.js prints VIDEO_READY on success; copy the MP4 out for convenience.
out="$ROOT/.cache/demo/${SESSION}.mp4"
if curl -fsS -o "$out" "$SERVER/api/demo/video/$SESSION" 2>/dev/null; then
  echo "done -> $out ($(du -h "$out" | cut -f1))"
else
  echo "no video produced (see render.js output)" >&2; exit 1
fi
