#!/usr/bin/env bash
# One-time setup for the object path's automatic background removal: a Python
# venv with rembg (U2Net salient-object matting via onnxruntime). The web server
# calls it as a subprocess — it is NOT a dependency of the engine or the Go
# server binary itself, just an optional matting toolchain for the demo.
#
#   nix develop -c scripts/setup_bgremove.sh
#
# After this, scripts/serve.sh auto-enables the "remove background" toggle for
# the Object model. The u2netp model (~4MB) downloads to ~/.u2net on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=.venv-bgremove

python -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "rembg[cli]" onnxruntime pillow
echo "installed rembg in $VENV — serve.sh will pick it up automatically."
