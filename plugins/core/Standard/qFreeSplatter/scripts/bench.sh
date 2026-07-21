#!/usr/bin/env bash
# Benchmark free-splatter.cpp against the upstream FreeSplatter PyTorch model.
#
# Times one forward pass (N views -> the [N,512,512,23] transformer logits) and
# compares the ggml engine (CPU with AVX/AVX512, and Vulkan GPU) against PyTorch
# (CPU, and CUDA GPU), at matching precisions, then prints a table.
#
#   nix develop -c scripts/bench.sh                 # default: 2 views
#   VIEWS=4 ITERS=10 nix develop -c scripts/bench.sh
#
# Notes:
#  - CPU uses the `release-portable` build (every ggml CPU ISA variant, best
#    picked at runtime). The plain `release` preset is SCALAR under Nix
#    (NIX_ENFORCE_NO_NATIVE strips -march=native) and ~13x slower -- do not use
#    it for CPU timing.
#  - PyTorch-on-CUDA needs a cu1xx torch build (.venv-torch-cu128) and the host
#    NVIDIA driver libs (/run/opengl-driver/lib) on LD_LIBRARY_PATH; both are set
#    up below. The CPU-only .venv-torch cannot use the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

VIEWS=${VIEWS:-2}
ITERS=${ITERS:-8}
CPU_ITERS=${CPU_ITERS:-3}
GGUF_F16=${GGUF_F16:-$ROOT/.cache/freesplatter-scene-f16.gguf}
GGUF_F32=${GGUF_F32:-$ROOT/.cache/freesplatter-scene-f32.gguf}
CKPT=${CKPT:-$ROOT/.cache/freesplatter-scene.safetensors}
TORCH=${TORCH_PY:-$ROOT/.venv-torch-cu128/bin/python}

# pip-CUDA torch finds libcuda.so.1 from the host driver here (NixOS).
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"

CPU_BIN=$ROOT/build/release-portable/bin/free_splatter-bench
VK_BIN=$ROOT/build/vulkan/free_splatter-bench

echo "building bench binaries (if needed) ..."
[ -x "$CPU_BIN" ] || { cmake --preset release-portable >/dev/null && \
    cmake --build --preset release-portable --target free_splatter-bench -j >/dev/null; }
[ -x "$VK_BIN" ]  || { cmake --preset vulkan >/dev/null && \
    cmake --build build/vulkan --target free_splatter-bench -j >/dev/null; }

RESULTS=$(mktemp)
trap 'rm -f "$RESULTS"' EXIT

run() {  # label, command...
    local label=$1; shift
    printf '>>> %s\n' "$label" >&2
    local line
    line=$("$@" 2>/dev/null | grep '^RESULT' || true)
    if [ -n "$line" ]; then echo "$label|$line" >> "$RESULTS"
    else printf '    (no result -- skipped or failed)\n' >&2; fi
}

# --- engine (ggml) ---
[ -f "$GGUF_F32" ] && run "engine | cpu    | f32 " "$CPU_BIN" --device cpu    --views "$VIEWS" --iters "$CPU_ITERS" --warmup 1 "$GGUF_F32"
[ -f "$GGUF_F16" ] && run "engine | cpu    | f16 " "$CPU_BIN" --device cpu    --views "$VIEWS" --iters "$CPU_ITERS" --warmup 1 "$GGUF_F16"
[ -f "$GGUF_F16" ] && run "engine | vulkan | f16 " "$VK_BIN"  --device vulkan --views "$VIEWS" --iters "$ITERS"     --warmup 2 "$GGUF_F16"
[ -f "$GGUF_F32" ] && run "engine | vulkan | f32 " "$VK_BIN"  --device vulkan --views "$VIEWS" --iters "$ITERS"     --warmup 2 "$GGUF_F32"

# --- PyTorch reference ---
if [ -x "$TORCH" ] && [ -f "$CKPT" ]; then
    run "torch  | cpu    | fp32" "$TORCH" scripts/bench_torch.py --ckpt "$CKPT" --device cpu  --dtype fp32 --views "$VIEWS" --iters "$CPU_ITERS" --warmup 1
    if "$TORCH" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        run "torch  | cuda   | fp16" "$TORCH" scripts/bench_torch.py --ckpt "$CKPT" --device cuda --dtype fp16 --views "$VIEWS" --iters "$ITERS" --warmup 3
        run "torch  | cuda   | fp32" "$TORCH" scripts/bench_torch.py --ckpt "$CKPT" --device cuda --dtype fp32 --views "$VIEWS" --iters "$ITERS" --warmup 3
        run "torch  | cuda   | bf16" "$TORCH" scripts/bench_torch.py --ckpt "$CKPT" --device cuda --dtype bf16 --views "$VIEWS" --iters "$ITERS" --warmup 3
    else
        echo ">>> torch CUDA unavailable (need .venv-torch-cu128 + /run/opengl-driver/lib)" >&2
    fi
else
    echo ">>> torch skipped (no $TORCH or $CKPT)" >&2
fi

# --- table ---
echo
printf '%-26s | %12s | %10s\n' "engine | device | dtype" "median (ms)" "scenes/s"
printf '%-26s-+-%12s-+-%10s\n' "$(printf '%.0s-' {1..26})" "------------" "----------"
sort "$RESULTS" | while IFS='|' read -r who dev dt rest; do
    med=$(echo "$rest" | grep -o 'median_ms=[0-9.]*' | cut -d= -f2)
    vps=$(echo "$rest" | grep -o 'views_per_s=[0-9.]*' | cut -d= -f2)
    sps=$(awk -v v="$vps" -v n="$VIEWS" 'BEGIN{ if (v>0) printf "%.3f", v/n; else print "-" }')
    printf '%-7s|%-8s|%-5s | %12s | %10s\n' "$who" "$dev" "$dt" "$med" "$sps"
done
echo
echo "(N=$VIEWS views, S=$((VIEWS*4096)) tokens; median of timed iters; lower ms / higher scenes/s is better)"
