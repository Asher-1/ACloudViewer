#!/usr/bin/env bash
# validate_gpu.sh — GPU-offload parity + speed validation for depth-anything.cpp.
#
# Runs on the CUDA dev/validation box (target HW: NVIDIA GB10 / Blackwell, ARM64,
# CUDA 13). It builds BOTH a CPU-only and a CUDA library + CLI, runs `da3-cli
# depth` on the same image on CPU and on the GPU, then compares the two depth
# maps (parity) and benchmarks both (speedup). It prints a clear PASS/FAIL.
#
# The C++/ggml code calls only ggml backend APIs (no direct CUDA), so the SAME
# sources compile for both builds; the only difference is -DDA_GGML_CUDA=ON.
#
# ----------------------------------------------------------------------------
# Required environment:
#   DA_GGUF   path to the model GGUF (e.g. models/depth-anything-giant-f32.gguf)
#   DA_IMAGE  path to an input image  (e.g. dumps/native_input.png)
# Optional:
#   DA_CUDA_DEV  GPU device name to pass as DA_DEVICE for the GPU run. If unset,
#                the GPU run leaves DA_DEVICE unset so the backend auto-picks the
#                first GPU/accelerator device (CUDA0 on a single-GPU box). The
#                device names ggml registers are logged at startup as
#                "da::Backend using device: <name>" — read one off a GPU run and
#                set DA_CUDA_DEV to pin a specific GPU.
#   DA_REPEAT    bench iterations (default 10).
#   DA_THREADS   CPU threads for the CPU run (default: nproc).
#   DA_TOL       max-abs parity tolerance (default 1e-2).
#   DA_CORR      min correlation (default 0.999).
#   PYTHON       python interpreter (default: python3, falls back to .venv).
# ----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${DA_GGUF:?set DA_GGUF to the model gguf path}"
: "${DA_IMAGE:?set DA_IMAGE to the input image path}"
REPEAT="${DA_REPEAT:-10}"
THREADS="${DA_THREADS:-$(nproc 2>/dev/null || echo 4)}"
TOL="${DA_TOL:-1e-2}"
CORR="${DA_CORR:-0.999}"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON="$ROOT/.venv/bin/python"

OUT_DIR="$ROOT/build-gpu-validate"
mkdir -p "$OUT_DIR"
CPU_PFM="$OUT_DIR/depth_cpu.pfm"
GPU_PFM="$OUT_DIR/depth_gpu.pfm"

echo "==> [1/4] Configure + build CPU-only (build-cpu)"
cmake -B build-cpu -DDA_BUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-cpu -j --target da3-cli >/dev/null
CPU_CLI="build-cpu/examples/cli/da3-cli"

echo "==> [2/4] Configure + build CUDA (build-cuda, -DDA_GGML_CUDA=ON)"
cmake -B build-cuda -DDA_BUILD_CLI=ON -DDA_GGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-cuda -j --target da3-cli >/dev/null
GPU_CLI="build-cuda/examples/cli/da3-cli"

echo "==> [3/4] Run depth: CPU (DA_DEVICE=cpu) and GPU"
echo "    CPU run ..."
DA_DEVICE=cpu "$CPU_CLI" depth --model "$DA_GGUF" --input "$DA_IMAGE" \
    --pfm "$CPU_PFM" --threads "$THREADS" >/dev/null

echo "    GPU run ..."
if [ -n "${DA_CUDA_DEV:-}" ]; then
    DA_DEVICE="$DA_CUDA_DEV" "$GPU_CLI" depth --model "$DA_GGUF" --input "$DA_IMAGE" \
        --pfm "$GPU_PFM" >/dev/null
else
    # DA_DEVICE unset -> backend auto-picks the first GPU device.
    env -u DA_DEVICE "$GPU_CLI" depth --model "$DA_GGUF" --input "$DA_IMAGE" \
        --pfm "$GPU_PFM" >/dev/null
fi

echo "==> [4/4] Benchmark (--repeat $REPEAT) + compare"
echo "    CPU bench:"
CPU_BENCH=$(DA_DEVICE=cpu "$CPU_CLI" depth --model "$DA_GGUF" --input "$DA_IMAGE" \
            --threads "$THREADS" --repeat "$REPEAT" 2>&1 | grep -E '^bench:' || true)
echo "      $CPU_BENCH"
echo "    GPU bench:"
if [ -n "${DA_CUDA_DEV:-}" ]; then
    GPU_BENCH=$(DA_DEVICE="$DA_CUDA_DEV" "$GPU_CLI" depth --model "$DA_GGUF" \
                --input "$DA_IMAGE" --repeat "$REPEAT" 2>&1 | grep -E '^bench:' || true)
else
    GPU_BENCH=$(env -u DA_DEVICE "$GPU_CLI" depth --model "$DA_GGUF" \
                --input "$DA_IMAGE" --repeat "$REPEAT" 2>&1 | grep -E '^bench:' || true)
fi
echo "      $GPU_BENCH"

# Extract "infer=<ms>ms/iter" from each bench line for a speedup number.
cpu_ms=$(printf '%s\n' "$CPU_BENCH" | sed -nE 's/.*infer=([0-9.]+)ms.*/\1/p')
gpu_ms=$(printf '%s\n' "$GPU_BENCH" | sed -nE 's/.*infer=([0-9.]+)ms.*/\1/p')

echo
echo "==> Parity + speed report"
"$PYTHON" - "$CPU_PFM" "$GPU_PFM" "$TOL" "$CORR" "${cpu_ms:-nan}" "${gpu_ms:-nan}" <<'PY'
import sys, struct, math

def read_pfm(path):
    with open(path, "rb") as f:
        hdr = f.readline().strip()
        assert hdr in (b"Pf", b"PF"), hdr
        ch = 1 if hdr == b"Pf" else 3
        w, h = map(int, f.readline().split())
        scale = float(f.readline().strip())
        endian = "<" if scale < 0 else ">"
        n = w * h * ch
        data = struct.unpack(f"{endian}{n}f", f.read(n * 4))
    return w, h, ch, list(data)

cpu_path, gpu_path, tol, corr_min, cpu_ms, gpu_ms = sys.argv[1:7]
tol = float(tol); corr_min = float(corr_min)
wc, hc, cc, a = read_pfm(cpu_path)
wg, hg, cg, b = read_pfm(gpu_path)

ok = True
if (wc, hc, cc) != (wg, hg, cg):
    print(f"FAIL: shape mismatch CPU={wc}x{hc}x{cc} GPU={wg}x{hg}x{cg}")
    sys.exit(1)

n = len(a)
diffs = [abs(a[i] - b[i]) for i in range(n)]
maxd = max(diffs)
meand = sum(diffs) / n
# Pearson correlation.
ma = sum(a) / n; mb = sum(b) / n
num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
da_ = math.sqrt(sum((x-ma)**2 for x in a))
db_ = math.sqrt(sum((x-mb)**2 for x in b))
corr = num / (da_*db_) if da_ > 0 and db_ > 0 else float("nan")

print(f"  shape       : {wc}x{hc} (ch={cc})")
print(f"  max|d|      : {maxd:.3e}   (tol {tol:.1e})")
print(f"  mean|d|     : {meand:.3e}")
print(f"  corr        : {corr:.6f}   (min {corr_min})")
parity = (maxd <= tol) and (corr >= corr_min)
print(f"  PARITY      : {'PASS' if parity else 'FAIL'}")

try:
    cms = float(cpu_ms); gms = float(gpu_ms)
    print(f"  CPU infer   : {cms:.1f} ms/iter")
    print(f"  GPU infer   : {gms:.1f} ms/iter")
    if gms > 0:
        print(f"  speedup     : {cms/gms:.2f}x")
except ValueError:
    print("  (bench timing unavailable — check the bench lines above)")

print()
print("RESULT:", "PASS" if parity else "FAIL")
sys.exit(0 if parity else 1)
PY
