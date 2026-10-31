#!/usr/bin/env bash
# YOLO upstream-parity benchmark: run the AICore bench matrix on one or more
# devices and gate it against the upstream ultralytics-ggml baseline.
#
# Usage:
#   tests/yolo/run_upstream_parity.sh \
#       [--upstream-bench <bench.jsonl>] [--models-dir <dir>] [--image <img>] \
#       [--devices cuda,vulkan,cpu] [--out-dir <dir>] [--warmup N] [--iters N] \
#       [--limit PCT] [--no-build]
#
# Defaults point at the canonical upstream checkout and the local build_app
# tree. Each device run writes <out-dir>/aicore-<device>.jsonl; the final gate
# is bench_compare.py against the upstream baseline (exit 1 on > limit% p50
# e2e regression).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
UPSTREAM_REPO="${UPSTREAM_REPO:-$HOME/develop/code/github/dl/ultralytics-ggml}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_app}"

UPSTREAM_BENCH="$UPSTREAM_REPO/cpp_ggml/benchmarks/bench.jsonl"
MODELS_DIR="$UPSTREAM_REPO/cpp_ggml/models/gguf"
IMAGE="$UPSTREAM_REPO/ultralytics/assets/bus.jpg"
DEVICES="cuda,vulkan,cpu"
OUT_DIR="$REPO_ROOT/build_app/yolo_parity"
WARMUP=20
ITERS=50
LIMIT=5
DO_BUILD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upstream-bench) UPSTREAM_BENCH="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --no-build) DO_BUILD=0; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$UPSTREAM_BENCH" ]]; then
    echo "upstream bench not found: $UPSTREAM_BENCH" >&2
    echo "set UPSTREAM_REPO or pass --upstream-bench" >&2
    exit 2
fi
if [[ ! -d "$MODELS_DIR" ]]; then
    echo "models dir not found: $MODELS_DIR" >&2
    exit 2
fi

if [[ "$DO_BUILD" == 1 ]]; then
    echo "==> building test_yolo_capi_performance"
    cmake --build "$BUILD_DIR" --target test_yolo_capi_performance -j"${BUILD_JOBS:-8}" \
        > /dev/null
fi

TEST_BIN="$BUILD_DIR/bin/aicore_tests/test_yolo_capi_performance"
if [[ ! -x "$TEST_BIN" ]]; then
    echo "test binary not found: $TEST_BIN" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
# NOTE: no LD_LIBRARY_PATH pointing INTO aicore_tests/ — a stale copy of the
# ggml core libs there would shadow the fresh ones in bin/. The test binaries
# carry an $ORIGIN/.. RPATH; only the Qt prefix needs to be on the path.
if [[ -d /opt/qt515/lib ]]; then
    export LD_LIBRARY_PATH="/opt/qt515/lib:$BUILD_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$BUILD_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

IFS=',' read -ra DEVICE_LIST <<< "$DEVICES"
rc=0
for device in "${DEVICE_LIST[@]}"; do
    out="$OUT_DIR/aicore-${device}.jsonl"
    echo "==> benching device=$device -> $out"
    # Upstream matrix threads: cpu rows used 8, cuda/vulkan rows used 32.
    # threads drive the CPU-side parts of the pipeline (preprocess, host-side
    # graph ops and backend-sched host work) — running the GPU rows with the
    # default 1 thread regresses preprocess ~7x and the CUDA graph up to ~50%,
    # which is a harness artifact, not a real regression.
    case "$device" in
        cpu)   export AICORE_TEST_YOLO_THREADS="${YOLO_CPU_THREADS:-8}" ;;
        cuda|vulkan) export AICORE_TEST_YOLO_THREADS="${YOLO_GPU_THREADS:-32}" ;;
        *)     unset AICORE_TEST_YOLO_THREADS ;;
    esac
    # One process per model, exactly like the upstream matrix generator
    # (yolo-cli bench --model M.gguf, one model per process). A single process
    # creating/releasing many CUDA sessions in a row can trip ggml's backend
    # teardown order (intermittent illegal memory access on large q8 models) —
    # a harness artifact the per-model split eliminates while keeping the
    # timing contract 1:1.
    : > "$out"
    : > "$OUT_DIR/aicore-${device}.log"
    for model in "$MODELS_DIR"/*.gguf; do
        if ! AICORE_TEST_YOLO_GGUF="$model" \
             AICORE_TEST_YOLO_DEVICE="$device" \
             AICORE_TEST_YOLO_IMAGE="$IMAGE" \
             AICORE_TEST_YOLO_WARMUP="$WARMUP" \
             AICORE_TEST_YOLO_ITERS="$ITERS" \
             "$TEST_BIN" >> "$out" 2>> "$OUT_DIR/aicore-${device}.log"; then
            echo "    FAILED: $(basename "$model") (see $OUT_DIR/aicore-${device}.log)" >&2
            rc=1
        fi
    done
    rows=$(grep -c '^{' "$out" || true)
    echo "    $rows model rows emitted"
done
unset AICORE_TEST_YOLO_THREADS

echo "==> gate: upstream vs AICore (limit ${LIMIT}% p50)"
cat "$OUT_DIR"/aicore-*.jsonl > "$OUT_DIR/aicore-all.jsonl" 2>/dev/null || true
if ! python3 "$REPO_ROOT/core/AICore/tests/yolo/bench_compare.py" \
        --upstream "$UPSTREAM_BENCH" \
        --aicore "$OUT_DIR/aicore-all.jsonl" \
        --limit "$LIMIT"; then
    rc=1
fi

exit $rc
