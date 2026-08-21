#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# CUDA graph parity: compare the integrated ggml-cuda build artifact against
# an upstream ultralytics-ggml checkout. Part of the "CUDA graph 36~88%"
# follow-up documented in core/AICore/docs/cuda_graph_parity.md.
#
# This script collects the EVIDENCE only — it does not change any build
# configuration. It prints the per-op profile of both sides, the ggml-cuda
# compile-definition diff, and (when cuobjdump is present) a SASS dump of
# both cubins for manual kernel comparison.
#
# Usage:
#   cuda_build_compare.sh [--integrated-build DIR] [--upstream DIR] [--model gguf]
#
# Defaults:
#   --integrated-build build_app          (repo-root-relative)
#   --upstream          dl/ultralytics-ggml
#   --model             yolov8n-f16.gguf (relative to $MODELS_DIR env or
#                         <upstream>/cpp_ggml/models/gguf)
# ----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INTEGRATED="${INTEGRATED_BUILD:-$REPO_ROOT/build_app}"
UPSTREAM="${UPSTREAM_DIR:-$REPO_ROOT/dl/ultralytics-ggml}"
MODEL="${MODEL:-yolov8n-f16.gguf}"
TEST_BIN="${TEST_BIN:-$INTEGRATED/bin/test_yolo_capi_performance}"

usage() { sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 1; }
while [ $# -gt 0 ]; do
    case "$1" in
        --integrated-build) INTEGRATED="$2"; shift 2 ;;
        --upstream) UPSTREAM="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) usage ;;
    esac
done

if [ ! -x "$TEST_BIN" ]; then
    echo "error: test binary not found: $TEST_BIN (build AICore with CUDA first)" >&2
    exit 1
fi

# Locate the model: --model may be an absolute path, relative to MODELS_DIR,
# or the upstream models dir.
MODEL_PATH="$MODEL"
if [ ! -f "$MODEL_PATH" ]; then
    for cand in "${MODELS_DIR:-}/$MODEL" "$UPSTREAM/cpp_ggml/models/gguf/$MODEL"; do
        if [ -f "$cand" ]; then MODEL_PATH="$cand"; break; fi
    done
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "error: model not found: $MODEL (set MODELS_DIR or pass an absolute path)" >&2
    exit 1
fi

IMAGE="${IMAGE:-}"
if [ -z "$IMAGE" ]; then
    for cand in "$UPSTREAM/ultralytics/assets/bus.jpg" \
                "$UPSTREAM/assets/bus.jpg"; do
        if [ -f "$cand" ]; then IMAGE="$cand"; break; fi
    done
fi
if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
    echo "error: benchmark image not found (set IMAGE=...)" >&2
    exit 1
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/build_app/cuda_build_compare}"
mkdir -p "$OUT_DIR"

echo "== CUDA build-artifact parity =="
echo "  model   : $MODEL_PATH"
echo "  image   : $IMAGE"
echo "  out dir : $OUT_DIR"
echo

# 1) Integrated side: per-op profile through the C API.
echo "== [1/4] integrated per-op profile =="
AICORE_TEST_YOLO_MODELS_DIR="$(dirname "$MODEL_PATH")" \
AICORE_TEST_YOLO_IMAGE="$IMAGE" \
AICORE_TEST_YOLO_DEVICE=cuda \
AICORE_TEST_YOLO_THREADS=32 \
AICORE_TEST_YOLO_PROFILE=1 \
    "$TEST_BIN" \
    > "$OUT_DIR/integrated.jsonl" \
    2> "$OUT_DIR/integrated.op_profile.log" || true
grep '^\[' "$OUT_DIR/integrated.op_profile.log" > "$OUT_DIR/integrated.op_profile.tsv" || true
echo "  wrote $OUT_DIR/integrated.op_profile.tsv ($(wc -l < "$OUT_DIR/integrated.op_profile.tsv") ops)"
echo

# 2) Upstream side: same model through yolo-cli bench (if the checkout is
#    present and built). The upstream bench prints its own per-op table on
#    stderr with the same "[op profile]" header when profiling is enabled.
UPSTREAM_BENCH="$UPSTREAM/build/bin/yolo-cli"
if [ -x "$UPSTREAM_BENCH" ]; then
    echo "== [2/4] upstream per-op profile (yolo-cli bench) =="
    (cd "$UPSTREAM" && "$UPSTREAM_BENCH" bench "$MODEL_PATH" "$IMAGE" 2>&1 \
        || true) | grep '^\[' > "$OUT_DIR/upstream.op_profile.tsv" || true
    echo "  wrote $OUT_DIR/upstream.op_profile.tsv ($(wc -l < "$OUT_DIR/upstream.op_profile.tsv") lines)"
else
    echo "== [2/4] upstream yolo-cli not found ($UPSTREAM_BENCH) — skipping =="
fi
echo

# 3) Compile-definition diff for the ggml-cuda translation units.
echo "== [3/4] ggml-cuda compile definitions diff =="
find "$INTEGRATED/ggml/src/ext_ggml-build" -name flags.make \
    -path '*ggml-cuda*' -print -quit >/dev/null 2>&1 || true
INT_FLAGS="$(find "$INTEGRATED/ggml/src/ext_ggml-build" -name flags.make 2>/dev/null | grep -i cuda | head -1 || true)"
UP_FLAGS="$(find "$UPSTREAM/build" -name flags.make 2>/dev/null | grep -i cuda | head -1 || true)"
if [ -n "$INT_FLAGS" ] && [ -n "$UP_FLAGS" ]; then
    grep -oE '\-D[A-Za-z0-9_]+(=[0-9]+)?' "$INT_FLAGS" | sort -u \
        > "$OUT_DIR/integrated.defs"
    grep -oE '\-D[A-Za-z0-9_]+(=[0-9]+)?' "$UP_FLAGS" | sort -u \
        > "$OUT_DIR/upstream.defs"
    diff -u "$OUT_DIR/upstream.defs" "$OUT_DIR/integrated.defs" \
        > "$OUT_DIR/defs.diff" || true
    if [ -s "$OUT_DIR/defs.diff" ]; then
        echo "  definitions differ — see $OUT_DIR/defs.diff:"
        cat "$OUT_DIR/defs.diff"
    else
        echo "  definitions identical (integrated.defs == upstream.defs)"
    fi
else
    echo "  flags.make not found on one side (integrated=$INT_FLAGS upstream=$UP_FLAGS)"
fi
echo

# 4) SASS dump (optional; requires cuobjdump from the CUDA toolkit).
if command -v cuobjdump >/dev/null 2>&1; then
    echo "== [4/4] SASS dumps =="
    INT_CUBINS="$(find "$INTEGRATED/ggml/src/ext_ggml-build" -name '*.cubin' -path '*ggml-cuda*' 2>/dev/null | head -1 || true)"
    UP_CUBINS="$(find "$UPSTREAM/build" -name '*.cubin' -path '*ggml-cuda*' 2>/dev/null | head -1 || true)"
    if [ -n "$INT_CUBINS" ]; then
        cuobjdump -sass "$INT_CUBINS" > "$OUT_DIR/integrated.sass" 2>/dev/null || true
        echo "  integrated SASS: $OUT_DIR/integrated.sass ($(wc -l < "$OUT_DIR/integrated.sass") lines)"
    fi
    if [ -n "$UP_CUBINS" ]; then
        cuobjdump -sass "$UP_CUBINS" > "$OUT_DIR/upstream.sass" 2>/dev/null || true
        echo "  upstream   SASS: $OUT_DIR/upstream.sass ($(wc -l < "$OUT_DIR/upstream.sass") lines)"
    fi
else
    echo "== [4/4] cuobjdump not found — SASS comparison skipped =="
fi
echo
echo "Done. Evidence in $OUT_DIR — next steps per docs/cuda_graph_parity.md §6."
