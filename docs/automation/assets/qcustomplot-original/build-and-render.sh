#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${ACV_SOURCE_ROOT:?Set ACV_SOURCE_ROOT to the ACloudViewer source tree}"
export NPROC=2 CMAKE_BUILD_PARALLEL_LEVEL=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cmake -S "$ROOT" -B "$ROOT/build" \
  -DACV_SOURCE_ROOT="$ACV_SOURCE_ROOT" \
  -DCMAKE_PREFIX_PATH="${QT_PREFIX:-/opt/qt515}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$ROOT/build" --parallel 2
mkdir -p "$ROOT/output"
QT_QPA_PLATFORM=offscreen "$ROOT/build/qcp_original_illustrations" "$ROOT/output" \
  > "$ROOT/render.log" 2>&1
python3 "$ROOT/optimize_pngs.py"
