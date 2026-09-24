#!/usr/bin/env bash
# Regenerate SAM 3D parity golden assets from the upstream reference tree.
# Run this ONLY when the upstream sam-3d-objects-ggml tree advances; the PR
# must pair the golden refresh with the upstream version bump (review checklist).
#
# Prereqs:
#   - upstream repo at $UPSTREAM (default: ~/develop/code/github/dl/sam-3d-objects-ggml)
#     with a built cpp_ggml/build-cuda/bin/sam3d-cli
#   - output root at $GOLDEN_ROOT (default: /tmp/sam3d_golden) - publish its
#     contents to the shared data repo afterwards and update the manifest row.
set -euo pipefail

UPSTREAM="${UPSTREAM:-$HOME/develop/code/github/dl/sam-3d-objects-ggml}"
CLI="$UPSTREAM/cpp_ggml/build-cuda/bin/sam3d-cli"
GOLDEN="${GOLDEN_ROOT:-/tmp/sam3d_golden}"
COND="$GOLDEN/cond"
NOISE="$GOLDEN/noise"
DBG="$GOLDEN/dump"
SEED=42
BLOCKS=168

mkdir -p "$COND" "$NOISE" "$DBG"

# 1) pinned conditions (image -> preprocessed inputs + cond tokens)
"$CLI" e2e "$COND" --conditions-out "$COND" --seed "$SEED" \
    2>&1 | tail -2 || true
# NOTE: conditions-out needs the source image; keep sacre_coeur1.jpg beside
# this script's assets and pass --image if the e2e variant requires it.

# 2) official contract noise (SS draws + SLAT initial state)
"$CLI" rng-dump --seed "$SEED" --sizes 6,3,32768,3,1 \
    --distribution-blocks "$BLOCKS" --out-dir "$NOISE"
mv "$NOISE/rng_00.samt" "$NOISE/ss_x0_6drotation_normalized.samt"
mv "$NOISE/rng_01.samt" "$NOISE/ss_x0_scale.samt"
mv "$NOISE/rng_02.samt" "$NOISE/ss_x0_shape.samt"
mv "$NOISE/rng_03.samt" "$NOISE/ss_x0_translation.samt"
mv "$NOISE/rng_04.samt" "$NOISE/ss_x0_translation_scale.samt"
rm -f "$NOISE"/rng_*.samt
# SLAT: run the upstream once with --stage ss to learn its coords count N,
# then: rng-dump --seed $SEED --sizes $((8*N)) --distribution-blocks $BLOCKS
#       (rename to slat_x0.samt) and export the coords from that run as
#       slat_coords.samt. Sizes must be refreshed whenever SS occupancy moves.

# 3) golden dumps under the production contract semantics
"$CLI" run "$COND" --noise-dir "$NOISE" --cond-manual-attention-exp \
    --ss-attention normal --dtype q4_k --seed "$SEED" \
    --debug-stage post_patch --dino-dbg "$DBG/post_patch.bin" \
    --dbg-dir "$DBG" --out "$DBG/upstream_golden.ply"
"$CLI" run "$COND" --noise-dir "$NOISE" --cond-manual-attention-exp \
    --ss-attention normal --dtype q4_k --seed "$SEED" \
    --debug-stage b0_q --dino-dbg "$DBG/b0_q.bin" \
    --out "$DBG/upstream_golden_b0q.ply"

# 4) record the golden gaussian count for the L2 gate
python3 - "$DBG" <<'PY'
import re, sys, pathlib
d = pathlib.Path(sys.argv[1])
log = (d / "upstream_golden_count.txt")
# grep the count from the ply header via the last CLI log line is fragile;
# instead parse the .ply file the runner produced (vertex count element).
ply = d / "upstream_golden.ply"
head = ply.read_bytes()[:4096].split(b"end_header")[0].decode("utf8", "replace")
m = re.search(r"element vertex (\d+)", head)
log.write_text(m.group(1) if m else "UNKNOWN")
print("golden gaussian count:", m.group(1) if m else "UNKNOWN")
PY

echo "golden tree ready under $GOLDEN"
