#!/usr/bin/env bash
# Bake the accumulating-reconstruction demo: run the engine over a stream of
# photos, fold them into one growing world (free_splatter-cli --accumulate), and
# lay out a self-contained web demo — the cloud reconstructed from 2 photos, then
# 3, then 4, …, plus the input thumbnails and the accumulate.html viewer.
#
#   MODEL=.cache/freesplatter-scene-f16.gguf \
#     scripts/make_accumulate_demo.sh OUT_DIR FRAME0 FRAME1 FRAME2 ...
#
# Then serve it:  cd OUT_DIR && python3 -m http.server 8080   (open index.html)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
MODEL=${MODEL:-.cache/freesplatter-scene-f16.gguf}
MAXSPLATS=${MAXSPLATS:-500000}
# GPU by default (CPU is ~50x slower); DEVICE=cpu is the explicit opt-in. The CLI
# is resolved to the matching build, and the vulkan CLI is built if missing (it
# isn't part of the default CPU build) — mirroring scripts/serve.sh for the .so.
DEVICE=${DEVICE:-vulkan}
if [ "$DEVICE" = cpu ]; then
  CLI=${CLI:-$ROOT/build/release/bin/free_splatter-cli}
else
  CLI=${CLI:-$ROOT/build/vulkan/bin/free_splatter-cli}
  if [ ! -x "$CLI" ]; then
    echo "building vulkan CLI ($CLI) ..." >&2
    cmake -S "$ROOT" -B "$ROOT/build/vulkan" -DFREE_SPLATTER_VULKAN=ON -DFREE_SPLATTER_BUILD_SHARED=ON >&2
    cmake --build "$ROOT/build/vulkan" --target free_splatter-cli >&2
  fi
fi

[ "$#" -ge 3 ] || { echo "usage: $0 OUT_DIR FRAME0 FRAME1 [FRAME2 ...]  (>=2 frames)" >&2; exit 1; }
OUT=$1; shift
FRAMES=("$@")
mkdir -p "$OUT"

# 1) input thumbnails: center-crop square + resize, mirroring the engine's own
#    preprocessing so the strip shows exactly the view that was reconstructed.
for i in "${!FRAMES[@]}"; do
  ffmpeg -y -loglevel error -i "${FRAMES[$i]}" \
    -vf "crop='min(iw,ih)':'min(iw,ih)',scale=360:360" "$OUT/view_$i.jpg"
done

# 2) accumulate: one engine pass over the stream writes acc_2.splat, acc_3.splat,
#    ... (the cloud after each KEPT photo) + a best-frame consensus acc_fused.splat.
#    --min-parallax gates by keyframe: a candidate frame is folded in only if its
#    triangulation angle vs the last KEPT frame clears MIN_PARALLAX deg (else its
#    depth is ill-conditioned and the model would invent it); skipped frames re-
#    anchor against the last kept one. This is the after-inference angle, which the
#    model over-reports, so keep it well above COLMAP's 1-2 deg. MIN_PARALLAX=0
#    folds in every frame. --refine (OFF) is the gaussian averaging pass (blurs).
RFLAG=; [ "${REFINE:-0}" = 1 ] && RFLAG=--refine
MINPAR=${MIN_PARALLAX:-8}
PFLAG=; [ "$MINPAR" != "0" ] && PFLAG="--min-parallax $MINPAR"
LOG="$OUT/accumulate.log"
"$CLI" --device "$DEVICE" --accumulate $RFLAG $PFLAG --fuse --fuse-mode best \
  --splat-prefix "$OUT/acc" --max-splats "$MAXSPLATS" "$MODEL" "${FRAMES[@]}" | tee "$LOG"

# 3) manifest.json: one step per KEPT photo (acc_n.splat + its n kept thumbnails),
#    then the consensus-fused step. With gating only the frames that carried enough
#    parallax are kept: frame 0 is the anchor and each "keep frame J" line adds
#    input frame J. Ungated (MIN_PARALLAX=0), every frame is kept.
if [ -z "$PFLAG" ]; then
  kept=(); for i in "${!FRAMES[@]}"; do kept+=("$i"); done
else
  kept=(0)
  while read -r j; do kept+=("$j"); done < <(grep -oP '^keep frame \K[0-9]+' "$LOG")
fi
nkept=${#kept[@]}
[ "$nkept" -ge 2 ] || { echo "only $nkept frame(s) kept (MIN_PARALLAX=$MINPAR too high?)" >&2; exit 1; }
thumb() { printf '"view_%s.jpg"' "$1"; }       # thumbnail referenced by original index
{
  echo '{ "steps": ['
  for ((k=2;k<=nkept;k++)); do
    imgs=""
    for ((j=0;j<k;j++)); do imgs+="$(thumb "${kept[$j]}")"; [ $j -lt $((k-1)) ] && imgs+=", "; done
    printf '    {"splat":"acc_%d.splat","images":[%s],"n":%d},\n' "$k" "$imgs" "$k"
  done
  allimgs=""
  for ((j=0;j<nkept;j++)); do allimgs+="$(thumb "${kept[$j]}")"; [ $j -lt $((nkept-1)) ] && allimgs+=", "; done
  printf '    {"splat":"acc_fused.splat","images":[%s],"n":%d,"label":"consensus-fused — best-frame, parallax-gated"}\n' "$allimgs" "$nkept"
  echo '  ] }'
} > "$OUT/manifest.json"
echo "kept frames (input indices): ${kept[*]}  of ${#FRAMES[@]}"

# 4) the viewer, self-contained next to its assets
cp "$ROOT/server/web/accumulate.html" "$OUT/index.html"   # single source; standalone-fallback to ./manifest.json
echo "demo baked -> $OUT"
echo "serve: (cd $OUT && python3 -m http.server 8080)  then open http://localhost:8080/"
