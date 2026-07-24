#!/usr/bin/env bash
# Auto-discover videos and bake each into an accumulating-reconstruction demo.
#
# Mirrors the drop-a-folder convenience of demo-photos/ (scripts/demo/bake.sh),
# but for VIDEO: drop one folder per scene under demo-vids/ with a clip inside —
#   demo-vids/office-corner/clip.mp4
#   demo-vids/flower-bed/clip.mp4
# — then run this. Each clip is sampled into frames, the parallax gate curates the
# well-conditioned subset (scripts/make_accumulate_demo.sh --min-parallax), and the
# result is a self-contained growing-reconstruction demo (accumulate.html) under
# .cache/demo/<name>/.
#
#   nix develop -c scripts/demo/bake-vids.sh                 # bake every demo-vids/ folder
#   MIN_PARALLAX=6 MAXFRAMES=30 nix develop -c scripts/demo/bake-vids.sh
#
# Env: VIDS (demo-vids/), OUTBASE (.cache/demo), MAXFRAMES (frames sampled per
# clip, default 24), MIN_PARALLAX (gate degrees, default 8; 0 disables).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
VIDS=${VIDS:-$ROOT/demo-vids}
OUTBASE=${OUTBASE:-$ROOT/.cache/demo}
MAXFRAMES=${MAXFRAMES:-24}
export MIN_PARALLAX=${MIN_PARALLAX:-8}

command -v ffmpeg >/dev/null || { echo "ffmpeg required" >&2; exit 1; }
mkdir -p "$VIDS"
[ -f "$VIDS/README.txt" ] || cat >"$VIDS/README.txt" <<'TXT'
Drop one folder per scene here with a video clip inside (a time-lapse / slow pan
works best), then run scripts/demo/bake-vids.sh:
  office-corner/  clip.mp4
  flower-bed/     clip.mp4
Each clip is sampled to frames, the parallax gate keeps the well-conditioned ones,
and the result is a growing-reconstruction demo under .cache/demo/<name>/.
TXT

shopt -s nullglob
baked=()
for D in "$VIDS"/*/; do
  D=${D%/}; name=$(basename "$D")
  vid=$(find "$D" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.mkv' -o -iname '*.m4v' \) | sort | head -1)
  [ -n "$vid" ] || { echo "  no video in $name/ — skipping"; continue; }
  safe=$(printf '%s' "$name" | tr -c 'A-Za-z0-9_-' '_')

  # Sample frames: decode all, then take an even stride down to ~MAXFRAMES. (Time-
  # lapse clips are short, so decoding all is cheap and avoids frame-count guessing.)
  fdir="$OUTBASE/_frames/$safe"; rm -rf "$fdir"; mkdir -p "$fdir"
  ffmpeg -y -loglevel error -i "$vid" "$fdir/all%04d.png"
  mapfile -t ALL < <(ls "$fdir"/all*.png | sort)
  nb=${#ALL[@]}
  [ "$nb" -ge 2 ] || { echo "  $name: <2 frames decoded — skipping"; continue; }
  stride=$(( (nb + MAXFRAMES - 1) / MAXFRAMES )); [ "$stride" -lt 1 ] && stride=1
  FR=(); for ((i=0;i<nb;i+=stride)); do FR+=("${ALL[$i]}"); done
  echo "== $name: $(basename "$vid")  ($nb frames -> ${#FR[@]} sampled, stride $stride, gate ${MIN_PARALLAX}deg) =="

  bash scripts/make_accumulate_demo.sh "$OUTBASE/$safe" "${FR[@]}"
  baked+=("$safe")
done

if [ "${#baked[@]}" -eq 0 ]; then
  echo "no video folders found under $VIDS (drop demo-vids/<name>/<clip>.mp4)"; exit 0
fi
echo
echo "baked ${#baked[@]} demo(s): ${baked[*]}"
echo "serve one with:  python3 -m http.server -d $OUTBASE/<name> 8080   # open http://localhost:8080/"
