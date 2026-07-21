#!/usr/bin/env bash
# Bake the demo's content: reconstruct a curated selection of scenes + objects
# through the running web server, save each as a .splat alongside downscaled
# copies of its source images, and emit a manifest.json that drives the
# storyboard page (web/demo.html). This is the "bake" stage of the
# bake -> render -> assemble pipeline (mirrors ~/c/LocalVQE/demo).
#
#   # with a server already up (scripts/serve.sh on :4001):
#   SERVER=http://127.0.0.1:4001 scripts/demo/bake.sh
#
# ADD YOUR OWN PHOTOS: drop a folder per scene/object under PHOTOS (default
# demo-photos/) — e.g. demo-photos/kitchen/ with 2 overlapping photos (scene) or
# demo-photos/mug/ with 3-4 photos around one object — then re-run this and reload
# the demo page. Each folder becomes a station, in alphabetical order. Conventions
# inside a folder (all optional):
#   model.txt  -> "scene" | "object" | "object-2dgs"  (else inferred: <=2 imgs=scene, else object)
#   label.txt  -> the caption shown on screen           (else the folder name)
#   nobg       -> an empty file; disables background removal for an object folder
# If PHOTOS has no folders, the built-in SAMPLES below are baked instead.
#
# Output goes to DEMO_DIR (default .cache/demo). Re-run to refresh; existing
# .splat files are reused unless FORCE=1.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

SERVER=${SERVER:-http://127.0.0.1:4001}
CACHE=${CACHE:-$ROOT/.cache}
DEMO_DIR=${DEMO_DIR:-$CACHE/demo}
PHOTOS=${PHOTOS:-$ROOT/demo-photos}     # drop folders of photos here
FPS=${FPS:-30}
WIDTH=${WIDTH:-1920}
HEIGHT=${HEIGHT:-1080}
THUMB_W=${THUMB_W:-640}    # downscale source images to this width for the billboards
FORCE=${FORCE:-0}

# Built-in sample stations (used only when PHOTOS has no folders):
# "name|model|remove_bg|label|relpath-under-CACHE|img,img,..."
SAMPLES=(
  "co3d_1|scene|0|Teddy bear · 2 photos|scenes/co3d_1|000.jpg,001.jpg"
  "re10k_1|scene|0|Bedroom · 2 photos|scenes/re10k_1|000008.png,000071.png"
  # office scenes removed per request — re-add by uncommenting:
  # "wild_1|scene|0|Office corner · 2 photos|scenes/wild_1|000.jpg,001.jpg"
  # "scannetpp_1|scene|0|Workstation · 2 photos|scenes/scannetpp_1|001.jpg,002.jpg"
  # object sample (wants bg removal; run bake inside `nix develop`):
  # "doll|object|1|Single object · 4 photos|objviews/doll|000.png,004.png,013.png,019.png"
)

command -v ffmpeg >/dev/null || { echo "ffmpeg required (for thumbnail downscale)" >&2; exit 1; }
curl -fsS "$SERVER/api/models" >/dev/null 2>&1 || {
  echo "no server at $SERVER — start one first (e.g. ADDR=:4001 scripts/serve.sh)" >&2
  exit 1
}
BG_AVAILABLE=$(curl -fsS "$SERVER/api/models" 2>/dev/null | python3 -c 'import sys,json;print(1 if json.load(sys.stdin).get("bg_remove") else 0)' 2>/dev/null || echo 0)

# Make the drop folder discoverable + self-documenting on first run.
mkdir -p "$PHOTOS" "$DEMO_DIR"
[ -f "$PHOTOS/README.txt" ] || cat >"$PHOTOS/README.txt" <<'TXT'
Drop one folder per scene/object here, then re-run scripts/demo/bake.sh and reload
http://<host>:4001/demo.html. Each folder = one station (alphabetical order):
  kitchen/   img1.jpg img2.jpg            # 2 overlapping photos -> "scene" model
  mug/       a.jpg b.jpg c.jpg d.jpg      # 3-4 photos around one object -> "object" model
Optional files inside a folder: model.txt ("scene"/"object"/"object-2dgs"),
label.txt (on-screen caption), nobg (empty file: skip background removal).
TXT

# Build the station list: the curated SAMPLES first, then ADD any dropped folders.
# (To drop a sample, comment it out in SAMPLES above — same as the office scenes.)
ENTRIES=()
for s in "${SAMPLES[@]}"; do
  IFS='|' read -r nm md bgf lb rd im <<<"$s"
  ENTRIES+=("$nm|$md|$bgf|$lb|$CACHE/$rd|$im")
done
samples_n=${#ENTRIES[@]}
shopt -s nullglob
for D in "$PHOTOS"/*/; do
  D=${D%/}
  mapfile -t imgfiles < <(find "$D" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -printf '%f\n' | sort)
  [ ${#imgfiles[@]} -gt 0 ] || { echo "  skipping empty folder $(basename "$D")"; continue; }
  bn=$(basename "$D"); n=${#imgfiles[@]}
  name=$(printf '%s' "$bn" | tr -c 'A-Za-z0-9_-' '_')
  if [ -f "$D/model.txt" ]; then model=$(tr -d '[:space:]' <"$D/model.txt")
  elif [ "$n" -le 2 ]; then model=scene; else model=object; fi
  if [ -f "$D/label.txt" ]; then label=$(head -1 "$D/label.txt"); else label="${bn//_/ } · $n photos"; fi
  bg=0; { [ "$model" = object ] && [ "$BG_AVAILABLE" = 1 ] && [ ! -f "$D/nobg" ]; } && bg=1
  imgs=$(IFS=,; echo "${imgfiles[*]}")
  ENTRIES+=("$name|$model|$bg|$label|$D|$imgs")
done
echo "stations: ${#ENTRIES[@]} ($samples_n sample(s) + $(( ${#ENTRIES[@]} - samples_n )) dropped folder(s))"
# DISCOVER_ONLY=1 prints the planned stations and exits (preview without baking).
if [ "${DISCOVER_ONLY:-0}" = 1 ]; then
  echo "planned stations (name | model | bg | label | src | images):"
  printf '  %s\n' "${ENTRIES[@]}"
  exit 0
fi

MANIFEST_STATIONS=""   # JSON array body, comma-joined
kept_names=()          # stations that made it in (for pruning removed ones)
i=0
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r name model bg label src imgs <<<"$entry"
  out_splat="$DEMO_DIR/$name.splat"
  out_imgdir="$DEMO_DIR/$name"
  mkdir -p "$out_imgdir"

  # build curl args: one -F images=@ per view, plus model + optional remove_bg
  curl_args=(-fsS -X POST "$SERVER/api/reconstruct" -F "model=$model")
  [ "$bg" = "1" ] && curl_args+=(-F "remove_bg=1")
  json_imgs=""
  j=0
  IFS=',' read -ra files <<<"$imgs"
  for fimg in "${files[@]}"; do
    f="$src/$fimg"
    [ -f "$f" ] || { echo "missing image: $f" >&2; exit 1; }
    curl_args+=(-F "images=@$f")
    # downscale a copy for the billboard (keeps demo dir small + textures cheap)
    thumb="$out_imgdir/img$j.jpg"
    ffmpeg -y -loglevel error -i "$f" -vf "scale='min($THUMB_W,iw)':-2" "$thumb"
    json_imgs+="${json_imgs:+,}\"$name/img$j.jpg\""
    j=$((j+1))
  done

  # Source signature: re-reconstruct when the folder's images (or model/bg) change,
  # so editing photos in a folder actually takes effect without FORCE.
  sig=$( { printf '%s|%s\n' "$model" "$bg"; for fimg in "${files[@]}"; do echo "$fimg $(stat -c '%s %Y' "$src/$fimg" 2>/dev/null)"; done; } | sort | sha1sum | cut -c1-16)
  sigfile="$DEMO_DIR/$name.sig"
  if [ "$FORCE" = "1" ] || [ ! -s "$out_splat" ] || [ "$(cat "$sigfile" 2>/dev/null)" != "$sig" ]; then
    echo "baking $name ($model${bg:+, bg=$bg}) from $src ..."
    # A single station failing (e.g. bg removal needs `nix develop` for libstdc++)
    # must not abort the whole bake — warn and skip it, keep the rest.
    if ! resp=$(curl "${curl_args[@]}" 2>/dev/null); then
      echo "  !! reconstruct failed for $name — skipping (run inside 'nix develop' if bg removal)" >&2
      continue
    fi
    id=$(printf '%s' "$resp" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])' 2>/dev/null) || { echo "  !! bad response for $name — skipping" >&2; continue; }
    nsp=$(printf '%s' "$resp" | python3 -c 'import sys,json;print(json.load(sys.stdin)["n_splats"])' 2>/dev/null)
    curl -fsS "$SERVER/api/splat/$id" -o "$out_splat" || { echo "  !! splat fetch failed for $name — skipping" >&2; continue; }
    printf '%s' "$sig" > "$sigfile"
    echo "  -> $out_splat ($nsp splats, $(du -h "$out_splat" | cut -f1))"
  else
    echo "keeping existing $out_splat (sources unchanged; FORCE=1 to rebuild)"
  fi

  [ -s "$out_splat" ] || { echo "  no splat for $name — skipping manifest entry" >&2; continue; }
  MANIFEST_STATIONS+="${MANIFEST_STATIONS:+,}"$'\n'"    {\"name\":\"$name\",\"model\":\"$model\",\"label\":\"$label\",\"splat\":\"$name.splat\",\"images\":[$json_imgs],\"offset\":[$(python3 -c "print($i*4.0)"),0,0]}"
  kept_names+=("$name")
  i=$((i+1))
done

[ "$i" -gt 0 ] || { echo "no stations baked — is the right model loaded?" >&2; exit 1; }

# prune outputs for stations that no longer exist (a folder/sample was removed)
keepset=" ${kept_names[*]} "
for f in "$DEMO_DIR"/*.splat; do
  [ -e "$f" ] || continue
  nm=$(basename "$f" .splat)
  case "$keepset" in *" $nm "*) ;; *) echo "pruned removed station: $nm"; rm -rf "$DEMO_DIR/$nm.splat" "$DEMO_DIR/$nm.sig" "$DEMO_DIR/$nm";; esac
done

cat >"$DEMO_DIR/manifest.json" <<JSON
{
  "fps": $FPS,
  "width": $WIDTH,
  "height": $HEIGHT,
  "timeline": { "travel": 1.6, "images_in": 1.0, "images_hold": 0.9, "images_shrink": 1.4, "build": 3.6, "settle": 1.0, "splat_lag": 0.5 },
  "stations": [$MANIFEST_STATIONS
  ]
}
JSON

echo "wrote $DEMO_DIR/manifest.json with $i stations"
echo "serve with: scripts/serve.sh has -demo-dir wired via make.sh, or add -demo-dir $DEMO_DIR"
