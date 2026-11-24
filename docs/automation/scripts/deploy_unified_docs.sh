#!/usr/bin/env bash
#
# Merge the built Sphinx documentation with the static website into a single
# GitHub Pages artifact, and derive the version metadata from that artifact.
#
# WHY THIS IS A STANDALONE SCRIPT
# -------------------------------
# GitHub rejects a workflow file when any single `run:` scalar exceeds 21000
# characters ("Exceeded max expression length 21000"). This step was 20354
# characters inline, which silently invalidated the whole workflow: no jobs
# were created, no logs were produced, and the only trace was an annotation
# buried in the run's HTML page. Keeping the logic in a file makes the
# workflow's inline length constant (~50 chars) no matter how the script
# grows, and makes the script testable with `bash -n` / shellcheck / a local
# dry run.
#
# SINGLE SOURCE OF VERSION TRUTH
# ------------------------------
# The set of version directories that actually exist in this deployment IS the
# truth. Upstream release listings are only an input for "which archives to
# try to download" -- never the answer to "which versions exist". Everything
# downstream (version switcher, 404 redirect, sitemap) reads the generated
# `documentation/versions.json`.
#
# INPUTS (env)
#   IS_RELEASE       "true" for a release build, otherwise development build
#   RELEASE_VERSION  release tag, e.g. v3.9.5          (release builds only)
#   DOC_VERSION      version directory name, e.g. v3.9.5 (release builds only)
#   SITE_URL         canonical site root, no trailing slash
#
# EXPECTED CWD LAYOUT
#   website-deploy/            static website (copied to the deploy root)
#   docs-output/               built Sphinx HTML
#   preserved-versions/        version archives restored from GitHub Releases
#   unified-deploy/            OUTPUT: the merged Pages artifact
#
# Exit codes: 0 success, 1 hard failure, 2 bad invocation.

set -euo pipefail

readonly SITE_URL="${SITE_URL:-https://asher-1.github.io/ACloudViewer}"
readonly IS_RELEASE="${IS_RELEASE:-false}"
readonly RELEASE_VERSION="${RELEASE_VERSION:-}"
readonly DOC_VERSION="${DOC_VERSION:-}"

readonly DOC_ROOT="unified-deploy/documentation"
readonly DEV_DIR="${DOC_ROOT}/dev"
# GitHub Pages' own 404 handler. Kept at the deploy root so it applies site-wide.
readonly NOT_FOUND_PAGE="unified-deploy/404.html"
readonly VERSIONS_JSON="${DOC_ROOT}/versions.json"

log()  { printf '%s\n' "$*"; }
rule() { log "────────────────────────────────────────────────────────────────"; }
die()  { log "❌ $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Redirect page writer.
#
# Emits a meta-refresh + JS redirect. `target` is a site-root-relative path,
# e.g. "./v3.9.5/index.html" or "./dev/index.html". Written as a function
# rather than a heredoc because heredocs inside YAML block scalars were the
# source of an earlier (and misdiagnosed) CI failure.
# ---------------------------------------------------------------------------
write_redirect() {
    local target="$1" title="$2" note="$3" out="$4"

    mkdir -p "$(dirname "$out")"
    {
        echo '<!DOCTYPE html>'
        echo '<html lang="en">'
        echo '<head>'
        echo '    <meta charset="utf-8">'
        echo "    <title>${title}</title>"
        echo "    <meta http-equiv=\"refresh\" content=\"0; url=${target}\">"
        echo "    <link rel=\"canonical\" href=\"${target}\">"
        echo '    <style>'
        echo '        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'
        echo '               display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;'
        echo '               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }'
        echo '        .container { text-align: center; padding: 2rem; }'
        echo '        a { color: #fff; text-decoration: underline; }'
        echo '    </style>'
        echo "    <script>window.location.replace(\"${target}\");</script>"
        echo '</head>'
        echo '<body>'
        echo '    <div class="container">'
        echo "        <h2>🔄 ${title}</h2>"
        echo "        <p>${note}</p>"
        echo "        <p>If not redirected automatically, <a href=\"${target}\">click here</a>.</p>"
        echo '    </div>'
        echo '</body>'
        echo '</html>'
    } > "$out"
}

# ---------------------------------------------------------------------------
# Collect version directory names present in this deployment, ascending.
# Only `vX.Y.Z` directories; `dev` is handled separately.
# ---------------------------------------------------------------------------
list_version_dirs() {
    local root="$1"
    [ -d "$root" ] || return 0
    find "$root" -mindepth 1 -maxdepth 1 -type d -name 'v*' -printf '%f\n' 2>/dev/null \
        | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
        | sort -V || true
}

# ---------------------------------------------------------------------------
# Emit documentation/versions.json -- the single input for every client-side
# version UI. Shaped like:
#   { "latest": "v3.9.5",
#     "generated_at": "...",
#     "versions": [ {"id":"v3.9.5","label":"v3.9.5","kind":"stable"}, ... ] }
# ---------------------------------------------------------------------------
write_versions_json() {
    local latest="$1"; shift
    local -a versions=("$@")
    local json="${VERSIONS_JSON}" first=1 entry kind id

    mkdir -p "$(dirname "$json")"
    {
        echo '{'
        echo "  \"latest\": \"${latest}\","
        echo "  \"generated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
        printf '  "versions": ['

        # dev first, so the switcher always offers a place to go.
        if [ -d "$DEV_DIR" ]; then
            printf '\n    {"id": "dev", "label": "Development (main)", "kind": "dev"}'
            first=0
        fi

        # Newest release first, matching user expectation in a dropdown.
        # `sort -Vr` needs at least one line; an empty array expands to a
        # single empty string under `set -u`, which the -n guard drops.
        for id in $(printf '%s\n' "${versions[@]-}" | sort -Vr); do
            [ -n "$id" ] || continue
            if [ "$id" = "$latest" ]; then kind="latest"; else kind="archive"; fi
            [ "$first" -eq 1 ] || printf ','
            printf '\n    {"id": "%s", "label": "%s", "kind": "%s"}' "$id" "$id" "$kind"
            first=0
        done

        printf '\n  ]\n}\n'
    } > "$json"

    log "  ✅ Wrote $(grep -c '"id"' "$json") entries to ${json} (latest=${latest:-none})"
}

# ---------------------------------------------------------------------------
# Regenerate sitemap.xml from the deployment itself. Deep pages are listed
# only for the newest version to avoid cross-version duplicate content;
# older archives are listed at their version root.
# ---------------------------------------------------------------------------
write_sitemap() {
    local latest="$1"
    local today; today="$(date +%F)"
    local doc_base page

    emit_url() {
        printf '    <url>\n        <loc>%s</loc>\n        <lastmod>%s</lastmod>\n' "$1" "$today"
        printf '        <changefreq>%s</changefreq>\n        <priority>%s</priority>\n    </url>\n' "$2" "$3"
    }

    {
        echo '<?xml version="1.0" encoding="UTF-8"?>'
        echo '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        emit_url "$SITE_URL/" weekly 1.0

        if [ -n "$latest" ] && [ -d "${DOC_ROOT}/${latest}" ]; then
            doc_base="$SITE_URL/documentation/${latest}"
            emit_url "${doc_base}/" weekly 0.9
            for page in \
                getting_started/introduction.html \
                getting_started/build_from_source.html \
                getting_started/quickstart.html \
                getting_started/installation.html \
                python_api/index.html \
                cpp_api/index.html \
                tutorial/index.html
            do
                [ -f "${DOC_ROOT}/${latest}/${page}" ] && emit_url "${doc_base}/${page}" weekly 0.7
            done
        fi

        for page in $(list_version_dirs "$DOC_ROOT"); do
            [ "$page" = "$latest" ] || emit_url "$SITE_URL/documentation/${page}/" monthly 0.4
        done

        [ -d "$DEV_DIR" ] && emit_url "$SITE_URL/documentation/dev/" weekly 0.5
        echo '</urlset>'
    } > unified-deploy/sitemap.xml

    log "  ✅ sitemap.xml: $(grep -c '<loc>' unified-deploy/sitemap.xml) URLs"
}

# ---------------------------------------------------------------------------
# Point the site-wide 404 handler at the newest version that really exists,
# so legacy flat deep links (/documentation/getting_started/foo.html) still
# resolve after the switch to a versioned layout.
# ---------------------------------------------------------------------------
calibrate_not_found_page() {
    local latest="$1"

    [ -f "$NOT_FOUND_PAGE" ] || { log "  ℹ️  no 404.html to calibrate"; return 0; }
    [ -n "$latest" ] || { log "  ℹ️  no version to calibrate 404.html with"; return 0; }

    # The page carries exactly one `var version = '...';` assignment and gates
    # on a vX.Y.Z|dev pattern, so an uncalibrated page degrades to a plain 404.
    sed -i "s/var version = '[^']*';/var version = '${latest}';/" "$NOT_FOUND_PAGE"
    log "  ✅ 404.html → ${latest}"
}

# ---------------------------------------------------------------------------
# Point the static website's documentation links at the version that was just
# deployed.
#
# The website is a static file authored long before any deployment knows what
# the newest version is. A hardcoded `/documentation/getting_started/...` is
# only valid while /documentation/ is a real directory; it became a redirect
# stub the moment the versioned layout landed, so those links 404'd in
# production. A hardcoded `/documentation/v3.9.4/...` would instead go stale
# on the next release.
#
# The source therefore carries an `@DOC_LATEST@` placeholder, and the version
# segment is filled in here -- from the same variable that drives the
# /documentation/ redirect, so the two can never disagree.
# ---------------------------------------------------------------------------
rewrite_website_doc_links() {
    local latest="${1:-dev}"
    local page count=0

    # index.html is the site; DOCUMENTATION.md is the human-readable map of
    # the same tree and carries the same links. Both ship the placeholder.
    for page in unified-deploy/index.html unified-deploy/DOCUMENTATION.md; do
        [ -f "$page" ] || continue
        grep -q '@DOC_LATEST@' "$page" || continue
        sed -i "s|@DOC_LATEST@|${latest}|g" "$page"
        count=$((count + 1))
    done

    if [ "$count" -gt 0 ]; then
        log "  ✅ website documentation links → /documentation/${latest}/"
    else
        log "  ℹ️  no @DOC_LATEST@ placeholder in the website"
    fi
}

# ---------------------------------------------------------------------------
# Propagate the newest version switcher into every archive. Historical
# archives were built against older templates and would otherwise ship a
# switcher that cannot see versions published after they were built.
# ---------------------------------------------------------------------------
propagate_version_switcher() {
    local latest="$1" src="" dir name css_src css_dest

    # Preference order matters. The dev build carries the switcher compiled
    # from the current source tree; an archive carries whatever was current
    # when that release shipped. Using the archive as the source would
    # propagate stale logic to every other archive, so always prefer the
    # freshest copy that exists in this deployment.
    for candidate in "${DEV_DIR}/_static/version_switch.js" \
                     "${DOC_ROOT}/${latest}/_static/version_switch.js"; do
        if [ -f "$candidate" ]; then src="$candidate"; break; fi
    done

    if [ -z "$src" ]; then
        log "  ⚠️  no version_switch.js found; archives keep their bundled copy"
        return 0
    fi

    log "  📁 Using switcher from ${src}"
    for dir in "${DOC_ROOT}"/*/; do
        name="$(basename "$dir")"
        [ "$name" = "dev" ] && continue
        # Create _static/ when an archive lacks it: an archive that shipped
        # without one would otherwise never pick up the current switcher.
        mkdir -p "${dir}_static"
        dest="${dir}_static/version_switch.js"
        # The newest archive may BE the source. Copying a file onto itself
        # fails under `set -e` and would abort the entire deployment.
        if [ "$(readlink -f "$src")" = "$(readlink -f "$dest")" ]; then
            log "    ⏭️  ${name}: already current"
            continue
        fi
        cp "$src" "$dest"
        log "    ✅ ${name}/_static/version_switch.js"

        # Some historical archives reference this stylesheet but omitted it.
        # Repair only missing assets; never replace an archive's own CSS.
        css_src="$(dirname "$src")/css/furo_overrides.css"
        css_dest="${dir}_static/css/furo_overrides.css"
        if [ ! -f "$css_dest" ] && [ -f "$css_src" ]; then
            mkdir -p "$(dirname "$css_dest")"
            cp "$css_src" "$css_dest"
            log "    ✅ restored missing ${name}/_static/css/furo_overrides.css"
        fi
    done
}

# ---------------------------------------------------------------------------
# Repair Doxygen mainpage pictures omitted by older release packages. Sources
# must belong to that same version; current dev screenshots are not substitutes.
# The workflow fetches only required release tags before this read-only step.
# ---------------------------------------------------------------------------
repair_archive_mainpage_images() {
    local name dir index image target source candidate repo commit temporary
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    while IFS= read -r name; do
        dir="${DOC_ROOT}/${name}"
        index="${dir}/cpp_api/api/index.html"
        [ -f "$index" ] || continue
        commit=""
        for image in AbstractionLayers.png MainUI.png ICP-registration.png Reconstruction.png SemanticAnnotation.png; do
            target="${dir}/cpp_api/api/${image}"
            [ ! -e "$target" ] && [ ! -L "$target" ] || continue
            grep -Eq "src=[\"']${image//./\\.}[\"']" "$index" || continue
            source=""
            for candidate in "${dir}/_images/${image}" "${dir}/images/${image}"; do
                if [ -f "$candidate" ] && [ ! -L "$candidate" ]; then
                    source="$candidate"
                    break
                fi
            done
            if [ -n "$source" ]; then
                cp "$source" "$target"
                log "  ✅ restored ${name}/cpp_api/api/${image} from same-version ${source}"
                continue
            fi
            if [ -z "$commit" ]; then
                commit="$(git -C "$repo" rev-parse --verify "refs/tags/${name}^{commit}")" || \
                    die "Cannot restore ${name}/${image}: same-release source tag is unavailable"
            fi
            temporary="$(mktemp "${target}.XXXXXX")"
            if ! git -C "$repo" show "${commit}:docs/images/${image}" > "$temporary" || [ ! -s "$temporary" ]; then
                # The checked mktemp path is the sole temporary file we own.
                [ "$(dirname "$temporary")" = "$(dirname "$target")" ] && \
                    [[ "$(basename "$temporary")" == "${image}."* ]] || die "Unexpected image staging path"
                rm -- "$temporary"
                die "Cannot restore ${name}/${image}: same-release image is unavailable at ${commit}"
            fi
            [ "$(dirname "$temporary")" = "$(dirname "$target")" ] && \
                [[ "$(basename "$temporary")" == "${image}."* ]] || die "Unexpected image staging path"
            mv -- "$temporary" "$target"
            log "  ✅ restored ${name}/cpp_api/api/${image} from release ${commit}"
        done
    done < <(list_version_dirs "$DOC_ROOT")
}

# v3.9.4 re-exported PyTorch 2.7 SummaryWriter docstrings but omitted their
# seven images. Preserve its original relative URLs and HTML; do not apply
# these version-specific assets to other releases or unknown references.
repair_archived_tensorboard_images() {
    local dir="${DOC_ROOT}/v3.9.4" page assets image target expected actual notice
    page="${dir}/python_api/cloudViewer.visualization.tensorboard_plugin.summary.SummaryWriter.html"
    [ -f "$page" ] || return 0
    grep -q 'python_api/_static/img/tensorboard/' "$page" || return 0
    assets="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../assets/torch-tensorboard"
    while IFS= read -r image; do
        case "$image" in
            add_histogram.png) expected=1cdb99f78f7678e24ae10a5598d94a8c4301b1b07937b2fe715ceb39e60d0208 ;;
            add_histogram_raw.png) expected=c5324136e40de96b18c230c6d92b441d603f17f47da3bd2cc101c80fca4982e0 ;;
            add_hparam.png) expected=a08ac24f8b106726fd2783cbd02647c43722c35e7615ea871134112c0b7b249a ;;
            add_image.png) expected=33577c29ac83db40e01bffe0d8fc7108273d4e3ab84a759a66f0eb5e2a69094a ;;
            add_images.png) expected=bbedd25f35cd9a89f7e9a76a2e812d29fb38a0c3f431fada60130ab2d8090912 ;;
            add_scalar.png) expected=2fb7ffb2ac36558b6060ce28d7b0ca70ef556771d608563f37785e7264d7dded ;;
            add_scalars.png) expected=d94436b2edb05443d6fa4023dd01d6c0eb79fec0e295ae28bc60fdc5047175d7 ;;
            *) die "Unknown v3.9.4 SummaryWriter TensorBoard image: ${image}" ;;
        esac
        target="${dir}/python_api/python_api/_static/img/tensorboard/${image}"
        [ ! -e "$target" ] && [ ! -L "$target" ] || continue
        [ -f "${assets}/${image}" ] || die "Missing verified PyTorch image: ${image}"
        actual="$(sha256sum "${assets}/${image}")"
        [ "${actual%% *}" = "$expected" ] || die "PyTorch image checksum mismatch: ${image}"
        mkdir -p "$(dirname "$target")"
        for notice in LICENSE README.md; do
            [ -s "${assets}/${notice}" ] || die "Missing PyTorch image attribution: ${notice}"
            [ -e "$(dirname "$target")/${notice}" ] || cp "${assets}/${notice}" "$(dirname "$target")/${notice}"
        done
        cp "${assets}/${image}" "$target"
        log "  ✅ restored v3.9.4 SummaryWriter ${image} from verified PyTorch 2.7 assets"
    done < <(grep -oE 'python_api/_static/img/tensorboard/[^"<>[:space:]]+' "$page" | \
             sed 's|.*/||' | sort -u)
}

# Bind recovered formula numbers to the exact historical TeX input, not to
# formula IDs from a newer dev build. Python uses only the standard library.
repair_archived_formulas() {
    local assets
    assets="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../assets/v3.9.4-formulas"
    python3 - "${DOC_ROOT}/v3.9.4/cpp_api/api" "$assets" <<'PY'
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys

api, assets = map(Path, sys.argv[1:])
if not api.is_dir():
    raise SystemExit(0)
references = set()
for page in api.rglob('*.html'):
    references.update(re.findall(r'''src=["'](form_\d+\.png)["']''', page.read_text(encoding='utf-8')))
missing = sorted(name for name in references if not (api / name).exists() and not (api / name).is_symlink())
if not missing:
    raise SystemExit(0)
expected_tex = '31adb22408e895f689f8f84405c8bb6d5e533418c444e69db4b857a28a8c5c94'
tex = api / '_formulas.tex'
if not tex.is_file() or hashlib.sha256(tex.read_bytes()).hexdigest() != expected_tex:
    raise SystemExit('Cannot restore v3.9.4 formulas: missing or mismatched original _formulas.tex')
manifest = json.loads((assets / 'manifest.json').read_text(encoding='utf-8'))
if manifest['original_tex_sha256'] != expected_tex:
    raise SystemExit('Cannot restore v3.9.4 formulas: source manifest mismatch')
rows = {row['image']: row for row in manifest['PNG_formula_mapping']}
for name in missing:
    source = assets / name
    if name not in rows or not source.is_file():
        raise SystemExit(f'Cannot restore v3.9.4 formula: unavailable verified asset {name}')
    if hashlib.sha256(source.read_bytes()).hexdigest() != rows[name]['png_sha256']:
        raise SystemExit(f'Cannot restore v3.9.4 formula: PNG checksum mismatch {name}')
for name in missing:
    shutil.copyfile(assets / name, api / name)
print(f'  ✅ restored {len(missing)} v3.9.4 formula PNGs from exact historical TeX')
PY
}

# ---------------------------------------------------------------------------
# Rewrite the pre-rename selector markup still baked into old archives.
# ---------------------------------------------------------------------------
fix_legacy_selector_markup() {
    local fixed=0 file

    while IFS= read -r file; do
        grep -q 'Latest (main)' "$file" 2>/dev/null || continue
        sed -i 's|<option value="latest">Latest (main)</option>|<option value="stable">Latest Stable</option><option value="dev">Development (main)</option>|g' "$file"
        sed -i 's|<option value="">Latest (main)</option>|<option value="stable">Latest Stable</option><option value="dev">Development (main)</option>|g' "$file"
        sed -i 's|>Latest (main)<|>Latest Stable<|g' "$file"
        fixed=$((fixed + 1))
    done < <(find "$DOC_ROOT" -name '*.html' -type f 2>/dev/null || true)

    log "  ✅ rewrote selector markup in ${fixed} file(s)"
}

# ===========================================================================
# Main
# ===========================================================================

rule
log "🔄 Merging website and documentation for unified deployment"
log "   /documentation/        → latest stable release (redirect)"
log "   /documentation/dev/    → development build"
log "   /documentation/vX.Y.Z/ → permanent version archives"
rule

[ -d website-deploy ] || die "website-deploy/ missing"
[ -d docs-output ]    || die "docs-output/ missing"

# A documentation tree with no entry page is not a degraded deployment, it is
# a broken one. Doxygen output (cpp_api/) survives a failed Sphinx run, so the
# directory can hold tens of thousands of files while /documentation/dev/
# itself serves 404. Refuse it here rather than publishing it.
[ -f docs-output/index.html ] || \
    die "docs-output/index.html is missing: the documentation build produced no entry page"

[ ! -e unified-deploy ] || die "unified-deploy already exists: use a fresh staging directory"
if [ "$IS_RELEASE" = "true" ]; then
    [[ "$DOC_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
        die "release DOC_VERSION must be a stable vX.Y.Z tag"
    [ "$RELEASE_VERSION" = "$DOC_VERSION" ] || die "release version inputs disagree"
fi
mkdir -p unified-deploy
cp -r website-deploy/. unified-deploy/
log "✅ Website copied to deploy root"

# ---------------------------------------------------------------------------
# This build's own documentation
# ---------------------------------------------------------------------------
# A release build without a version would silently take the dev branch below:
# the release docs would overwrite /documentation/dev/, no archive would be
# created, and /documentation/vX.Y.Z/ would never exist. Nothing would fail.
# The workflow always supplies DOC_VERSION for a release event, so reaching
# here means the two inputs disagree -- refuse rather than guess.
if [ "$IS_RELEASE" = "true" ] && [ -z "$DOC_VERSION" ]; then
    die "IS_RELEASE=true but DOC_VERSION is empty: a release build must name the version directory it publishes to"
fi

if [ "$IS_RELEASE" = "true" ]; then
    rule
    log "🎯 Release build → /documentation/${DOC_VERSION}/"
    rule
    mkdir -p "${DOC_ROOT}/${DOC_VERSION}"
    cp -r docs-output/. "${DOC_ROOT}/${DOC_VERSION}/"
    log "✅ Deployed /documentation/${DOC_VERSION}/"
else
    rule
    log "🔧 Development build → /documentation/dev/"
    rule
    mkdir -p "$DEV_DIR"
    cp -r docs-output/. "$DEV_DIR/"
    log "✅ Deployed /documentation/dev/"
fi

# ---------------------------------------------------------------------------
# Restore archives downloaded from GitHub Releases
# ---------------------------------------------------------------------------
rule
log "📦 Restoring version archives"
rule

if [ -d preserved-versions/documentation ]; then
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        target="${DOC_ROOT}/${name}"
        if [ -d "$target" ]; then
            log "  ⏭️  ${name} (already deployed by this build)"
            continue
        fi
        [ -f "preserved-versions/documentation/${name}/index.html" ] || \
            die "preserved ${name} has no index.html; refusing to lose a historical entry page"
        mkdir -p "$target"
        cp -r "preserved-versions/documentation/${name}/." "$target/"
        log "  ✅ restored ${name}"
    done < <(list_version_dirs preserved-versions/documentation)

    # `dev` is not a vX.Y.Z directory, so it is preserved separately. A
    # release build produces only the new version archive; without this the
    # entire dev tree would be dropped from the Pages artifact on every
    # release, and /documentation/dev/ would start 404ing.
    if [ ! -d "$DEV_DIR" ] && [ -d "preserved-versions/documentation/dev" ]; then
        [ -f preserved-versions/documentation/dev/index.html ] || \
            die "preserved dev has no index.html; refusing to advertise incomplete documentation"
        mkdir -p "$DEV_DIR"
        cp -r "preserved-versions/documentation/dev/." "$DEV_DIR/"
        log "  ✅ restored dev (preserved across this release deploy)"
    fi
else
    log "  ℹ️  no preserved archives"
fi

# ---------------------------------------------------------------------------
# Single source of truth: the version directories that actually exist
# ---------------------------------------------------------------------------
rule
log "🧭 Deriving version metadata from the deployment"
rule

mapfile -t DEPLOYED_VERSIONS < <(list_version_dirs "$DOC_ROOT")
LATEST_VERSION=""
if [ "${#DEPLOYED_VERSIONS[@]}" -gt 0 ]; then
    LATEST_VERSION="$(printf '%s\n' "${DEPLOYED_VERSIONS[@]}" | sort -V | tail -1)"
fi

log "  versions present: ${DEPLOYED_VERSIONS[*]-none}"
log "  latest resolved:  ${LATEST_VERSION:-none}"

if [ -n "$LATEST_VERSION" ]; then
    write_redirect "./${LATEST_VERSION}/index.html" \
        "Redirecting to Latest Stable Documentation" \
        "Serving the newest published release." \
        "${DOC_ROOT}/index.html"
    log "  ✅ /documentation/ → /documentation/${LATEST_VERSION}/"
else
    # No release has ever been deployed: dev is the only thing that exists.
    write_redirect "./dev/index.html" \
        "Redirecting to Development Documentation" \
        "No stable release has been published yet." \
        "${DOC_ROOT}/index.html"
    log "  ✅ /documentation/ → /documentation/dev/"
fi

write_versions_json "$LATEST_VERSION" "${DEPLOYED_VERSIONS[@]:-}"
rewrite_website_doc_links "${LATEST_VERSION:-dev}"
calibrate_not_found_page "$LATEST_VERSION"

# ---------------------------------------------------------------------------
# Keep every archive's switcher current
# ---------------------------------------------------------------------------
rule
log "🔄 Propagating version switcher to all archives"
rule
repair_archive_mainpage_images
repair_archived_tensorboard_images
repair_archived_formulas
# Original examples fill known missing QCustomPlot illustrations in both new
# and archived API trees; existing historical images and HTML stay unchanged.
for api_dir in "${DOC_ROOT}"/*/cpp_api/api; do
    [ -d "$api_dir" ] || continue
    python3 "$(dirname "${BASH_SOURCE[0]}")/install_qcp_doc_images.py" --api-root "$api_dir"
    # Historical HTML already loads doxygen.css. Refresh the shared header
    # layout through that stylesheet without rewriting archived API pages.
    python3 - "$api_dir" "$(dirname "${BASH_SOURCE[0]}")/../../source/_static/css/doxygen_overrides.css" <<'PYSTYLE'
from pathlib import Path
import shutil
import sys
api, source = map(Path, sys.argv[1:])
stylesheet = api / 'doxygen.css'
if stylesheet.is_file():
    shutil.copyfile(source, api / source.name)
    content = stylesheet.read_text(encoding='utf-8')
    directive = '@import url("doxygen_overrides.css");\n'
    if directive not in content:
        stylesheet.write_text(directive + content, encoding='utf-8')
PYSTYLE
done
propagate_version_switcher "$LATEST_VERSION"
fix_legacy_selector_markup

write_sitemap "$LATEST_VERSION"

# ---------------------------------------------------------------------------
# Officially supported site size and the tar acceptance ceiling differ.
# Project policy: warn beyond the 1 GB supported range, but only block at the
# documented tar ceiling. GB is decimal here: the tar must be <10,000,000,000 B.
# The gzip count is transfer telemetry, not an independent acceptance gate.
# Real deployment, timeout and live-site verification remain mandatory.
# ---------------------------------------------------------------------------
readonly PAGES_SUPPORTED_BYTES=1000000000
readonly PAGES_TAR_LIMIT_BYTES=10000000000

check_pages_size_budget() {
    local measurements raw_bytes tar_bytes gzip_bytes ver_mb name
    if ! measurements=$(python3 - <<'PYMEASURE'
# PAGES_ARCHIVE_MEASUREMENT
from pathlib import Path
import subprocess
import tempfile
import zlib

root = Path('unified-deploy')
raw = 0
for path in root.rglob('*'):
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise SystemExit(f'Unsupported Pages entry: {path}')
    if path.is_file():
        raw += path.stat().st_size
# Match upload-pages-artifact@v3's GNU tar inputs. Count the full tar stream,
# including headers, padding and end blocks, and compress that same stream.
with tempfile.TemporaryFile() as errors:
    process = subprocess.Popen([
        'tar', '--dereference', '--hard-dereference', '--directory', str(root),
        '-cf', '-', '--exclude=.git', '--exclude=.github', '.'
    ], stdout=subprocess.PIPE, stderr=errors)
    compressor = zlib.compressobj(level=6, wbits=31)
    tar_bytes = gzip_bytes = 0
    try:
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b''):
            tar_bytes += len(chunk)
            gzip_bytes += len(compressor.compress(chunk))
        gzip_bytes += len(compressor.flush())
    finally:
        process.stdout.close()
        code = process.wait()
    if code:
        errors.seek(0)
        raise SystemExit('Pages tar measurement failed: ' + errors.read().decode(errors='replace'))
print(raw, tar_bytes, gzip_bytes)
PYMEASURE
    ); then
        log "❌ Cannot validate Pages archive; refusing publication."
        return 1
    fi
    read -r raw_bytes tar_bytes gzip_bytes <<< "$measurements"
    if ! [[ "$raw_bytes" =~ ^[0-9]+$ && "$tar_bytes" =~ ^[0-9]+$ && "$gzip_bytes" =~ ^[0-9]+$ ]]; then
        log "❌ Invalid Pages size measurements."
        return 1
    fi
    log "   Published content: ${raw_bytes} bytes"
    log "   Uncompressed tar: ${tar_bytes} bytes (must be below ${PAGES_TAR_LIMIT_BYTES} bytes; 10 GB decimal)"
    log "   Gzip transfer estimate: ${gzip_bytes} bytes (informational, not Actions ZIP size)"
    if [ "$tar_bytes" -ge "$PAGES_TAR_LIMIT_BYTES" ]; then
        log "❌ Pages tar reaches the 10 GB acceptance ceiling; refusing deployment."
        return 1
    fi
    if [ "$raw_bytes" -gt "$PAGES_SUPPORTED_BYTES" ] || [ "$tar_bytes" -gt "$PAGES_SUPPORTED_BYTES" ]; then
        log "⚠️  Site/tar exceeds GitHub Pages' officially supported 1 GB range."
        log "   Proceeding under the selected project policy; deployment may still fail or time out."
        log "   Successful upload is not a substitute for deployment and live verification."
    fi
    if [ -d "$DOC_ROOT" ]; then
        for name in $(list_version_dirs "$DOC_ROOT"); do
            ver_mb=$(du -sm "${DOC_ROOT}/${name}" | cut -f1)
            log "     ${name}: ${ver_mb} MB on disk"
        done
        if [ -d "$DEV_DIR" ]; then
            ver_mb=$(du -sm "$DEV_DIR" | cut -f1)
            log "     dev: ${ver_mb} MB on disk"
        fi
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
rule
log "📊 Deployment summary"
log "   files: $(find unified-deploy -type f | wc -l)"
log "   size:  $(du -sh unified-deploy | cut -f1)"
log "   docs:  $(find "$DOC_ROOT" -type f 2>/dev/null | wc -l) files"
log "   latest: ${LATEST_VERSION:-none (dev only)}"
log "   stable: ${SITE_URL}/documentation/"
log "   dev:    ${SITE_URL}/documentation/dev/"
for name in "${DEPLOYED_VERSIONS[@]:-}"; do
    [ -n "$name" ] && log "   ${name}: ${SITE_URL}/documentation/${name}/"
done
rule

if ! check_pages_size_budget; then
    die "deployment exceeds the GitHub Pages artifact limit"
fi

log "✅ Unified deployment prepared"
