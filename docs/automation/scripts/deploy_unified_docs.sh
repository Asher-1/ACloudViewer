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
    local latest="$1" src="" dir name

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
    done
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
        mkdir -p "$target"
        cp -r "preserved-versions/documentation/${name}/." "$target/"
        log "  ✅ restored ${name}"
    done < <(list_version_dirs preserved-versions/documentation)

    # `dev` is not a vX.Y.Z directory, so it is preserved separately. A
    # release build produces only the new version archive; without this the
    # entire dev tree would be dropped from the Pages artifact on every
    # release, and /documentation/dev/ would start 404ing.
    if [ ! -d "$DEV_DIR" ] && [ -d "preserved-versions/documentation/dev" ]; then
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
propagate_version_switcher "$LATEST_VERSION"
fix_legacy_selector_markup

write_sitemap "$LATEST_VERSION"

# ---------------------------------------------------------------------------
# GitHub Pages refuses artifacts over 1 GB.
#
# upload-pages-artifact only WARNS about an oversized artifact and
# deploy-pages then fails at the very end, after a 20-30 minute build. The
# live site keeps serving the previous deployment, so the failure looks like
# "the deploy half worked" -- new versions 404 while old ones stay reachable,
# which is exactly the confusing state this check prevents.
#
# The limit applies to the COMPRESSED artifact, so measuring the tree with
# `du` would both false-positive (documentation compresses to roughly 1/3)
# and miss the real threshold. This gzips the tree the same way
# upload-pages-artifact does and compares the resulting byte count.
#
# Measured cause the first time this fired: one restored archive built with
# the old Doxyfile (2.0 GB extracted) plus the new dev tree produced a
# 1343 MB artifact, 300 MB over. Sizes are reported per component so the
# culprit is obvious.
# ---------------------------------------------------------------------------
readonly PAGES_LIMIT_KB=$((1024 * 1024))

check_pages_size_budget() {
    local artifact_kb raw_mb ver_mb name

    raw_mb=$(du -sm unified-deploy | cut -f1)

    # -c: total only; the size is compared, the bytes themselves are discarded.
    artifact_kb=$(tar -czf - -C unified-deploy . 2>/dev/null | wc -c)
    artifact_kb=$((artifact_kb / 1024))

    log "   ${raw_mb} MB on disk, ${artifact_kb} KB compressed (GitHub Pages limit: 1048576 KB)"

    if [ -d "$DOC_ROOT" ]; then
        for name in $(list_version_dirs "$DOC_ROOT"); do
            ver_mb=$(du -sm "${DOC_ROOT}/${name}" | cut -f1)
            log "     ${name}: ${ver_mb} MB"
        done
        if [ -d "$DEV_DIR" ]; then
            ver_mb=$(du -sm "$DEV_DIR" | cut -f1)
            log "     dev: ${ver_mb} MB"
        fi
    fi

    if [ "$artifact_kb" -gt "$PAGES_LIMIT_KB" ]; then
        log ""
        log "❌ Compressed artifact is ${artifact_kb} KB; GitHub Pages rejects anything over ${PAGES_LIMIT_KB} KB."
        log "   deploy-pages would fail AFTER the build, leaving the previous site live"
        log "   and the new version unreachable. Failing here instead."
        log ""
        log "   Most likely cause: a version archive in the deployment was built with"
        log "   an older Doxyfile. Rebuild it with the current docs/Doxyfile.in, e.g."
        log "     DOT_IMAGE_FORMAT=svg  SOURCE_BROWSER=NO  INCLUDED_BY_GRAPH=NO"
        log "   and re-upload the release asset, or drop that version."
        return 1
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
