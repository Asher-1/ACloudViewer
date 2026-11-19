#!/usr/bin/env bash
#
# Assemble the static-website half of the GitHub Pages artifact.
#
# WHY THIS IS A STANDALONE SCRIPT
# -------------------------------
# This logic used to be a 130-line inline `run:` block in
# .github/workflows/documentation.yml. It could not be tested, it counted
# against the 21000-character limit that silently invalidates a workflow file,
# and reviewing a change to it meant scrolling through YAML. The deploy half
# was already extracted (deploy_unified_docs.sh) and has contract tests; this
# brings the website half to the same standard.
#
# INPUTS (env)
#   IS_RELEASE       "true" for a release build, otherwise development build
#   RELEASE_VERSION  release tag, e.g. v3.9.5 (release builds only)
#   GITHUB_TOKEN     optional; lets `gh release list` see private repositories
#
# EXPECTED CWD LAYOUT
#   docs/            the static website sources (copied, minus build output)
#   libs/cloudViewer/version.txt   fallback version source
#   website-deploy/  OUTPUT: the website tree handed to the deploy job
#
# Exit codes: 0 success, 1 hard failure.

set -euo pipefail

readonly IS_RELEASE="${IS_RELEASE:-false}"
readonly RELEASE_VERSION="${RELEASE_VERSION:-}"
readonly WEBSITE_DIR="website-deploy"
readonly VERSION_FILE="libs/cloudViewer/version.txt"

log()  { printf '%s\n' "$*"; }
die()  { log "❌ $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Copy the static website, excluding anything the docs build produces.
# `docs/_out` and `docs/_build` are build output; the rest is authored.
# ---------------------------------------------------------------------------
copy_website() {
    [ -d docs ] || die "docs/ missing"

    mkdir -p "$WEBSITE_DIR"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --exclude='_out' --exclude='_build' --exclude='*.pyc' \
              --exclude='__pycache__' --exclude='.pytest_cache' \
              docs/ "$WEBSITE_DIR/"
    else
        # rsync is not guaranteed on every runner image; cp is enough here.
        cp -r docs/. "$WEBSITE_DIR/"
        rm -rf "$WEBSITE_DIR/_out" "$WEBSITE_DIR/_build"
        find "$WEBSITE_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
        find "$WEBSITE_DIR" -name '*.pyc' -type f -delete 2>/dev/null || true
    fi

    log "✅ Website copied to ${WEBSITE_DIR}/"
}

# ---------------------------------------------------------------------------
# Decide which version the website advertises.
#
# The site must match /documentation/, which serves the newest stable release.
# A release build knows its tag; a development build has to look it up.
#
# The GitHub query is used first because it reflects what is actually
# published. version.txt is only a fallback and can legitimately name an
# unreleased version (it is bumped before the release is cut), so it must not
# be preferred over the release list.
# ---------------------------------------------------------------------------
resolve_website_version() {
    local latest

    if [ "$IS_RELEASE" = "true" ] && [ -n "$RELEASE_VERSION" ]; then
        printf '%s' "$RELEASE_VERSION"
        return 0
    fi

    latest=$(gh release list --limit 10 --json tagName \
                -q '.[].tagName' 2>/dev/null \
             | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
             | head -1 || true)
    if [ -n "$latest" ]; then
        printf '%s' "$latest"
        return 0
    fi

    local major minor patch
    if [ -f "$VERSION_FILE" ]; then
        major=$(awk '/CLOUDVIEWER_VERSION_MAJOR/{print $2}' "$VERSION_FILE")
        minor=$(awk '/CLOUDVIEWER_VERSION_MINOR/{print $2}' "$VERSION_FILE")
        patch=$(awk '/CLOUDVIEWER_VERSION_PATCH/{print $2}' "$VERSION_FILE")
        if [ -n "$major" ] && [ -n "$minor" ] && [ -n "$patch" ]; then
            printf 'v%s.%s.%s' "$major" "$minor" "$patch"
            return 0
        fi
    fi

    printf 'dev'
}

# ---------------------------------------------------------------------------
# Fill in the version placeholders.
#
# @VERSION@            -> the version shown on the page (WEBSITE_VERSION)
# @LATEST_STABLE_VERSION@ -> what 404.html redirects legacy deep links to
# @DOC_LATEST@         -> deliberately left for deploy_unified_docs.sh, which
#                         knows the version that was actually deployed
# ---------------------------------------------------------------------------
apply_version_placeholders() {
    local version="$1"

    if [ -f "${WEBSITE_DIR}/index.html" ]; then
        if grep -q '@VERSION@' "${WEBSITE_DIR}/index.html"; then
            sed -i "s/@VERSION@/${version}/g" "${WEBSITE_DIR}/index.html"
            log "  ✅ index.html @VERSION@ → ${version}"
        else
            log "  ℹ️  no @VERSION@ in index.html"
        fi
    fi

    if [ -f "${WEBSITE_DIR}/404.html" ]; then
        sed -i "s/@LATEST_STABLE_VERSION@/${version}/g" "${WEBSITE_DIR}/404.html"
        log "  ✅ 404.html @LATEST_STABLE_VERSION@ → ${version}"
    fi
}

# ---------------------------------------------------------------------------
# Ship downloads_data.json, or a schema-correct empty one.
#
# docs/script.js reads `version_metadata` as an ARRAY (it calls .find()) and
# the download map as `download_data`. Emitting anything else renders an empty
# table with no error.
# ---------------------------------------------------------------------------
install_downloads_data() {
    if [ -f "docs/downloads_data.json" ]; then
        cp docs/downloads_data.json "${WEBSITE_DIR}/"
        log "  ✅ downloads_data.json ($(du -h docs/downloads_data.json | cut -f1), $(jq '.version_metadata | length' docs/downloads_data.json 2>/dev/null || echo '?') versions)"
    else
        log "  ⚠️  docs/downloads_data.json missing; writing an empty, schema-correct placeholder"
        printf '{"generated_at":"%s","version_metadata":[],"download_data":{}}\n' \
            "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
            > "${WEBSITE_DIR}/downloads_data.json"
    fi
}

# ---------------------------------------------------------------------------
# DOCUMENTATION.md mirrors the tree it links to.
#
# Every link here must resolve. /documentation/ is a redirect stub under the
# versioned layout, so the real entries are cpp_api.html and tutorial/, and
# both live under the deployed version directory. python_api/ is intentionally
# absent: it only exists when the Python bindings import, which the build does
# not guarantee.
# ---------------------------------------------------------------------------
write_documentation_readme() {
    cat > "${WEBSITE_DIR}/DOCUMENTATION.md" <<'DOCEOF'
# ACloudViewer Documentation

Full documentation is available at: [/documentation/](/documentation/)

## Documentation Structure

- **Main Website**: Current page (project info, downloads, getting started)
- **Documentation**: [/documentation/](/documentation/) (versioned C++ API, tutorials, guides)

## Quick Links

- [C++ API Reference](/documentation/@DOC_LATEST@/cpp_api.html)
- [Tutorials](/documentation/@DOC_LATEST@/tutorial/)
- [Getting Started](/documentation/@DOC_LATEST@/getting_started/introduction.html)
DOCEOF
    log "  ✅ DOCUMENTATION.md written"
}

main() {
    log "📋 Preparing the main website for deployment"
    copy_website

    local website_version
    website_version="$(resolve_website_version)"
    log "  version: ${website_version}"
    apply_version_placeholders "${website_version}"

    install_downloads_data
    write_documentation_readme

    log ""
    log "✅ Website ready: $(find "$WEBSITE_DIR" -type f | wc -l) files, $(du -sh "$WEBSITE_DIR" | cut -f1)"
}

main "$@"
