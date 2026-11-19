#!/usr/bin/env bash
# Contract tests for prepare_website.sh.
#
# Builds a synthetic layout and asserts the outputs, mirroring the style of
# test_deploy_unified_docs.sh. These cover the decisions that are invisible
# once the site is deployed: which version the page advertises, whether the
# 404 redirect survives, and whether the downloads schema matches what
# docs/script.js reads.
#
# Usage: bash docs/automation/scripts/test_prepare_website.sh
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_website.sh"
SCRATCH="$(mktemp -d)"
trap 'chmod -R u+w "$SCRATCH" 2>/dev/null || true; rm -r "$SCRATCH"' EXIT

pass=0
fail=0
ok()    { printf '  ✅ %s\n' "$*"; pass=$((pass + 1)); }
bad()   { printf '  ❌ %s\n' "$*"; fail=$((fail + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# new_case <name> -> isolated workdir with a minimal website + version.txt
new_case() {
    local d="${SCRATCH}/$1"
    mkdir -p "$d/docs" "$d/libs/cloudViewer"
    cat > "$d/docs/index.html" <<'EOF'
<html><body><span id="ver">@VERSION@</span></body></html>
EOF
    cat > "$d/docs/404.html" <<'EOF'
<script>var version = '@LATEST_STABLE_VERSION@';</script>
EOF
    echo '{"site":"data"}' > "$d/docs/downloads_data.json"
    # Build output that must NOT be copied
    mkdir -p "$d/docs/_out/html" "$d/docs/_build"
    echo '<html>built</html>' > "$d/docs/_out/html/index.html"
    echo 'v3.9.4' > "$d/docs/_build/marker"
    printf 'CLOUDVIEWER_VERSION_MAJOR 3\nCLOUDVIEWER_VERSION_MINOR 9\nCLOUDVIEWER_VERSION_PATCH 5\n' \
        > "$d/libs/cloudViewer/version.txt"
    printf '%s\n' "$d"
}

run_prepare() {
    local d="$1" is_release="$2" rel_ver="$3"
    ( cd "$d" && env IS_RELEASE="$is_release" RELEASE_VERSION="$rel_ver" \
        bash "$SCRIPT" > "$d/prepare.log" 2>&1 )
}

echo "═══ Case 1: release build uses the release tag ═══"
d=$(new_case c1)
run_prepare "$d" true v9.9.9
check "index.html carries the release tag" \
    "grep -q 'v9\.9\.9' '$d/website-deploy/index.html'"
check "no @VERSION@ survives" \
    "! grep -q '@VERSION@' '$d/website-deploy/index.html'"
check "404 calibrated to the release tag" \
    "grep -q \"var version = 'v9\.9\.9'\" '$d/website-deploy/404.html'"

echo "═══ Case 2: development build never advertises an unreleased version ═══"
# version.txt says 3.9.5 but the newest published release is v3.9.4. The site
# must follow what is deployed, not the bump that has not shipped yet.
d=$(new_case c2)
mkdir -p "$d/bin"
cat > "$d/bin/gh" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *"release list"*) printf 'main-devel\nv3.9.4\nv3.9.3\n' ;;
  *) exit 1 ;;
esac
EOF
chmod +x "$d/bin/gh"
( cd "$d" && env PATH="$d/bin:$PATH" IS_RELEASE=false RELEASE_VERSION="" \
    bash "$SCRIPT" > "$d/prepare.log" 2>&1 )
check "adopts the newest published release, not version.txt" \
    "grep -q 'v3\.9\.4' '$d/website-deploy/index.html'"
check "does not ship the unreleased 3.9.5" \
    "! grep -q 'v3\.9\.5' '$d/website-deploy/index.html'"
check "prerelease tag is not chosen" \
    "! grep -q 'main-devel' '$d/website-deploy/index.html'"

echo "═══ Case 3: falls back to version.txt when no release is listed ═══"
d=$(new_case c3)
mkdir -p "$d/bin"
cat > "$d/bin/gh" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
chmod +x "$d/bin/gh"
( cd "$d" && env PATH="$d/bin:$PATH" IS_RELEASE=false RELEASE_VERSION="" \
    bash "$SCRIPT" > "$d/prepare.log" 2>&1 )
check "falls back to version.txt" \
    "grep -q 'v3\.9\.5' '$d/website-deploy/index.html'"

echo "═══ Case 4: build output is excluded from the website tree ═══"
d=$(new_case c4)
run_prepare "$d" false ""
check "_out is not copied" "[ ! -e '$d/website-deploy/_out' ]"
check "_build is not copied" "[ ! -e '$d/website-deploy/_build' ]"
check "authored files are copied" "[ -f '$d/website-deploy/index.html' ]"

echo "═══ Case 5: downloads_data.json keeps the schema the site reads ═══"
d=$(new_case c5)
run_prepare "$d" false ""
check "existing downloads_data.json is shipped verbatim" \
    "grep -q '\"site\":\"data\"' '$d/website-deploy/downloads_data.json'"

echo "═══ Case 6: a missing downloads_data.json gets a schema-correct placeholder ═══"
d=$(new_case c6)
rm -f "$d/docs/downloads_data.json"
run_prepare "$d" false ""
check "placeholder is created" "[ -f '$d/website-deploy/downloads_data.json' ]"
check "version_metadata is an array" \
    "python3 -c \"import json;d=json.load(open('$d/website-deploy/downloads_data.json'));assert isinstance(d['version_metadata'],list), d['version_metadata']\""
check "download_data key is present" \
    "python3 -c \"import json;d=json.load(open('$d/website-deploy/downloads_data.json'));assert 'download_data' in d\""
check "legacy download_links key is gone" \
    "! grep -q 'download_links' '$d/website-deploy/downloads_data.json'"

echo "═══ Case 7: DOCUMENTATION.md leaves @DOC_LATEST@ for the deploy step ═══"
# deploy_unified_docs.sh owns that placeholder because only it knows which
# version was actually deployed.
d=$(new_case c7)
run_prepare "$d" false ""
check "DOCUMENTATION.md exists" "[ -f '$d/website-deploy/DOCUMENTATION.md' ]"
check "@DOC_LATEST@ survives for the deploy step" \
    "grep -q '@DOC_LATEST@' '$d/website-deploy/DOCUMENTATION.md'"
check "links point at the versioned layout" \
    "grep -q '/documentation/@DOC_LATEST@/cpp_api.html' '$d/website-deploy/DOCUMENTATION.md'"

echo
echo "──────────────────────────────"
printf 'PASS: %d   FAIL: %d\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
