#!/usr/bin/env bash
# Local dry-run harness for deploy_unified_docs.sh.
# Builds synthetic layouts under a scratch dir and asserts the outputs.
# Usage: bash docs/automation/scripts/test_deploy_unified_docs.sh
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_unified_docs.sh"
REPO_ROOT="$(cd "$(dirname "$SCRIPT")/../../.." && pwd)"
SCRATCH="$(mktemp -d)"
trap 'chmod -R u+w "$SCRATCH" 2>/dev/null || true; rm -r "$SCRATCH"' EXIT

pass=0
fail=0

ok()   { printf '  ✅ %s\n' "$*"; pass=$((pass + 1)); }
bad()  { printf '  ❌ %s\n' "$*"; fail=$((fail + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# new_case <name> -> creates an isolated workdir, echoes its path
new_case() {
    local d="${SCRATCH}/$1"
    mkdir -p "$d/website-deploy/images" "$d/docs-output/_static/css"
    echo '/* current shared styles */' > "$d/docs-output/_static/css/furo_overrides.css"
    echo '<html>site</html>' > "$d/website-deploy/index.html"
    echo '<html>doc</html>'  > "$d/docs-output/index.html"
    echo '/*switcher*/'      > "$d/docs-output/_static/version_switch.js"
    cp "${REPO_ROOT}/docs/404.html" "$d/website-deploy/404.html"
    printf '%s\n' "$d"
}

run_deploy() {
    local d="$1"; shift
    ( cd "$d" && env IS_RELEASE="$1" RELEASE_VERSION="$2" DOC_VERSION="$3" \
        bash "$SCRIPT" > "$d/deploy.log" 2>&1 )
}

echo "═══ Case 1: release build, no pre-existing archives ═══"
d=$(new_case c1); run_deploy "$d" true v3.9.5 v3.9.5
check "versions.json is valid JSON" "python3 -m json.tool '$d/unified-deploy/documentation/versions.json' >/dev/null 2>&1"
check "latest == v3.9.5" "grep -q '\"latest\": \"v3.9.5\"' '$d/unified-deploy/documentation/versions.json'"
check "v3.9.5 dir created" "[ -d '$d/unified-deploy/documentation/v3.9.5' ]"
check "root redirect -> v3.9.5" "grep -q 'url=\./v3\.9\.5/index\.html' '$d/unified-deploy/documentation/index.html'"
check "404 calibrated to v3.9.5" "grep -q \"var version = 'v3.9.5'\" '$d/unified-deploy/404.html'"
check "sitemap references v3.9.5" "grep -q '/documentation/v3.9.5/' '$d/unified-deploy/sitemap.xml'"
check "no dev entry" "! grep -q '\"dev\"' '$d/unified-deploy/documentation/versions.json'"

echo "═══ Case 2: dev build + restored archives v3.9.3, v3.9.4 ═══"
d=$(new_case c2)
mkdir -p "$d/preserved-versions/documentation/v3.9.3" "$d/preserved-versions/documentation/v3.9.4"
echo a > "$d/preserved-versions/documentation/v3.9.3/index.html"
echo b > "$d/preserved-versions/documentation/v3.9.4/index.html"
mkdir -p "$d/preserved-versions/documentation/v3.9.3/_static/css"
echo '/* archived custom styles */' > "$d/preserved-versions/documentation/v3.9.3/_static/css/furo_overrides.css"
run_deploy "$d" false "" ""
check "versions.json is valid JSON" "python3 -m json.tool '$d/unified-deploy/documentation/versions.json' >/dev/null 2>&1"
check "latest == v3.9.4 (highest archive, not dev)" "grep -q '\"latest\": \"v3.9.4\"' '$d/unified-deploy/documentation/versions.json'"
check "root redirect -> v3.9.4" "grep -q 'url=\./v3\.9\.4/index\.html' '$d/unified-deploy/documentation/index.html'"
check "dev entry present" "grep -q '\"id\": \"dev\"' '$d/unified-deploy/documentation/versions.json'"
check "dev listed first" "[ \"\$(grep -o '\"id\": \"[^\"]*\"' '$d/unified-deploy/documentation/versions.json' | head -1)\" = '\"id\": \"dev\"' ]"
check "newest release listed before older" "python3 -c \"
import json,sys
v=[e['id'] for e in json.load(open('$d/unified-deploy/documentation/versions.json'))['versions']]
sys.exit(0 if v.index('v3.9.4') < v.index('v3.9.3') else 1)\""
check "v3.9.4 marked latest, v3.9.3 marked archive" "grep -q '\"id\": \"v3.9.4\", \"label\": \"v3.9.4\", \"kind\": \"latest\"' '$d/unified-deploy/documentation/versions.json'"
check "switcher propagated into archive" "cmp -s '$d/unified-deploy/documentation/dev/_static/version_switch.js' '$d/unified-deploy/documentation/v3.9.4/_static/version_switch.js'"
check "missing archived stylesheet is restored" "cmp -s '$d/docs-output/_static/css/furo_overrides.css' '$d/unified-deploy/documentation/v3.9.4/_static/css/furo_overrides.css'"
check "existing archived stylesheet is preserved" "grep -q 'archived custom styles' '$d/unified-deploy/documentation/v3.9.3/_static/css/furo_overrides.css'"
check "404 calibrated to v3.9.4" "grep -q \"var version = 'v3.9.4'\" '$d/unified-deploy/404.html'"

echo "═══ Case 3: dev build, never released ═══"
d=$(new_case c3); run_deploy "$d" false "" ""
check "versions.json is valid JSON" "python3 -m json.tool '$d/unified-deploy/documentation/versions.json' >/dev/null 2>&1"
check "latest is empty" "grep -q '\"latest\": \"\"' '$d/unified-deploy/documentation/versions.json'"
check "root redirect -> dev" "grep -q 'url=\./dev/index\.html' '$d/unified-deploy/documentation/index.html'"
check "404 left uncalibrated placeholder" "grep -q \"var version = '@LATEST_STABLE_VERSION@'\" '$d/unified-deploy/404.html'"
check "only dev in list" "[ \"\$(grep -c '\"id\"' '$d/unified-deploy/documentation/versions.json')\" -eq 1 ]"

echo "═══ Case 4: release build where an older archive is restored ═══"
d=$(new_case c4)
mkdir -p "$d/preserved-versions/documentation/v3.9.4"
echo b > "$d/preserved-versions/documentation/v3.9.4/index.html"
run_deploy "$d" true v3.9.5 v3.9.5
check "latest == v3.9.5" "grep -q '\"latest\": \"v3.9.5\"' '$d/unified-deploy/documentation/versions.json'"
check "both versions listed" "[ \"\$(grep -c '\"id\"' '$d/unified-deploy/documentation/versions.json')\" -eq 2 ]"
check "valid JSON" "python3 -m json.tool '$d/unified-deploy/documentation/versions.json' >/dev/null 2>&1"

echo "═══ Case 5: malformed version dirs are ignored ═══"
d=$(new_case c5)
mkdir -p "$d/preserved-versions/documentation/vBOGUS" "$d/preserved-versions/documentation/v3.9.4-beta"
echo x > "$d/preserved-versions/documentation/vBOGUS/index.html"
run_deploy "$d" false "" ""
check "bogus dirs not in versions.json" "! grep -q 'vBOGUS' '$d/unified-deploy/documentation/versions.json'"
check "valid JSON despite noise" "python3 -m json.tool '$d/unified-deploy/documentation/versions.json' >/dev/null 2>&1"

echo "═══ Case 6: release deploy preserves /documentation/dev/ ═══"
# A release build produces only the new archive. Pages replaces the whole
# artifact, so without restoration the dev tree — and every link to it —
# would disappear on each release.
d=$(new_case c6)
mkdir -p "$d/preserved-versions/documentation/dev/_static" \
         "$d/preserved-versions/documentation/v3.9.4"
echo '<html>dev</html>' > "$d/preserved-versions/documentation/dev/index.html"
echo '/*switcher*/'     > "$d/preserved-versions/documentation/dev/_static/version_switch.js"
echo '<html>old</html>' > "$d/preserved-versions/documentation/v3.9.4/index.html"
run_deploy "$d" true v3.9.5 v3.9.5
check "dev restored on a release deploy" "[ -f '$d/unified-deploy/documentation/dev/index.html' ]"
check "dev still listed in versions.json" "grep -q '\"id\": \"dev\"' '$d/unified-deploy/documentation/versions.json'"
check "new release is latest" "grep -q '\"latest\": \"v3.9.5\"' '$d/unified-deploy/documentation/versions.json'"
check "older archive still restored" "[ -f '$d/unified-deploy/documentation/v3.9.4/index.html' ]"
check "still valid JSON" "python3 -m json.tool '$d/unified-deploy/documentation/versions.json' >/dev/null 2>&1"

echo "═══ Case 7: website documentation links follow the deployed version ═══"
# The website is static and ships an `@DOC_LATEST@` placeholder. Fixing the
# version in the source would go stale on the next release; leaving it
# unexpanded (or hardcoding a flat /documentation/ path) 404s as soon as
# /documentation/ becomes a redirect stub. Both paths are asserted here.
d=$(new_case c7)
printf '%s\n' \
  '<a href="https://example.test/documentation/@DOC_LATEST@/getting_started/build_from_source.html">b</a>' \
  > "$d/website-deploy/index.html"
run_deploy "$d" true v3.9.5 v3.9.5
check "placeholder expanded to the new release" \
  "grep -q '/documentation/v3\.9\.5/getting_started/build_from_source\.html' '$d/unified-deploy/index.html'"
check "no placeholder survives into the artifact" \
  "! grep -q '@DOC_LATEST@' '$d/unified-deploy/index.html'"

echo "═══ Case 8: website links fall back to dev when nothing is released ═══"
d=$(new_case c8)
printf '%s\n' \
  '<a href="https://example.test/documentation/@DOC_LATEST@/getting_started/build_from_source.html">b</a>' \
  > "$d/website-deploy/index.html"
run_deploy "$d" false "" ""
check "placeholder expanded to dev" \
  "grep -q '/documentation/dev/getting_started/build_from_source\.html' '$d/unified-deploy/index.html'"
check "no placeholder survives into the artifact" \
  "! grep -q '@DOC_LATEST@' '$d/unified-deploy/index.html'"

echo "═══ Case 9: an entry-page-less docs tree is rejected ═══"
# Regression: a Sphinx run that produced nothing usable still leaves Doxygen's
# cpp_api/ behind, so docs-output/ can hold ~96k files and no index.html. The
# deploy used to accept that, publish it, and serve 404 at /documentation/dev/.
# Such a tree must now be refused before anything is uploaded.
d=$(new_case c9)
rm -f "$d/docs-output/index.html"
set +e
run_deploy "$d" false "" ""
rc=$?
set -e
check "deploy fails when docs-output has no index.html" "[ $rc -ne 0 ]"
check "failure names the missing entry page" \
  "grep -q 'index.html' '$d/deploy.log'"

echo
echo "──────────────────────────────"
printf 'PASS: %d   FAIL: %d\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
