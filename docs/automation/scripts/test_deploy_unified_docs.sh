#!/usr/bin/env bash
# Local dry-run harness for deploy_unified_docs.sh.
# Builds synthetic layouts under a scratch dir and asserts the outputs.
# Usage: bash docs/automation/scripts/test_deploy_unified_docs.sh
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_unified_docs.sh"
REPO_ROOT="$(cd "$(dirname "$SCRIPT")/../../.." && pwd)"
RESTORE_SCRIPT="$(dirname "$SCRIPT")/restore_release_archives.sh"
SCRATCH="$(realpath "$(mktemp -d "${TMPDIR:-/tmp}/acloudviewer-deploy-test.XXXXXX")")"
cleanup() {
    local resolved
    resolved="$(realpath -m -- "$SCRATCH")"
    [ "$resolved" = "$SCRATCH" ] && [[ "$(basename "$resolved")" == acloudviewer-deploy-test.* ]] \
        || { echo "Refusing unexpected test cleanup path: $resolved" >&2; return 1; }
    chmod -R u+w "$resolved" 2>/dev/null || true
    rm -r -- "$resolved"
}
trap cleanup EXIT

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

echo "═══ Case 2b: repair only referenced same-release mainpage images ═══"
d=$(new_case c2b)
archive="$d/preserved-versions/documentation/v3.9.4"
mkdir -p "$archive/cpp_api/api" "$archive/_images" "$archive/images" "$d/docs-output/cpp_api/api"
echo stable > "$archive/index.html"
for image in AbstractionLayers.png MainUI.png ICP-registration.png Reconstruction.png SemanticAnnotation.png; do
    printf '<img src="%s">\n' "$image" >> "$archive/cpp_api/api/index.html"
    if [ "$image" = Reconstruction.png ]; then
        printf 'same-release-%s' "$image" > "$archive/images/$image"
    else
        printf 'same-release-%s' "$image" > "$archive/_images/$image"
    fi
    printf 'wrong-dev-%s' "$image" > "$d/docs-output/cpp_api/api/$image"
done
# Existing archive content must win even when another same-version source exists.
echo existing-history > "$archive/cpp_api/api/MainUI.png"
printf '/* archived Doxygen theme */\n' > "$archive/cpp_api/api/doxygen.css"
printf '/* dev Doxygen theme */\n' > "$d/docs-output/cpp_api/api/doxygen.css"
run_deploy "$d" false "" ""
for image in AbstractionLayers.png ICP-registration.png Reconstruction.png SemanticAnnotation.png; do
    check "same-release $image restored without dev substitution" \
        "grep -qx 'same-release-$image' '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/$image'"
done
check "existing archive mainpage picture is never overwritten" \
    "grep -qx existing-history '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/MainUI.png'"
check "archived mainpage HTML is unchanged by image repair" \
    "cmp -s '$archive/cpp_api/api/index.html' '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/index.html'"

for version in dev v3.9.4; do
    api="$d/unified-deploy/documentation/$version/cpp_api/api"
    check "$version gets current Doxygen layout" \
        "cmp -s '$REPO_ROOT/docs/source/_static/css/doxygen_overrides.css' '$api/doxygen_overrides.css'"
    check "$version loads the Doxygen layout from its existing stylesheet" \
        "grep -q '^@import.*doxygen_overrides.css' '$api/doxygen.css'"
done
check "historical Doxygen theme is retained" \
    "grep -q 'archived Doxygen theme' '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/doxygen.css'"

previous_archive="$d/unified-deploy/documentation/v3.9.4"
d=$(new_case c2b-repeat)
mkdir -p "$d/preserved-versions/documentation/v3.9.4"
cp -a "$previous_archive/." "$d/preserved-versions/documentation/v3.9.4/"
run_deploy "$d" false "" ""
check "redeploy adds no duplicate Doxygen imports" \
    "[ \"\$(grep -c '^@import' '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/doxygen.css')\" = 1 ]"

echo "═══ Case 2c: only the matching release tag can supply omitted assets ═══"
for fault in none unavailable; do
    d=$(new_case "c2c-$fault")
    archive="$d/preserved-versions/documentation/v3.9.4"
    mkdir -p "$archive/cpp_api/api" "$d/bin" "$d/release-images" "$d/docs-output/cpp_api/api"
    echo stable > "$archive/index.html"
    for image in AbstractionLayers.png MainUI.png ICP-registration.png Reconstruction.png SemanticAnnotation.png; do
        printf '<img src="%s">\n' "$image" >> "$archive/cpp_api/api/index.html"
        printf 'release-tag-%s' "$image" > "$d/release-images/$image"
        printf 'wrong-dev-%s' "$image" > "$d/docs-output/cpp_api/api/$image"
    done
    cat > "$d/bin/git" <<'GIT'
#!/usr/bin/env python3
import os,sys
from pathlib import Path
args=sys.argv[1:]
assert args[:2]==['-C',os.environ['TEST_SCRIPT_REPO']],args
args=args[2:]
root=Path(os.environ['TEST_IMAGE_ROOT'])
with (root/'git-calls.log').open('a') as log: log.write(' '.join(args)+'\n')
if args==['rev-parse','--verify','refs/tags/v3.9.4^{commit}']:
    print('a'*40)
elif len(args)==2 and args[0]=='show' and args[1].startswith('a'*40+':docs/images/'):
    if os.environ['TEST_IMAGE_FAULT']=='unavailable': raise SystemExit(1)
    name=args[1].split('/')[-1]
    sys.stdout.buffer.write((root/'release-images'/name).read_bytes())
else:
    raise AssertionError(args)
GIT
    chmod +x "$d/bin/git"
    set +e
    ( cd "$d" && env PATH="$d/bin:$PATH" TEST_SCRIPT_REPO="$REPO_ROOT" \
        TEST_IMAGE_ROOT="$d" TEST_IMAGE_FAULT="$fault" IS_RELEASE=false RELEASE_VERSION='' DOC_VERSION='' \
        bash "$SCRIPT" > "$d/deploy.log" 2>&1 )
    rc=$?
    set -e
    if [ "$fault" = none ]; then
        check "same-release tag extraction succeeds" "[ $rc -eq 0 ]"
        for image in AbstractionLayers.png MainUI.png ICP-registration.png Reconstruction.png SemanticAnnotation.png; do
            check "release tag supplies exact $image bytes" \
                "cmp -s '$d/release-images/$image' '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/$image'"
        done
        check "source commit identity is recorded" "grep -q 'from release aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' '$d/deploy.log'"
    else
        check "missing same-release source stops deployment" "[ $rc -ne 0 ]"
        check "missing source reports correct release and image" \
            "grep -q 'Cannot restore v3.9.4/AbstractionLayers.png' '$d/deploy.log'"
        check "failed extraction publishes no partial image" \
            "[ ! -e '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/AbstractionLayers.png' ] && ! compgen -G '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/AbstractionLayers.png.*' >/dev/null"
    fi
done

echo "═══ Case 2d: unreferenced images are not manufactured ═══"
d=$(new_case c2d)
archive="$d/preserved-versions/documentation/v3.9.4"
mkdir -p "$archive/cpp_api/api"
echo stable > "$archive/index.html"
echo '<html>API without showcase images</html>' > "$archive/cpp_api/api/index.html"
run_deploy "$d" false "" ""
check "unreferenced mainpage images are not added" \
    "[ ! -e '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/MainUI.png' ] && ! grep -q 'from release' '$d/deploy.log'"

echo "═══ Case 2e: archived PyTorch 2.7 figures preserve original URLs ═══"
d=$(new_case c2e)
archive="$d/preserved-versions/documentation/v3.9.4"
page=python_api/cloudViewer.visualization.tensorboard_plugin.summary.SummaryWriter.html
mkdir -p "$archive/python_api" "$archive/python_api/python_api/_static/img/tensorboard" \
    "$d/preserved-versions/documentation/v3.9.5/python_api"
echo stable > "$archive/index.html"
echo future > "$d/preserved-versions/documentation/v3.9.5/index.html"
for image in add_histogram.png add_histogram_raw.png add_hparam.png add_image.png add_images.png add_scalar.png add_scalars.png; do
    printf '<a href="python_api/_static/img/tensorboard/%s"><img src="python_api/_static/img/tensorboard/%s"></a>\n' "$image" "$image" >> "$archive/$page"
done
cp "$archive/$page" "$d/preserved-versions/documentation/v3.9.5/$page"
echo existing-release-image > "$archive/python_api/python_api/_static/img/tensorboard/add_scalar.png"
run_deploy "$d" false "" ""
for image in add_histogram.png add_histogram_raw.png add_hparam.png add_image.png add_images.png add_scalars.png; do
    check "archived $image uses verified upstream bytes" \
        "cmp -s '$REPO_ROOT/docs/automation/assets/torch-tensorboard/$image' '$d/unified-deploy/documentation/v3.9.4/python_api/python_api/_static/img/tensorboard/$image'"
done
check "existing archived TensorBoard image is untouched" \
    "grep -qx existing-release-image '$d/unified-deploy/documentation/v3.9.4/python_api/python_api/_static/img/tensorboard/add_scalar.png'"
check "SummaryWriter HTML stays byte-identical" \
    "cmp -s '$archive/$page' '$d/unified-deploy/documentation/v3.9.4/$page'"
check "upstream copyright and source attribution are distributed" \
    "cmp -s '$REPO_ROOT/docs/automation/assets/torch-tensorboard/LICENSE' '$d/unified-deploy/documentation/v3.9.4/python_api/python_api/_static/img/tensorboard/LICENSE' && [ -s '$d/unified-deploy/documentation/v3.9.4/python_api/python_api/_static/img/tensorboard/README.md' ]"
check "known Torch assets are not applied to an unknown archive" \
    "[ ! -e '$d/unified-deploy/documentation/v3.9.5/python_api/python_api/_static/img/tensorboard/add_image.png' ]"

for fault in unknown missing; do
    d=$(new_case "c2e-$fault")
    archive="$d/preserved-versions/documentation/v3.9.4"
    mkdir -p "$archive/python_api"
    echo stable > "$archive/index.html"
    if [ "$fault" = unknown ]; then
        echo '<img src="python_api/_static/img/tensorboard/unreviewed.png">' > "$archive/$page"
        tested_script="$SCRIPT"
    else
        echo '<img src="python_api/_static/img/tensorboard/add_image.png">' > "$archive/$page"
        mkdir -p "$d/isolated/docs/automation/scripts"
        cp "$SCRIPT" "$d/isolated/docs/automation/scripts/deploy_unified_docs.sh"
    cp "$(dirname "$SCRIPT")/install_qcp_doc_images.py" "$d/isolated/docs/automation/scripts/"
        tested_script="$d/isolated/docs/automation/scripts/deploy_unified_docs.sh"
    fi
    set +e
    ( cd "$d" && env IS_RELEASE=false RELEASE_VERSION='' DOC_VERSION='' bash "$tested_script" > "$d/deploy.log" 2>&1 )
    rc=$?
    set -e
    check "$fault TensorBoard resource fails explicitly" "[ $rc -ne 0 ] && grep -Eq 'Unknown v3.9.4 SummaryWriter|Missing verified PyTorch' '$d/deploy.log'"
done

echo "═══ Case 2f: formula recovery is bound to historical TeX ═══"
for fault in none tex checksum; do
    d=$(new_case "c2f-$fault")
    archive="$d/preserved-versions/documentation/v3.9.4"
    mkdir -p "$archive/cpp_api/api" "$d/isolated/docs/automation/scripts" \
        "$d/isolated/docs/automation/assets"
    echo stable > "$archive/index.html"
    echo '<img src="form_0.png"><img src="form_1.png">' > "$archive/cpp_api/api/index.html"
    echo existing-formula > "$archive/cpp_api/api/form_1.png"
    cp "$SCRIPT" "$d/isolated/docs/automation/scripts/deploy_unified_docs.sh"
    cp "$(dirname "$SCRIPT")/install_qcp_doc_images.py" "$d/isolated/docs/automation/scripts/"
    cp -r "$REPO_ROOT/docs/automation/assets/v3.9.4-formulas" "$d/isolated/docs/automation/assets/"
    python3 - "$d" "$fault" <<'PYTEST'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);fault=sys.argv[2]
assets=root/'isolated/docs/automation/assets/v3.9.4-formulas'
m=json.loads((assets/'manifest.json').read_text())
tex=m['original_tex'].encode()
assert hashlib.sha256(tex).hexdigest()==m['original_tex_sha256']
(root/'preserved-versions/documentation/v3.9.4/cpp_api/api/_formulas.tex').write_bytes(tex)
if fault=='checksum':(assets/'form_0.png').write_bytes(b'corrupt')
PYTEST
    if [ "$fault" = tex ]; then echo wrong-version > "$archive/cpp_api/api/_formulas.tex"; fi
    set +e
    ( cd "$d" && env IS_RELEASE=false RELEASE_VERSION='' DOC_VERSION='' \
        bash "$d/isolated/docs/automation/scripts/deploy_unified_docs.sh" > "$d/deploy.log" 2>&1 )
    rc=$?
    set -e
    if [ "$fault" = none ]; then
        check "formula PNG restored from verified same-version mapping" \
            "[ $rc -eq 0 ] && cmp -s '$REPO_ROOT/docs/automation/assets/v3.9.4-formulas/form_0.png' '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/form_0.png'"
        check "existing formula PNG is not overwritten" \
            "grep -qx existing-formula '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/form_1.png'"
        check "formula repair preserves historical HTML" \
            "cmp -s '$archive/cpp_api/api/index.html' '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/index.html'"
    else
        check "formula $fault mismatch fails explicitly" \
            "[ $rc -ne 0 ] && grep -q 'Cannot restore v3.9.4 formula' '$d/deploy.log'"
        check "formula $fault failure does not publish PNG" \
            "[ ! -e '$d/unified-deploy/documentation/v3.9.4/cpp_api/api/form_0.png' ]"
    fi
done

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

echo "═══ Case 10: incomplete preserved dev is rejected on release ═══"
d=$(new_case c10)
mkdir -p "$d/preserved-versions/documentation/dev/cpp_api"
echo partial > "$d/preserved-versions/documentation/dev/cpp_api/index.html"
set +e
run_deploy "$d" true v3.9.5 v3.9.5
rc=$?
set -e
check "invalid preserved dev blocks publication" "[ $rc -ne 0 ]"
check "preserved dev failure names its entry page" \
    "grep -q 'preserved dev has no index.html' '$d/deploy.log'"

echo "═══ Case 11: reused staging is rejected rather than merged ═══"
d=$(new_case c11)
run_deploy "$d" false "" ""
set +e
run_deploy "$d" false "" ""
rc=$?
set -e
check "reused staging cannot retain deleted pages" "[ $rc -ne 0 ]"

echo "═══ Case 12: content beyond 1 GiB warns without dropping pages ═══"
d=$(new_case c12)
python3 - "$d/docs-output/expanded.txt" <<'PYSPARSE'
import sys
with open(sys.argv[1], 'wb') as stream:
    stream.truncate(1073741825)
PYSPARSE
set +e
run_deploy "$d" false "" ""
rc=$?
set -e
check "raw content beyond 1 GiB is not an independent blocker" "[ $rc -eq 0 ]"
check "supported-size warning is visible" "grep -qi 'warn\|warning\|⚠' '$d/deploy.log'"
check "large source and entry page are retained" \
    "[ -f '$d/unified-deploy/documentation/dev/index.html' ] && [ \"\$(stat -c %s '$d/unified-deploy/documentation/dev/expanded.txt')\" = 1073741825 ]"

echo "═══ Case 13: exact uncompressed tar boundary is authoritative ═══"
# Exercise the real owner function through its full script, replacing only the
# marked measurement process. No 10 GB stream or copied threshold algorithm.
real_python=$(command -v python3)
for spec in 'below:1 9999999999 1000000001:0' 'at:1 10000000000 1:1' 'above:1 10000000001 1:1' 'process-error:ERROR:1'; do
    label=${spec%%:*};rest=${spec#*:};measurement=${rest%:*};expected=${rest##*:}
    d=$(new_case "c13-$label")
    mkdir -p "$d/bin"
    printf '#!%s\n' "$real_python" > "$d/bin/python3"
    cat >> "$d/bin/python3" <<'PYMEASURE'
import os,subprocess,sys
if sys.argv[1:]==['-']:
    source=sys.stdin.read()
    if '# PAGES_ARCHIVE_MEASUREMENT' in source:
        if os.environ['TEST_MEASUREMENT']=='ERROR':
            print('simulated tar measurement failure',file=sys.stderr)
            raise SystemExit(23)
        print(os.environ['TEST_MEASUREMENT'])
        raise SystemExit(0)
    result=subprocess.run([os.environ['REAL_PYTHON'],'-'],input=source,text=True)
    raise SystemExit(result.returncode)
os.execv(os.environ['REAL_PYTHON'],[os.environ['REAL_PYTHON'],*sys.argv[1:]])
PYMEASURE
    chmod +x "$d/bin/python3"
    set +e
    ( cd "$d" && env PATH="$d/bin:$PATH" REAL_PYTHON="$real_python" TEST_MEASUREMENT="$measurement" \
        IS_RELEASE=false RELEASE_VERSION='' DOC_VERSION='' bash "$SCRIPT" > "$d/deploy.log" 2>&1 )
    rc=$?
    set -e
    if [ "$expected" = 0 ]; then
        check "tar below 10 GB passes even when gzip exceeds 1 GB" "[ $rc -eq 0 ]"
    else
        check "tar boundary or measurement $label fails closed" "[ $rc -ne 0 ]"
    fi
done

echo "═══ Case 13b: unsupported publication file fails measurement ═══"
d=$(new_case c13-filetype)
mkfifo "$d/docs-output/unsupported.pipe"
set +e
run_deploy "$d" false "" ""
rc=$?
set -e
check "nonregular publication input cannot be silently omitted" "[ $rc -ne 0 ]"

echo "═══ Case 13c: tar producer failure is not a valid size ═══"
d=$(new_case c13-tarerror)
mkdir -p "$d/bin"
printf '#!/usr/bin/env bash\necho simulated-tar-failure >&2\nexit 7\n' > "$d/bin/tar"
chmod +x "$d/bin/tar"
set +e
( cd "$d" && env PATH="$d/bin:$PATH" IS_RELEASE=false RELEASE_VERSION='' DOC_VERSION='' \
    bash "$SCRIPT" > "$d/deploy.log" 2>&1 )
rc=$?
set -e
check "tar subprocess failure blocks publication" \
    "[ $rc -ne 0 ] && grep -q 'Pages tar measurement failed' '$d/deploy.log'"

# Reuse this harness's isolated cases for the release-restoration boundary.
# No test reaches GitHub: the fake accepts only the exact gh operations used
# by the helper and copies tiny local tar fixtures for known asset names.
new_restore_case() {
    local d
    d=$(new_case "$1")
    mkdir -p "$d/bin" "$d/assets"
    printf '%s\n' '[{"tagName":"v3.9.4","isDraft":false},{"tagName":"v3.9.5","isDraft":false},{"tagName":"main-devel","isDraft":false},{"tagName":"v9.9.9","isDraft":true}]' > "$d/releases.json"
    printf '%s\n' '{"assets":[{"name":"acloudviewer-v3.9.4-docs.tar.gz"}]}' > "$d/v3.9.4.json"
    # Lexically smaller SHA is newer: selecting by name must fail this case.
    printf '%s\n' '{"assets":[{"name":"acloudviewer-ffff-docs.tar.gz","createdAt":"2026-01-01T00:00:00Z"},{"name":"downloads_data.json"},{"name":"acloudviewer-0000-docs.tar.gz","createdAt":"2026-02-01T00:00:00Z"}]}' > "$d/main-devel.json"
    cat > "$d/bin/gh" <<'GH'
#!/usr/bin/env bash
set -euo pipefail
root="${FAKE_GH_ROOT:?}"
printf '%s\n' "$*" >> "$root/gh-calls.log"
case "$1 $2" in
    'release list')
        [ ! -e "$root/list-error" ] || exit 23
        cat "$root/releases.json" ;;
    'release view')
        [ ! -e "$root/view-error-$3" ] || exit 24
        cat "$root/$3.json" ;;
    'release download')
        release="$3"; shift 3
        pattern='' dir=''
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --pattern) pattern="$2"; shift 2 ;;
                --dir) dir="$2"; shift 2 ;;
                *) exit 90 ;;
            esac
        done
        [ ! -e "$root/download-error-$release" ] || exit 25
        [ ! -e "$root/download-omitted" ] || exit 0
        [ -n "$dir" ] && [ -n "$pattern" ] || exit 91
        cp "$root/assets/$release/$pattern" "$dir/$pattern" ;;
    *) exit 92 ;;
esac
GH
    chmod +x "$d/bin/gh"
    printf '%s\n' "$d"
}

make_restore_archive() {
    local d="$1" release="$2" asset="$3" content="$4" tree
    tree="$(mktemp -d "$d/archive-tree.XXXXXX")"
    mkdir -p "$d/assets/$release" "$tree/cpp_api"
    echo 'C++ reference' > "$tree/cpp_api/index.html"
    if [ -n "$content" ]; then
        printf '%s\n' "$content" > "$tree/index.html"
    fi
    tar -czf "$d/assets/$release/$asset" -C "$tree" .
}

run_restore() {
    local d="$1" dev="${2:-true}" current="${3:-v3.9.5}"
    if ( cd "$d" && env PATH="$d/bin:$PATH" FAKE_GH_ROOT="$d" \
        bash "$RESTORE_SCRIPT" "$d/restored" "$dev" "$current" > "$d/restore.log" 2>&1 ); then
        restore_rc=0
    else
        restore_rc=$?
    fi
}

valid_restore_archives() {
    make_restore_archive "$1" v3.9.4 acloudviewer-v3.9.4-docs.tar.gz stable
    make_restore_archive "$1" main-devel acloudviewer-ffff-docs.tar.gz old-dev
    make_restore_archive "$1" main-devel acloudviewer-0000-docs.tar.gz newest-dev
}

echo "═══ Case 13: inventory and known-asset transport failures stop restoration ═══"
for fault in list-error view-error-v3.9.4 download-error-v3.9.4 \
             view-error-main-devel download-error-main-devel; do
    d=$(new_restore_case "restore-$fault")
    valid_restore_archives "$d"
    touch "$d/$fault"
    run_restore "$d"
    check "$fault fails closed" "[ $restore_rc -ne 0 ]"
    check "$fault does not advertise a dev tree" "[ ! -e '$d/restored/dev' ]"
done

echo "═══ Case 14: successful empty asset inventories are not failures ═══"
d=$(new_restore_case restore-no-assets)
printf '%s\n' '{"assets":[]}' > "$d/v3.9.4.json"
printf '%s\n' '{"assets":[{"name":"downloads_data.json"}]}' > "$d/main-devel.json"
run_restore "$d"
check "releases with no matching asset may continue" "[ $restore_rc -eq 0 ]"
check "no matching asset means no download attempt" "! grep -q 'release download' '$d/gh-calls.log'"
check "absent assets create no fake version directories" "[ ! -e '$d/restored/dev' ] && [ ! -e '$d/restored/v3.9.4' ]"

echo "═══ Case 15: both stable and dev must have root entry pages ═══"
for invalid in stable dev; do
    d=$(new_restore_case "restore-invalid-$invalid")
    valid_restore_archives "$d"
    if [ "$invalid" = stable ]; then
        make_restore_archive "$d" v3.9.4 acloudviewer-v3.9.4-docs.tar.gz ''
        target=v3.9.4
    else
        make_restore_archive "$d" main-devel acloudviewer-0000-docs.tar.gz ''
        target=dev
    fi
    run_restore "$d"
    check "invalid $invalid archive blocks publication" "[ $restore_rc -ne 0 ]"
    check "invalid $invalid tree is never moved into destination" "[ ! -e '$d/restored/$target' ]"
    check "invalid $invalid failure names root index.html" "grep -q 'root index.html' '$d/restore.log'"
    check "invalid $invalid staging is cleaned" "! compgen -G '$d/restored/.restore.*' >/dev/null"
done

echo "═══ Case 16: restore history and newest dev by creation time ═══"
d=$(new_restore_case restore-valid)
valid_restore_archives "$d"
run_restore "$d"
check "known stable and dev assets restore successfully" "[ $restore_rc -eq 0 ]"
check "stable archive contents are retained" "grep -qx stable '$d/restored/v3.9.4/index.html'"
check "newest createdAt wins instead of SHA ordering" "grep -qx newest-dev '$d/restored/dev/index.html'"
check "only the selected dev asset is downloaded" "! grep -q 'release download main-devel --pattern acloudviewer-ffff' '$d/gh-calls.log'"
check "current release is skipped before asset lookup" "! grep -q 'release view v3.9.5' '$d/gh-calls.log' && [ ! -e '$d/restored/v3.9.5' ]"
check "draft releases are ignored" "! grep -q 'release view v9.9.9' '$d/gh-calls.log'"
check "unrelated release assets are not downloaded" "! grep -q 'release download.*downloads_data.json' '$d/gh-calls.log'"
check "successful restoration removes temporary staging" "! compgen -G '$d/restored/.restore.*' >/dev/null"

echo "═══ Case 17: malformed inventories are not empty inventories ═══"
for malformed in releases assets duplicate; do
    d=$(new_restore_case "restore-bad-$malformed")
    case "$malformed" in
        releases) printf '%s\n' '[{"tagName":"v3.9.4"}]' > "$d/releases.json" ;;
        assets) printf '%s\n' '{"assets":null}' > "$d/v3.9.4.json" ;;
        duplicate) printf '%s\n' '{"assets":[{"name":"acloudviewer-v3.9.4-docs.tar.gz"},{"name":"acloudviewer-v3.9.4-docs.tar.gz"}]}' > "$d/v3.9.4.json" ;;
    esac
    run_restore "$d"
    check "malformed $malformed fails closed" "[ $restore_rc -ne 0 ]"
    check "malformed $malformed cannot restore a guessed tree" "[ ! -e '$d/restored/v3.9.4' ]"
done

echo "═══ Case 18: ambiguous or unusable dev creation times are rejected ═══"
for time_fault in missing invalid tie; do
    d=$(new_restore_case "restore-time-$time_fault")
    printf '%s\n' '{"assets":[]}' > "$d/v3.9.4.json"
    case "$time_fault" in
        missing) jq 'del(.assets[0].createdAt)' "$d/main-devel.json" > "$d/replacement.json" ;;
        invalid) jq '.assets[0].createdAt="not-a-date"' "$d/main-devel.json" > "$d/replacement.json" ;;
        tie) jq '.assets[0].createdAt=.assets[2].createdAt' "$d/main-devel.json" > "$d/replacement.json" ;;
    esac
    cp "$d/replacement.json" "$d/main-devel.json"
    run_restore "$d"
    check "dev timestamp $time_fault fails explicitly" "[ $restore_rc -ne 0 ]"
    check "dev timestamp $time_fault is not guessed from a name" "! grep -q 'release download main-devel' '$d/gh-calls.log'"
done

echo "═══ Case 19: successful transport still requires the declared tar ═══"
for transport in omitted corrupt; do
    d=$(new_restore_case "restore-$transport")
    valid_restore_archives "$d"
    if [ "$transport" = omitted ]; then
        touch "$d/download-omitted"
    else
        echo 'not a tarball' > "$d/assets/v3.9.4/acloudviewer-v3.9.4-docs.tar.gz"
    fi
    run_restore "$d"
    check "$transport archive cannot publish a version" "[ $restore_rc -ne 0 ] && [ ! -e '$d/restored/v3.9.4' ]"
done

echo "═══ Case 20: optional dev is controlled by event, not guessed ═══"
d=$(new_restore_case restore-no-dev-event)
valid_restore_archives "$d"
touch "$d/view-error-main-devel"
run_restore "$d" false
check "main builds do not restore redundant dev assets" "[ $restore_rc -eq 0 ] && [ ! -e '$d/restored/dev' ]"
check "disabled dev restore never inspects main-devel" "! grep -q 'release view main-devel' '$d/gh-calls.log'"
d=$(new_restore_case restore-no-dev-release)
printf '%s\n' '[{"tagName":"v3.9.4","isDraft":false}]' > "$d/releases.json"
valid_restore_archives "$d"
run_restore "$d"
check "a genuinely missing main-devel release is allowed" "[ $restore_rc -eq 0 ] && [ -f '$d/restored/v3.9.4/index.html' ] && [ ! -e '$d/restored/dev' ]"
d=$(new_restore_case restore-empty-release-list)
echo '[]' > "$d/releases.json"
run_restore "$d"
check "successful empty release list is allowed" "[ $restore_rc -eq 0 ]"

echo "═══ Case 21: existing historical restore targets are never replaced ═══"
d=$(new_restore_case restore-existing)
valid_restore_archives "$d"
mkdir -p "$d/restored/v3.9.4"
echo preserved > "$d/restored/v3.9.4/index.html"
run_restore "$d"
check "existing restored history is not overwritten" "[ $restore_rc -ne 0 ] && grep -qx preserved '$d/restored/v3.9.4/index.html'"

echo "═══ Case 22: a full release page is not a complete inventory ═══"
d=$(new_restore_case restore-truncated-list)
python3 - "$d/releases.json" <<'PY'
import json
import sys
with open(sys.argv[1], 'w', encoding='utf-8') as stream:
    json.dump([{'tagName': 'draft-' + str(i), 'isDraft': True}
               for i in range(1000)], stream)
PY
run_restore "$d"
check "inventory cap counts drafts too and fails closed" "[ $restore_rc -ne 0 ]"
check "inventory truncation cannot look like no stable releases" "grep -q '1000-release limit' '$d/restore.log'"

echo
echo "──────────────────────────────"
printf 'PASS: %d   FAIL: %d\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
