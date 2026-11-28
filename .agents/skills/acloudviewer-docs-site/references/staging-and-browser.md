# Actual artifact staging and browser acceptance

## Contract and version inventory

One Pages artifact ships marketing at `/`, newest-doc redirect at `/documentation/`, dev and immutable `vX.Y.Z` under `/documentation/`, disk-derived `versions.json`, regenerated sitemap, and version-calibrated `404.html`. In this project these are served beneath `/ACloudViewer/`. `versions.json` shape consumed by `docs/script.js`:

```json
{"latest":"v3.9.4","generated_at":"…","versions":[{"id":"dev","label":"Development (main)","kind":"dev"},{"id":"v3.9.4","label":"v3.9.4","kind":"latest"}]}
```

`dev` first if present; then releases descending; kinds `dev|latest|archive`. The directory matcher is `^v[0-9]+\.[0-9]+\.[0-9]+$`: prerelease-like tags can fall out of manifest, sitemap and restoration; if prereleases become supported, change the common matching contract, not one probe. Build the expected version list from the independently validated archive inventory plus new output. Only releases with actual docs assets can be restored (historically 2 of 18 had assets: `main-devel` and `v3.9.4`; earlier tags such as `v3.9.3` require building that tag). Release archives live on that release as `acloudviewer-<tag>-docs.tar.gz`, dev archives on `main-devel` as `acloudviewer-<sha>-docs.tar.gz`; patterns must remain disjoint. `main-devel` retains newest dev; restore dev on release builds because Pages replaces the deployment. Probe dev only when present per that restore policy.

Website preparation resolves `@VERSION@` and `@LATEST_STABLE_VERSION@`: release tag → newest **published** release → `version.txt` → `dev`; version.txt gets bumped before release, so it must not take precedence. `@DOC_LATEST@` is expanded only in unified deployment, once the actual tree is known. Refuse release mode with missing `DOC_VERSION`, otherwise release docs can overwrite dev. Missing docs archive on a release that never shipped docs is a warning; an existing but corrupt/unrestorable archive is fatal, not grounds for silently deleting a version. Never create a published version dir before its staged extraction has a valid `index.html`. Replace destination subtree when redeploying, because `cp -r src/. dst/` leaves deleted files behind. Reuse the CI-resolved archive name when uploading; historically renaming then copying to the same name aborted release builds.

## Replay commands

Only start after strict Sphinx exit 0 with final HTML entry point. A failed Sphinx run can leave ~96k Doxygen files in the output with no index. Select fresh `STAGE` outside the source/mutated output; set `HTML` to verified HTML, `NEW_ID` to the tag or full SHA selected by the current workflow, and `HIST_ARCHIVE` to the real downloaded release asset. Use a new stage after each change; reuse unchanged verified HTML/archives when only packaging changes. The historical local runner `build_headless_docs/evidence/run_native_ci_deploy.sh` was scratch evidence, not a required repo script. **TODO per checkout:** copy the exact `prepare_website.sh` invocation/inputs and the archive entry layout from the current workflow and record them in the run report (the evidence excerpts do not show the complete prepare command). Reproduce local file preparation/merge steps only, not workflow upload or publishing steps.

```bash
repo=$(pwd)
test -f "$HTML/index.html"
test -f "$HTML/cpp_api/api/index.html"
test ! -e "$STAGE"
mkdir -p "$STAGE/archives" "$STAGE/docs-output" "$STAGE/preserved-versions/documentation"
cp -a "$HTML/." "$STAGE/docs-output/"
# For a flat HTML-root archive; match the current workflow if it wraps the root.
new_archive="$STAGE/archives/acloudviewer-$NEW_ID-docs.tar.gz"
tar -C "$STAGE/docs-output" -czf "$new_archive" .
sha256sum "$new_archive" "$HIST_ARCHIVE"
tar -tzf "$HIST_ARCHIVE" > "$STAGE/historical-members.txt"
```

Use the workflow's restore layout: inspect members before extracting real assets, reject escaping paths, extract to a temporary directory, find the expected HTML root, require `index.html`, and move it intact into `"$STAGE/preserved-versions/documentation/$TAG"`. Repeat for each archive; take CSS/HTML checksums before merge. Run the workflow's website preparation against a staging copy of the actual `docs/` inputs and release metadata so `website-deploy/` is under `STAGE`; verify placeholders, downloads data, and no build-only roots. Then run merge in staging with the captured workflow values (`SITE_URL` for the project prefix and no upstream upload):

```bash
(
  cd "$STAGE"
  test -f docs-output/index.html
  test -d website-deploy
  test -d preserved-versions/documentation
  IS_RELEASE="$IS_RELEASE" RELEASE_VERSION="$RELEASE_VERSION" \
    DOC_VERSION="$DOC_VERSION" SITE_URL="$SITE_URL" \
    bash "$repo/docs/automation/scripts/deploy_unified_docs.sh"
)
```

Keep exit codes, logs and test summaries for each phase. The merge writes `unified-deploy/`. Contract guards from the repository root:

```bash
bash -n docs/automation/scripts/prepare_website.sh
bash -n docs/automation/scripts/deploy_unified_docs.sh
bash docs/automation/scripts/test_prepare_website.sh
bash docs/automation/scripts/test_deploy_unified_docs.sh
python3 docs/automation/scripts/verify_docs_site.py
node --check docs/source/_static/version_switch.js
```

Run test scripts in the proper shell/Python (Playwright browser installed in the latter); compile venv and browser env need not be the same. Verify actual browser interpreter if Playwright is missing. A passed fixture or shell syntax check does not establish actual artifact parity.

## Size gate and history preservation

If staging is too big, first run `du -h --max-depth=1 "$STAGE/website-deploy"`, `du -h --max-depth=1 "$HTML"`, `du -h --max-depth=1 docs/doxygen`, and `du -sh --apparent-size "$STAGE/unified-deploy"` if present. Record tar/gzip byte counts (`wc -c < "$new_archive"` and merge report; or `set -o pipefail; tar -C "$STAGE/unified-deploy" -czf - . | wc -c`). The previous site-root copy included ~1.4 GB of Doxygen intermediates and ~236 MB of doctrees in addition to versioned rendered docs; the new standalone docs tar was ~491 MiB in that replay. Exclude `doxygen` and `doctrees` from website **root** alongside `_out`, `_build` and bytecode using rsync exclusions and equivalent removal under the staging-only cp destination. Test both actual branches, with fixtures for each excluded directory and retained authored assets, deliberately omitting rsync from the fallback PATH while keeping other required tools (including `cat`). Validate C++ HTML under `documentation/<version>/cpp_api/api/` still exists and real archives remain. Do not raise the limit, remove versions or rebuild C++ just to fix the copy.

The workflow gzip budget is 1 GiB (`1073741824` bytes or `1048576` KiB); check compressed data separately from raw site/disk limits and report both. The gzip proxy was intentionally conservative compared with Pages transport compression; a passing proxy is not a live publication guarantee. Historical measurements from the Doxygen sizing investigation (not current expectations): SVG instead of PNG 1947→921 MB raw and 1177→145 MB archived, also avoids 32767 px truncation; `INCLUDED_BY_GRAPH=NO` ~197 MB saved; exclude `qSIBR/SIBR` ~72 MB; `SOURCE_BROWSER=NO` ~361 MB with 2303 header pages kept via `VERBATIM_HEADERS`; `HIDE_UNDOC_MEMBERS` saved only 15 MB and wasn't worth the loss. SVG shares textual compression across graphs. A tuned per-version sample was ~769 MB on disk / 587 MB content / 145 MB zip-equivalent gzip archive; the older v3.9.4-era archive was ~243 MB. For growth, consider re-archiving from the release source with approved size changes rather than dropping history. Keep the archive inventory and report measured bytes, not these examples.

## Historical switcher, CSS and 404

Deployment propagates current `_static/version_switch.js` to archives. Preserve historical HTML verbatim: retain its callable API (at least `window.ACloudViewerVersionSwitcher.updateSidebarSelector`) and support its existing select IDs (`#docs-version-select-sidebar`, `#docs-version-select-main`) along with `select[data-docs-version]`. Inspect the historical calls, arguments and DOM before adjusting the shared JS. Add a legacy template fixture exercising guest HTML calling the new API and selecting a version; actual historical page must also pass. Do not copy dev HTML over archives, or rewrite archive structure to suit new JS. For shared styles, transplant `_static/css/furo_overrides.css` only when missing from a version that needs it; never overwrite a pre-existing archive stylesheet. In merge contracts include one old archive lacking it and one containing its own CSS; check filled file equals shared source, existing file equals original, and compare archived HTML before/after.

In `docs/404.html`, omit the initial relative logo `src`; it triggers a fetch before JS can correct it. Resolve the prefix from the document path's `/documentation/` boundary, not the current nested 404 directory, then assign `images/ACloudViewer_logo_horizontal.png` beneath that prefix. Check root/non-doc 404 behavior for the configured site prefix too. Preserve the old flat deep route's suffix, query string and fragment when redirecting to latest; already-versioned misses must not be redirected in a loop.

## Browser on the actual tree

Use a Pages-like local server rooted at `unified-deploy/` but mounted under `/ACloudViewer/`; directory index and original-URL custom 404 response must work. Do not use a fixture-serving entry point as if it targets real output, or test only at `/`. The observed run used a scratch Playwright harness (`verify_actual_site.py`) taking the actual stage and independent versions; **TODO if absent:** inspect the current verifier for real-artifact support or implement the following checklist in a local harness and record its actual command, browser executable, mount, version inventory and report path. Never claim it ran just because a standalone fixture script exists.

1. Check artifact entry files, expected manifest IDs/types/order/latest, sitemap and served entry/redirect targets. Compare archive vs dev files to select common tutorial and API deep paths before launching; observed real v3.9.4 had `tutorial/geometry/surface_reconstruction.html`, `python_api/cloudViewer.geometry.html`, and `cpp_api.html`; no `python_api/index.html`.
2. In actual browser inspect selectors and available versions on dev and at least one old-template page that calls the legacy API. Drive switches both ways on common deep pages and assert destination path, selected version and successful page load. Test missing-page fallback separately; it is not evidence of deep-path preservation.
3. Exercise nested legacy missing routes with query/fragment; check final versioned URL, same query/fragment, no loop for already-versioned misses (both dev and historical). Check images at nested 404s even when navigation succeeds. Observe requests from the start of navigation, not only final DOM; detect even transient wrong logo fetch.
4. Collect local CSS/JS/image HTTP failures and transport `requestfailed` events, and all `pageerror` messages. Intentional page-level 404s are routes under test, not resource 404s. Compare archived HTML and custom style digests plus overlay missing stylesheet; verify loaded CSS/JS/images in both new and archived pages. External network failures should be separated, not silently renamed as local successes.
5. Write structured PASS/FAIL report with URL, expected/observed values, local resource and uncaught JS errors. Use exit status **and** report; final local run measured 106/0 actual checks, zero local resource errors and JS exceptions, 926587 KiB compressed (~905 MiB) with dev+v3.9.4; fixture counts were prepare 22/22, deploy 39/39, browser 32/32. Report current counts rather than hard-coding those. If deployment is pending behind strict Sphinx, say not executed.
