# Isolated integration, maintenance and capacity

## Isolated integration

Record original branch, full HEAD, `git status --short --branch` and staged binary diff outside the checkout. Verify the current remote main (`git ls-remote origin refs/heads/main`); if needed fetch main into local remote-tracking refs (read-only against the remote); compare fetched SHA to remote, and stop/recheck on a race. Record `BASE` as that complete verified commit, reject a moving/stale base; use `git worktree add --detach "$WT" "$BASE"` with a new path outside the original tree. Verify `git -C "$WT" rev-parse HEAD` and clean status. Never checkout, reset or abort in the user's tree. If a cherry-pick/rebase trial conflicts on a prerequisite, save its status/diff then use the corresponding `git -C "$WT" cherry-pick --abort` or `git -C "$WT" rebase --abort` **only if that operation is active there**. Check HEAD/status after abort. No whole mixed commit (including icons), and no wholesale replacement of the main version of conflicting scripts.

Read the source commits with `git show --stat` and inspect `BASE` versions, prerequisite hunks and current callers. Review cross-directory dependencies one by one: workflow invocations/env, Python/shell tests, Docker/helper, sources and static inputs; each included path needs an owner/call-site or contract reason, each deliberately excluded hunk a rationale. Reimplement the intended behavior atop current main where existing code diverges. In the detached tree review the exact edits (including new files). The following exports a candidate without touching either index; `paths` is the reviewed Bash array of exact repo-relative inputs, `EVIDENCE` is an absolute fresh directory, and the disposable index must not yet exist:

```bash
set -eo pipefail
idx="$EVIDENCE/review.index"
patch="$EVIDENCE/docs.patch"
test ! -e "$idx" && test ! -e "$patch"
export GIT_INDEX_FILE="$idx"
git -C "$WT" read-tree "$BASE"
git -C "$WT" add -A -- "${paths[@]}"
git -C "$WT" diff --cached --check "$BASE" -- "${paths[@]}"
git -C "$WT" diff --cached --name-status "$BASE" > "$EVIDENCE/patch-paths.txt"
git -C "$WT" diff --cached --binary --full-index "$BASE" -- "${paths[@]}" > "$patch"
test -s "$patch"
sha256sum "$patch" > "$EVIDENCE/patch.sha256"
git -C "$WT" apply --cached --reverse --check "$patch"
git -C "$WT" apply --cached --reverse "$patch"
git -C "$WT" diff --cached --exit-code "$BASE"
sha256sum -c "$EVIDENCE/patch.sha256"
git -C "$WT" apply --cached --check "$patch"
unset GIT_INDEX_FILE
```

Check allowlist against the complete name list; compare patch with both tip and main dependency contracts. To test application on a clean base, use another detached worktree at `BASE`: run `git apply --check`, `git apply` and `git apply --reverse --check` there with the hashed patch. Don't apply forward over existing edits. Record clean starting state, base, patch SHA, patch paths and final diff. Recheck original staging and HEAD, observed remote refs and write-operation ledger; local fixture commands cannot push, dispatch, or alter release assets/Pages.

## Offline maintenance matrix

Confirm [workflow](<.github/workflows/documentation.yml>) calls and current tests before invoking them; use the same verified `DOC_PY` and PATH from the repo root. Record a separate exit code and test log for each command and the observed inner path status (a passing test can assert an inner nonzero). Verify script syntax individually and YAML parsing, `git diff --check`, notebook/workflow guards when touched. Known suites:

```bash
export PATH="$(dirname "$DOC_PY"):$PATH"
bash docs/automation/scripts/test_prepare_website.sh
bash docs/automation/scripts/test_deploy_unified_docs.sh
"$DOC_PY" docs/automation/scripts/test_scan_releases.py
"$DOC_PY" docs/automation/scripts/test_docs_parallelism.py
"$DOC_PY" docs/automation/scripts/test_docstring_pipeline.py
"$DOC_PY" docs/automation/scripts/verify_docs_site.py
```

Inspect the current guards for these cases; add a minimal owner-level fixture where missing. **TODO per checkout:** copy exact release helper commands, tag variables and fixture selectors from the workflow/tests, rather than guessing endpoints or flags. Use temp trees, deterministic clocks and offline API/upload/download substitutes; block live mutations and inspect the calls.

- Successful valid empty release/asset list is allowed; release-list/API failure, invalid JSON/shape, download failure for a known asset, corrupt archive and missing archive entry are not empty. For each, assert nonzero from the owner, no loss of existing versions, and no publish call. A release that never had an asset is distinct from failure to retrieve an advertised asset.
- No `index.html` for dev means no dev in publishable inventory. Release builds skip restoring the current release's old duplicate; restore eligible dev and other releases exactly once. Validate release tag format and all required environment before extraction or placement. Use the actual stable tag contract, not shell text interpolation or an unvalidated event value. For archive source fetch, limit CI to exact needed tag refs; the merge script reads refs from its own checkout, not the staging CWD, and never silently fetches or substitutes HEAD.
- Dev backup replacement: unique SHA-based asset name, no forced overwrite. Use local fake receipts to verify failures preserve the last good backup; only after successful upload and read-back verification of new asset identity/integrity may cleanup target old backups, never the new asset or unrelated attachments. Record upload/verify/delete ordering and pre/post inventories.
- Downloads: inspect the current `download_data` and `version_metadata` shapes and actual date fields; use a `version_metadata` array as an array. Assert displayed version count and cutoff against UTC with boundary cases, include later pages in fixtures, follow actual pagination to exhaustion, and treat malformed/API errors as failures rather than an empty result. Where live read-only verification is requested, check displayed download URLs against the current release assets, not just link counts.
- Doxygen exit 0 without actual `cpp_api/api/index.html` must fail, as must missing/bad required formula PNGs. Retain Doxygen and aggregate-parallel guard output; separate fixture from real Doxygen evidence.

## Artifacts and capacity

1. Use [exact run lookup](<exact-sha-ci.md>) and the run artifact inventory (`gh api --paginate "repos/$REPO/actions/runs/$RUN_ID/artifacts?per_page=100"`). Select the exact workflow-provided artifact by name and ID and confirm run association/head SHA, expiration, size and available digest. Download the specific ID's ZIP via the authenticated Actions artifacts endpoint (avoid sending credentials to a redirected host or saving signed URLs). Hash actual ZIP bytes with `sha256sum`, compare to the Actions digest or independent trusted checksum; if unavailable, label the hash as a local fingerprint, not verified remote integrity. Preserve metadata, attempt, URL, member list, exact ZIP size/hash and inner docs archive size/hash if wrapped. Inspect member paths, links, duplicate/escaping paths and root before safe extraction to temp. Verify HTML root/index plus C++ API index, expected source/API/tutorial coverage and actual notebook entries/links against a known-good CI artifact and current workflow. A genuine CI artifact may ship tutorial HTML with GitHub notebook source links but no `.ipynb`; assert the actual contract and source links, not native file counts.
2. For every version record version ID, provenance/full source SHA (resolve historical tag to commit or report unknown, don't substitute current HEAD), asset identity/hash, member root and entry. Hash archive inputs, existing files and developer tree before/after; count added/changed/deleted files, with any intended overlay (e.g. switcher) explicitly separated from source repair. Assemble website and all available published history using [staging replay](<staging-and-browser.md>) in fresh main and release stages with their actual env and manifests. Save separate evidence for each real tree or label unproduced tree pending. Use the probe even on a complete staged tree with an expected capacity refusal, without publishing. Capture refusal exit and log line, no publish call, unchanged dev source/tree and archives, and the preserved historical entries. Do not equate exit 1 alone with capacity refusal.
3. Record byte counts separately: actual Actions ZIP/inner tar.gz/valid release archives, uncompressed extracted file content, per-build Doxygen/Sphinx size, and final merged **published file bytes** with subtotals for website, dev and each archived version. Sum logical file lengths in the effective upload tree, not allocated disk blocks; detect symlinks/hardlink or archive transport differences explicitly. Break down by directory and extension (`.html`, `.svg`, `.png`, etc.), inspect the rendered Doxyfile actually used and archive exclusion paths, and label any hypothetical change/savings as an estimate until measured. Historical sizing examples in [staging](<staging-and-browser.md#size-gate-and-history-preservation>) are not current quotas. Report verified compressed gate (`1073741824` bytes in the cited workflow) separately from published content: consult and quote the current [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits) and applicable upload contract, cite units and date, and report measured bytes, margin/overage and threshold for each. Do not infer 1 GB equals 1 GiB. A gzip pass when actual content exceeds the applicable limit is **release blocked**; do not drop historical pages, disable checks or raise limits to claim a pass.
