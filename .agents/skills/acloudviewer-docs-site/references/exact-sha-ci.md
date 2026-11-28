# Documentation CI: evidence and exact SHA

## Diagnose and archive the failed run first

Use the target repository, workflow, run ID and full expected SHA. Example API commands below use `REPO=Asher-1/ACloudViewer`; set variables for the actual case and choose evidence destinations that will not be overwritten by the follow-up. Check the run and job metadata before logs; case-sensitive step names are read from GitHub. Distinguish `Build documentation Docker image` from `Verify live deployment` and the probe/guard jobs.

```bash
set -eo pipefail
old="$EVIDENCE/ci-run-$RUN_ID"
mkdir -p "$old"
gh api "repos/$REPO/actions/runs/$RUN_ID" > "$old/run.json"
test "$(jq -r .head_sha "$old/run.json")" = "$EXPECTED_SHA"
gh api "repos/$REPO/actions/runs/$RUN_ID/jobs" > "$old/jobs.json"
# Check exactly one expected build job in the actual attempt before selecting its ID.
jq -r '.jobs[] | [.id,.name,.status,.conclusion] | @tsv' "$old/jobs.json"
gh api "repos/$REPO/actions/jobs/$JOB_ID/logs" > "$old/build.log"
test -s "$old/build.log"
sha256sum "$old/build.log"
```

Capture dispatch receipt if available, monitor and result metadata, relevant step statuses and the **full** failed job log, not just grep. If there were manual reruns, select the job ID from the appropriate attempt and record attempt as well as run ID. Check log contents/format, not just nonzero size (an error response is not a log). If the authenticated job logs endpoint redirects and you need a fallback, fetch its signed URL without forwarding GitHub credentials to the redirected host; do not store or print tokens/signed URLs. `BlobNotFound` while the job is active means logs may not be ready. If the archive command returned nonzero, independently check each expected file, SHA/run identity, byte sizes/checksums and missing parts, then report what was preserved. A `... | head` under pipefail caused exit 141 after copies completed; do not hide it with `|| true` or rerun mutations on that assumption. A full status check independent of the archive pipeline also avoids SIGPIPE.

Read first substantive failure, not the last Python unwind or the first matching CMake feature-probe `Failed`. Use `rg -n 'ModuleNotFoundError|Traceback|Warning, treated as error|CMake Error|fatal error|error:|Killed|No space left|Successfully executed|Failed to execute' "$old/build.log"`, then inspect context. Correlate phase counts, CMake config, layer and timestamps. A Qt `Connection timed out` at package installation or an OOM is not automatically caused by a docs patch. `apt` defaults to no retries; the Dockerfile used Acquire retries/timeouts, three tries with `--fix-missing`, then `dpkg -s qtbase5-dev` to assert actual installation, not a loop exit alone. Character-split Docker stderr can be reconstructed from the selected layer after removing `#<n> <time> ` prefixes; for notebooks prefer structured cell outputs. Compare last green/first red SHA if needed, e.g. use `gh api "repos/$REPO/actions/runs?per_page=60" --jq '.workflow_runs[] | select(.name=="Documentation") | [.created_at,.conclusion,.head_sha]'`. Do not push speculative fixes to distinguish infra from code.

Historical example: [run 44716154180](https://github.com/Asher-1/ACloudViewer/actions/runs/44716154180), SHA `8cfaf0c08071638d7a44232fcc18afc902280f7c`, passed guards and 28 notebooks before failing in parallel Sphinx's parent `recv` on `ml3d`. [Follow-up run 44729538026](https://github.com/Asher-1/ACloudViewer/actions/runs/44729538026) matched `7ee53d511a7b6a1f8870ebe9e04adc650f5633c3`; at the evidence cutoff the complete remote build was not known to have passed. These IDs are historical evidence, never dispatch parameters or current conclusions by default.

## Guard ladder and focused commit

Run `git diff --check`, Python compile checks (`python3 -m py_compile docs/make_docs.py` and touched Python files), shell syntax individually (`bash -n <script>`), and parse changed workflow YAML. From the repo root run `python3 docs/automation/scripts/check_notebook_headings.py` and `python3 docs/automation/scripts/check_workflow_limits.py`; GitHub silently invalidated a workflow near a 21000-character `run:` scalar (zero jobs, sometimes only an annotation). Then run affected no-compile guards, full strict builder/site checks per scope. Don't confuse a changed failure step with complete success. Existing guards also protect: zero generated API pages despite swallowed import errors; template exceptions leading to dangling Getting Started links; notebooks ignored under only `CI` rather than `GITHUB_ACTIONS`; downloads keyed as `download_links` instead of `download_data`; treating `version_metadata` array as an object instead of `map()`; fetching only 5 of 18 releases (limit moved to 100). Doxygen output check remains independently necessary.

Only after local verification and authorization, select exact paths. Example follow-up for the parent import issue:

```bash
files=(.github/workflows/documentation.yml docs/source/conf.py docs/automation/scripts/test_sphinx_ml_imports.py)
branch=$(git branch --show-current)
test -n "$branch"
test "$(git rev-parse HEAD)" = "$VERIFIED_BASE"
git diff --cached --binary > "$EVIDENCE/index-before.patch"
git diff -- HEAD -- "${files[@]}"
git diff --cached --name-only -- "${files[@]}"
# Stop if selected files contain unrelated changes; isolate/review before committing.
# Add the intended new file if untracked, not the whole index.
git add -- "${files[@]}"
git commit --only -m 'fix(docs): register ML root before parallel Sphinx workers' -- "${files[@]}"
sha=$(git rev-parse HEAD)
git show --format=fuller --name-status "$sha"
git diff --cached --binary > "$EVIDENCE/index-after.patch"
```

`git commit --only -- <paths>` uses working-tree contents of those paths; review them (including formerly untracked files) before commit. Bare `git commit` would include already-staged icons even after a limited `git add`; plain `git reset` clears unrelated staging and is also wrong. Verify only intended names in commit and the **unrelated** staged diff unchanged (whole patch can differ if some related changes were staged before); if not, stop before push. Push only to verified current feature branch on the authorized remote, never force. Example for `origin`:

```bash
git push --porcelain origin "HEAD:refs/heads/$branch"
remote_sha=$(git ls-remote origin "refs/heads/$branch" | cut -f1)
test "$remote_sha" = "$sha"
```

## Dispatch and observe once

Inspect the workflow's actual `workflow_dispatch` input names and types at that branch, confirm repository/branch and freshly verified remote SHA. The observed dispatch was for Documentation with `developer_build=ON`. A minimal authenticated dispatch pattern, **only if this key and value match the current workflow**, is:

```bash
gh workflow run documentation.yml --repo "$REPO" --ref "$branch" -f developer_build=ON
```

Record UTC dispatch time, repository/workflow, ref, requested SHA, inputs and receipt. One accepted request is not a run ID and branch ref resolution can race. If output is ambiguous, inspect runs before even considering a retry; do not dispatch again while looking up or monitoring. Match workflow, `workflow_dispatch`, branch, dispatch time window and full SHA; if more than one candidate still matches, disambiguate with run metadata/actor or report uncertainty. Never replace the target with the newest run of the branch or with a push-triggered run. Example read-only discovery:

```bash
gh api "repos/$REPO/actions/workflows/documentation.yml/runs?event=workflow_dispatch&per_page=60" \
  --jq ".workflow_runs[] | select(.head_sha == \"$sha\") | [.id,.html_url,.head_branch,.created_at,.status,.conclusion] | @tsv"
# After choosing MATCHED_ID by metadata:
gh api "repos/$REPO/actions/runs/$MATCHED_ID" > "$EVIDENCE/matched-run.json"
test "$(jq -r .head_sha "$EVIDENCE/matched-run.json")" = "$sha"
gh api "repos/$REPO/actions/runs/$MATCHED_ID/jobs" > "$EVIDENCE/matched-jobs.json"
```

Use notifications or one tracked read-only monitor for the chosen ID. Collect relevant jobs, no busy-polling or duplicate dispatch; `in_progress` is not green. Report local strict results, exact remote `docs-guards`, `build-docs`, and `deploy-docs` independently, with SHA, ID, URL and status/conclusion. `should_deploy` in the observed workflow is true for main or a release event only; a manual feature-branch run may build/upload a CI artifact without publishing Pages. Even a green feature run is not evidence of live deployment. If remote fails, preserve this attempt's metadata/logs and identify the new first actual error before proposing another focused fix. If local-only, stop before every mutating GitHub operation.
