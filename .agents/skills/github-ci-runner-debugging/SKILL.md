---
name: github-ci-runner-debugging
description: It diagnoses GitHub Actions runner and dependency failures, reconciles exact-SHA PR checks, and maintains a single read-only CI observer when users ask to make CI green, check merge readiness, recover monitoring, audit duplicate triggers, or verify a fix on new main.
---

# GitHub CI Runner Debugging

Own the evidence acquisition, root-cause, exact-commit reconciliation and observer handoff protocol. Read the repository CI rules for its current platform matrix and known signatures; do not freeze an incident's workflow counts into policy. Documentation-site behavior belongs to the docs-site skill; this skill tracks its CI and deployment evidence without treating either as the other.

## Unified C++ Test Evidence

For this repository, CI invokes the root `tests/Run.cmake` interface; individual
modules own registration through `cloudviewer_add_test` or existing forwarding
helpers. Adding a module must not require a CI module-list edit. Do not introduce
another test runner, dependency list, profile policy, or parallelism policy in a
workflow or compatibility target.

Consolidating a test entry does not authorize changing test implementations,
assertions, source enablement, case names, dynamic downloads or fixture inputs.
Preserve these contracts, including CVViewer's existing data acquisition; report
pre-existing failures separately instead of changing cases to obtain green CI.

A green selected profile proves only that profile. Record the exact profile,
compiled targets, CTest selections, GoogleTest case counts, skips and disabled
cases separately. Restoring a source line to CMake proves neither compilation
nor execution. `PROFILE=all`, asset-backed inference, GPU execution, graphics,
and multi-gigabyte reductions need their own observed results. Linking GoogleTest
while a handwritten main owns all assertions is not a completed unit migration;
retain custom protocol entry points only for audited integration consumers.

Audit duplicate registrations by executable plus arguments, environment, working
directory and actual device/input selection. Ignore missing command fields for
unbuilt targets rather than grouping all such rows as duplicates. In a CPU-only
configuration, CPU/backend matrix aliases may resolve to the same command; omit
the redundant backend registration while preserving non-CPU matrix coverage.

## Steps

1. **Fix scope, authorization and identity first.** Record repository, PR/branch, full 40-character target SHA, UTC sampling time and requested action. Read-only means no commits, remote ref changes, merge, dispatch, rerun, cancellation or workflow edits. Permission to diagnose is not permission to retry or merge. Record explicit authorization and its gates separately from observed results. Resolve the current directory with `pwd`; do not infer it from an implementation checkout. Read the applicable repository CI and dependency-owner rules before changing code.

2. **Get remote PR identity through REST, including CLI compatibility fallback.** Set `REPO=owner/repo`, `PR` to the requested number and `EVIDENCE` to an agreed writable evidence directory; use the available `gh` executable, not a historical machine-specific path. Example read-only calls:
   ```bash
   mkdir -p "$EVIDENCE"
   date -u +%FT%TZ > "$EVIDENCE/sampled-at.txt"
   gh api "repos/$REPO/pulls/$PR" > "$EVIDENCE/pr.json"
   gh api "repos/$REPO/pulls/$PR" --jq '{number,state,merged,mergeable,mergeable_state,head_sha:.head.sha,head_ref:.head.ref,base_sha:.base.sha,base_ref:.base.ref,merge_commit_sha,html_url}'
   gh api "repos/$REPO/branches/main" --jq '{name,sha:.commit.sha}'
   ```
   If `gh pr view --json baseRefOid` is unsupported, use this PR REST response for **both** head and base; do not omit base or substitute an older local ref. Record an unknown/unstable mergeability result as unknown. Re-query before a write. Retain separate identities for PR head, tested synthetic merge commit, PR base and latest main.

3. **Reconcile runs, jobs and commit checks for the exact SHA with pagination.** Set `SHA` to the verified full target; the following are GETs. Save raw responses with sample time and endpoint as source:
   ```bash
   gh api --paginate "repos/$REPO/actions/runs?head_sha=$SHA&per_page=100" > "$EVIDENCE/runs-pages.json"
   gh api --paginate "repos/$REPO/commits/$SHA/check-runs?filter=all&per_page=100" > "$EVIDENCE/checks-pages.json"
   gh api --paginate "repos/$REPO/actions/runs/$RUN/jobs?per_page=100" > "$EVIDENCE/jobs-$RUN-pages.json"
   gh api "repos/$REPO/actions/runs/$RUN" --jq '{id,head_sha,head_branch,event,path,run_attempt,status,conclusion,created_at,updated_at,html_url}'
   gh api --paginate "repos/$REPO/actions/runs/$RUN/jobs?per_page=100" --jq '.jobs[]|{id,name,status,conclusion,started_at,completed_at,html_url,active:[.steps[]?|select(.status=="in_progress")|{name,number,started_at}],failed:[.steps[]?|select(.conclusion=="failure")|{name,number}]}'
   ```
   Pagination output may contain successive JSON documents; parse each page rather than treating concatenated pages as one JSON object. For a historical attempt, use `repos/$REPO/actions/runs/$RUN/attempts/$ATTEMPT/jobs?per_page=100`. Reconcile by full SHA, PR association, branch, workflow path, event and attempt, not workflow display name alone. PR workflows may test a synthetic merge SHA: identify it explicitly and retain its relationship to the PR head. Include independent check producers such as CodeQL, not just jobs visible in one run list.

   Inspect workflow definitions at the target revision, their `on` events, branch/path filters, job `if`, dependencies, matrix and concurrency. Build the expected set for this PR/event, then classify observed checks as **success / justified conditional skip / failure / cancelled / queued or running / expected but not observed**. Report dynamically computed counts with the sampled set and denominator. `skipped`, cancellation, missing checks and nonterminal `conclusion: null` are not success. For every unfinished item report run ID, attempt, job ID, URL, current actual step and step start time; a queued job has no active step. Example: docs-guards success plus build-docs queued means Documentation unfinished and Pages pending, not published.

4. **Acquire the smallest decisive failure evidence, then full logs.** Read run metadata → jobs/steps → annotations → complete failed-job log. Map the job to its actual check-run ID from metadata/URL before querying annotations; do not assume equal numeric IDs.
   ```bash
   gh api --paginate "repos/$REPO/check-runs/$CHECK/annotations?per_page=100" > "$EVIDENCE/annotations-$CHECK-pages.json"
   gh api "repos/$REPO/actions/jobs/$JOB/logs" > "$EVIDENCE/job-$JOB.log"
   ```
   Record full SHA, run/attempt/job/check IDs, failing step, first substantive error with surrounding log lines and source URL. If logs are not yet available, report that limitation and retain annotations; silence is not a pass. Check **every** shell result's `[exit code: N]`, HTTP status and final background-job exit; an outer tool `ok` is not command success.

   Let `gh` use its authenticated session. If a custom downloader needs a token, obtain it inside the consuming process, e.g. Python `subprocess.check_output(['gh','auth','token'], text=True).strip()`, and keep it only in memory for authorized API requests. Never print/store the token, dump credential files, put it in evidence, or forward authorization headers to arbitrary redirected hosts. A local download/archive interruption is local evidence loss, not proof of remote CI failure.

5. **Classify the current attempt, not a previous explanation.** Compare last green and current HEAD; inspect only signatures still alive at the target commit. All configurations failing at one phase suggests a shared contract; one configuration suggests platform/toolchain, but neither proves root cause. One cancelled sibling is **not automatically concurrency cancellation**. For cancelled runs, especially zero-job runs, inspect trigger, timestamps, concurrency policy, newer runs, annotations and whether any runner executed code. A hosted-runner acquisition annotation supports capacity failure before build, not compilation failure.

   When a later attempt entered `Config and build Wheel`, reacquire that attempt's complete log. Repeated `connectivity_check` DNS code 6 is a configuration-stage dependency preflight failure, not the old runner-capacity issue, a compiler error, documentation failure or an unrelated model download failure. Distinguish a probe URL from actual required payload URLs. DNS failure alone does not establish permanent endpoint retirement or payload unavailability. If an assumed local build directory is absent, record the gap and search source/configure metadata; do not claim to have inspected nonexistent child outputs.

6. **Reproduce and repair only the owning mechanism, when authorized.** Use the smallest experiment, then actual generated build commands/cache and relinked consumers. Apply the [build and forensic principles](<references/build-forensics.md>) and the specialized [dependency validation protocol](<references/dependency-validation.md>) for firmware migrations or transient external downloads. For ACloudViewer RealSense, inspect the pinned upstream recipe, local probe entry and ExternalProject owner; unify probe and real firmware base through typed `CMAKE_CACHE_ARGS`, retaining version, all declared hashes and firmware/TM2. Verify every declared real payload, child cache and wrong-hash negative control before claiming local validation. Do not fix CI by disabling required features, suppressing errors or weakening checksum/TLS validation. `git diff --check` and a minimal diff are necessary, not sufficient. Report local checks, target-platform CI and publication separately.

7. **Retry only under explicit scope, and reconcile ambiguous writes before another POST.** If authorized to retry only unexecuted/failed work, retain annotations and initial failure evidence. Supported endpoints for that scope are:
   ```bash
   gh api -X POST "repos/$REPO/actions/jobs/$JOB/rerun"
   # Only if the first request demonstrably did not take effect and scope permits:
   gh api -X POST "repos/$REPO/actions/runs/$RUN/rerun-failed-jobs"
   ```
   Capture response, stderr, HTTP error and exit code as the retry receipt. A 502 does not prove rejection or success. **Before fallback**, GET run `run_attempt/status/conclusion` and paginated jobs, compare with the pre-request sample and check for queued/new work. A queued state may precede incrementing `run_attempt`. If accepted or still ambiguous, do not duplicate the write; continue read-only verification. Only after confirming no effect use failed-only fallback within authorization, then verify the new attempt and actual job acquisition. Never rerun successful platforms or claim eventual success from a queued response.

8. **Merge only with authorization and the applicable gate; preserve independent merge evidence.** Normally all expected checks must be reconciled and terminal acceptable before merging. If the user explicitly changes the order to direct main verification, record that authorization, all unfinished/failed checks and residual risks; it is not evidence of a green branch and does not override repository protection. Local preconditions imposed by the task (such as firmware hash/cache validation) still apply.

   Immediately before merging, re-fetch REST PR head/base, latest remote main, current failures and the full diff/file scope; ensure the tested head and approved scope have not changed. Use the repository's supported authorized merge method with head-match protection; do not invent flags or bypass protections. **TODO if unavailable:** identify that supported method before writing. Save PR URL/number, authorization, pre-merge head and main, validation report, command/API receipt and actual resulting main SHA. A merge receipt is separate from CI acceptance.

   If an accepted test merge commit exists, compare its source tree to the new main:
   ```bash
   gh api "repos/$REPO/git/commits/$TEST_MERGE_SHA" --jq .tree.sha
   gh api "repos/$REPO/git/commits/$NEW_MAIN_SHA" --jq .tree.sha
   ```
   Report **both complete commit SHAs and both tree SHAs**. Equal trees establish identical tracked source content, not identical commit/event/environment or a passed main deployment. Unequal trees require diff and revalidation; do not transfer acceptance blindly. Switch all Actions queries to `head_sha=$NEW_MAIN_SHA`. Old branch/old main results remain historical evidence, not new-main success.

9. **Maintain exactly one real read-only observer and recover failures without duplicate work.** Reuse the existing observer's timestamped samples; use its saved background ID plus actual process/job status to verify liveness (for a known PID, `ps -p "$PID" -o pid,etime,args`). A saved ID, old message or subagent name is not proof of a live process. If absent, take one immediate REST snapshot before recovery. Track each background ID you start; collect relevant outputs via `job_output`, and stop obsolete local watchers via `job_kill` (not GitHub cancellation). Do not busy-poll or sleep in the foreground. A managed background loop may sample roughly every 600 seconds, drill into failures at the next observation, and avoid redundant API/log/browser work.

   Before replacement, confirm the old observer has exited or stop it and confirm termination. Hand the sole successor repository, PR, full target SHA, run/attempt/job IDs, last sample time/status/active steps, unresolved gates and evidence locations. Transfer Docs artifact metadata once; do not reacquire already valid material. If ordinary subagent startup fails with empty provider or disallowed model route, do not keep guessing models or use an unauthorized explicit route. Use **permitted inherited** `subagent_fork` with `cwd=<verified workspace>`, `name=<unique observer>`, `description=<read-only CI scope>` and `prompt=<handoff plus no remote writes/dispatch/rerun/merge>` only if available and authorized. Otherwise the current agent alone continues read-only. A failed launcher is not a running observer. Moving to a new SHA ends the old scope; explicitly idle/stop stale observers instead of allowing overlapping requests.

10. **Audit duplicate runs by causality, not counts.** For same-SHA repeats, list run ID, event (`push`, `schedule`, `workflow_dispatch`, `workflow_run`), attempt and created/start/end times. Inspect the workflow version actually used, cold-build/cache behavior, upload conditions, release-asset replacement, downstream `workflow_run` download maintenance and documentation dispatch chain. Serial `concurrency` queues are not deduplication. Separate independent validation value from publishing side effects, directory-only maintenance from full documentation rebuild cost, and expected conditional skips from unfinished gates. Unknown semantic changes or schedule delays remain unconfirmed. An audit is read-only: no cancelling, rerunning or edits.

11. **Report meaningful changes and evidence boundaries.** Give the full SHA and UTC sample time, endpoint/source, computed status buckets, run/job links, active steps or first substantive error and next unresolved gate. Keep separate rows for the PR (and any subsequent PR), current main CI, and Pages/publication. Distinguish successful native docs publication, blocked upstream platform wave, actual lightweight follow-up deployment and no-change/no-loop closure; never infer the full chain from one successful job. Notify on publication, new failure, target change or completion of the full expected check set, not repetitive unchanged samples. Preserve original failed-attempt evidence and worktrees needed for unresolved acceptance; do not mark the overall objective complete while required gates are pending.

## Pitfalls

- Unsupported PR CLI fields must lead to REST identity retrieval, not missing base SHA. First-page run lists, short SHAs and a single green Docs job cannot establish merge readiness.
- A cancellation may mean capacity, concurrency, manual action or upstream gating; annotations and timeline decide. A retry can fail for a different reason and needs a new diagnosis.
- A 502 or background tool wrapper marked `ok` is not a successful retry. Check exit status, then remote effect before another write.
- Only fixing a preflight probe can leave actual payload downloads on the retired host. Only changing a parent variable can leave an ExternalProject child on the old value. Inspect generated child configuration.
- Old-main Pages success, local payload validation and a successful merge are three different facts, none of which proves new-main CI or new deployment success.
- Do not equate an observer's stale ID with liveness or spawn duplicate observers after model-routing failures. Current-agent fallback is valid and read-only.

## Verification

- Identity record contains full PR head/base, relevant test merge and latest/new main SHAs, UTC sample time and REST sources; paginated runs/jobs/checks including CodeQL are reconciled against actual trigger conditions.
- Every nonterminal check has an ID/link and real active step/start time or an explicit queued/no-step state. Skips have an explained condition; missing checks remain unresolved.
- Failure evidence contains the current attempt's annotations and complete job log, first substantive error and command exit status. Local evidence failures are labelled separately.
- Authorized repairs pass syntax checks (`bash -n` for touched shell scripts; use the repository's YAML parser/gate), local configure/generated-command inspection, affected rebuild/relink and narrow then original failed gate. Do not stack speculative pushes; observe one focused fix at its new full SHA.
- Firmware/download-specific local tests satisfy the [dependency checklist](<references/dependency-validation.md>); unavailable target-platform builds and publication checks remain explicitly pending.
- Any retry/merge has authorization and a receipt. New-main observation uses the new full SHA; tree-equivalence evidence is recorded when applicable but does not replace event-specific CI.
- Exactly one verified observer owns the scope, all relevant background outputs are collected, and PR, main and Pages verdicts are independently supported.
