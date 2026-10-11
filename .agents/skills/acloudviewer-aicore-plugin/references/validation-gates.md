# AICore Validation Gates

This file owns the executable validation policy. The runner and probes are
authoritative; this document defines what a report is allowed to claim.

## Manifest Contract

`core/AICore/scripts/validation_manifest.json` is the single matrix owner. A
scenario owns its model globs, required dependencies, probe, backend, pipeline
and quantization coverage. Every catalog model must expand into at least one
scenario in the requested tier.

The manifest uses schema `2`. `accuracy_gate` remains a short display label for
compatibility, but new scenarios must add:

```json
"accuracy": {
  "reference": "cpu_reference|upstream|golden|probe_invariant",
  "reference_id": "stable probe or fixture identity",
  "checks": [
    {"metric": "kpt_median_px", "op": "<=", "value": 0.005}
  ],
  "require_metrics": ["kpt_median_px"]
}
```

Supported operators are `<`, `<=`, `>`, `>=`, `==`, `!=`, and `finite`.
Metric names may be complete flattened JSON paths or a unique leaf name. Any
numeric leaf a row's own `checks`/`require_metrics` name is collected as a
metric; no runner whitelist entry is needed. A
missing or non-finite required metric fails the scenario. A legacy string is
reported as `probe_legacy`; it is accepted only by the developer profile, is
not a numeric threshold, and must not be used as release or upstream-accuracy
evidence.

Use `upstream` or a pinned `golden` reference for semantic correctness. Use
`cpu_reference` only for a backend parity claim. Use `probe_invariant` for
finite/non-empty/role/geometry conditions implemented by the probe and record
those conditions in its output and tests. All other references must declare at
least one `checks` or `require_metrics` entry. A `probe_invariant` row may leave
both lists empty only when a nonzero probe exit code enforces the named
conditions; it proves structural correctness, not upstream semantic accuracy.

An `upstream` row that executes an external reference program pins the
program's source identity (revision, clean tree, submodule revision and the
sha256 of any submodule working-tree diff) and fails when the reference
binary is older than any file it is compiled from: a source pin over a stale
binary certifies a different program (`run_map_anything_upstream_parity.py`).

Every model glob of a selected task (`model_glob`, `owned_globs`,
`covered_globs`, `required_globs`, `light_globs`) must match at least one
`test_catalog_dump_urls --json` row; the preflight fails closed on a pattern
that names an unpublished file (`local_only` rows excepted). Under the default
`exact` fingerprint policy a probe must print at least one string leaf whose
name contains `hash`/`sha`/`digest` (a byte hash of its typed outputs).

## Profiles and Claims

The runner has explicit profiles:

```bash
# Fast local diagnosis. Not release evidence.
python3 core/AICore/scripts/validate_all.py --profile developer \
  --build build_app --backend cuda --output build_app/Testing/aicore.json

# Release/regression claim. Requires a same-host baseline.
python3 core/AICore/scripts/validate_all.py --profile release \
  --build build_app --baseline-build build_app.before --backend cuda \
  --full --output build_app/Testing/aicore-release.json
```

Developer defaults are 2 process repeats, 1 warmup, and 2 timed forwards.
Release defaults are 3 repeats, 2 warmups, and 10 timed forwards; release
requires `--baseline` or `--baseline-build`, rejects `--allow-incomplete`,
rejects every selected row whose accuracy is still legacy prose, and rejects
weaker sampling overrides.
Baseline comparison requires identical backend, profile, tier, input/model
digests, host identity, and sampling protocol. It compares p50/p95 and other
timing fields with both relative and absolute noise floors.

`--allow-incomplete` is diagnostic-only. Missing downloads, exit-77 probes,
uncovered models, and resource skips are reported as `INCOMPLETE`; that report
cannot support a release, parity, accuracy, or speed claim. `--full` is needed
for a complete matrix claim. The light tier is a declared subset, not an
implicit local-cache discovery.

## Required Evidence Ladder

1. Build and export/ABI contract tests.
2. Probe-level numeric accuracy against the declared reference.
3. Repeated-forward stability and context recreation.
4. Real checkpoint execution on every claimed backend and quantization.
5. Controlled release-profile A/B for performance changes.
6. Plugin still/live/cancel/stale/error/teardown workflow tests.
7. Agent automation closure: native HELP schema, RPC whitelist, exact CLI
   registry, MCP tools, installed-plugin L3 checks on all three platforms,
   and L4 against the packaged GUI under the offscreen QPA.

Parity is not upstream truth. A stable output hash is not semantic accuracy.
Weight quantization is not evidence of lower activation/KV memory or lower
latency; inspect the graph reservation and report the requested frame/window.

## New Task / Model Gate

Update the owning task catalog, stable URL/cache folder, digest header, catalog
dump, probe target, structured manifest rows, and runner tests together. Add
negative tests for bad downloads, missing coverage, malformed accuracy checks,
and incomplete filtering. Preserve the JSON and Markdown report from the exact
revision and dirty state used for the claim.

## Automation Gate

A new AICore plugin or command is incomplete until all of these identities are
the same: native command keyword, JSON-RPC `aicore.batch` whitelist, Python AI
registry, CLI output, and MCP task resolution. Option metadata has one runtime
owner: the native `HELP RESULT_JSON` schema. Python must not duplicate defaults.

Run L1/L2/L5 without a binary (Linux, macOS, Windows), then L3 against the
packaged binary. L3 queries
every registry task, requires a non-empty option list containing `RESULT_JSON`,
and requires schema `acloudviewer.aicore.batch/1`. Linux CPU/CUDA, macOS, and
Windows installer jobs are separate evidence; success on one platform does not
cover another. Missing plugins and unknown commands fail rather than skip.

L4 runs on the Ubuntu installer job: the GUI starts under `offscreen` on
Xvfb with software Mesa, must serve `ping` before the suite starts, and must
still be alive after it. A GUI that cannot start (no GL context, port taken)
exits or logs instead of crashing or waiting on a modal dialog; that is what
makes an unattended start observable. The CLI-Anything change that a new
registry row depends on must reach its default branch before the
ACloudViewer change is pushed.
