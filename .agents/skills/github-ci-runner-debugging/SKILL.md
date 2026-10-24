---
name: github-ci-runner-debugging
description: Locate and fix remote GitHub Actions runner failures. Use when a run is red, when asked to make CI green, or when a fix must be verified before pushing. Complements .agents/rules/acloudviewer-ci-debugging.mdc (platform matrix, known failure patterns).
---

# CI Runner Failure Debugging Skill

This skill owns the **evidence acquisition and root-cause protocol** for remote
CI failures. The rules file owns the platform matrix and known failure patterns;
this file owns how to think. Every principle below was distilled from a real
incident; the incidents themselves are not recorded — the mechanisms are.

## Access ladder (evidence acquisition, cheapest first)

All read-only; use the REST API, not a CLI dependency.

1. **Run list** — which workflows fail, which commit, since when. Drill down
   per workflow+branch.
2. **Job/steps** — the failing step name narrows the phase (image build vs
   test vs upload).
3. **Check-run annotations** — `exit code N` + workflow line number. Often
   sufficient to classify without logs.
4. **Full job log** (needs a token; load it into a shell variable, never echo
   or commit it).

Poll with a background loop (10 min interval), never with blocking sleeps in
the main thread; drill down immediately on any failure signal.

## Triage protocol (fail-stop order)

1. **Matrix scope**: all configs failing at the same step = systematic
   (build-system/contract) issue; one config = platform/toolchain issue; one
   job cancelled while siblings succeeded = concurrency cancel, not a failure.
2. **Signature classification**:
   | Signature | Typical root cause |
   |---|---|
   | exit 127 on a binary path | script and build system disagree on an output path |
   | exit N inside docker RUN layers | image build layer error |
   | timeout | stage exceeded job limit (raise only after profiling) |
   | test failure / abort / segfault | code or environment regression |
3. **Last-green correlation**: find the last `success` commit, then diff the
   suspect paths. Multiple signatures = multiple independent bugs; fix only
   the ones alive at current HEAD, and check whether newer commits already
   fixed older signatures.
4. **Root cause before fix**: reproduce the mechanism locally with the
   smallest possible experiment. Never infer from comments — they can state
   the falsehood that caused the bug. Never accept "local passes" without
   verifying the binary was actually relinked against the changed library.
5. **In-situ verification**: after a CMake edit, re-configure the existing
   build tree and read the generated link/compile command files — that is the
   ground truth of what the build will do, in seconds.

## Distilled principles

### Build system & CMake

- **Per-config target properties beat generic overrides.** In multi-config
  generators a target's `RUNTIME_OUTPUT_DIRECTORY_<CONFIG>` (snapshotted from
  the global variable at `add_executable()` time) silently wins over a later
  generic `RUNTIME_OUTPUT_DIRECTORY` set via `set_target_properties`. When a
  per-target override must survive the dev-layout macros, re-derive it with
  `$<CONFIG>` in the property value (one property covers every configuration),
  and gate platform-specific output properties behind `if(MSVC)` /
  `if(CMAKE_CONFIGURATION_TYPES)` unless every consumer of the flat path is
  updated in the same change.
- **ExternalProject `CMAKE_ARGS` are passed verbatim to the sub-build.**
  Generator expressions are NOT evaluated — they land as literal strings in
  the child cache. Select platform-dependent flags with plain configure-time
  `if(WIN32) set(...)` in the parent, then splice the variable.
- **Guard every bare flag addition.** `-march=` is rejected by MSVC and
  non-x86 targets; `--retry-all-errors` needs curl ≥ 7.71 while older base
  images ship 7.68; macro `ARGN` is not a real list variable (capture with
  `set()` before `list()` commands). A flag that works locally can hard-fail
  another platform in the same commit.
- **Target lifetime == link lifetime.** Any `target_link_libraries` /
  `target_compile_options` on a conditionally-created target (tests behind
  `TESTS_ENABLED`) must live inside the same condition. The structural fix is
  to extend the creation macro with an explicit `LIBS` section rather than
  guarding the call sites one by one.
- **ISA/codegen flags are binary-wide ODR contracts, not library-private
  details.** Mixing two `-march=` regimes in one binary corrupts the heap via
  vague-linkage winners of shared inline functions; and INTERFACE propagation
  through `$<LINK_ONLY>` carries link deps but NOT compile options — a
  per-target fix will silently miss aggregated/object-library consumers. One
  emission point, one ISA for every TU of every binary.
- **Audit build artifacts by (mtime, content-tail), not existence.** CMake
  does not rewrite unchanged `flags.make` files, and renamed targets leave
  zombie directories whose stale files masquerade as live drift. Reconcile
  any artifact list against the live target registry before drawing
  conclusions.

### Platform differences (Windows vs Unix)

- **Filesystem case-insensitivity turns "different" names into the same
  file.** Two targets whose outputs differ only by case (an archive
  `flann.lib` and an executable's import library `Flann.lib` in the same
  directory) are one physical file; parallel build tasks writing it
  concurrently fail with `LNK1114 error code 32` (sharing violation).
  Eliminate the physical collision (isolate output directories per target
  family) rather than disabling the features involved.
- **OpenSSL-based libcurl has no CA store on Windows.** Its CA paths are
  auto-detected from `/etc/ssl` at build time; on Windows every HTTPS transfer
  fails with `CURLE_SSL_CACERT (code 60)`. The established cross-platform
  pattern (upstream Open3D) is `CURLOPT_SSL_VERIFYPEER/HOST = false` in the
  downloader plus a post-transfer content hash — integrity comes from the
  pinned digest, not from the peer check; the channel stays encrypted.
- **Real-time AV scanning holds fresh files.** A link or archive writing a
  new file can fail with a sharing violation if the scanner has an open
  handle; this is probabilistic (one task in thousands) and correlates with
  "same round a single failure, previous round all green". Prefer fixes that
  remove the collision surface over retries.
- **Transient network errors inside image builds need explicit retries.** A
  bare `curl … | bash` pipe both lacks retry and swallows the curl failure
  (bash executes an empty script; the real error surfaces later with a
  misleading message). Retry the whole fetch into a file gated on non-empty,
  using constructs available on the oldest supported base image.
- **Environment residue is a debugging hazard.** A stray `PYTHONPATH` pointing
  at a build staging tree (with bundled mismatched libraries) produces
  misleading import errors that look like code bugs. When local reproduction
  suddenly breaks after environment changes, reset the environment variables
  to a clean state before touching code.

### Test & forensics discipline

- **A crashing C++ extension loses pytest's captured output.** faulthandler
  dumps go to stderr unbuffered; the test's own progress goes to stdout,
  buffered, and dies with the process. To name the failing test/call:
  run pytest with `-v`, a fixed seed (`--randomly-seed=42`), and
  `PYTHONUNBUFFERED=1`, and have the test log step markers to **stderr with
  flush** (the same channel as the dump). The dumped "started-but-unfinished"
  verbose line names the aborting test.
- **Buffered progress output is not execution position.** "[22%] + N dots"
  only says what happened to be flushed; the real position can be far ahead.
  Never infer progress from buffered dots.
- **pytest-randomly shuffles the collected list; same seed reproduces the
  order only for the same collection.** Local trees that cannot collect all
  files (missing optional deps) cannot reproduce CI order. Fix the seed in
  CI so any failure is reproducible somewhere, and read the verbose output
  for names.
- **Skip a test on a platform when the surface was never verified there** —
  but only with the upstream evidence attached (e.g. "upstream runs this
  suite exclusively on Linux"), and keep the diagnostic markers in the file
  so the skip can be lifted by a later diagnosis. A skip that deletes the
  diagnostic loses the case forever.
- **PowerShell functions that must fail must `throw`, not `return 1`** — a
  bare return never sets `$LASTEXITCODE`, so callers checking it silently
  swallow the failure. Same idea: a "not found" branch that returns success
  turns a missing test into a green run.
- **Distinguish "silence" from "no failure".** A cancelled job prints nothing
  — an all-zero stats report or an empty log section may mean the observer
  never ran, not that the thing is broken. Anything that must be measured or
  flushed belongs at the end of the build step itself, while the process tree
  is alive.
- **Diagnostics must survive the crash they diagnose.** Put state-logging on
  the failure path of silent failure modes (e.g. log the counts that define
  the broken invariant before returning the error). The next occurrence then
  self-reports instead of costing another debugging round.
- **Environment-independent nondeterminism in one test, across rounds and
  configs, points at state pollution or a race, not at code.** Collect the
  timeline (timestamps), audit every link-maintenance path, rule out UB with
  an ASAN build, and only then accept "environment-specific" with a
  diagnostic left in place.

### Upstream alignment

- Before designing a Windows fix, check what the upstream project does on
  Windows for the same surface: which tests it runs (and skips), which flags
  it passes, how its downloader handles certificates. A test surface the
  upstream never verified is a superset — aligning the verification boundary
  (skip with rationale) is legitimate; shipping product features for that
  surface is a separate decision with its own risk.
- Upstream reference clones are the fastest evidence source; workflow files
  answer "does upstream run X on platform Y" in minutes.

## Verification ladder before pushing

1. Syntax-check the touched scripts (`bash -n`, YAML parse).
2. Re-configure the local build tree; read the generated commands — the
   ground truth.
3. Rebuild the affected targets locally; relink anything that consumes a
   changed library.
4. Run the narrowest relevant gates, then the full gate that failed.
5. CI is the final gate: one focused commit, watch the run, do not stack
   speculative fixes — each push costs a full rebuild.
