---
name: github-ci-runner-debugging
description: Locate and fix remote GitHub Actions runner failures for this repository. Use when a run on github.com/Asher-1/ACloudViewer/actions is red, when asked to make CI green, or when a fix must be verified before pushing. Complements .agents/rules/acloudviewer-ci-debugging.mdc (platform matrix, known failure patterns).
---

# GitHub CI Runner Failure Debugging Skill

Methodology proven on real incidents (runs 38918190650, 38852073143, 36713152602).
The rule file [acloudviewer-ci-debugging.mdc](../../rules/acloudviewer-ci-debugging.mdc)
owns the platform matrix and known failure patterns; this skill owns the
**evidence acquisition and root-cause protocol** for remote failures.

## Access ladder (evidence acquisition, cheapest first)

All read-only; `gh` CLI is not required. Use the REST API.

1. **Run list** (public): `GET /repos/Asher-1/ACloudViewer/actions/runs?per_page=30`
   and per-workflow/branch drill-down:
   `GET .../actions/workflows/ubuntu.yml/runs?branch=<branch>&per_page=30`.
   Yields: which workflows fail, which commit, since when.
2. **Job/steps** (public): `GET .../actions/runs/<run_id>/jobs` — the failing
   step name narrows the phase (Docker build vs Docker test vs upload).
3. **Check-run annotations** (public): `GET /repos/.../check-runs/<job_id>/annotations`
   — `level=failure` entries give `Process completed with exit code N` and the
   workflow line number. Often sufficient to classify without logs.
4. **Full job log** (needs token): `GET /repos/.../actions/jobs/<job_id>/logs`.
   A token exists in `~/.git-credentials` (`https://Asher-1:<token>@github.com`).
   Load it into a shell variable, never echo or commit it:
   `TOKEN=$(grep -o 'https://Asher-1:[^@]*@github.com' ~/.git-credentials | sed 's|https://Asher-1:||;s|@github.com||')`
   Then `curl -sL -H "Authorization: Bearer $TOKEN" -o /tmp/log.txt <url>`.

## Triage protocol (fail-stop order)

1. **Matrix scope**: which configs fail vs pass? All configs of a workflow
   failing at the same step = systematic (build-system/contract) issue; one
   config = platform/toolchain issue; one job cancelled while its matrix
   siblings succeeded = concurrency cancel, not a failure.
2. **Signature classification** from the annotation/first error:
   | Signature | Meaning | Typical root cause |
   |---|---|---|
   | exit 127 + "No such file or directory" on a binary path | runner script and build system disagree on an output path | output-path contract broken (see contract table) |
   | exit N inside `#26 <t> ...` docker RUN lines | image build layer failure | apt/conda/patch/source error in Dockerfile.ci |
   | timeout | long stage exceeded job limit | raise limit only after profiling the stage |
   | test failure (gtest/gtest_* reported) | code or environment regression | bisect with last-green commit |
3. **Last-green correlation**: list runs per workflow+branch to find the last
   `success` commit, then `git log <last-green>..HEAD -- <suspect-paths>` to
   shortlist the introducing commit. A failure stream with several signatures
   means several independent bugs — fix the ones alive at current HEAD only,
   and check whether newer commits already fixed older signatures.
4. **Root cause before fix**: reproduce the mechanism locally with the smallest
   possible experiment (standalone CMake/shell project), never infer from
   comments alone — comments can state falsehoods that caused the bug.
5. **In-situ verification without a full rebuild**: after editing CMake, run
   `cmake .` in the existing build tree and read the generated link command,
   e.g. `grep -o '\-o [^ ]*' libs/CVViewer/tests/CMakeFiles/tests.dir/link.txt`.
   This is the ground truth of where Make will emit the binary, in seconds.

## Output-path contract for the unit-test runner (single source of truth)

| Generator | Emitted path | Consumers expecting it |
|---|---|---|
| Unix Makefiles (CI Docker, build_app) | `bin/cloudviewer_test/tests` | `docker/docker_test.sh` (`./bin/cloudviewer_test/tests`), `util/ci_utils.sh run_cpp_unit_tests`, `libs/CVViewer/tests/README.md` |
| MSVC multi-config | `bin\cloudviewer_test\<Config>\tests.exe` | `.github/workflows/windows.yml` (`Join-Path $buildDir "bin\cloudviewer_test\${{ matrix.CONFIG }}\tests.exe"`) |

Owner: `libs/CVViewer/tests/CMakeLists.txt`. The per-config
`RUNTIME_OUTPUT_DIRECTORY_<CONFIG>` properties are MSVC-only on purpose.

## CMake pitfall that caused the 2026-10 outage (generalizable)

A per-config target property carrying `$<CONFIG>` (e.g.
`RUNTIME_OUTPUT_DIRECTORY_RELEASE ".../$<CONFIG>"`) is **not** ignored by
single-config generators: with `CMAKE_BUILD_TYPE=Release` the `_RELEASE`
property wins over the generic `RUNTIME_OUTPUT_DIRECTORY`, and `$<CONFIG>`
evaluates to `Release` — the binary lands in a config subdirectory that no
Unix consumer knows about. Reproduced with a 6-line standalone project.
Rule: per-config output properties go inside `if(MSVC)` (or `if(CMAKE_CONFIGURATION_TYPES)`)
unless every consumer of the flat path is updated in the same change.

## Verification ladder before pushing

1. `bash -n` the touched shell scripts (syntax).
2. Re-configure the local build tree and check the emitted link path
   (link.txt) — see triage step 5.
3. Local rebuild of the affected target when affordable
   (`cmake --build build_app --target tests -j4`).
4. CI is the final gate: push one focused commit, watch the run, do not stack
   speculative fixes — each push costs a full Docker build (~3 h on CUDA).

## Incident log (index, not narrative)

- 2026-10-16 `aff48cd6`: Ubuntu/Ubuntu CUDA `Docker test` exit 127 — per-config
  output properties moved `tests` to `bin/cloudviewer_test/Release/tests`;
  fixed MSVC-only guard in `libs/CVViewer/tests/CMakeLists.txt`.
- 2026-10-16 `aff48cd6`: Windows `Build wheel (ON, 3.12)` cancelled at 361.7min
  = the 6h runner limit (the app matrix jobs never time out). Root causes:
  wheel job had no sccache steps (fixed: added the same two-step wiring +
  COLD_BUILD env), and sccache stats always reported 0 on Windows — the daemon
  lives inside the build step's process tree, GitHub runners terminate that
  tree on step exit, and the daemon's stats are in-memory, so any post-build
  stats observer sees a fresh daemon. Cache writes are real (repo cache
  8.4GB/~12k entries; macos.yml's actions/cache entries are few by
  comparison). Fix: stats printed by build_win_app.ps1 / build_win_wheel.ps1
  at the end of the build step, while the daemon is still alive.
- 2026-10-15 `c984168a`/`993e5133`: Agent Integration Level 3 headless — EGL
  needed a display platform; fixed with `xvfb-run` (6dafa5db27) after software
  GL (f7ba7bf9bd).
- 2026-09-30 `c97643af`: focal Docker build — vendored boost nvp clash;
  fixed in 89833cff07.

## Windows-specific traps learned here

- A step that runs a tool *after* the build step cannot observe in-build
  daemons' state on Windows runners (process tree is killed on step exit).
  Anything that must be measured or flushed (stats, caches, logs) belongs at
  the end of the build step itself.
- "0 requests" in sccache stats with compiler commands visibly prefixed by
  sccache is a **timing artifact, not a dead cache**: check the repo-wide
  cache usage (`GET /repos/.../actions/cache/usage`) to confirm entries are
  being written before concluding the cache is broken.
- `mozilla-actions/sccache-action` does not export `SCCACHE_GHA_ENABLED` by
  itself; wire it manually (see windows.yml "Enable sccache launcher").
