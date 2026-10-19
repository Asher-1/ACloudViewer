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

- 2026-10-19 `2eba52d3c`: Windows `Test wheel (3.12)` — the forensics
  stack (-v + PYTHONUNBUFFERED + seed=42) named the aborting test on the
  first retry: `test_parity_batches.py::test_database_management_surface`
  (faulthandler dump, then the flush of its started-but-unfinished -v line;
  the previous run's "[22%] + 35 dots" was a stdio-buffer artifact, NOT the
  execution position — the real position was ~88%, i.e. ~758/862 tests in,
  with torch already loaded). Static audit of its 7 C++ calls (Database
  open/clears/reads/read_rig_with_sensor/static pair-id helpers/`with db`
  transaction) found all upstream-consistent; no SQLite error line before
  the dump (the `exit(EXIT_FAILURE)` path of SQLite3CallHelper was ruled
  out). Step-level stderr markers (`[db_mgmt_surface] ...` on the same
  unbuffered channel as the dump, incl. forced `del`+`gc.collect()`
  destructor markers) landed in the test — the next CI run names the exact
  C++ call. Local Linux (gcc 11 CUDA tree): 9 markers green, test PASSED,
  61/61 file+sensor regression — Windows-only, not locally reproducible.
  NOTE: a stray PYTHONPATH pointing at the build staging package (its
  bundled Qt 5.15 libs) poisons local cloudViewer imports with a misleading
  Qt private-symbol OSError; `unset PYTHONPATH` first when debugging.
- 2026-10-18 `4fddd764b`: Windows `Test wheel (3.12)` — `Fatal Python
  error: Aborted` 4 minutes into the python suite (stderr, immediately
  visible) while the stdout progress dots stayed buffered: the run showed
  only 22 dots in `test_parity_batches.py [ 22%]`, then nothing for ~6h
  until the runner limit killed it (the aborted interpreter never released
  the output pipe). Root cause of the silence, not of the abort: the
  windows.yml pytest invocation had no `-v`, no fixed seed and no
  `PYTHONUNBUFFERED` (ubuntu's run_python_tests passes -v; the same suite
  passes there in ~6min). Forensics fix: `-v --randomly-seed=42` +
  `PYTHONUNBUFFERED: '1'` on both windows wheel test steps and in
  run_python_tests, plus `timeout-minutes: 60` on the test-wheel job.
  Local repro matrix (CUDA tree, gcc 11): seed 42 full-tree run passes
  404 (torch-dependent files cannot be collected locally), single-file
  runs pass 56/56 — the abort is Windows/MSVC-specific or order+platform
  specific; the next CI run will name the exact test.
- 2026-10-18 `4fddd764b`: cuda-noble `Docker build` — `curl: (6) Could not
  resolve host: pyenv.run` at Dockerfile.ci's pyenv layer (exit 127, job
  117709564908). Transient resolver blip killed a ~2h image build because
  the bare `curl https://pyenv.run | bash` pipe had no retry AND no
  pipefail (bash executed an empty script; the error surfaced later as
  "pyenv: command not found"). Fix: bash for-loop retry (5x5s) with the
  installer landing in a file gated on `-s` non-empty; curl
  `--retry-all-errors` was rejected because it needs curl >= 7.71 and
  focal (7.68) is a supported base image.
- 2026-10-18 `4fddd764b`: cuda-focal `Docker test` —
  automatic_reconstruction_test 173-1 (GlobalMapperProducesModel) again,
  but the zero-residual diagnostic finally fired: `41 3D points, 5
  configured images, 5 registered frames` inside the retriangulation BA
  (2.3 ms after "Iterative bundle adjustment done"; the retriangulation
  heading IS present). Static audit found every link-maintenance path
  (Load/BeginReconstruction/MergePoints3D/AddPoint3D/FilterTracks,
  boost::unordered_node_map reference stability) upstream-consistent; an
  ASAN build of the full test passes clean locally (CUDA tree, gcc 11).
  Failure is environment-nondeterministic (cpu-focal failed on b96f8a2f5,
  passes on 4fddd764b). Next-round evidence: the zero-residual path now
  runs a two-way link audit (empty-track point count, dangling track
  elements, per-image linked-observation counts) — read it before any
  further code change.
- 2026-10-18 `4fddd764b`: Windows wheel SUCCESS on the first NMC run
  (Build wheel ON,3.12 ~5.9h vs the VS-generator double-pass 6h timeouts),
  confirming the sccache launcher fix, the webrtc dependency edge, and
  CLOUDVIEWER_WHEEL_SKIP_CPU_PASS being unnecessary (CPU+CUDA double pass
  fits when sccache actually runs).
- 2026-10-18 `8cfc7088d`: Windows `Build wheel (ON, 3.12, false)` cancelled at
  6h with all-zero sccache stats — REAL zero, not a timing artifact: the
  wheel job ran the default "Visual Studio 17 2022" generator (ci_utils.ps1
  Get-CMakeGeneratorArgs default) and CMAKE_C/CXX_COMPILER_LAUNCHER is only
  honored by Makefile/Ninja generators, so MSBuild compiled the whole CPU+
  CUDA double pass without ever invoking sccache. Fix: wheel job env
  CLOUDVIEWER_GENERATOR="Ninja Multi-Config" + ilammy/msvc-dev-cmd (same
  wiring as the app matrix job). Follow-up gaps found in the app->wheel NMC
  comparison (all fixed): (1) Build-PipPackage lacked the
  -DCMAKE_CONFIGURATION_TYPES=Release pin that Build-GuiApp already carries
  (NMC ignores CMAKE_BUILD_TYPE); (2) ExternalProject_CMAKE_ARGS did not
  forward CMAKE_CONFIGURATION_TYPES, so the 3rdparty sub-builds would see an
  empty config under NMC; (3) Run-CppUnitTests probed only
  bin\<Config>\tests.exe while the NMC layout is
  bin\cloudviewer_test\<Config>\tests.exe, AND its not-found branch used a
  bare `return 1` which never sets $LASTEXITCODE — the caller comparing
  $LASTEXITCODE silently swallowed the missing-test case. It now probes both
  layouts and throws. Keep in mind the stats steps only observe a
  live daemon when printed in-step (build_win_wheel.ps1 tail), and a
  cancelled job prints nothing. NOTE: the OFF+Debug app-matrix entries are
  env-setup-only (build_win_app.ps1 is Release-only); their success does NOT
  exercise Run-CppUnitTests.
- 2026-10-18 `8cfc7088d`: Ubuntu `Test wheel (3.13, false)` —
  `Fatal Python error: Segmentation fault` in
  test_synthetic.py::test_synthesize_noise. Same ISA-mixing root as the
  undistortion abort: pybind.so archives ColmapLib SSE2 TUs together with
  pybind/CloudViewer v3 TUs. Reproduced 8/8 core dumps locally with the
  CI-state cuda/pybind.so; after the global baseline tail
  (CMakeLists.txt USE_SIMD branch appends -march=x86-64 last for
  BUILD_RECONSTRUCTION) the rebuilt pybind.so passes 10/10. NOTE: the
  CMake INTERFACE propagation of the baseline stops at direct
  target_link_libraries consumers ($<LINK_ONLY> carries link deps but not
  compile options), so the global tail is the only complete fix.
- 2026-10-18 `8cfc7088d`: Ubuntu jammy+noble `Docker test` —
  `image/undistortion_test` abort (`free()/munmap_chunk(): invalid pointer` in
  TestUndistortReconstruction) and macOS ColmapLib build error
  (`unsupported argument 'x86-64' to option '-march='`), same root: the
  ColmapLib-only `-march=x86-64` override (PRIVATE, unconditional). Fixed by
  x86+GNU/Clang guard and PUBLIC `$<COMPILE_LANGUAGE:CXX>` propagation in
  `libs/Reconstruction/src/CMakeLists.txt` — ISA mixing inside one binary is
  an ODR-level heap-corruption hazard (reproduced 5/5 locally, flags-matched
  1-TU experiment passes 6/6).
- 2026-10-17 `b96f8a2f5`: Windows app matrix `Build ACloudViewer App` —
  configure error at `libs/Reconstruction/src/mvs/CMakeLists.txt:89`
  ("Cannot specify link libraries for target poisson_meshing_test which is
  not built by this project"): 30e9fdd981 added bare target_link_libraries
  outside the macro; the Windows app matrix passes -DBUILD_UNIT_TESTS=OFF →
  set(TESTS_ENABLED ${BUILD_UNIT_TESTS}) → COLMAP_ADD_TEST creates no target
  → the post-macro links reference a target that does not exist. Forward fix
  (upstream colmap_mvs shape): the COLMAP_ADD_TEST macro gained an optional
  trailing `LIBS <libs...>` section (upstream LINK_LIBS parity, links inside
  the same TESTS_ENABLED block, ARGN captured via set() because macro ARGN
  is not a real list variable); poisson_meshing_test's direct poisson_recon
  link was deleted as redundant (ColmapLib PUBLIC ${COLMAP_INTERNAL_LIBRARIES}
  already propagates the solver, matching upstream colmap_mvs
  PRIVATE_LINK_LIBS colmap_poisson_recon), poisson_recon_lib_test keeps its
  direct PoissonReconLib.h dependency via `LIBS poisson_recon_lib`. Same run, Ubuntu focal `Docker test` 173-test gate:
  1 failure = controllers/automatic_reconstruction_test
  GlobalMapperProducesModel — retriangulation left the BA problem with zero
  residuals and the fork's silent `return false` surfaced as "Global mapping
  failed" with no diagnostic; zero-residual-failure semantics are
  upstream-consistent (default summary = FAILURE -> IsSolutionUsable()==false);
  fixes = upstream-aligned retriangulation sequence (DeleteAllPoints2DAndPoints3D
  + BeginReconstruction's Load() restore) and a state-logging ERROR on
  the zero-residual path.
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
