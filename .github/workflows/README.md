# GitHub Actions Workflows

Reference for every workflow in `.github/workflows/`: trigger/runner/matrix
configuration, the compiler-cache mechanism of each job, and everything each
job uploads (artifacts and releases).

## 1. Workflow Inventory

| Workflow | File | Triggers | Runner(s) | Matrix / cells | Jobs |
|---|---|---|---|---|---|
| Windows | `windows.yml` | `workflow_dispatch` (`developer_build`, `cold_build`), `schedule` (Mon 03:00 UTC, cold-build gate), `push: main`, `release: released`, `pull_request` | `windows-2022` | app: `CONFIG` (Release/Debug) x `BUILD_CUDA_MODULE` (ON/OFF), py3.12; agent: CUDA ON/OFF; wheel: CUDA ON x py3.10–3.12 (PR: py3.12); test-wheel: py3.10–3.12 (PR: py3.12) | `windows`, `agent-integration`, `build-wheel`, `test-wheel` |
| Ubuntu | `ubuntu.yml` | `workflow_dispatch`, `push: main`, `release: released`, `pull_request` | `ubuntu-latest` (+ `ubuntu-20.04/22.04/24.04` for agent job) | docker: `cpu-focal`, `cpu-jammy`, `cpu-noble` | `skip-check`, `build-and-run-docker`, `agent-integration`, `aicore-tests` |
| Ubuntu CUDA | `ubuntu-cuda.yml` | same as Ubuntu | `ubuntu-latest` | docker: `cuda-focal`, `cuda-jammy`, `cuda-noble` | `skip-check`, `build-and-run-docker` |
| Ubuntu Wheel | `ubuntu-wheel.yml` | `workflow_dispatch`, `push: main`, `release: released`, `pull_request` | `ubuntu-latest` (test: `ubuntu-22.04`) | wheel: py3.10–3.13 (PR: py3.12 only); fork PRs skipped | `skip-check`, `build-wheel`, `test-wheel` |
| MacOS | `macos.yml` | `workflow_dispatch`, `push: main`, `release: released`, `pull_request` | macOS runners (`runner.arch` = ARM64/X64) | app: py3.12; wheel: py3.10–3.12 (PR: py3.12) | `MacOS`, `agent-integration` (macos-15), `build-wheel`, `test-wheel` |
| CodeQL Security Analysis | `codeql.yml` | `push: main`, `pull_request: main`, `schedule` (Mon 00:00 UTC), `workflow_dispatch` (`cpp-optimization`) | `ubuntu-24.04` | Python + C/C++ analyses | `analyze-python`, `analyze-cpp` |
| Documentation | `documentation.yml` | `workflow_dispatch`, `push: main`, `pull_request`, `release` | `ubuntu-latest` | — | `build-docs`, `deploy-docs` |
| Style Check | `style.yml` | `workflow_dispatch`, `push: main`, `pull_request` | `ubuntu-latest` | — | `style-check` (auto-formats PR branches, plus Vulkan/RealSense script checks) |
| Agent Integration | `agent-integration.yml` | `workflow_dispatch`, `push`/`pull_request` with `agent-integration/**`, `core/AICore/**`, plugin/AICore paths | per matrix | OS x Python (newest interpreter on macOS/Windows) | `test-basic` (Level 1+2+5), more levels |
| Prebuild Ubuntu deps (VTK/PCL) | `prebuild-deps.yml` | `workflow_dispatch` only | docker (ubuntu) | per-codename tarball `vtk-pcl-deps-<codename>-vtk9.3.1-pcl1.14.1.tar.gz` | single build job (skips when the asset already exists) |
| Update Downloads Data | `update-downloads.yml` | `workflow_run` after Windows/MacOS/Ubuntu, `release` events, `workflow_dispatch` | `ubuntu-latest` | — | refreshes `downloads_data.json` |
| Clean release | `update-release.yml` | `workflow_run` after Windows/MacOS/Ubuntu/Ubuntu CUDA/Ubuntu Wheel | `ubuntu-latest` | — | idempotent release-asset cleanup |
| VTK Packages | `vtk_packages.yml` | `workflow_dispatch` only | Linux / Windows / macOS | per-OS VTK archives | `Linux`, `Windows`, `MacOS` |
| WebRTC | `webrtc.yml` | `workflow_dispatch` only (pin commits) | Linux / Windows | — | WebRTC static libs build |

Notes:
- All build workflows honor `[skip ci]` in the head-commit message.
- PR runs on forks are skipped for publish-capable jobs (Ubuntu Wheel uses an
  explicit `skip-check` job; Windows/macOS use repo checks).
- Windows `COLD_BUILD=true` (releases, weekly schedule, explicit dispatch
  input) disables the compile cache end-to-end so shipped artifacts stay
  reproducible from clean sources; builds keep the full 360 min budget.

## 2. Compiler-Cache Mechanisms

Cache never gates CI: every failure path degrades to a cold build
(`continue-on-error` everywhere; restore failure = cold build).

| Workflow / job | Mechanism | Key / storage | Path on disk | Restore / save policy |
|---|---|---|---|---|
| Windows `windows` (app) | sccache daemon + stash artifact pool | `sccache-win-app-cuda${ON/OFF}` (apache/infrastructure-actions/stash; one artifact per cell, newest run wins) | `${{ runner.temp }}\sccache` | restore → launcher wired via `CLOUDVIEWER_COMPILER_LAUNCHER` → trim (drop entries untouched since `SCCACHE_POOL_T0`, success only) → save on success/failure/cancelled build; retention 14 d, compression 0; skipped when `COLD_BUILD=true`; build timeout 330 min cached / 360 min cold |
| Windows `build-wheel` | sccache daemon + stash artifact pool | `sccache-win-wheel-cuda${ON/OFF}-py${py}`; **seeded** from both `sccache-win-app-cuda*` pools when its own stash missed (cold-start deadlock breaker) | `${{ runner.temp }}\sccache` | same trim/save policy as the app job (20 GB ceiling for the two-pass CUDA+CPU build) |
| Windows `agent-integration`, `test-wheel` | none | — | — | downloads artifacts only |
| Ubuntu / Ubuntu CUDA (docker build) | ccache tarball on release assets | `ccache-ubuntu-${CI_CONFIG}.tar.gz` on `Asher-1/cloudViewer_downloads` release `ci-ccache` (free, no total quota, 2 GiB per-asset limit) | `.ccache/` in the checkout, COPY'd into the docker image, exported back by `docker/docker_build.sh` | curl restore (plain HTTPS, fork PRs work) → save via `gh release upload --clobber` (`ACCESS_TOKEN`); tarballs < 50 MB are treated as cancelled-run debris and not uploaded; **no branch dimension** (content-addressed keys make cross-branch sharing safe) |
| Ubuntu Wheel (docker build) | ccache tarball on release assets | `ccache-ubuntu-wheel-py${py}.tar.gz` on the same `ci-ccache` release | `.ccache/` | same policy as Ubuntu; per-python pool names |
| MacOS `MacOS` (app) | ccache + stash artifact pool | `ccache-macos-${runner.arch}-app`; legacy release-asset seed `ccache-macos-wheel-py*.tar.gz` (cloudViewer_downloads) when the stash missed | `${{ github.workspace }}/.ccache` | restore → `ccache -z` in build → save on success/failure build (no partial-run pruning); retention 14 d |
| MacOS `build-wheel` | ccache + stash artifact pool | `ccache-macos-${runner.arch}-wheel-py${py}` (+ same legacy seed) | `${{ github.workspace }}/.ccache` | same policy as the app job |
| CodeQL | CodeQL built-in caches (traps/queries) | managed by `github/codeql-action` | runner-managed | not repo-visible; no `actions/cache` steps |
| Documentation, Style, VTK Packages, WebRTC, Prebuild deps, Update-* | none | — | — | prebuilt-deps are the cache for normal CI (see `ubuntu-prebuild-deps` release below) |

## 3. Artifact & Release Uploads

Artifact uploads marked *always* run with `if: always()` +
`continue-on-error: true` (evidence survives failed builds).

| Workflow / job | Name | Path | When |
|---|---|---|---|
| Windows `windows` | `windows-diagnostics-<CONFIG>-cuda<CUDA>` | `${{ runner.temp }}\windows-build.log`, `build-timings.jsonl`, `sccache-*.json`, `ninja-*`, `build\**\.ninja_log` | always |
| Windows `windows` | `ACloudViewer-<ver>-<cuda/cpu>-win64.exe` (installer, `PACKAGE_INSTALLER_NAME`) | `${{ env.INSTALL_DIR }}\<installer>` | Release lane (`BUILD_SHARED_LIBS=OFF`) |
| Windows `windows` | release `main-devel` asset | installer `.exe` | `push: main` |
| Windows `windows` | tag release asset (softprops) | installer `.exe` | `release` tag |
| Windows `agent-integration` | — (downloads the installer artifact) | — | — |
| Windows `build-wheel` | `windows-wheel-diagnostics-py<py>-cuda<CUDA>` | `windows-wheel-build.log` + the same profiling evidence set | always |
| Windows `build-wheel` | `cloudviewer-*.whl` artifact | `${{ env.INSTALL_DIR }}/cloudviewer*.whl` (CUDA + CPU) | build success |
| Windows `build-wheel` | release `main-devel` / tag release | wheels | `push: main` / tag |
| Ubuntu `build-and-run-docker` | `ubuntu-build-log-<CONFIG>` | `/tmp/ubuntu-build.log` | always |
| Ubuntu `build-and-run-docker` | `ACloudViewer-devel-linux-x86_64-<CONFIG>` | `ACloudViewer-*.run` | build success |
| Ubuntu | release `main-devel` / tag release | `.run` installer | `push: main` / tag |
| Ubuntu CUDA | `ubuntu-cuda-build-log-<CONFIG>` | `/tmp/ubuntu-cuda-build.log` | always |
| Ubuntu CUDA | `ACloudViewer-devel-linux-x86_64-<CONFIG>` | `ACloudViewer-*run` | build success |
| Ubuntu CUDA | release `main-devel` / tag release | `.run` installer | `push: main` / tag |
| Ubuntu Wheel `build-wheel` | `ubuntu-wheel-build-log-py<py>` | `/tmp/ubuntu-wheel-build.log` | always |
| Ubuntu Wheel | `cloudviewer-*.whl` artifact | CUDA + CPU wheels in workspace root | build success |
| Ubuntu Wheel | release `main-devel` / tag release | wheels | `push: main` / tag |
| MacOS `MacOS` | `macos-build-log-<arch>` | `/tmp/macos-build.log`, `build/Testing/Temporary/LastTest.log`, `~/Library/Logs/DiagnosticReports/test_*.ips` | always |
| MacOS `MacOS` | `ACloudViewer-app-macosx-<arch>` | `${{ env.INSTALL_DIR }}/${{ env.DEVEL_PKG_NAME }}` (DMG) | build success |
| MacOS | release `main-devel` / tag release | DMG | `push: main` / tag |
| MacOS `build-wheel` | `macos-wheel-build-log-py<py>` | `/tmp/macos-wheel-build.log` | always |
| MacOS `build-wheel` | `cloudviewer-*.whl` artifact | `build/lib/python_package/pip_package/` | build success |
| MacOS `build-wheel` | release `main-devel` / tag release | wheels | `push: main` / tag |
| CodeQL `analyze-cpp` | `codeql-cpp-evidence-<opt>-<run_id>` | `codeql-evidence/`, `build/**/compile_commands.json` | after C++ analysis |
| Documentation `build-docs` | docs archive artifact (`acloudviewer-*-docs.tar.gz`) | repo-root archive | build success |
| Documentation | `docs-output`, `website-deploy` | build outputs, retention 1 d (hand-off to deploy job) | build success |
| Documentation | release `main-devel` / versioned release | docs archive | `push: main` / `release` |
| Prebuild deps | `${TARBALL}` artifact + `ubuntu-prebuild-deps` release | `vtk-pcl-deps-<codename>-*.tar.gz` | build success (published once, then skipped) |
| VTK Packages | `vtk_linux_*` / `vtk_windows_*` / `vtk_macos_*` | `vtk*.tar.gz` + `checksum*.txt` | build success |
| WebRTC | `webrtc_release_<os>` | `webrtc_*.tar.gz` / `webrtc_*.zip` + `checksum_*.txt` | build success |
| Update Downloads Data | release `main-devel` | `downloads_data.json` | after build workflows |

## 4. Evidence Files Consumed from Artifacts

- `build-timings.jsonl` / `sccache-*.json` / `ninja-*` / `.ninja_log`
  (Windows diagnostics artifact): the only sources for stage-level timeout
  attribution and compiler-cache A/B once the ephemeral runner is gone.
  Written by `util/ci_utils.ps1` (`Invoke-CMakeStage`, `Write-SccacheStats`),
  whose contracts are unit-tested by `util/tests/test_ci_utils.ps1`.
- `.ccache-stats.txt` (Ubuntu docker builds): rendered into the job summary
  by the "Report ccache stats" step.

## 5. Maintenance Notes

- Consolidated platform test scripts (invoked one-line from their workflow;
  each is also runnable locally):
  - Windows: `scripts/platforms/windows/test_cumesh_smoke.ps1` — the CUDA
    lane's C++ contract test: a standalone plain-CUDA consumer (outside the
    main build tree) builds and its binaries are DLL-self-contained under a
    system-only PATH. It cannot live inside `tests.exe` by construction: the
    subject is standalone-ness of a consumer of the build, not code in it.
  - Windows: `scripts/platforms/windows/test_installer.ps1` — silent
    installer install, DLL-path setup, `ACV_BINARY` detection and
    `STATUS_DLL_NOT_FOUND` triage (plus `-Cleanup`).
  - macOS: `scripts/platforms/mac/test_installer.sh` — DMG attach, installer
    copy off the volume, Qt IFW silent install, detach retry, `ACV_BINARY`
    detection (plus `--cleanup`), invoked one-line from `macos.yml`.
- Tool-library unit tests: `util/tests/test_ci_utils.ps1` (pwsh 7.3+, run
  from `windows.yml`) covers the build-diagnostics layer of
  `util/ci_utils.ps1`; `util/tests/test_ci_utils.sh` (bash only, run from
  `ubuntu.yml` before any step sources the library) covers the strict-mode
  sourcing contract, the `CLOUDVIEWER_CMAKE_BIN` guard, the AICore `USE_*`
  option resolvers and the `python_release_deps_required` gate of
  `util/ci_utils.sh`.
- Windows C++ testing is unified in `Run-CppUnitTests`
  (`util/ci_utils.ps1`): `tests.exe` + foundation portability CTest
  executables + qPythonRuntime native gate — all unconditional parts of the
  entry (core/tests registers the foundation executables under
  `BUILD_UNIT_TESTS`, which every caller enables), shared by `windows.yml`,
  `build_win_app.ps1` and `build_win_wheel.ps1`.
- Windows compile-cache incident history and tuning rationale live in the
  step comments of `windows.yml` (ghac quota flood, release-asset 2 GiB
  thrashing, sccache version pinning).
