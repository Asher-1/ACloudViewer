# Qt / VTK / PCL Version Matrix & Environment Recipes

## Official compatibility evidence (verified via primary sources)

| Claim | Source |
|---|---|
| VTK supports Qt6 since **9.1** via `VTK_QT_VERSION` (`Auto`/`5`/`6`); adds `VTK::GUISupportQtQuick` | https://docs.vtk.org/en/latest/release_details/9.1.html |
| ParaView **6.0.0** fully supports Qt6; official binaries built/packaged with Qt6 | https://www.kitware.com/paraview-6-0-0-release-notes/ |
| VTK **9.4** loads OpenGL at runtime via `glad`; removes `VTK::opengl` module | https://docs.vtk.org/en/latest/release_details/9.4.html |

**Conclusion**: VTK 9.4.2 + Qt 6.5.3 is an officially supported combination. Qt5→Qt6
breakage in this repo is linkage/behavior, not version incompatibility.

## Upstream CloudCompare (reference, not justification)

| Line | Qt | Evidence |
|---|---|---|
| Stable release v2.13.2 (2024-07-11) | Qt 5.15.x | `.ci/conda-win.yml` `qt=5.15.*`; Ubuntu 22.04 system Qt5 |
| master (next 2.14) | **Qt 6.8.3** (Windows release pinned) | `.ci/conda-win.yml` `qt6-main==6.8.3`; macOS `>=6.4`; Ubuntu 24.04 system Qt6 |

- master switched to Qt6 on 2025-10-22 (`Compile with Qt6 (and Visual Studio 2022)`).
- 2025-11-05: conda-forge `qt6-main 6.9.3_1` broke Windows CI → pinned back to **6.8.3 LTS**.
- Removed Core5Compat dependency and QGamepad; macOS FBO uses raw OpenGL calls.
- Upstream master CI uses **PCL 1.15.1** (same as our Qt6 deps).
- Lesson: prefer **LTS** Qt6 (6.5.x / 6.8.x); non-LTS 6.9/6.10 carry package/behavior risk.

## Ubuntu apt Qt versions (Launchpad API + packages.ubuntu.com, verified)

| Ubuntu | Codename | Qt6 (`qt6-base-dev`) | Qt5 (`qtbase5-dev`) |
|---|---|---|---|
| 20.04 | focal | — (Qt5 only) | 5.12.8 |
| 22.04 | jammy | — (Qt5 only; Qt6 via aqt/self-build) | 5.15.3 |
| 24.04 | noble | 6.4.2 | 5.15.13 |
| **26.04** | **resolute** | **6.10.2** (`6.10.2+dfsg-7`, Release pocket) | **5.15.18** (`5.15.18+dfsg-1ubuntu1`) |

- 26.04 `qt6-5compat-dev` = 6.10.2-1. Qt5 and Qt6 coexist → dual-compat testable via apt.
- **Qt6 system packages only on Ubuntu 24.04+**; on 22.04 Qt6 must be self-deployed (aqt).

## Local environment recipes (this machine, Ubuntu 22.04)

### Qt5 baseline (`build_app`)
- Qt 5.14.2 at `/opt/Qt5.14.2/5.14.2/gcc_64` (`Qt5_DIR` in CMakeCache).
- VTK 9.3.1 / PCL 1.14.1 (Qt5-linked), system paths.

### Qt6 (`build_app_qt6`)
- Qt 6.5.3 (LTS) via aqt at `/opt/qt6/6.5.3/gcc_64` (non-conda, root-free).
- Qt6-linked deps at `/opt/qt6-deps`: **VTK 9.4.2** (`-DVTK_QT_VERSION=6`) + **PCL 1.15.1**
  built against that Qt6 VTK.
- Configure mirrors `build_app` options + `-DUSE_QT6=ON`; rpath includes
  `/opt/qt6-deps/lib` and `/opt/qt6/6.5.3/gcc_64/lib`; `--disable-new-dtags`.
- GLEW built with hidden visibility (see root-causes #1).

### Qt 6.4.2 (24.04 parity) + Qt 6.10.2 (26.04 parity) — VALIDATED locally
- aqt installs (mirror `https://qt.mirror.constant.com`; jaist timed out, tsinghua
  served checksum-corrupt archives): `/opt/qt6/6.4.2/gcc_64`, `/opt/qt6/6.10.2/gcc_64`
  (aqt 3.3.0 arch names: 6.4.2=`gcc_64`, 6.10.2=`linux_gcc_64`; modules
  `-m qtwebsockets qtshadertools` — qtsvg/qtwayland are in the base archive).
- Per-Qt deps prefixes (a Qt minor needs its OWN VTK/PCL — forward-ABI only):
  `/opt/qt64-deps`, `/opt/qt610-deps`, each **VTK 9.4.2** (`-DVTK_QT_VERSION=6`,
  `GUISupportQtQuick=DONT_WANT`, `VTK_WRAP_PYTHON=OFF`) + **PCL 1.15.1** (`WITH_QT=AUTO`).
  Recipe script: `/tmp/build_qt_deps.sh <QT_DIR> <PREFIX> <JOBS> <TAG>`.
- App trees `build_app_qt64` / `build_app_qt610` mirror `build_app_qt6` options.
- **Result**: full build 0 FAILED on both; qAnyLabeling 44/44 on both; VTK/layout GUI
  tests pass on both. Two Qt6.10-specific test fixes were required (root-causes #12, #13),
  both verified back-compatible on Qt 5.14.2 / 6.5.3 / 6.4.2 / 6.10.2.
- **Loader caveat (local aqt only)**: aqt Qt lives in a non-system path; transitive
  `libQt6OpenGL.so.6` (pulled via Qt libs' `RUNPATH=$ORIGIN`) is not resolved by the exe
  `DT_RPATH` alone for some GUI tests → run those with
  `LD_LIBRARY_PATH=<qt>/lib:<deps>/lib`. CI installs Qt6 via apt into system ldconfig
  paths, so this caveat does NOT apply to CI.
- **Confirmed pairing for CI**: Qt 6.4.2 (noble) and Qt 6.10.2 (resolute) both use
  VTK 9.4.2 + PCL 1.15.1.

## CI structure (landed, unified QT_MAJOR)

Single source of truth: **Qt major is a pure function of the Ubuntu release**.
`docker/docker_build.sh` `set_qt_vars()` derives `QT_MAJOR` / `QT_BASE_DIR` from
`UBUNTU_VERSION` (focal/jammy=5 + versioned qt5 prefix; noble/resolute=6 + `/usr`)
and passes `--build-arg QT_MAJOR` into `Dockerfile.ci` / `Dockerfile.wheel`. There
are **no parallel Qt6 files** (`docker_build_qt6.sh`, `Dockerfile.ci.qt6`,
`Dockerfile.wheel.qt6` were deleted — they duplicated the path with wrong package
names and mislabeled 26.04 as oracular).

- `.github/workflows/prebuild-deps.yml`: matrix focal/jammy/noble/**resolute**, each
  row carries `qt_major` / `vtk_version` / `pcl_version` / `qt_base_dir`
  (Qt5 → 9.3.1/1.14.1, Qt6 → 9.4.2/1.15.1). Builds via `Dockerfile.ci`
  `PREBUILD_DEPS_ONLY=ON` + `QT_MAJOR`, publishes
  `vtk-pcl-deps-<codename>-vtk<V>-pcl<P>.tar.gz` to the `ubuntu-prebuild-deps`
  release. The version pair in the name keeps Qt5/Qt6 artifacts distinct; each Qt
  minor needs its own VTK/PCL (forward-ABI only), hence one tarball per codename.
- `ubuntu.yml` / `ubuntu-cuda.yml`: matrix `cpu-*` / `cuda-*` ×
  focal/jammy/noble/**resolute** (+`-release`), all routed through
  `docker_build.sh` → `docker_test.sh` (same flat option names). agent-integration
  cells run on `ubuntu-latest` (the .run installer bundles its Qt libs; GitHub has
  no 26.04 hosted runner).
- `cuda-resolute` base image: `nvidia/cuda:13.5.0-cudnn-devel-ubuntu26.04` — NVIDIA
  publishes **only CUDA 13.x for ubuntu26.04** (Docker Hub tag API verified); the
  repo's arch list for toolkit ≥12 (75/80/86/89/90) is fully supported by CUDA 13.
  CUDA 13 caps host GCC at 14 while resolute defaults to gcc-15 → `Dockerfile.ci`
  installs gcc-14/g++-14 and shadows gcc/cc/g++/c++ via `/usr/local/bin` symlinks
  (only when `BUILD_CUDA_MODULE=ON` and codename=resolute).
- `ubuntu-wheel.yml`: `cuda_wheel_*` now builds on **noble/Qt6** (12.6.3 CUDA image
  exists for 24.04). Wheel tag follows build glibc (`python/setup.py`), so the
  artifact is `manylinux_2_39` and `test-wheel` runs on `ubuntu-24.04` (pip on a
  22.04 runner would refuse the wheel).
- Qt6 install branch (both Dockerfiles): `qt6-base-dev qt6-base-dev-tools
  qt6-declarative-dev qt6-svg-dev qt6-websockets-dev qt6-wayland-dev qt6-tools-dev
  qt6-shadertools-dev qml6-module-qtquick* libqt6opengl6-dev libgl1-mesa-dev
  libglx-dev` + `/etc/profile.d/qt6.sh` exporting `QT_PLUGIN_PATH` (sourced by
  `build_gui_app.sh` / `build_cloudviewer_whl.sh`, which auto-detect Qt6 via
  `qmake6` → `USE_QT6=ON`). Correct apt names are `qt6-svg-dev` /
  `qt6-websockets-dev` (NOT `libqt6svg6-dev` / `libqt6websockets6-dev`).
- Verification gate for CI edits (no local runner): `bash -n` every touched
  script, YAML-parse every workflow, source `docker_build.sh` and assert each
  matrix cell's `QT_MAJOR`/`BASE_IMAGE`, and cross-check every workflow
  `CI_CONFIG` has a case in both docker scripts.