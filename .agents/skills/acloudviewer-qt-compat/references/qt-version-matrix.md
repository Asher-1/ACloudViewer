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

### Qt 6.10.2 validation (Ubuntu 26.04 parity, pending)
- Goal: prove the Qt6 port holds on 6.10.2 (26.04's native Qt) before pushing CI.
- Deploy Qt 6.10.2 via aqt to `/opt/qt6/6.10.2/gcc_64`; rebuild VTK 9.4.2 / PCL 1.15.1
  against it into a separate deps prefix; configure `build_app_qt610` mirroring
  `build_app_qt6`; build + run the affected ctest subset. Record results here.

## CI structure (current)

- `.github/workflows/prebuild-deps.yml`: builds the heavy source dep layer once per
  Ubuntu release via `docker/Dockerfile.ci` `PREBUILD_DEPS_ONLY=ON`, publishes
  `vtk-pcl-deps-<codename>-vtk<V>-pcl<P>.tar.gz` to the `ubuntu-prebuild-deps` release.
  Current: VTK 9.3.1 / PCL 1.14.1, `QT_BASE_DIR=/usr/lib/x86_64-linux-gnu/qt5`,
  matrix focal/jammy/noble.
- `ubuntu.yml` / `ubuntu-cuda.yml` / `ubuntu-wheel.yml` consume the tarball (download +
  extract) instead of recompiling VTK/PCL.
- Qt6/26.04 plan: add a resolute matrix entry with native Qt6 (`qt6-base-dev`),
  Qt6-linked VTK/PCL prebuild, and `USE_QT6=ON` app build. (Implementation tracked in
  the task list; update this section with the verified recipe once landed.)