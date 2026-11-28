---
name: acloudviewer-qt-compat
description: ACloudViewer Qt 5.12+/Qt6 dual-compatibility engineering contract. Use when porting or debugging Qt5→Qt6 issues (GL/VTK crashes, GLEW symbol hijack, rpath, offscreen GL, QFileDialog/QAction behavior changes), adding cross-version shims to QtCompat.h, wiring Qt6-linked VTK/PCL dependencies, or extending Linux CI (prebuild-deps / ubuntu / ubuntu-cuda / ubuntu-wheel) to Ubuntu 26.04 native Qt6.
---

# Qt 5.12+ / Qt6 Dual-Compatibility Skill

Single source of truth for cross-version shims: [`core/include/QtCompat.h`](../../../core/include/QtCompat.h).
Detailed root-cause ledger: [`references/root-causes.md`](references/root-causes.md).
Official version evidence + environment recipes: [`references/qt-version-matrix.md`](references/qt-version-matrix.md).

## When to use

- A Qt6 build crashes at startup in `libqxcb-glx-integration.so` / `QOpenGLContext::create()`.
- GL/VTK symbols resolve to the wrong library, or a test segfaults **after all tests passed**.
- Adding a helper that must behave identically on Qt 5.12 … Qt 6.x.
- Building Qt6-linked VTK/PCL, or pushing Linux CI to Ubuntu 26.04 (native Qt 6.10.2).
- A Qt5 test passes but the same test fails on Qt6 (or vice versa).

## Golden rules (derived, not conventional)

1. **It is almost never "VTK incompatible with Qt6".** VTK ≥ 9.1 officially supports
   Qt6 (`VTK_QT_VERSION=6`); ParaView 6.0 official binaries are Qt6-built. The real
   causes are this repo's GL symbol linkage, rpath propagation, and Qt6 *behavior*
   changes. See the version matrix before blaming versions.
2. **Solve at the owner.** Qt version differences go into `QtCompat.h` (QtCore/QtWidgets
   level) — never scattered `#if QT_VERSION` at call sites. A shim needing a module not
   every consumer links (e.g. Qt Svg) stays in the private holder (qlementine port's
   `QtCompat.hpp`), because the unified header is consumed repo-wide via `CVTools.h`.
3. **Forward-fix, never evade.** A failing test is an unlocated bug. Do not skip, weaken
   assertions, or mark "legacy" to silence it. Locate the root cause (gdb / minimal
   probe) and fix the owning code path. Both Qt5 and Qt6 must pass after the fix.
4. **QApplication lifetime is a stack object in `main()`.** A function-local `static
   QApplication` is destroyed inside `__run_exit_handlers` and crashes in
   `~QGuiApplicationPrivate` (SIGSEGV after every test passed). Custom `main` + stack
   `QApplication` + link `gtest` (not `gtest_main`).
5. **Verify on every affected Qt major before handing back.** A fix is not done until
   the relevant `ctest` subset passes on **both** `build_app` (Qt5) and `build_app_qt6`
   (Qt6). Build with `-j24` max (OOM guard).

## Root-cause quick table

| Symptom | Essence | Fix owner |
|---|---|---|
| Startup `0x0` call in `libqxcb-glx-integration.so` | GLEW static symbols hijack Qt's GL resolution | `3rdparty/glew/CMakeLists.txt` `*_VISIBILITY_PRESET hidden` |
| Same crash only on Qt 6.4-generation builds (Qt5/6.5+ pass) | Precompiled filament `libbluegl.a` GL symbols hijack Qt 6.4 Gui | `--exclude-libs,libbluegl.a` on `3rdparty_filament` in `3rdparty/find_dependencies.cmake` (never `objcopy --localize-symbols`: it breaks `libbackend.a`'s static refs; verify by relinking `CloudViewerApp` without system libGL) |
| Qt6/VTK libs not found at runtime | DT_RUNPATH not transitive to indirect deps | `--disable-new-dtags` (DT_RPATH) + shared-lib rpath; `tests/Runtime.cmake`, `libs/Reconstruction/cmake/CMakeHelper.cmake` |
| VTK 9.4 GL header / `VTK::opengl` missing | 9.4 loads GL at runtime via glad | VTK-version macro gating in VtkEngine |
| Qt6 offscreen test: no GL context | offscreen platform has no `QOpenGLWidget` GL | `xvfb-run -a` + `QT_QPA_PLATFORM=xcb` (qAnyLabeling tests CMake) |
| `QFileDialog` accepts stale dir / won't close | Qt6 `selectFile()` async until model loads | `QtCompat.h` feature 28 `qtCompatFileDialogSelectAndAccept` |
| Qt6 `QAction::trigger()` does nothing | Qt6 honors enabled state (Qt5 fired disabled) | open the owning panel first (test side); app behavior is correct |
| Test SIGSEGV after `[ PASSED ]` | static QApplication torn down at atexit | custom `main` + stack QApplication; link `gtest` |
| Markdown bold export differs | Qt6 `QFont::Bold` weight export | `QtCompat.h` feature 27 |
| qlementine EP compile fail on Qt5 | stale `Qt5_DIR` cache / 6.7-only API | `QtCompat.h` features 23/24; check `CMakeCache` `*_DIR` first |

## Mandatory workflow

1. **Classify** — reproduce on both Qt5 and Qt6 baselines. Same failure on both = a
   version-independent bug (fix benefits both); Qt6-only = behavior/porting issue.
2. **Locate** — gdb backtrace for crashes; minimal standalone probe for behavior
   changes (compile a tiny Qt program, do not guess). Read the owning code path.
3. **Fix at the owner** — shim → `QtCompat.h`; linkage → the dependency's CMake;
   test harness → the test registration / `main`. Delete the wrong path, don't guard it.
4. **Verify both majors** — rebuild affected targets in `build_app` and `build_app_qt6`,
   run the focused `ctest -R` subset, repeat flaky-prone tests 3×.
5. **Record** — append the new symptom→essence→fix triple to `references/root-causes.md`.

## ctest / build pitfalls (verified)

- `ctest -N -V` prints the **original** `add_test` command, not the `COMMAND` property
  override (e.g. `xvfb-run` wrapper). Read `CTestTestfile.cmake` for the real command.
- `ENVIRONMENT` list entries apply in order; a later duplicate wins. Remove stale
  `QT_QPA_PLATFORM=offscreen` before appending `xcb`.
- Ninja **target** name ≠ ctest **test** name (e.g. target `test_qanylabeling_run_path`,
  test `test_qanylabeling_attribute_run_path` = same binary + `--attribute`).
- Never run two ninja processes in the **same** build dir concurrently; wait for the
  full build before a target build. Different build dirs are independent.
- The Bash tool resets cwd to `/tmp` between calls — always `cd` to an absolute path.
- `qmake -project` scans the whole cwd; never run it at the repo root (hangs).

## CI / Ubuntu 26.04

See [`references/qt-version-matrix.md`](references/qt-version-matrix.md) for the apt
version facts and the landed CI wiring. Ubuntu 26.04 ships Qt 6.10.2 natively
(`qt6-base-dev`) and Qt 5.15.18 (`qtbase5-dev`) — no aqt layer needed there.
Linux CI is unified on one `QT_MAJOR` knob derived from the Ubuntu release in
`docker_build.sh` (`set_qt_vars`); never add a parallel Qt6 Dockerfile/script.
`cuda-resolute` uses CUDA 13.x + gcc-14 (NVIDIA has no 12.6.3 ubuntu26.04 image);
wheels build on noble/Qt6 (`manylinux_2_39`, test runner ≥ 24.04).