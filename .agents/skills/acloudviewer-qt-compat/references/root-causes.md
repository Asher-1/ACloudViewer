# Qt5/Qt6 Root-Cause Ledger

Each entry: **Symptom → Essence → Fix (owner) → Verification**. Append new entries here.

## 1. Startup SIGSEGV: GLEW static symbol hijack

- **Symptom**: `0x0000000000000000 in ?? ()` called from
  `/opt/qt6/.../plugins/xcbglintegrations/libqxcb-glx-integration.so` →
  `QOpenGLContext::create()` → `QGuiApplicationPrivate::init()` → `ecvApplicationBase`.
- **Essence**: the executable statically links GLEW with default symbol visibility, so
  GLEW's `glGetString`/`glXGetProcAddress`-resolved functions are exported as dynamic
  symbols. The dynamic linker binds Qt's xcb-glx plugin GL calls to GLEW's
  **uninitialized** function pointers → call through NULL.
- **Fix**: `3rdparty/glew/CMakeLists.txt` set `C_VISIBILITY_PRESET hidden` and
  `CXX_VISIBILITY_PRESET hidden` on the GLEW target. Verify with
  `nm -D build_app_qt6/bin/ACloudViewer | grep glGetString` → must be `U` (undefined,
  resolved from system libGL at runtime), not `T`/`W`.
- **Verification**: GUI smoke under xvfb starts and renders; no startup crash.

## 2. Runtime library resolution: DT_RPATH vs DT_RUNPATH

- **Symptom**: Qt6/VTK 9.4 libraries under `/opt/qt6*` not found when launching the exe
  or loading plugins, despite rpath being set.
- **Essence**: `DT_RUNPATH` (new dtags) is **not transitive** — it only applies to the
  direct dependencies of the object that carries it, not indirect ones. `DT_RPATH`
  (old dtags) is transitive.
- **Fix**: link with `-Wl,--disable-new-dtags` to emit `DT_RPATH`; add the dependency
  lib dirs (`/opt/qt6-deps/lib`, `/opt/qt6/6.5.3/gcc_64/lib`) to shared-library rpath.
  Owners: `tests/Runtime.cmake` (test targets), `libs/Reconstruction/cmake/CMakeHelper.cmake`,
  and configure-level `CMAKE_BUILD_RPATH`/`CMAKE_INSTALL_RPATH` for the app.
- **Verification**: the rpath-dependent test subset (12 tests) passes; `ldd` resolves all libs.

## 3. VTK 9.4 glad runtime GL loading

- **Symptom**: VTK 9.4 build fails on `VTK::opengl` module / GL headers.
- **Essence**: VTK 9.4 "loads the appropriate OpenGL library at runtime using `glad`
  and no longer provides the `VTK::opengl` module" (official 9.4 release notes).
- **Fix**: gate GL header inclusion / module linkage by VTK version macros in VtkEngine.
- **Verification**: VtkEngine compiles against VTK 9.4.2 (Qt6) and 9.3.1 (Qt5).

## 4. Qt6 offscreen platform has no GL

- **Symptom**: GL-widget tests fail/skip under `QT_QPA_PLATFORM=offscreen` on Qt6.
- **Essence**: the offscreen platform plugin does not provide a real OpenGL context for
  `QOpenGLWidget`.
- **Fix**: run GL tests under `xvfb-run -a` with `QT_QPA_PLATFORM=xcb`. Owner:
  `plugins/core/Standard/qAnyLabeling/tests/CMakeLists.txt` (COMMAND override + remove
  stale `offscreen` ENVIRONMENT entry before appending `xcb`).
- **Verification**: 452/449 GL workflow tests pass.

## 5. QFileDialog::selectFile() is async on Qt6 (feature 28)

- **Symptom**: test times out "Select a specific flags file", or exports to the wrong
  (stale) directory.
- **Essence**: Qt5 `selectFile()` takes effect synchronously; Qt6 only applies it once
  the directory model loads. Until then `selectedFiles()` reports the **old** directory
  (non-empty but wrong), and `accept()` validates `selectedFiles()` — rejecting the
  close or accepting the wrong path. In Directory mode `directory()` does not follow
  `selectFile`.
- **Fix**: `QtCompat.h` feature 28 `qtCompatFileDialogSelectAndAccept()` polls until
  `QDir::cleanPath(selectedFiles().value(0)) == QDir::cleanPath(path)` then accepts.
  `afterAccept` uses `qApp` as context (a stack chooser is destroyed right after accept).
- **Verification**: 452 passes 5/5; classifier export lands in the intended directory.

## 6. QAction::trigger() honors enabled state on Qt6

- **Symptom**: 483 `attribute_run_path` — `launchBatch selected=0`, batch range dialog
  never appears, status stays "Inference inputs changed; run again". Qt5 passes.
- **Essence**: Qt6 `QAction::activate()` returns early when the action is disabled;
  Qt5 fired `triggered` even for disabled actions. `autoRun` is legitimately disabled
  because `m_modelPanel->isVisible()` is false — the test never opened the AI panel.
- **Fix (test side, app is correct)**: toggle the `auto_label` toolbar action
  (`setChecked(true)`) to open the panel before triggering `auto_run`. Applied in both
  the attribute and relate Auto-Run sections of `test_run_path.cpp`.
- **Verification**: 483 passes 3/3 on Qt6; default run_path passes; Qt5 still passes.

## 7. Test SIGSEGV after all tests passed: static QApplication at atexit

- **Symptom**: 476/477 — `[ PASSED ] 9 tests` then SIGSEGV in
  `QGuiApplicationPrivate::~QGuiApplicationPrivate()` from `__run_exit_handlers`.
- **Essence**: a function-local `static QApplication` (in `SetUpTestSuite`) is destroyed
  during C++ atexit teardown, interleaved with Qt's own global-static destructors →
  crash. QApplication must be destroyed while Qt internals are still alive.
- **Fix**: custom `int main()` with a **stack** `QApplication` (destroyed when main
  returns, before atexit), `InitGoogleTest` + `RUN_ALL_TESTS`; remove the static
  `SetUpTestSuite` app; link `gtest` instead of `gtest_main`. Owners:
  `test_relation_filter.cpp`, `test_relate_options.cpp`, qAnyLabeling tests `CMakeLists.txt`.
- **Verification**: 476/477 pass on Qt6 and Qt5 (was SEGFAULT on both).

## 8. Model-test asset provisioning (486 LLM batch)

- **Symptom**: 486 `llm_batch_models` fails "Every real LLM asset and plugin must exist".
- **Essence**: the test reads `AICORE_TEST_LLM_TEXT_MODEL` / `_VISION_MODEL` / `_MMPROJ`
  but nothing wired them. The AICore model-test env owner is
  `aicore_configure_model_test_assets()` (`core/AICore/tests/CMakeLists.txt`), resolving
  paths from the `AICORE_TEST_ASSET_ROOT` cache var (`~/cloudViewer_data/extract`).
  core is processed before plugins, so the plugin test wires its own env from that same
  cache var (filenames mirror `core/AICore/src/tasks/llm/model_catalog.cpp`).
- **Fix**: in qAnyLabeling tests `CMakeLists.txt`, append the three LLM env paths
  (`llm_models/qwen2.5-0.5b-instruct-q8_0.gguf`, `SmolVLM-500M-Instruct-Q8_0.gguf`,
  `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf`) to `test_qanylabeling_llm_batch_models`.
- **Verification**: 486 passes (24.7s real inference) on Qt6 and Qt5.

## 9. qlementine ExternalProject port (features 23/24)

- **Symptom**: qlementine EP fails on Qt 5.14.2 (`QTabBar::isTabVisible`,
  `SE_PushButtonBevel` missing); `-Wswitch` error on `SE_CustomBase+N`.
- **Essence**: those are Qt 6.7+ APIs; `SE_CustomBase+N` is out of enum range and cannot
  be a `case` label under `-Wswitch -Werror`.
- **Fix**: `QtCompat.h` features 23 (`qtCompatTabBarIsTabVisible`) / 24
  (`qtCompatPushButtonBevelSubElement`); intercept out-of-range values with `if` before
  the `switch`. SVG shim stays in the port-private `QtCompat.hpp` (module boundary).
  `qlementine.cmake` folds the unified header SHA-256 into `source_id`.
- **Pitfalls**: `find_package` prefers cached `Qt5_DIR` over `CMAKE_PREFIX_PATH` — check
  `CMakeCache` first. EP patch regeneration needs `git add -A` then `git diff --cached`
  (plain `git diff` omits new untracked files). Use `CMAKE_SOURCE_DIR`, not
  `PROJECT_SOURCE_DIR`, in included subproject context.

## 10. Bare Xvfb window-activation race (452)

- **Symptom**: "Host default shortcut was not suspended" intermittently.
- **Essence**: a bare Xvfb display has no window manager; show/activate events arrive
  async and interleave. `syncHostShortcuts` is driven by WindowActivate/Deactivate →
  `singleShot(0)` reading `activeWindow()`; a late Deactivate restores the host action.
- **Fix (test side)**: re-assert activation until the suspension survives a settling
  window (20 consecutive stable iterations), drop trailing blind `processEvents`.
- **Verification**: 452 passes 5/5.

## 11. Fake virtual override wrapped in `#if QT_VERSION` (evasive-fix trap)

- **Symptom**: `model_viewer_widget.h` switched `cleanupGL()` `override` on/off via
  `#if QT_VERSION`; flipping the branch only moves the compile error between build trees.
- **Essence**: `QOpenGLWidget::cleanupGL()` does not exist in **any** Qt version (the only
  virtuals are `initializeGL/resizeGL/paintGL`) — both branches override a non-existent
  virtual. The real invariant is "GPU resource deletion needs a current GL context";
  the `#if` made Qt5 run a context-less manual `Release()` and Qt6 skip release entirely
  (silent runtime divergence; `QOpenGLVertexArrayObjectPrivate::destroy()` does
  `if (vao && ctx)` — no ctx, no delete).
- **Fix**: delete the whole `#if` block and the manual cleanup; express the invariant
  version-neutrally — derived destructor body calls `makeCurrent()` (no `doneCurrent()`),
  letting value-member `QOpenGLShaderProgram/VAO/Buffer` RAII destructors delete under a
  current context. Delete the dead `painter Release()` (its only caller was the removed
  cleanup). Matches upstream COLMAP `~ModelViewerWidget() { makeCurrent(); }`.
- **Lesson**: never flip/add a `#if QT_VERSION` branch to make one tree compile — grep the
  actually-installed Qt headers to confirm which virtuals exist, then express the real
  invariant version-neutrally. Verify with `nm -C` that the old symbols vanish and rebuild
  downstream TUs (single-target build is not proof).
- **Verification**: ColmapLib builds on Qt 5.14.2 and Qt 6.5.3; net -63 lines, zero
  `QT_VERSION` branches; GPU resources provably released on both.