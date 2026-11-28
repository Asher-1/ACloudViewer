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
- **Qt 6.4 is worse (no GL is not the only problem)**: the full GUI under
  `QT_QPA_PLATFORM=offscreen` on Qt 6.4.2 (noble) segfaults on the first
  expose as soon as a `QOpenGLWidget` is on screen —
  `QPlatformBackingStore::rhiFlush` ← `QWidgetRepaintManager::flush` ←
  `QApplicationPrivate::notify_helper` ← `processExposeEvent` (Qt's own
  bug; Qt 6.10 and Qt 5.15 only print "QOpenGLWidget is not supported on
  this platform" and stay up). Reproduced locally: qt64 offscreen rc=139,
  qt64/qt610/qt5 xcb and qt610/qt5 offscreen alive after 15 s. So a step
  that already runs inside `xvfb-run` must not force `offscreen`; owner
  `.github/workflows/ubuntu.yml` Level 4 (`QT_QPA_PLATFORM: xcb`).

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

## 12. Qt 6.10 `QJsonValueRef == QJsonArray` ambiguous overload

- **Symptom**: `test_run_path.cpp` `readJson(sidecar)["shapes"] == detected` fails to
  compile on Qt 6.10.2: "ambiguous overload for operator== (QJsonValueRef and const
  QJsonArray)". Compiles fine on Qt 5.x / 6.5.3.
- **Essence**: Qt 6.10 added `Q_DECLARE_EQUALITY_COMPARABLE` overloads
  (`QJsonValueConstRef==QJsonValue` and `QJsonArray==QJsonValue`). A non-const
  `QJsonObject::operator[]` yields `QJsonValueRef`; comparing it to a concrete
  `QJsonArray` now has two viable conversion paths → ambiguity. Older Qt had fewer
  overloads so it resolved.
- **Fix (version-neutral)**: compare concrete arrays —
  `readJson(sidecar)["shapes"].toArray() == detected` (and `.toArray()` on both sides
  when both are refs). `QJsonArray==QJsonArray` is unambiguous on every Qt version and
  semantically identical (the value IS an array). Do NOT add `#if QT_VERSION` casts.
- **Verification**: builds + passes on Qt 5.14.2 / 6.5.3 / 6.4.2 / 6.10.2.

## 13. Qt 6.10 canonicalizes HTTP header names (case)

- **Symptom**: `test_chat.cpp` asserts `headers.contains("x-api-key: test-key")` on the
  raw request captured by a local QTcpServer; passes on 6.5.3, fails on 6.10.2.
- **Essence**: Qt 6.10's `QNetworkAccessManager` (QHttpHeaders) normalizes header NAMES
  to canonical title-case: `setRawHeader("x-api-key",...)` is sent as `X-Api-Key`,
  `anthropic-version` as `Anthropic-Version`. Header VALUES are unchanged. HTTP header
  names are case-insensitive (RFC 7230), so the test's case-sensitive name match was
  always fragile.
- **Fix (version-neutral)**: match header names case-insensitively —
  `headers.toLower().contains("x-api-key: test-key")`. Keep request-line/path checks
  case-sensitive; value-only substrings (e.g. `Bearer test-key`) are unaffected.
- **Diagnosis tip**: dump the captured `headers` QByteArray (fprintf %s) — grep shows
  only the first line because the buffer embeds `\r\n`; use `grep -A15`.
- **Verification**: chat test passes on Qt 5.14.2 / 6.5.3 / 6.4.2 / 6.10.2.

## 14. Concurrent builds sharing `3rdparty_downloads` truncate a fetch

- **Symptom**: two app build trees configured/building in parallel both fetch the same
  ExternalProject archive (e.g. spirula) into the shared `3rdparty_downloads/` cache;
  one writes a truncated tar → other's extraction misses files (`Languages.h` missing,
  `cmake -E tar: Truncated tar archive`).
- **Essence**: the download cache is a shared mutable resource with no lock; concurrent
  writers race. Not a Qt issue.
- **Fix**: after the race, the cached tar is usually complete (`gzip -t` OK). Remove the
  affected build tree's `<dep>/` ExternalProject dir entirely and RE-CONFIGURE (deleting
  only the stamp dir breaks ninja's generated rule graph: "patch-info.txt missing, no
  known rule"). Then rebuild. Avoid by serializing fetch-heavy phases or per-tree caches.
- **Verification**: reconfigure + rebuild extracts cleanly; build proceeds.
## 15. Startup SIGSEGV on Qt 6.4 + new Mesa: precompiled bluegl symbol hijack

- **Symptom**: the installed `.run` binary segfaults at `QApplication` init
  (`QOpenGLContext::create()` → `libqxcb-glx-integration.so` → call to `0x0`)
  on Ubuntu 24.04/26.04 runners with Mesa ≥ 25.2; Qt5 and Qt ≥ 6.5 builds pass
  on the same host. `LD_DEBUG=bindings` shows `libQt6Gui.so.6` binding
  `glGetString` to the **ACloudViewer executable itself**, not libGL.
- **Essence**: same class as entry 1, different owner: the precompiled filament
  archive `libbluegl.a` ships real GL entry points (`glGetString`, …, ~2574
  symbols) as globals in its assembly object. When the link pulls that member
  in, the executable exports them and interposes Qt 6.4 Gui's direct
  `glGetString` reference; bluegl's pointer table is still uninitialized →
  jump to NULL. Qt ≥ 6.5 resolves GL through its own dispatch and never binds
  the exe symbol, which is why only the 6.4-generation build crashes. Mesa
  version is a red herring — the container's llvmpipe stack was healthy
  (`glxinfo` OK).
- **Fix**: `3rdparty/find_dependencies.cmake`, right after
  `import_3rdparty_library(3rdparty_filament ...)`, adds
  `target_link_options(3rdparty_filament INTERFACE "LINKER:--exclude-libs,libbluegl.a")`
  for GNU on Linux — the repo's existing `import_3rdparty_library` HIDDEN
  idiom, covering both the download and the source-build branch. Only the
  dynamic export stops; `libbackend.a`'s references (`glInsertEventMarkerEXT`,
  `glPushGroupMarkerEXT`, `glPopGroupMarkerEXT`) keep resolving to bluegl
  statically. Do NOT use `--exclude-libs,ALL`: plugins legitimately consume
  exe-exported LZ4 symbols.
- **Failed attempt (lesson)**: `objcopy --localize-symbols` on the archive
  (`localize-bluegl` ExternalProject step, reverted) made bluegl's symbols
  STB_LOCAL, and a local symbol cannot satisfy references from another
  object: CI cpu-noble/cuda-noble/cuda-resolute failed to link `CloudViewerApp`
  with `undefined reference to glInsertEventMarkerEXT` from
  `libbackend.a(OpenGLDriver.cpp.o)`. The dev trees passed only because the
  host link line also carries `/usr/lib/x86_64-linux-gnu/libGL.so` (mesa
  exports those EXT functions). A CI-equivalent check must (a) link
  `CloudViewerApp` — the executable CI actually links; `CloudViewer` is a
  static lib — and (b) drop the system `libGL.so` from the link line.
- **Diagnosis tip**: `nm -D <exe> | grep " T gl"` — any hit is a hijack risk;
  `LD_DEBUG=bindings` proves which object Qt bound to. Decisive runtime proof:
  gdb-rewrite the exe's leaked GL wrappers to jump into real libGL → crash
  disappears and the same CLI command produces its output file.
- **Verification**: CI-equivalent relink of the Qt 6.4 `CloudViewerApp` without
  system libGL: with `--exclude-libs` rc=0 and 0 gl* dynamic exports; without
  it rc=0 and 2574 exports (the leak condition, reproduced). Three trees
  (Qt 5.15.2 / 6.4.2 / 6.10.2): both executables export 0 gl* symbols, LZ4
  exports intact, headless PCV rc=0 (1875-byte output), binding contracts
  50/50. Earlier decisive proof: CI noble artifact + gdb redirect of the
  leaked GL wrappers → PCV output produced.

## 16. Qt ≥ 6.5 forces C.UTF-8: C-locale child assertions must accept both

- **Symptom**: `test_binding_contracts.py` unicode-path cases assert the
  embedded-Python child's `LC_CTYPE == "C"`; fails only on Qt ≥ 6.5 builds
  (`C.UTF-8` observed), passes on Qt5 / Qt 6.4.
- **Essence**: Qt ≥ 6.5 switches a C/POSIX `LC_CTYPE` to `C.UTF-8` at
  `QCoreApplication` init ("Qt depends on a UTF-8 locale"); libQt6Core strings
  contain the message, 6.4/Qt5 do not. Not glibc, not Python (isolated config).
- **Fix**: accept `("C", "C.UTF-8")` for `child_locale`; keep the strict
  `filesystem_encoding == "utf-8"` assertion — that is the real Unicode guard.

## 17. Windows install: `cloudViewer_install_ext` takes ONE file per call

- **Symptom**: Windows Release build compiles for ~72 min (exit 0) then the
  install step fails with `file cannot create directory:
  .../build/external/bin/opencv_videoio_ffmpeg470_64.dllC:/dev/cloudViewer_install/bin/ACloudViewer`.
- **Essence**: `cloudViewer_install_ext` (`cmake/CMakeInclude.cmake`) is
  positional: `ARGV1` = the single source file, `ARGV2` = the destination,
  `ARGV3` = the suffix. `opencv_build.cmake` passed a two-element file list
  (`opencv_world470.dll` + the ffmpeg plugin DLL), so the second path became
  the destination. Not a compile error — the main job log is silent because
  `Tee-Object` buffers; the real error is at the end of the
  `windows-diagnostics-<Config>-cuda<ON|OFF>` artifact's `_temp/windows-build.log`
  and `_temp/build-timings.jsonl` (`app-ON-build` exit 0,
  `app-ON-install-and-package` exit 1).
- **Fix**: loop `foreach(_opencv_install_file ${_opencv_install_files})`
  with one `cloudViewer_install_ext( FILES ${_opencv_install_file}
  ${INSTALL_DESTINATIONS} "")` per file — the form used before the ffmpeg DLL
  was appended, and the same shape as `suitesparse_build.cmake`.
- **Verification**: Windows-only path; no local Windows build — CI is the gate.

## 18. Wheel on CUDA 13.x: `torch==<pin>+cu<toolkit>` does not exist

- **Symptom**: wheel resolute (CUDA 13.5 image) `pip install` fails with
  `No matching distribution found for torch==2.7.1+cu135`.
- **Essence**: PyTorch publishes a fixed, version-dependent CUDA build set
  (2.7.1: cu118/cu126/cu128; cu13x only from 2.9.0 on). Bumping the tag alone
  is not enough: `torch.utils.cpp_extension._check_cuda_version` raises when
  the toolkit CUDA **major** differs from the one torch was built with (a
  minor mismatch only warns), so CUDA 13.x needs a torch with a cu13x build.
- **Fix**: `util/ci_utils.sh` `resolve_torch_cuda_build` reads the same pip
  index (`TORCH_REPO_URL`) and picks the lowest torch ≥ `TORCH_VER` with a
  same-major `+cuNNN` not above the toolkit (else the lowest same-major tag);
  unreachable index → pinned version + raw tag (old behavior). Callers that
  only use `+cpu` (`test_wheel`, tiered unit-test lane) are unaffected.
- **Verification**: resolver cases (118/121/126/128/129/130/132/135/112,
  unreachable, empty) pass; `util/tests/test_ci_utils.sh` 22/22. The CUDA
  13.5 container itself is not buildable locally (image pull blocked) — CI is
  the gate for `CloudViewer-ML` building against the resolved torch.
- **CI verdict (494e2fe65f, wheel resolute)**: the build now passes; the
  failure moved to `Test wheel resolute` step "Test Python package (CUDA)":
  `Version mismatch: CloudViewer needs PyTorch version 2.9.*, but version
  2.7.1+cpu is installed!`. The resolver had only been applied to the build
  side; `test_wheel` still installed the ML repo's `requirements-torch.txt`,
  which pins `torch==2.7.1+cpu`, while the ops were compiled against 2.9 and
  `cloudViewer/ml/torch/__init__.py` checks `_build_config["Pytorch_VERSION"]`
  major.minor on import. Lesson: a version that one stage *derives* must be
  read back from the artifact by every later stage, never re-derived from the
  pin.
- **Fix (test side)**: `test_wheel` reads
  `cloudViewer._build_config['Pytorch_VERSION']` from the installed wheel and,
  when it differs from `TORCH_VER`, installs a copy of the ML requirements
  with `torch==<built>+cpu` and `torchvision` unpinned (pip picks the release
  that requires exactly that torch: 0.24.0+cpu ↔ 2.9.0). Equal version → the
  file is used unchanged (CPU wheel, macOS, CUDA ≤ 12 cells).
- **Verification**: `bash -n`, `test_ci_utils.sh` 22/22, three-branch
  simulation with a stub `cloudViewer` (2.9.0+cu135 → rewritten file;
  2.7.1+cpu and empty → original file), and a pip dry-run of the rewritten
  file for cp313 resolving `torch-2.9.0+cpu torchvision-0.24.0+cpu`.
- **Open**: `util/ci_utils.ps1` still builds the raw tag; Windows CI is on
  CUDA 12.6 (`+cu126` exists), so port the resolver when its CUDA is bumped.

## 19. Windows: a file that is still open cannot be removed or replaced

- **Symptom**: Windows unified headless C++ gate (343/347):
  `test_qanylabeling_exporter` "Cannot stage existing PPOCR artifact:
  Label.txt"; `test_qanylabeling_video_ui` "exported clip replace or JPEG
  failed" followed by a 1200 s timeout. Green on Linux/macOS.
- **Essence**: POSIX unlink/rename ignores open handles; Windows refuses
  both. Both tests held the output open themselves: `test_exporter.cpp`
  kept reading `ppocrLabels`/`ppocrGt` (`QFile`) while the replacement
  export moved them aside (`AnnotationPpocr.cpp` `movePath` →
  `QFile::remove`), and `test_video_ui.cpp` kept `cv::VideoCapture
  exportedClip(clipExport)` open while the second export replaced the clip
  through `QSaveFile`.
- **Fix**: the test releases what it holds before asking for a replacement
  (`ppocrLabels.close(); ppocrGt.close();` / `exportedClip.release();`).
  Production code is unchanged — it is the caller's handle, not an exporter
  bug. Also fixed the Windows-only runner assertion
  `Path(...).name == "test_relate_temporal_replay"` → `.stem`
  (`core/AICore/scripts/test_validate_all.py`; `.exe` suffix).
- **Verification**: three test targets pass on Qt 6.4.2 and Qt 5.15.2 trees;
  `test_validate_all` 166/166. No local Windows build — CI is the gate.
- **The 1200 s hang is a cascade, not a second bug**: once an assertion fails,
  the dialog is still dirty and the fixture's `TearDown() { dialog.close(); }`
  blocks in `closeEvent → confirmSave → QMessageBox::question` (reproduced on
  Linux with Qt 6.4; stack taken by launching the test under gdb — ptrace_scope
  forbids attaching — and interrupting the inferior).
  `set_tests_properties(test_qanylabeling_video_ui PROPERTIES TIMEOUT 300)`
  bounds it; the product behaviour (ask before discarding) is unchanged.

## 20. Windows: OpenCV narrow-string file APIs take the ANSI code page

- **Symptom**: `test_qanylabeling_video_dataset` `ExportsUnsavedUnicodeLabel`
  fails only on Windows ("Failed to write frame"): the RawFrames export
  writes into `rawframes/行人/…`.
- **Essence**: `cv::imwrite(path.toStdString(), …)` and friends call
  `fopen` on a `const char*`; on Windows that is the ANSI code page, so a
  UTF-8 path with non-ASCII characters cannot be opened. `QFile` uses the
  wide API. (The Video path is safe: ffmpeg is a `QProcess` with
  `QStringList` arguments.)
- **Fix**: `VideoSegmentExporter.cpp` encodes with `cv::imencode(".jpg")`
  and writes the buffer through `QFile`. No repository-wide `imwrite`
  wrapper exists; the one `imwrite` in qAnyLabeling `src` was the only
  writer.
- **Verification**: 6/6 on Qt 6.4.2 and Qt 5.15.2 trees (Linux cannot fail
  this way — it only proves no regression); CI is the gate for the Unicode
  path.
- **Open**: `source.openVideo(request.videoPath.toStdString())` has the
  same narrow-path shape for a Unicode *input* video; not changed (no
  failing test, no Windows build to validate a fix).

## 21. Timeline drag test driven by pixel percentages lands on a half-frame

- **Symptom**: `test_qanylabeling_video_ui`
  `LabelTimelineExportsSplitHistoryAndDropRoundTrip` fails on Qt 6.4.2 with
  "exported frame range is incorrect" (segment `[1600, 2000)` instead of
  `[1400, 1800)`), then hangs (entry 19). Green on Qt 5.15.2. Never seen in
  Linux CI: the `test-ci` label set excludes qAnyLabeling tests; only the
  Windows unified headless gate runs them.
- **Essence**: the test converted percentages of the timeline width into drag
  coordinates (`trackWidth * 55 / 100`). `VideoSegmentTimeline::snapTime`
  rounds to the nearest frame (`llround(ms / frame)`), and 55 % / 75 % are the
  half-frame marks of a 2000 ms, 5 fps clip (1100 ms = frame 5.5, 1500 ms =
  frame 7.5). Whether a landing point sits just below or exactly on the mark
  depends on the widget width, which differs per Qt version: Qt 6.4 lays the
  timeline out 904 px wide (track 880 = 20 × 44, so 55 % is exactly 1100.0 →
  rounds up to 1200), Qt 5.15 901 px (track 877, 55 % is 1099 ms → 1000).
  Evidence: temporary prints of `mousePressEvent` (track 12..891, hit
  Right/1) and of `segmentEdited` (receiving `1200,1600` directly, no
  `editRange` in between). The plugin behaves as designed.
- **Fix**: the test maps times to pixels (`xAt(ms) = 12 + trackWidth * ms /
  2000`) and uses landings ≥ 50 ms away from every half-frame mark (1050,
  1250, 1450, 1150, 1650): create `[1000, 1200)`, grow the right handle to
  1400, slide the body to `[1400, 1800)`. Assertions unchanged.
- **Verification**: 3/3 on Qt 6.4.2, Qt 6.10.2 and Qt 5.15.2 trees.
- **Rule**: a synthetic drag on a snapping widget must be specified in the
  snapped domain (frames) with margin, never as a percentage of an unknown
  width.

## 22. A hidden `QProgressDialog` pops up by itself 4 s after construction

- **Symptom**: `test_qanylabeling_chat` `ProviderAndWorkspaceWorkflows`
  on Ubuntu 20.04 (Qt 5.12.8) either hangs until the ctest timeout
  (cuda-focal: `***Timeout 1200.10 sec`, log silent after the `R,G,p`
  sequence) or segfaults inside `QLineEdit::setText` with XMM registers
  full of `0x7a` ('z'). Both are the same timer.
- **Essence**: `ChatbotDialog` constructed
  `QProgressDialog(0, 0)` + `setWindowModality(WindowModal)` +
  `setMinimumDuration(0)` + `hide()`. The private constructor arms
  `forceTimer->start(showTime = 4000)`. `setMinimumDuration()` only
  re-arms while `bar->value() == bar->minimum()`; a fresh busy bar is
  `-1` vs minimum `0`, so the timer survives and `hide()` does not stop
  it. ~4 s after construction `forceShow()` shows the hidden modal
  dialog from a `QTimer::timeout`:
  - `QWidgetPrivate::show_helper` closes every active `Qt::Popup` when a
    non-popup top-level window is shown → the open `ChatModelPopup`
    (`WA_DeleteOnClose`) is deleted → a raw `QLineEdit*` cached by the
    test dangles.
  - `showModalWindow()` prepends it to `modalWindowList`, so
    `QApplication::activeModalWidget()` is no longer the
    `ChatBatchOptions` dialog whose `exec()` is running → the test's
    single-shot watchdog never clicks Ok → `exec()` never returns.
  Proof: a 20-line program reproduces the self-show on Qt 5.12.8; an
  in-process event filter with `backtrace()` captured
  `show_helper ← QDialog::setVisible ← QProgressDialog::forceShow ←
  QTimer::timeout` closing the popup; delaying the batch launch to
  3950–4010 ms after construction hung 7/7, 3700–3940 and 4020–4100
  passed 100 %.
- **Fix**: call `reset()` right after construction (owner side) in
  `ChatbotDialog` and in `AnnotationClassifier::remoteProgress`;
  `reset()` stops `forceTimer`. Nothing else changes: `configureBatch`
  still calls `setRange/setValue/show()` on demand.
- **Verification**: container Qt 5.12.8, the same 3950–4010 sweep 0/6
  hangs; 48 concurrent instances (`--cpus 2`) 0 segfaults vs 25–29 %
  before. Green on Qt 5.14.2, 6.4.2 and 6.10.2 trees. The residual
  `test_chat.cpp:2375` 5 s wait failure under 12× oversubscription is
  pre-existing (A/B with the unfixed plugin: 3/45 vs 3/48) and does not
  occur in serial CI.
- **Rule**: a `QProgressDialog` that is created hidden must be followed by
  `reset()` (or be created lazily at first use); `setMinimumDuration(0)`
  + `hide()` do not cancel the constructor-armed force-show timer.

## 23. `_opencv_videoio` consumer list lagged behind the plugins linking `video_base`

- **Symptom**: found while repairing entry 17 on Windows: the
  bundled OpenCV is built without `videoio`/ffmpeg whenever only
  qAnyLabeling, qRAM or qRelateAnything need video decoding, so those
  plugins would have no decoder at run time.
- **Essence**: `3rdparty/opencv/opencv_build.cmake` decides whether the
  bundled OpenCV builds `videoio` + the ffmpeg backend from a hard-coded
  list of plugin options. qAnyLabeling, qRAM and qRelateAnything link
  `video_base` but were not in the list (the list had not been kept in
  sync with the `video_base` consumers).
- **Fix**: add the three options to the `_opencv_videoio` condition (the
  list must mirror the `video_base` consumers in the plugin CMakeLists).
- **Verification**: Windows CI only (no local Windows toolchain); recorded
  as a pending gate until the Windows job of the fixing SHA is green.
