# Launch identity, native GUI and selection verification

This is an engineering verification route alongside supported CLI operation. Source changes must obey current scoped repository rules and the existing owner; do not build a second automation interface. Preserve the user's live scene and installation. Historical target names and paths below are examples to confirm, not assumptions about another checkout.

## RPC probe and acceptance layers

1. Record `pwd`, CLI executable, relevant environment overrides and display; read current CLI help/docs.
2. Run a read-only bounded supported probe, e.g. through the shell tool with `timeout_ms=10000`:

   ```bash
   cli-anything-acloudviewer --json --mode gui scene list
   ```

   If a hard shell bound is needed and `timeout` is available, use `timeout 10s` around the same command. Record timeout status separately from application status. `info` or `ui list-actions` can further diagnose documented capabilities, but a traceback is not a scene list. Do not retry indefinitely, guess binary flags, restart the user's app, or silently switch to headless for scene acceptance.
3. Distinguish these layers:

   | Layer | What it proves | What it does not prove |
   |---|---|---|
   | Configure/build | Selected sources compiled/linked | Package launches or inputs work |
   | Identified package launch | Exact package loads dependencies and opens | Fullscreen or current-scene correctness |
   | Production-linked native Qt/VTK + Xvfb | Targeted data/GPU/view regression | User's current model and desktop interaction |
   | Isolated WM + native input | Exact identified app/probe under chosen mode/input path | Another build or live user session |
   | Current-scene GUI | Intended live scene responds to actual acceptance actions | Other untested modes |

A native test can be a useful alternative **verification layer** when RPC is unavailable; it is never equivalent to current-scene GUI acceptance. If no safe current-scene path exists, report it as untested, preserving the scene.

## Artifact identity and package launch

### Resolve what actually runs

1. Use supported CLI to discover installed binary information:

   ```bash
   command -v cli-anything-acloudviewer
   cli-anything-acloudviewer check
   cli-anything-acloudviewer --json --mode headless info
   ```

   Record mode and any `ACV_BINARY` value. If resolved path is not exposed by this version, inspect the installed harness's binary-discovery implementation/configuration after locating its package; do not invent an `info` response field or launch flag. An override for a headless call does not replace a process already serving RPC.
2. If CLI resolves a launcher such as `ACloudViewer.sh`, read it and trace its relative directory, final executable, environment, library/plugin paths and arguments. Distinguish old installation, new installation, generated deploy package and `build_app/bin/ACloudViewer`. Do not overwrite any of them during diagnosis.
3. For each observed executable, record identity. Variables below must be set from discovered absolute paths:

   ```bash
   stat -c '%y %s %n' "$INSTALLED_EXE" "$PACKAGE_EXE" "$BUILD_EXE"
   readelf -n "$INSTALLED_EXE"
   readelf -n "$PACKAGE_EXE"
   readelf -n "$BUILD_EXE"
   # If Build ID is unavailable, compare complete content identities:
   sha256sum "$INSTALLED_EXE" "$PACKAGE_EXE" "$BUILD_EXE"
   ```

   On Linux inspect the relevant running PID's `/proc/<pid>/exe` and command line when permitted; record deleted/replaced executable status. Outside ELF use platform-supported content identity. Time/size support diagnosis but are weaker than Build ID or hash. A different path does not necessarily mean different content; matching IDs do not prove bundled dependency completeness.
4. Historical evidence: CLI used an August installation while source fixes built in November; generated package and build could have matching IDs while package launch still failed. Never carry those dates/paths/IDs forward as current facts.

### Diagnose startup before interpreting “window not found”

- Capture application stdout/stderr and WM logs separately. A missing window can be a dynamic loader error, not an input/focus bug.
- Read the actual application log first. Inspect actual package `lib` and plugin directories and the launcher's library search environment.
- Inspect needed library names and candidate SONAMEs rather than guessing version links:

  ```bash
  readelf -d "$PACKAGE_EXE"
  # For each implicated package/plugin/library and discovered candidate:
  readelf -d "$CANDIDATE_LIB"
  ldconfig -p
  # Bounded search of known build tree, not repeated whole-home scans:
  find build_app -maxdepth 8 -name 'libopencv_world.so*' -print
  grep -nE 'OpenCV|OPENCV|CMAKE_PREFIX_PATH|CONDA_PREFIX' build_app/CMakeCache.txt
  ```

  Narrow further to `build_app/bin`, `build_app/external/lib` or the OpenCV build path revealed by cache/source. Distinguish no matches, command failure and search timeout. Stop repeating a timed-out home-wide glob.
- Evidence once showed `libopencv_world.so.407` missing from generated package/system directories while build tree held `libopencv_world.so.4.7.0` and an unversioned file. This requires checking ELF SONAME/NEEDED and dependency provenance; it does **not** authorize `ln -s` between apparent versions or prove compatibility by filename alone.
- An authorized test may use a narrowly scoped runtime library environment to diagnose; disclose it as a diagnostic environment, not a repaired distributable. Without permission do not install OS dependencies, copy libraries into the package, patch launchers, replace old installs, or add version symlinks.
- Re-run identified package launch only if safe and authorized, in isolated config/session, with bounded duration. Report build identity, package load success and actual interaction as separate results.

## Fullscreen and native input

### Scope and ownership

- Change the existing owner of F9 main-window fullscreen (historically `app/MainWindow.cpp`), not unrelated layout logic. F9 entry/exit should not create a floating exit overlay or leave one obscuring controls. Preserve F9/Escape keyboard behavior.
- F11 view fullscreen exit control belongs to the view/tabbed fullscreen owner (historically `app/ecvTabbedMultiViewWidget.cpp`); retain it. Confirm modifier-specific modes from current action definitions rather than guessing.
- Do not change default toolbar loading/order, fixed rows or layout restoration as a workaround for input bugs. Layout restoration and default/reset layout are distinct behaviors; validate relevant ones separately if touched.
- Bubble View's own exit, point-size and line-width controls are independent of fullscreen exit controls. Do not remove or repurpose them.

### Compile a production-linked probe when needed

Prefer existing test targets. If a high-risk path is uncovered, make the smallest probe using the actual full build:

1. Read `build_app/compile_commands.json`; select the relevant production translation unit (e.g. `app/main.cpp` for app-linked Qt probes). Parse its `command` with `shlex.split`, or use its `arguments` list if available; honor the entry's working directory.
2. Reuse its compiler, definitions and includes, replacing source and object output for the probe. Inspect the actual app link recipe (historically `build_app/app/CMakeFiles/ACloudViewer.dir/link.txt`) and replace only app entry/output appropriately. For another generator inspect its actual generated build recipe; TODO: derive generator-specific linkage if it is not exposed. Do not fabricate link flags or reuse incompatible ABI objects.
3. Link actual production VTK/app libraries and initialize the real application/view host. A bare uninitialized QVTK widget can crash for fixture reasons. Evidence required actual ViewManager/VtkDisplayTools host initialization after gdb diagnosis, rather than pretending a minimal widget was equivalent.
4. A QAction `.trigger()` test is not native keyboard dispatch. Explicitly distinguish synthetic Qt key-event, action-trigger, and OS-level XTest coverage. Fullscreen focus/reparenting needs native input and a real window manager.

If the checked-out script actually exists and its help confirms arguments, prior supported probe recipe was:

```bash
python3 app/tests/shortcuts/run_vtk_probe.py --build build_app --output build_shortcuts/vtk
ACV_SHORTCUT_GRAPHICS_TEST=1 QT_QPA_PLATFORM=xcb xvfb-run -a \
  python3 app/tests/shortcuts/run_vtk_probe.py --build build_app --output build_shortcuts/vtk-graphics
```

Do not assume those source files or environment contracts exist on every branch. Inspect before use; otherwise TODO: use the current existing probe or implement the narrow uncovered test at its owner. The evidence does not establish a completed global shortcut audit.

### Isolated desktop/input conditions

- Check available `xvfb-run`, `dbus-run-session`, Mutter/X11, XTest/native probe support, screenshot utilities (e.g. `xwd`, `xprop`). Do not assume `xdotool` or ImageMagick `import` exists, and do not install them without permission.
- Run isolated Xvfb with GLX, an isolated D-Bus session, Mutter `--x11 --sm-disable`, `QT_QPA_PLATFORM=xcb`, and a unique temporary `XDG_CONFIG_HOME`. Use only resolved runtime library paths appropriate to the exact artifact, not copied developer-machine paths.
- Example shell envelope after checking tools and resolving output directory; the actual app/probe launch must use the existing wrapper or documented toolchain, not guessed app options:

  ```bash
  # Set OUT to an authorized absolute output directory first.
  mkdir -p "$OUT"
  export OUT
  timeout --kill-after=5s 120s xvfb-run -a \
    -s '-screen 0 1600x1000x24 +extension GLX' \
    dbus-run-session -- bash -c '
      set -eu
      : "${OUT:?output directory required}"
      export XDG_CONFIG_HOME
      XDG_CONFIG_HOME=$(mktemp -d)
      export QT_QPA_PLATFORM=xcb
      # Start the discovered WM and existing app/probe here, each with logs.
      # TODO: supply the checked-out probe command and actual resolved launcher.
      # Track all child PIDs/process group and install scoped cleanup traps.
    '
  ```

  This is an envelope, not a runnable success probe until the marked launch/input/cleanup work is supplied. Use a unique managed process group/session for the entire test and capture its ID before launch. Export paths or define them **inside** the nested shell: otherwise `$OUT` can be lost and logs accidentally target `/installed-app.log`.
- Bound window discovery, focus acquisition and every native action. Use readiness events/conditions with deadlines rather than unbounded waits or sleep-based busy polling.
- Maintain a latest-artifact matrix: theme/layout (e.g. Fusion/Custom if relevant), mode, entry action, actual focused window, Escape exit, same-shortcut exit, menu exit, F11 exit-button click, resulting window state, overlay count/visibility, retained Bubble state/exit and point-size/line-width state. Cover combined Bubble + point/line controls explicitly; an individual-control pass does not prove the combination.
- If focus/menu input stalls, collect logs/state, cancel the managed job and terminate only its isolated process group, then collect final status. Verify no isolated WM/Xvfb/probe descendants remain. Never use broad app-name kills, guessed PIDs, or the user's desktop display for destructive cleanup.
- A completed compile with a hung native probe is only build success. Earlier tests before an overlay removal do not validate the new build. Record the exact failed step, tested binary ID, timeout/cancellation and remaining gaps; do not mark the overall interaction objective complete.

## Selection and source mapping regression

Read the existing owners before changing anything: `libs/VtkEngine/Tools/SelectionTools/`, `libs/VtkEngine/Visualization/VtkVis.cpp`, `VtkVis.h`, GUI selection consumers and RPC selection methods (historically `plugins/core/Standard/qJSonRPCPlugin/src/JsonRPCMethodsView.cpp`). Do not apply these as blind edits.

### Required correction order

1. **Delete incorrect feedback and isolate capture.** Audit the main-window secondary `pointIndicesSelected` route: face selections must not be reinterpreted as vertex IDs and overwrite a correct face highlight. Preserve legitimate external table/chart point selections. During hardware selection capture, helpers/nonselectable props must neither contribute IDs nor occlude real geometry. `PickableOff` alone failed the point-depth negative control. Temporarily exclude their visibility for capture and restore original visibility and original pickability on success, failure and early exit; do not permanently disable label clicking/dragging/expansion.
2. **Qualify identity and state.** Keep source-object identity, block identity, field association (point/cell), stable source IDs and representation-local IDs distinct. Local ID 22 on two different objects is not one selection; point 22 is not cell 22. Set union/subtraction/toggle must operate on qualified source identity rather than raw actor/local integers. Audit replace/add/subtract/toggle semantics against current implementation. Publish complete atomic selection state (including deselections/multi-source membership), not incremental fragments that leave stale consumers.
3. **Rebuild by view representation.** Selection belongs to a source; each view resolves it using its own actor, mapping, geometry order and transform. Do not broadcast one view's generated overlay geometry to other windows. A view not displaying the source gets no foreign overlay. Test ordinary, compare and orthographic views as applicable. Formal selection and hover have distinct overlays but share source-to-local conversion, including generic conversion paths. View teardown must unbind all consumers; removed DB sources must not be dereferenced from stale raw pointers.

### Helper geometry negative control

Use actual DB mesh and point cloud fixtures and real hardware selection under Xvfb, not just mocked ID filtering.

- Establish baseline **qualified source ID sets** with no helper.
- Add, then remove, each of point marker, connecting line and auxiliary triangle/face (including temporary measurement geometry registered for rendering but not a DB-selectable source).
- For rectangle and polygon selections compare complete resulting qualified source sets to baseline at both add/remove stages. Include point and face association where supported, transformed actors and multiple objects with equal numeric IDs. Merely checking “label ID absent” misses depth occlusion of genuine points.
- Assert helper visibility and pickability exactly restored after capture (including initially hidden/nonpickable cases and failure cleanup). Do not globally force `PickableOn` on restoration.
- For `PolygonSelect`, inspect API coordinate-count contract: four 2D vertices in flat coordinates require count **8**, not 4.
- Through/frustum selection must preserve source-ID mapping and correctly intersect geometry; do not approximate acceptance solely by actor bounds.
- Validate overlay coordinates/bounds in each actor's transform, not only selected ID counts.

### Face highlight/export pollution negative control

A regression exposed `vtkOriginalPointIds` changing from source mapping `{33,22,11}` to local `{0,1,2}` after face extraction. Treat those values as a reproducible fixture example, not universal IDs.

1. Build a shared real source displayed in two views with different point order, local indices, actors and transforms. Include multi-source cases with colliding numeric IDs. Establish that point selection and hover initially resolve stable source IDs correctly in both views.
2. Before any face highlight, snapshot the **display input's** `vtkOriginalPointIds` (array existence, tuple count and every value). Also check relevant source/block/field identity and cell/source mappings where used. Assert expected nonidentity mapping explicitly so a fixture already using `{0,1,2}` cannot mask contamination.
3. Generate face selection/highlight through production code. Assert the original display input mapping is unchanged immediately afterward, not merely that extracted output has an ID array. Repeat separately before/after the export extraction path. Preserve pre-fix failure evidence or a failing negative-control run: this differentiates source mapping corruption from a bad local-index or hover fixture.
4. If the extractor is mutating shared input attribute containers, isolate them at both highlighter and exporter ownership points. Observed fix pattern (adapt names to existing code):

   ```cpp
   vtkNew<vtkPolyData> selectionInput;
   selectionInput->ShallowCopy(polyData);
   extractor->SetInputData(0, selectionInput);
   ```

   This shares large geometry but uses independent attribute containers so adding/replacing `vtkOriginal*Ids` during extraction does not alter the display data's container. Inspect the exact VTK path: shallow copying still shares underlying arrays, so if values are modified in place, isolate the affected array too. Do not claim arbitrary mutation safety from `ShallowCopy` alone.
5. After face highlight **and** after export, repeat point selection and hover using stable source IDs in each view; assert correct target-local IDs, expected transformed coordinates/bounds and no overlays in views without the source. Verify hover shares the conversion path but not the committed-selection overlay. Repeat relevant add/remove helper cases, set operations and complete notification checks.
6. Add minimal lifetime checks for closing a view/unbinding pipeline consumers and removing a source without dereferencing it. Keep tests scoped to uncovered high-risk paths rather than introducing a parallel framework.

## Targeted build and test execution

1. Discover before running:

   ```bash
   cmake --build build_app --target help
   ctest --test-dir build_app -N
   # Read actual definitions too:
   sed -n '1,240p' libs/VtkEngine/tests/CMakeLists.txt
   ```

   Confirm whether `test_composite_block_registry`, `test_lod_actor_bounds`, `QVTK_ENGINE_LIB`, `ACloudViewer` and `QJSON_RPC_PLUGIN` exist. Historical `/fast` variants are Make-generator-specific; prefer normal targets unless verified. Do not rerun a test name blindly if absent. TODO: identify branch-specific equivalent when targets are missing; report a blocked verification dependency instead of inventing one. Do not reconfigure a user's full build indiscriminately.
2. After confirming those targets, one serialized build example is:

   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=8 cmake --build build_app \
     --target QVTK_ENGINE_LIB ACloudViewer QJSON_RPC_PLUGIN \
              test_composite_block_registry test_lod_actor_bounds -j 8
   ```

   Choose lower parallelism for memory limits. Changes to selection structures/API require rebuilding ABI-dependent RPC plugin as well as engine and app; a test executable relink alone is insufficient. Collect every previous relevant build job before starting a conflicting one. Do not run two formatters/builds over the same files/tree concurrently.
3. Run discovered tests under the available graphical host:

   ```bash
   xvfb-run -a ctest --test-dir build_app \
     -R '^(composite_block_registry|lod_actor_bounds)$' --output-on-failure
   ```

   Match actual names from `ctest -N`. Use a bounded managed job/tool timeout and ensure descendants are cleaned up. Keep build and CTest logs separately so each exit status is attributable. If using `tee`, enable `set -o pipefail` and preserve the real command status; never let `echo` mask failure.
4. Read `build_app/Testing/Temporary/LastTest.log` and any relevant failure logs. Report test-case and CTest-test counts accurately (one CTest entry can wrap many cases). Record failed/skipped/no-match items, not just the final shell status. Historical 25-test/6-probe passes are not reusable evidence for a different build.
5. For narrow code formatting use only touched files and inspect `git diff --check`; do not run a whole-file formatter concurrently with another writer/build. If a compile error shows GLEW/Qt OpenGL header ordering, inspect includes: evidence required `<vtk_glew.h>` before Qt OpenGL headers in the affected production translation unit. Rebuild after the correction; do not bypass the production header path to make the test pass.

## Report and completion gate

For every relevant run, retain:

- Exact artifact paths, CLI resolution/wrapper and Build ID/hash, runtime library environment and configuration isolation.
- Supported GUI probe command, outcome/traceback/timeout and whether it reached the intended existing scene.
- Actual CMake targets, build exit codes, Xvfb/CTest command, test/log paths and actual pass/fail/skip counts.
- Native input matrix: mode/theme, entry and exit path, focus result, overlay and control states, exact latest artifact. Label clicking, dragging and expansion must be marked untested if only native data/GPU tests ran; unchanged label event code alone is not interaction proof.
- Before/after source-map assertions for **both** highlight and export; view-local conversion/transform and later hover results; helper add/remove source-set negative controls and state restoration.
- Cancellation/timeout details, collected job outputs and scoped process-group cleanup. No relevant orphan jobs; no interruption to the user's scene.
- Separate verdicts for source repair, build, generated-package launch, native regression, isolated WM/input, and live current-scene acceptance. Explicit remaining items, including combined Bubble/point/line controls or menu-focus paths.

Do not report overall GUI recovery or mark that objective complete if the latest interaction test hung, current required scene replay is absent, or the ordinary CLI still launches another version. Do not promote “shortcut audit started” into conclusions about universal conflict detection or immediate rebinding. Those require a separate audit of registration → category/list → defaults/user config → conflict scope → actual dispatch, including QAction/QShortcut/hard-coded VTK paths and fullscreen reparenting.
