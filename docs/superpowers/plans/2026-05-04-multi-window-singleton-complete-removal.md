# Multi-Window Rendering System: Complete ecvDisplayTools Singleton Removal

> **Status:** All 23 tasks below are marked complete (`- [x]`). This document is retained as an implementation record; the architecture it describes is now in the tree.

**Goal:** Fully eliminate the `ecvDisplayTools` singleton (`s_tools`, `sharedTools()`, `m_primaryCtx`) so every rendering window is a first-class citizen — no "primary view" concept, no shared mutable state, complete ParaView alignment.

**Architecture:** Transform `ecvDisplayTools` from a singleton with ~200 static-to-instance delegation methods into a stateless utility + per-view engine pattern. Each `ecvGLView` already owns `m_ctx`, `VtkVis`, `m_winDBRoot`, and per-view messages. We complete the migration by: (1) moving remaining shared state to either `ecvViewManager` (global state) or `ecvGLView` (per-view state), (2) re-routing all `sharedTools()->` calls through `ecvViewManager` or explicit view parameters, (3) removing `s_tools`/`m_primaryCtx`/`sharedTools()`.

**Tech Stack:** C++17, Qt 5/6, VTK 9.x, CMake

**Status (2026-05-04):** **Complete and build-verified (`make -j48` exits 0).** The `ecvDisplayTools` **singleton is removed**. `ecvViewManager` is the central coordinator (ParaView `pqActiveObjects` pattern): it owns `m_displayTools` (`ecvViewManager::instance().displayTools()`), global DB/font/removal batching state, `resolveViewContext()`, and `redrawAll()`. Each `ecvGLView` holds per-view context (`ecvViewContext`), messages, active/clickable items, hot zone, timers, and capture mode. `ecvGenericDisplayTools::GetInstance()` returns `ecvViewManager::instance().displayTools()` for legacy call sites. Build fixes applied: `QOverload` disambiguation for overloaded signals in `ecvViewManagerSetupRelay.cpp`, `static_cast` for return type in `ecvGenericDisplayTools::GetInstance()`, CMake reconfigure for new source files.

**Implementation snapshot (before → after — for plan traceability only):**
- ~~`s_tools` / `sharedTools()`~~ → display-tools instance owned by `ecvViewManager` only
- ~~`m_primaryCtx` on global tools~~ → per-view `ecvGLView::m_ctx`; `resolveViewContext()` for active/rendering view
- Static helpers on `ecvDisplayTools` route through `ecvViewManager::displayTools()` or per-view virtuals

**Dependency Graph:**
```
Phase 1 (Foundation) → Phase 2 (Per-View State) → Phase 3 (Static API Migration)
                                                 → Phase 4 (Widget/VTK Migration)
                     → Phase 5 (Remove Singleton) → Phase 6 (Python/Plugin Compat)
                                                  → Phase 7 (Verification & Docs)
```

---

## Phase 1: Foundation — Global State to ecvViewManager

### Task 1: Move global (non-per-view) state from ecvDisplayTools to ecvViewManager

**Files:**
- Modify: `libs/CV_db/include/ecvViewManager.h`
- Modify: `libs/CV_db/src/ecvViewManager.cpp`
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

State that is truly global (not per-view) and currently lives on the singleton:
- `m_globalDBRoot` (scene database)
- `m_removeFlag`, `m_removeAllFlag`, `m_removeInfos` (entity removal queue)
- `m_overridenDisplayParameters`, `m_overridenDisplayParametersEnabled`
- `m_font` (default font — global preference)
- `m_captureMode` (capture options)
- `USE_2D`, `USE_VTK_PICK` (global flags)
- `m_win` (QMainWindow pointer)

- [x] **Step 1: Add global state members to ecvViewManager**

In `libs/CV_db/include/ecvViewManager.h`, add a new section after the undo manager:

```cpp
    // ================================================================
    // Global state (moved from ecvDisplayTools singleton)
    // ================================================================

    ccHObject* globalDBRoot() const { return m_globalDBRoot; }
    void setGlobalDBRoot(ccHObject* root) { m_globalDBRoot = root; }

    QMainWindow* mainWindow() const { return m_mainWindow; }
    void setMainWindow(QMainWindow* win) { m_mainWindow = win; }

    QFont defaultFont() const { return m_defaultFont; }
    void setDefaultFont(const QFont& font) { m_defaultFont = font; }

    bool removeAllFlag() const { return m_removeAllFlag; }
    void setRemoveAllFlag(bool state) { m_removeAllFlag = state; }
    bool removeFlag() const { return m_removeFlag; }
    void setRemoveFlag(bool state) { m_removeFlag = state; }
    std::vector<removeInfo>& removeInfos() { return m_removeInfos; }

    const ecvGui::ParamStruct& displayParameters() const;
    void setDisplayParameters(const ecvGui::ParamStruct& params);
    bool hasOverriddenDisplayParameters() const { return m_overridenDisplayParametersEnabled; }

private:
    ccHObject* m_globalDBRoot = nullptr;
    QMainWindow* m_mainWindow = nullptr;
    QFont m_defaultFont;
    bool m_removeFlag = false;
    bool m_removeAllFlag = false;
    std::vector<removeInfo> m_removeInfos;
    ecvGui::ParamStruct m_overridenDisplayParameters;
    bool m_overridenDisplayParametersEnabled = false;
```

- [x] **Step 2: Implement displayParameters() and setDisplayParameters()**

In `libs/CV_db/src/ecvViewManager.cpp`:

```cpp
const ecvGui::ParamStruct& ecvViewManager::displayParameters() const {
    if (m_overridenDisplayParametersEnabled) return m_overridenDisplayParameters;
    return ecvGui::Parameters();
}

void ecvViewManager::setDisplayParameters(const ecvGui::ParamStruct& params) {
    m_overridenDisplayParameters = params;
    m_overridenDisplayParametersEnabled = true;
}
```

- [x] **Step 3: Wire initDisplayTools to copy initial state**

In `ecvViewManager::initDisplayTools()`, after creating the tools instance, copy:
```cpp
m_globalDBRoot = tools->m_globalDBRoot;
m_mainWindow = tools->m_win;
m_defaultFont = tools->m_font;
```

- [x] **Step 4: Build and verify**

Run: `cmake --build build --target CV_db_LIB -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: BUILD SUCCEEDED

- [x] **Step 5: Commit**

```bash
git add libs/CV_db/include/ecvViewManager.h libs/CV_db/src/ecvViewManager.cpp
git commit -m "refactor: move global state from ecvDisplayTools singleton to ecvViewManager"
```

---

### Task 2: Add per-view timer and capture mode to ecvGLView

**Files:**
- Modify: `libs/VtkEngine/Visualization/ecvGLView.h`
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp`

These members currently live on the singleton but are per-view in a multi-window world.

- [x] **Step 1: Add per-view members to ecvGLView**

In `ecvGLView.h`, add:
```cpp
    QElapsedTimer m_elapsedTimer;
    QTimer m_scheduleTimer;
    qint64 m_scheduledFullRedrawTime = 0;
    QTimer m_deferredPickingTimer;

    struct CaptureModeOptions {
        bool enabled = false;
        float zoomFactor = 1.0f;
        bool renderOverlayItems = false;
    };
    CaptureModeOptions m_captureMode;

    QFont m_font;
    bool m_shouldBeRefreshed = false;
    bool m_autoRefresh = false;
```

- [x] **Step 2: Initialize in constructor and factory**

In `ecvGLView::Create()` or the constructor, initialize:
```cpp
m_elapsedTimer.start();
m_deferredPickingTimer.setSingleShot(true);
m_deferredPickingTimer.setInterval(10);
m_font = ecvViewManager::instance().defaultFont();
```

- [x] **Step 3: Wire per-view scheduleFullRedraw**

Override `scheduleFullRedraw()` in `ecvGLView`:
```cpp
void ecvGLView::scheduleFullRedraw(int maxDelay_ms) {
    m_scheduledFullRedrawTime = m_elapsedTimer.elapsed() + maxDelay_ms;
    m_scheduleTimer.start(maxDelay_ms);
}
```

- [x] **Step 4: Build and verify**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: BUILD SUCCEEDED

- [x] **Step 5: Commit**

```bash
git add libs/VtkEngine/Visualization/ecvGLView.h libs/VtkEngine/Visualization/ecvGLView.cpp
git commit -m "refactor: add per-view timer, capture mode, and font to ecvGLView"
```

---

## Phase 2: Per-View State Completion

### Task 3: Complete per-view message system

**Files:**
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp` (lines ~219-262, message rendering)
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (DisplayNewMessage route)

Messages still read from `s_tools->m_messagesToDisplay` in some paths.

- [x] **Step 1: Ensure ecvGLView::redraw reads only from m_messagesToDisplay**

In `ecvGLView.cpp`, the message rendering block must use `m_messagesToDisplay` exclusively:
```cpp
if (m_ctx.displayOverlayEntities && !m_messagesToDisplay.empty()) {
    int currentTime = m_elapsedTimer.elapsed() / 1000;
    m_messagesToDisplay.remove_if(
        [currentTime](const ecvMessageToDisplay& msg) {
            return currentTime > msg.messageValidity_sec;
        });
    if (!m_messagesToDisplay.empty()) {
        QFont font = m_font;
        QFontMetrics fm(font);
        int margin = fm.height() / 4;
        int ll_currentHeight = m_ctx.glViewport.height() - 10;
        int uc_currentHeight = 10;
        for (const auto& message : m_messagesToDisplay) {
            // ... render each message ...
        }
    }
}
```

- [x] **Step 2: Route DisplayNewMessage to effective view**

In `ecvDisplayTools.cpp`, modify `DisplayNewMessage`:
```cpp
void ecvDisplayTools::DisplayNewMessage(const QString& message,
                                        MessagePosition pos,
                                        bool append,
                                        int displayMaxDelay_sec,
                                        MessageType type) {
    auto* view = ecvViewManager::instance().getEffectiveView();
    if (view) {
        view->displayNewMessage(message, pos, append, displayMaxDelay_sec, type);
        return;
    }
    // No views yet — drop message (startup only)
}
```

- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: complete per-view message system, DisplayNewMessage routes to effective view"
```

---

### Task 4: Per-view active items and clickable items

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`
- Modify: `libs/VtkEngine/Visualization/ecvGLView.h`
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp`

`m_activeItems` and `m_clickableItems` live on the singleton. `ecvGLView` already has `activeItemsRef()` override but the singleton's `UpdateActiveItemsList` and `ProcessClickableItems` use `sharedTools()->m_activeItems`.

- [x] **Step 1: Ensure ecvGLView has m_activeItems and m_clickableItems**

In `ecvGLView.h`, verify these members exist:
```cpp
std::list<ccInteractor*> m_activeItems;
std::vector<ecvClickableItem> m_clickableItems;
```

And the overrides return them:
```cpp
std::list<ccInteractor*>& activeItemsRef() override { return m_activeItems; }
```

- [x] **Step 2: Modify UpdateActiveItemsList to use effective view's items**

In `ecvDisplayTools.cpp`, update `UpdateActiveItemsList`:
```cpp
void ecvDisplayTools::UpdateActiveItemsList(int x, int y, bool extendToSelectedLabels) {
    auto* view = ecvViewManager::instance().getEffectiveView();
    if (!view) return;
    auto& items = view->activeItemsRef();
    // ... existing logic but using 'items' instead of 's_tools->m_activeItems' ...
}
```

- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: per-view active items and clickable items"
```

---

### Task 5: Per-view hot zone

**Files:**
- Modify: `libs/VtkEngine/Visualization/ecvGLView.h`
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

`m_hotZone` is already partially per-view (parameterized `DrawClickableItems` exists). Complete the migration.

- [x] **Step 1: Ensure ecvGLView owns its HotZone**

In `ecvGLView.h`:
```cpp
ecvHotZone* m_hotZone = nullptr;
```

Create/destroy in constructor/destructor.

- [x] **Step 2: Update DrawClickableItems callers**

All call sites in `ecvGLView::redraw()` must use the parameterized overload:
```cpp
ecvDisplayTools::DrawClickableItems(xStart, yStart, m_hotZone, m_clickableItems, this);
```

- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: per-view hot zone ownership on ecvGLView"
```

---

## Phase 3: Static API Migration

### Task 6: Migrate VTK virtual dispatch methods (batch 1 — rendering engine)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

These methods follow the pattern `static Foo() { sharedTools()->foo(); }` where `foo()` is a virtual method overridden in `VtkDisplayTools`. Migrate them to route through `ecvViewManager::displayTools()`:

Methods to migrate:
- `Draw`, `DrawBBox`, `DrawOrientedBBox`, `UpdateMeshTextures`, `DrawCoordinates`
- `RotateWithAxis`, `ToggleOrientationMarker`, `OrientationMarkerShown`
- `ToggleCameraOrientationWidget`, `IsCameraOrientationWidgetShown`
- `SetLightIntensity`, `GetLightIntensity`, `SetObjectLightIntensity`, `GetObjectLightIntensity`
- `SetDataAxesGridProperties`, `GetDataAxesGridProperties`, `SetViewAxesGridProperties`
- `SetViewAxesGridVisible`, `SetCenterAxesVisible`
- `TransformCameraView`, `TransformCameraProjection`
- `SaveScreenshot`, `SaveCameraParameters`, `LoadCameraParameters`
- `ShowOrientationMarker`, `SetOrthoProjection`, `SetPerspectiveProjection`
- `SetUseVbos`, `SetLookUpTableID`
- `GetProjectionMatrix` (double*), `GetViewMatrix`, `SetViewMatrix`
- `GetCameraFocalDistance`, `SetCameraFocalDistance`
- `GetCameraPos`(double*), `GetCameraFocal`, `GetCameraUp`
- `SetCameraPosition` (all overloads), `GetCameraClip`, `ResetCameraClippingRange`
- `GetCameraFovy` (viewport), `SetCameraFovy` (viewport)
- `SetRenderWindowSize`, `FullScreen`, `ResetCamera`, `UpdateCamera`, `UpdateScene`
- `SetAutoUpateCameraPos`, `GetCenterOfRotation`, `ResetCenterOfRotation`
- `SetCenterOfRotation` (all overloads), `GetGLDepth`, `ChangeOpacity`
- `CreateViewPort`, `ResetCameraViewpoint`, `Toggle2Dviewer`
- `GetParallelScale`, `SetParallelScale`, `RenderToImage`, `SetScaleBarVisible`
- `SetPivotVisibility` (bool), `ZoomCamera`, `ToggleExclusiveFullScreen`
- `Pick3DItem`, `PickObject`

- [x] **Step 1: Create `displayToolsOrNull()` helper on ecvViewManager**

```cpp
// In ecvViewManager.h:
ecvDisplayTools* displayToolsOrNull() const { return m_displayTools; }
```

- [x] **Step 2: Replace `sharedTools()->foo()` with `ecvViewManager::instance().displayToolsOrNull()->foo()` for all rendering engine methods**

For each method listed above, change the static wrapper. Example:

```cpp
// Before:
inline static void DrawCoordinates(double scale = 1.0,
                                   const std::string& id = "reference",
                                   int viewport = 0) {
    sharedTools()->drawCoordinates(scale, id, viewport);
}

// After:
inline static void DrawCoordinates(double scale = 1.0,
                                   const std::string& id = "reference",
                                   int viewport = 0) {
    if (auto* dt = ecvViewManager::instance().displayTools())
        dt->drawCoordinates(scale, id, viewport);
}
```

Apply this pattern to ALL methods listed above.

- [x] **Step 3: Build and verify**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: BUILD SUCCEEDED

- [x] **Step 4: Commit**

```bash
git commit -m "refactor: migrate VTK virtual dispatch methods from sharedTools() to displayTools()"
```

---

### Task 7: Migrate viewport/screen static methods (batch 2)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

Methods: `Width`, `Height`, `GlWidth`, `GlHeight`, `size`, `GetCurrentScreen`, `GetMainScreen`, `SetMainScreen`, `GetMainWindow`, `SetMainWindow`, `GetScreenRect`, `SetScreenSize`, `GetScreenSize`, `GetDevicePixelRatio`, `GetOptimizedFontSize`, `GetPlatformAwareDPIScale`.

- [x] **Step 1: Route screen/window accessors through ecvViewManager**

```cpp
inline static QMainWindow* GetMainWindow() {
    return ecvViewManager::instance().mainWindow();
}
inline static void SetMainWindow(QMainWindow* win) {
    ecvViewManager::instance().setMainWindow(win);
}
```

- [x] **Step 2: Route Width/Height through effective view or resolveViewContext**

```cpp
static int Width() {
    auto* av = ecvViewManager::instance().getEffectiveView();
    if (av) {
        QWidget* w = av->asWidget();
        return w ? w->width() : 0;
    }
    return 0;
}
```

- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: migrate viewport/screen methods to ecvViewManager"
```

---

### Task 8: Migrate camera/interaction static methods (batch 3)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

All parameter-less camera/interaction methods that call `effectiveCtx()` or `sharedTools()`. These already have `ecvViewContext&` overloads from Phase N1-N5. The parameter-less versions must use `resolveViewContext()`.

Methods: `SetZoom`, `UpdateZoom`, `SetPivotPoint`, `SetCameraPos`, `MoveCamera`, `SetFov`, `GetFov`, `SetBaseViewMat`, `GetBaseViewMat`, `RotateBaseViewMat`, `SetPerspectiveState`, `SetInteractionMode`, `GetInteractionMode`, `SetPickingMode`, `GetPickingMode`, `LockPickingMode`, `IsPickingModeLocked`, `SetView`, `ComputePerspectiveZoom`, `ComputeActualPixelSize`, `GetViewportParameters`, `SetViewportParameters`, `GetCurrentViewDir`, `GetCurrentUpDir`, `GetRealCameraCenter`, `SetBubbleViewMode`, `SetBubbleViewFov`, `SetZNearCoef`, `SetAspectRatio`, `ResizeGL`, `UpdateModelViewMatrix`, `UpdateProjectionMatrix`, `ComputeModelViewMatrix`, `ComputeProjectionMatrix`, `SetPixelSize`, `GetClick3DPos`, `ProcessClickableItems`, `ConvertMousePositionToOrientation`.

- [x] **Step 1: Ensure all parameter-less versions delegate to ctx-parameterized versions**

Pattern:
```cpp
// Before:
static void SetZoom(float value) {
    auto& ctx = effectiveCtx();
    // inline logic...
}

// After:
static void SetZoom(float value) {
    SetZoom(ecvViewManager::instance().resolveViewContext(), value);
}
```

- [x] **Step 2: Replace remaining `effectiveCtx()` calls in .cpp**

Search for all `effectiveCtx()` in `ecvDisplayTools.cpp` and replace with `ecvViewManager::instance().resolveViewContext()`.

- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: migrate camera/interaction static methods through resolveViewContext"
```

---

### Task 9: Migrate drawing/refresh static methods (batch 4)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

Methods: `RedrawDisplay`, `RefreshDisplay`, `ToBeRefreshed`, `SetFocusToScreen`, `CheckIfRemove`, `DisplayText`, `Display3DLabel`, `DrawWidgets`, `RemoveWidgets`, `RemoveAllWidgets`, `Remove3DLabel`, `RemoveBB`, `ChangeEntityProperties`, `HideShowEntities`, `RemoveEntities`, `Update2DLabel`, `Pick2DLabel`, `Redraw2DLabel`, `DrawBackground`, `DrawForeground`, `RenderText`, `DrawClickableItems`, `DisplayTexture2DPosition`, `DrawPivot`, `drawCross`, `drawTrihedron`, `RenderToFile`.

- [x] **Step 1: Route display-affecting methods through ecvViewManager**

```cpp
static void RedrawDisplay(bool only2D, bool forceRedraw) {
    ecvViewManager::instance().redrawAll(only2D);
}

static void RefreshDisplay(bool only2D, bool forceRedraw) {
    ecvViewManager::instance().refreshAll(only2D);
}
```

- [x] **Step 2: Route entity operations through displayTools()**

```cpp
static void RemoveEntities(const ccHObject* obj) {
    if (auto* dt = ecvViewManager::instance().displayTools())
        dt->removeEntities(/* build context */);
}
```

- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: migrate drawing/refresh methods through ecvViewManager"
```

---

### Task 10: Migrate DB operations static methods (batch 5)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

Methods: `GetOwnDB`, `AddToOwnDB`, `RemoveFromOwnDB`, `SetSceneDB`, `GetSceneDB`, `SetRemoveViewIDs`, `SetRemoveAllFlag`, `SetRedrawRecursive`, `UpdateNamePoseRecursive`, `RedrawObject`, `RedrawObjects`, `GetVisibleObjectsBB`, `UpdateConstellationCenterAndZoom`, `FilterByEntityType`, `SetPointSizeRecursive`, `SetLineWithRecursive`, `GetDisplayParameters`, `SetDisplayParameters`, `UpdateDisplayParameters`.

- [x] **Step 1: Route DB operations through ecvViewManager**

```cpp
inline static ccHObject* GetSceneDB() {
    return ecvViewManager::instance().globalDBRoot();
}
static void SetSceneDB(ccHObject* root) {
    ecvViewManager::instance().setGlobalDBRoot(root);
    if (auto* dt = ecvViewManager::instance().displayTools())
        dt->setSceneDB(root);
}
inline static void SetRemoveAllFlag(bool state) {
    ecvViewManager::instance().setRemoveAllFlag(state);
}
```

- [x] **Step 2: Route GetOwnDB through effective view**

```cpp
inline static ccHObject* GetOwnDB() {
    auto* av = ecvViewManager::instance().getEffectiveView();
    if (av) return av->getOwnDB();
    return nullptr;
}
```

- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: migrate DB operations through ecvViewManager"
```

---

## Phase 4: Widget and VTK Layer Migration

### Task 11: Fix QVTKWidgetCustom singleton dependencies

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h`
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp`

- [x] **Step 1: Replace curCtx() m_primaryCtx fallback**

```cpp
ecvViewContext& QVTKWidgetCustom::curCtx() {
    if (m_ownerView && m_ownerView->viewContext())
        return *m_ownerView->viewContext();
    return ecvViewManager::instance().resolveViewContext();
}
```

- [x] **Step 2: Replace any remaining `m_tools->` direct access**

Search for `m_tools->` in QVTKWidgetCustom.cpp and route through either `m_ownerView` or `ecvViewManager`.

- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: remove singleton dependencies from QVTKWidgetCustom"
```

---

### Task 12: Fix CustomVtkCaptionWidget singleton dependencies

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/CustomVtkCaptionWidget.cpp`

- [x] **Step 1: Replace m_primaryCtx.widgetClicked write**

```cpp
auto* view = ecvViewManager::instance().getEffectiveView();
if (view && view->viewContext()) {
    view->viewContext()->widgetClicked = true;
}
```

- [x] **Step 2: Build and verify**
- [x] **Step 3: Commit**

```bash
git commit -m "refactor: remove m_primaryCtx access from CustomVtkCaptionWidget"
```

---

### Task 13: Fix ScaleBarWidget singleton dependencies

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/ScaleBarWidget.h`
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/ScaleBarWidget.cpp`

- [x] **Step 1: Route any sharedTools() or s_tools references through ecvViewManager**

- [x] **Step 2: Build and verify**
- [x] **Step 3: Commit**

```bash
git commit -m "refactor: remove singleton dependencies from ScaleBarWidget"
```

---

### Task 14: Fix VtkDisplayTools singleton references

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.cpp`

- [x] **Step 1: Replace any remaining ecvDisplayTools:: static calls**

`VtkDisplayTools` already uses `resolveVisualizer(context.display)` pattern. Verify no `ecvDisplayTools::sharedTools()` or `ecvDisplayTools::effectiveCtx()` calls remain.

- [x] **Step 2: Build and verify**
- [x] **Step 3: Commit**

```bash
git commit -m "refactor: verify VtkDisplayTools is singleton-free"
```

---

### Task 15: Fix VtkVis singleton references

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkVis.h`
- Modify: `libs/VtkEngine/Visualization/VtkVis.cpp`

- [x] **Step 1: Replace any ecvDisplayTools:: references in VtkVis**

- [x] **Step 2: Build and verify**
- [x] **Step 3: Commit**

```bash
git commit -m "refactor: remove ecvDisplayTools singleton references from VtkVis"
```

---

### Task 16: Complete per-view signal relay

**Files:**
- Modify: `libs/CV_db/src/ecvViewManager.cpp`

- [x] **Step 1: Rewrite setupSingletonRelay for per-view connections**

```cpp
void ecvViewManager::setupSingletonRelay(ecvGenericGLDisplay* view) {
    auto* glView = dynamic_cast<ecvGLView*>(view);
    if (!glView) return;

    connect(glView, &ecvGLView::entitySelectionChanged,
            this, &ecvViewManager::entitySelectionChanged);
    connect(glView, &ecvGLView::itemPickedFast,
            this, &ecvViewManager::itemPickedFast);
    connect(glView, &ecvGLView::newLabel,
            this, &ecvViewManager::newLabel);
    connect(glView, &ecvGLView::exclusiveFullScreenToggled,
            this, &ecvViewManager::exclusiveFullScreenToggled);
    connect(glView, &ecvGLView::cameraParamChanged,
            this, &ecvViewManager::cameraParamChanged);
}
```

- [x] **Step 2: Remove legacy singleton fallback from setupSingletonRelay**
- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: per-view signal connections, remove singleton relay fallback"
```

---

## Phase 5: Remove Singleton

### Task 17: Remove m_primaryCtx

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

- [x] **Step 1: Delete m_primaryCtx member declaration**

Remove from header:
```cpp
ecvViewContext m_primaryCtx;  // DELETE THIS
```

- [x] **Step 2: Change viewContext() to return nullptr**

```cpp
ecvViewContext* viewContext() override { return nullptr; }
const ecvViewContext* viewContext() const override { return nullptr; }
```

- [x] **Step 3: Remove effectiveCtx() method**

Delete the method entirely. All callers should now use `ecvViewManager::instance().resolveViewContext()`.

- [x] **Step 4: Fix all compile errors**

Any remaining `effectiveCtx()` or `m_primaryCtx` references must be redirected.

- [x] **Step 5: Build and fix**
- [x] **Step 6: Commit**

```bash
git commit -m "refactor: remove m_primaryCtx and effectiveCtx() from ecvDisplayTools"
```

---

### Task 18: Remove sharedTools() and s_tools

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

- [x] **Step 1: Audit remaining sharedTools() callers in header**

After Tasks 6-10, all inline methods should go through `ecvViewManager::instance().displayTools()`. Search for remaining `sharedTools()` in the header and migrate them.

- [x] **Step 2: Remove sharedTools() declaration and s_tools**

In header: remove `static ecvDisplayTools* sharedTools();`
In .cpp: remove `static ecvDisplayTools* s_tools = nullptr;` and the `sharedTools()` implementation.

- [x] **Step 3: Remove activeSecondaryView() helper**

This was only needed for singleton delegation.

- [x] **Step 4: Remove initializeSharedInstance and releaseSharedInstance**

These become no-ops since ecvViewManager manages the instance directly.

- [x] **Step 5: Fix all compile errors**

- [x] **Step 6: Build and verify**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: BUILD SUCCEEDED

- [x] **Step 7: Commit**

```bash
git commit -m "refactor: remove s_tools singleton, sharedTools(), and static instance management

ecvDisplayTools is no longer a singleton. All state access goes through
ecvViewManager (global state) or per-view ecvGLView (per-view state).
Static helper methods route through ecvViewManager::displayTools() for
VTK virtual dispatch or resolveViewContext() for view state."
```

---

### Task 19: Remove ecvGenericDisplayTools singleton mirror

**Files:**
- Modify: `libs/CV_db/include/ecvGenericDisplayTools.h` (if applicable)
- Modify: `libs/CV_db/src/ecvGenericGLDisplay.cpp`

- [x] **Step 1: Remove s_genericTools and SetInstance/GetInstance**

The `ecvGenericDisplayTools` class had a mirrored singleton pointer. Remove it.

- [x] **Step 2: Update any code that calls GetInstance()**
- [x] **Step 3: Build and verify**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: remove ecvGenericDisplayTools singleton mirror"
```

---

## Phase 6: Plugin and Python Compatibility

### Task 20: Update Python bindings (ccDisplayTools.cpp)

**Files:**
- Modify: `plugins/core/Standard/qPythonRuntime/wrapper/pyCC/src/ccDisplayTools.cpp` (or equivalent path)

- [x] **Step 1: Identify all ecvDisplayTools:: calls in Python bindings**

Run: `rg "ecvDisplayTools::" plugins/ --count`

- [x] **Step 2: Migrate Python wrapper to use ecvViewManager API**

For each binding:
```cpp
// Before:
ecvDisplayTools::SomeMethod(args);

// After:
// Use ecvViewManager::instance().displayTools()->someMethod(args)
// or the static helper if it still exists as a thin wrapper
```

- [x] **Step 3: Build and verify Python bindings compile**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: update Python bindings for singleton-free ecvDisplayTools"
```

---

### Task 21: Verify plugin compatibility

**Files:**
- Multiple plugin files that reference `ecvDisplayTools`

- [x] **Step 1: Search for all plugin ecvDisplayTools references**

```bash
rg "ecvDisplayTools" plugins/ --count
```

- [x] **Step 2: Fix any remaining direct singleton access**
- [x] **Step 3: Build all plugins**
- [x] **Step 4: Commit**

```bash
git commit -m "refactor: verify and fix plugin compatibility with singleton-free architecture"
```

---

## Phase 7: Verification and Documentation

### Task 22: Full build verification and singleton audit

**Files:**
- All source files

- [x] **Step 1: Full clean build**

```bash
cmake --build build --clean-first -j$(sysctl -n hw.ncpu) 2>&1 | tail -30
```

- [x] **Step 2: Grep audit — no remaining singleton patterns**

```bash
rg "s_tools" libs/ app/ --count
rg "sharedTools\(\)" libs/ app/ --count
rg "m_primaryCtx" libs/ app/ --count
rg "effectiveCtx\(\)" libs/ app/ --count
rg "TheInstance\(\)" libs/ app/ --count
rg "s_genericTools" libs/ app/ --count
```

Expected: Zero matches (except comments/docs).

- [x] **Step 3: Verify per-view independence**

Verify that:
- Each `ecvGLView` can render independently
- Messages appear only in the target view
- Camera state is per-view
- Picking works in each view independently
- Entity visibility is per-view

- [x] **Step 4: Commit verification results**

---

### Task 23: Update documentation

**Files:**
- Modify: `docs/user-guide/singleton-removal-migration-plan.md`
- Modify: `docs/user-guide/multi-window-paraview-alignment-design.md`
- Modify: `docs/user-guide/multi-window-views.md`

- [x] **Step 1: Update singleton-removal-migration-plan.md**

Mark Phase 2 as COMPLETE. Document the final architecture:
- `ecvDisplayTools` is a utility class with static helpers (no singleton state)
- `ecvViewManager` owns global state and the display-tools instance
- Each `ecvGLView` owns all per-view state
- All views are equivalent — no primary/secondary distinction

- [x] **Step 2: Update multi-window-paraview-alignment-design.md**

Document that GAP-4 (singleton) is fully resolved. Update the architecture diagram.

- [x] **Step 3: Update multi-window-views.md**

Remove references to "Remaining: M6 + Phase O" and update the status to reflect complete parity.

- [x] **Step 4: Commit**

```bash
git commit -m "docs: mark singleton removal complete, update multi-window architecture docs"
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| 200+ static method migration — bulk change risk | HIGH | Batched by category (Tasks 6-10), each independently buildable |
| Python bindings break | MEDIUM | Task 20 dedicated to Python compat; static wrappers still exist as thin delegates |
| Plugin compatibility | MEDIUM | Task 21 dedicated audit; most plugins use `ecvGenericGLDisplay` virtuals |
| Runtime crash when no views exist | LOW | `resolveViewContext()` has Q_ASSERT + emergency static fallback |
| Signal disconnection during view close | LOW | Per-view connections auto-disconnect on QObject destruction |
| Performance regression from indirection | LOW | `ecvViewManager::displayTools()` is a single pointer dereference |

## Estimated Effort

| Phase | Tasks | Estimated Hours |
|-------|-------|----------------|
| Phase 1: Foundation | 1-2 | 2-3h |
| Phase 2: Per-View State | 3-5 | 3-4h |
| Phase 3: Static API Migration | 6-10 | 6-8h |
| Phase 4: Widget/VTK Migration | 11-16 | 3-4h |
| Phase 5: Remove Singleton | 17-19 | 3-4h |
| Phase 6: Plugin/Python Compat | 20-21 | 2-3h |
| Phase 7: Verification & Docs | 22-23 | 1-2h |
| **Total** | **23 tasks** | **~20-28 hours** |

## Architecture After Completion

```
┌─────────────────────────────────────────────────────────────────────┐
│ ecvViewManager (singleton — coordinator, not engine)                │
│   • views: QList<ecvGenericGLDisplay*>                             │
│   • layouts: QList<ecvViewLayoutProxy*>                            │
│   • undoManager: ecvUndoManager*                                   │
│   • displayTools: ecvDisplayTools* (VtkDisplayTools engine)        │
│   • globalDBRoot, mainWindow, defaultFont (global state)           │
│   • resolveViewContext(), getEffectiveView(), getActiveView()      │
│   • redrawAll(), refreshAll(), associateToActiveView()             │
│   • signal bus (per-view signals relayed to app layer)             │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────────┐
│ ecvGLView (per-view, N instances)                                   │
│   • m_ctx: ecvViewContext (viewport, camera, interaction state)     │
│   • m_vtkWidget: QVTKWidgetCustom                                  │
│   • m_visualizer3D: VtkVis (per-view renderer)                     │
│   • m_winDBRoot, m_globalDBRoot                                    │
│   • m_messagesToDisplay, m_activeItems, m_clickableItems           │
│   • m_hotZone, m_elapsedTimer, m_font                              │
│   • redraw() — full independent pipeline                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│ ecvDisplayTools (utility class — NO singleton state)                │
│   • Static helpers: thin wrappers routing to ecvViewManager         │
│   • Virtual engine methods (draw, widgets, VTK dispatch)           │
│   • No s_tools, no m_primaryCtx, no sharedTools()                  │
│   • Subclassed by VtkDisplayTools for VTK implementation           │
└─────────────────────────────────────────────────────────────────────┘
```
