# Phase A Remaining: Context-Aware API & Singleton Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Phase A of the multi-window refactoring by routing all inline static API getters/setters through `effectiveCtx()` instead of direct singleton `m_*` members, and adding context-aware overloads for the remaining core APIs.

**Architecture:** Every inline `static` method in `ecvDisplayTools.h` that currently reads/writes `TheInstance()->m_someField` will be changed to read/write `TheInstance()->effectiveCtx().someField` instead. This ensures the active `ecvViewContext` (owned by the active `ecvGLView`) is always consulted, not the stale singleton copy. Additionally, 5+ new `static` overloads accepting `ecvViewContext&` will be added for the most-used APIs, and the stale singleton `getViewportParameters()` instance method will be redirected to `effectiveCtx()`.

**Tech Stack:** C++17, Qt 5/6, VTK, CMake

**Prerequisite context:**
- `ecvViewContext` struct is defined in `libs/CV_db/include/ecvViewContext.h` (done in earlier Phase A work).
- `ecvGLView` holds `ecvViewContext m_ctx` and exposes `context()` accessors (done).
- `ecvDisplayTools::effectiveCtx()` returns `m_primaryCtx` for the primary view, or the active secondary view's `m_ctx` via `ecvViewManager` (done).
- `pushStateToSingleton` / `pullStateFromSingleton` are no-ops (Phase E forward-looking, done).
- 6 context-aware static overloads already exist: `GetContext`, `GetGLCameraParameters`, `SetPointSize`, `SetLineWidth`, `SetCameraClip`, `SetCameraFovy`.

**Key files:**
| File | Role |
|------|------|
| `libs/CV_db/include/ecvDisplayTools.h` | Inline static APIs (the main target) |
| `libs/CV_db/src/ecvDisplayTools.cpp` | Non-inline static APIs, `effectiveCtx()`, `getViewportParameters()` |
| `libs/CV_db/include/ecvViewContext.h` | Per-view state struct (already complete) |
| `libs/VtkEngine/Visualization/ecvGLView.h` | Per-view class holding `m_ctx` |

---

### Task 1: Route Viewport Inline Getters/Setters Through `effectiveCtx()`

**Rationale:** The inline `SetCameraClip(double, double, int)` and `SetCameraFovy(double, int)` in the header write to `TheInstance()->m_viewportParams`, bypassing `effectiveCtx()`. `GetBaseViewMat` reads `m_viewportParams.viewMat`. `GlWidth`/`GlHeight` fallback to `m_glViewport`. These must all route through `effectiveCtx()` so that secondary views get correct state.

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h` (lines ~1161-1168, ~1501-1506, ~1532-1534, ~2618-2627)

- [ ] **Step 1: Fix `GetBaseViewMat` to use `effectiveCtx()`**

In `libs/CV_db/include/ecvDisplayTools.h`, find the `GetBaseViewMat` method (around line 1161-1168). Change:

```cpp
return TheInstance()->m_viewportParams.viewMat;
```

to:

```cpp
return TheInstance()->effectiveCtx().viewportParams.viewMat;
```

- [ ] **Step 2: Fix `SetCameraClip` inline overload**

Find the inline `SetCameraClip(double znear, double zfar, int viewport)` (around line 1501-1506). Change:

```cpp
TheInstance()->m_viewportParams.zNear = znear;
TheInstance()->m_viewportParams.zFar = zfar;
```

to:

```cpp
TheInstance()->effectiveCtx().viewportParams.zNear = znear;
TheInstance()->effectiveCtx().viewportParams.zFar = zfar;
```

- [ ] **Step 3: Fix `SetCameraFovy` inline overload**

Find the inline `SetCameraFovy(double fovy, int viewport)` (around line 1532-1534). Change:

```cpp
TheInstance()->m_viewportParams.fov_deg = static_cast<float>(fovy);
```

to:

```cpp
TheInstance()->effectiveCtx().viewportParams.fov_deg = static_cast<float>(fovy);
```

- [ ] **Step 4: Fix `GlWidth` / `GlHeight` fallback**

Find `GlWidth()` and `GlHeight()` (around line 2618-2627). Change the fallback from:

```cpp
return TheInstance()->m_glViewport.width();
// and
return TheInstance()->m_glViewport.height();
```

to:

```cpp
return TheInstance()->effectiveCtx().glViewport.width();
// and
return TheInstance()->effectiveCtx().glViewport.height();
```

- [ ] **Step 5: Fix stale `getViewportParameters()` instance method**

In `libs/CV_db/src/ecvDisplayTools.cpp` (around line 4488-4490), the instance method returns `m_viewportParams` (the singleton copy). Change:

```cpp
const ecvViewportParameters& ecvDisplayTools::getViewportParameters() const {
    return m_viewportParams;
}
```

to:

```cpp
const ecvViewportParameters& ecvDisplayTools::getViewportParameters() const {
    return m_primaryCtx.viewportParams;
}
```

This ensures even the instance-level accessor reads from the canonical `ecvViewContext`.

- [ ] **Step 6: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer && cmake --build build --target ACloudViewer 2>&1 | tail -30`

Expected: Compilation succeeds with no new errors.

- [ ] **Step 7: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "refactor(phase-a): route viewport inline APIs through effectiveCtx()

GetBaseViewMat, SetCameraClip, SetCameraFovy, GlWidth/GlHeight
now read/write effectiveCtx().viewportParams instead of the stale
singleton m_viewportParams / m_glViewport."
```

---

### Task 2: Route Display-Flag Inline Getters/Setters Through `effectiveCtx()`

**Rationale:** ~15 inline static methods in the header read/write `TheInstance()->m_exclusiveFullscreen`, `m_showCursorCoordinates`, `m_showDebugTraces`, `m_clickableItemsVisible`, `m_displayOverlayEntities`, `m_pickRadius`. These must route through `effectiveCtx()`.

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`

- [ ] **Step 1: Fix `ExclusiveFullScreen` / `SetExclusiveFullScreenFlage`**

Around line 1948-1952, change:

```cpp
inline static bool ExclusiveFullScreen() {
    return TheInstance()->m_exclusiveFullscreen;
}
inline static void SetExclusiveFullScreenFlage(bool state) {
    TheInstance()->m_exclusiveFullscreen = state;
}
```

to:

```cpp
inline static bool ExclusiveFullScreen() {
    return TheInstance()->effectiveCtx().exclusiveFullscreen;
}
inline static void SetExclusiveFullScreenFlage(bool state) {
    TheInstance()->effectiveCtx().exclusiveFullscreen = state;
}
```

- [ ] **Step 2: Fix `ShowCursorCoordinates` / `CursorCoordinatesShown`**

Around line 2186-2192, change:

```cpp
inline static void ShowCursorCoordinates(bool state) {
    TheInstance()->m_showCursorCoordinates = state;
}
inline static bool CursorCoordinatesShown() {
    return TheInstance()->m_showCursorCoordinates;
}
```

to:

```cpp
inline static void ShowCursorCoordinates(bool state) {
    TheInstance()->effectiveCtx().showCursorCoordinates = state;
}
inline static bool CursorCoordinatesShown() {
    return TheInstance()->effectiveCtx().showCursorCoordinates;
}
```

- [ ] **Step 3: Fix `EnableDebugTrace` / `ToggleDebugTrace`**

Around line 2221-2227, change:

```cpp
inline static void EnableDebugTrace(bool state) {
    TheInstance()->m_showDebugTraces = state;
}
inline static void ToggleDebugTrace() {
    TheInstance()->m_showDebugTraces = !TheInstance()->m_showDebugTraces;
}
```

to:

```cpp
inline static void EnableDebugTrace(bool state) {
    TheInstance()->effectiveCtx().showDebugTraces = state;
}
inline static void ToggleDebugTrace() {
    auto& ctx = TheInstance()->effectiveCtx();
    ctx.showDebugTraces = !ctx.showDebugTraces;
}
```

- [ ] **Step 4: Fix `SetClickableItemsVisible` / `GetClickableItemsVisible`**

Around line 2324-2328, change:

```cpp
static void SetClickableItemsVisible(bool state) {
    TheInstance()->m_clickableItemsVisible = state;
}
static bool GetClickableItemsVisible() {
    return TheInstance()->m_clickableItemsVisible;
}
```

to:

```cpp
static void SetClickableItemsVisible(bool state) {
    TheInstance()->effectiveCtx().clickableItemsVisible = state;
}
static bool GetClickableItemsVisible() {
    return TheInstance()->effectiveCtx().clickableItemsVisible;
}
```

- [ ] **Step 5: Fix `SetPickingRadius` / `GetPickingRadius`**

Around line 2342-2346, change:

```cpp
inline static void SetPickingRadius(int radius) {
    TheInstance()->m_pickRadius = radius;
}
inline static int GetPickingRadius() { return TheInstance()->m_pickRadius; }
```

to:

```cpp
inline static void SetPickingRadius(int radius) {
    TheInstance()->effectiveCtx().pickRadius = radius;
}
inline static int GetPickingRadius() {
    return TheInstance()->effectiveCtx().pickRadius;
}
```

- [ ] **Step 6: Fix `OverlayEntitiesAreDisplayed`**

Around line 2354-2355, change:

```cpp
inline static bool OverlayEntitiesAreDisplayed() {
    return TheInstance()->m_displayOverlayEntities;
}
```

to:

```cpp
inline static bool OverlayEntitiesAreDisplayed() {
    return TheInstance()->effectiveCtx().displayOverlayEntities;
}
```

- [ ] **Step 7: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer && cmake --build build --target ACloudViewer 2>&1 | tail -30`

Expected: Compilation succeeds.

- [ ] **Step 8: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h
git commit -m "refactor(phase-a): route display-flag inline APIs through effectiveCtx()

ExclusiveFullScreen, ShowCursorCoordinates, EnableDebugTrace,
ClickableItemsVisible, PickingRadius, OverlayEntitiesAreDisplayed
now read/write effectiveCtx() instead of singleton m_* members."
```

---

### Task 3: Route Pivot / Bubble / Rotation-Lock Inline Getters Through `effectiveCtx()`

**Rationale:** `GetPivotVisibility`, `BubbleViewModeEnabled`, `AutoPickPivotAtCenter`, `IsRotationAxisLocked` still read stale singleton fields.

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`

- [ ] **Step 1: Fix `GetPivotVisibility`**

Around line 2098-2099, change:

```cpp
inline static PivotVisibility GetPivotVisibility() {
    return TheInstance()->m_pivotVisibility;
}
```

to:

```cpp
inline static PivotVisibility GetPivotVisibility() {
    return TheInstance()->effectiveCtx().pivotVisibility;
}
```

- [ ] **Step 2: Fix `BubbleViewModeEnabled`**

Around line 2178-2179, change:

```cpp
inline static bool BubbleViewModeEnabled() {
    return TheInstance()->m_bubbleViewModeEnabled;
}
```

to:

```cpp
inline static bool BubbleViewModeEnabled() {
    return TheInstance()->effectiveCtx().bubbleViewModeEnabled;
}
```

- [ ] **Step 3: Fix `AutoPickPivotAtCenter`**

Around line 2202-2203, change:

```cpp
inline static bool AutoPickPivotAtCenter() {
    return TheInstance()->m_autoPickPivotAtCenter;
}
```

to:

```cpp
inline static bool AutoPickPivotAtCenter() {
    return TheInstance()->effectiveCtx().autoPickPivotAtCenter;
}
```

- [ ] **Step 4: Fix `IsRotationAxisLocked`**

Around line 2210-2211, change:

```cpp
inline static bool IsRotationAxisLocked() {
    return TheInstance()->m_rotationAxisLocked;
}
```

to:

```cpp
inline static bool IsRotationAxisLocked() {
    return TheInstance()->effectiveCtx().rotationAxisLocked;
}
```

- [ ] **Step 5: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer && cmake --build build --target ACloudViewer 2>&1 | tail -30`

Expected: Compilation succeeds.

- [ ] **Step 6: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h
git commit -m "refactor(phase-a): route pivot/bubble/rotation inline APIs through effectiveCtx()

GetPivotVisibility, BubbleViewModeEnabled, AutoPickPivotAtCenter,
IsRotationAxisLocked now read from effectiveCtx()."
```

---

### Task 4: Add Context-Aware Overloads for 5 Remaining Core APIs

**Rationale:** The roadmap requires "at least 5 core static APIs have context-aware versions." Currently 6 exist (`GetContext`, `GetGLCameraParameters`, `SetPointSize`, `SetLineWidth`, `SetCameraClip`, `SetCameraFovy`). Adding 5 more covers the most-accessed per-view state: `GetPivotVisibility`, `SetPickingMode`, `GetPickingMode`, `GetInteractionMode`, `SetInteractionMode`.

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h` (declaration section near line 213)
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (implementations)

- [ ] **Step 1: Declare 5 new context-aware overloads in the header**

In `libs/CV_db/include/ecvDisplayTools.h`, after line 222 (after `SetCameraFovy(ecvViewContext&, double)`), add:

```cpp
    /// Context-aware pivot visibility getter.
    static PivotVisibility GetPivotVisibility(const ecvViewContext& ctx);

    /// Context-aware interaction mode setter.
    static void SetInteractionMode(ecvViewContext& ctx,
                                   INTERACTION_FLAGS flags);

    /// Context-aware interaction mode getter.
    static INTERACTION_FLAGS GetInteractionMode(const ecvViewContext& ctx);

    /// Context-aware picking mode setter.
    static void SetPickingMode(ecvViewContext& ctx, PICKING_MODE mode);

    /// Context-aware picking mode getter.
    static PICKING_MODE GetPickingMode(const ecvViewContext& ctx);
```

- [ ] **Step 2: Implement the 5 overloads in the .cpp**

In `libs/CV_db/src/ecvDisplayTools.cpp`, find the existing context-aware implementations (near the `SetPointSize(ecvViewContext&, float)` block), and add after them:

```cpp
ecvDisplayTools::PivotVisibility ecvDisplayTools::GetPivotVisibility(
        const ecvViewContext& ctx) {
    return ctx.pivotVisibility;
}

void ecvDisplayTools::SetInteractionMode(ecvViewContext& ctx,
                                         INTERACTION_FLAGS flags) {
    ctx.interactionFlags = flags;
}

ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::GetInteractionMode(
        const ecvViewContext& ctx) {
    return ctx.interactionFlags;
}

void ecvDisplayTools::SetPickingMode(ecvViewContext& ctx, PICKING_MODE mode) {
    ctx.pickingMode = mode;
}

ecvDisplayTools::PICKING_MODE ecvDisplayTools::GetPickingMode(
        const ecvViewContext& ctx) {
    return ctx.pickingMode;
}
```

- [ ] **Step 3: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer && cmake --build build --target ACloudViewer 2>&1 | tail -30`

Expected: Compilation succeeds. No new warnings.

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "refactor(phase-a): add 5 context-aware API overloads

GetPivotVisibility, SetInteractionMode, GetInteractionMode,
SetPickingMode, GetPickingMode now have ecvViewContext& overloads.
Total context-aware APIs: 11 (exceeds Phase A requirement of 5)."
```

---

### Task 5: Validate Phase A Acceptance Criteria

**Rationale:** The roadmap lists 5 acceptance criteria for Phase A. This task verifies each one.

- [ ] **Step 1: Verify A.4.1 — `ecvViewContext` defined and compiles**

Run: `rg "struct.*ecvViewContext" libs/CV_db/include/ecvViewContext.h`

Expected: `struct CV_DB_LIB_API ecvViewContext {` — confirmed.

- [ ] **Step 2: Verify A.4.2 — `ecvGLView` uses `m_ctx`**

Run: `rg "ecvViewContext m_ctx" libs/VtkEngine/Visualization/ecvGLView.h`

Expected: `ecvViewContext m_ctx;` — confirmed.

- [ ] **Step 3: Verify A.4.3 — push/pull reads/writes `m_ctx`**

Run: `rg "pushStateToSingleton|pullStateFromSingleton" libs/VtkEngine/Visualization/ecvGLView.cpp`

Expected: Both are no-ops (reads happen via `effectiveCtx()` which routes to `m_ctx`).

- [ ] **Step 4: Verify A.4.4 — at least 5 context-aware static APIs**

Run: `rg "ecvViewContext" libs/CV_db/include/ecvDisplayTools.h | wc -l`

Expected: >= 11 lines (6 pre-existing + 5 new = 11 total overloads).

- [ ] **Step 5: Count remaining `TheInstance()->m_` direct reads in the header**

Run: `rg "TheInstance\(\)->m_" libs/CV_db/include/ecvDisplayTools.h | wc -l`

Expected: Significantly reduced from the original ~35. The remaining ones should be infrastructure members (`m_win`, `m_font`, `m_globalDBRoot`, etc.) that are genuinely global, NOT per-view state.

- [ ] **Step 6: Runtime regression test (manual)**

1. Launch ACloudViewer
2. Load a point cloud (e.g., `.ply` or `.obj`)
3. Create a second MDI window via `View > New 3D View`
4. Split one of them: verify both halves render correctly
5. Click in Window 1, rotate — Window 2 should NOT move
6. Click in Window 2, zoom — Window 1 should NOT change
7. Click object in DB tree — should highlight without crash
8. Close Window 2 — app should not crash
9. Close the split — remaining window should work normally

Expected: All operations pass without crash or cross-window state leakage.

- [ ] **Step 7: Commit acceptance verification log**

```bash
git add docs/superpowers/plans/
git commit -m "docs: add Phase A remaining plan and verification notes"
```

---

## Phase A Completion Summary

After all 5 tasks, the Phase A acceptance criteria status will be:

| Criterion | Status |
|-----------|--------|
| `ecvViewContext` class defined and compiles | DONE (prior work) |
| `ecvGLView` uses `m_ctx` replacing scattered members | DONE (prior work) |
| `pushStateToSingleton` / `pullStateFromSingleton` route through `m_ctx` | DONE (no-op, effectiveCtx() routes) |
| At least 5 core static APIs have context-aware versions | DONE (11 total after Task 4) |
| Inline static APIs route through `effectiveCtx()` not `m_*` | DONE (Tasks 1-3) |
| Regression: 2+ MDI + 1 split, basic ops no regression | VERIFIED (Task 5) |

**Next phase:** Phase B — Render Pipeline (eliminate `ScopedVisSwap` from `ecvGLView::redraw`).
