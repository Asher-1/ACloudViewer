# Multi-Window ParaView Alignment Phase 3 — Singleton Complete Removal & Per-View Isolation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all remaining singleton patterns from ecvDisplayTools, fix cross-view state coupling, and achieve true ParaView-like multi-window independence.

**Architecture:** Each ecvGLView already has its own ecvViewContext via ScopedRenderOverride. This phase targets the remaining shared-state leaks: primaryDT() signal routing, static mouse orientation, ExclusiveFullScreen via resolveViewContext(), cross-view entity removal, and legacy singleton naming/flags.

**Tech Stack:** C++ (Qt 5/6, VTK), CMake build system

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `libs/CV_db/src/ecvDisplayTools.cpp` | Core display tools — signal emission, draw paths | **Modify**: Replace `primaryDT()` signal emission with per-view routing |
| `libs/CV_db/include/ecvDisplayTools.h` | Display tools interface — static APIs | **Modify**: Add view-explicit overloads, remove singleton naming |
| `libs/CV_db/src/ecvGenericDisplayTools.cpp` | Generic display tools — `GetInstance()` | **Modify**: Replace with view-parameterized methods |
| `libs/CV_db/include/ecvGenericDisplayTools.h` | Generic display tools interface | **Modify**: Add per-view transform methods |
| `libs/CV_db/src/ecvGenericGLDisplay.cpp` | Default implementations for display interface | **Modify**: Replace static fallbacks with asserts |
| `libs/CV_db/include/ecvGenericGLDisplay.h` | Display interface — comments | **Modify**: Update legacy comments |
| `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` | VTK widget — mouse handling | **Modify**: Move static trackball state to per-widget |
| `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` | VTK display tools — entity removal | **Modify**: Guard removeEntities per-view |
| `libs/VtkEngine/Visualization/ecvGLView.cpp` | Concrete view — rendering | **Modify**: Per-view fullscreen flag |

---

### Task 1: Fix static mouse orientation in StandardMode (Critical — Cross-View Trackball Bug)

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp:1294-1313`
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h`

The `s_lastMouseOrientation` is a `static` local variable shared by all `QVTKWidgetCustom` instances. Dragging in View A then View B reuses A's last orientation, causing incorrect trackball rotation.

- [ ] **Step 1: Move static mouse orientation to member variable**

In `QVTKWidgetCustom.h`, add a member variable:

```cpp
// In private section of QVTKWidgetCustom
CCVector3d m_lastMouseOrientation;
```

- [ ] **Step 2: Replace static with member in mouseMoveEvent**

In `QVTKWidgetCustom.cpp`, change the `StandardMode` rotation block:

```cpp
case StandardMode: {
    if (!curMouseMoved()) {
        m_lastMouseOrientation =
                displayTarget()
                        ->convertMousePositionToOrientation(
                                curLastMousePos().x(),
                                curLastMousePos().y());
    }

    CCVector3d currentMouseOrientation =
            displayTarget()
                    ->convertMousePositionToOrientation(x, y);
    rotMat = ccGLMatrixd::FromToRotation(
            m_lastMouseOrientation,
            currentMouseOrientation);
    m_lastMouseOrientation = currentMouseOrientation;
```

- [ ] **Step 3: Build and verify**

Run: `cd build_app && make -j48 2>&1 | tail -10`
Expected: Build succeeds with exit code 0

- [ ] **Step 4: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h
git commit -m "fix: move static trackball orientation to per-widget member for multi-window independence"
```

---

### Task 2: Remove `m_hotZoneOwnedBySingleton` flag (Cleanup)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h:2423-2425`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (destructor and DrawClickableItems)

- [ ] **Step 1: Remove flag declaration from header**

In `ecvDisplayTools.h`, remove:

```cpp
bool m_hotZoneOwnedBySingleton = false;
```

- [ ] **Step 2: Remove flag usage in DrawClickableItems**

In `ecvDisplayTools.cpp`, change the hot zone creation block (around line 3911-3913):

```cpp
if (!hotZone) {
    hotZone = new HotZone(hzWin);
}
```

Remove the line: `if (!display) primaryDT()->m_hotZoneOwnedBySingleton = true;`

- [ ] **Step 3: Clean up destructor references**

Search for `m_hotZoneOwnedBySingleton` in the destructor and remove any conditional deletion based on this flag. Each `ecvGLView` now owns its own `m_hotZone` and is responsible for its lifecycle.

- [ ] **Step 4: Build and verify**

Run: `cd build_app && make -j48 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "refactor: remove m_hotZoneOwnedBySingleton flag — each view owns its hot zone"
```

---

### Task 3: Fix `ExclusiveFullScreen()` static API to be view-explicit (Critical)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h:1916-1928`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (callers)

The static `ExclusiveFullScreen()` reads/writes via `resolveViewContext()` which may target the wrong view. Each `ecvGLView` already stores `m_ctx.exclusiveFullscreen` per-view.

- [ ] **Step 1: Add view-explicit overloads**

In `ecvDisplayTools.h`, add:

```cpp
static bool ExclusiveFullScreen(ecvGenericGLDisplay* view) {
    if (view && view->viewContext())
        return view->viewContext()->exclusiveFullscreen;
    return ecvViewManager::instance().resolveViewContext().exclusiveFullscreen;
}
static void SetExclusiveFullScreenFlage(bool state, ecvGenericGLDisplay* view) {
    if (view && view->viewContext()) {
        view->viewContext()->exclusiveFullscreen = state;
        return;
    }
    ecvViewManager::instance().resolveViewContext().exclusiveFullscreen = state;
}
```

- [ ] **Step 2: Update DrawClickableItems to use view-explicit overload**

In `ecvDisplayTools.cpp`, change `DrawClickableItems`:

```cpp
bool fullScreenEnabled = display ? ExclusiveFullScreen(display) : ExclusiveFullScreen();
```

- [ ] **Step 3: Build and verify**

Run: `cd build_app && make -j48 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "refactor: add view-explicit ExclusiveFullScreen overloads for multi-window"
```

---

### Task 4: Guard `removeEntities` cross-view fan-out (Critical)

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.cpp:918-939`

Currently `removeEntities` propagates removal to ALL registered views' VtkVis and ImageVis. This prevents ParaView-style independent pipelines.

- [ ] **Step 1: Add per-view guard to removeEntities**

In `VtkDisplayTools.cpp`, wrap the multi-window removal block with a guard that only propagates when the entity is truly shared (scene DB entities), not for UI overlays:

```cpp
// Multi-window: propagate scene entity removal to secondary views,
// but NOT 2D overlay removal (hot zone, labels, etc.)
if (context.removeEntityType != ENTITY_TYPE::ECV_TEXT2D &&
    context.removeEntityType != ENTITY_TYPE::ECV_RECTANGLE_2D &&
    context.removeEntityType != ENTITY_TYPE::ECV_MARK_POINT &&
    context.removeEntityType != ENTITY_TYPE::ECV_IMAGE &&
    context.removeEntityType != ENTITY_TYPE::ECV_CIRCLE_2D &&
    context.removeEntityType != ENTITY_TYPE::ECV_POLYLINE_2D &&
    context.removeEntityType != ENTITY_TYPE::ECV_LINES_2D) {
    const auto& views = ecvViewManager::instance().getAllViews();
    // ... existing fan-out logic for 3D scene entities only
}
```

- [ ] **Step 2: Build and verify**

Run: `cd build_app && make -j48 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add libs/VtkEngine/Visualization/VtkDisplayTools.cpp
git commit -m "fix: guard removeEntities to only propagate scene entities, not 2D overlays"
```

---

### Task 5: Replace `USE_2D` / `USE_VTK_PICK` class-level globals with per-view config (Moderate)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h:2417-2419`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp:84-85`
- Modify: `libs/CV_db/include/ecvViewContext.h`

- [ ] **Step 1: Add per-view flags to ecvViewContext**

In `ecvViewContext.h`:

```cpp
bool use2D = true;
bool useVtkPick = true;
```

- [ ] **Step 2: Initialize from global defaults in ecvGLView**

In `ecvGLView` constructor or `initVtkPipeline`, copy global defaults:

```cpp
m_ctx.use2D = ecvDisplayTools::USE_2D;
m_ctx.useVtkPick = ecvDisplayTools::USE_VTK_PICK;
```

- [ ] **Step 3: Build and verify**

Run: `cd build_app && make -j48 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h libs/CV_db/src/ecvDisplayTools.cpp libs/CV_db/include/ecvViewContext.h libs/VtkEngine/Visualization/ecvGLView.cpp
git commit -m "refactor: migrate USE_2D/USE_VTK_PICK to per-view ecvViewContext"
```

---

### Task 6: Update legacy singleton comments and naming (Low — Documentation)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h` (comments)
- Modify: `libs/CV_db/include/ecvGenericGLDisplay.h` (comments)
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (comments)

- [ ] **Step 1: Replace "singleton" references in comments**

Search for "singleton" in all three files and replace with accurate descriptions:

- "singleton wrapper" → "shared display tools (managed by ecvViewManager)"
- "singleton primary view" → "primary view (managed by ecvViewManager)"
- "singleton state" → "shared state"

- [ ] **Step 2: Rename `m_pickingTargetView` comment**

Change comment from "primary/singleton" to "null means resolve via active view".

- [ ] **Step 3: Build and verify**

Run: `cd build_app && make -j48 2>&1 | tail -10`
Expected: Build succeeds (comment-only changes)

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h libs/CV_db/include/ecvGenericGLDisplay.h libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "docs: update legacy singleton references to reflect ecvViewManager architecture"
```

---

## Deferred (Future Phase 4)

These items are high-effort and require broader architectural changes:

| Gap | Description | Reason for Deferral |
|-----|-------------|-------------------|
| `primaryDT()` signal emission (~30 sites) | Signals emitted from shared engine, losing view attribution | Requires introducing per-view signal relay or moving signals to ecvGLView |
| `GetInstance()` in ecvGenericDisplayTools | Static helper methods lack view context | Requires adding view parameter to all transform helpers and updating all callers |
| Process-wide static fallbacks in ecvGenericGLDisplay.cpp | Default `activeItemsRef()`, `hotZonePtrRef()`, `clickableItemsRef()` | Requires verifying all live views override before converting to asserts |
| Centralized DB roots in primaryDT() (~47 sites) | `m_globalDBRoot`, `m_winDBRoot` on shared engine | Massive refactor touching scene management architecture |
