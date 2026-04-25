# Phase E: Singleton Cleanup — `ecvDisplayTools` Slimdown

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete all per-view `m_*` members from `ecvDisplayTools` that are already in `ecvViewContext`. Delete `pushStateToSingleton` / `pullStateFromSingleton`. Delete `ScopedHotZoneRender`. Reduce `s_tools.instance->m_*` reads to < 50 (only global members).

**Architecture:** After Phases A-D, all code paths read/write per-view state through `ecvViewContext` (via `effectiveCtx()` or `ownerCtx()`). The singleton still holds duplicate per-view `m_*` members that are no longer read. Phase E deletes these duplicates, removes the push/pull mechanism, and cleans up the now-unused swap RAII guards.

**Tech Stack:** C++17, Qt 5/6, VTK, CMake

---

## Current state (post Phase D)

| Metric | Expected count |
|---|---|
| `s_tools.instance->m_*` in `ecvDisplayTools.cpp` | ~50-80 (was ~102, will reduce) |
| `m_tools->m_*` in `QVTKWidgetCustom.cpp` | ~6 (global: timer, deferred timer, hotZone) |
| `push/pullStateToSingleton` | 8 refs (all empty body / declarations) |
| `ScopedHotZoneRender` | ~8 refs |
| Per-view `m_*` on singleton | ~30 fields (mirrored in ecvViewContext) |

---

### Task 1: Delete Per-View `m_*` Members From `ecvDisplayTools`

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h` (member declarations)
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (initialization, any remaining reads)

- [ ] **Step 1: List per-view members to delete**

These members have equivalents in `ecvViewContext` and should be removed from `ecvDisplayTools`:

| Singleton member | ecvViewContext equivalent |
|---|---|
| `m_viewportParams` | `viewportParams` |
| `m_glViewport` | `glViewport` |
| `m_viewMatd` / `m_projMatd` | `viewMatd` / `projMatd` |
| `m_validModelviewMatrix` / `m_validProjectionMatrix` | `validModelviewMatrix` / `validProjectionMatrix` |
| `m_cameraToBBCenterDist` / `m_bbHalfDiag` | `cameraToBBCenterDist` / `bbHalfDiag` |
| `m_interactionFlags` | `interactionFlags` |
| `m_pickingMode` / `m_pickingModeLocked` / `m_pickRadius` | `pickingMode` / `pickingModeLocked` / `pickRadius` |
| `m_allowRectangularEntityPicking` | `allowRectangularEntityPicking` |
| `m_lastMousePos` / `m_lastMouseMovePos` | `lastMousePos` / `lastMouseMovePos` |
| `m_mouseMoved` / `m_mouseButtonPressed` / `m_ignoreMouseReleaseEvent` | `mouseMoved` / `mouseButtonPressed` / `ignoreMouseReleaseEvent` |
| `m_touchInProgress` / `m_touchBaseDist` | `touchInProgress` / `touchBaseDist` |
| `m_bubbleViewModeEnabled` / `m_bubbleViewFov_deg` | `bubbleViewModeEnabled` / `bubbleViewFov_deg` |
| `m_pivotVisibility` / `m_pivotSymbolShown` / `m_autoPickPivotAtCenter` | `pivotVisibility` / `pivotSymbolShown` / `autoPickPivotAtCenter` |
| `m_clickableItemsVisible` / `m_displayOverlayEntities` | `clickableItemsVisible` / `displayOverlayEntities` |
| `m_exclusiveFullscreen` / `m_showCursorCoordinates` / `m_showDebugTraces` | `exclusiveFullscreen` / `showCursorCoordinates` / `showDebugTraces` |
| `m_rotationAxisLocked` / `m_lockedRotationAxis` | `rotationAxisLocked` / `lockedRotationAxis` |
| `m_sunLightEnabled` / `m_customLightEnabled` / `m_customLightPos` | `sunLightEnabled` / `customLightEnabled` / `customLightPos` |
| `m_lastClickTime_ticks` / `m_widgetClicked` | `lastClickTime_ticks` / `widgetClicked` |
| `m_last_picked_point` / `m_last_point_index` / `m_last_picked_id` | `lastPickedPoint` / `lastPointIndex` / `lastPickedId` |

- [ ] **Step 2: Search for remaining reads of each field**

For each field, run: `rg 's_tools\.instance->m_FIELD\|TheInstance\(\)->m_FIELD' libs/CV_db/`

Any remaining reads MUST be migrated to `effectiveCtx()` before deletion.

- [ ] **Step 3: Delete member declarations from header**

Remove each `m_*` declaration from `ecvDisplayTools.h`.

- [ ] **Step 4: Delete initialization from constructor/init**

In `ecvDisplayTools.cpp`, remove initialization of deleted fields.

- [ ] **Step 5: Build — fix compilation errors**

Each error points to a remaining singleton read that was missed. Fix by routing through `effectiveCtx()`.

- [ ] **Step 6: Commit**

```bash
git add libs/CV_db/
git commit -m "refactor(phase-e): delete per-view m_* members from ecvDisplayTools

~30 per-view member fields removed from the singleton. All reads
now go through ecvViewContext via effectiveCtx() or ownerCtx()."
```

---

### Task 2: Delete `pushStateToSingleton` / `pullStateFromSingleton`

**Files:**
- Modify: `libs/CV_db/include/ecvGenericGLDisplay.h` (virtual declarations)
- Modify: `libs/VtkEngine/Visualization/ecvGLView.h` (override declarations)
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp` (implementations)

- [ ] **Step 1: Verify push/pull are empty bodies**

```bash
rg 'pushStateToSingleton|pullStateFromSingleton' libs/ -A 2
```

Expected: All implementations are `{}` (empty).

- [ ] **Step 2: Search for callers**

```bash
rg 'pushStateToSingleton\(\)|pullStateFromSingleton\(\)' libs/ app/
```

If callers exist, remove those calls.

- [ ] **Step 3: Delete declarations and implementations**

Remove from `ecvGenericGLDisplay.h`, `ecvGLView.h`, and `ecvGLView.cpp`.

- [ ] **Step 4: Build and verify**

- [ ] **Step 5: Commit**

```bash
git add libs/
git commit -m "refactor(phase-e): delete pushStateToSingleton/pullStateFromSingleton

These methods have been empty since Phase A. With per-view state
living in ecvViewContext, there is nothing to push or pull."
```

---

### Task 3: Delete `ScopedHotZoneRender`

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.h` (class declaration)
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` (implementation)
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp` (usage in `redraw`)

**Prerequisite:** Hot zone drawing must already be per-view (Phase B deferred Task 2b, implemented during Phase C). `ecvGLView::redraw` must draw hot zone directly without `ScopedHotZoneRender`.

- [ ] **Step 1: Replace usage in `ecvGLView::redraw` with direct draw**

The `ScopedHotZoneRender` usage in `redraw` should already be replaced by a direct `drawHotZoneForView()` call (from Phase B Task 2b).

- [ ] **Step 2: Delete `ScopedHotZoneRender` class**

Remove the class declaration from `VtkDisplayTools.h` and implementation from `VtkDisplayTools.cpp`.

- [ ] **Step 3: Delete `beginPrimaryRender` / `endPrimaryRender`**

If `RedrawDisplay` no longer uses the legacy fallback (Task 3a replaced it with delegation), these can be deleted.

- [ ] **Step 4: Build and verify**

- [ ] **Step 5: Commit**

```bash
git add libs/VtkEngine/
git commit -m "refactor(phase-e): delete ScopedHotZoneRender and beginPrimaryRender

Drawing pipeline is fully per-view. No more singleton pointer swapping."
```

---

### Task 4: Delete Remaining 6 `m_tools->m_*` in `QVTKWidgetCustom`

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp`

- [ ] **Step 1: Migrate timer/hotZone to appropriate owner**

- `m_timer` → move to `ecvViewContext` or keep as app-global via `ecvDisplayTools::timer()`
- `m_deferredPickingTimer` → move to per-view or keep as app-global
- `m_hotZoneOwnedBySingleton` → remove (hot zone is now per-view)

- [ ] **Step 2: Remove `m_tools` member entirely**

After all reads are migrated, `QVTKWidgetCustom` no longer needs `m_tools`.

- [ ] **Step 3: Build and verify**

- [ ] **Step 4: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h \
        libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp
git commit -m "refactor(phase-e): remove m_tools from QVTKWidgetCustom

QVTKWidgetCustom no longer holds an ecvDisplayTools pointer.
All state access goes through m_ownerView->context()."
```

---

### Task 5: Phase E Acceptance Verification

- [ ] **Step 1: Count singleton direct reads**

```bash
rg 's_tools\.instance->m_' libs/CV_db/src/ecvDisplayTools.cpp | wc -l
```

Expected: < 50 (only global: `m_globalDBRoot`, `m_win`, `m_winDBRoot`, `m_timer`, `m_font`, etc.)

- [ ] **Step 2: Count `m_tools` in widget**

```bash
rg 'm_tools' libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp | wc -l
```

Expected: 0.

- [ ] **Step 3: Verify no push/pull references**

```bash
rg 'pushStateToSingleton\|pullStateFromSingleton' libs/ app/
```

Expected: 0.

- [ ] **Step 4: Verify no ScopedVisSwap/ScopedHotZoneRender**

```bash
rg 'ScopedVisSwap\|ScopedHotZoneRender' libs/
```

Expected: 0 (or comments only).

- [ ] **Step 5: Runtime regression — full feature test**

1. Multi-window with data — independent rendering
2. Picking in each view — correct
3. Hot zone in each view — functional
4. View creation/closure — no crashes
5. All overlay tools — functional
6. Properties panel — updates on view switch
