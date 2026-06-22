# Phase C: Interaction Pipeline — `QVTKWidgetCustom` De-Singleton

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `QVTKWidgetCustom` event handlers operate directly on the owning `ecvGLView`'s context instead of the singleton `ecvDisplayTools`. Eliminate `m_tools->m_*` direct reads.

**Architecture:** `QVTKWidgetCustom.h` already declares `ecvGLView* m_ownerView = nullptr` and has Phase C accessor stubs (lines 215-264). The migration has 3 stages: (1) wire `m_ownerView` during construction, (2) replace the 32 accessor-block `m_tools->m_*` returns with `m_ownerView->context()`, (3) replace the 6 non-accessor `m_tools->m_*` uses in event handlers.

**Tech Stack:** C++17, Qt 5/6, VTK, CMake

---

## Current state

| Metric | Count |
|---|---|
| `m_tools->m_*` reads/writes | **38** lines |
| `m_tools` type | `ecvDisplayTools*` |
| `m_ownerView` | Declared but `nullptr` |
| Accessor block (lines 93-245) | **32** lines |
| Event handler uses (lines 717-1510) | **6** lines |

### Categorized `m_tools->m_*` reads

| Category | Count | Members |
|---|---|---|
| Mouse state | 10 | `m_lastMousePos`, `m_lastMouseMovePos`, `m_mouseMoved`, `m_mouseButtonPressed`, `m_ignoreMouseReleaseEvent`, `m_widgetClicked`, `m_touchInProgress`, `m_touchBaseDist`, `m_showCursorCoordinates`, `m_lastClickTime_ticks` |
| Interaction flags | 1 | `m_interactionFlags` |
| Picking | 9 | `m_pickingMode`, `m_pickingModeLocked`, `m_pickRadius`, `m_allowRectangularEntityPicking`, `m_last_point_index`, `m_last_picked_id`, `m_rectPickingPoly`, `m_deferredPickingTimer` x2 |
| Viewport params | 2 | `m_viewportParams` (const + non-const) |
| Bubble/Pivot | 5 | `m_bubbleViewModeEnabled`, `m_bubbleViewFov_deg`, `m_pivotVisibility`, `m_pivotSymbolShown`, `m_autoPickPivotAtCenter` |
| Other | 11 | `m_clickableItemsVisible`, `m_customLightEnabled`, `m_customLightPos`, `m_rotationAxisLocked`, `m_lockedRotationAxis`, `m_activeItems`, `m_hotZone`, `m_timer`, `m_hotZoneOwnedBySingleton` |

---

### Task 1: Wire `m_ownerView` During Construction

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h` (constructor signature)
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` (constructor body)
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp` (pass `this` when constructing widget)

- [ ] **Step 1: Read `QVTKWidgetCustom` constructor**

Read `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` first 50 lines and `QVTKWidgetCustom.h` constructor declaration.

- [ ] **Step 2: Add `ecvGLView*` parameter to constructor**

Add `ecvGLView* ownerView = nullptr` parameter. Store it in `m_ownerView`.

```cpp
QVTKWidgetCustom::QVTKWidgetCustom(QWidget* parent, ecvGLView* ownerView)
    : QVTKOpenGLNativeWidget(parent), m_ownerView(ownerView) {
    // existing init ...
}
```

- [ ] **Step 3: Pass `this` from `ecvGLView` when creating widget**

In `ecvGLView.cpp`, where `m_vtkWidget = new QVTKWidgetCustom(...)`, add `this` as the owner view parameter.

- [ ] **Step 4: Add `ownerCtx()` helper to `QVTKWidgetCustom`**

```cpp
ecvViewContext& QVTKWidgetCustom::ownerCtx() {
    if (m_ownerView) return m_ownerView->context();
    return m_tools->effectiveCtx();  // fallback for primary view
}
const ecvViewContext& QVTKWidgetCustom::ownerCtx() const {
    if (m_ownerView) return m_ownerView->context();
    return m_tools->effectiveCtx();
}
```

- [ ] **Step 5: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`

- [ ] **Step 6: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h \
        libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp \
        libs/VtkEngine/Visualization/ecvGLView.cpp
git commit -m "refactor(phase-c): wire m_ownerView in QVTKWidgetCustom

QVTKWidgetCustom now receives its owning ecvGLView* at construction.
ownerCtx() returns the owning view's ecvViewContext, falling back to
the singleton's effectiveCtx() for the primary view."
```

---

### Task 2: Replace Accessor Block `m_tools->m_*` (32 lines)

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp:93-245`

- [ ] **Step 1: Read accessor block**

Read `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` lines 88-250.

- [ ] **Step 2: Replace all 32 accessor returns**

Each accessor currently returns `m_tools->m_fieldName`. Replace with `ownerCtx().fieldName`:

```cpp
// Before (example, line 93):
ecvGenericGLDisplay::INTERACTION_FLAGS& QVTKWidgetCustom::interactionFlags() {
    return m_tools->m_interactionFlags;
}

// After:
ecvGenericGLDisplay::INTERACTION_FLAGS& QVTKWidgetCustom::interactionFlags() {
    return ownerCtx().interactionFlags;
}
```

Apply this pattern to all 32 accessors. Fields that exist in `ecvViewContext`:
- `interactionFlags`, `viewportParams`, `lastMousePos`, `lastMouseMovePos`, `mouseMoved`, `mouseButtonPressed`, `ignoreMouseReleaseEvent`, `pickingMode`, `pickingModeLocked`, `pickRadius`, `allowRectangularEntityPicking`, `touchInProgress`, `touchBaseDist`, `clickableItemsVisible`, `bubbleViewModeEnabled`, `bubbleViewFov_deg`, `pivotVisibility`, `pivotSymbolShown`, `autoPickPivotAtCenter`, `customLightEnabled`, `customLightPos`, `rotationAxisLocked`, `lockedRotationAxis`, `showCursorCoordinates`, `lastClickTime_ticks`, `widgetClicked`

Fields NOT in `ecvViewContext` (need singleton fallback or addition):
- `m_activeItems` → add to `ecvViewContext` or keep via `m_tools`
- `m_hotZone` → keep via `m_tools` (Phase B deferred)
- `m_timer`, `m_deferredPickingTimer` → keep via `m_tools` (global)
- `m_hotZoneOwnedBySingleton` → keep via `m_tools`
- `m_rectPickingPoly` → keep via `m_tools`
- `m_last_point_index`, `m_last_picked_id` → available in `ecvViewContext` as `lastPointIndex`, `lastPickedId`

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`

- [ ] **Step 4: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp
git commit -m "refactor(phase-c): replace 32 accessor m_tools->m_* with ownerCtx()

QVTKWidgetCustom accessor block now reads/writes through ownerCtx()
(the owning view's ecvViewContext) instead of the singleton."
```

---

### Task 3: Replace Event Handler `m_tools->m_*` (6 lines)

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` (lines 717, 754, 897, 900, 1489, 1510)

- [ ] **Step 1: Read each event handler usage**

Read each line to understand context:
- Line 717: `m_tools->m_timer` — global timer, keep via `m_tools`
- Line 754: `m_tools->m_deferredPickingTimer` — global, keep via `m_tools`
- Line 897, 900: `m_tools->m_hotZoneOwnedBySingleton` — keep via `m_tools`
- Line 1489: `m_tools->m_timer` — keep via `m_tools`
- Line 1510: `m_tools->m_deferredPickingTimer` — keep via `m_tools`

- [ ] **Step 2: Replace with named accessor or keep as m_tools fallback**

For global/shared members (timer, deferred picking timer, hotZoneOwnedBySingleton), these should remain on `m_tools` until Phase E. Add clear comments explaining this is intentional.

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`

- [ ] **Step 4: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp
git commit -m "refactor(phase-c): document remaining 6 m_tools->m_* as Phase E targets

Timer, deferred picking timer, and hotZoneOwnedBySingleton are
global/shared state — intentionally kept on m_tools until Phase E."
```

---

### Task 4: Eliminate Foreign Wheel Patch

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` (`wheelEvent`)

- [ ] **Step 1: Read `wheelEvent` implementation**

Read around line 770-810.

- [ ] **Step 2: Remove `pushStateToSingleton` / `pullStateFromSingleton` calls**

Since all state now routes through `ownerCtx()`, the push/pull around foreign wheel events is redundant. Remove these calls. The wheel event operates directly on the owning view's context via `ownerCtx()`.

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`

- [ ] **Step 4: Runtime test**

1. Open two views with data
2. Wheel-zoom in each — isolated, no cross-contamination
3. No push/pull overhead

- [ ] **Step 5: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp
git commit -m "refactor(phase-c): remove push/pull from wheelEvent

With ownerCtx() routing, QVTKWidgetCustom reads/writes the correct
per-view state directly. push/pullStateToSingleton no longer needed
in wheelEvent — foreign wheel handled by direct context access."
```

---

### Task 5: Phase C Acceptance Verification

- [ ] **Step 1: Count remaining `m_tools->m_*`**

Run: `rg 'm_tools->m_' libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp | wc -l`

Expected: **6** (timer, deferred timer, hotZoneOwnedBySingleton — intentional Phase E targets).

- [ ] **Step 2: Verify `ownerCtx()` usage**

Run: `rg 'ownerCtx\(\)' libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp | wc -l`

Expected: ~32 (one per accessor).

- [ ] **Step 3: Runtime regression**

1. Multi-window rotate/zoom/pan — isolated
2. Picking in each view — correct entity
3. Hot zone works in each view
4. Bubble view mode in non-primary view
5. No crashes on view close

---

## Phase C Completion Summary

| Acceptance Criterion | Task |
|---|---|
| `QVTKWidgetCustom` has `m_ownerView` wired | Task 1 |
| Accessor block uses `ownerCtx()` | Task 2 |
| Event handlers documented, globals intentionally on `m_tools` | Task 3 |
| Foreign wheel no longer needs push/pull | Task 4 |
| `m_tools->m_*` count: 38 → 6 | Task 5 verification |
