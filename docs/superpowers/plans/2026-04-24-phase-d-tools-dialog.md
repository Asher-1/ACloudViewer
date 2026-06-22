# Phase D: Tools & Dialog Refactoring — Explicit View Binding

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** All overlay dialogs and tools explicitly know which `ecvGLView` they operate on. Picking pipeline uses per-view context. Properties panel refreshes on view switch.

**Architecture:** `ccOverlayDialog` already has `bindToView(ecvGenericGLDisplay*)` / `m_boundView` (Phase D prep). The migration has 3 stages: (1) make `ccPickingHub` picking use per-view context, (2) ensure all 17 `ccOverlayDialog` subclasses use `bindToView` or `followActiveView`, (3) connect properties panel to `activeViewChanged`.

**Tech Stack:** C++17, Qt 5/6, VTK, CMake

---

## Current state

- **17 `ccOverlayDialog` subclasses** (11 in `app/`, 3 in plugins, 1 in `CVAppCommon`)
- `ccOverlayDialog` already has: `bindToView()`, `getBoundView()`, `m_boundView`, `linkWith()`
- `ccPickingHub` tracks `m_activeWindow` (`QWidget*`), updated on MDI activation
- Properties panel updated via `rebindToolsToActiveView` → `setVisualizer()`
- `doPicking()` reads from `effectiveCtx()` (Phase A bridging)

---

### Task 1: Make `doPicking` Use Per-View Context Explicitly

**Files:**
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (`doPicking` method)

- [ ] **Step 1: Read current `doPicking`**

Read the `doPicking` method to see which context reads exist.

- [ ] **Step 2: Verify all reads go through `effectiveCtx()`**

After Phase A, `doPicking` should read `lastPickedPoint`, `lastPointIndex`, `lastPickedId`, `pickingMode`, `pickRadius` from `effectiveCtx()`. Verify no direct `m_last_picked_*` reads remain.

- [ ] **Step 3: If direct reads remain, replace with `effectiveCtx()`**

Replace any `s_tools.instance->m_pickingMode` with `s_tools.instance->effectiveCtx().pickingMode`, etc.

- [ ] **Step 4: Build and verify**

- [ ] **Step 5: Commit**

```bash
git add libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "refactor(phase-d): doPicking reads entirely from effectiveCtx()

All picking state (mode, radius, last picked point/index/id) now
read from the active view's ecvViewContext, not singleton members."
```

---

### Task 2: Audit and Wire `bindToView` in Overlay Dialogs

**Files:**
- Audit: All 17 `ccOverlayDialog` subclasses

Subclass list:
1. `ecvAnimationParamDlg` — `app/ecvAnimationParamDlg.h`
2. `ccTracePolylineTool` — `app/ecvTracePolylineTool.h`
3. `ccGraphicalTransformationTool` — `app/ecvGraphicalTransformationTool.h`
4. `ecvDeepSemanticSegmentationTool` — `app/ecvDeepSemanticSegmentationTool.h`
5. `ecvFilterByLabelDlg` — `app/ecvFilterByLabelDlg.h`
6. `ccGraphicalSegmentationTool` — `app/ecvGraphicalSegmentationTool.h`
7. `ccPointPairRegistrationDlg` — `app/ecvPointPairRegistrationDlg.h`
8. `ecvCameraParamEditDlg` — `libs/CVAppCommon/include/ecvCameraParamEditDlg.h`
9. `ecvAnnotationsTool` — `app/ecvAnnotationsTool.h`
10. `ccCloudLayersDlg` — `plugins/core/Standard/qCloudLayers/include/ccCloudLayersDlg.h`
11. `ecvFilterTool` — `app/ecvFilterTool.h`
12. `ecvMeasurementTool` — `app/ecvMeasurementTool.h`
13. `ccPointPickingGenericInterface` — `app/ecvPointPickingGenericInterface.h`
14. `ccMapDlg` — `plugins/core/Standard/qCompass/include/ccMapDlg.h`
15. `ccCompassDlg` — `plugins/core/Standard/qCompass/include/ccCompassDlg.h`
16. `ccMPlaneDlg` — `plugins/core/Standard/qMPlane/src/ccMPlaneDlg.h`

- [ ] **Step 1: For each subclass, check if it already calls `bindToView` or `linkWith`**

```bash
rg 'bindToView\|linkWith' app/ecv*.h app/ecv*.cpp app/cc*.h app/cc*.cpp \
    libs/CVAppCommon/ plugins/ -l
```

- [ ] **Step 2: Add `followActiveView()` to tools that should track the active view**

Tools like segmentation, measurement, annotation should follow the active view. Add in their constructors or `start()` methods:

```cpp
void ecvMeasurementTool::start() {
    followActiveView();  // from ccOverlayDialog
    // existing start logic ...
}
```

- [ ] **Step 3: Add `bindToView()` to tools that operate on a fixed view**

Camera edit dialog, animation dialog should bind to the view they were opened from:

```cpp
void ecvCameraParamEditDlg::linkWith(QWidget* win) {
    ccOverlayDialog::linkWith(win);
    auto* display = ecvGenericGLDisplay::FromWidget(win);
    if (display) bindToView(display);
}
```

- [ ] **Step 4: Build and verify**

- [ ] **Step 5: Commit**

```bash
git add app/ libs/CVAppCommon/ plugins/
git commit -m "refactor(phase-d): overlay dialogs use bindToView/followActiveView

All 17 ccOverlayDialog subclasses now explicitly track their target
view. Tools that follow active view use followActiveView(); tools
that operate on a fixed view use bindToView()."
```

---

### Task 3: Connect Properties Panel to `activeViewChanged`

**Files:**
- Modify: `app/MainWindow.cpp` (`rebindToolsToActiveView` or init)

- [ ] **Step 1: Verify current properties delegate update path**

Current: `rebindToolsToActiveView` → `m_ccRoot->getPropertiesDelegate()->setVisualizer(activeViewer)`.

- [ ] **Step 2: Ensure properties refresh includes per-view display parameters**

When active view changes, the properties panel should reflect that view's display parameters (point size, line width, etc.) rather than singleton defaults.

- [ ] **Step 3: Build and verify**

- [ ] **Step 4: Runtime test**

1. Open two views with different point sizes
2. Switch active view — properties panel updates to show new view's parameters
3. Change parameter in properties — affects only active view

- [ ] **Step 5: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "refactor(phase-d): properties panel refreshes on activeViewChanged

PropertiesDelegate::setVisualizer called with the new active view's
VTK visualizer when activeViewChanged fires."
```

---

### Task 4: Phase D Acceptance Verification

- [ ] **Step 1: Verify all dialogs use view binding**

```bash
rg 'bindToView\|followActiveView' app/ libs/CVAppCommon/ plugins/ -l
```

Expected: All 17 subclass files (or their parent constructors).

- [ ] **Step 2: Verify `doPicking` has no direct singleton picking reads**

```bash
rg 's_tools\.instance->m_last_pick\|s_tools\.instance->m_pickingMode' libs/CV_db/src/ecvDisplayTools.cpp
```

Expected: No matches (all via `effectiveCtx()`).

- [ ] **Step 3: Runtime regression**

1. Segmentation tool in Window 1 — draws on Window 1 only
2. Switch active view — tool follows or stays bound
3. Measurement tool works in non-primary view
4. Close bound view — tool closes gracefully
