# ParaView Alignment Remaining Gaps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all remaining ParaView multi-window alignment gaps (documented and undocumented) to reach 100% parity across layout persistence, view association, selection, and zoom behavior.

**Architecture:** Fix 7 concrete issues across 3 categories: (A) session persistence robustness, (B) model/view association correctness, (C) cross-view selection/zoom consistency. Each task is independent and can be committed separately.

**Tech Stack:** C++ / Qt 5 / VTK / CMake

---

## Current State

| Category | Documented Gap | Status |
|----------|---------------|--------|
| GAP-T: Source Undo | `ecvPropertyChangeCommand` etc. | Deferred (LOW, 4-6 weeks) |
| Phase P: View Type Registry | Only needed for SpreadSheet/Chart | Deferred (LOW, optional) |
| Layout camera persistence | NOT documented — discovered in audit | **NEW — HIGH** |
| `restoreLayoutState` leak | NOT documented — discovered in audit | **NEW — HIGH (latent)** |
| Zoom-on-selection view mismatch | NOT documented — discovered in audit | **NEW — HIGH** |
| Plugin association hazard | Partially documented | **NEW — MEDIUM** |
| DB tree per-view eye icon | Not explicitly documented as gap | **MEDIUM** (deferred) |

This plan addresses the 5 HIGH/MEDIUM gaps. GAP-T and Phase P are explicitly deferred.

---

## File Map

| File | Responsibility | Tasks |
|------|---------------|-------|
| `libs/CV_db/src/ecvViewLayoutProxy.cpp` | KD-tree layout model, JSON serialization | 1 |
| `libs/CV_db/include/ecvViewLayoutProxy.h` | Layout proxy header | 1 |
| `app/ecvTabbedMultiViewWidget.cpp` | Tab container, save/restore layout | 1, 2 |
| `app/ecvTabbedMultiViewWidget.h` | Tab container header | 2 |
| `app/MainWindow.cpp` | Main app, zoom, model loading | 3, 4 |
| `libs/CV_db/src/ecvViewManager.cpp` | Active view, model association | 4 |
| `libs/CV_db/include/ecvViewManager.h` | View manager header | 4 |
| `libs/VtkEngine/Visualization/VtkVis.h` | Per-view VTK visualizer (camera state) | 1 |
| `libs/VtkEngine/Visualization/VtkVis.cpp` | VTK camera save/load | 1 |

---

### Task 1: Layout Session Restore — Save/Load Per-View Camera State ✅

**Problem:** `ecvViewLayoutProxy::saveState()` serializes the KD-tree split structure and `view_id`, but does **not** save any camera state (position, focal point, view angle, projection mode). After session restore, all views reset to default cameras instead of restoring the user's previous viewpoint.

**ParaView pattern:** `vtkSMViewProxy` serializes all camera properties (position, focal point, view up, parallel scale, parallel projection flag) as part of the XML state file.

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkVis.h`
- Modify: `libs/VtkEngine/Visualization/VtkVis.cpp`
- Modify: `libs/CV_db/src/ecvViewLayoutProxy.cpp`
- Modify: `libs/CV_db/include/ecvViewLayoutProxy.h`
- Modify: `app/ecvTabbedMultiViewWidget.cpp`

- [ ] **Step 1: Add camera serialization to VtkVis**

Read `VtkVis.h` to find the `CameraParams` struct, then add `toJson()` / `fromJson()` methods.

```cpp
// In VtkVis.h, add to existing CameraParams struct:
QJsonObject toJson() const;
static CameraParams fromJson(const QJsonObject& obj);

// In VtkVis.h, add public methods:
QJsonObject saveCameraToJson() const;
void loadCameraFromJson(const QJsonObject& json);
```

```cpp
// In VtkVis.cpp:
QJsonObject VtkVis::CameraParams::toJson() const {
    QJsonObject obj;
    obj["pos_x"] = position[0];
    obj["pos_y"] = position[1];
    obj["pos_z"] = position[2];
    obj["fp_x"] = focalPoint[0];
    obj["fp_y"] = focalPoint[1];
    obj["fp_z"] = focalPoint[2];
    obj["up_x"] = viewUp[0];
    obj["up_y"] = viewUp[1];
    obj["up_z"] = viewUp[2];
    obj["view_angle"] = viewAngle;
    obj["parallel_scale"] = parallelScale;
    obj["parallel_projection"] = parallelProjection;
    return obj;
}

VtkVis::CameraParams VtkVis::CameraParams::fromJson(const QJsonObject& obj) {
    CameraParams p;
    p.position[0] = obj["pos_x"].toDouble();
    p.position[1] = obj["pos_y"].toDouble();
    p.position[2] = obj["pos_z"].toDouble();
    p.focalPoint[0] = obj["fp_x"].toDouble();
    p.focalPoint[1] = obj["fp_y"].toDouble();
    p.focalPoint[2] = obj["fp_z"].toDouble();
    p.viewUp[0] = obj["up_x"].toDouble();
    p.viewUp[1] = obj["up_y"].toDouble();
    p.viewUp[2] = obj["up_z"].toDouble();
    p.viewAngle = obj["view_angle"].toDouble(30.0);
    p.parallelScale = obj["parallel_scale"].toDouble(1.0);
    p.parallelProjection = obj["parallel_projection"].toBool(false);
    return p;
}

QJsonObject VtkVis::saveCameraToJson() const {
    auto* cam = getCurrentRenderer() ? getCurrentRenderer()->GetActiveCamera() : nullptr;
    if (!cam) return {};
    CameraParams p;
    cam->GetPosition(p.position);
    cam->GetFocalPoint(p.focalPoint);
    cam->GetViewUp(p.viewUp);
    p.viewAngle = cam->GetViewAngle();
    p.parallelScale = cam->GetParallelScale();
    p.parallelProjection = (cam->GetParallelProjection() != 0);
    return p.toJson();
}

void VtkVis::loadCameraFromJson(const QJsonObject& json) {
    if (json.isEmpty()) return;
    auto p = CameraParams::fromJson(json);
    applyCameraState(p);
}
```

- [ ] **Step 2: Verify VtkVis changes compile**

Run: `cmake --build build --target CV_VtkEngine -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Build succeeds with no errors.

- [ ] **Step 3: Extend layout JSON to include camera state**

In `ecvViewLayoutProxy.cpp`, modify `saveState()` to include camera data for each leaf cell's view:

```cpp
// In saveState(), inside the cell serialization loop, after writing view_id:
if (cell.view) {
    auto* glView = dynamic_cast<ecvGLView*>(cell.view);
    if (glView && glView->getVisualizer3D()) {
        cellObj["camera"] = glView->getVisualizer3D()->saveCameraToJson();
    }
}
```

In `loadState()`, after resolving the view for a cell, apply camera:

```cpp
// After assignView(cell.index, view):
if (cellObj.contains("camera")) {
    auto* glView = dynamic_cast<ecvGLView*>(view);
    if (glView && glView->getVisualizer3D()) {
        glView->getVisualizer3D()->loadCameraFromJson(cellObj["camera"].toObject());
    }
}
```

- [ ] **Step 4: Verify layout save/load with camera**

Run: `cmake --build build --target ACloudViewer -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Build succeeds. Manual test: create 2 views, rotate cameras differently, close/reopen app, verify cameras are restored.

- [ ] **Step 5: Commit**

```bash
git add libs/VtkEngine/Visualization/VtkVis.h libs/VtkEngine/Visualization/VtkVis.cpp \
       libs/CV_db/src/ecvViewLayoutProxy.cpp libs/CV_db/include/ecvViewLayoutProxy.h
git commit -m "feat: persist per-view camera state in layout session JSON"
```

---

### Task 2: Fix `restoreLayoutState` Layout Leak ✅

**Problem:** `ecvTabbedMultiViewWidget::restoreLayoutState()` tears down extra tabs by calling `destroyAllViews()` + `deleteLater()` on the widget, but does NOT call `ecvViewManager::unregisterLayout(layout)`. This leaks `ecvViewLayoutProxy*` entries in `ecvViewManager`'s registry. (Compare with `closeTab()` which properly calls `unregisterLayout`.)

**Files:**
- Modify: `app/ecvTabbedMultiViewWidget.cpp`

- [ ] **Step 1: Add `unregisterLayout` before teardown**

Search `restoreLayoutState` in `ecvTabbedMultiViewWidget.cpp` for the loop that removes extra tabs:

```cpp
// Find the block that removes excess tabs during restore.
// Before the widget->deleteLater() call, add:
auto* layout = widget->layoutManager();
if (layout) {
    ecvViewManager::instance().unregisterLayout(layout);
}
```

- [ ] **Step 2: Verify build**

Run: `cmake --build build --target ACloudViewer -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add app/ecvTabbedMultiViewWidget.cpp
git commit -m "fix: unregister layout proxy before tearing down tabs in restoreLayoutState"
```

---

### Task 3: Fix `zoomOnSelectedEntities` View Mismatch ✅

**Problem:** `MainWindow::zoomOnSelectedEntities()` always uses `getActiveGLView()` for the zoom target, ignoring the owning view of the selected entities. Other zoom helpers (`zoomOn`, `zoomOnEntities`) correctly call `findViewForEntity` first. This causes incorrect camera manipulation when the user selects entities in View B but View A is active.

**ParaView pattern:** Zoom operations always target the view that contains the representation.

**Files:**
- Modify: `app/MainWindow.cpp`

- [ ] **Step 1: Read current `zoomOnSelectedEntities` implementation**

Read `MainWindow.cpp` around line 6100 to understand the current logic.

- [ ] **Step 2: Add owner-view resolution**

```cpp
void MainWindow::zoomOnSelectedEntities() {
    const auto& selected = getSelectedEntities();
    if (selected.empty()) return;

    // Find the view that owns the first selected entity.
    auto& vm = ecvViewManager::instance();
    auto* ownerView = vm.findViewForEntity(selected.front());
    if (ownerView && ownerView != vm.getActiveView()) {
        vm.setActiveView(ownerView);
    }

    ecvGLView* win = getActiveGLView();
    if (!win) return;

    ccBBox box;
    for (ccHObject* entity : selected) {
        box += entity->getDisplayBB_recursive(false, win);
    }
    if (box.isValid()) {
        win->zoomGlobal(box);
    }
}
```

- [ ] **Step 3: Verify build**

Run: `cmake --build build --target ACloudViewer -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix: zoomOnSelectedEntities targets the entity's owner view"
```

---

### Task 4: Harden Plugin Model Association ✅

**Problem:** `ecvViewManager::associateToActiveView()` only sets the display when `getDisplay()` is null. Plugins calling `m_app->addToDB(obj)` with objects that already have a display set (e.g., cloned objects, objects from import pipelines) will silently skip association, potentially keeping the object attached to a stale or wrong view. Additionally, long-running plugin dialogs can shift the active view before association.

**ParaView pattern:** `pqObjectBuilder::createRepresentation()` always creates a new representation for the current active view, regardless of prior state.

**Files:**
- Modify: `libs/CV_db/src/ecvViewManager.cpp`
- Modify: `libs/CV_db/include/ecvViewManager.h`
- Modify: `app/MainWindow.cpp`

- [ ] **Step 1: Add `forceAssociateToView` method**

```cpp
// In ecvViewManager.h, add:
void forceAssociateToView(ccHObject* obj, ecvGenericGLDisplay* view);

// In ecvViewManager.cpp:
void ecvViewManager::forceAssociateToView(ccHObject* obj,
                                           ecvGenericGLDisplay* view) {
    if (!obj || !view) return;
    obj->setDisplay_recursive(view);
}
```

- [ ] **Step 2: Use `forceAssociateToView` in the addToDB(ccHObject*) overload for user-initiated loads**

In `MainWindow::addToDB(ccHObject*, ...)`, after the existing `associateToActiveView` call, add a force path when the object still has a mismatched display:

```cpp
// After: vm.associateToActiveView(obj);
// Add:
auto* activeView = vm.getActiveView();
if (activeView && obj->getDisplay() && obj->getDisplay() != activeView) {
    CVLog::Print("[addToDB] Rebinding entity to active view");
    vm.forceAssociateToView(obj, activeView);
}
```

- [ ] **Step 3: Verify build**

Run: `cmake --build build --target ACloudViewer -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/src/ecvViewManager.cpp libs/CV_db/include/ecvViewManager.h app/MainWindow.cpp
git commit -m "feat: add forceAssociateToView for robust plugin model association"
```

---

### Task 5: Fix `copyPrimaryViewConfig` Timing ✅

**Problem:** `MainWindow::copyPrimaryViewConfig()` reads from `dt->effectiveCtx()` when seeding a new view. If the effective context points to a different view than intended (e.g., during a tab switch or event processing), the new view inherits incorrect camera/interaction state.

**Files:**
- Modify: `app/MainWindow.cpp`

- [ ] **Step 1: Read current `copyPrimaryViewConfig` implementation**

Search for `copyPrimaryViewConfig` in `MainWindow.cpp` and read the function.

- [ ] **Step 2: Use explicit source view parameter**

Instead of relying on `effectiveCtx()`, pass the explicit source view:

```cpp
void MainWindow::copyPrimaryViewConfig(ecvGLView* newView) {
    // Use the active view as the source, not effectiveCtx()
    auto* sourceView = dynamic_cast<ecvGLView*>(
        ecvViewManager::instance().getActiveView());
    if (!sourceView || sourceView == newView) return;

    const auto& srcCtx = sourceView->context();
    auto& dstCtx = newView->context();

    // Copy relevant display parameters
    dstCtx.viewportParams.defaultPointSize = srcCtx.viewportParams.defaultPointSize;
    dstCtx.viewportParams.defaultLineWidth = srcCtx.viewportParams.defaultLineWidth;
    dstCtx.viewportParams.perspectiveView = srcCtx.viewportParams.perspectiveView;
    dstCtx.viewportParams.objectCenteredView = srcCtx.viewportParams.objectCenteredView;

    // Reset transient interaction state for the new view
    dstCtx.resetInteractionState();
}
```

- [ ] **Step 3: Verify build**

Run: `cmake --build build --target ACloudViewer -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix: copyPrimaryViewConfig uses explicit source view instead of effectiveCtx"
```

---

### Task 6: Update Documentation ✅

**Files:**
- Modify: `docs/user-guide/multi-window-paraview-alignment-design.md`

- [ ] **Step 1: Update the alignment matrix**

In §2.9 (Session Persistence), update the project file row:
```
| Project file | `.pvsm` (SM state XML) | `.acv` project file + session camera restore | **ALIGNED** |
```

Add a new row for camera persistence:
```
| Camera session restore | SM property serialize | `saveCameraToJson()` / `loadCameraFromJson()` in layout state | **ALIGNED** |
```

- [ ] **Step 2: Update §4 remaining gaps**

Add a note about the fixed gaps:
- Layout camera persistence: RESOLVED
- `restoreLayoutState` leak: RESOLVED
- `zoomOnSelectedEntities` view mismatch: RESOLVED
- Plugin association hardening: RESOLVED

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/multi-window-paraview-alignment-design.md
git commit -m "docs: update alignment matrix with fixed layout/zoom/association gaps"
```

---

### Task 7: Update `multi-window-views.md` Session Persistence Section ✅

**Files:**
- Modify: `docs/user-guide/multi-window-views.md`

- [ ] **Step 1: Add camera persistence documentation**

Search for the session persistence section and add documentation about the new camera save/restore in layout JSON.

- [ ] **Step 2: Commit**

```bash
git add docs/user-guide/multi-window-views.md
git commit -m "docs: document per-view camera session persistence"
```

---

## Deferred Items (Not In This Plan)

| Item | Reason | Priority |
|------|--------|----------|
| GAP-T: Source Undo | Very high complexity (4-6 weeks); separate plan exists | LOW |
| Phase P: View Type Registry | Only needed for SpreadSheet/Chart views | LOW |
| DB Tree per-view eye icon | Current global+per-view-in-properties approach is acceptable | LOW |
| DB selection multi-view focus | First-of-selection wins is acceptable UX; ParaView does similar | LOW |
| Wheel-to-activate view | ParaView also doesn't activate on wheel; consistent behavior | LOW |

---

## Self-Review Checklist

1. **Spec coverage**: All 5 HIGH/MEDIUM gaps from the audit are covered (Tasks 1-5). Documentation updates in Tasks 6-7.
2. **Placeholder scan**: No TBD/TODO placeholders found.
3. **Type consistency**: `CameraParams::toJson/fromJson` matches existing `CameraParams` struct. `forceAssociateToView` uses same types as `associateToActiveView`.
