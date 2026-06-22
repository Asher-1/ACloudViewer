# Multi-Window Three Bugs Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three multi-window rendering system bugs: click-only activation, default-window deletion crash, and new-window config inheritance.

**Architecture:** Harden the existing `ecvViewManager` / `ecvDisplayTools` / `VtkDisplayTools` / `MainWindow` lifecycle and config-copy paths. No new classes needed — this is a bug-fix plan, not a refactoring plan.

**Tech Stack:** C++ / Qt 5 / VTK / CMake

---

## Assessment Summary

| Bug | Status | Remaining Work |
|-----|--------|---------------|
| **1. Click-only activation** | **Already implemented** | Minor: MDI tab `subWindowActivated` also activates (same as ParaView — acceptable). Canvas hover/focus do NOT activate. DB tree selection does NOT activate. |
| **2. Default window deletion crash** | **Partially protected** | `adoptNewPrimary` does not update `m_mainScreen` (dangling pointer). `SetCurrentScreen(nullptr)` crashes. Edge cases in `prepareViewClose`. |
| **3. New window config inheritance** | **Partially implemented** | `copyPrimaryViewConfig` always clones `m_primaryCtx`, not the active view's effective context. If user configured a secondary view, new windows don't inherit those settings. |

---

### Task 1: Harden `SetCurrentScreen` null safety

**Files:**
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp:4472-4475`

- [ ] **Step 1: Add null guard to `SetCurrentScreen`**

The current code crashes immediately if `widget` is null because it calls `widget->update()` unconditionally.

```cpp
void ecvDisplayTools::SetCurrentScreen(QWidget* widget) {
    s_tools.instance->m_currentScreen = widget;
    if (widget) {
        widget->update();
    }
}
```

- [ ] **Step 2: Verify compile**

Run: `cmake --build build --target CV_db_LIB 2>&1 | tail -20`
Expected: Build succeeds (only one line changed, no API change).

- [ ] **Step 3: Commit**

```bash
git add libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "fix: guard SetCurrentScreen against nullptr widget"
```

---

### Task 2: Update `m_mainScreen` in `adoptNewPrimary`

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.cpp:141-167`

After `adoptNewPrimary` runs, `GetMainScreen()` still returns the old (destroyed) widget pointer — a use-after-free. The fix is to call `SetMainScreen(widget)` inside `adoptNewPrimary`.

- [ ] **Step 1: Add `SetMainScreen` call in `adoptNewPrimary`**

```cpp
void VtkDisplayTools::adoptNewPrimary(VtkVisPtr vis,
                                     QVTKWidgetCustom* widget) {
    if (!vis || !widget) return;

    if (m_visualizer3D && m_visualizer3D != vis) {
        disconnect(m_visualizer3D.get(),
                   &ecvGenericVisualizer3D::interactorPointPickedEvent,
                   this, &ecvDisplayTools::onPointPicking);
    }
    connect(vis.get(),
            &ecvGenericVisualizer3D::interactorPointPickedEvent,
            this, &ecvDisplayTools::onPointPicking,
            Qt::UniqueConnection);

    m_visualizer3D = vis;
    m_vtkWidget = widget;
    SetMainScreen(widget);
    SetCurrentScreen(widget);

    m_primaryVis = nullptr;
    m_primaryWidget = nullptr;

    if (widget->localHotZone()) {
        m_hotZone = widget->localHotZone();
        m_primaryCtx.clickableItemsVisible =
                widget->localClickableItemsVisible();
    }
}
```

The only change is adding `SetMainScreen(widget);` before `SetCurrentScreen(widget);`.

- [ ] **Step 2: Verify compile**

Run: `cmake --build build --target VtkEngine 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add libs/VtkEngine/Visualization/VtkDisplayTools.cpp
git commit -m "fix: update m_mainScreen in adoptNewPrimary to prevent dangling pointer"
```

---

### Task 3: Harden `prepareViewClose` edge cases

**Files:**
- Modify: `app/MainWindow.cpp:2545-2605`

Two edge cases need handling:
1. When `closingPrimary && !newPrimary` (no surviving `ecvGLView` with a pipeline) — should block close or gracefully degrade.
2. When the last view is being closed — the tab-close handler already blocks this (`subs.size() <= 1`), but the in-frame Close button also needs protection.

- [ ] **Step 1: Add fail-safe when no new primary is found**

```cpp
void MainWindow::prepareViewClose(QWidget* viewFrame) {
    if (!viewFrame) return;

    auto findViewInFrame = [](QWidget* root) -> ecvGenericGLDisplay* {
        if (!root) return nullptr;
        auto* d = ecvGenericGLDisplay::FromWidget(root);
        if (d) return d;
        for (auto* child : root->findChildren<QWidget*>()) {
            d = ecvGenericGLDisplay::FromWidget(child);
            if (d) return d;
        }
        return nullptr;
    };

    auto* closingDisplay = findViewInFrame(viewFrame);
    if (!closingDisplay) return;

    auto* glView = dynamic_cast<ecvGLView*>(closingDisplay);

    ecvViewManager::instance().unregisterView(closingDisplay);

    auto* primaryDT = static_cast<Visualization::VtkDisplayTools*>(
            ecvDisplayTools::TheInstance());
    QWidget* primaryScreen = ecvDisplayTools::GetCurrentScreen();
    bool closingPrimary = false;

    if (glView && glView->getVtkWidget() == primaryScreen) {
        closingPrimary = true;
    } else if (closingDisplay == ecvDisplayTools::TheInstance()) {
        closingPrimary = true;
    }

    if (closingPrimary && primaryDT) {
        const auto& views = ecvViewManager::instance().getAllViews();
        ecvGLView* newPrimary = nullptr;
        for (auto* v : views) {
            auto* gv = dynamic_cast<ecvGLView*>(v);
            if (gv && gv != glView && gv->getVisualizer3D() &&
                gv->getVtkWidget()) {
                newPrimary = gv;
                break;
            }
        }
        if (newPrimary) {
            primaryDT->adoptNewPrimary(newPrimary->getVisualizer3DSP(),
                                       newPrimary->getVtkWidget());
            rebindToolsToActiveView(newPrimary);
        } else {
            CVLog::Warning("[prepareViewClose] No surviving ecvGLView to "
                           "adopt as primary — singleton pipeline may be "
                           "invalid until a new view is created.");
            primaryDT->SetCurrentScreen(nullptr);
        }
    }

    if (glView && glView->getVisualizer3D()) {
        Visualization::VtkCameraLink::instance().removeView(
                glView->getVisualizer3D());
    }
}
```

The key change: when `closingPrimary && !newPrimary`, log a warning and safely null out the current screen (which is now safe after Task 1's null guard).

- [ ] **Step 2: Verify compile**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix: handle edge case in prepareViewClose when no new primary is available"
```

---

### Task 4: Fix `copyPrimaryViewConfig` to use effective (active) context

**Files:**
- Modify: `app/MainWindow.cpp:2411-2455`

Currently `copyPrimaryViewConfig` always clones `m_primaryCtx`. If the user changed settings (e.g., disabled auto-pick rotation center) while a secondary `ecvGLView` was active, those changes live in that view's `m_ctx`, not in `m_primaryCtx`. New windows would then get stale settings.

The fix: use `effectiveCtx()` which returns the active view's context.

- [ ] **Step 1: Change config source from `viewContext()` to `effectiveCtx()`**

```cpp
void MainWindow::copyPrimaryViewConfig(ecvGLView* view) {
    if (!view) return;

    auto* primaryInstance = ecvDisplayTools::TheInstance();
    auto* primaryDT =
            static_cast<Visualization::VtkDisplayTools*>(primaryInstance);
    if (!primaryDT) return;

    auto* primaryVis = primaryDT->get3DViewer();
    auto* newVis = view->getVisualizer3D();
    auto* newWidget = view->getVtkWidget();
    if (!primaryVis || !newVis || !newWidget) return;

    // Clone the effective (active) view's context — this is the view the
    // user is currently looking at.  If a secondary ecvGLView is active,
    // its m_ctx is returned; otherwise m_primaryCtx is the fallback.
    // This ensures new windows match the user's current working state
    // (CC pattern: inherit from the view you're splitting/cloning).
    const ecvViewContext& srcCtx = primaryInstance->effectiveCtx();
    view->context() = srcCtx;

    // Reset view-specific transient state that should NOT be inherited
    view->context().mouseMoved = false;
    view->context().mouseButtonPressed = false;
    view->context().touchInProgress = false;
    view->context().ignoreMouseReleaseEvent = false;
    view->context().widgetClicked = false;
    view->context().lastClickTime_ticks = 0;

    // Background color: copy from current draw context + set on renderer
    CC_DRAW_CONTEXT ctx;
    ecvDisplayTools::GetContext(ctx);
    newWidget->setBackgroundColor(ecvTools::TransFormRGB(ctx.backgroundCol),
                                  ecvTools::TransFormRGB(ctx.backgroundCol2),
                                  ctx.drawBackgroundGradient);

    ecvColor::Rgbf bkg1 = ecvTools::TransFormRGB(ctx.backgroundCol);
    ecvColor::Rgbf bkg2 = ecvTools::TransFormRGB(ctx.backgroundCol2);
    newVis->setBackgroundColor(bkg1.r, bkg1.g, bkg1.b, bkg2.r, bkg2.g, bkg2.b,
                               ctx.drawBackgroundGradient);

    // Camera orientation widget (ParaView-style gizmo)
    if (primaryVis->IsCameraOrientationWidgetShown()) {
        newVis->ToggleCameraOrientationWidget(true);
    }

    // Orientation marker axes (corner trihedron)
    if (primaryVis->pclMarkerAxesShown()) {
        newVis->showPclMarkerAxes(newVis->getRenderWindowInteractor());
    }

    view->redraw();
}
```

Key changes:
1. `primaryInstance->viewContext()` → `primaryInstance->effectiveCtx()` — reads from active view, not always primary.
2. Added transient state reset — mouse/touch state should not carry over to a new view.

- [ ] **Step 2: Verify compile**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix: new windows inherit config from active view, not always primary context"
```

---

### Task 5: Audit `GetMainScreen()` call sites

**Files:**
- Audit: `libs/CV_db/src/LineSet.cpp:35`
- Audit: `app/ecvPointPairRegistrationDlg.cpp:145`
- Audit: `app/reconstruction/ReconstructionWidget.cpp:24`
- Audit: `app/MainWindow.cpp:717`

`GetMainScreen()` returns `m_mainScreen` which, after Task 2, will always track the current primary widget. But some call sites use it as "any valid widget" (truthiness check) while others use it as a parent widget. Verify these are safe.

- [ ] **Step 1: Audit `LineSet.cpp` usage**

```35:35:libs/CV_db/src/LineSet.cpp``` uses `GetMainScreen()` as a guard: `if (MACRO_Draw3D(context) && ecvDisplayTools::GetMainScreen())`. After Task 2, this is always valid when a view exists. **No change needed.**

- [ ] **Step 2: Audit `ecvPointPairRegistrationDlg.cpp` usage**

```145:145:app/ecvPointPairRegistrationDlg.cpp``` uses it as a truthiness check for `ecvRedrawScope`. **No change needed.**

- [ ] **Step 3: Audit `ReconstructionWidget.cpp` usage**

```24:24:app/reconstruction/ReconstructionWidget.cpp``` uses `GetMainScreen()` as a **parent widget** for `QWidget` construction. If the primary window is closed and re-homed, this widget's parent might be stale if constructed before the re-home. However, this is a startup-only construction. **No change needed for this plan.**

- [ ] **Step 4: Audit `MainWindow.cpp:717` usage**

Uses `GetMainScreen()` to get the initial widget during `initial()`. This is always the first widget created. **No change needed.**

- [ ] **Step 5: Commit (audit-only, no code changes)**

No code changes from this task — just verification that existing call sites are safe after Tasks 1-2.

---

### Task 6: Verify click-only activation behavior

This task confirms that Bug 1 (click-only activation) is already correctly implemented.

- [ ] **Step 1: Verify `QVTKWidgetCustom::mousePressEvent` calls `setActiveView`**

```680:691:libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp``` shows click-to-activate with explicit ParaView-style comment. **Already correct.**

- [ ] **Step 2: Verify `mouseMoveEvent` does NOT activate**

```843:847:libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp``` has explicit comment: "Do NOT call setActiveView here". **Already correct.**

- [ ] **Step 3: Verify `FocusIn` does NOT activate**

```4847:4854:app/MainWindow.cpp``` has explicit `case QEvent::FocusIn: break;` with comment. **Already correct.**

- [ ] **Step 4: Verify no `enterEvent` override on `QVTKWidgetCustom`**

No `enterEvent` override exists in `QVTKWidgetCustom.h`. **Already correct.**

- [ ] **Step 5: Document in roadmap**

Bug 1 is already fixed. Activation paths:
- Canvas mouse click → `setActiveView` ✓
- MDI tab click → `subWindowActivated` → `setActiveView` ✓ (same as ParaView)
- Hover/focus/DB-tree → **no activation** ✓

No code changes needed.

---

## Self-Review Checklist

1. **Spec coverage:** All 3 bugs addressed. Bug 1 verified as already fixed. Bug 2 fixed by Tasks 1-3. Bug 3 fixed by Task 4. Task 5 is defensive audit. Task 6 is verification.

2. **Placeholder scan:** No TBDs, TODOs, or "fill in later" entries. All code blocks contain complete implementations.

3. **Type consistency:** `effectiveCtx()` returns `ecvViewContext&` (confirmed in `ecvDisplayTools.cpp:81-90`). `SetMainScreen` takes `QWidget*` (confirmed in `ecvDisplayTools.h`). All types match.
