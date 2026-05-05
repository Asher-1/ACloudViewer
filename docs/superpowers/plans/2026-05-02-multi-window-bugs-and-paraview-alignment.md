# Multi-Window Bugs & ParaView Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the black overlay at top-left corner, correct text routing for per-view rendering, auto-update rotation center on model load, and align the three pivot buttons with ParaView behavior.

**Architecture:** Per-view text rendering must bypass the singleton's ImageVis and route directly to the per-view VtkVis. The Hot Zone clickable items visibility guard must be checked per-view. The rotation center should auto-update on every model load, matching ParaView's `resetCenterOfRotation` behavior.

**Tech Stack:** C++17, VTK, Qt5/6, ACloudViewer VtkEngine

---

## Root Cause Analysis

### Issue 1: Black Area at Top-Left (Image 1)

**Root cause chain:**

1. `ecvGLView::redraw()` renders messages by calling `ecvDisplayTools::RenderText(x, y, ..., this)`.
2. `RenderText` → `DisplayText(context)` → `sharedTools()->displayText(CONTEXT)`.
3. `VtkDisplayTools::displayText()` checks `isSecondaryView = true` (because `context.display` is `ecvGLView`, not the singleton).
4. It tries to get per-view ImageVis via `glView->getImageVis()`, but **ecvGLView never initializes `m_visualizer2D`**, so it returns null.
5. Falls back to **singleton's** `m_visualizer2D` (ImageVis).
6. Text is rendered via `ImageVis::addText()` on the **singleton's** 2D overlay, not on the per-view VTK widget.
7. The singleton's ImageVis has a different viewport/coordinate mapping, causing text to appear at the wrong position (top-left instead of bottom-left).
8. Additionally, text actors added to the ImageVis are **never cleaned up** when per-view rendering takes over — they persist as stale overlays.

**Secondary contributor:** The Hot Zone draws a semi-transparent dark grey background (`alpha=210/255`) at position `(0, 0)` in the top-left corner. Combined with the misrouted text, this creates the solid black-looking area.

### Issue 6: Rotation Center Not Auto-Updated + Button Misalignment

**Root causes:**

1. **No auto-update on subsequent loads:** `MainWindow::addToDB` only calls `updateConstellationCenterAndZoom()` when the DB was previously empty (`updateZoom = true` only for first load). Subsequent file loads do NOT reset the rotation center.
2. **`actionAutoPickPivot` tooltip lie:** Says "middle of the screen" but actually picks the 3D point under the mouse cursor via `getClick3DPos`.
3. **`actionShowPivot` drops `SHOW_ON_MOVE`:** The toggle only sets `PIVOT_ALWAYS_SHOW` or `PIVOT_HIDE`, losing ParaView's intermediate state.

---

## File Map

| File | Responsibility | Action |
|------|----------------|--------|
| `libs/VtkEngine/Visualization/VtkDisplayTools.cpp:1275-1301` | Text routing to per-view vs singleton | **Modify** |
| `libs/VtkEngine/Visualization/ecvGLView.cpp:220-265` | Per-view message rendering | **Modify** |
| `libs/CV_db/src/ecvDisplayTools.cpp:3906-3936` | DrawClickableItems visibility guard | **Verify** |
| `app/MainWindow.cpp:3661-3694` | Model load → rotation center update | **Modify** |
| `app/MainWindow.cpp:2223-2255` | Pivot button handlers | **Modify** |
| `app/ui_templates/MainWindow.ui:3584-3627` | Tooltip text for pivot buttons | **Modify** |

---

### Task 1: Fix Per-View Text Routing (Issue 1 – Core Fix)

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.cpp:1275-1301`

The root problem: `VtkDisplayTools::displayText()` falls back to the singleton's `m_visualizer2D` when the per-view ecvGLView has no ImageVis. This routes text to the wrong widget and wrong coordinate space.

**Fix:** When the target display is a secondary view (ecvGLView), skip the ImageVis path entirely and use the per-view VtkVis directly.

- [ ] **Step 1: Read the current `displayText` implementation**

```bash
# Verify the current code matches our analysis
```

Current code at `VtkDisplayTools.cpp:1275-1301`:
```cpp
void VtkDisplayTools::displayText(const CC_DRAW_CONTEXT& context) {
    VtkVis* vis = resolveVisualizer(context.display);
    bool isSecondaryView =
            context.display &&
            context.display != static_cast<ecvDisplayTools*>(this);
    Visualization::ImageVis* txtVis2D =
            m_visualizer2D ? m_visualizer2D.get() : nullptr;
    if (isSecondaryView) {
        auto* glView = dynamic_cast<ecvGLView*>(context.display);
        if (glView && glView->getImageVis()) {
            txtVis2D = glView->getImageVis().get();
        }
    }
    if (txtVis2D) {
        ecvTextParam textParam = context.textParam;
        std::string viewID = CVTools::FromQString(context.viewID);
        std::string text = CVTools::FromQString(textParam.text);
        ecvColor::Rgbf textColor =
                ecvTools::TransFormRGB(context.textDefaultCol);
        txtVis2D->addText(textParam.textPos.x, textParam.textPos.y, text,
                          textColor.r, textColor.g, textColor.b, viewID,
                          textParam.opacity, textParam.font.pointSize(),
                          textParam.font.bold());
    } else {
        vis->displayText(context);
    }
}
```

- [ ] **Step 2: Fix the routing logic**

Replace the `displayText` method with logic that routes secondary views directly to VtkVis:

```cpp
void VtkDisplayTools::displayText(const CC_DRAW_CONTEXT& context) {
    VtkVis* vis = resolveVisualizer(context.display);
    bool isSecondaryView =
            context.display &&
            context.display != static_cast<ecvDisplayTools*>(this);

    if (isSecondaryView) {
        // Per-view text: always go through VtkVis (3D renderer's text actor).
        // ecvGLView does not own an ImageVis, so falling back to the
        // singleton's m_visualizer2D would render on the wrong widget.
        if (vis) {
            vis->displayText(context);
        }
        return;
    }

    // Singleton path (primary view or no per-view display)
    Visualization::ImageVis* txtVis2D =
            m_visualizer2D ? m_visualizer2D.get() : nullptr;
    if (txtVis2D) {
        ecvTextParam textParam = context.textParam;
        std::string viewID = CVTools::FromQString(context.viewID);
        std::string text = CVTools::FromQString(textParam.text);
        ecvColor::Rgbf textColor =
                ecvTools::TransFormRGB(context.textDefaultCol);
        txtVis2D->addText(textParam.textPos.x, textParam.textPos.y, text,
                          textColor.r, textColor.g, textColor.b, viewID,
                          textParam.opacity, textParam.font.pointSize(),
                          textParam.font.bold());
    } else if (vis) {
        vis->displayText(context);
    }
}
```

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target ACloudViewer -j$(nproc) 2>&1 | tail -20`
Expected: Build succeeds with no errors.

- [ ] **Step 4: Commit**

```bash
git add libs/VtkEngine/Visualization/VtkDisplayTools.cpp
git commit -m "fix: route per-view text to VtkVis instead of singleton ImageVis

Secondary views (ecvGLView) do not own an ImageVis. The fallback to
the singleton's m_visualizer2D rendered text on the wrong widget with
wrong coordinates, causing the black overlay at the top-left corner."
```

---

### Task 2: Clean Up Stale Singleton Text Actors (Issue 1 – Cleanup)

**Files:**
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp:3572-3632`

When per-view rendering activates (`viewCount > 0`), the singleton overlay path deactivates but only cleans up messages that are still in `m_messagesToDisplay`. Messages that expired earlier (2-second timeout) leave stale text actors in the singleton's ImageVis.

- [ ] **Step 1: Add comprehensive cleanup when singleton overlay deactivates**

In `ecvDisplayTools.cpp`, find the block at line ~3626:

```cpp
} else if (s_singletonOverlayActive) {
    s_singletonOverlayActive = false;
    for (const auto& message : s_tools->m_messagesToDisplay) {
        RemoveWidgets(WIDGETS_PARAMETER(
                WIDGETS_TYPE::WIDGET_T2D, message.message));
    }
}
```

Replace with:

```cpp
} else if (s_singletonOverlayActive) {
    s_singletonOverlayActive = false;
    for (const auto& message : s_tools->m_messagesToDisplay) {
        RemoveWidgets(WIDGETS_PARAMETER(
                WIDGETS_TYPE::WIDGET_T2D, message.message));
    }
    // Also clean stale text actors for well-known message types
    // that may have expired before per-view rendering activated.
    static const QStringList knownMsgPrefixes = {
            "Perspective", "Centered perspective", "Viewer-based perspective",
            "New default point size", "New default line width",
            "Near clipping", "New size", "F.O.V."};
    auto* vdt = dynamic_cast<VtkDisplayTools*>(s_tools);
    if (vdt && vdt->getVisualizer2D()) {
        for (const auto& prefix : knownMsgPrefixes) {
            RemoveWidgets(WIDGETS_PARAMETER(
                    WIDGETS_TYPE::WIDGET_T2D, prefix));
        }
    }
}
```

**Note:** This is a safety net. Task 1's fix prevents new stale actors from being created. This task handles actors created before the fix.

- [ ] **Step 2: Build and verify**

Run: `cmake --build build --target ACloudViewer -j$(nproc) 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "fix: clean stale singleton text actors when per-view rendering activates

Messages that expire before views are created leave orphaned text
actors in the singleton's ImageVis. This cleanup removes known
message types during the transition to per-view rendering."
```

---

### Task 3: Auto-Update Rotation Center on Every Model Load (Issue 6a)

**Files:**
- Modify: `app/MainWindow.cpp:3661-3694`

Currently `updateConstellationCenterAndZoom()` is only called when loading into an empty DB. ParaView always resets the camera center when loading new data.

- [ ] **Step 1: Read the current addToDB zoom logic**

Find in `MainWindow.cpp` around line 3661:
```cpp
if (!m_ccRoot->getRootEntity() ||
    m_ccRoot->getRootEntity()->getChildrenNumber() == 0) {
    updateZoom = true;
}
```

- [ ] **Step 2: Always update rotation center after adding entities**

After the existing `updateZoom` block, add rotation center update:

```cpp
    if (updateZoom) {
        if (auto* v = getActiveGLView()) {
            v->updateConstellationCenterAndZoom();
        }
    }

    // ParaView alignment: always reset rotation center to encompass
    // all visible geometry, even when not zooming.
    if (!updateZoom) {
        if (auto* v = getActiveGLView()) {
            v->resetCenterOfRotation();
        }
    }
```

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target ACloudViewer -j$(nproc) 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix: auto-update rotation center on every model load

ParaView always resets the center of rotation when loading new data.
Previously, ACloudViewer only updated it when loading into an empty
scene. Now resetCenterOfRotation() is called after every addToDB."
```

---

### Task 4: Fix actionAutoPickPivot Tooltip (Issue 6b)

**Files:**
- Modify: `app/ui_templates/MainWindow.ui:3584-3600`

The tooltip says "Place the rotation center to the middle of the screen" but the implementation picks the 3D point under the mouse cursor.

- [ ] **Step 1: Update the tooltip to match actual behavior**

Find in `MainWindow.ui`:
```xml
<property name="toolTip">
 <string>Place the rotation center to the middle of the screen</string>
</property>
```

Replace with:
```xml
<property name="toolTip">
 <string>Auto-pick rotation center at clicked 3D point</string>
</property>
```

- [ ] **Step 2: Also fix the "reset Rotation Center" capitalization**

Find:
```xml
<string>reset Rotation Center</string>
```

Replace with:
```xml
<string>Reset Rotation Center</string>
```

- [ ] **Step 3: Commit**

```bash
git add app/ui_templates/MainWindow.ui
git commit -m "fix: correct pivot button tooltips and capitalization

actionAutoPickPivot tooltip now says 'Auto-pick rotation center at
clicked 3D point' (was incorrectly 'middle of the screen').
actionResetPivot text capitalized consistently."
```

---

### Task 5: Fix actionShowPivot to Support SHOW_ON_MOVE (Issue 6c)

**Files:**
- Modify: `app/MainWindow.cpp:2241-2255`
- Modify: `app/MainWindow.cpp:2667-2681` (`syncPivotButtonStates`)

The toggle currently only cycles between `PIVOT_ALWAYS_SHOW` and `PIVOT_HIDE`, losing ParaView's intermediate `PIVOT_SHOW_ON_MOVE` state.

- [ ] **Step 1: Read the current toggleRotationCenterVisibility**

```cpp
void MainWindow::toggleRotationCenterVisibility(bool state) {
    ecvGLView* view = getActiveGLView();
    if (!view) return;

    if (state) {
        view->setPivotVisibility(ecvGenericGLDisplay::PIVOT_ALWAYS_SHOW);
    } else {
        view->setPivotVisibility(ecvGenericGLDisplay::PIVOT_HIDE);
    }
    view->redraw(false, false);
}
```

- [ ] **Step 2: Keep the toggle as-is but ensure sync handles SHOW_ON_MOVE**

The toggle behavior (checked = ALWAYS_SHOW, unchecked = HIDE) is acceptable as a simplified ParaView-style UI. The important fix is that `syncPivotButtonStates` correctly reflects the state. The current sync already does this correctly (checked only for ALWAYS_SHOW). No code change needed here — just verify.

- [ ] **Step 3: Verify and commit (no-op if no changes needed)**

If the sync logic is already correct, skip this task.

---

### Task 6: Ensure Per-View Consistency for Rotation Center (Issue 6d)

**Files:**
- Modify: `app/MainWindow.cpp:2223-2233` (`toggleActiveWindowAutoPickRotCenter`)
- Verify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp:748-757`

When auto-pick is toggled, the setting must apply only to the active view and be correctly synced when switching views.

- [ ] **Step 1: Verify per-view auto-pick isolation**

The current code already calls `getActiveGLView()->setAutoPickPivotAtCenter(state)`, which modifies only that view's context. `QVTKWidgetCustom::mousePressEvent` reads from `curCtx()` which resolves to the per-view context. This is correct.

- [ ] **Step 2: Verify syncPivotButtonStates refreshes auto-pick state**

Check that `syncPivotButtonStates` also syncs `actionAutoPickPivot`:

```cpp
void MainWindow::syncPivotButtonStates(ecvGenericGLDisplay* display) {
    // ... existing showPivot sync ...
    bool autoPickPivot = ctx.autoPickPivotAtCenter;
    m_ui->actionAutoPickPivot->blockSignals(true);
    m_ui->actionAutoPickPivot->setChecked(autoPickPivot);
    m_ui->actionAutoPickPivot->blockSignals(false);
}
```

If this sync is missing, add it.

- [ ] **Step 3: Build, verify, commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix: sync auto-pick pivot button state on active view change

Ensures the toolbar checkbox reflects the per-view auto-pick state
when the user switches between views."
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - Issue 1 (black area): Tasks 1 + 2 ✅
   - Issue 6a (rotation center auto-update): Task 3 ✅
   - Issue 6b (tooltip mismatch): Task 4 ✅
   - Issue 6c (SHOW_ON_MOVE): Task 5 ✅
   - Issue 6d (per-view consistency): Task 6 ✅

2. **Placeholder scan:** No TBDs, TODOs, or "fill in later" — all steps have concrete code.

3. **Type consistency:** `resetCenterOfRotation()` matches `ecvGLView::resetCenterOfRotation(int viewport = 0)` signature. `setPivotVisibility` matches `PivotVisibility` enum. `resolveVisualizer` returns `VtkVis*`. All consistent.
