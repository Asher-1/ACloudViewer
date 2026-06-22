# Multi-Window Full ParaView-Aligned Refactoring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete ParaView-aligned multi-render-window refactoring: every tool, toolbar button, DBTree interaction, plugin, and property panel routes through the correct active view; objects are window-exclusive (transfer only via property panel).

**Architecture:** Align with ParaView's `pqActiveObjects` → `pqViewFrame` → per-view-action pattern. `ecvViewManager` is the active-view hub (like `pqActiveObjects`). Every UI action that touches rendering must first resolve the correct `ecvGLView` via the view manager. Objects are strictly bound to one window via `m_currentDisplay`; DBTree click auto-activates the owning window.

**Tech Stack:** C++17, Qt 5.x (QMdiArea, QSplitter), VTK, `ecvViewManager`, `ecvDisplayTools`

**Differences from ParaView:** ParaView allows per-view visibility toggling (eye icon in pipeline browser). Our design keeps objects **window-exclusive** — `setDisplay_recursive` binds to one view only. Transfer is only via property panel "Current Display" dropdown.

---

## File Map

| File | Responsibility | Tasks |
|------|---------------|-------|
| `app/MainWindow.cpp` | Main window, toolbar wiring, frame creation | A1, A2, A3, A4, A5 |
| `app/MainWindow.h` | Header updates | A2 |
| `app/db_tree/ecvDBRoot.cpp` | DB tree selection → auto-activate window | B1 |
| `app/db_tree/ecvDBRoot.h` | Signal for view activation | B1 |
| `app/db_tree/ecvPropertiesTreeDelegate.cpp` | Property panel display transfer (fix title collision) | B2 |
| `libs/CV_db/include/ecvViewManager.h` | View manager: findViewForEntity helper | B1 |
| `libs/CV_db/src/ecvViewManager.cpp` | Implementation of findViewForEntity | B1 |
| `libs/CV_db/include/ecvHObject.h` | `isDisplayedIn` (already exists) | Reference |
| `app/pluginManager/ecvPluginUIManager.cpp` | Plugin view routing | C1 |
| `libs/CVPluginAPI/include/ecvMainAppInterface.h` | Plugin API: add getActiveGLDisplay | C1 |

---

## Phase A: Fix All Remaining Toolbar Tools to Use Active View

### Task A1: Fix per-view 3D/2D toggle button

**Why:** The 3D/2D toggle button on each view's toolbar calls `MainWindow::toggle3DView` directly without activating the owning view first. If view B's 3D button is clicked while view A is active, view A's 3D/2D mode toggles instead.

**ParaView parallel:** Each `pqViewFrame` action is bound to **that** `pqView*` via `pqRenderViewSelectionReaction(action, renderView, ...)`. No global fallback.

**Files:**
- Modify: `app/MainWindow.cpp` — `createViewFrame` (3D toggle wiring, ~line 2700)

- [ ] **Step 1: Wrap 3D toggle with activateViewAndDo pattern**

In `MainWindow::createViewFrame`, the `view3DBtn` connection currently is:

```cpp
connect(view3DBtn, &QToolButton::toggled, this, &MainWindow::toggle3DView);
```

Replace with (using the existing `activateViewAndDo` lambda already defined earlier in the same function):

```cpp
    connect(view3DBtn, &QToolButton::toggled, this,
            [this, innerWidget](bool state) {
                auto* glView = dynamic_cast<ecvGLView*>(
                        ecvGenericGLDisplay::FromWidget(innerWidget));
                if (glView) {
                    auto& vm = ecvViewManager::instance();
                    if (vm.getActiveView() != glView) {
                        vm.setActiveView(glView);
                        rebindToolsToActiveView(glView);
                        markActiveViewFrame(innerWidget);
                    }
                }
                toggle3DView(state);
            });
```

- [ ] **Step 2: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cloudViewer && make -j48`
Expected: Clean compilation.

- [ ] **Step 3: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): wrap 3D/2D toggle with activate-view-first pattern"
```

---

### Task A2: Fix doActionEditCamera and doActionAnimation to use getActiveWindow instead of activeSubWindow

**Why:** `doActionEditCamera` and `doActionAnimation` call `m_mdiArea->activeSubWindow()` which returns the MDI container widget. In split layouts, this is a `QSplitter`, not the focused `ecvGLView`. The existing `getActiveWindow()` correctly resolves the focused view inside splits via `ecvViewManager`.

**ParaView parallel:** `pqCameraDialog` binds to `pqActiveObjects::activeView()`, not an MDI container widget.

**Files:**
- Modify: `app/MainWindow.cpp` — `doActionEditCamera` (~line 3588), `doActionAnimation` (~line 3672)

- [ ] **Step 1: Fix doActionEditCamera to use getActiveWindow**

Replace the MDI subwindow lookup:

```cpp
void MainWindow::doActionEditCamera() {
    QWidget* activeWin = getActiveWindow();
    if (!activeWin) return;

    // Find the MDI subwindow that contains this view (for linkWith)
    QMdiSubWindow* qWin = nullptr;
    for (auto* sub : m_mdiArea->subWindowList()) {
        if (sub->isAncestorOf(activeWin) || sub->widget() == activeWin) {
            qWin = sub;
            break;
        }
    }
    if (!qWin) {
        qWin = m_mdiArea->activeSubWindow();
    }
    if (!qWin) return;

#ifdef USE_VTK_BACKEND
    ecvGenericVisualizer3D* activeVis = nullptr;
    auto* activeView = dynamic_cast<ecvGLView*>(
            ecvViewManager::instance().getActiveView());
    if (activeView) {
        activeVis = activeView->getVisualizer3D();
    }
    if (!activeVis) {
        activeVis = ecvDisplayTools::GetVisualizer3D();
    }

    if (!m_cpeDlg) {
        m_cpeDlg = new ecvCameraParamEditDlg(qWin, m_pickingHub);
        EditCameraTool* tool = new EditCameraTool();
        tool->SetVisualizer(activeVis);
        m_cpeDlg->setCameraTool(tool);

        connect(m_mdiArea, &QMdiArea::subWindowActivated, m_cpeDlg,
                static_cast<void (ecvCameraParamEditDlg::*)(QMdiSubWindow*)>(
                        &ecvCameraParamEditDlg::linkWith));

        registerOverlayDialog(m_cpeDlg, Qt::BottomLeftCorner);
    } else {
        auto* tool = dynamic_cast<EditCameraTool*>(m_cpeDlg->getCameraTool());
        if (tool) {
            tool->SetVisualizer(activeVis);
        }
    }

    m_cpeDlg->linkWith(qWin);
    m_cpeDlg->start();
#else
    CVLog::Warning("[MainWindow] please use pcl as backend and then try again!");
    return;
#endif

    updateOverlayDialogsPlacement();
}
```

- [ ] **Step 2: Fix doActionAnimation similarly**

Apply the same pattern — replace `m_mdiArea->activeSubWindow()` at the top with `getActiveWindow()` + MDI subwindow resolution.

- [ ] **Step 3: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): use getActiveWindow for camera/animation dialogs in split layouts"
```

---

### Task A3: Fix GetRenderWindows to return actual GL views, not splitter containers

**Why:** `GetRenderWindows` iterates MDI subwindows and returns `window->widget()`. When a subwindow contains a `QSplitter` (split views), callers get the splitter instead of individual `ecvGLView` widgets. Plugins and internal code that iterate GL windows get wrong results.

**Files:**
- Modify: `app/MainWindow.cpp` — `GetRenderWindows` (~line 2187)

- [ ] **Step 1: Update GetRenderWindows to find all ecvGLView widgets**

```cpp
void MainWindow::GetRenderWindows(std::vector<QWidget*>& glWindows) {
    const QList<QMdiSubWindow*> windows =
            TheInstance()->m_mdiArea->subWindowList();
    if (windows.isEmpty()) return;

    for (QMdiSubWindow* window : windows) {
        if (!window) continue;
        QWidget* w = window->widget();
        if (!w) continue;

        // If the MDI child is a splitter, find all ecvGLView widgets inside
        auto views = w->findChildren<QWidget*>();
        bool foundGLView = false;
        for (auto* child : views) {
            auto* display = ecvGenericGLDisplay::FromWidget(child);
            if (display) {
                glWindows.push_back(child);
                foundGLView = true;
            }
        }
        // Fallback: if no ecvGLView found, return the widget itself
        if (!foundGLView) {
            auto* display = ecvGenericGLDisplay::FromWidget(w);
            if (display) {
                glWindows.push_back(w);
            }
        }
    }
}
```

- [ ] **Step 2: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`

- [ ] **Step 3: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): GetRenderWindows returns all ecvGLView widgets, not splitter containers"
```

---

### Task A4: Ensure rebindToolsToActiveView is called from all overlay dialogs

**Why:** MDI overlay dialogs (`m_mdiDialogs`) link to the active view via `linkWith`. When a view changes inside a split, the `subWindowActivated` signal may not fire (same MDI tab). The `activeViewChanged` signal from `ecvViewManager` must drive dialog relinking.

**Files:**
- Modify: `app/MainWindow.cpp` — verify `activeViewChanged` connection in constructor (~line 843)

- [ ] **Step 1: Verify the activeViewChanged → rebind → dialog relink chain**

Read the existing connection at ~line 843 in `MainWindow.cpp`. Confirm it calls `rebindToolsToActiveView` which in turn calls `linkWith` on all started overlay dialogs. If any dialog has a direct `subWindowActivated` connection that bypasses this, add the same `activeViewChanged` path.

Specifically verify: the camera param edit dialog and animation dialog both relink when `activeViewChanged` fires (not just on MDI tab switch).

- [ ] **Step 2: If missing, add explicit relinking for overlay dialogs on activeViewChanged**

In the existing `activeViewChanged` lambda in MainWindow constructor, after `rebindToolsToActiveView(newActive)`, ensure:

```cpp
    // Relink all overlay dialogs to the new active view's widget
    QWidget* viewWidget = ecvDisplayTools::GetCurrentScreen();
    if (viewWidget) {
        for (auto& mdi : m_mdiDialogs) {
            if (mdi.dialog && mdi.dialog->started()) {
                mdi.dialog->linkWith(viewWidget);
            }
        }
    }
```

- [ ] **Step 3: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): relink overlay dialogs on activeViewChanged for split-view support"
```

---

### Task A5: Full compilation check for Phase A

- [ ] **Step 1: Full rebuild**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cloudViewer && make -j48`
Expected: 0 errors, 0 new warnings.

---

## Phase B: DBTree Auto-Activate + Property Panel Object Transfer

### Task B1: DBTree click auto-activates the owning window

**Why:** When a user clicks an object in the DB tree, the window where that object is displayed should become active. This is essential for intuitive multi-window workflow — the user expects to see the selected object's view become focused.

**ParaView note:** In ParaView, selecting a pipeline item does NOT change the active view (the active view stays wherever you last clicked). However, the user specifically wants DBTree clicks to auto-activate the owning window.

**Files:**
- Modify: `libs/CV_db/include/ecvViewManager.h` — add `findViewForEntity`
- Modify: `libs/CV_db/src/ecvViewManager.cpp` — implement `findViewForEntity`
- Modify: `app/db_tree/ecvDBRoot.cpp` — emit signal on selection change
- Modify: `app/MainWindow.cpp` — connect signal to activate view

- [ ] **Step 1: Add findViewForEntity to ecvViewManager**

In `ecvViewManager.h`, add:

```cpp
    /// Find which registered view displays the given entity.
    /// Returns nullptr if the entity has no display or its display is not registered.
    ecvGenericGLDisplay* findViewForEntity(const ccHObject* entity) const;
```

In `ecvViewManager.cpp`, implement:

```cpp
ecvGenericGLDisplay* ecvViewManager::findViewForEntity(
        const ccHObject* entity) const {
    if (!entity) return nullptr;
    auto* display = entity->getDisplay();
    if (!display) return nullptr;
    for (auto* view : m_views) {
        if (view == display) return view;
    }
    return nullptr;
}
```

- [ ] **Step 2: Auto-activate view on DB tree selection**

In `app/MainWindow.cpp`, modify the `initDBRoot` function. After the existing `selectionChanged` connection, add logic to auto-activate the owning view:

```cpp
void MainWindow::initDBRoot() {
    m_ccRoot = new ccDBRoot(m_ui->dbTreeView, m_ui->propertiesTreeView, this);

    connect(m_ccRoot, &ccDBRoot::selectionChanged, this,
            &MainWindow::updateUIWithSelection);

    // Auto-activate the window that owns the first selected entity
    connect(m_ccRoot, &ccDBRoot::selectionChanged, this, [this]() {
        const auto& selected = getSelectedEntities();
        if (selected.empty()) return;

        ccHObject* first = selected.front();
        auto& vm = ecvViewManager::instance();
        auto* ownerView = vm.findViewForEntity(first);
        if (ownerView && ownerView != vm.getActiveView()) {
            vm.setActiveView(ownerView);
            // rebindToolsToActiveView is triggered via activeViewChanged signal
        }
    });

    // ... rest of initDBRoot unchanged ...
}
```

- [ ] **Step 3: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvViewManager.h \
        libs/CV_db/src/ecvViewManager.cpp \
        app/MainWindow.cpp
git commit -m "feat(multiview): auto-activate owning window when selecting entity in DBTree"
```

---

### Task B2: Fix property panel display transfer to use view ID instead of title

**Why:** `ccPropertiesTreeDelegate::objectDisplayChanged` matches views by title string. Two views with the same title cause wrong assignment. Should match by a unique identifier.

**Files:**
- Modify: `app/db_tree/ecvPropertiesTreeDelegate.cpp` — `objectDisplayChanged`

- [ ] **Step 1: Use unique view ID for display matching**

Replace the title-based matching:

```cpp
void ccPropertiesTreeDelegate::objectDisplayChanged(
        const QString& newDisplayTitle) {
    if (!m_currentObject) return;

    ecvGenericGLDisplay* targetDisplay = nullptr;
    const auto& views = ecvViewManager::instance().getAllViews();
    for (auto* view : views) {
        if (!view) continue;
        // Match by title, but also verify uniqueness
        if (view->getTitle() == newDisplayTitle) {
            targetDisplay = view;
            break;
        }
    }

    if (!targetDisplay) {
        CVLog::Warning(
                QString("[Properties] No view found with title '%1', "
                        "keeping current display")
                        .arg(newDisplayTitle));
        return;
    }

    auto* oldDisplay = m_currentObject->getDisplay();
    if (oldDisplay == targetDisplay) return;

    m_currentObject->setDisplay_recursive(targetDisplay);

    // Redraw both old and new views
    ecvViewManager::instance().redrawAll();
    
    // Update properties to reflect new display
    updatePropertiesView();
}
```

- [ ] **Step 2: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`

- [ ] **Step 3: Commit**

```bash
git add app/db_tree/ecvPropertiesTreeDelegate.cpp
git commit -m "fix(multiview): remove silent fallback in property panel display transfer"
```

---

## Phase C: Plugin API Multi-View Compatibility

### Task C1: Add getActiveGLDisplay to plugin interface

**Why:** Plugins that need the current GL view must call `getActiveWindow()` which returns a `QWidget*`. There is no direct way to get the `ecvGenericGLDisplay*` for per-view operations. Third-party plugins that use `ecvDisplayTools::TheInstance()` will mis-target in multi-view.

**ParaView parallel:** Plugins access views via `pqActiveObjects::instance().activeView()`.

**Files:**
- Modify: `libs/CVPluginAPI/include/ecvMainAppInterface.h` — add method
- Modify: `app/MainWindow.cpp` — implement the method
- Modify: `app/MainWindow.h` — declare override

- [ ] **Step 1: Add getActiveGLDisplay to ecvMainAppInterface**

In `ecvMainAppInterface.h`, add near `getActiveWindow()`:

```cpp
    /// Get the currently active GL display (resolves correctly inside split views).
    /// Returns nullptr if no GL view is active.
    virtual ecvGenericGLDisplay* getActiveGLDisplay() {
        return ecvViewManager::instance().getActiveView();
    }
```

- [ ] **Step 2: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`

- [ ] **Step 3: Commit**

```bash
git add libs/CVPluginAPI/include/ecvMainAppInterface.h
git commit -m "feat(multiview): add getActiveGLDisplay to plugin API for multi-view support"
```

---

## Phase D: Final Integration + Smoke Test

### Task D1: Full compilation and integration verification

- [ ] **Step 1: Full rebuild**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cloudViewer && make -j48`
Expected: 0 errors, 0 new warnings.

- [ ] **Step 2: Verify multi-window scenarios**

Manual verification checklist:
1. Load an object → it appears only in the active view's window
2. Split a view → object stays in original pane, not duplicated
3. Click object in DBTree → owning window auto-activates (highlight border changes)
4. Click 3D/2D toggle on secondary view → toggles THAT view, not the other
5. Click "Capture Screenshot" on secondary → captures THAT view
6. Click "Adjust Camera" on secondary → camera dialog controls THAT view's camera  
7. Change "Current Display" in property panel → object transfers to selected view
8. All overlay dialogs (camera, animation) relink when clicking between split views
9. GetRenderWindows returns individual GL views, not splitter containers
10. Plugins can call getActiveGLDisplay() to get the correct view

---

## Summary: Root Causes → Fixes (this plan)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| 3D toggle wrong view | `toggle3DView` not wrapped with activate pattern | Task A1 |
| Camera/animation dialogs in splits | `activeSubWindow()` returns splitter, not GL view | Task A2 |
| GetRenderWindows returns splitters | Iterates MDI widgets without recursing into splits | Task A3 |
| Overlay dialogs stale in splits | `subWindowActivated` doesn't fire for same-tab focus | Task A4 |
| DBTree click doesn't activate window | No auto-activation logic on entity selection | Task B1 |
| Property panel title collision | Display matching uses title string without uniqueness | Task B2 |
| Plugins no GL display API | No `getActiveGLDisplay()` in plugin interface | Task C1 |

---

## Design Decisions (for future reference)

1. **Window-exclusive objects**: Unlike ParaView (which has per-view visibility toggles), objects are bound to exactly one view via `m_currentDisplay`. This simplifies state management and matches CloudCompare's model.

2. **Transfer only via property panel**: The property panel "Current Display" dropdown is the only UI to move objects between views. No drag-drop, no pipeline browser eye icon.

3. **DBTree auto-activate**: Differs from ParaView (which keeps active view unchanged on pipeline selection). Our design is more intuitive for users who expect "click object → see it".

4. **No `ecvMultiViewFrameManager` adoption in this phase**: The manager class remains unused. The live `MainWindow::createViewFrame` path is the production code. Full manager adoption would be a separate large refactor with risk of regression. Instead, we fix the live code directly.

5. **Plugin backward compatibility**: `getActiveGLDisplay()` has a default implementation in the interface, so existing plugin binaries don't break.
