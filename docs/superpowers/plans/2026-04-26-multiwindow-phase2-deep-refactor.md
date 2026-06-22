# Multi-Window Phase 2: Deep Module Compatibility Refactoring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all remaining multi-window issues discovered in the audit: object orphaning on view close, tab-close with splitters, properties panel singleton coupling, dialog relinking, `getActiveGLView` fallback inconsistency, and transform/filter tool view resolution.

**Architecture:** Continue aligning with ParaView's per-view-owns-its-state model. The pattern: every code path that touches rendering must resolve the target view via `ecvViewManager::instance().getActiveView()` (or the entity's `getDisplay()`) rather than `ecvDisplayTools::TheInstance()`. When a view closes, its objects are reassigned to a surviving view (not left unbound).

**Tech Stack:** C++17, Qt 5.x (QMdiArea, QSplitter), VTK, `ecvViewManager`, `ecvDisplayTools`

---

## File Map

| File | Responsibility | Tasks |
|------|---------------|-------|
| `libs/CV_db/src/ecvViewManager.cpp` | View lifecycle, entity reassignment | E1, E2 |
| `libs/CV_db/include/ecvViewManager.h` | API additions | E1 |
| `app/MainWindow.cpp` | Tab close with splitters, GLView fallback, tool binding | E2, E3, E5 |
| `app/MainWindow.h` | Header updates | E3 |
| `libs/CV_db/src/ecvDrawableObject.cpp` | Entity display reassignment helper | E1 |

---

## Phase E: Object Lifecycle + View Close Safety

### Task E1: Reassign orphaned objects to surviving view on close

**Why:** When a view is closed, `detachEntitiesFromView` sets `m_currentDisplay = nullptr` on all objects that were in that view. This makes them "unbound" — visible in ALL views (`isDisplayedIn` returns true for `nullptr` display). The user requirement says objects are window-exclusive. They must be reassigned to a surviving view, not left floating.

**ParaView parallel:** When a view is closed in ParaView, representations are removed from the closed view. The source still exists in the pipeline but is not auto-shown in another view — it becomes invisible until the user explicitly makes it visible elsewhere.

**Our design choice:** Reassign orphaned objects to the view that becomes active after close (matching the "objects always belong to one window" requirement).

**Files:**
- Modify: `libs/CV_db/src/ecvViewManager.cpp` — `detachEntitiesFromView`
- Modify: `libs/CV_db/include/ecvViewManager.h` — add `reassignEntitiesToView` declaration

- [ ] **Step 1: Change detachEntitiesFromView to reassign instead of nullify**

In `libs/CV_db/src/ecvViewManager.cpp`, replace the current `detachEntitiesFromView`:

```cpp
void ecvViewManager::detachEntitiesFromView(ecvGenericGLDisplay* closingView) {
    if (!closingView) return;

    ccHObject* sceneDB = ecvDisplayTools::GetSceneDB();
    if (!sceneDB) return;

    // Find a surviving view to receive orphaned entities.
    // Prefer the view that will become active (last in list, excluding closing).
    ecvGenericGLDisplay* recipient = nullptr;
    for (int i = m_views.size() - 1; i >= 0; --i) {
        if (m_views[i] != closingView) {
            recipient = m_views[i];
            break;
        }
    }

    reassignEntitiesFromView(sceneDB, closingView, recipient);
}

void ecvViewManager::reassignEntitiesFromView(
        ccHObject* root,
        ecvGenericGLDisplay* fromView,
        ecvGenericGLDisplay* toView) {
    if (!root) return;

    if (root->getDisplay() == fromView) {
        if (toView) {
            root->setDisplay(toView);
        } else {
            root->removeFromDisplay(fromView);
        }
    }

    for (unsigned i = 0; i < root->getChildrenNumber(); ++i) {
        reassignEntitiesFromView(root->getChild(i), fromView, toView);
    }
}
```

- [ ] **Step 2: Add declaration to ecvViewManager.h**

```cpp
    void reassignEntitiesFromView(ccHObject* root,
                                  ecvGenericGLDisplay* fromView,
                                  ecvGenericGLDisplay* toView);
```

- [ ] **Step 3: Build and verify**

Run: `source /Users/asher/opt/anaconda3/etc/profile.d/conda.sh && conda activate cloudViewer && cd /Users/asher/develop/code/github/ACloudViewer/build_app && cmake --build . --target ACloudViewer -- -j12`
Expected: Clean compilation.

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/src/ecvViewManager.cpp libs/CV_db/include/ecvViewManager.h
git commit -m "fix(multiview): reassign orphaned objects to surviving view on close instead of nullifying"
```

---

### Task E2: Fix tab close to handle all views inside a QSplitter

**Why:** When closing an MDI tab that contains a `QSplitter` with multiple `ecvGLView` instances, `prepareViewClose` only finds and unregisters ONE view (the first child match). Other views in the splitter are only cleaned up when Qt destroys them, causing inconsistent ordering of `detachEntitiesFromView`, primary promotion, and tool rebinding.

**Files:**
- Modify: `app/MainWindow.cpp` — `prepareViewClose`

- [ ] **Step 1: Find all views in a frame recursively before closing**

In `MainWindow.cpp`, modify the beginning of `prepareViewClose` to find ALL `ecvGenericGLDisplay` instances in the frame:

```cpp
void MainWindow::prepareViewClose(QWidget* viewFrame) {
    if (!viewFrame) return;

    // Collect ALL GL views in this frame (handles QSplitter with multiple views)
    std::vector<ecvGenericGLDisplay*> viewsToClose;
    auto* directDisplay = ecvGenericGLDisplay::FromWidget(viewFrame);
    if (directDisplay) {
        viewsToClose.push_back(directDisplay);
    }
    const auto children = viewFrame->findChildren<QWidget*>();
    for (QWidget* child : children) {
        auto* display = ecvGenericGLDisplay::FromWidget(child);
        if (display && display != directDisplay) {
            viewsToClose.push_back(display);
        }
    }

    if (viewsToClose.empty()) return;

    // Process each view
    for (auto* closingDisplay : viewsToClose) {
        // ... existing per-view cleanup logic (unregister, primary promotion, etc.)
```

The key change is wrapping the existing single-view cleanup in a loop over all discovered views.

- [ ] **Step 2: Ensure primary promotion only runs once**

Inside the loop, track whether primary promotion has already been handled:

```cpp
    bool primaryHandled = false;
    for (auto* closingDisplay : viewsToClose) {
        bool closingPrimary = false;
        // ... determine if this view is the primary ...
        
        ecvViewManager::instance().unregisterView(closingDisplay);

        if (closingPrimary && !primaryHandled) {
            primaryHandled = true;
            // ... existing primary adoption logic ...
        }
    }
```

- [ ] **Step 3: Build and verify**

Run: `source /Users/asher/opt/anaconda3/etc/profile.d/conda.sh && conda activate cloudViewer && cd /Users/asher/develop/code/github/ACloudViewer/build_app && cmake --build . --target ACloudViewer -- -j12`

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): handle all views in QSplitter when closing MDI tab"
```

---

### Task E3: Fix getActiveGLView fallback inconsistency

**Why:** `MainWindow::getActiveGLView()` falls back to `ecvDisplayTools::TheInstance()` when `FromWidget` fails, but `getActiveGLDisplay()` returns the `ecvViewManager` active view directly. This inconsistency means callers of `getActiveGLView` can get a different (wrong) view than callers of `getActiveGLDisplay`.

**Files:**
- Modify: `app/MainWindow.cpp` — `getActiveGLView`

- [ ] **Step 1: Read the current getActiveGLView implementation**

Read `app/MainWindow.cpp` around line 2406 to understand the current fallback chain.

- [ ] **Step 2: Align getActiveGLView with getActiveGLDisplay**

Replace the implementation to use `ecvViewManager` first:

```cpp
ecvGLView* MainWindow::getActiveGLView() {
    auto* activeDisplay = ecvViewManager::instance().getActiveView();
    if (activeDisplay) {
        auto* glView = dynamic_cast<ecvGLView*>(activeDisplay);
        if (glView) return glView;
    }
    // Final fallback to primary singleton (for backward compat)
    return dynamic_cast<ecvGLView*>(ecvDisplayTools::TheInstance());
}
```

- [ ] **Step 3: Build and verify**

Run: `source /Users/asher/opt/anaconda3/etc/profile.d/conda.sh && conda activate cloudViewer && cd /Users/asher/develop/code/github/ACloudViewer/build_app && cmake --build . --target ACloudViewer -- -j12`

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): align getActiveGLView with ecvViewManager for consistent view resolution"
```

---

### Task E4: Fix transform/filter/measurement tools to explicitly resolve active view

**Why:** `activateTranslateRotateMode`, `doActionMeasurementMode`, and `doActionFilterMode` all call `ecvDisplayTools::GetVisualizer3D()` and `ecvDisplayTools::GetCurrentScreen()`. While these are usually correct after `rebindToolsToActiveView`, there are race conditions where focus changes between rebind and tool activation. Explicit resolution is safer.

**Files:**
- Modify: `app/MainWindow.cpp` — tool activation functions

- [ ] **Step 1: Read the current activateTranslateRotateMode**

Read the function to understand where `GetVisualizer3D()` and `GetCurrentScreen()` are used.

- [ ] **Step 2: Add explicit active view resolution at entry**

At the start of each tool activation function, add an explicit active view check:

```cpp
void MainWindow::activateTranslateRotateMode() {
    // Ensure tools target the correct view
    auto* activeDisplay = ecvViewManager::instance().getActiveView();
    auto* activeView = dynamic_cast<ecvGLView*>(activeDisplay);
    if (activeView) {
        rebindToolsToActiveView(activeView);
    }
    // ... existing implementation using GetVisualizer3D() / GetCurrentScreen() ...
```

Apply the same pattern to `doActionMeasurementMode` and `doActionFilterMode`.

- [ ] **Step 3: Build and verify**

Run: `source /Users/asher/opt/anaconda3/etc/profile.d/conda.sh && conda activate cloudViewer && cd /Users/asher/develop/code/github/ACloudViewer/build_app && cmake --build . --target ACloudViewer -- -j12`

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): explicit active view resolution in tool activation functions"
```

---

### Task E5: Full compilation + integration verification

- [ ] **Step 1: Full rebuild**

Run: `source /Users/asher/opt/anaconda3/etc/profile.d/conda.sh && conda activate cloudViewer && cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j12`
Expected: 0 errors, 0 new warnings.

---

## Summary: Root Causes → Fixes (Phase 2)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Objects visible in all views after owning view closes | `detachEntitiesFromView` nullifies display instead of reassigning | Task E1 |
| Tab close misses views in QSplitter | `prepareViewClose` only handles first found view | Task E2 |
| `getActiveGLView` returns wrong view | Falls back to `TheInstance()` instead of `ecvViewManager` | Task E3 |
| Tool activation race condition | Tools read globals before rebind completes | Task E4 |
