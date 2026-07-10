# CloudCompare Plugin Migration Guide for ACloudViewer

This document guides developers migrating plugins from CloudCompare (CC) to ACloudViewer (ACV).
ACloudViewer has completed a paradigm shift from CC's QMdiArea + native OpenGL to a
**ParaView-style Tab+Split layout with VTK backend**.

---

## 1. Architecture Differences at a Glance

| Aspect | CloudCompare | ACloudViewer |
|--------|-------------|--------------|
| Window container | `QMdiArea` + `QMdiSubWindow` | `ecvTabbedMultiViewWidget` + `QSplitter` (KD-tree) |
| GL view class | `ccGLWindow` / `ccGLWindowInterface` | `vtkGLView` implementing `ecvGenericGLDisplay` |
| View coordinator | None (MainWindow directly) | `ecvViewManager` singleton |
| Rendering backend | Native OpenGL + FBO | VTK `vtkGenericOpenGLRenderWindow` |
| Entity-view binding | Strict 1:1 (`m_currentDisplay == context.display`) | 1:1 primary + per-view representation + view linking |
| Unbound entity | Not drawn anywhere | **Drawn in ALL views** (global visibility) |
| View close behavior | Entity stays enabled, display set to null | Entity **disabled** + unbound |

---

## 2. API Migration Table

### 2.1 Main App Interface

| CloudCompare (`ccMainAppInterface`) | ACloudViewer (`ecvMainAppInterface`) | Notes |
|-------------------------------------|--------------------------------------|-------|
| `getActiveGLWindow()` → `ccGLWindowInterface*` | `getActiveGLDisplay()` → `ecvGenericGLDisplay*` | Also: `getActiveWindow()` → `QWidget*` |
| `createGLWindow(win, widget)` | `vtkGLView::Create()` + `addViewWidget()` | No direct equivalent in plugin API |
| `destroyGLWindow(win)` | Close view via layout system | No direct equivalent |
| `disableAllBut(win)` | `disableAll()` / `enableAll()` + `freezeUI(true)` | Disables entire tab container |
| `redrawAll(only2D)` | `redrawAll(only2D, forceRedraw)` | Extra `forceRedraw` parameter |
| `refreshAll(only2D)` | `refreshAll(only2D, forceRedraw)` | Extra `forceRedraw` parameter |

### 2.2 GL Window Interface

| CloudCompare (`ccGLWindowInterface`) | ACloudViewer (`ecvGenericGLDisplay` / `vtkGLView`) | Notes |
|--------------------------------------|---------------------------------------------------|-------|
| `asQObject()` | `asWidget()` | Returns `QWidget*` |
| `signalEmitter()` | Via `ecvViewManager` signals | No per-view signal emitter |
| `FromWidget(widget)` | `ecvGenericGLDisplay::FromWidget(widget)` | Static registry lookup |
| `FromEmitter(obj)` | Not available | Use `ecvViewManager::activeViewChanged` |
| `Create()` | `ecvGenericGLDisplay::CreateView(parent, ...)` | Factory method (plugins) |
| `setGlFilter(filter)` | Not supported | VTK replaces GL filters |
| `getGlFilter()` | Not supported | — |
| `renderToImage(...)` | `renderToFile(...)` | Different signature |
| `toCenteredGLCoordinates(x, y)` | Not available | Use VTK coordinate transforms |
| `getScreenSize()` | `QSize(glWidth(), glHeight())` | — |
| `redraw(only2D, resetLOD)` | `redraw(only2D, forceRedraw)` | 2nd param semantics differ! |
| `setDisplayParameters(p, thisWindowOnly)` | Same signature | **Always pass `true` for per-view** |

### 2.3 Overlay Dialog

| CloudCompare | ACloudViewer | Notes |
|---|---|---|
| `linkWith(ccGLWindowInterface*)` | `linkWith(QWidget*)` | Widget-based |
| Manual `activeViewChanged` handling | `ccOverlayDialog::start()` auto-relinks | Base class handles view switching |
| `m_associatedWin` is `ccGLWindow*` | `m_associatedWin` is `QWidget*` | — |
| `bindToView(...)` | `bindToView(ecvGenericGLDisplay*)` | Lock to specific view |

### 2.4 Picking

| CloudCompare | ACloudViewer | Notes |
|---|---|---|
| `onActiveWindowChanged(QMdiSubWindow*)` | `onActiveViewWidgetChanged(QWidget*)` | Signal name/type changed |
| `PickedItem::uvw` filled | `PickedItem::uvw` may be empty | VTK picker limitation |
| Single active window implicit | `PickedItem::pickView` — explicit pick source | Multi-view aware |

---

## 3. Common Migration Patterns

### 3.1 Getting the Active View

**Before (CC):**
```cpp
ccGLWindowInterface* win = m_app->getActiveGLWindow();
if (!win) return;
win->redraw(true, false);
```

**After (ACV):**
```cpp
ecvGenericGLDisplay* view = m_app->getActiveGLDisplay();
if (!view) return;
view->redraw(true, true);
```

Or for widget-level operations:
```cpp
QWidget* win = m_app->getActiveWindow();
if (!win) return;
```

### 3.2 Per-View Display Parameters

**Before (CC):**
```cpp
ccGui::ParamStruct params = m_glWindow->getDisplayParameters();
params.backgroundCol = white;
m_glWindow->setDisplayParameters(params, true);  // per-window
```

**After (ACV):**
```cpp
auto* view = ecvViewManager::instance().getEffectiveView();
if (!view) return;
ecvGui::ParamStruct params = view->getDisplayParameters();
params.backgroundCol = ecvColor::white;
view->setDisplayParameters(params, true);  // MUST pass true for per-view!
```

> **WARNING:** Calling `setDisplayParameters(params)` without `true` writes to the
> global `ecvGui::Set()`, polluting all views. Always use `true` when modifying
> parameters for a tool/dialog-specific view.

### 3.3 Overlay Dialog Lifecycle

**Before (CC):**
```cpp
m_dialog->linkWith(m_app->getActiveGLWindow());
m_app->registerOverlayDialog(m_dialog, Qt::TopRightCorner);
m_app->freezeUI(true);
m_app->disableAllBut(m_app->getActiveGLWindow());
```

**After (ACV):**
```cpp
m_dialog->linkWith(m_app->getActiveWindow());
m_app->registerOverlayDialog(m_dialog, Qt::TopRightCorner);
m_app->freezeUI(true);
// disableAllBut removed — base class ccOverlayDialog::start() handles
// view-switching automatically via activeViewChanged signal
m_dialog->start();
```

### 3.4 Adding Temporary Objects to View

**Before (CC):**
```cpp
m_glWindow->addToOwnDB(tempObject);
```

**After (ACV):**
```cpp
auto* view = ecvViewManager::instance().getEffectiveView();
if (view) {
    view->addToOwnDB(tempObject, true);
}
```

> **Note:** If the user switches views, ownDB objects remain in the original view.
> For tools that need to follow view switches, migrate ownDB objects in the
> `activeViewChanged` callback or use `ccOverlayDialog::linkWith()` which
> handles relinking automatically.

### 3.5 Event Filter Installation

**Before (CC):**
```cpp
m_app->getActiveGLWindow()->asQObject()->installEventFilter(this);
```

**After (ACV):**
```cpp
QWidget* win = m_app->getActiveWindow();
if (win) {
    win->installEventFilter(this);
}
```

For multi-view awareness (if your tool needs to follow view switches):
```cpp
connect(&ecvViewManager::instance(), &ecvViewManager::activeViewChanged,
        this, [this](ecvGenericGLDisplay*, ecvGenericGLDisplay*) {
    QWidget* w = m_app->getActiveWindow();
    if (w && !m_filteredWindows.contains(w)) {
        w->installEventFilter(this);
        m_filteredWindows.insert(w);
    }
});
```

---

## 4. Key Behavioral Differences

### 4.1 Entity Visibility Semantics

- **CC:** `m_currentDisplay == nullptr` → entity not drawn anywhere
- **ACV:** `m_currentDisplay == nullptr` → entity drawn in **ALL** views (global)

To bind an entity to a specific view:
```cpp
entity->setDisplay_recursive(targetView);
```

### 4.2 View Creation (New 3D View)

- **CC:** `new3DView()` immediately creates a new `ccGLWindow` with scene
- **ACV:** Creates empty Tab; user must click "Create Render View" (ParaView design)

For plugins needing a dedicated view:
```cpp
m_app->addViewWidget(myViewWidget);
```

### 4.3 Window Close Behavior

- **CC:** Entity stays enabled, `m_currentDisplay` set to null
- **ACV:** Entity **disabled** and unbound; user must re-enable in DB tree

### 4.4 `redraw()` vs `refresh()`

- **CC:** `redrawAll()` forces full repaint; `refreshAll()` only repaints dirty windows
- **ACV:** `redrawAll()` defaults to `refreshAll()` internally; use `forceRedraw=true` for forced full repaint

---

## 5. Signals & Slots Migration

| CC Signal (`ccGLWindowSignalEmitter`) | ACV Equivalent |
|---------------------------------------|----------------|
| `mouseWheelRotated(float)` | `ecvViewManager::mouseWheelRotated` |
| `leftButtonClicked(int, int)` | Via VTK interactor / `ecvViewManager` |
| `mouseMoved(int, int, Qt::MouseButtons)` | `ecvViewManager::mouseMoved` |
| `itemPicked(...)` | `ecvViewManager::itemPicked` |
| `aboutToClose(ccGLWindow*)` | `ecvViewManager::viewClosing` |

---

## 6. GL Filter Plugins

GL filter plugins (`ccGLPluginInterface`) are **not supported** in ACloudViewer.
The VTK rendering pipeline replaces the CC OpenGL FBO-based filter mechanism.

To achieve similar effects (EDL, SSAO), use VTK rendering passes or
post-processing shaders through the `VtkVis` pipeline.

---

## 7. Multi-Window Best Practices (Lessons from Plugin Audit)

### 7.1 The `m_bindView` Pattern

Interactive tools and dialogs that span multiple user interactions **must** cache the
view at activation time and use it consistently throughout their lifecycle. Do NOT call
`getEffectiveView()` on every operation — the user may switch views mid-operation.

```cpp
class MyToolDialog : public ccOverlayDialog {
    ecvGenericGLDisplay* m_bindView = nullptr;

    void onStart() {
        m_bindView = ecvViewManager::instance().getEffectiveView();
        if (m_bindView) {
            m_bindView->setPickingMode(...);
            m_bindView->addToOwnDB(m_tempObj);
        }
    }

    void onMouseMove(int x, int y) {
        if (!m_bindView) return;
        ccGLCameraParameters cam;
        m_bindView->getGLCameraParameters(cam);
        // use cam for coordinate transforms...
    }

    void onStop() {
        if (m_bindView) {
            m_bindView->removeFromOwnDB(m_tempObj);
            m_bindView->setPickingMode(DEFAULT_PICKING);
        }
    }
};
```

**Plugins already fixed with this pattern:**
- `qCloudLayers/ccMouseCircle` — ownDB bound to construction-time view
- `qCompass/ccMouseCircle` — same fix
- `qSRA/distanceMapGenerationDlg` — all ownDB ops use `m_bindView`
- `qAnimation/qAnimationDlg` — rendering & resize use `m_bindView`
- `qCloudLayers/ccCloudLayersDlg` — picking & interaction use `m_bindView`
- `qCanupo/qCanupo2DViewDialog` — dedicated `vtkGLView` instance

### 7.2 Creating Dedicated GL Views (qCanupo Pattern)

For plugins needing a separate 2D/3D view (like CC's `m_glWindow` in dialog),
use the backend-agnostic factory in `ecvGenericGLDisplay` — **no VtkEngine dependency**:

```cpp
// In constructor:
m_glView = ecvGenericGLDisplay::CreateView(parentWindow, true, false);
if (m_glView) {
    ecvGui::ParamStruct params = m_glView->getDisplayParameters();
    params.backgroundCol = ecvColor::white;
    m_glView->setDisplayParameters(params, true);
    m_glView->setPerspectiveState(false, true);  // 2D mode

    viewFrame->setLayout(new QHBoxLayout());
    viewFrame->layout()->addWidget(m_glView->asWidget());

    // Connect per-view signals via signalSource() (no vtkGLView dependency)
    if (QObject* src = m_glView->signalSource()) {
        connect(src, SIGNAL(leftButtonClicked(int,int)), this, SLOT(onLeftClick(int,int)));
        connect(src, SIGNAL(mouseMoved(int,int,Qt::MouseButtons)), this, SLOT(onMouseMove(int,int,Qt::MouseButtons)));
    }
}

// In destructor:
if (m_glView) {
    if (QObject* src = m_glView->signalSource()) {
        src->disconnect(this);
        delete src;  // deletes the concrete view (e.g. vtkGLView)
    }
    m_glView = nullptr;
}
```

No extra CMake linkage needed — the factory is in CV_db.

### 7.3 RPC/API Multi-View Support (qJSonRPC Pattern)

For remote API plugins, add optional `view_id` parameter:

```cpp
ecvGenericGLDisplay* resolveView(const QMap<QString, QVariant>& params) {
    if (params.contains("view_id")) {
        return ecvViewManager::instance().findView(params["view_id"].toInt());
    }
    return ecvViewManager::instance().getEffectiveView();
}
```

And expose `view.list` to let clients discover views.

### 7.4 Python Binding Per-View Access

Python scripts access per-view methods through `ccGenericGLDisplay`:

```python
vm = cloudViewer.ccViewManager.instance()
view = vm.findView(view_id)  # or getEffectiveView()
view.setViewportParameters(params)
view.redraw()
view.addToOwnDB(obj)
```

### 7.5 Actor Lifecycle in Custom Entities (qCompass Pattern)

Entities that create VTK actors in `drawMeOnly()` must track which view they were
drawn in, so cleanup targets the correct view — not just `getEffectiveView()`:

```cpp
class MyEntity : public ccPolyline {
    ecvGenericGLDisplay* m_lastDrawnView = nullptr;
    QStringList m_actorViewIds;

    void drawMeOnly(CC_DRAW_CONTEXT& context) override {
        ecvGenericGLDisplay* view = context.display
                ? context.display
                : ecvViewManager::instance().getEffectiveView();
        if (!view) return;
        m_lastDrawnView = view;  // cache for cleanup

        // ... create actors with viewIDs ...
    }

    void removeActors() {
        ecvGenericGLDisplay* view = m_lastDrawnView
                ? m_lastDrawnView
                : ecvViewManager::instance().getEffectiveView();
        if (!view) { m_actorViewIds.clear(); return; }
        for (const auto& id : m_actorViewIds) {
            CC_DRAW_CONTEXT ctx;
            ctx.display = view;
            ctx.removeViewID = id;
            ctx.removeEntityType = ENTITY_TYPE::ECV_MESH;
            view->removeEntities(ctx);
        }
        m_actorViewIds.clear();
    }
};
```

**Entities fixed with this pattern:**
- `qCompass/ccTrace` — segment & waypoint marker actors
- `qCompass/ccPointPair` — point/body/head mesh actors
- `qCompass/ccSNECloud` — normal line actors

### 7.6 Removed Dead Code: `scheduleFullRedraw` / `m_scheduleTimer`

The timer mechanism (`m_scheduleTimer`, `m_scheduledFullRedrawTime`,
`scheduleFullRedraw()`, `checkScheduledRedraw()`, `cancelScheduledRedraw()`) has been
**completely removed** from `ecvDisplayTools`, `ecvGenericGLDisplay`, and `vtkGLView`.
This was dead code carried over from CC's auto-refresh pattern with zero callers.

### 7.7 `m_bindView` Safety: View Deletion During Plugin Execution

**Risk**: `ecvGenericGLDisplay` is not a `QObject`, so `QPointer` cannot be used. If
a user closes a view while a plugin holds a raw `m_bindView` pointer, a use-after-free
crash can occur.

**Two-layer defense** (implemented):

1. **UI lock**: `MainWindow::freezeUI(true)` now disables all "Close View" buttons
   via `ecvTabbedMultiViewWidget::setViewCloseButtonsEnabled(false)`, preventing the
   user from closing any view while a plugin is active.

2. **Signal safety net**: `ccOverlayDialog::start()` connects to
   `ecvViewManager::viewUnregistered`. If the bound view is unregistered
   (`onBoundViewUnregistered`), `m_boundView` is nulled and `stop(false)` is called
   automatically to prevent dangling pointer access.

Plugin-specific `m_bindView` fields are cleaned up in each plugin's `stop()` override.

### 7.8 Patterns That Don't Need Fixing

- **One-shot operations** (export, screenshot, compute): Using `getEffectiveView()` at
  call time is correct — the user expects the operation on the currently active view
- **Entity creation** (`setDisplay(getEffectiveView())`): Correct for newly created
  entities that should appear in the current view

---

## 8. Checklist for Plugin Migration

- [ ] Replace `getActiveGLWindow()` with `getActiveGLDisplay()` or `getActiveWindow()`
- [ ] Replace `ccGLWindowInterface*` types with `ecvGenericGLDisplay*` or `QWidget*`
- [ ] Add `true` to all `setDisplayParameters()` calls that should be per-view
- [ ] Replace `linkWith(ccGLWindow*)` with `linkWith(QWidget*)`
- [ ] Use `ccOverlayDialog::start()`/`stop()` for automatic view-switch handling
- [ ] Remove `disableAllBut()` calls (use `freezeUI()` instead)
- [ ] Replace `signalEmitter()` connections with `ecvViewManager` signals
- [ ] Add null checks for all view/window pointer retrieval
- [ ] **Cache view at tool activation** (`m_bindView` pattern) for interactive tools
- [ ] **Track `m_lastDrawnView`** in custom entities with VTK actor lifecycle
- [ ] **Use `context.display`** in `drawMeOnly()` instead of `getEffectiveView()`
- [ ] **Use `ecvGenericGLDisplay::CreateView()`** for dialog-embedded separate views
- [ ] **Add `view_id` support** for remote API methods
- [ ] Test with multiple split views open simultaneously
- [ ] Test view switching while tool is active
- [ ] Test view close while tool is active

---

## 8. Quick Reference: ecvViewManager API

```cpp
// Get the singleton
auto& vm = ecvViewManager::instance();

// Active view (UI-focused)
ecvGenericGLDisplay* view = vm.getActiveView();

// Effective view (rendering context — may differ during redrawAll)
ecvGenericGLDisplay* view = vm.getEffectiveView();

// Active widget
QWidget* widget = vm.activeWidget();

// All registered views
QList<ecvGenericGLDisplay*> views = vm.getAllViews();

// View count
int count = vm.viewCount();

// Batch operations
vm.redrawAll(only2D, forceRedraw);
vm.refreshAll(only2D, forceRedraw);

// Entity management
vm.moveEntityToView(entity, targetView);
vm.associateToActiveView(entity);

// Signals
connect(&vm, &ecvViewManager::activeViewChanged, ...);
connect(&vm, &ecvViewManager::viewRegistered, ...);
connect(&vm, &ecvViewManager::viewClosing, ...);
```
