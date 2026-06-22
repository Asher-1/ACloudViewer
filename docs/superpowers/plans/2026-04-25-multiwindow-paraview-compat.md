# Multi-Window ParaView-Compatible Refactoring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all multi-window bugs (widget scaling, show-name leakage, Edit Camera, toolbar action routing, capture/2D toggle) by aligning rendering state routing with ParaView's per-view architecture.

**Architecture:** Each `ecvGLView` owns a self-contained `ecvViewContext`. All static `ecvDisplayTools` methods that read viewport/camera state must resolve to the **currently rendering** view's context (via `effectiveCtx()` with `ScopedRenderOverride`). Entity drawing must use `context.display` for camera/viewport queries, not the global "active" view. Per-view toolbar actions must activate the correct view before operating.

**Tech Stack:** C++17, Qt 5.x (QMdiArea, QSplitter), VTK, `ecvViewManager::ScopedRenderOverride`

---

## File Map

| File | Responsibility | Action |
|------|---------------|--------|
| `libs/VtkEngine/Visualization/ecvGLView.cpp` | Per-view redraw + camera params | Modify (Tasks 1, 2) |
| `libs/CV_db/src/ecvDisplayTools.cpp` | `DisplayText`, `RenderText`, `DrawClickableItems` viewport routing | Modify (Tasks 3, 7) |
| `libs/CV_db/src/ecvHObject.cpp` | Entity name drawing (show name) | Modify (Task 4) |
| `libs/VtkEngine/Tools/CameraTools/EditCameraTool.cpp` | Camera tool per-view binding | Modify (Task 5) |
| `libs/VtkEngine/Tools/CameraTools/EditCameraTool.h` | Camera tool API | Modify (Task 5) |
| `app/ecvMultiViewFrameManager.cpp` | Per-view toolbar button wiring | Modify (Task 5) |
| `app/MainWindow.cpp` | Toolbar action routing + screenshot + camera | Modify (Tasks 5, 6, 8) |

---

### Task 1: Add ScopedRenderOverride to ecvGLView::redraw

**Why:** When `ecvGLView::redraw()` is triggered directly (Qt resize, user interaction) rather than through `ecvDisplayTools::RedrawDisplay`, no `ScopedRenderOverride` is active. Any code calling `effectiveCtx()` during such a redraw gets the **primary** view's context, causing wrong viewport dimensions for widgets/overlays.

**ParaView parallel:** Each `vtkPVView` render always has its own `vtkRenderWindow` context; there is no "wrong view" ambiguity.

**Files:**
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp:111-125`

- [ ] **Step 1: Add ScopedRenderOverride at start of ecvGLView::redraw**

```cpp
void ecvGLView::redraw(bool only2D, bool forceRedraw) {
    if (!m_visualizer3D || !m_vtkWidget) return;

    // Ensure effectiveCtx() resolves to THIS view's context for the
    // entire duration of drawing. This is critical when redraw() is
    // triggered directly (resize, interaction) rather than through
    // ecvDisplayTools::RedrawDisplay which sets its own guard.
    ecvViewManager::ScopedRenderOverride viewGuard(this);

    // Sync per-view glViewport from the actual widget so that
    // ComputeActualPixelSize() (which reads effectiveCtx().glViewport)
    // returns correct dimensions for this sub-window.
    const int dpr = static_cast<int>(m_vtkWidget->devicePixelRatioF());
    m_ctx.glViewport = QRect(0, 0,
                             m_vtkWidget->width() * dpr,
                             m_vtkWidget->height() * dpr);

    // --- Build draw context from per-view state ---
    CC_DRAW_CONTEXT context;
    getContext(context);
    context.forceRedraw = forceRedraw;
    // ... rest unchanged ...
```

- [ ] **Step 2: Build and verify compilation**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && conda activate cloudViewer && make -j48`
Expected: No new errors or warnings from `ecvGLView.cpp`.

- [ ] **Step 3: Commit**

```bash
git add libs/VtkEngine/Visualization/ecvGLView.cpp
git commit -m "fix(multiview): add ScopedRenderOverride to ecvGLView::redraw for correct effectiveCtx"
```

---

### Task 2: Fix ecvGLView::getGLCameraParameters DPR consistency

**Why:** `ecvGLView::getGLCameraParameters` uses logical pixels (`m_vtkWidget->width()`) while the primary path in `ecvDisplayTools::GetGLCameraParameters` uses `Width() * GetDevicePixelRatio()`. This inconsistency causes 3D→2D projection mismatches (names/labels appear at wrong positions/sizes in secondary views).

**ParaView parallel:** `vtkPVView::PPI` + `vtkRenderWindow::SetDPI` ensures consistent pixel-space across views.

**Files:**
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp:270-281`

- [ ] **Step 1: Apply DPR scaling to viewport in getGLCameraParameters**

```cpp
void ecvGLView::getGLCameraParameters(ccGLCameraParameters& params) const {
    if (!m_vtkWidget) return;
    const int dpr = static_cast<int>(m_vtkWidget->devicePixelRatioF());
    params.viewport[0] = 0;
    params.viewport[1] = 0;
    params.viewport[2] = m_vtkWidget->width() * dpr;
    params.viewport[3] = m_vtkWidget->height() * dpr;
    params.perspective = m_ctx.viewportParams.perspectiveView;
    params.fov_deg = m_ctx.viewportParams.fov_deg;
    params.pixelSize = m_ctx.viewportParams.pixelSize;
    params.modelViewMat = m_ctx.viewMatd;
    params.projectionMat = m_ctx.projMatd;
}
```

- [ ] **Step 2: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`
Expected: Clean compilation.

- [ ] **Step 3: Commit**

```bash
git add libs/VtkEngine/Visualization/ecvGLView.cpp
git commit -m "fix(multiview): apply DPR scaling in ecvGLView::getGLCameraParameters for consistent projection"
```

---

### Task 3: Route DisplayText/RenderText through correct view's viewport

**Why:** `DisplayText` and `RenderText` use `s_tools.instance->effectiveCtx().glViewport.height()` for Y-coordinate flip and layout. The `display` parameter is accepted but **completely ignored**. When the painting view differs from the effective view, text renders at wrong Y positions or in the wrong window.

**ParaView parallel:** Each `vtkTextSourceRepresentation` is bound to a specific view via `AddToView`/`RemoveFromView`; labels use the view's own renderer viewport.

**Files:**
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp:3703-3740` (RenderText)
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp:3773-3857` (DisplayText)

- [ ] **Step 1: Add helper to get viewport height from display or effectiveCtx**

At the top of `ecvDisplayTools.cpp` (near `activeSecondaryView()`), add:

```cpp
static int viewportHeightFor(ecvGenericGLDisplay* display) {
    if (display && display != s_tools.instance) {
        auto* ctx = display->viewContext();
        if (ctx) return ctx->glViewport.height();
    }
    return s_tools.instance->effectiveCtx().glViewport.height();
}
```

- [ ] **Step 2: Update RenderText (2D overload) to use display parameter**

In `ecvDisplayTools::RenderText(int x, int y, ...)` (line 3703):

```cpp
void ecvDisplayTools::RenderText(
        int x,
        int y,
        const QString& str,
        const QFont& font,
        const ecvColor::Rgbub& color,
        const QString& id,
        ecvGenericGLDisplay* display) {
    CC_DRAW_CONTEXT context;
    if (id.isEmpty()) {
        context.viewID = str;
    } else {
        context.viewID = id;
    }

    context.textParam.text = str;
    context.textParam.display3D = false;
    context.textParam.font = font;
    context.textParam.font.setPointSize(font.pointSize());

    context.textDefaultCol = color;
    int vpH = viewportHeightFor(display);
    if (context.textParam.display3D) {
        context.textParam.textScale = GetPlatformAwareDPIScale();
        CCVector3d input3D(x, vpH - y, 0);
        CCVector3d output2D;
        ToWorldPoint(input3D, output2D);
        context.textParam.textPos.x = output2D.x;
        context.textParam.textPos.y = output2D.y;
        context.textParam.textPos.z = output2D.z;
    } else {
        context.textParam.textPos.x = x;
        context.textParam.textPos.y = vpH - y;
        context.textParam.textPos.z = 0;
    }
    DisplayText(context);
}
```

- [ ] **Step 3: Update DisplayText to use display parameter for Y-flip**

In `ecvDisplayTools::DisplayText(const QString& text, int x, int y, ...)` (line 3773):

Replace all `s_tools.instance->effectiveCtx().glViewport.height()` with `viewportHeightFor(display)`:

```cpp
void ecvDisplayTools::DisplayText(
        const QString& text,
        int x,
        int y,
        unsigned char align,
        float bkgAlpha,
        const unsigned char* rgbColor,
        const QFont* font,
        const QString& id,
        ecvGenericGLDisplay* display) {
    int vpH = viewportHeightFor(display);
    int x2 = x;
    int y2 = vpH - 1 - y;

    // actual text color
    const unsigned char* col =
            (rgbColor ? rgbColor : GetDisplayParameters().textDefaultCol.rgb);

    QFont realFont = (font ? *font : s_tools.instance->m_font);
    QFont textFont = realFont;
    QFontMetrics fm(textFont);
    int margin = fm.height() / 4;

    if (align != ALIGN_DEFAULT || bkgAlpha != 0.0f) {
        QRect rect = fm.boundingRect(text);

        if (align & ALIGN_HMIDDLE)
            x2 -= rect.width() / 2;
        else if (align & ALIGN_HRIGHT)
            x2 -= rect.width();
        if (align & ALIGN_VMIDDLE)
            y2 += rect.height() / 2;
        else if (align & ALIGN_VBOTTOM)
            y2 += rect.height();

        if (bkgAlpha != 0.0f) {
            const float invertedCol[4] = {(255 - col[0]) / 255.0f,
                                          (255 - col[0]) / 255.0f,
                                          (255 - col[0]) / 255.0f, bkgAlpha};

            int xB = x2;
            int yB = vpH - y2;

            WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, id);
            param.text = text;

            if (id.isEmpty()) {
                param.viewID = text;
                RemoveWidgets(param);
            }

            param.color.r = invertedCol[0];
            param.color.g = invertedCol[1];
            param.color.b = invertedCol[2];
            param.color.a = invertedCol[3];
            param.rect =
                    QRect(xB - margin, yB - margin, rect.width() + 2 * margin,
                          static_cast<int>(rect.height() + 1.5 * margin));

#ifdef Q_OS_MAC
            param.rect.setWidth(param.rect.width() * 2);
            param.rect.moveTop(std::min(vpH, param.rect.y() + 2 * margin));
#endif

            DrawWidgets(param, true);
        }
    }

    if (align & ALIGN_VBOTTOM)
        y2 -= margin;
    else if (align & ALIGN_VMIDDLE)
        y2 -= margin / 2;

    ecvColor::Rgbub textColor(col);
    RenderText(x2, y2, text, realFont, textColor, id, display);
}
```

- [ ] **Step 4: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`
Expected: Clean compilation.

- [ ] **Step 5: Commit**

```bash
git add libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "fix(multiview): route DisplayText/RenderText through display parameter for correct per-view Y-flip"
```

---

### Task 4: Fix ccHObject::draw to use context.display for camera params

**Why:** `ccHObject::draw()` calls `ecvDisplayTools::GetGLCameraParameters(camera)` at line 1552 which goes through the global `effectiveCtx()` path. When the entity is being drawn in a secondary view, this can return the primary view's camera, causing the show-name to project to wrong screen coordinates and appear in the wrong window.

**ParaView parallel:** Each `vtkPVDataRepresentation` renders within its own view's pipeline; camera parameters come from the view-specific `vtkSMRenderViewProxy`.

**Files:**
- Modify: `libs/CV_db/src/ecvHObject.cpp:1547-1559`

- [ ] **Step 1: Use context.display for camera parameters when drawing names**

Replace the name-drawing block in `ccHObject::draw()`:

```cpp
    if (shouldDrawName) {
        if (MACRO_Draw3D(context)) {
            ccBBox bBox = getBB_recursive(true);
            if (bBox.isValid()) {
                ccGLCameraParameters camera;
                // Use the draw context's display for per-view camera params
                // so the 3D→2D projection uses the correct view's matrices.
                if (context.display &&
                    context.display != ecvDisplayTools::TheInstance()) {
                    context.display->getGLCameraParameters(camera);
                } else {
                    ecvDisplayTools::GetGLCameraParameters(camera);
                }

                CCVector3 C = bBox.getCenter();
                camera.project(C, m_nameIn3DPos);
            }
        } else if (MACRO_Draw2D(context) && MACRO_Foreground(context)) {
            drawNameIn3D();
        }
    }
```

- [ ] **Step 2: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`
Expected: Clean compilation.

- [ ] **Step 3: Commit**

```bash
git add libs/CV_db/src/ecvHObject.cpp
git commit -m "fix(multiview): use context.display for per-view camera params in show-name projection"
```

---

### Task 5: Fix EditCameraTool per-view binding + wire buttons

**Why:** `EditCameraTool` stores the visualizer in process-wide statics (`s_viewer`, `s_camera`). `doActionEditCamera` constructs the tool with the primary `ecvDisplayTools::GetVisualizer3D()`. `ecvMultiViewFrameManager::createViewFrame` creates the "Adjust Camera" button but never wires it. When clicked on a secondary view's toolbar, nothing happens.

**ParaView parallel:** `pqCameraDialog` connects to `pqActiveObjects::viewChanged` signal and rebinds `pqPropertyLinks` to the new `vtkSMRenderViewProxy`.

**Files:**
- Modify: `libs/VtkEngine/Tools/CameraTools/EditCameraTool.cpp:80-102`
- Modify: `libs/VtkEngine/Tools/CameraTools/EditCameraTool.h`
- Modify: `app/ecvMultiViewFrameManager.cpp:86-91`
- Modify: `app/MainWindow.cpp:3576-3604`

- [ ] **Step 1: Make EditCameraTool rebind-safe (eliminate static s_viewer)**

In `EditCameraTool.cpp`, change the static variables to member variables:

```cpp
// Remove these statics:
// static vtkSmartPointer<vtkCamera> s_camera = nullptr;
// static Visualization::VtkVis* s_viewer = nullptr;

EditCameraTool::EditCameraTool(ecvGenericVisualizer3D* viewer)
    : ecvGenericCameraTool() {
    SetVisualizer(viewer);
    updateCameraParameters();
}

EditCameraTool::~EditCameraTool() {}

void EditCameraTool::SetVisualizer(ecvGenericVisualizer3D* viewer) {
    if (viewer) {
        m_viewer = reinterpret_cast<Visualization::VtkVis*>(viewer);
        if (!m_viewer) {
            CVLog::Warning("[EditCameraTool::setVisualizer] viewer is Null!");
        }
    } else {
        CVLog::Warning("[EditCameraTool::setVisualizer] viewer is Null!");
    }
}

void EditCameraTool::UpdateCameraInfo() {
    if (!m_viewer) {
        SetVisualizer(ecvDisplayTools::GetVisualizer3D());
    }

    m_camera = m_viewer->getVtkCamera();
    OldCameraParam = CurrentCameraParam;

    m_camera->GetViewUp(CurrentCameraParam.viewUp.u);
    m_camera->GetFocalPoint(CurrentCameraParam.focal.u);
    m_camera->GetPosition(CurrentCameraParam.position.u);
    m_camera->GetClippingRange(CurrentCameraParam.clippRange.u);
    CurrentCameraParam.viewAngle = m_camera->GetViewAngle();
    CurrentCameraParam.eyeAngle = m_camera->GetEyeAngle();
    m_viewer->getCenterOfRotation(CurrentCameraParam.pivot.u);
    CurrentCameraParam.rotationFactor = m_viewer->getRotationFactor();
}

void EditCameraTool::UpdateCamera() {
    if (!m_viewer) {
        SetVisualizer(ecvDisplayTools::GetVisualizer3D());
    }

    m_camera = m_viewer->getVtkCamera();

    m_camera->SetViewUp(CurrentCameraParam.viewUp.u);
    m_camera->SetFocalPoint(CurrentCameraParam.focal.u);
    // ... (rest of UpdateCamera uses m_camera and m_viewer instead of s_camera and s_viewer)
```

- [ ] **Step 2: Add member variables to EditCameraTool.h**

In `EditCameraTool.h`, add:

```cpp
private:
    vtkSmartPointer<vtkCamera> m_camera;
    Visualization::VtkVis* m_viewer = nullptr;
```

Also update any methods that used `s_camera` / `s_viewer` to use `m_camera` / `m_viewer`.

- [ ] **Step 3: Update doActionEditCamera to rebind on active view change**

In `MainWindow.cpp` `doActionEditCamera()`:

```cpp
void MainWindow::doActionEditCamera() {
    QMdiSubWindow* qWin = m_mdiArea->activeSubWindow();
    if (!qWin) return;

#ifdef USE_VTK_BACKEND
    // Determine which visualizer to bind: use the active ecvGLView if
    // available, otherwise fall back to the primary singleton.
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
        EditCameraTool* tool = new EditCameraTool(activeVis);
        m_cpeDlg->setCameraTool(tool);

        connect(m_mdiArea, &QMdiArea::subWindowActivated, m_cpeDlg,
                static_cast<void (ecvCameraParamEditDlg::*)(QMdiSubWindow*)>(
                        &ecvCameraParamEditDlg::linkWith));

        registerOverlayDialog(m_cpeDlg, Qt::BottomLeftCorner);
    } else {
        // Rebind to the current active view's visualizer
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

- [ ] **Step 4: Wire ecvMultiViewFrameManager's btnAdjustCamera**

In `ecvMultiViewFrameManager.cpp`, the `createViewFrame` method must connect the button. Add after the button creation (around line 91):

```cpp
    // Edit camera
    auto* editCamBtn =
            makeToolBtn(QIcon(":/Resources/images/svg/pqEditCamera.svg"),
                        tr("Adjust Camera"));
    editCamBtn->setObjectName("btnAdjustCamera");
    tbLayout->addWidget(editCamBtn);

    // Wire Edit Camera to activate this view and trigger the dialog
    connect(editCamBtn, &QToolButton::clicked, innerWidget, [this, innerWidget]() {
        // Find the ecvGLView that owns this frame's content widget
        auto* glView = findGLViewForWidget(innerWidget);
        if (glView) {
            ecvViewManager::instance().setActiveView(glView);
        }
        // Emit signal so MainWindow can handle camera dialog
        emit editCameraRequested(innerWidget);
    });
```

Add the signal and helper to `ecvMultiViewFrameManager.h`:

```cpp
signals:
    void editCameraRequested(QWidget* viewWidget);
```

- [ ] **Step 5: Connect the signal in MainWindow setup**

In `MainWindow` where the `ecvMultiViewFrameManager` is created, connect the signal:

```cpp
connect(m_viewFrameManager, &ecvMultiViewFrameManager::editCameraRequested,
        this, &MainWindow::doActionEditCamera);
```

- [ ] **Step 6: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`
Expected: Clean compilation.

- [ ] **Step 7: Commit**

```bash
git add libs/VtkEngine/Tools/CameraTools/EditCameraTool.cpp \
        libs/VtkEngine/Tools/CameraTools/EditCameraTool.h \
        app/ecvMultiViewFrameManager.cpp \
        app/ecvMultiViewFrameManager.h \
        app/MainWindow.cpp
git commit -m "fix(multiview): rebind EditCameraTool per-view, wire per-frame Adjust Camera button"
```

---

### Task 6: Per-view toolbar "activate-then-act" pattern

**Why:** Per-view toolbar buttons (3D/2D toggle, Capture Screenshot, Edit Camera) connect directly to MainWindow slots (`toggle3DView`, `doActionScreenShot`, `doActionEditCamera`) without first activating the owning view. If the user last clicked in view A, but then clicks the "Capture" button on view B's toolbar, the action executes on view A's pipeline.

**ParaView parallel:** `pqViewFrame` buttons always route through `pqActiveObjects::setActiveView(thisView)` before dispatching actions; `pqCameraDialog` rebinds via `pqActiveObjects::viewChanged`.

**Files:**
- Modify: `app/MainWindow.cpp:2686-2710`

- [ ] **Step 1: Create helper lambda to activate owning view before action**

In `MainWindow::createViewFrame`, add a helper that wraps toolbar button connections to first activate the view:

```cpp
    // Helper: activate the owning view before dispatching any per-view toolbar action.
    // This mirrors ParaView's pqViewFrame which always calls setActiveView first.
    auto activateViewAndDo = [this, innerWidget](auto slotFn) {
        return [this, innerWidget, slotFn]() {
            auto* display = ecvGenericGLDisplay::FromWidget(innerWidget);
            if (display) {
                auto& vm = ecvViewManager::instance();
                if (vm.getActiveView() != display) {
                    vm.setActiveView(display);
                    rebindToolsToActiveView(display);
                }
            }
            (this->*slotFn)();
        };
    };
```

- [ ] **Step 2: Update button connections to use activate-then-act pattern**

Replace the existing button connections (lines 2686-2708):

```cpp
    // 3D/2D toggle — per-view
    auto* view3DBtn =
            makeToolBtn(QIcon(":/Resources/images/3D3.png"), tr("3D View"));
    view3DBtn->setCheckable(true);
    view3DBtn->setChecked(true);
    connect(view3DBtn, &QToolButton::toggled, this, [this, innerWidget](bool state) {
        auto* display = ecvGenericGLDisplay::FromWidget(innerWidget);
        if (display) {
            auto& vm = ecvViewManager::instance();
            if (vm.getActiveView() != display) {
                vm.setActiveView(display);
                rebindToolsToActiveView(display);
            }
        }
        toggle3DView(state);
    });
    tbLayout->addWidget(view3DBtn);

    // Capture screenshot — per-view
    auto* captureBtn =
            makeToolBtn(QIcon(":/Resources/images/svg/pqCaptureScreenshot.svg"),
                        tr("Capture Screenshot"));
    connect(captureBtn, &QToolButton::clicked, this, activateViewAndDo(&MainWindow::doActionScreenShot));
    tbLayout->addWidget(captureBtn);

    // Edit camera — per-view
    auto* editCamBtn =
            makeToolBtn(QIcon(":/Resources/images/svg/pqEditCamera.svg"),
                        tr("Adjust Camera"));
    connect(editCamBtn, &QToolButton::clicked, this, activateViewAndDo(&MainWindow::doActionEditCamera));
    tbLayout->addWidget(editCamBtn);
```

- [ ] **Step 3: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`
Expected: Clean compilation.

- [ ] **Step 4: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): per-view toolbar buttons activate owning view before action dispatch"
```

---

### Task 7: Fix DrawClickableItems viewport dimensions

**Why:** `DrawClickableItems` uses `effectiveCtx().glViewport.width()` and `.height()` for layout calculations (positioning clickable hot-zone items). After Task 1 adds `ScopedRenderOverride` to `ecvGLView::redraw`, `effectiveCtx()` will already resolve to the correct view. But `DrawClickableItems` must also handle the case where it's called outside a scoped guard (e.g. during idle update).

**Files:**
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (DrawClickableItems function)

- [ ] **Step 1: Verify DrawClickableItems uses effectiveCtx consistently**

After Task 1 is complete, `DrawClickableItems` should already work correctly because `ScopedRenderOverride` ensures `effectiveCtx()` returns the right context during any per-view draw. Verify by reading the function and confirming all viewport references go through `effectiveCtx()`.

If any hardcoded `s_tools.instance->m_primaryCtx` references exist, replace with `s_tools.instance->effectiveCtx()`.

- [ ] **Step 2: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`
Expected: Clean compilation.

- [ ] **Step 3: Commit (if changes needed)**

```bash
git add libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "fix(multiview): ensure DrawClickableItems uses effectiveCtx for viewport dimensions"
```

---

### Task 8: Fix doActionScreenShot to render from active view

**Why:** `doActionScreenShot` calls `ecvDisplayTools::RenderToFile` which renders from the singleton's current pipeline. After Task 6's `activateViewAndDo` pattern, the singleton will be properly rebound to the active view before rendering. But `RenderToFile` may use cached dimensions from the primary. This task ensures the capture size matches the target view.

**Files:**
- Modify: `app/MainWindow.cpp:3676-3688`

- [ ] **Step 1: Use active view dimensions for screenshot dialog**

```cpp
void MainWindow::doActionScreenShot() {
    QWidget* win = getActiveWindow();
    if (!win) return;

    // Use the active view's widget for accurate dimensions
    auto* activeView = dynamic_cast<ecvGLView*>(
            ecvViewManager::instance().getActiveView());
    QWidget* sizeSource = win;
    if (activeView && activeView->getVtkWidget()) {
        sizeSource = activeView->getVtkWidget();
    }

    ccRenderToFileDlg rtfDlg(static_cast<unsigned>(sizeSource->width()),
                             static_cast<unsigned>(sizeSource->height()), this);

    if (rtfDlg.exec()) {
        QApplication::processEvents();
        ecvDisplayTools::RenderToFile(rtfDlg.getFilename(), rtfDlg.getZoom(),
                                      rtfDlg.dontScalePoints(),
                                      rtfDlg.renderOverlayItems());
    }
}
```

- [ ] **Step 2: Build and verify**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && make -j48`
Expected: Clean compilation.

- [ ] **Step 3: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "fix(multiview): use active view dimensions for screenshot dialog"
```

---

### Task 9: Full compilation + smoke test

- [ ] **Step 1: Full rebuild**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && conda activate cloudViewer && make -j48`
Expected: 0 errors, 0 new warnings.

- [ ] **Step 2: Launch application**

Run: `cd /Users/asher/develop/code/github/ACloudViewer/build_app && ./bin/ACloudViewer`
Expected: Application launches without crash.

- [ ] **Step 3: Verify multi-window scenarios**

Manual verification checklist:
1. Split a view horizontally → both panes should be 50/50, resizable via splitter handle
2. Load an object → widgets/overlays should scale correctly in each sub-window
3. Enable "show name" → name should only appear in the window where the object is displayed
4. Click "Adjust Camera" on a secondary view toolbar → camera dialog should control that view's camera
5. Click "Capture Screenshot" on a secondary view toolbar → screenshot captures that view (not primary)
6. Click "3D View" toggle on a secondary view toolbar → 2D/3D mode toggles for that specific view
7. Use selection tools (rubber band, point picking) → they should target the view where the tool was activated
8. Plugin interactions should target the active view

---

## Summary of Root Causes → Fixes

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| 1. Widget scale | `effectiveCtx()` resolves to wrong view during direct `ecvGLView::redraw` | Task 1: `ScopedRenderOverride` in `ecvGLView::redraw` |
| 1b. DPR mismatch | `ecvGLView::getGLCameraParameters` uses logical px, primary uses device px | Task 2: Apply DPR scaling in secondary view camera params |
| 2. Show name leak | `DisplayText`/`RenderText` ignore `display` param | Task 3: Route Y-flip through `viewportHeightFor(display)` |
| 2b. Wrong camera | `GetGLCameraParameters` uses global effective view for name projection | Task 4: Use `context.display->getGLCameraParameters` |
| 3. Edit Camera | Static `s_viewer`; button not wired | Task 5: Instance members; wire button |
| 4. Toolbar routing | Per-view buttons don't activate their owning view before action | Task 6: `activateViewAndDo` pattern |
| 5. HotZone layout | `DrawClickableItems` may use wrong viewport during non-scoped draws | Task 7: Verify effectiveCtx usage |
| 6. Screenshot size | `doActionScreenShot` uses wrong widget dimensions | Task 8: Use active view widget for size |
