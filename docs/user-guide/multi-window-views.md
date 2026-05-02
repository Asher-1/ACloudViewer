# ACloudViewer Multi-Window 3D Views — Architecture & Implementation Reference

> Date: 2026-04-30
> Version: 5.2 (Phase M completed + Phase N completed — effectiveCtx elimination done)
>
> **Status (v5.2)**: Phase M (M1-M5) and Phase N (N1-N5) completed. VtkDisplayTools is now a pure engine service, ecvGLView is the sole view type, effectiveCtx() calls reduced from 307 to 76 (acceptable patterns only). Remaining: M6 + Phase O.
> - `Init()`/`TheInstance()`/`HasInstance()`/`ReleaseInstance()` are gone from public API
> - `ecvViewManager` now owns the shared `ecvDisplayTools` instance lifecycle
> - `ecvGLView` routes through `m_displayTools` (typed as `VtkDisplayTools*`) instead of static singleton calls
> - Nested types (`HotZone`, `MessageToDisplay`, etc.) extracted to `ecvDisplayTypes.h`
> - 12 `ecvViewManager::shared*()` forwarders replace all non-core `ecvDisplayTools::` calls
> - See [singleton-removal-migration-plan.md](singleton-removal-migration-plan.md) for full changelog
> - See [multi-window-refactor-roadmap-Vtk-vs-CC.md](multi-window-refactor-roadmap-Vtk-vs-CC.md) §10 for Phase M TODOs (M1–M6, all ✅)
> - See [multi-window-paraview-alignment-design.md](multi-window-paraview-alignment-design.md) for the ParaView ↔ ACloudViewer full alignment design (96.7% aligned, Phase M–O complete)
>
> **Note**: Some code examples below still reference `TheInstance()` patterns from the pre-migration architecture. These are preserved as historical context for understanding the design evolution.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Sequence Diagrams](#sequence-diagrams)
4. [Core Classes & Source Locations](#core-classes--source-locations)
  - [Core Classes \& Source Locations](#core-classes--source-locations)
    - [1. ecvViewManager](#1-ecvviewmanager)
      - [Class Declaration](#class-declaration)
      - [Public Methods](#public-methods)
      - [ScopedRenderOverride (RAII)](#scopedrenderoverride-raii)
      - [Private Members](#private-members)
      - [Signals](#signals)
    - [2. ecvRepresentationManager](#2-ecvrepresentationmanager)
      - [Class Declaration](#class-declaration-1)
      - [Public Methods](#public-methods-1)
      - [Internal Data Structures](#internal-data-structures)
      - [Cleanup Callback Mechanism](#cleanup-callback-mechanism)
      - [Signals](#signals-1)
    - [3. ecvViewRepresentation](#3-ecvviewrepresentation)
      - [Class Declaration](#class-declaration-2)
      - [RenderMode Enum](#rendermode-enum)
      - [Properties Struct](#properties-struct)
      - [Methods](#methods)
      - [`effectivePointSize()` Fallback Chain](#effectivepointsize-fallback-chain)
    - [4. ecvGLView](#4-ecvglview)
      - [Factory: `ecvGLView::Create()`](#factory-ecvglviewcreate)
      - [`initVtkPipeline()`](#initvtkpipeline)
      - [Destructor](#destructor)
      - [`redraw()` — The Core Render Loop](#redraw--the-core-render-loop)
    - [5. QVTKWidgetCustom](#5-qvtkwidgetcustom)
      - [Active View on Mouse Interaction](#active-view-on-mouse-interaction)
    - [6. VtkVis — Text/Caption Rendering](#6-vtkvis--textcaption-rendering)
      - [`addText()`](#addtext)
      - [`updateText()`](#updatetext)
      - [`addCaption()`](#addcaption)
      - [`updateCaption()`](#updatecaption)
    - [7. VtkDisplayTools](#7-vtkdisplaytools)
      - [`resolveVisualizer(display)`](#resolvevisualizerdisplay)
      - [Actor Cleanup Callback Registration](#actor-cleanup-callback-registration)
    - [8. ecvDisplayTools](#8-ecvdisplaytools)
      - [`UpdateScreen()`](#updatescreen)
  - [Entity Visibility \& View Isolation](#entity-visibility--view-isolation)
    - [`ccHObject::isDisplayedIn()`](#cchobjectisdisplayedin)
      - [Three-Way Logic](#three-way-logic)
    - [ScopedRenderOverride](#scopedrenderoverride)
  - [Draw Pipeline](#draw-pipeline)
    - [`ccHObject::draw()`](#cchobjectdraw)
    - [Per-View Representation Wiring](#per-view-representation-wiring)
    - [`ecvGLView::redraw()`](#ecvglviewredraw)
  - [cc2DLabel Rendering](#cc2dlabel-rendering)
    - [`cc2DLabel::drawMeOnly()`](#cc2dlabeldrawmeonly)
    - [ABC Legend Text (DisplayText)](#abc-legend-text-displaytext)
    - [Caption Box (DrawWidgets)](#caption-box-drawwidgets)
    - [Font Size Adaptive Algorithm](#font-size-adaptive-algorithm)
  - [VTK Caption Widget System](#vtk-caption-widget-system)
    - [`VtkVis::addCaption()`](#vtkvisaddcaption)
    - [`VtkVis::updateCaption()`](#vtkvisupdatecaption)
    - [Caption Box Sizing Algorithm](#caption-box-sizing-algorithm)
    - [`VtkVis::addText()` / `updateText()`](#vtkvisaddtext--updatetext)
  - [Entity Binding \& Load Path](#entity-binding--load-path)
    - [`MainWindow::addToDB()`](#mainwindowaddtodb)
    - [Zoom Behavior](#zoom-behavior)
    - [`zoomOn()` — Active View Sync](#zoomon--active-view-sync)
    - [`zoomOnEntities()` — Same Pattern](#zoomonentities--same-pattern)
  - [VTK Actor Lifecycle \& Cleanup](#vtk-actor-lifecycle--cleanup)
    - [Creation Flow](#creation-flow)
    - [Destruction Flow (View Close)](#destruction-flow-view-close)
    - [Destruction Flow (Entity Delete)](#destruction-flow-entity-delete)
  - [How to Use](#how-to-use)
    - [1. Creating a New 3D View](#1-creating-a-new-3d-view)
    - [2. View Title Bar (ParaView pqViewFrame style)](#2-view-title-bar-paraview-pqviewframe-style)
    - [3. Switching Active View](#3-switching-active-view)
    - [4. Moving Entities Between Views](#4-moving-entities-between-views)
    - [5. Closing a View](#5-closing-a-view)
  - [Programmatic API](#programmatic-api)
    - [Creating a new view](#creating-a-new-view)
    - [Moving an entity to a view](#moving-an-entity-to-a-view)
    - [Querying views](#querying-views)
    - [Per-view representation](#per-view-representation)
    - [Connecting to signals](#connecting-to-signals)
  - [FAQ](#faq)
  - [Known Issues \& Future Work](#known-issues--future-work)

---

## Overview

ACloudViewer supports **multiple independent 3D view windows**, each with its own VTK rendering pipeline (renderer, render window, interactor), viewport camera, and entity visibility settings. The design follows **ParaView's `pqActiveObjects` + per-view representation** pattern.

Key capabilities:

- View the same scene from different camera angles simultaneously
- Show/hide specific entities per view
- Apply different display properties (opacity, point size, render mode) per view
- Work with a primary + N secondary views inside the MDI area
- Newly loaded entities bind to the **active** view (not all views)
- Each view's 2D labels (ABC text, captions) render independently with DPI-aware sizing

---

## Architecture Diagram

### System-Level Layout

```
┌──────────────────────────────────────────────────────────┐
│                     MainWindow (QMainWindow)             │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                   QMdiArea                          │ │
│  │                                                     │ │
│  │  ┌──────────────────┐  ┌──────────────────┐         │ │
│  │  │   Primary View   │  │   New 3D View    │  ...    │ │
│  │  │                  │  │                  │         │ │
│  │  │ ecvDisplayTools  │  │   ecvGLView      │         │ │
│  │  │ (singleton)      │  │   (per-window)   │         │ │
│  │  │                  │  │                  │         │ │
│  │  │ ┌──────────────┐ │  │ ┌──────────────┐ │         │ │
│  │  │ │QVTKWidget    │ │  │ │QVTKWidget    │ │         │ │
│  │  │ │  vtkRenderer │ │  │ │  vtkRenderer │ │         │ │
│  │  │ │  vtkRenWin   │ │  │ │  vtkRenWin   │ │         │ │
│  │  │ │  VtkVis      │ │  │ │  VtkVis      │ │         │ │
│  │  │ └──────────────┘ │  │ └──────────────┘ │         │ │
│  │  └──────────────────┘  └──────────────────┘         │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌───────────────┐  ┌───────────────────────────┐        │
│  │  DB Tree      │  │  Properties Panel         │        │
│  │  (ccDBRoot)   │  │  ccPropertiesTreeDelegate │        │
│  │               │  │   Current Display: [___]  │        │
│  │  Right-click: │  │                           │        │
│  │  "Move to     │  │                           │        │
│  │   View"       │  │                           │        │
│  └───────────────┘  └───────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### Class Relationships

```
                     ┌──────────────────────┐
                     │  ecvGenericGLDisplay  │ (interface)
                     └──────────┬───────────┘
                                │ implements
                     ┌──────────┴───────────┐
                     │                      │
                     ▼                      ▼
              ecvDisplayTools          ecvGLView
              (primary window)         (per-window)
              singleton                owns its own VtkVis
                     │                      │
                     └──────┬───────────────┘
                            │ registered in
                            ▼
                  ┌─────────────────────┐
                  │  ecvViewManager     │
                  │  (singleton)        │
                  │                     │
                  │  m_activeView       │──▶ which view gets user input
                  │  m_renderingView    │──▶ which view is painting now
                  │  m_views[]          │──▶ all registered views
                  │  m_activeSource     │──▶ selected ccHObject
                  │  m_layouts[]        │──▶ layout proxies
                  └─────────────────────┘
                            │ queries
                            ▼
              ┌─────────────────────────┐     ┌───────────────────────┐
              │ ecvRepresentationManager│────▶│ ecvViewRepresentation │
              │ (singleton)             │     │                       │
              │                         │     │  entity : ccHObject*  │
              │  Key: (entity, view)    │     │  view   : display*    │
              │  QHash storage          │     │  visibility override  │
              │  QReadWriteLock         │     │  opacity, pointSize   │
              │  CleanupCallback        │     │  renderMode, ...      │
              └─────────────────────────┘     └───────────────────────┘
```

---

## Sequence Diagrams

### 1. Secondary View Redraw (ecvGLView::redraw)

```
ecvGLView::redraw(only2D=false, forceRedraw=true)
    │
    ├──▶ ScopedRenderOverride viewGuard(this)
    │        └── m_renderingView = this                   [ecvGLView.cpp:119]
    │
    ├──▶ Sync glViewport from widget                      [ecvGLView.cpp:124-127]
    │        context.glW = m_vtkWidget->width()
    │        context.glH = m_vtkWidget->height()
    │
    ├──▶ getContext(context)                               [ecvGLView.cpp:131]
    │        └── context.display = this                   [ecvGLView.cpp:349]
    │
    ├──▶ Background setup                                  [ecvGLView.cpp:134-161]
    │
    ├──▶ [3D Pass] m_globalDBRoot->draw(context)           [ecvGLView.cpp:166]
    │    │
    │    └──▶ ccHObject::draw(context)                     [ecvHObject.cpp:1445]
    │         │
    │         ├── isDisplayedIn(context.display) ?          [ecvHObject.cpp:1467]
    │         │    └── getEffectiveView() returns THIS view (via ScopedRenderOverride)
    │         │    └── m_currentDisplay matches THIS view → draw = true
    │         │
    │         ├── ensureRepresentation(entity, display)     [ecvHObject.cpp:1473]
    │         │    └── ecvRepresentationManager creates/finds (entity, view) pair
    │         │
    │         ├── Check visibility override from rep         [ecvHObject.cpp:1479-1481]
    │         │
    │         └── drawMeOnly(context)                       [ecvHObject.cpp:~1516]
    │              └── VtkDisplayTools::draw*()
    │                   └── resolveVisualizer(context.display) → ecvGLView's VtkVis
    │                        └── Add/update actors in THIS view's VtkVis
    │
    ├──▶ [3D Pass] m_winDBRoot->draw(context)              [ecvGLView.cpp:170]
    │
    ├──▶ [2D Pass] m_globalDBRoot->draw(context)           [ecvGLView.cpp:176]
    │    │
    │    └──▶ cc2DLabel::drawMeOnly(context)               [ecv2DLabel.cpp:868]
    │         ├── isRedraw() || context.forceRedraw → proceed
    │         ├── myDisp == context.display → proceed
    │         ├── drawMeOnly3D(context) → 3D markers/widgets
    │         └── drawMeOnly2D(context) → ABC text + caption box
    │              ├── DisplayText("A",...,legendId,context.display)  [ecv2DLabel.cpp:1274]
    │              │    └── VtkVis::addText/updateText (via legendId)
    │              └── DrawWidgets(param) → caption widget             [ecv2DLabel.cpp:1739]
    │                   └── VtkVis::addCaption/updateCaption
    │
    ├──▶ [2D Pass] m_winDBRoot->draw(context)              [ecvGLView.cpp:180]
    │
    ├──▶ ScopedHotZoneRender                               [ecvGLView.cpp:186-193]
    │    ├── Save primary pipeline state
    │    ├── Swap to THIS view's VtkVis/widget
    │    ├── DrawClickableItems()
    │    └── ~ScopedHotZoneRender: restore primary state
    │
    └──▶ ~ScopedRenderOverride: m_renderingView = saved
```

### 2. Active View Switching (User Clicks Different View)

```
User clicks on secondary view's QVTKWidgetCustom
    │
    ├──▶ QVTKWidgetCustom::mousePressEvent()               [QVTKWidgetCustom.cpp:548]
    │    ├── display = ecvGenericGLDisplay::FromWidget(this) [s_displayRegistry lookup]
    │    └── ecvViewManager::setActiveView(display)          [ecvViewManager.cpp:39]
    │         ├── m_activeView = display
    │         ├── updateActiveRepresentation()
    │         └── triggerSignals()
    │              └── emit activeViewChanged(newActive, oldActive) [ecvViewManager.cpp:82]
    │
    └──▶ Signal received by MainWindow                      [MainWindow.cpp:728-743]
         │
         ├── rebindToolsToActiveView(newActive)             [MainWindow.cpp:2629-2647]
         │    │
         │    ├── [secondary view?]
         │    │    └── VtkDisplayTools::switchActiveView(     [VtkDisplayTools.cpp:131]
         │    │            glView->getVisualizer3DSP(),
         │    │            glView->getVtkWidget())
         │    │         ├── Save primary pipeline (m_primaryVis/Widget)
         │    │         ├── Reconnect picking signal to new VtkVis
         │    │         ├── m_visualizer3D = secondary VtkVis
         │    │         ├── m_vtkWidget = secondary widget
         │    │         └── SetCurrentScreen(widget)
         │    │
         │    └── [primary view?]
         │         └── VtkDisplayTools::restorePrimaryView()
         │              ├── m_visualizer3D = m_primaryVis
         │              ├── m_vtkWidget = m_primaryWidget
         │              └── SetCurrentScreen(m_primaryWidget)
         │
         ├── m_pickingHub->onActiveViewWidgetChanged(widget)
         │
         ├── markActiveViewFrame(widget)
         │    └── Update colored border + bold title on active view
         │
         └── updateMenus()
```

### 3. Object Loading (File → addToDB → View Binding)

```
User: File → Open → selects "bunny.ply"
    │
    ├──▶ FileIOFilter::LoadFromFile("bunny.ply")
    │    └── Returns ccHObject* (point cloud / mesh)
    │
    ├──▶ MainWindow::addToDB(obj, updateZoom=true)          [MainWindow.cpp:~3700]
    │    │
    │    ├── Insert into global DB tree                      [MainWindow.cpp:~3720]
    │    │    └── m_ccRoot->addElement(obj, autoExpand)
    │    │
    │    ├── Bind to active view                             [MainWindow.cpp:3741]
    │    │    └── ecvViewManager::associateToActiveView(obj) [ecvViewManager.cpp:219]
    │    │         └── obj->setDisplay_recursive(activeView)
    │    │              └── Now obj->m_currentDisplay = activeView
    │    │
    │    ├── Zoom to fit                                     [MainWindow.cpp:3743-3752]
    │    │    ├── [active == primary?]
    │    │    │    └── ecvDisplayTools::ZoomGlobal()
    │    │    └── [active == secondary?]
    │    │         └── ecvGLView::zoomGlobal()
    │    │
    │    └── Trigger redraw
    │         └── ecvDisplayTools::InvalidateViewport()
    │              └── RedrawDisplay()
    │                   ├── Primary view draws → isDisplayedIn(primary) → false (bound to secondary)
    │                   └── Secondary view draws → isDisplayedIn(secondary) → true → visible!
    │
    └── Entity appears ONLY in the active (secondary) view
```

### 4. View Creation & Destruction Lifecycle

```
═══════════════ VIEW CREATION ═══════════════

MainWindow::new3DView()
    │
    ├──▶ ecvGLView::Create(mainWindow)                     [ecvGLView.cpp:69]
    │    │
    │    ├── new ecvGLView(parent)
    │    │
    │    ├── initVtkPipeline()                              [ecvGLView.cpp:82]
    │    │    ├── m_vtkWidget = new QVTKWidgetCustom(...)
    │    │    ├── m_vtkWidget->setOwnerView(this)
    │    │    ├── Create vtkRenderer + vtkGenericOpenGLRenderWindow
    │    │    ├── m_visualizer3D = make_shared<VtkVis>(...)
    │    │    ├── m_vtkWidget->SetRenderWindow(renderWindow)
    │    │    └── m_visualizer3D->initialize()
    │    │
    │    ├── m_winDBRoot = new ccHObject("DB.GLView_<ID>")
    │    │
    │    ├── RegisterGLDisplay(m_vtkWidget, view)           [ecvGLView.cpp:76]
    │    │    └── s_displayRegistry[widget] = view
    │    │
    │    └── ecvViewManager::registerView(view)              [ecvGLView.cpp:77]
    │         ├── m_views.append(view)
    │         ├── emit viewRegistered(view)
    │         └── emit viewCountChanged(count)
    │
    ├──▶ VtkCameraLink::addView(vis)                        [if link enabled]
    │
    └──▶ view->setSceneDB(globalDBRoot)


═══════════════ VIEW DESTRUCTION ═══════════════

ecvGLView::~ecvGLView()                                    [ecvGLView.cpp:42]
    │
    ├──▶ emit aboutToClose(this)                            [line 43]
    │
    ├──▶ m_globalDBRoot->removeFromDisplay_recursive(this)   [line 46]
    │    └── For each entity bound to this view: setDisplay(nullptr)
    │
    ├──▶ ecvRepresentationManager::removeRepresentationsForView(this) [line 49]
    │    │
    │    └── For each (entity, this) pair in QHash:
    │         │
    │         ├── m_actorCleanup(entity, this)               [CleanupCallback]
    │         │    └── VtkDisplayTools lambda:                [VtkDisplayTools.cpp:110-120]
    │         │         ├── viewID = entity->getViewId()
    │         │         ├── vis = resolveVisualizer(this)     → this view's VtkVis
    │         │         ├── vis->removePointCloud(viewID)
    │         │         ├── vis->removePolygonMesh(viewID)
    │         │         └── vis->removeShape(viewID)
    │         │
    │         └── m_representations.erase(key)
    │
    ├──▶ ecvViewManager::unregisterView(this)                [line 51]
    │    ├── m_views.removeOne(this)
    │    ├── [was active?] → setActiveView(next available)
    │    ├── emit viewUnregistered(this)
    │    └── emit viewCountChanged(count)
    │
    ├──▶ UnregisterGLDisplay(m_vtkWidget)                    [line 54]
    │    └── s_displayRegistry.remove(widget)
    │
    └──▶ delete m_hotZone, m_rectPickingPoly, m_winDBRoot
```

### 5. cc2DLabel ABC Text Rendering Flow

```
ecvGLView::redraw() → m_globalDBRoot->draw(context) → ... → cc2DLabel::draw(context)
    │
    ├──▶ ccHObject::draw(context)                           [ecvHObject.cpp:1445]
    │    └── drawInThisContext = isDisplayedIn(context.display) = true
    │
    └──▶ cc2DLabel::drawMeOnly(context)                     [ecv2DLabel.cpp:868]
         │
         ├── Guard: m_pickedPoints.empty() → return
         ├── Guard: !MACRO_Foreground → return
         ├── Guard: !isRedraw() && !context.forceRedraw → return    [line 881]
         ├── Guard: myDisp != context.display → return              [line 886]
         │
         ├──▶ [3D Pass] drawMeOnly3D(context)               [ecv2DLabel.cpp:896]
         │    │
         │    ├── For each picked point:
         │    │    ├── Project 3D → 2D:  camera.project(P3D, pos2D)
         │    │    └── Store in m_pickedPoints[j].pos2D
         │    │
         │    ├── Draw 3D markers (sphere/cross)              [line ~1107-1114]
         │    │    └── DrawWidgets(WIDGET_SPHERE/POINT)
         │    │
         │    └── Draw 3D leader line if 2-point label        [line ~1064-1068]
         │
         └──▶ [2D Pass] drawMeOnly2D(context)               [ecv2DLabel.cpp:1197]
              │
              ├── Get camera parameters                       [line 1227-1236]
              │    └── If context.display != primary → get from context.display
              │
              ├── Project 3D points → 2D screen coordinates   [line ~1240-1250]
              │
              ├── ═══ ABC LEGEND TEXT ═══                     [line 1260-1284]
              │    │
              │    │  For each point j (0..count-1):
              │    │
              │    ├── title = "A" / "B" / "C" (or "P#idx")   [line 1262-1268]
              │    │
              │    ├── legendId = "{viewId}_legend_{j}"         [line 1270-1272]
              │    │    └── Unique ID enables updateText() (no recreation)
              │    │
              │    └── ecvDisplayTools::DisplayText(             [line 1274-1283]
              │              title,
              │              pos2D.x + markerTextShift,
              │              pos2D.y + markerTextShift,
              │              ALIGN_DEFAULT,
              │              0.55f,                              ← semi-transparent background
              │              white, &font, legendId,
              │              context.display)                    ← routes to correct VtkVis
              │         │
              │         └──▶ VtkDisplayTools::DisplayText()
              │              └── resolveVisualizer(context.display) → correct VtkVis
              │                   ├── [actor exists?] VtkVis::updateText(text, x, y, legendId)
              │                   └── [new?] VtkVis::addText(text, x, y, fontSize, r, g, b, legendId)
              │
              ├── ═══ LABEL BODY (title, measurements) ═══   [line 1291-1601]
              │    └── Build m_labelROI from QFontMetrics, margins, tab content
              │
              └── ═══ CAPTION BOX ═══                         [line 1666-1739]
                   │
                   ├── Build WIDGETS_PARAMETER param
                   │    ├── type = WIDGET_CAPTION
                   │    ├── center = centroid of picked points (3D anchor)
                   │    ├── pos = from m_labelROI (2D position)
                   │    ├── text = m_historyMessage.join("\n")
                   │    └── fontSize = adaptive calculation:
                   │         refDim = min(logicalW, logicalH)       [line 1734]
                   │         fontSize = clamp(refDim/42, 10, 28)    [line 1735-1736]
                   │
                   └── ecvDisplayTools::DrawWidgets(param)    [line 1739]
                        │
                        └──▶ VtkDisplayTools → resolveVisualizer(context.display)
                             ├── [exists?] VtkVis::updateCaption(...)  [VtkVis.cpp:1903]
                             │    ├── SetAnchorPosition(3D)            [line 1924]
                             │    ├── SetPosition(pos2D/winSize)       [line 1930]
                             │    ├── SetPosition2(baseW, captionH)    [line 1941]
                             │    │    └── adaptive sizing algorithm
                             │    └── Update text, font, background
                             │
                             └── [new?] VtkVis::addCaption(...)        [VtkVis.cpp:1994]
                                  ├── Create vtkCaptionRepresentation
                                  ├── SetAnchorPosition + SetPosition + SetPosition2
                                  ├── Create CustomVtkCaptionWidget
                                  └── Store in m_widget_map
```

### 6. Camera Link Synchronization

```
User rotates camera in View A (VtkCameraLink enabled)
    │
    ├──▶ VTK interaction → vtkRenderWindow::Render()
    │
    ├──▶ vtkRenderWindow fires EndEvent
    │
    ├──▶ VtkCameraLink::OnRenderEnd(caller, ...)            [VtkCameraLink.cpp:~111]
    │    │
    │    ├── Find source VtkVis from caller (vtkRenderWindow*)
    │    │
    │    ├── Check m_updating guard → false → proceed
    │    │
    │    └── syncCamerasFrom(sourceVis)                      [VtkCameraLink.cpp:~135]
    │         │
    │         ├── m_updating = true                          ← re-entry guard
    │         │
    │         ├── Get source camera: position, focal, viewUp,
    │         │   clipping, viewAngle, parallelScale, parallelProjection
    │         │
    │         ├── For each OTHER LinkedView (not source):
    │         │    │
    │         │    ├── Set target camera = source camera params
    │         │    │    ├── SetPosition, SetFocalPoint, SetViewUp
    │         │    │    ├── SetClippingRange, SetViewAngle
    │         │    │    ├── SetParallelScale, SetParallelProjection
    │         │    │    └── Copy center of rotation
    │         │    │
    │         │    └── target->getRenderWindow()->Render()
    │         │         └── EndEvent fires again BUT m_updating=true → skip
    │         │
    │         └── m_updating = false                         ← release guard
    │
    └── All views now show same camera angle
```

### 7. "Move to View" (DB Tree Right-Click)

```
User right-clicks entity in DB Tree → "Move to View" → "3D View 1001"
    │
    ├──▶ Context menu built                                  [ecvDBRoot.cpp:2462-2478]
    │    ├── Check viewCount() > 1
    │    ├── Add "None (All Views)" action
    │    └── For each view: add action with view->getTitle()
    │
    ├──▶ ccDBRoot::moveSelectedToView(view, indexes)         [ecvDBRoot.cpp:2493-2502]
    │    │
    │    ├── For each selected entity:
    │    │    └── item->setDisplay_recursive(view)
    │    │         └── Recursively sets m_currentDisplay on entity and children
    │    │
    │    ├── ecvViewManager::redrawAll()
    │    │    └── All views redraw:
    │    │         ├── Old view: isDisplayedIn() = false → entity disappears
    │    │         └── New view: isDisplayedIn() = true → entity appears
    │    │
    │    └── updatePropertiesView()
    │         └── Properties panel "Current Display" dropdown updates
    │
    └── Entity now visible only in "3D View 1001"
```

### 8. View Close & Cleanup

```
User clicks ✕ on secondary view "3D View 1001"
    │
    ├──▶ Qt closeEvent triggers ecvGLView destructor
    │
    ├──▶ ecvGLView::~ecvGLView()                            [ecvGLView.cpp:42-67]
    │    │
    │    ├──▶ emit aboutToClose(this)
    │    │    └── MainWindow::onViewClosing(view) [if connected]
    │    │         └── VtkCameraLink::removeView(vis)
    │    │
    │    ├──▶ m_globalDBRoot->removeFromDisplay_recursive(this)
    │    │    └── All entities bound to THIS view:
    │    │         m_currentDisplay = nullptr (now visible in remaining view)
    │    │
    │    ├──▶ RepresentationManager::removeRepresentationsForView(this)
    │    │    │
    │    │    └── For EACH (entity, this) representation:
    │    │         │
    │    │         ├── CleanupCallback(entity, this)
    │    │         │    └── VtkDisplayTools lambda:
    │    │         │         ├── resolveVisualizer(this) → this view's VtkVis
    │    │         │         ├── removePointCloud(viewID)
    │    │         │         ├── removePolygonMesh(viewID)
    │    │         │         └── removeShape(viewID)
    │    │         │
    │    │         ├── emit representationRemoved(entity, this)
    │    │         └── m_representations.erase(key)
    │    │
    │    ├──▶ ViewManager::unregisterView(this)
    │    │    ├── m_views.removeOne(this)
    │    │    ├── [was active?] setActiveView(remaining view)
    │    │    │    └── emit activeViewChanged → rebindToolsToActiveView
    │    │    ├── emit viewUnregistered(this)
    │    │    └── emit viewCountChanged(count - 1)
    │    │
    │    ├──▶ UnregisterGLDisplay(m_vtkWidget)
    │    │
    │    └──▶ Delete: m_hotZone, m_rectPickingPoly, m_winDBRoot
    │
    └── Entities now visible in remaining primary view (m_currentDisplay == nullptr)
```

---

## Core Classes & Source Locations

### 1. ecvViewManager

> **ParaView equivalent**: `pqActiveObjects` (active view/source tracking) + `pqServerManagerModel` (view registration)

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecvViewManager.h` |
| **Implementation** | `libs/CV_db/src/ecvViewManager.cpp` |

#### Class Declaration

- **Line**: `ecvViewManager.h:33` — `class CV_DB_LIB_API ecvViewManager : public QObject`

#### Public Methods

| Method | Header Line | Description |
|--------|-------------|-------------|
| `instance()` | `:37` | Singleton accessor |
| `getActiveView()` | `:43` | Returns the UI-active view (where user last clicked) |
| `setActiveView(view)` | `:44` | Changes active view; emits `activeViewChanged` |
| `getEffectiveView()` | `:65` | Returns `m_renderingView` if set, else `m_activeView`. Used by `isDisplayedIn()` during rendering |
| `activeSource()` | `:71` | Returns the currently selected `ccHObject` in DB tree |
| `setActiveSource(source)` | `:72` | Sets active source; updates active representation |
| `activeRepresentation()` | `:74` | Returns `ecvViewRepresentation` for (activeSource, activeView) |
| `registerView(view)` | `:80` | Adds view to `m_views`; emits `viewRegistered` + `viewCountChanged` |
| `unregisterView(view)` | `:81` | Removes view; if it was active, picks next available; emits signals |
| `registerLayout(layout)` | `:87` | Registers a `ecvViewLayoutProxy` |
| `unregisterLayout(layout)` | `:88` | Unregisters a layout proxy |
| `allLayouts()` | `:89` | Returns all registered layout proxies |
| `activeLayout()` | `:90` | Returns the layout containing the active view |
| `getAllViews()` | `:96` | Returns `QList<ecvGenericGLDisplay*>` of all views |
| `viewCount()` | `:97` | Returns number of registered views |
| `findView(uniqueID)` | `:98` | Finds view by its `getUniqueID()` |
| `findViewForEntity(entity)` | `:99` | Finds the view that `entity->getDisplay()` points to |
| `refreshAll(only2D)` | `:105` | Calls `update()` on each view's widget |
| `redrawAll(only2D, forceRedraw, includePrimary)` | `:106-108` | Full redraw of all views |
| `saveLayout(geometryOf)` | `:116` | Serializes layout to JSON |
| `restoreLayout(layout, apply)` | `:120` | Restores layout from JSON |
| `associateToActiveView(obj)` | `:126` | Binds entity tree to active view recursively via `setDisplay_recursive()` |
| `detachEntitiesFromView(view)` | `:127` | Resets all entities bound to `view` to `nullptr` |
| `reassignEntitiesFromView(root, from, to)` | `:128-130` | Moves entities from one view to another |

#### ScopedRenderOverride (RAII)

- **Header Lines**: `:48–62`
- Temporarily sets `m_renderingView` to the current view during `ecvGLView::redraw()`
- Makes `getEffectiveView()` return the rendering view instead of the UI-active view
- Essential for multi-view isolation during paint

#### Private Members

| Member | Header Line | Purpose |
|--------|-------------|---------|
| `m_activeView` | `:155` | The UI-active view (user last clicked) |
| `m_renderingView` | `:156` | Temporarily set during rendering via `ScopedRenderOverride` |
| `m_activeSource` | `:157` | Currently selected DB tree entity |
| `m_activeRepresentation` | `:158` | Representation for (activeSource, activeView) |
| `m_cachedView/Source/Representation` | `:161-163` | For `triggerSignals()` diffing |
| `m_views` | `:165` | `QList<ecvGenericGLDisplay*>` — all registered views |
| `m_layouts` | `:166` | `QList<ecvViewLayoutProxy*>` — all layout proxies |

#### Signals

| Signal | Header Line | Emitted When |
|--------|-------------|--------------|
| `activeViewChanged(new, old)` | `:134` | User clicks a different view |
| `activeSourceChanged(source)` | `:136` | DB tree selection changes |
| `activeRepresentationChanged(repr)` | `:137` | Active (source, view) pair changes |
| `activeLayoutChanged(layout)` | `:138` | Active layout changes |
| `viewRegistered(view)` | `:140` | New view created and registered |
| `viewUnregistered(view)` | `:141` | View closed and unregistered |
| `viewCountChanged(count)` | `:142` | Number of views changed |
| `layoutRegistered(layout)` | `:144` | Layout proxy registered |
| `layoutUnregistered(layout)` | `:145` | Layout proxy unregistered |

---

### 2. ecvRepresentationManager

> **ParaView equivalent**: `pqServerManagerModel` (representation registry)

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecvRepresentationManager.h` |
| **Implementation** | `libs/CV_db/src/ecvRepresentationManager.cpp` |

#### Class Declaration

- **Line**: `ecvRepresentationManager.h:33` — `class CV_DB_LIB_API ecvRepresentationManager : public QObject`

#### Public Methods

| Method | Header Line | Description |
|--------|-------------|-------------|
| `instance()` | `:37` | Singleton accessor |
| `getRepresentation(entity, view)` | `:42-43` | Read-only lookup; returns `nullptr` if not found |
| `ensureRepresentation(entity, view)` | `:46-47` | Get-or-create; emits `representationAdded` on creation |
| `getRepresentationsForEntity(entity)` | `:51-52` | All representations for one entity across all views |
| `getRepresentationsForView(view)` | `:53-54` | All representations for one view across all entities |
| `removeRepresentationsForEntity(entity)` | `:58` | Removes all reps for entity; invokes cleanup callback |
| `removeRepresentationsForView(view)` | `:59` | Removes all reps for view; invokes cleanup callback |
| `removeRepresentation(entity, view)` | `:60` | Removes single rep; invokes cleanup callback |
| `count()` | `:62` | Total number of (entity, view) pairs |
| `setActorCleanupCallback(cb)` | `:68` | Registers VTK-layer cleanup lambda |

#### Internal Data Structures

```cpp
// ecvRepresentationManager.h:78-81
using Key = QPair<ccHObject*, ecvGenericGLDisplay*>;
QHash<Key, std::shared_ptr<ecvViewRepresentation>> m_representations;
CleanupCallback m_actorCleanup;  // set by VtkEngine layer
mutable QReadWriteLock m_lock;   // for thread-safe access
```

#### Cleanup Callback Mechanism

The `CleanupCallback` (`ecvRepresentationManager.h:66-67`) is a `std::function<void(ccHObject*, ecvGenericGLDisplay*)>`.

**Registration**: `VtkDisplayTools` constructor at `VtkDisplayTools.cpp:109-120`.

**Invocation**: On every removal path (`removeRepresentationsForEntity`, `removeRepresentationsForView`, `removeRepresentation`), the callback is invoked **before** erasing the map entry. This ensures VTK actors are cleaned up before the representation is destroyed.

#### Signals

| Signal | Header Line |
|--------|-------------|
| `representationAdded(rep)` | `:71` |
| `representationRemoved(entity, view)` | `:72` |
| `representationChanged(rep)` | `:73` |

---

### 3. ecvViewRepresentation

> **ParaView equivalent**: `vtkSMRepresentationProxy`

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecvViewRepresentation.h` |
| **Implementation** | `libs/CV_db/src/ecvViewRepresentation.cpp` |

#### Class Declaration

- **Line**: `ecvViewRepresentation.h:24` — `class CV_DB_LIB_API ecvViewRepresentation`

#### RenderMode Enum

```cpp
// ecvViewRepresentation.h:26-32
enum class RenderMode : int {
    Inherit = -1,
    Points = 0,
    Wireframe = 1,
    Surface = 2,
    SurfaceWithEdges = 3
};
```

#### Properties Struct

```cpp
// ecvViewRepresentation.h:51-62
struct Properties {
    std::optional<float> opacity;
    std::optional<float> pointSize;
    std::optional<float> lineWidth;
    std::optional<RenderMode> renderMode;
    std::optional<bool> edgeVisibility;
    std::optional<int> scalarFieldIndex;
    std::optional<bool> showScalarField;
    std::optional<bool> showColors;
    std::optional<bool> showNormals;
    std::optional<float> normalScale;
};
```

All fields are `std::optional` — when unset, the entity's own global property is used.

#### Methods

| Method | Header Line | Description |
|--------|-------------|-------------|
| `getEntity()` | `:37` | Returns the bound `ccHObject*` |
| `getView()` | `:38` | Returns the bound `ecvGenericGLDisplay*` |
| `isVisible()` | `:42` | Returns visibility override value (default: entity's `m_visible`) |
| `setVisible(v)` | `:43` | Sets per-view visibility override |
| `hasVisibilityOverride()` | `:44-46` | `m_visibilityOverride.has_value()` |
| `clearVisibilityOverride()` | `:47` | Removes override, reverts to entity default |
| `properties()` | `:64-65` | Const/non-const access to Properties struct |
| `setProperties(props)` | `:66` | Bulk-sets properties, marks dirty |
| `effectiveOpacity()` | `:68` | Returns optional override or `entity->getOpacity()` |
| `effectivePointSize()` | `:69` | Returns optional override → entity point cloud size → `1.0f` |
| `isDirty()` | `:73` | Needs VTK actor update |
| `setDirty(d)` | `:74` | Marks for next render update |

#### `effectivePointSize()` Fallback Chain

```
ecvViewRepresentation.cpp:49-59
1. m_properties.pointSize (per-view override)  →  if set, return it
2. m_entity is CV_TYPES::POINT_CLOUD           →  cloud->getPointSize()
3. fallback                                     →  1.0f
```

---

### 4. ecvGLView

> **ParaView equivalent**: `pqRenderView` (owns renderer, render window, interactor)

| File | Path |
|------|------|
| **Header** | `libs/VtkEngine/Visualization/ecvGLView.h` |
| **Implementation** | `libs/VtkEngine/Visualization/ecvGLView.cpp` |

#### Factory: `ecvGLView::Create()`

**Location**: `ecvGLView.cpp:69-79`

```
1. new ecvGLView(parent)
2. initVtkPipeline(parent, stereoMode)
3. Create m_winDBRoot = new ccHObject("DB.GLView_<uniqueID>")
4. ecvGenericGLDisplay::RegisterGLDisplay(m_vtkWidget, view)
5. ecvViewManager::instance().registerView(view)  ← registers with ViewManager
```

#### `initVtkPipeline()`

**Location**: `ecvGLView.cpp:82-105`

```
1. m_vtkWidget = new QVTKWidgetCustom(parent, primaryTools, stereoMode)
2. m_vtkWidget->setOwnerView(this)
3. Create vtkRenderer + vtkGenericOpenGLRenderWindow
4. m_visualizer3D = make_shared<VtkVis>(renderer, renderWindow, style, title, false)
5. m_vtkWidget->SetRenderWindow(renderWindow)
6. Setup interactor, custom interactor style
7. m_visualizer3D->initialize()
```

Each `ecvGLView` owns:
- **`m_vtkWidget`** — the Qt widget for OpenGL rendering
- **`m_visualizer3D`** — its own `VtkVis` instance (independent actor map, renderer)
- **`m_winDBRoot`** — per-window DB root for window-local objects

#### Destructor

**Location**: `ecvGLView.cpp:42-67`

```
1. emit aboutToClose(this)
2. m_globalDBRoot->removeFromDisplay_recursive(this)
3. ecvRepresentationManager::instance().removeRepresentationsForView(this)
    → invokes CleanupCallback → VTK actors removed
4. ecvViewManager::instance().unregisterView(this)
5. ecvGenericGLDisplay::UnregisterGLDisplay(m_vtkWidget)
6. delete m_hotZone, m_rectPickingPoly, m_winDBRoot
```

#### `redraw()` — The Core Render Loop

**Location**: `ecvGLView.cpp:111-191`

Detailed in [ecvGLView::redraw()](#ecvglviewredraw) section below.

---

### 5. QVTKWidgetCustom

| File | Path |
|------|------|
| **Header** | `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h` |
| **Implementation** | `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` |

#### Active View on Mouse Interaction

- **`mousePressEvent()`** (`QVTKWidgetCustom.cpp:548-559`): On mouse press, calls `ecvGenericGLDisplay::FromWidget(this)` → if not active, calls `ecvViewManager::setActiveView(display)`. This is how clicking a view makes it active.
- **`mouseDoubleClickEvent()`** (`QVTKWidgetCustom.cpp:612-619`): Same active-view logic.
- **`resolveDisplay()`** (`QVTKWidgetCustom.cpp:275-278`): Returns the `ecvGenericGLDisplay*` for this widget.
- **`setOwnerView()` / `ownerView()`** (`QVTKWidgetCustom.h:197-198`): Links widget to its `ecvGLView`.
- **`curCtx()`** (`QVTKWidgetCustom.cpp:87-95`): Delegates to owner view or primary tools for CC_DRAW_CONTEXT.

---

### 6. VtkVis — Text/Caption Rendering

| File | Path |
|------|------|
| **Header** | `libs/VtkEngine/Visualization/VtkVis.h` |
| **Implementation** | `libs/VtkEngine/Visualization/VtkVis.cpp` |

#### `addText()`

**Location**: `VtkVis.cpp:1010-1040`

Creates a `vtkTextActor` at pixel position `(xpos, ypos)` with given font size and color. Stores in `shape_actor_map_[id]`. Used for ABC legend text rendering.

#### `updateText()`

**Location**: `VtkVis.cpp:1042-1053`

Looks up existing `vtkTextActor` by ID in `shape_actor_map_`, updates input text and position. If the ID exists, updates in-place (no recreation). If not found, returns false.

#### `addCaption()`

**Location**: `VtkVis.cpp:1994-2112`

Detailed in [VTK Caption Widget System](#vtk-caption-widget-system) below.

#### `updateCaption()`

**Location**: `VtkVis.cpp:1903-1956`

Detailed in [VTK Caption Widget System](#vtk-caption-widget-system) below.

---

### 7. VtkDisplayTools

> Bridge between CV_db display entities and VTK rendering.

| File | Path |
|------|------|
| **Header** | `libs/VtkEngine/Visualization/VtkDisplayTools.h` |
| **Implementation** | `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` |

#### `resolveVisualizer(display)`

**Location**: `VtkDisplayTools.cpp:387-398`

Determines which `VtkVis` instance to use for a given display:

```
1. display == nullptr or display == this (primary)  →  m_primaryVis or m_visualizer3D
2. display is ecvGLView with getVisualizer3D()      →  cast to VtkVis*
3. fallback                                          →  m_visualizer3D.get()
```

This is the **critical routing function** — every draw call goes through `resolveVisualizer()` to pick the correct VTK pipeline. Used at `VtkDisplayTools.cpp:419, 587, 611, 725, 746, 821, 846, 965, 1034, 1066, 1094, 1143, 1205, 1453`.

#### Actor Cleanup Callback Registration

**Location**: `VtkDisplayTools.cpp:109-120`

```cpp
ecvRepresentationManager::instance().setActorCleanupCallback(
    [this](ccHObject* entity, ecvGenericGLDisplay* view) {
        if (!entity) return;
        std::string viewID = CVTools::FromQString(entity->getViewId());
        VtkVis* vis = resolveVisualizer(view);
        if (vis && vis->contains(viewID)) {
            vis->removePointCloud(viewID);
            vis->removePolygonMesh(viewID);
            vis->removeShape(viewID);
        }
    });
```

Uses `resolveVisualizer()` to find the correct `VtkVis` → calls public removal APIs.

---

### 8. ecvDisplayTools

> Shared display tools instance managed by `ecvViewManager`. Provides display utility functions. **No longer a public singleton** — access via `ecvViewManager::instance().displayTools()` or per-view `m_displayTools`.

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecvDisplayTools.h` |
| **Implementation** | `libs/CV_db/src/ecvDisplayTools.cpp` |

#### `UpdateScreen()`

**Location**: `ecvDisplayTools.cpp:1490-1499`

```cpp
void ecvDisplayTools::UpdateScreen() {
    if (QWidget* w = GetCurrentScreen()) {
        w->update();
    }
    UpdateScene();
    // Multi-view refresh
    if (ecvViewManager::instance().viewCount() > 1) {
        ecvViewManager::instance().refreshAll();
    }
}
```

Ensures that DB tree visibility toggles and other global changes propagate to **all** views, not just the primary.

---

## Entity Visibility & View Isolation

### `ccHObject::isDisplayedIn()`

**Location**: `libs/CV_db/src/ecvHObject.cpp:1862-1874`

```cpp
bool ccHObject::isDisplayedIn(const ecvGenericGLDisplay* display) const {
    if (display == nullptr) return true;                     // (1)

    if (m_currentDisplay == nullptr) {                       // (2) entity not bound
        if (ecvViewManager::instance().viewCount() <= 1) {
            return true;                                     // (2a) single view: backward compat
        }
        const ecvGenericGLDisplay* effective =
                ecvViewManager::instance().getEffectiveView();
        return (effective == nullptr || display == effective); // (2b) multi-view: only in rendering view
    }
    return (m_currentDisplay == display);                     // (3) bound: exact match
}
```

**Header doc**: `libs/CV_db/include/ecvHObject.h:576-583`

#### Three-Way Logic

| Condition | Behavior |
|-----------|----------|
| `display == nullptr` | Legacy mode (no view context), draw always |
| `m_currentDisplay == nullptr` + single view | Backward compat: draw in the single view |
| `m_currentDisplay == nullptr` + multi-view | Draw only in the **effective** (rendering) view — prevents "pollution" |
| `m_currentDisplay == display` | Exact match, draw here |
| `m_currentDisplay != display` | Skip this view |

### ScopedRenderOverride

**Location**: `libs/CV_db/include/ecvViewManager.h:48-62`

```
ecvGLView::redraw() {
    ScopedRenderOverride viewGuard(this);  // ← m_renderingView = this
    ...
    m_globalDBRoot->draw(context);         // isDisplayedIn() checks getEffectiveView()
    ...
}   // ~ScopedRenderOverride restores m_renderingView
```

During rendering, `getEffectiveView()` returns `m_renderingView` (the view being painted), NOT `m_activeView` (the view the user last clicked). This is what makes multi-view isolation work correctly — each view only draws entities that belong to it.

---

## Draw Pipeline

### `ccHObject::draw()`

**Location**: `libs/CV_db/src/ecvHObject.cpp:1445`

```
Step 1: Remove check (line 1447-1452)
  ├── If entity has remove flag → RemoveEntities + return

Step 2: Enabled check (line 1457-1462)
  ├── If disabled → hideObject_recursive + return

Step 3: Visibility + view isolation (line 1466-1467)
  ├── drawInThisContext = (m_visible || m_selected) && isDisplayedIn(context.display)

Step 4: Per-view representation (lines 1469-1481)
  ├── If drawable + not fixed ID → ensureRepresentation(entity, display)    [line 1472-1474]
  ├── Else → getRepresentation (read-only lookup)                           [line 1475-1477]
  ├── If visibility override exists → drawInThisContext = viewRep->isVisible() [line 1479-1481]

Step 5: Opacity from representation (lines 1483-1487)
  ├── If rep has opacity override → use effectiveOpacity()
  └── Else → use entity's getOpacity()

Step 6: 3D geometry draw (line ~1507-1523)
  ├── If visible + drawInThisContext + forceRedraw → drawMeOnly(context)

Step 7: forceRedraw recovery (line ~1525-1538)
  ├── If entity needs forceRedraw but context didn't have it
  └── → Create newContext with forceRedraw=true, recursive draw

Step 8: Children traversal (line ~1577-1580)
  └── For each child → child->draw(context)
```

### Per-View Representation Wiring

**Key lines**: `ecvHObject.cpp:1469-1481`

The `ensureRepresentation()` call (line 1473) is what populates the `ecvRepresentationManager` registry. Every entity that gets drawn in a view automatically gets a representation created. This enables:

- Per-view visibility overrides
- Per-view opacity overrides
- Per-view point size, render mode, etc.
- Proper cleanup when views are closed

### `ecvGLView::redraw()`

**Location**: `libs/VtkEngine/Visualization/ecvGLView.cpp:111`

```
1. Guard: if no visualizer or widget → return                    [line 112]
2. ScopedRenderOverride viewGuard(this)                          [line 119]
   → getEffectiveView() now returns THIS view
3. Sync per-view glViewport from widget dimensions               [line 124-127]
4. Build CC_DRAW_CONTEXT from per-view state                     [line 130-132]
5. Background setup                                              [line 134-161]
6. 3D pass: m_globalDBRoot->draw(context)                        [line 164-166]
7. 3D pass: m_winDBRoot->draw(context)                           [line 168-170]
8. 2D foreground: m_globalDBRoot->draw(context)                  [line 174-176]
9. 2D foreground: m_winDBRoot->draw(context)                     [line 178-180]
10. Hot zone / clickable items                                   [line 182-193]
```

`context.display` is set to `this` (the ecvGLView), so every `isDisplayedIn()` call during the tree walk knows which view is being painted.

---

## cc2DLabel Rendering

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecv2DLabel.h` |
| **Implementation** | `libs/CV_db/src/ecv2DLabel.cpp` |

### `cc2DLabel::drawMeOnly()`

**Location**: `ecv2DLabel.cpp:868-894`

```cpp
void cc2DLabel::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (m_pickedPoints.empty()) return;
    if (!MACRO_Foreground(context)) return;
    if (MACRO_VirtualTransEnabled(context)) return;

    // Camera-dependent: always update on zoom/rotate/pan
    if (!isRedraw() && !context.forceRedraw) {    // ← line 881
        return;
    }

    // Per-view isolation for labels
    ecvGenericGLDisplay* myDisp = getDisplay();
    if (myDisp && context.display && myDisp != context.display) {
        return;                                     // ← line 887
    }

    if (MACRO_Draw3D(context))
        drawMeOnly3D(context);                      // ← line 891
    else if (MACRO_Draw2D(context))
        drawMeOnly2D(context);                      // ← line 893
}
```

**Important**: Line 881 uses `context.forceRedraw` (not just `isRedraw()`), so labels update on **camera changes** (zoom, rotate, pan), not only on model changes. This was a bug fix — previously `isRedraw()` alone would skip label updates on scroll-wheel zoom.

### ABC Legend Text (DisplayText)

**Location**: `ecv2DLabel.cpp:1260-1284`

```cpp
for (size_t j = 0; j < count; j++) {
    QString title;
    if (count == 1) title = getName();
    else if (count == 3) title = ABC[j];          // "A", "B", "C"
    else title = QString("P#%0").arg(m_pickedPoints[j].index);

    // Unique ID per legend to enable VtkVis::updateText() (not recreate)
    QString legendId = QString("%1_legend_%2")     // ← line 1270
                               .arg(this->getViewId())
                               .arg(j);

    ecvDisplayTools::DisplayText(
            title,
            static_cast<int>(m_pickedPoints[j].pos2D.x)
                + context.labelMarkerTextShift_pix,
            static_cast<int>(m_pickedPoints[j].pos2D.y)
                + context.labelMarkerTextShift_pix,
            ecvDisplayTools::ALIGN_DEFAULT,
            0.55f,                                 // ← semi-transparent black background
            ecvColor::white.rgb, &font, legendId,
            context.display);
}
```

Key design decisions:
1. **`legendId`** uses `viewId_legend_0`, `viewId_legend_1`, etc. — unique IDs allow `VtkVis::updateText()` to update existing `vtkTextActor`s in-place instead of deleting and recreating every frame.
2. **Opacity `0.55f`** — semi-transparent black background behind white text for readability.
3. **`context.display`** parameter — routes text to the correct `VtkVis` instance via `resolveVisualizer()`.

### Caption Box (DrawWidgets)

**Location**: `ecv2DLabel.cpp:1666-1739`

The caption widget (showing measurement info like distances, angles, coordinates) uses `ecvDisplayTools::DrawWidgets()` with `WIDGET_CAPTION` type.

Key parameters built at:
- **3D anchor**: Centroid of picked points → `param.center` (line ~1691)
- **2D position**: `param.pos` from `m_labelROI` with platform-specific Y flip (line ~1710-1726)
- **Font size**: Adaptive calculation at line 1731-1738
- **Background color**: `param.color.a` from `defaultBkgColor.a / 255.0f`
- **Text**: `m_historyMessage.join("\n").trimmed()` (line 1729-1730)

### Font Size Adaptive Algorithm

**Location**: `ecv2DLabel.cpp:1731-1738`

```cpp
const float logicalH = context.glH / context.devicePixelRatio;
const float logicalW = context.glW / context.devicePixelRatio;
const float refDim = std::min(logicalW, logicalH);    // ← min of logical dimensions
int captionFontSize =
        std::max(10, std::min(static_cast<int>(refDim / 42.0f), 28));
```

| Window Size | refDim | Font Size |
|-------------|--------|-----------|
| 420px | 420 | 10 (min clamp) |
| 800px | 800 | 19 |
| 1176px | 1176 | 28 (max clamp) |
| 1920px | 1080 (height) | 25 |

Uses `min(logicalW, logicalH)` as reference dimension for consistent scaling across window aspect ratios.

---

## VTK Caption Widget System

### `VtkVis::addCaption()`

**Location**: `VtkVis.cpp:1994-2112`

Creates a `CustomVtkCaptionWidget` with `vtkCaptionRepresentation`.

```
Step 1: Check duplicate ID                                   [line 2007-2012]
Step 2: Create vtkCaptionRepresentation                      [line 2015-2016]
Step 3: Set anchor position (3D world coords)                [line 2018-2019]
Step 4: Set 2D normalized position (pos2D / winSize)         [line 2021-2026]
Step 5: Calculate adaptive box size (SetPosition2)           [line 2028-2037]
Step 6: Configure caption actor (text, border, leader)       [line 2040-2070]
Step 7: Configure text properties (font, color, background)  [line 2046-2070]
Step 8: Create CustomVtkCaptionWidget                        [line 2072-2109]
Step 9: Store in m_widget_map                                [line 2108-2109]
```

### `VtkVis::updateCaption()`

**Location**: `VtkVis.cpp:1903-1956`

Updates an existing caption widget by `viewID`:

```
Step 1: Find widget by ID                                    [line 1914]
Step 2: Cast to CustomVtkCaptionWidget                       [line 1917-1919]
Step 3: Get vtkCaptionRepresentation                         [line 1921-1922]
Step 4: Update 3D anchor position (SetAnchorPosition)        [line 1924]
Step 5: Update 2D normalized position                        [line 1926-1930]
        pos2D.x / winW, pos2D.y / winH
Step 6: Recalculate adaptive box size                        [line 1932-1941]
Step 7: Update caption text                                  [line 1943-1944]
Step 8: Update text properties (color, font, background)     [line 1946-1952]
```

**Critical fix**: Line 1930 (`rep->SetPosition(...)`) — previously `pos2D` was ignored in `updateCaption()`, causing the text box to detach from the 3D handle on zoom.

### Caption Box Sizing Algorithm

Used in both `addCaption()` (line 2028-2037) and `updateCaption()` (line 1932-1941):

```cpp
const int lineCount = 1 + count('\n' in text);
const double refH = 800.0;
const double scaleFactor = std::clamp(refH / winH, 0.6, 1.6);
const double baseW = std::clamp(0.30 * scaleFactor, 0.18, 0.50);
const double perLineH = 0.045 * scaleFactor;
const double captionH = std::clamp(perLineH * lineCount + 0.03, 0.06, 0.55);
captionRepresentation->SetPosition2(baseW, captionH);
```

| Window Height | scaleFactor | baseW | 1-line captionH | 3-line captionH |
|---------------|-------------|-------|------------------|------------------|
| 400px | 1.6 (max) | 0.48 | 0.102 | 0.246 |
| 800px | 1.0 | 0.30 | 0.075 | 0.165 |
| 1200px | 0.667 | 0.20 | 0.060 (min) | 0.120 |
| 1600px | 0.6 (min) | 0.18 (min) | 0.060 (min) | 0.111 |

`SetPosition2(w, h)` sets the caption box size as a **fraction of the render window** (normalized 0.0–1.0). Both width and height are adaptive to the actual window size.

### `VtkVis::addText()` / `updateText()`

**Locations**: `VtkVis.cpp:1010-1040` / `VtkVis.cpp:1042-1053`

Used for the ABC legend text (small text next to 3D handle markers).

- `addText()` creates a `vtkTextActor` at pixel `(xpos, ypos)`, stores in `shape_actor_map_[id]`
- `updateText()` looks up by `id`, updates position and text in-place
- The ABC rendering uses unique `legendId`s (`viewId_legend_0`, etc.) so `updateText()` can update existing actors instead of always creating new ones. This significantly improves rendering efficiency.

---

## Entity Binding & Load Path

### `MainWindow::addToDB()`

**Location**: `app/MainWindow.cpp:3741`

```cpp
// ParaView-style: bind newly loaded objects to the active view
ecvViewManager::instance().associateToActiveView(obj);
```

When a new entity is loaded (file open, import, etc.), it is bound to the **currently active** view via `setDisplay_recursive()`. This prevents newly loaded objects from appearing in all views simultaneously.

### Zoom Behavior

**Location**: `app/MainWindow.cpp:3743-3752`

```cpp
if (updateZoom) {
    auto* activeView = ecvViewManager::instance().getActiveView();
    if (!activeView || activeView == ecvDisplayTools::TheInstance()) {
        ecvDisplayTools::ZoomGlobal();           // primary view
    } else {
        auto* glView = dynamic_cast<ecvGLView*>(activeView);
        if (glView) glView->zoomGlobal();        // secondary view only
    }
}
```

Zoom is applied **only** to the active view after loading, not all views.

### `zoomOn()` — Active View Sync

**Location**: `app/MainWindow.cpp:5155-5168`

```cpp
void MainWindow::zoomOn(ccHObject* object) {
    if (!object) return;
    auto& vm = ecvViewManager::instance();
    auto* ownerView = vm.findViewForEntity(object);
    if (ownerView && ownerView != vm.getActiveView()) {
        vm.setActiveView(ownerView);   // switch to the view that owns the entity
    }
    if (ecvDisplayTools::GetCurrentScreen()) {
        ccBBox box = object->getDisplayBB_recursive(false);
        ecvDisplayTools::UpdateConstellationCenterAndZoom(&box);
    }
}
```

When double-clicking an entity in the DB tree, the application first switches the active view to the one that owns the entity, then zooms.

### `zoomOnEntities()` — Same Pattern

**Location**: `app/MainWindow.cpp:6108-6125`

Same active-view sync + zoom + `refreshAll()` for full multi-view consistency.

---

## VTK Actor Lifecycle & Cleanup

### Creation Flow

```
User loads file
  → MainWindow::addToDB()
    → associateToActiveView(obj)                        [MainWindow.cpp:3741]
    → ecvDisplayTools::ZoomGlobal() triggers redraw
      → ccHObject::draw()
        → ensureRepresentation(entity, display)          [ecvHObject.cpp:1473]
          → ecvRepresentationManager creates entry
        → drawMeOnly() → VtkDisplayTools
          → resolveVisualizer(context.display)            [VtkDisplayTools.cpp:387]
          → VtkVis::addPointCloud/addPolygonMesh/etc.
```

### Destruction Flow (View Close)

```
ecvGLView::~ecvGLView()
  → m_globalDBRoot->removeFromDisplay_recursive(this)    [ecvGLView.cpp:46]
  → ecvRepresentationManager::removeRepresentationsForView(this)  [ecvGLView.cpp:49]
    → For each (entity, this) pair:
      → m_actorCleanup(entity, this)                     [ecvRepresentationManager.cpp:104-106]
        → VtkDisplayTools lambda:                        [VtkDisplayTools.cpp:110-120]
          → resolveVisualizer(view)
          → vis->removePointCloud(viewID)
          → vis->removePolygonMesh(viewID)
          → vis->removeShape(viewID)
      → erase from m_representations
  → ecvViewManager::unregisterView(this)                 [ecvGLView.cpp:51]
  → ecvGenericGLDisplay::UnregisterGLDisplay(m_vtkWidget) [ecvGLView.cpp:54]
```

### Destruction Flow (Entity Delete)

```
Entity delete from DB tree
  → ecvRepresentationManager::removeRepresentationsForEntity(entity)
    → For each (entity, view) pair:
      → m_actorCleanup(entity, view)
      → erase from m_representations
```

---

## How to Use

### 1. Creating a New 3D View

| Method | Action |
|--------|--------|
| **"+" Button** | Click `+` on tab bar → new layout with fresh 3D view |
| **Split View** | Title bar: ┃ (horizontal) or ━ (vertical) split |
| **Right-click** | Title bar context menu → Split Horizontal/Vertical/Close |
| **Menu** | `Display → New 3D View` or ViewToolBar button |
| **Link Cameras** | `Display → Link Cameras` — syncs camera across all views |

### 2. View Title Bar (ParaView pqViewFrame style)

```
┌──────────────────────────────────────────────────────────┐
│ [spacer]                    RenderView1  │ ┃ ━ □ ✕ │     │
│                                          split buttons   │
├──────────────────────────────────────────────────────────┤
│              [3D Rendering Content]                      │
└──────────────────────────────────────────────────────────┘
```

### 3. Switching Active View

Click on a view to make it active. The active view determines:
- Where newly loaded entities will be displayed
- Which view receives keyboard/mouse interaction
- Which view toolbar tools operate on

### 4. Moving Entities Between Views

**Properties Panel**: Select entity → "Current Display" dropdown → choose target view.

**DB Tree Right-click**: Right-click entity → "Move to View" → choose target view.

| Setting | Meaning |
|---------|---------|
| **None** | Show in all views (default for single-view) |
| **Primary View** | Primary window only |
| **3D View N** | That specific secondary view only |

### 5. Closing a View

When a view is closed:
1. All representations for that view are cleaned up (VTK actors removed)
2. View is unregistered from `ecvViewManager`
3. If it was active, the next available view becomes active
4. Entities previously bound to that view are reset to `nullptr`

---

## Programmatic API

### Creating a new view

```cpp
#include <Visualization/ecvGLView.h>
#include <ecvViewManager.h>

// From MainWindow:
ecvGLView* view = new3DView();

// Or manually:
auto* view = ecvGLView::Create(mainWindow);
view->setSceneDB(dbRoot->getRootEntity());
```

### Moving an entity to a view

```cpp
entity->setDisplay_recursive(targetView);  // bind to specific view
entity->setDisplay_recursive(nullptr);     // show in all views
```

### Querying views

```cpp
auto& mgr = ecvViewManager::instance();
int count = mgr.viewCount();
auto* active = mgr.getActiveView();
const auto& all = mgr.getAllViews();
auto* view = mgr.findView(uniqueID);
auto* viewForEntity = mgr.findViewForEntity(entity);
```

### Per-view representation

```cpp
auto& repMgr = ecvRepresentationManager::instance();

auto* rep = repMgr.ensureRepresentation(entity, view);
rep->setVisible(false);                          // hide in this view only
rep->properties().opacity = 0.5f;
rep->properties().renderMode = ecvViewRepresentation::RenderMode::Wireframe;
rep->setDirty();

// Batch query
auto reps = repMgr.getRepresentationsForEntity(entity);
auto reps = repMgr.getRepresentationsForView(view);

// Cleanup (automatic on view close)
repMgr.removeRepresentationsForView(view);
```

### Connecting to signals

```cpp
connect(&ecvViewManager::instance(), &ecvViewManager::activeViewChanged,
        this, [](ecvGenericGLDisplay* newView, ecvGenericGLDisplay* oldView) {
    // handle view switch
});

connect(&ecvViewManager::instance(), &ecvViewManager::viewCountChanged,
        this, [](int count) {
    // update UI (e.g., enable/disable "Move to View" menu)
});
```

---

## FAQ

**Q: Can I have more than 2 views?**
A: Yes. Each split or new-view action creates an additional view. The MDI area supports tiling and cascading.

**Q: Does each view have its own camera?**
A: Yes. Each view has independent `ecvViewportParameters` (camera center, view matrix, perspective/ortho, zoom).

**Q: Is geometry data duplicated across views?**
A: No. The scene DB (`ccHObject` tree and underlying `vtkPolyData`) is shared. Each view maintains its own VTK actors that reference the shared geometry data.

**Q: What happens when I load a new file?**
A: The entity is added to the global scene DB and bound to the active view via `associateToActiveView()` (`MainWindow.cpp:3741`). It only appears in the active view.

**Q: Why does ABC text not follow the handle on zoom?**
A: Fixed. The `drawMeOnly()` guard at `ecv2DLabel.cpp:881` now checks `context.forceRedraw` in addition to `isRedraw()`, ensuring labels update on all camera changes including scroll-wheel zoom.

**Q: Why is the caption box too large on small windows?**
A: Fixed. Both `VtkVis::addCaption()` and `updateCaption()` use an adaptive sizing algorithm based on `scaleFactor = clamp(800/winH, 0.6, 1.6)` instead of a fixed `0.38 × 0.25` ratio.

**Q: How does multi-view isolation work?**
A: `isDisplayedIn()` (`ecvHObject.cpp:1862`) checks `m_currentDisplay` against the rendering view. `ScopedRenderOverride` ensures `getEffectiveView()` returns the correct view during paint. Unbound entities (nullptr display) only show in the rendering view when multiple views exist.

---

## Active View Switching — `switchActiveView` / `restorePrimaryView`

> Replaces the historical `ScopedVisSwap` pattern. Now a **persistent** switch instead of RAII scoped.

| File | Path |
|------|------|
| **Header** | `libs/VtkEngine/Visualization/VtkDisplayTools.h:441-447` |
| **Implementation** | `libs/VtkEngine/Visualization/VtkDisplayTools.cpp:131-156` |

### `switchActiveView(vis, widget)`

**Location**: `VtkDisplayTools.cpp:131-156`

When the user clicks a different view, `MainWindow::rebindToolsToActiveView()` (`MainWindow.cpp:2629-2647`) calls this method to **persistently** swap the singleton `VtkDisplayTools`'s `m_visualizer3D` and `m_vtkWidget` to the secondary view's pipeline:

```
1. Save the primary pipeline on first switch (m_primaryVis, m_primaryWidget)  [line 136-139]
2. Disconnect picking signal from old VtkVis                                   [line 143-147]
3. Connect picking signal to new VtkVis                                        [line 148-151]
4. m_visualizer3D = vis; m_vtkWidget = widget                                 [line 153-154]
5. SetCurrentScreen(widget)                                                    [line 155]
```

### `restorePrimaryView()`

**Location**: `VtkDisplayTools.cpp:158+`

Called when user clicks back to the primary view. Restores `m_primaryVis` → `m_visualizer3D` and `m_primaryWidget` → `m_vtkWidget`.

### Connection: `activeViewChanged` → `rebindToolsToActiveView`

**Location**: `MainWindow.cpp:728-743`

```cpp
connect(&ecvViewManager::instance(), &ecvViewManager::activeViewChanged,
        this, [this](ecvGenericGLDisplay* newActive, ecvGenericGLDisplay*) {
            rebindToolsToActiveView(newActive);
            // ... picking hub, view frame highlight, menu update
        });
```

`rebindToolsToActiveView()` (`MainWindow.cpp:2629-2647`):
- If `newActive` is an `ecvGLView` → `switchActiveView(glView->getVisualizer3DSP(), glView->getVtkWidget())`
- Else → `restorePrimaryView()` + `SetCurrentScreen(primaryWidget)`

---

## ScopedHotZoneRender — Per-View Hot Zone Rendering

| File | Path |
|------|------|
| **Header** | `libs/VtkEngine/Visualization/VtkDisplayTools.h:461-493` |
| **Implementation** | `libs/VtkEngine/Visualization/VtkDisplayTools.cpp:285-350` |

**Purpose**: RAII helper that **temporarily** swaps `VtkDisplayTools`'s pipeline to a specific view's `VtkVis` for the duration of hot zone / clickable item rendering. Used inside `ecvGLView::redraw()`.

### Constructor (`VtkDisplayTools.cpp:285-331`)

```
1. Save current state: m_visualizer3D, m_visualizer2D, m_vtkWidget, glViewport, hotZone, clickableItems
2. Swap to target view's VtkVis + widget                            [line 302-304]
3. Update GL viewport from widget dimensions                        [line 306-309]
4. Setup local ImageVis (2D) if needed                              [line 311-323]
5. Increment m_scopedVisSwapDepth                                   [line 325]
6. Create HotZone if needed; assign to dt->m_hotZone                [line 327-330]
```

### `draw()` (`VtkDisplayTools.cpp:333-337`)

```cpp
void ScopedHotZoneRender::draw() {
    int yStart = 0;
    ecvDisplayTools::DrawClickableItems(0, yStart);
    m_clickableItems = m_dt->m_clickableItems;
}
```

### Destructor (`VtkDisplayTools.cpp:339-350`)

```
1. Decrement m_scopedVisSwapDepth
2. Restore: m_hotZone, m_clickableItems, m_visualizer3D, m_visualizer2D,
   m_vtkWidget, m_primaryCtx.glViewport, SetCurrentScreen
```

### Usage — Only in `ecvGLView::redraw()`

**Location**: `ecvGLView.cpp:183-194`

```cpp
{
    auto* primaryDT = static_cast<VtkDisplayTools*>(ecvDisplayTools::TheInstance());
    if (primaryDT) {
        VtkDisplayTools::ScopedHotZoneRender hzRender(
                primaryDT, m_visualizer3D, m_vtkWidget,
                m_hotZone, m_ctx, m_clickableItems);
        hzRender.draw();
    }
}
```

---

## Camera Link System (`VtkCameraLink`)

> **ParaView equivalent**: `vtkSMCameraLink`

| File | Path |
|------|------|
| **Header** | `libs/VtkEngine/Visualization/VtkCameraLink.h` |
| **Implementation** | `libs/VtkEngine/Visualization/VtkCameraLink.cpp` |

### Class: `VtkCameraLink`

**Location**: `VtkCameraLink.h:29`

Singleton that synchronizes cameras across all registered `VtkVis` views.

### Public API

| Method | Line | Description |
|--------|------|-------------|
| `instance()` | `:31` | Singleton accessor |
| `setEnabled(bool)` | `:33` | Enable/disable — installs or removes vtkRenderWindow observers |
| `isEnabled()` | `:34` | Query state |
| `addView(VtkVis*)` | `:36` | Register a view for camera sync |
| `removeView(VtkVis*)` | `:37` | Unregister a view |
| `clear()` | `:38` | Remove all views |
| `setSyncInteractiveRenders(bool)` | `:40` | Toggle interactive render sync |

### Internal Mechanism

```
1. setEnabled(true) → installObservers()
   → For each LinkedView, attach vtkCallbackCommand to vtkRenderWindow::EndEvent

2. User rotates/zooms in View A
   → vtkRenderWindow::EndEvent fires
   → OnRenderEnd() static callback
     → Identifies source VtkVis from vtkObject* caller
     → syncCamerasFrom(source)
       → For each OTHER view:
         → Copy camera: position, focal point, view up, clipping, view angle,
           parallel scale, parallel projection, center of rotation
         → Render() target window
       → Re-entry guard: m_updating prevents infinite loop
```

### Private Members

| Member | Line | Purpose |
|--------|------|---------|
| `m_enabled` | `:59` | Global enable flag |
| `m_updating` | `:60` | Re-entry guard to prevent infinite render loops |
| `m_syncInteractive` | `:61` | Sync during interactive (mouse drag) renders |
| `m_views` | `:68` | `vector<LinkedView>` — each has `VtkVis*`, observer, observerTag |

### UI Toggle

**Location**: `MainWindow.cpp:1378-1392`

```cpp
auto* linkCamerasAction = new QAction(tr("Link Cameras"), this);
linkCamerasAction->setCheckable(true);
connect(linkCamerasAction, &QAction::toggled, this, [](bool checked) {
    VtkCameraLink::instance().setEnabled(checked);
});
m_ui->ViewToolBar->addAction(linkCamerasAction);
displayMenu->addAction(linkCamerasAction);
```

---

## CC_DRAW_CONTEXT (ccGLDrawContext)

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecvDrawContext.h` |
| **Alias** | `using CC_DRAW_CONTEXT = ccGLDrawContext;` at line `722` |

### Struct Definition: `ecvDrawContext.h:572`

### Key Fields

| Field | Line | Type | Default | Purpose |
|-------|------|------|---------|---------|
| `drawingFlags` | `:573` | `int` | `0` | Bitfield: CC_DRAW_2D, CC_DRAW_3D, CC_DRAW_FOREGROUND, CC_VIRTUAL_TRANS_ENABLED |
| `forceRedraw` | `:574` | `bool` | `true` | Forces geometry redraw even when entity's model hasn't changed |
| `display` | `:582` | `ecvGenericGLDisplay*` | `nullptr` | The view that owns this context; used by `isDisplayedIn()` for view isolation |
| `viewID` | `:586` | `QString` | `"unnamed"` | Current entity's view identifier for VTK actor lookup |
| `opacity` | `:591` | `float` | `1.0` | Per-draw opacity (may be overridden by representation) |
| `visible` | `:593` | `bool` | `true` | Entity visibility flag |
| `glW` | `:607` | `int` | `0` | OpenGL screen width (pixels) |
| `glH` | `:608` | `int` | `0` | OpenGL screen height (pixels) |
| `devicePixelRatio` | `:609` | `float` | `1.0f` | HiDPI scaling factor (1 for standard, 2 for Retina) |
| `backgroundCol` | `:631` | `Rgbub` | default | Background color for view |
| `labelMarkerSize` | `:649` | `float` | `5` | Label marker radius (pixels) |
| `labelMarkerTextShift_pix` | `:650` | `float` | `5` | ABC text offset from marker center (pixels) |
| `labelOpacity` | `:654` | `unsigned` | `100` | Label background opacity (0-100) |
| `renderZoom` | `:611` | `float` | `1.0f` | Render zoom factor for UI element scaling |

### How Context Is Built Per-View

**`ecvGLView::getContext()`** — `ecvGLView.cpp:347-356`:

```cpp
void ecvGLView::getContext(ccGLDrawContext& context) const {
    ecvDisplayTools::GetContext(context, m_ctx);
    context.display = const_cast<ecvGLView*>(this);   // ← route to THIS view
    if (m_vtkWidget) {
        context.glW = m_vtkWidget->width();            // ← actual widget size
        context.glH = m_vtkWidget->height();
        context.devicePixelRatio = m_vtkWidget->devicePixelRatioF();
    }
}
```

---

## Redraw Flag System (`m_modelRedraw` / `m_forceRedraw`)

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecvDrawableObject.h` |

### Members

| Member | Line | Type | Purpose |
|--------|------|------|---------|
| `m_modelRedraw` | `:326` | `bool` | Set when entity's model data changes (geometry, colors, etc.) |
| `m_forceRedraw` | `:327` | `bool` | Set when entity needs forced redraw on next cycle |

### Accessors

| Method | Line | Description |
|--------|------|-------------|
| `isRedraw()` | `:59` | Returns `m_modelRedraw` |
| `setRedraw(state)` | `:61` | Sets `m_modelRedraw` |
| `setForceRedraw(state)` | `:63` | Sets `m_forceRedraw` |

### Interaction in `ccHObject::draw()`

**Location**: `ecvHObject.cpp:1445+`

```
Main geometry draw:
  if (m_visible && drawInThisContext && context.forceRedraw)   [line ~1507]
    → drawMeOnly(context)

forceRedraw recovery:                                          [line ~1529-1537]
  if (!context.forceRedraw && m_forceRedraw && !hasExist)
    → Create newContext with forceRedraw = true
    → setRedrawFlagRecursive(true)
    → Recursive draw(newContext)

Post-draw cleanup:                                             [line ~1629-1631]
  → setRedraw(true)
  → setForceRedraw(false)
```

### Impact on cc2DLabel

**`cc2DLabel::drawMeOnly()`** at `ecv2DLabel.cpp:881`:

```cpp
if (!isRedraw() && !context.forceRedraw) {
    return;  // Skip ONLY if neither flag is set
}
```

- `isRedraw()` returns `m_modelRedraw` — set when label data changes
- `context.forceRedraw` — set when camera changes (zoom, rotate, pan)
- Both must be false to skip drawing → labels always update on camera changes

---

## Signal Connections — Complete Map

### Signals WITH Active Connections

| Signal | Connected In | Line | Slot/Lambda | Behavior |
|--------|-------------|------|-------------|----------|
| `ecvViewManager::activeViewChanged` | `MainWindow.cpp` | `728-743` | Lambda | `rebindToolsToActiveView()`, update picking hub, mark active view frame, update menus |
| `ecvViewManager::activeViewChanged` | `ecvMultiViewWidget.cpp` | `64-68` | Lambda | `markActive(newActive)` — highlights active view in multi-view UI |
| `ecvViewManager::activeViewChanged` | `ccMPlaneDlgController.cpp` | `192-202` | Lambda | Install event filter on active window; link dialog |
| `ecvViewManager::activeViewChanged` | `ccCompass.cpp` | `561-573` | Lambda | Install event filter; link dialogs |
| `ecvViewLayoutProxy::layoutChanged` | `ecvMultiViewWidget.cpp` | `80` | `reload()` | Rebuild Qt splitter UI from proxy tree |

### Signals WITHOUT Active Connections (Declared Only)

| Signal | Declared At | Notes |
|--------|------------|-------|
| `activeSourceChanged` | `ecvViewManager.h:136` | Emitted at `ecvViewManager.cpp:88` — no subscribers found |
| `viewRegistered` | `ecvViewManager.h:140` | Emitted at `ecvViewManager.cpp:105` — no subscribers found |
| `viewUnregistered` | `ecvViewManager.h:141` | Emitted at `ecvViewManager.cpp:122` — no subscribers found |
| `viewCountChanged` | `ecvViewManager.h:142` | Emitted at `ecvViewManager.cpp:106,123` — no subscribers found |
| `activeLayoutChanged` | `ecvViewManager.h:138` | **Never emitted** — declaration only |
| `representationAdded` | `ecvRepresentationManager.h:71` | Emitted at `ecvRepresentationManager.cpp:48` — no subscribers |
| `representationRemoved` | `ecvRepresentationManager.h:72` | Emitted at `ecvRepresentationManager.cpp:90,108,124` — no subscribers |
| `representationChanged` | `ecvRepresentationManager.h:73` | **Never emitted** — declaration only |

---

## ecvGenericGLDisplay — Widget ↔ Display Registry

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecvGenericGLDisplay.h:206-213` |
| **Implementation** | `libs/CV_db/src/ecvGenericGLDisplay.cpp:17-20, 85-100` |

### Internal Storage

```cpp
// ecvGenericGLDisplay.cpp:17-20
namespace {
QMutex s_registryMutex;
QMap<QWidget*, ecvGenericGLDisplay*> s_displayRegistry;
}
```

### Static Methods

| Method | Lines | Description |
|--------|-------|-------------|
| `FromWidget(QWidget*)` | `85-89` | Thread-safe lookup: `QWidget*` → `ecvGenericGLDisplay*` |
| `RegisterGLDisplay(QWidget*, display)` | `91-94` | Thread-safe insert |
| `UnregisterGLDisplay(QWidget*)` | `96-99` | Thread-safe remove |

### Usage

- **Registration**: `ecvGLView::Create()` at `ecvGLView.cpp:76` — `RegisterGLDisplay(m_vtkWidget, view)`
- **Unregistration**: `ecvGLView::~ecvGLView()` at `ecvGLView.cpp:54` — `UnregisterGLDisplay(m_vtkWidget)`
- **Lookup**: `QVTKWidgetCustom::mousePressEvent()` at `QVTKWidgetCustom.cpp:548` — `FromWidget(this)` to determine which display was clicked

---

## ecvViewLayoutProxy — Binary Tree Layout Model

> **ParaView equivalent**: `vtkSMViewLayoutProxy`

| File | Path |
|------|------|
| **Header** | `libs/CV_db/include/ecvViewLayoutProxy.h` |
| **Implementation** | `libs/CV_db/src/ecvViewLayoutProxy.cpp` |

### Purpose

Models a binary tree of split cells (heap-indexed vector). Each leaf cell can hold an `ecvGenericGLDisplay*` (a view). The tree supports horizontal/vertical splits, collapse, swap, and fraction adjustment.

### Key Methods (Header Lines)

| Method | Line Range | Description |
|--------|-----------|-------------|
| `Split(index, dir, fraction)` | `:45-50` | Split a leaf into two children |
| `AssignView(index, view)` | `:52` | Place a view in a leaf cell |
| `RemoveView(view)` | `:54` | Remove view from tree |
| `CollapseCell(index)` | `:56` | Merge two children back into parent |
| `SwapCells(idx1, idx2)` | `:58` | Swap two leaf cells |
| `SetSplitFraction(index, frac)` | `:60` | Adjust split ratio |
| `EqualizeSplits()` | `:62` | Make all splits 50/50 |
| `MaximizeCell(index)` | `:64` | Toggle maximize/restore |
| `GetView(index)` | `:94` | Get view at leaf cell |
| `GetViews()` | `:96` | All views in tree order |
| `GetDirection(index)` | `:98` | Split direction at node |
| `GetSplitFraction(index)` | `:100` | Split fraction at node |
| Static `firstChild/secondChild/parent` | `:117-121` | Heap index math |

### Signal: `layoutChanged`

**Location**: `ecvViewLayoutProxy.h:162`

Connected in `ecvMultiViewWidget.cpp:80`:
```cpp
connect(m_layout, &ecvViewLayoutProxy::layoutChanged,
        this, &ecvMultiViewWidget::reload);
```

Triggers `reload()` which reconstructs the Qt splitter tree from the proxy's binary tree.

---

## Properties Panel — "Current Display" Dropdown

| File | Path |
|------|------|
| **Implementation** | `app/db_tree/ecvPropertiesTreeDelegate.cpp` |

### Row Creation

**Location**: `ecvPropertiesTreeDelegate.cpp:598`

Adds a persistent editor row with label "Current Display" and type `OBJECT_CURRENT_DISPLAY`.

### Combobox Population

**Location**: `ecvPropertiesTreeDelegate.cpp:1351-1365`

```cpp
case OBJECT_CURRENT_DISPLAY: {
    QComboBox* comboBox = new QComboBox(parent);
    const auto& views = ecvViewManager::instance().getAllViews();
    for (auto* view : views) {
        if (view) {
            comboBox->addItem(view->getTitle(), view->getUniqueID());
        }
    }
    connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ccPropertiesTreeDelegate::objectDisplayIndexChanged);
}
```

### On Selection Change

**Location**: `ecvPropertiesTreeDelegate.cpp:3734-3750` — `objectDisplayIndexChanged`

Resolves `findView(viewID)` → `setDisplay_recursive(targetDisplay)` → `redrawAll()`.

---

## DB Tree — "Move to View" Context Menu

| File | Path |
|------|------|
| **Implementation** | `app/db_tree/ecvDBRoot.cpp` |

### Menu Build

**Location**: `ecvDBRoot.cpp:2462-2478`

```cpp
if (toggleVisibility && ecvViewManager::instance().viewCount() > 1) {
    QMenu* moveMenu = menu.addMenu(tr("Move to View"));
    moveMenu->addAction(tr("None (All Views)"), [this, selectedIndexes]() {
        moveSelectedToView(nullptr, selectedIndexes);
    });
    for (auto* view : ecvViewManager::instance().getAllViews()) {
        moveMenu->addAction(view->getTitle(), [this, view, selectedIndexes]() {
            moveSelectedToView(view, selectedIndexes);
        });
    }
}
```

Submenu only appears when `viewCount() > 1`.

### Action: `moveSelectedToView`

**Location**: `ecvDBRoot.cpp:2493-2502`

```cpp
void ccDBRoot::moveSelectedToView(ecvGenericGLDisplay* view,
                                  const QModelIndexList& indexes) {
    for (const auto& idx : indexes) {
        auto* item = static_cast<ccHObject*>(idx.internalPointer());
        if (!item) continue;
        item->setDisplay_recursive(view);
    }
    ecvViewManager::instance().redrawAll();
    updatePropertiesView();
}
```

---

## Primary View Draw Path (ecvDisplayTools::RedrawDisplay)

| File | Path |
|------|------|
| **Implementation** | `libs/CV_db/src/ecvDisplayTools.cpp` |

### Entry: `RedrawDisplay()`

**Location**: `ecvDisplayTools.cpp:3243+`

```
1. Check update flags, capture mode                              [line ~3285-3330]
2. Per-view delegation to secondary views                        [line ~3305-3317]
3. beginPrimaryRender()                                          [line ~3320]
4. Build CC_DRAW_CONTEXT                                         [line ~3340-3380]

3D pass — Draw3D():                                              [line ~3425-3470]
  5. drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND
  6. m_globalDBRoot->draw(CONTEXT)                               [line ~3445]
  7. m_winDBRoot->draw(CONTEXT)                                  [line ~3449]

2D foreground — DrawForeground():                                [line ~3583-3597]
  8. drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND
  9. m_globalDBRoot->draw(CONTEXT)                               [line ~3594]
 10. m_winDBRoot->draw(CONTEXT)                                  [line ~3596]
```

The primary view's `CONTEXT.display` is set to `ecvDisplayTools::TheInstance()` (the singleton), which is how `isDisplayedIn()` knows it's the primary window.

---

## Known Issues & Future Work

### Multi-Window Issues

| Issue | Status | Location |
|-------|--------|----------|
| Right-click "Show in View" / "Hide in View" context menu | Not yet implemented | Planned for `ecvDBRoot.cpp` |
| Per-view scalar field switching | Properties struct supports it, not wired to UI | `ecvViewRepresentation.h:57-58` |
| Per-view render mode switching | Properties struct supports it, not wired to UI | `ecvViewRepresentation.h:55` |
| Layout persistence across sessions | `saveLayout()` / `restoreLayout()` exist, not integrated into session save | `ecvViewManager.h:114-120` |
| `activeLayoutChanged` signal never emitted | Declared but no `emit` call exists | `ecvViewManager.h:138` |
| `representationChanged` signal never emitted | Declared but no `emit` call exists | `ecvRepresentationManager.h:73` |
| `activeSourceChanged`, `viewRegistered`, `viewUnregistered`, `viewCountChanged` | Emitted but no subscribers | Available for future UI extensions |

---

## cc2DLabel Optimization Plan (CloudCompare Alignment)

> Based on detailed comparison of CloudCompare (`/Users/asher/develop/code/autopilot/MVS/CloudCompare` (macOS) / `/home/ludahai/develop/code/github/CloudCompare` (Linux)) and ACloudViewer (`cc2DLabel` VTK implementation).

### Background: Key Differences Between CloudCompare and ACloudViewer

| Dimension | CloudCompare (Raw OpenGL) | ACloudViewer (VTK Backend) |
|-----------|--------------------------|---------------------------|
| **Label body** | Self-drawn `GL_QUADS` + `GL_LINE_LOOP` rectangle with precise QFontMetrics sizing | `vtkCaptionActor2D` widget with `SetPosition2` heuristic sizing |
| **ABC text** | `displayText` → QPainter → texture → GL textured quad | `vtkTextActor` via `DisplayText` → `addText/updateText` |
| **Leader/arrow** | Self-drawn `GL_TRIANGLE_FAN` wedge from ROI edge to projected 3D point | `vtkCaptionActor2D` built-in 2D leader line (no wedge control) |
| **Position** | `m_screenPos` normalized [0,1], every frame: `xStart = glW * m_screenPos[0]` | Same normalized, passed to `vtkCaptionRepresentation::SetPosition()` |
| **Resize** | `context.glW * m_screenPos` auto-adapts; `m_labelROI` recomputed each frame | `SetPosition2` uses adaptive `scaleFactor = clamp(800/winH, 0.6, 1.6)` |
| **Text rendering** | QPainter → QImage → `QOpenGLTexture` → GL textured quad | `vtkTextActor` (VTK internal font rasterizer, Arial) |
| **Mouse drag** | `move2D()` updates `m_screenPos` as normalized delta | Same `move2D()` mechanism |
| **Collapse** | `acceptClick(MiddleButton)` toggles `m_showFullBody` | Same mechanism |
| **HiDPI** | `move2D` receives `dx * devicePixelRatio`, draws in viewport coords | Mixed logical/physical coords with potential inconsistency |

### Optimization Items (by Priority)

#### Phase 1: Critical Bug Fixes (P0)

##### O1: `Redraw2DLabel` Filter Logic Bug

**Problem**: When `isDisplayedIn2D() == true && isPointLegendDisplayed() == false`, `Redraw2DLabel` **skips** the label refresh. This means camera movements don't update caption position for labels that show 2D panel but not point legend.

**Location**: `libs/CV_db/src/ecvDisplayTools.cpp:3684-3685`

**Current code**: `if (!l || (l->isDisplayedIn2D() && !l->isPointLegendDisplayed())) continue;`

**Fix**: Change condition to skip only when label needs neither 2D panel nor legend:
```cpp
if (!l || (!l->isDisplayedIn2D() && !l->isPointLegendDisplayed())) continue;
```

**CloudCompare reference**: `ccGLWindowInterface.cpp` always redraws labels on camera change — no filter.

---

##### O2: HiDPI `setPosition` Inconsistency

**Problem**: When placing a label after picking, `setPosition` divides by `glViewport` width/height (physical pixels) instead of logical dimensions. But `drawMeOnly2D` uses `logicalW = context.glW / devicePixelRatio` to convert `m_screenPos` back. On HiDPI screens (DPR=2), the label appears at half the intended position.

**Location**: `libs/CV_db/src/ecvDisplayTools.cpp:855-859`

**Fix**: Divide by logical dimensions instead:
```cpp
float logicalW = viewport.width() / devicePixelRatio;
float logicalH = viewport.height() / devicePixelRatio;
label->setPosition(x / logicalW, 1.0f - y / logicalH);
```

**CloudCompare reference**: `cc2DLabel.cpp:1585-1589` — `xStart = context.glW * m_screenPos[0]` where `glW` is the viewport width (consistent).

---

##### O3: `WIDGET_T2D` Fall-Through in `drawWidgets`

**Problem**: The `WIDGET_T2D` case in `VtkDisplayTools::drawWidgets` lacks a `break` statement, causing unintentional fall-through to `WIDGET_LINE_3D`.

**Location**: `libs/VtkEngine/Visualization/VtkDisplayTools.cpp:1213-1241`

**Fix**: Add `break;` after `WIDGET_T2D` case.

---

#### Phase 2: Display Quality (P1)

##### O4: Caption Width Should Be Content-Aware

**Problem**: `SetPosition2(baseW, captionH)` uses `baseW = clamp(0.30 * scaleFactor, 0.18, 0.50)` — a fixed ratio that's too wide for short text and too narrow for long measurement data.

**Current location**: `VtkVis.cpp:1932-1941` (updateCaption), `VtkVis.cpp:2028-2037` (addCaption)

**Fix**: Estimate width from longest line length:
```cpp
size_t maxLineLen = 0;
for (auto& line : split_by_newline(text)) {
    maxLineLen = std::max(maxLineLen, line.size());
}
const double charWidthRatio = 0.012 * scaleFactor;  // ~0.012 per char at 800px
const double baseW = std::clamp(charWidthRatio * maxLineLen + 0.04, 0.12, 0.55);
```

**CloudCompare reference**: `cc2DLabel.cpp:1537-1578` — uses exact `QFontMetrics::width()` for each column + margins.

---

##### O5: Tab Column Alignment with Proportional Fonts

**Problem**: Tab content is padded with ASCII spaces to align columns. With proportional fonts (like Arial used by `vtkTextActor`), columns don't align properly.

**Location**: `ecv2DLabel.cpp:1633-1658`

**Options**:
- A) Use monospace font (Courier) for tab body text via `vtkTextProperty::SetFontFamilyToCourier()`
- B) Split each column into a separate `vtkTextActor` with calculated X offset
- C) (Minimal) Increase space padding multiplier from average to max character width

**CloudCompare reference**: Same issue exists (`cc2DLabel.cpp:1554-1558`), but less visible with OpenGL texture-based text.

---

##### O6: Caption Background Styling Control

**Problem**: The label body visual is entirely determined by `vtkCaptionActor2D`'s built-in frame. Can't control background opacity, border radius, or exact padding independently.

**Location**: `VtkVis.cpp:2040-2070` (addCaption text property setup)

**Current**:
```cpp
textProperty->SetBackgroundColor(1.0 - r, 1.0 - g, 1.0 - b);
textProperty->SetBackgroundOpacity(std::max(a, 0.55));
textProperty->SetLineSpacing(1.2);
```

**Improvement**: Adjust `BackgroundOpacity` to match the `defaultBkgColor.a` value from `CC_DRAW_CONTEXT` rather than hard-coded 0.55. For long-term improvement, consider replacing `vtkCaptionActor2D` with a combination of:
- `vtkActor2D` + `vtkPolyDataMapper2D` for the background rectangle
- `vtkTextActor` for the text
- Custom leader line via `vtkPolyData` lines

**CloudCompare reference**: `cc2DLabel.cpp:1705-1725` — explicit `GL_QUADS` fill + `GL_LINE_LOOP` border with exact color control.

---

##### O7: Leader Line (Arrow) Customization

**Problem**: VTK's `vtkCaptionActor2D` leader line is a simple straight line. CloudCompare draws a wedge-shaped arrow that visually connects the label's edge to the projected 3D point.

**Location**: `VtkVis.cpp:2042` — `actor2D->ThreeDimensionalLeaderOff()`; uses 2D straight leader.

**CloudCompare reference**: `cc2DLabel.cpp:1620-1702` — calculates arrow direction from `m_labelROI` center to nearest `pos2D` of picked points, draws `GL_TRIANGLE_FAN` wedge.

**Long-term**: Custom `vtkCaptionRepresentation` subclass or separate `vtkPolyDataMapper2D` for arrow.

---

#### Phase 3: Interaction Quality (P2)

##### O8: Drag Doesn't Immediately Update Caption Position

**Problem**: After `move2D()` updates `m_screenPos`, the VTK caption widget doesn't update until the next full redraw cycle. On some platforms this causes visible lag during drag.

**Location**: `ecv2DLabel.cpp:231-238` (move2D), `QVTKWidgetCustom.cpp:1246-1270` (drag dispatch)

**Fix**: After `move2D()`, immediately call `ecvDisplayTools::Redraw2DLabel()` or `update2DLabelView()` to force caption position sync:
```cpp
// In QVTKWidgetCustom after move2D loop:
ecvDisplayTools::Redraw2DLabel();  // already called at line ~1270
```

Verify: `Redraw2DLabel()` must not be filtered by O1's bug condition.

---

##### O9: Hover Visual Feedback (Optional Enhancement)

**Problem**: No visual feedback when mouse hovers over a label. CloudCompare also lacks this.

**Suggestion**: Add `vtkCaptionWidget::AddObserver(vtkCommand::EnterEvent, ...)` to change border color on hover.

---

##### O10: Collapse Toggle Only Works with Middle Button

**Problem**: `acceptClick` only responds to `Qt::MiddleButton`. Many laptop users don't have a middle mouse button.

**Location**: `ecv2DLabel.cpp:852-866`

**Fix**: Also accept `Qt::RightButton`:
```cpp
if (button == Qt::MiddleButton || button == Qt::RightButton) {
    // ... toggle m_showFullBody
}
```

**CloudCompare reference**: `cc2DLabel.cpp:968-976` — also middle button only. This would be an improvement over both.

---

#### Phase 4: Architecture Cleanup (P3)

##### O11: `pos2D` Documentation Bug

**Problem**: `ecv2DLabel.h` header comment says `pos2D` is updated in `drawMeOnly3D`, but it's actually updated in `drawMeOnly2D`.

**Location**: `libs/CV_db/include/ecv2DLabel.h:128-131`

**Fix**: Update comment to say "Updated in `drawMeOnly2D()` via `camera.project()`".

---

##### O12: Platform Y-Coordinate Handling

**Problem**: `drawMeOnly2D` uses `#ifdef Q_OS_MAC` for Y-coordinate flipping. This is fragile and hard to test.

**Location**: `ecv2DLabel.cpp:1710-1726`

**Fix**: Unify Y handling by converting to VTK's coordinate convention (bottom-left origin) once at the beginning of `drawMeOnly2D`, then using consistent math throughout. The `#ifdef` should only exist at the final conversion point.

---

##### O13: Missing Signal Emissions

**Problem**: `representationChanged` and `activeLayoutChanged` signals are declared but never emitted.

**Locations**:
- `ecvRepresentationManager.h:73` — `representationChanged`
- `ecvViewManager.h:138` — `activeLayoutChanged`

**Fix**:
- Emit `representationChanged` in `ecvViewRepresentation::setProperties()` and `setVisible()`
- Emit `activeLayoutChanged` in `ecvViewManager::registerLayout()` and when active view changes layout context

---

### Implementation Priority Matrix

```
        Impact
High ┃ O1  O2  O4
     ┃ O6
     ┃
Med  ┃ O3  O8  O10
     ┃ O5
     ┃
Low  ┃ O7  O9  O11
     ┃ O12 O13
     ┗━━━━━━━━━━━━━━━
       Low  Med  High
             Effort
```

### Recommended Phases

| Phase | Items | Est. Effort | Description |
|-------|-------|-------------|-------------|
| **1** | O1, O2, O3 | 1-2 hours | Critical bug fixes (HiDPI, refresh, fall-through) |
| **2** | O4, O6 | 2-3 hours | Caption display quality (width, background) |
| **3** | O8, O10 | 1 hour | Interaction improvements (drag, collapse) |
| **4** | O5, O7, O11, O12, O13 | 4-6 hours | Architecture and long-term improvements |

---

## cc2DLabel Rendering Architecture: QPainter Overlay Strategy (Plan D+)

> This section documents the recommended approach for replacing the VTK `vtkCaptionActor2D` based label rendering with QPainter overlay, achieving near-100% visual and behavioral parity with CloudCompare while maintaining all existing mouse interactions.

### Background: Why Replace the VTK Caption Widget?

| Problem | Root Cause |
|---------|------------|
| Caption box size not content-aware | `SetPosition2` uses heuristic ratio, not actual text metrics |
| Tab column misalignment | ASCII space padding + proportional font (Arial) in vtkTextActor |
| No wedge-shaped arrow (leader) | `vtkCaptionActor2D` only supports straight 2D leader line |
| Cannot control background opacity/color precisely | VTK text property BackgroundOpacity is limited |
| Font rendering quality lower than Qt | VTK's internal rasterizer vs Qt's FreeType + subpixel rendering |
| Core profile compatibility concern | Raw GL calls (`glBegin/glEnd`) not available in OpenGL 3.2+ core profile |
| Performance with many labels | Each label = 1 caption widget + N text actors = heavy VTK overhead |

### VTK+OpenGL Hybrid Rendering Options Evaluated

| Approach | Description | Pros | Cons | Verdict |
|----------|-------------|------|------|---------|
| **A: vtkRenderer EndEvent + Raw OpenGL** | Inject GL calls after VTK render | Closest to CloudCompare | **Core profile kills `glBegin/glEnd`** | Rejected |
| **B: vtkActor2D + vtkPolyDataMapper2D** | Use VTK 2D pipeline for rectangle + text + lines | Pure VTK, core-profile compatible | Multiple actors per label, complex lifecycle | Possible but heavy |
| **C: QPainter Overlay** | Override `paintEvent` in `QVTKWidgetCustom`, draw 2D labels with QPainter after VTK render | Best text quality, simple code, efficient | Does not participate in VTK picking | **Recommended** |
| **D+: Hybrid (3D=VTK, 2D=QPainter)** | Keep VTK for 3D markers + picking; use QPainter for 2D label panel + ABC text + arrows | Best of both: VTK picking + Qt rendering | 2 interaction adaptations needed | **Selected** |

### Architecture: Plan D+ (Selected)

```
Rendering Pipeline:
─────────────────

ecvGLView::redraw()
    │
    ├── [VTK 3D Pass]  ─── ccHObject::draw(3D context) ─── cc2DLabel::drawMeOnly3D()
    │                       │
    │                       ├── 3D point markers (WIDGET_POINT)         ← VTK (keep)
    │                       ├── 3D line between 2 points (WIDGET_LINE)  ← VTK (keep)
    │                       └── 3D triangle face for 3-point labels     ← VTK (keep)
    │
    ├── [VTK 2D Pass]  ─── ccHObject::draw(2D context) ─── cc2DLabel::drawMeOnly2D()
    │                       │
    │                       ├── 3D → 2D projection: camera.project(P3D, pos2D)    ← keep
    │                       ├── m_labelROI calculation (QFontMetrics)               ← keep
    │                       ├── m_screenPos → screen coords conversion             ← keep
    │                       ├── m_historyMessage (tab content) assembly             ← keep
    │                       │
    │                       ├── [REMOVED] WIDGET_CAPTION → VtkVis::addCaption
    │                       ├── [REMOVED] DisplayText(ABC) → VtkVis::addText
    │                       │
    │                       └── [NEW] Store computed layout in m_overlayData
    │                            ├── backgroundRect (m_labelROI in absolute coords)
    │                            ├── textLines (title + body rows + formatting)
    │                            ├── arrowPoints (wedge from ROI edge to pos2D centroid)
    │                            ├── legendTexts (ABC + positions)
    │                            └── selected/colors/fonts
    │
    ├── [VTK Render]  ─── vtkRenderWindow::Render()  ← 3D geometry only now
    │
    └── [QPainter Pass]  ─── QVTKWidgetCustom::paintOverlay()     ← NEW
                              │
                              └── For each cc2DLabel with m_overlayData:
                                   ├── QPainter::fillRect(backgroundRect, bkgColor)
                                   ├── QPainter::drawRect(backgroundRect, borderColor)
                                   ├── QPainter::drawPolygon(arrowPoints)    ← wedge arrow
                                   ├── QPainter::drawText(title, titleFont)
                                   ├── QPainter::drawText(body rows, bodyFont)
                                   └── For each legend (A/B/C):
                                        ├── QPainter::fillRect(textBkg, semiTransparent)
                                        └── QPainter::drawText(letter, pos)
```

### Interaction Compatibility Matrix

| # | Interaction | CloudCompare Method | ACloudViewer Current | After Plan D+ | Compatible? |
|---|------------|--------------------|--------------------|---------------|-------------|
| 1 | **Label body drag** | `move2D()` → `m_screenPos += delta/screenSize` (`cc2DLabel.cpp:268-276`) | `QVTKWidgetCustom::updateActivateditems` (1246-1270) → `move2D()` | **No change** — `move2D()` only operates on `m_screenPos` (normalized), QPainter reads it | YES |
| 2 | **Middle-click collapse** | `acceptClick(MiddleButton)` → `m_labelROI.contains()` → toggle `m_showFullBody` (`cc2DLabel.cpp:968-976`) | `QVTKWidgetCustom::mouseReleaseEvent` (1303-1319) → `acceptClick()` | **No change** — `m_labelROI` is computed in `drawMeOnly2D()` every frame, `acceptClick` is screen-space hit test | YES |
| 3 | **3D marker picking** | `pointPicking(clickPos, camera)` — 3D ray vs marker sphere (`cc2DLabel.cpp:1847-1922`) | `ecvDisplayTools.cpp:1241-1265` → `pointPicking()`, VTK 3D picking | **No change** — 3D markers still use VTK `WIDGET_POINT` rendering and picking | YES |
| 4 | **Label selection (DB tree)** | FAST_PICKING → `cc2DLabel` entity → `entitySelectionChanged` (`ccGLWindowInterface.cpp:5713-5735`) | `CustomVtkCaptionWidget` click → `SetAssociatedLabel()` → DB tree select | **ADAPTATION NEEDED** — QPainter labels not in VTK picking pipeline | See A1 below |
| 5 | **Active items list** | `updateActiveItemsList` → FAST_PICKING → `m_activeItems.push_back(label)` (`ccGLWindowInterface.cpp:1981-2024`) | `QVTKWidgetCustom::updateActivateditems` → VTK widget search | **ADAPTATION NEEDED** — Need 2D ROI hit test instead of VTK picking | See A2 below |
| 6 | **2D connecting segments** | `GL_LINES`/`GL_LINE_LOOP` in 2D overlay between projected pos2D (`cc2DLabel.cpp:1292-1315`) | `WIDGET_LINE_3D` in 3D space | **IMPROVED** — `QPainter::drawLine()` in 2D (matches CC exactly) | YES (better) |
| 7 | **Wedge arrow** | `GL_TRIANGLE_FAN` from ROI edge to pos2D centroid (`cc2DLabel.cpp:1620-1702`) | None (VTK straight leader line) | **NEW** — `QPainter::drawPolygon()` wedge (matches CC exactly) | YES (new) |
| 8 | **Camera change update** | `pos2D` recomputed every frame in `drawMeOnly3D/2D` via `camera.project()` | `drawMeOnly2D` projects + `forceRedraw` ensures updates | **No change** — `drawMeOnly2D` still computes layout every frame | YES |
| 9 | **Mouse wheel zoom** | `onWheelEvent` → `redraw()` → pos2D recomputed, `m_screenPos` unchanged | Fixed: `forceRedraw` ensures label update on zoom | **No change** | YES |
| 10 | **Window resize** | `xStart = glW * m_screenPos[0]` recomputed every frame | Same logic | **No change** — QPainter uses `widget->width() * m_screenPos[0]` | YES |
| 11 | **Hover highlight** | Not in CloudCompare | Not in ACloudViewer | Optional: `mouseMoveEvent` → ROI hit test → cursor change | N/A |
| 12 | **Selected state display** | `isSelected()` → red markers/borders (`cc2DLabel.cpp:1133, 1303, 1600`) | `isSelected()` → red | **No change** — QPainter checks `isSelected()` for border color | YES |
| 13 | **Right-click context menu** | Not specific to cc2DLabel | Not specific | **No change** | YES |

**Result: 10/13 fully compatible, 2 need small adaptation, 1 optional enhancement**

### Adaptation A1: Label Panel Click → Selection

**Problem**: `CustomVtkCaptionWidget` currently handles clicks on the caption → selects entity in DB tree. QPainter-drawn labels are invisible to VTK picking.

**Solution**: Add 2D hit test in `QVTKWidgetCustom::mousePressEvent()` before VTK picking:

```
Location: libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp:548+

// Before VTK picking, check if click hits any cc2DLabel's ROI
for (each cc2DLabel in scene) {
    QRect roi(label->m_lastScreenPos[0], label->m_lastScreenPos[1],
              label->m_labelROI.width(), label->m_labelROI.height());
    // Convert mouse pos to label coordinate system (account for Y flip)
    QPoint labelPos(event->pos().x(),
                    widget->height() - 1 - event->pos().y());
    if (roi.contains(labelPos)) {
        // Select this label in DB tree (same as CustomVtkCaptionWidget::widgetClicked)
        emit entitySelected(label);
        return;  // Don't fall through to VTK picking
    }
}
// Fall through to VTK picking for 3D objects
```

**CloudCompare equivalent**: `ccGLWindowInterface.cpp:6065-6079` — `doPicking` with `INTERACT_2D_ITEMS` finds cc2DLabel via FAST_PICKING, then `entitySelectionChanged`.

**Estimated code**: ~25 lines

### Adaptation A2: Active Items List for Drag

**Problem**: `updateActivateditems` currently finds active labels through VTK widget search. QPainter labels need 2D ROI hit test.

**Solution**: Add 2D ROI-based search in `QVTKWidgetCustom::updateActivateditems()`:

```
Location: libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp:1246+

// Before VTK widget search, check 2D labels
for (each cc2DLabel in scene) {
    QRect roi(...);
    if (roi.contains(mousePos)) {
        m_activeItems.push_back(label);
        return;
    }
}
// Fall through to existing VTK widget-based search
```

**CloudCompare equivalent**: `ccGLWindowInterface.cpp:1981-2024` — `updateActiveItemsList` uses FAST_PICKING → checks if hit is cc2DLabel → adds to `m_activeItems`.

**Estimated code**: ~20 lines

### Implementation Phases

#### Phase 1: Infrastructure (No Visual Change)

| Step | File | Change | Lines |
|------|------|--------|-------|
| 1.1 | `QVTKWidgetCustom.h` | Add `paintOverlay()` virtual method, `m_overlayCallback` member | ~10 |
| 1.2 | `QVTKWidgetCustom.cpp` | Override `paintEvent()` — call `QVTKOpenGLNativeWidget::paintEvent()` first, then `paintOverlay()` | ~15 |
| 1.3 | `ecvGLView.h/cpp` | Register overlay callback that iterates cc2DLabel objects in scene | ~30 |
| 1.4 | `ecv2DLabel.h` | Add `struct OverlayData` and `drawOverlay2D(QPainter&, int w, int h)` declaration | ~20 |

#### Phase 2: Rendering Replacement (Visual Change)

| Step | File | Change | Lines |
|------|------|--------|-------|
| 2.1 | `ecv2DLabel.cpp:drawMeOnly2D()` | Replace `WIDGET_CAPTION` + `DisplayText` calls with `m_overlayData` population | ~50 |
| 2.2 | `ecv2DLabel.cpp` (new method) | Implement `drawOverlay2D()`: `QPainter` calls for background, text, arrows, ABC | ~150 |
| 2.3 | `ecv2DLabel.cpp:drawMeOnly2D()` | Port CloudCompare's arrow/wedge calculation (`cc2DLabel.cpp:1620-1702`) | ~80 |

#### Phase 3: Interaction Adaptation

| Step | File | Change | Lines |
|------|------|--------|-------|
| 3.1 | `QVTKWidgetCustom.cpp:mousePressEvent` | Add 2D label ROI hit test before VTK picking (Adaptation A1) | ~25 |
| 3.2 | `QVTKWidgetCustom.cpp:updateActivateditems` | Add 2D label ROI search for drag activation (Adaptation A2) | ~20 |

#### Phase 4: Cleanup

| Step | File | Change | Lines |
|------|------|--------|-------|
| 4.1 | `VtkDisplayTools.cpp` | Remove `WIDGET_CAPTION` handling for cc2DLabel path | ~20 |
| 4.2 | `VtkVis.cpp` | `addCaption`/`updateCaption` now only used by non-label captions | ~0 (keep for other uses) |
| 4.3 | `ecvDisplayTools.cpp` | Update `Redraw2DLabel` to trigger QPainter overlay update | ~10 |

**Total estimated new/changed code: ~430 lines**

### Performance Comparison

| Metric | Current (VTK) | After (QPainter) |
|--------|--------------|------------------|
| Actors per label | 1 `vtkCaptionWidget` + N `vtkTextActor`s + widget overhead | 0 VTK actors for 2D elements |
| 100 labels render time | ~100 VTK actor updates per frame | Single QPainter pass |
| Text quality | VTK rasterizer (no subpixel rendering) | Qt FreeType (subpixel + hinting) |
| Font metrics accuracy | VTK internal metrics | QFontMetrics (pixel-perfect) |
| Memory per label (2D) | ~3-5 VTK objects + texture cache | ~200 bytes OverlayData struct |
| HiDPI handling | Manual `devicePixelRatio` conversion | Automatic (QPainter respects DPR) |

### CloudCompare Code Reuse Map

| CloudCompare Function | File:Lines | Reuse in ACloudViewer |
|----------------------|------------|----------------------|
| Arrow wedge calculation | `cc2DLabel.cpp:1620-1702` | Port to `drawOverlay2D()` — calculate arrow points from `m_labelROI` to `pos2D` centroid |
| Background rectangle | `cc2DLabel.cpp:1705-1712` | `QPainter::fillRect(m_labelROI, bkgColor)` |
| Border rectangle | `cc2DLabel.cpp:1714-1725` | `QPainter::drawRect(m_labelROI, borderPen)` |
| Title text | `cc2DLabel.cpp:1766-1775` | `QPainter::drawText(titleRect, title, titleFont)` |
| Body text (tab rows) | `cc2DLabel.cpp:1778-1836` | `QPainter::drawText(rowRect, rowText, bodyFont)` |
| ABC legend text | `cc2DLabel.cpp:1317-1342` | `QPainter::drawText(pos, letter)` with filled background rect |
| 2D connecting segments | `cc2DLabel.cpp:1292-1315` | `QPainter::drawLine(pos2D[0], pos2D[1])` |
| `m_labelROI` computation | `cc2DLabel.cpp:1573-1582` | **Already aligned** in ACloudViewer `ecv2DLabel.cpp:1563-1601` |
| `m_screenPos` → screen | `cc2DLabel.cpp:1585-1590` | **Already aligned** in ACloudViewer `ecv2DLabel.cpp:1576-1579` |
| `move2D()` | `cc2DLabel.cpp:268-276` | **Already aligned** in ACloudViewer `ecv2DLabel.cpp:231-238` |
| `acceptClick()` | `cc2DLabel.cpp:968-976` | **Already aligned** in ACloudViewer `ecv2DLabel.cpp:852-866` |
| `pointPicking()` | `cc2DLabel.cpp:1847-1922` | **Already aligned** in ACloudViewer `ecv2DLabel.cpp:1743-1811` |

---

## Next Steps — Architecture Evolution (2026-04-30)

### Completed Milestones

| Phase | Status | Key Result |
|-------|--------|------------|
| A–K (multi-window layout) | **DONE** | ParaView-aligned KD-tree layout, Tab system, per-view context, push/pull eliminated |
| L (singleton API cleanup) | **DONE** | `TheInstance()`/`Init()`/`ReleaseInstance()` removed, 0 non-core `ecvDisplayTools::` refs |
| M1–M5 (structural gaps) | **DONE** | VtkDisplayTools 拆分、QVTKWidgetCustom 统一、ecvGLView 唯一视图、2D Overlay 参数化、Python API |
| N1–N5 (effectiveCtx elimination) | **DONE** | 307→76 calls, all remaining in acceptable patterns |

### Remaining Architectural Gap

~~The primary remaining gap is that the **main view is still a `VtkDisplayTools*` instance** (not an `ecvGLView`)~~ — **已解决** (Phase M1-M3, 2026-04-30).

**当前状态**: Phase M1-M6 + Phase N + Phase O 全部完成 ✅。Per-view camera undo/redo 已在 VtkVis + MainWindow 中实现 ✅。ParaView 对齐率 94.5%。

See **[multi-window-refactor-roadmap-Vtk-vs-CC.md](multi-window-refactor-roadmap-Vtk-vs-CC.md) §10** for the Phase M TODOs (all ✅), and **[multi-window-paraview-alignment-design.md](multi-window-paraview-alignment-design.md)** for the full alignment design.

| Phase | Title | Priority | Pre-req | Est. | Status |
|-------|-------|----------|---------|------|--------|
| **M1** | VtkDisplayTools 职责拆分 (→纯引擎服务) | HIGH | -- | 2-3 周 | ✅ |
| **M2** | QVTKWidgetCustom 统一 (消除 m_tools ~90+ 引用) | HIGH | M1 partial | 2 周 | ✅ |
| **M3** | ecvGLView 成为唯一视图类型 | HIGH | M1+M2 | 1-2 周 | ✅ |
| **M4** | 2D Overlay 管线参数化 (消除 ScopedHotZoneRender) | MEDIUM | M3 | 1-2 周 | ✅ |
| **M5** | Python API 现代化 | LOW | M1 | 1 周 | ✅ |
| **M6** | Per-View 表示完善 | LOW | -- | 2-3 周 | ✅ |
