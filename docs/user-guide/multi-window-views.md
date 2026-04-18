# ACloudViewer Multi-Window 3D Views User Guide

> Date: 2026-04-13
> Version: 2.0 (ParaView-aligned)

---

## Overview

ACloudViewer now supports **multiple independent 3D view windows**. Each view has its own VTK rendering pipeline (renderer, render window, interactor), viewport camera, and entity visibility settings. This allows you to:

- View the same scene from different camera angles simultaneously
- Show/hide specific entities per view
- Apply different display properties (opacity, point size, render mode) per view
- Work with a primary + secondary views inside the MDI area

---

## Architecture Diagram

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

### Key Components

```
┌─────────────────────┐     ┌──────────────────────┐
│  ecvViewManager     │────▶│  ecvGenericGLDisplay │ (interface)
│  (singleton)        │     └──────────┬───────────┘
│                     │                │ implements
│  - activeView       │     ┌──────────┴───────────┐
│  - allViews[]       │     │                      │
│  - registerView()   │     ▼                      ▼
│  - setActiveView()  │  ecvDisplayTools      ecvGLView
│  - redrawAll()      │  (primary window)     (per-window)
└─────────────────────┘

┌─────────────────────────┐     ┌───────────────────────┐
│ ecvRepresentationManager│────▶│ ecvViewRepresentation │
│ (singleton)             │     │                       │
│                         │     │  - entity             │
│  Key: (entity, view)    │     │  - view               │
│  - getRepresentation()  │     │  - visibility override│
│  - ensureRepresentation │     │  - opacity, pointSize │
│                         │     │  - renderMode         │
└─────────────────────────┘     └───────────────────────┘
```

---

## How to Use

### 1. Creating a New 3D View

There are multiple ways to create a new 3D view:

**Method A: "+" Button on Tab Bar**
Click the small `+` button at the end of the tab bar. This creates a new layout with a fresh 3D view.

**Method B: Split View (from title bar)**
Each view has a title bar with Split Horizontal and Split Vertical buttons on the right side. Clicking these splits the current view into two panels within the same layout tab.

**Method C: Right-click Context Menu**
Right-click on any view's title bar for a context menu with Split Horizontal, Split Vertical, and Close options.

**Method D: Menu / Toolbar**
`Display → New 3D View` or click the New 3D View button in the ViewToolBar.

**Method E: Link Cameras**
`Display → Link Cameras` or the lock icon in the ViewToolBar toggles camera synchronization across all views. When enabled, rotating/panning/zooming any view mirrors the change to all other views in real-time.

All new views share the same scene database (global DB) and inherit the primary view's configuration (background color, oriented box, bubble widget, etc.).

### 2. View Title Bar (ParaView pqViewFrame style)

Each view has a ParaView-style title bar:

```
 ┌──────────────────────────────────────────────────────────┐
 │ [spacer]                    RenderView1  │ ┃ ━ □ ✕ │     │
 │                                          split buttons   │
 ├──────────────────────────────────────────────────────────┤
 │                                                          │
 │              [3D Rendering Content]                      │
 │                                                          │
 └──────────────────────────────────────────────────────────┘
```

- **Title** (right-aligned): View name (RenderView1, RenderView2, etc.)
- **Split Horizontal (┃)**: Splits view side-by-side
- **Split Vertical (━)**: Splits view top-bottom
- **Maximize (□)**: Toggles maximize/restore
- **Close (✕)**: Closes the view (disabled when only 1 view remains)

The active view shows a colored border and bold/underlined title.

### 3. Switching Active View

Simply **click on a view** to make it active. The active view determines:
- Which view receives keyboard/mouse interaction
- Where newly loaded entities will be displayed
- Which view toolbar tools operate on
- The camera parameters shown in the toolbar

All tools (selection, measurement, camera edit, segmentation, etc.) automatically rebind to the active view when you switch.

### 3. Moving Entities Between Views

There are two ways to move entities between views:

#### Method A: Properties Panel (Current Display dropdown)

1. Select an entity in the **DB Tree**
2. In the **Properties** panel, find the **"Current Display"** dropdown
3. Select the target view from the list
4. The entity will immediately move to the selected view

```
 Properties
 ┌─────────────────────────────┐
 │ Name:     bunny.ply         │
 │ Visible:  ☑                 │
 │ Current Display: [View ▼]   │  ◀── Change this
 │ ┌─────────────────────────┐ │
 │ │ None                    │ │  ← Show in all views
 │ │ Primary View            │ │  ← Primary window only
 │ │ 3D View 1001            │ │  ← Secondary view only
 │ │ 3D View 1002            │ │
 │ └─────────────────────────┘ │
 │ Colors:   ☑                 │
 │ ...                         │
 └─────────────────────────────┘
```

**Display association semantics:**

| Setting | Meaning |
|---------|---------|
| **None** | Entity is shown in **all views** (default) |
| **Primary View** | Entity is shown only in the primary window |
| **3D View 1001** | Entity is shown only in that specific secondary view |

#### Method B: DB Tree Context Menu (Right-click)

1. Right-click one or more entities in the **DB Tree**
2. Select **"Move to View"** submenu
3. Choose the target view

```
 DB Tree
 ┌─────────────────────────────┐
 │ ▶ bunny.ply (right-click)   │
 │ ┌─────────────────────────┐ │
 │ │ Toggle                  │ │
 │ │ Toggle Visibility       │ │
 │ │ Delete                  │ │
 │ │ ────────────────────    │ │
 │ │ Move to View ▶          │ │  ◀── Submenu
 │ │ ┌───────────────────┐   │ │
 │ │ │ None (All Views)  │   │ │
 │ │ │ Primary View      │   │ │
 │ │ │ 3D View 1001      │   │ │
 │ │ └───────────────────┘   │ │
 │ │ ────────────────────    │ │
 │ │ Expand                  │ │
 │ │ Collapse                │ │
 │ └─────────────────────────┘ │
 └─────────────────────────────┘
```

> **Note:** The "Move to View" submenu only appears when there are **2 or more views** open.

### 4. Per-View Display Properties (Representation Layer)

Each (entity, view) pair can have independent display properties managed by the **Representation Layer**:

```
 Same entity "bunny.ply" in two views:

 ┌─── Primary View ──────────┐  ┌─── 3D View 1001 ─────────┐
 │                           │  │                          │
 │   [bunny shown as         │  │   [bunny shown as        │
 │    solid surface,         │  │    wireframe,            │
 │    opacity 1.0]           │  │    opacity 0.5]          │
 │                           │  │                          │
 └───────────────────────────┘  └──────────────────────────┘
```

The representation system supports overriding:
- **Visibility** — show/hide per view without changing the entity's global visibility
- **Opacity** — different transparency per view
- **Point Size** — different point rendering sizes
- **Line Width** — different line widths
- **Render Mode** — Points / Wireframe / Surface / Surface+Edges
- **Scalar Field Display** — different SF per view
- **Color/Normal Display** — toggle per view

### 5. Closing a View

Close a secondary view by clicking its MDI sub-window's close button (×). When a view is closed:
1. All representations associated with that view are automatically cleaned up
2. The view is unregistered from `ecvViewManager`
3. If it was the active view, the next available view becomes active
4. Entities previously bound to that view have their display reset to `nullptr` (visible in all remaining views)

---

## Entity Visibility Logic

The draw pipeline uses **three-way logic** to determine if an entity should be drawn in a given view:

```
 isDisplayedIn(display) logic:
 ┌────────────────────────────────────────────────────────┐
 │                                                        │
 │  display == nullptr        → Legacy mode, draw always  │
 │  entity.display == nullptr → Unbound, draw in all      │
 │  entity.display == display → Match, draw here          │
 │  entity.display != display → Skip this view            │
 │                                                        │
 └────────────────────────────────────────────────────────┘
```

Additionally, the **Representation layer** can override visibility:

```
 if (entity passes isDisplayedIn):
   if (representation exists for (entity, view)):
     if (representation has visibility override):
       use representation.isVisible()
```

---

## Draw Pipeline Flow

```
 ecvGLView::redraw()
   │
   ├─ Create CC_DRAW_CONTEXT with display = this
   │
   ├─ globalDBRoot->draw(context)
   │    │
   │    ├─ ccHObject::draw(context)
   │    │    │
   │    │    ├─ Check isDisplayedIn(context.display)
   │    │    ├─ Check representation visibility override
   │    │    │
   │    │    └─ ecvDisplayTools → VtkDisplayTools::draw()
   │    │         │
   │    │         ├─ ScopedVisSwap: temporarily swap
   │    │         │   m_visualizer3D to this view's VtkVis
   │    │         │
   │    │         ├─ drawPointCloud / drawMesh / ...
   │    │         │   (automatically use swapped VtkVis)
   │    │         │
   │    │         └─ ~ScopedVisSwap: restore original
   │    │
   │    └─ Recurse to children
   │
   └─ VtkVis::RenderWindow->Render()
```

---

## Programmatic API

### Creating a new view (C++)

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
#include <ecvHObject.h>
#include <ecvGenericGLDisplay.h>

// Move to a specific view
entity->setDisplay_recursive(targetView);

// Show in all views (unbind)
entity->setDisplay_recursive(nullptr);
```

### Querying views

```cpp
#include <ecvViewManager.h>

auto& mgr = ecvViewManager::instance();
int count = mgr.viewCount();
auto* active = mgr.getActiveView();
const auto& all = mgr.getAllViews();
auto* view = mgr.findView(uniqueID);
```

### Per-view representation

```cpp
#include <ecvRepresentationManager.h>

auto& repMgr = ecvRepresentationManager::instance();

// Get or create representation for (entity, view)
auto* rep = repMgr.ensureRepresentation(entity, view);
rep->setVisible(false);  // Hide in this view only
rep->properties().opacity = 0.5f;
rep->properties().renderMode = ecvViewRepresentation::RenderMode::Wireframe;
rep->setDirty();

// Query
auto reps = repMgr.getRepresentationsForEntity(entity);
auto reps = repMgr.getRepresentationsForView(view);

// Cleanup (automatic on view close)
repMgr.removeRepresentationsForView(view);
```

---

## FAQ

**Q: Can I have more than 2 views?**
A: Yes. Each `Ctrl+Shift+N` creates an additional view. The MDI area supports tiling (`Window → Tile`) and cascading.

**Q: Does each view have its own camera?**
A: Yes. Each view has independent `ecvViewportParameters` (camera center, view matrix, perspective/ortho, zoom).

**Q: Is geometry data duplicated across views?**
A: No. The scene DB (`ccHObject` tree and underlying `vtkPolyData`) is shared. Each view maintains its own VTK actors that reference the shared geometry data.

**Q: What happens when I load a new file?**
A: The entity is added to the global scene DB. If `m_currentDisplay` is `nullptr` (default), it appears in all views. If the active view is set, entities may be associated with the active view via `ecvViewManager::associateToActiveView()`.

**Q: How do I reset an entity to show in all views?**
A: Set its "Current Display" to "None" in the Properties panel, or right-click → "Move to View" → "None (All Views)".
