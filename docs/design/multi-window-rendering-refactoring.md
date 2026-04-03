# ACloudViewer Multi-Window Rendering Refactoring Plan

> Status: **Draft — Pending Review**
> Date: 2026-04-01
> References: ParaView, CloudCompare, MeshLab

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture](#2-current-architecture)
3. [Target Architecture](#3-target-architecture)
4. [Architecture Constraints](#4-architecture-constraints)
5. [Reference Frameworks Comparison](#5-reference-frameworks-comparison)
6. [Phase Breakdown](#6-phase-breakdown)
   - [Phase 1: ecvGenericGLDisplay Interface](#phase-1-ecvgenericgldisplay-interface)
   - [Phase 2: ecvViewManager](#phase-2-ecvviewmanager)
   - [Phase 3: CC_DRAW_CONTEXT + ccDrawableObject](#phase-3-cc_draw_context--ccdrawableobject)
   - [Phase 4: ecvViewRepresentation + ecvRepresentationManager](#phase-4-ecvviewrepresentation--ecvrepresentationmanager)
   - [Phase 5: ccHObject::draw() Per-Window Filtering](#phase-5-cchobjectdraw-per-window-filtering)
   - [Phase 6: ecvDisplayTools Adaption](#phase-6-ecvdisplaytools-adaption)
   - [Phase 7: ecvGLView (Per-Window VTK Pipeline)](#phase-7-ecvglview-per-window-vtk-pipeline)
   - [Phase 8: VtkDisplayTools Routing + Per-View Properties](#phase-8-vtkdisplaytools-routing--per-view-properties)
   - [Phase 9: MainWindow MDI Management](#phase-9-mainwindow-mdi-management)
   - [Phase 10: Compilation Verification + Call-Site Migration](#phase-10-compilation-verification--call-site-migration)
7. [Draw Chain Walkthrough](#7-draw-chain-walkthrough)
8. [Backward Compatibility](#8-backward-compatibility)
9. [Cross-Platform Checklist](#9-cross-platform-checklist)
10. [Performance & Memory Checklist](#10-performance--memory-checklist)
11. [VtkVis viewport Parameter Decision](#11-vtkviss-viewport-parameter-decision)
12. [File Inventory](#12-file-inventory)

---

## 1. Problem Statement

ACloudViewer currently uses a **process-level singleton** `ecvDisplayTools` that holds ALL rendering state (viewport parameters, camera, scene DB roots, interaction state, etc.) in a single instance. This makes multi-window rendering impossible:

- `CC_DRAW_CONTEXT` carries no display/window reference.
- `ccDrawableObject` has no display association (`m_currentDisplay`).
- `VtkDisplayTools` owns a single `VtkVis` / `QVTKWidgetCustom` pair.
- `MainWindow` only manages one 3D view in the MDI area.

The goal is to support **multiple independent 3D rendering windows** — like CloudCompare and ParaView — where each window has its own camera, viewport, and VTK rendering pipeline, while sharing a common scene database.

---

## 2. Current Architecture

### Singleton Chain

```
ecvGenericDisplayTools (static s_genericTools*)
  └── ecvDisplayTools (static s_tools.instance via ecvSingleton<T>)
        └── VtkDisplayTools (concrete, created in MainWindow::initial())
              ├── QVTKWidgetCustom* m_vtkWidget     (single)
              ├── VtkVisPtr m_visualizer3D           (single)
              │     ├── vtkRenderer
              │     ├── vtkRenderWindow
              │     ├── vtkRenderWindowInteractor
              │     ├── cloud_actor_map_    (keyed by viewID string)
              │     └── shape_actor_map_    (keyed by viewID string)
              └── ImageVisPtr m_visualizer2D          (single)
```

### Draw Dispatch (Single Window)

```
ecvDisplayTools::Draw3D()
  → m_globalDBRoot->draw(CONTEXT)
    → ccHObject::draw(CONTEXT)
      → drawMeOnly(CONTEXT)
        → ecvDisplayTools::Draw(context, this)    // static
          → TheInstance()->draw(context, obj)       // virtual
            → VtkDisplayTools::draw(context, obj)   // always uses m_visualizer3D
```

### Key Per-View State on ecvDisplayTools (Instance Members)

| Category | Members |
|----------|---------|
| Window/screen | `m_currentScreen`, `m_mainScreen`, `m_win` |
| Viewport/camera | `m_viewportParams`, `m_glViewport`, `m_viewMatd`, `m_projMatd`, `m_cameraToBBCenterDist`, `m_bbHalfDiag` |
| Scene DB | `m_globalDBRoot` (shared), `m_winDBRoot` (view-local) |
| Interaction | `m_interactionFlags`, `m_pickingMode`, `m_activeItems`, `m_lastMousePos`, etc. |
| Lights | `m_sunLightPos`, `m_customLightPos`, enable flags |
| Rendering | `m_overridenDisplayParameters`, `m_font`, `m_hotZone`, etc. |
| Timers | `m_timer`, `m_scheduleTimer`, `m_shouldBeRefreshed` |

---

## 3. Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MainWindow (MDI)                      │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│   │  MdiSub 1    │ │  MdiSub 2    │ │  MdiSub 3    │   │
│   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
│          │                │                │            │
│   ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐   │
│   │ ecvGLView 1  │ │ ecvGLView 2  │ │ ecvGLView 3  │   │
│   │ (primary,    │ │ (extra)      │ │ (extra)      │   │
│   │  backed by   │ │              │ │              │   │
│   │  singleton)  │ │              │ │              │   │
│   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
│          │                │                │            │
│   ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐   │
│   │ QVTK Widget  │ │ QVTK Widget  │ │ QVTK Widget  │   │
│   │ + VtkVis     │ │ + VtkVis     │ │ + VtkVis     │   │
│   │ + vtkRenWin  │ │ + vtkRenWin  │ │ + vtkRenWin  │   │
│   └──────────────┘ └──────────────┘ └──────────────┘   │
│                                                          │
│   ┌──────────────────────────────────────────────────┐  │
│   │         ecvViewManager (singleton)                │  │
│   │  - activeView / allViews management               │  │
│   │  - signal: activeViewChanged                      │  │
│   │  - refreshAll() / redrawAll()                     │  │
│   └──────────────────────────────────────────────────┘  │
│                                                          │
│   ┌──────────────────────────────────────────────────┐  │
│   │    ecvRepresentationManager (singleton)           │  │
│   │  - per-(entity, view) display state               │  │
│   │  - VTK actor cleanup callback (from VtkEngine)    │  │
│   └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│               Shared Scene DB Tree                        │
│  ccHObject (m_currentDisplay -> ecvGenericGLDisplay*)     │
│  ├── PointCloud A  (display = view1)                      │
│  ├── Mesh B        (display = nullptr → all windows)      │
│  └── PointCloud C  (display = view2)                      │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Architecture Constraints

### Constraint 1: Module Scope Isolation

```
┌─ CV_db (libs/CV_db) ──────────────────────────────────┐
│  Pure data/interface layer, ZERO VTK dependency         │
│  ecvGenericGLDisplay.h     (abstract interface)         │
│  ecvViewRepresentation.h   (per-view props, pure data)  │
│  ecvRepresentationManager.h (registry, no VTK refs)     │
│  ecvViewManager.h           (active view mgmt)          │
│  ecvDrawableObject.h        (m_currentDisplay)          │
│  ecvDrawContext.h           (display pointer)            │
│  ❌ NEVER #include <vtk*.h>                             │
│  ❌ NEVER reference VtkVis, QVTKWidgetCustom, etc.      │
└────────────────────────────────────────────────────────┘
          ▲ depends only on Qt Core/Widgets
          │
┌─ VtkEngine (libs/VtkEngine) ──────────────────────────┐
│  VTK rendering backend implementation                   │
│  ecvGLView.h    (implements ecvGenericGLDisplay)        │
│  VtkDisplayTools.h  (resolveVisualizer routing)         │
│  VtkVis.h            (VTK pipeline)                     │
│  QVTKWidgetCustom.h  (Qt/VTK bridge)                   │
│  ✅ CAN #include <vtk*.h>                               │
└────────────────────────────────────────────────────────┘
          ▲ depends on CV_db + VTK + Qt
          │
┌─ App (app/) ──────────────────────────────────────────┐
│  MainWindow, MDI management, UI                         │
│  ✅ CAN reference CV_db and VtkEngine                   │
└────────────────────────────────────────────────────────┘
```

**Key isolation technique**: `ecvRepresentationManager` uses a **callback** (`std::function`) for VTK actor cleanup, registered by VtkEngine at init time — no VTK types leak into CV_db headers.

### Constraint 2: Cross-Platform Compatibility (Windows Symbol Export)

| Module | Export Macro | Header |
|--------|-------------|--------|
| CV_db new classes | `CV_DB_LIB_API` | `#include "CV_db.h"` |
| VtkEngine new classes | `QVTK_ENGINE_LIB_API` | `#include "qVTK.h"` |
| App layer | no export needed | — |

Windows-specific notes:
- `Q_OBJECT` classes require MOC processing (CMake `AUTOMOC` covers this).
- `std::function` members defined in `.cpp`, not exposed in headers.
- `QHash` `qHash` specializations implemented in `.cpp`.
- `std::optional` as struct members is fine; avoid as function parameters crossing DLL boundaries.

### Constraint 3: Performance & Memory Stability

| Strategy | Origin | Implementation |
|----------|--------|----------------|
| **Shared geometry data** | MeshLab `MLSceneGLSharedDataContext` | Same `vtkPolyData` shared by multiple view actors |
| **Lazy representation creation** | ParaView proxy lazy init | `getRepresentation()` returns nullptr → legacy path |
| **O(1) lookup** | ParaView `pqActiveObjects` | `QHash<QPair<ptr,ptr>>` indexing |
| **Weak references** | MeshLab `QPointer` | `ecvViewManager` monitors view lifecycle |
| **Batched refresh debounce** | ParaView deferred render timer | `ecvViewManager::redrawAll()` merges requests |

---

## 5. Reference Frameworks Comparison

| Dimension | CloudCompare | ParaView | MeshLab | **ACloudViewer (proposed)** |
|-----------|-------------|----------|---------|---------------------------|
| Entity-window binding | Direct pointer `m_currentDisplay` | Representation Proxy per (data, view) | Per-view visibility map | **Direct pointer + Representation layer** |
| Per-view display differences | Not supported (1 entity → 1 window) | Full support (independent repr per view) | Per-view visibility only | **Representation with per-view properties** |
| Active view tracking | Implicit (MDI activation + `FromWidget`) | `pqActiveObjects` singleton + `vtkSMProxySelectionModel` | `currentgla` | **`ecvViewManager` (signal-driven)** |
| VTK pipeline | N/A (OpenGL direct) | Per view: independent `vtkRenderWindow` + `vtkRenderer` | N/A (QGLWidget + vcg) | **Per view: independent VtkVis** |
| Layout | QMdiArea | `pqMultiViewWidget` (splits/tabs, serializable) | QSplitter | **QMdiArea (Phase 1) + future splitter** |
| Draw context | `context.display` pointer | Representation proxy filtering | Per-GLArea paintGL | **`context.display` pointer** |
| Shared GPU resources | N/A | Shared data objects + per-view mappers | `MLSceneGLSharedDataContext` | **Shared `vtkPolyData`, per-view actors** |

---

## 6. Phase Breakdown

### Phase 1: ecvGenericGLDisplay Interface

**New files**:
- `libs/CV_db/include/ecvGenericGLDisplay.h`
- `libs/CV_db/src/ecvGenericGLDisplay.cpp`

**Module**: CV_db | **Risk**: Low

```cpp
// ecvGenericGLDisplay.h
#pragma once
#include "CV_db.h"
#include <QString>

class ccHObject;
class ccDrawableObject;
class QWidget;
class ecvViewportParameters;

/// Per-window display interface for multi-window rendering.
///
/// Each 3D view window implements this interface, holding its own
/// viewport parameters, camera state, and scene/window DB roots.
/// The global ecvDisplayTools singleton delegates to the "active"
/// ecvGenericGLDisplay instance so that existing static call sites
/// keep working while new code can address a specific window.
///
/// Design references:
///   CloudCompare: ccGenericGLDisplay
///   ParaView:     pqView / pqRenderView
///   MeshLab:      GLArea
class CV_DB_LIB_API ecvGenericGLDisplay {
public:
    virtual ~ecvGenericGLDisplay() = default;

    // ── Identification ──
    virtual int getUniqueID() const = 0;
    virtual QString getTitle() const = 0;

    // ── Refresh / redraw ──
    virtual void redraw(bool only2D = false, bool forceRedraw = true) = 0;
    virtual void refresh(bool only2D = false) = 0;
    virtual void toBeRefreshed() = 0;

    // ── Viewport / camera ──
    virtual const ecvViewportParameters& getViewportParameters() const = 0;
    virtual void setViewportParameters(const ecvViewportParameters& params) = 0;
    virtual void setPerspectiveState(bool state, bool objectCenteredView) = 0;
    virtual bool perspectiveView() const = 0;
    virtual bool objectCenteredView() const = 0;

    // ── Scene / own DB ──
    virtual void setSceneDB(ccHObject* root) = 0;
    virtual ccHObject* getSceneDB() = 0;
    virtual ccHObject* getOwnDB() = 0;
    virtual void addToOwnDB(ccHObject* obj, bool noDependency = true) = 0;
    virtual void removeFromOwnDB(ccHObject* obj) = 0;

    // ── Qt widget bridge ──
    virtual QWidget* asWidget() = 0;
    virtual const QWidget* asWidget() const = 0;

    // ── Display parameters (per-window overrides) ──
    virtual bool hasOverriddenDisplayParameters() const = 0;

    // ── Lifecycle notification (ref: CloudCompare aboutToBeRemoved) ──
    virtual void aboutToBeRemoved(ccDrawableObject* /*obj*/) {}

    // ── Static registry: QWidget* -> ecvGenericGLDisplay* ──
    static ecvGenericGLDisplay* FromWidget(QWidget* widget);
    static void RegisterGLDisplay(QWidget* widget, ecvGenericGLDisplay* display);
    static void UnregisterGLDisplay(QWidget* widget);
};
```

**Implementation** (`ecvGenericGLDisplay.cpp`):

```cpp
#include "ecvGenericGLDisplay.h"
#include <QMap>
#include <QMutex>
#include <QMutexLocker>

namespace {
QMutex s_registryMutex;
QMap<QWidget*, ecvGenericGLDisplay*> s_displayRegistry;
}

ecvGenericGLDisplay* ecvGenericGLDisplay::FromWidget(QWidget* widget) {
    QMutexLocker lock(&s_registryMutex);
    auto it = s_displayRegistry.find(widget);
    return (it != s_displayRegistry.end()) ? it.value() : nullptr;
}

void ecvGenericGLDisplay::RegisterGLDisplay(QWidget* widget,
                                            ecvGenericGLDisplay* display) {
    QMutexLocker lock(&s_registryMutex);
    s_displayRegistry.insert(widget, display);
}

void ecvGenericGLDisplay::UnregisterGLDisplay(QWidget* widget) {
    QMutexLocker lock(&s_registryMutex);
    s_displayRegistry.remove(widget);
}
```

**CMake**: Add to `libs/CV_db/include/CMakeLists.txt` and `libs/CV_db/src/CMakeLists.txt`.

---

### Phase 2: ecvViewManager

**New files**:
- `libs/CV_db/include/ecvViewManager.h`
- `libs/CV_db/src/ecvViewManager.cpp`

**Module**: CV_db | **Risk**: Low

```cpp
// ecvViewManager.h
#pragma once
#include "CV_db.h"
#include <QObject>
#include <QList>

class ecvGenericGLDisplay;
class ccHObject;

/// Global view manager. Replaces stacking active-window static methods
/// on ecvDisplayTools. Inspired by ParaView's pqActiveObjects but lighter.
class CV_DB_LIB_API ecvViewManager : public QObject {
    Q_OBJECT
public:
    static ecvViewManager& instance();

    // ── Active view ──
    ecvGenericGLDisplay* getActiveView() const;
    void setActiveView(ecvGenericGLDisplay* view);

    // ── View registration ──
    void registerView(ecvGenericGLDisplay* view);
    void unregisterView(ecvGenericGLDisplay* view);

    // ── Query ──
    const QList<ecvGenericGLDisplay*>& getAllViews() const;
    int viewCount() const;
    ecvGenericGLDisplay* findView(int uniqueID) const;

    // ── Batch operations ──
    void refreshAll(bool only2D = false);
    void redrawAll(bool only2D = false, bool forceRedraw = true);

    // ── Entity-view association helpers ──
    void associateToActiveView(ccHObject* obj);
    void detachEntitiesFromView(ecvGenericGLDisplay* view);

signals:
    void activeViewChanged(ecvGenericGLDisplay* newActive,
                           ecvGenericGLDisplay* oldActive);
    void viewRegistered(ecvGenericGLDisplay* view);
    void viewUnregistered(ecvGenericGLDisplay* view);
    void viewCountChanged(int count);

private:
    ecvViewManager();
    ecvGenericGLDisplay* m_activeView = nullptr;
    QList<ecvGenericGLDisplay*> m_views;
};
```

**Design rationale** (vs adding static methods to ecvDisplayTools):
- **Separation of concerns**: View management ≠ rendering tools.
- **Signal-driven**: `activeViewChanged` lets UI components auto-respond (properties panel, toolbar, status bar).
- **Minimal invasion**: ecvDisplayTools (2760 lines, 200+ static methods) remains untouched.
- **Testable**: Independent class, easier to unit test.

---

### Phase 3: CC_DRAW_CONTEXT + ccDrawableObject

**Modified files**:
- `libs/CV_db/include/ecvDrawContext.h`
- `libs/CV_db/include/ecvDrawableObject.h`
- `libs/CV_db/src/ecvDrawableObject.cpp`

**Module**: CV_db | **Risk**: Low

#### ecvDrawContext.h changes

```cpp
class ecvGenericGLDisplay;  // forward declaration

struct ccGLDrawContext {
    // ... existing members ...

    /// The display (window) that owns this draw context.
    /// nullptr means "draw in all windows" (backward-compatible default).
    ecvGenericGLDisplay* display = nullptr;
};
```

#### ecvDrawableObject.h changes

```cpp
class ecvGenericGLDisplay;  // forward declaration

class CV_DB_LIB_API ccDrawableObject {
public:
    // ── Display association (multi-window support) ──
    virtual ecvGenericGLDisplay* getDisplay() const { return m_currentDisplay; }
    virtual void setDisplay(ecvGenericGLDisplay* display);
    virtual void removeFromDisplay(const ecvGenericGLDisplay* display);

    // ... existing draw, visibility, transform methods ...

protected:
    // ... existing members ...

    /// Currently associated GL display (window).
    /// nullptr = show in any window that draws this entity (legacy mode).
    ecvGenericGLDisplay* m_currentDisplay = nullptr;
};
```

#### ecvDrawableObject.cpp changes

```cpp
void ccDrawableObject::setDisplay(ecvGenericGLDisplay* display) {
    if (m_currentDisplay == display) return;
    m_currentDisplay = display;
    setRedraw(true);
}

void ccDrawableObject::removeFromDisplay(const ecvGenericGLDisplay* display) {
    if (m_currentDisplay == display) {
        m_currentDisplay = nullptr;
        setRedraw(true);
    }
}
```

**Backward compatibility**: `m_currentDisplay` defaults to `nullptr`, meaning "show in all windows". Existing code never calls `setDisplay()`, so behavior is unchanged.

---

### Phase 4: ecvViewRepresentation + ecvRepresentationManager

**New files**:
- `libs/CV_db/include/ecvViewRepresentation.h`
- `libs/CV_db/src/ecvViewRepresentation.cpp`
- `libs/CV_db/include/ecvRepresentationManager.h`
- `libs/CV_db/src/ecvRepresentationManager.cpp`

**Module**: CV_db | **Risk**: Low

#### ecvViewRepresentation

```cpp
// ecvViewRepresentation.h
#pragma once
#include "CV_db.h"
#include <optional>

class ccHObject;
class ecvGenericGLDisplay;

/// Per-(entity, view) display state.
/// Analogous to ParaView's vtkSMRepresentationProxy: tracks how one data
/// object appears in one specific view.
class CV_DB_LIB_API ecvViewRepresentation {
public:
    enum class RenderMode : int {
        Inherit = -1,  // use entity's default
        Points = 0,
        Wireframe = 1,
        Surface = 2,
        SurfaceWithEdges = 3
    };

    ecvViewRepresentation(ccHObject* entity, ecvGenericGLDisplay* view);
    ~ecvViewRepresentation() = default;

    ccHObject* getEntity() const { return m_entity; }
    ecvGenericGLDisplay* getView() const { return m_view; }

    // ── Visibility (per-view override) ──
    bool isVisible() const;
    void setVisible(bool v);
    bool hasVisibilityOverride() const { return m_visibilityOverride.has_value(); }
    void clearVisibilityOverride();

    // ── Per-view display properties ──
    // Tier 1 (Phase 1 must-have):
    struct Properties {
        std::optional<float> opacity;
        std::optional<float> pointSize;
        std::optional<float> lineWidth;
        std::optional<RenderMode> renderMode;
        std::optional<bool> edgeVisibility;

        // Tier 2 (Phase 1 optional):
        std::optional<int> scalarFieldIndex;
        std::optional<bool> showScalarField;
        std::optional<bool> showColors;
        std::optional<bool> showNormals;
        std::optional<float> normalScale;

        // Tier 3 (future):
        // std::optional<bool> showTextures;
        // std::optional<bool> showMaterials;
        // std::optional<QString> colorMapName;
    };

    const Properties& properties() const { return m_properties; }
    Properties& properties() { return m_properties; }
    void setProperties(const Properties& props);

    float effectiveOpacity() const;
    float effectivePointSize() const;

    // ── Dirty state (needs VTK actor update) ──
    bool isDirty() const { return m_dirty; }
    void setDirty(bool d = true) { m_dirty = d; }

private:
    ccHObject* m_entity;
    ecvGenericGLDisplay* m_view;
    std::optional<bool> m_visibilityOverride;
    Properties m_properties;
    bool m_dirty = true;
};
```

#### ecvRepresentationManager

```cpp
// ecvRepresentationManager.h
#pragma once
#include "CV_db.h"
#include <QObject>
#include <QHash>
#include <QPair>
#include <functional>
#include <memory>

class ccHObject;
class ecvGenericGLDisplay;
class ecvViewRepresentation;

/// Central registry for all (entity, view) representations.
/// Analogous to ParaView's pqServerManagerModel for representation tracking.
class CV_DB_LIB_API ecvRepresentationManager : public QObject {
    Q_OBJECT
public:
    static ecvRepresentationManager& instance();

    // ── Lookup ──
    ecvViewRepresentation* getRepresentation(ccHObject* entity,
                                              ecvGenericGLDisplay* view) const;
    ecvViewRepresentation* ensureRepresentation(ccHObject* entity,
                                                 ecvGenericGLDisplay* view);

    // ── Batch queries ──
    QList<ecvViewRepresentation*> getRepresentationsForEntity(ccHObject* entity) const;
    QList<ecvViewRepresentation*> getRepresentationsForView(ecvGenericGLDisplay* view) const;

    // ── Cleanup ──
    void removeRepresentationsForEntity(ccHObject* entity);
    void removeRepresentationsForView(ecvGenericGLDisplay* view);
    void removeRepresentation(ccHObject* entity, ecvGenericGLDisplay* view);

    int count() const;

    // ── VTK actor cleanup callback ──
    // Registered by VtkEngine layer at init time.
    // This avoids VTK types leaking into CV_db headers.
    using CleanupCallback = std::function<void(ccHObject* entity,
                                                ecvGenericGLDisplay* view)>;
    void setActorCleanupCallback(CleanupCallback cb);

signals:
    void representationAdded(ecvViewRepresentation* rep);
    void representationRemoved(ccHObject* entity, ecvGenericGLDisplay* view);
    void representationChanged(ecvViewRepresentation* rep);

private:
    ecvRepresentationManager() = default;

    using Key = QPair<ccHObject*, ecvGenericGLDisplay*>;
    QHash<Key, std::unique_ptr<ecvViewRepresentation>> m_representations;
    CleanupCallback m_actorCleanup;
};
```

**Scope isolation**: The `CleanupCallback` allows the VtkEngine layer to register VTK-specific cleanup logic without any VTK types appearing in CV_db headers.

---

### Phase 5: ccHObject::draw() Per-Window Filtering

**Modified files**:
- `libs/CV_db/include/ecvHObject.h`
- `libs/CV_db/src/ecvHObject.cpp`

**Module**: CV_db | **Risk**: Medium

#### New methods on ecvHObject.h

```cpp
/// Recursively associate this entity and all children with a display.
void setDisplay_recursive(ecvGenericGLDisplay* display);

/// Recursively clear display association for entities bound to the given display.
void removeFromDisplay_recursive(const ecvGenericGLDisplay* display);

/// Returns true if this entity should be drawn in the given display.
/// Three-way compatible logic:
///   display == nullptr:        legacy mode, no filtering
///   m_currentDisplay == nullptr: entity unbound, draw in all windows
///   m_currentDisplay == display: normal match
bool isDisplayedIn(const ecvGenericGLDisplay* display) const;
```

#### Modified ccHObject::draw()

```cpp
void ccHObject::draw(CC_DRAW_CONTEXT& context) {
    // ... existing early checks ...

    // Entity must be visible or selected, AND displayed in this context
    bool drawInThisContext = ((m_visible || m_selected) &&
                               isDisplayedIn(context.display));

    // Representation layer: per-view visibility override
    if (drawInThisContext && context.display) {
        auto* rep = ecvRepresentationManager::instance()
                        .getRepresentation(this, context.display);
        if (rep && rep->hasVisibilityOverride()) {
            drawInThisContext = rep->isVisible();
        }
    }

    if (!drawInThisContext) {
        // Still recurse children (they may be bound to this display)
        for (auto child : m_children) {
            child->draw(context);
        }
        return;
    }

    // ... rest of existing draw logic (drawMeOnly, etc.) ...
}

bool ccHObject::isDisplayedIn(const ecvGenericGLDisplay* display) const {
    return (display == nullptr ||
            m_currentDisplay == nullptr ||
            m_currentDisplay == display);
}
```

**Backward compatibility**: When both `display` and `m_currentDisplay` are `nullptr` (the default), `isDisplayedIn()` returns `true` — behavior is unchanged.

---

### Phase 6: ecvDisplayTools Adaption

**Modified files**:
- `libs/CV_db/include/ecvDisplayTools.h`
- `libs/CV_db/src/ecvDisplayTools.cpp`

**Module**: CV_db | **Risk**: Medium

Make `ecvDisplayTools` implement `ecvGenericGLDisplay` (as the "primary window" display):

```cpp
class CV_DB_LIB_API ecvDisplayTools : public QObject,
                                       public ecvGenericDisplayTools,
                                       public ecvGenericGLDisplay {
    Q_OBJECT
public:
    // ── ecvGenericGLDisplay implementation ──
    int getUniqueID() const override { return m_uniqueID; }
    QString getTitle() const override { return "Primary View"; }
    void redraw(bool only2D, bool forceRedraw) override;
    void refresh(bool only2D) override;
    void toBeRefreshed() override;
    const ecvViewportParameters& getViewportParameters() const override;
    void setViewportParameters(const ecvViewportParameters& params) override;
    void setPerspectiveState(bool state, bool objectCenteredView) override;
    bool perspectiveView() const override;
    bool objectCenteredView() const override;
    void setSceneDB(ccHObject* root) override;
    ccHObject* getSceneDB() override;
    ccHObject* getOwnDB() override;
    void addToOwnDB(ccHObject* obj, bool noDependency) override;
    void removeFromOwnDB(ccHObject* obj) override;
    QWidget* asWidget() override;
    const QWidget* asWidget() const override;
    bool hasOverriddenDisplayParameters() const override;

    // ... existing 200+ static methods remain unchanged ...
};
```

#### Changes in Init() and GetContext()

```cpp
void ecvDisplayTools::Init(ecvDisplayTools* displayTools,
                           QMainWindow* win, bool stereoMode) {
    // ... existing init logic ...

    // Register with ViewManager
    ecvViewManager::instance().registerView(s_tools.instance);
    ecvViewManager::instance().setActiveView(s_tools.instance);
}

void ecvDisplayTools::GetContext(CC_DRAW_CONTEXT& CONTEXT) {
    // ... existing context filling ...

    // Set the display pointer to the active view
    CONTEXT.display = ecvViewManager::instance().getActiveView();
}
```

**Key point**: The singleton's existing static methods are NOT modified. The only changes are:
1. New `ecvGenericGLDisplay` override methods (delegating to existing member functions).
2. `Init()` registers with `ecvViewManager`.
3. `GetContext()` sets `CONTEXT.display`.

---

### Phase 7: ecvGLView (Per-Window VTK Pipeline)

**New files**:
- `libs/VtkEngine/Visualization/ecvGLView.h`
- `libs/VtkEngine/Visualization/ecvGLView.cpp`

**Module**: VtkEngine | **Risk**: Medium

```cpp
// ecvGLView.h
#pragma once
#include "qVTK.h"
#include <ecvGenericGLDisplay.h>
#include <ecvViewportParameters.h>
#include <QObject>
#include <memory>

class ccHObject;
class QMainWindow;
class QVTKWidgetCustom;

namespace Visualization {
class VtkVis;
using VtkVisPtr = std::shared_ptr<VtkVis>;
}

/// Per-window 3D view implementing ecvGenericGLDisplay backed by VTK.
///
/// Each ecvGLView owns an independent:
///   - QVTKWidgetCustom (Qt/VTK bridge widget)
///   - VtkVis (vtkRenderer + vtkRenderWindow + vtkInteractor)
///   - Viewport parameters (camera, perspective, zoom)
///   - Window-local DB root (m_winDBRoot)
///   - Reference to the shared global scene DB
///
/// Design references:
///   ParaView pqRenderView:  per-view vtkRenderWindow
///   CloudCompare ccGLWindow: per-window camera/viewport
///   MeshLab GLArea:          independent paintGL, shared document
class QVTK_ENGINE_LIB_API ecvGLView : public QObject,
                                        public ecvGenericGLDisplay {
    Q_OBJECT

public:
    static ecvGLView* Create(QMainWindow* parent, bool stereoMode = false);
    ~ecvGLView() override;

    // ── ecvGenericGLDisplay interface (full implementation) ──
    int getUniqueID() const override { return m_uniqueID; }
    QString getTitle() const override { return m_title; }
    void redraw(bool only2D = false, bool forceRedraw = true) override;
    void refresh(bool only2D = false) override;
    void toBeRefreshed() override;
    const ecvViewportParameters& getViewportParameters() const override;
    void setViewportParameters(const ecvViewportParameters& params) override;
    void setPerspectiveState(bool state, bool objectCenteredView) override;
    bool perspectiveView() const override;
    bool objectCenteredView() const override;
    void setSceneDB(ccHObject* root) override;
    ccHObject* getSceneDB() override;
    ccHObject* getOwnDB() override;
    void addToOwnDB(ccHObject* obj, bool noDependency = true) override;
    void removeFromOwnDB(ccHObject* obj) override;
    QWidget* asWidget() override;
    const QWidget* asWidget() const override;
    bool hasOverriddenDisplayParameters() const override;
    void aboutToBeRemoved(ccDrawableObject* obj) override;

    // ── VTK-specific (not exposed to CV_db layer) ──
    QVTKWidgetCustom* getVtkWidget() const { return m_vtkWidget; }
    Visualization::VtkVis* getVisualizer3D() const;

signals:
    void aboutToClose(ecvGLView* self);
    void viewActivated(ecvGLView* self);

protected:
    explicit ecvGLView(QMainWindow* parent);

private:
    void initVtkPipeline(QMainWindow* parent, bool stereoMode);
    void fillContext(CC_DRAW_CONTEXT& ctx) const;

    int m_uniqueID;
    QString m_title;
    QVTKWidgetCustom* m_vtkWidget = nullptr;
    Visualization::VtkVisPtr m_visualizer3D;
    ecvViewportParameters m_viewportParams;
    ccHObject* m_globalDBRoot = nullptr;  // shared (not owned)
    ccHObject* m_winDBRoot = nullptr;     // per-window (owned)
    bool m_shouldBeRefreshed = false;
    bool m_overriddenDisplayParamsEnabled = false;

    static int s_nextWindowID;
};
```

#### ecvGLView::redraw() — self-contained draw cycle

```cpp
void ecvGLView::redraw(bool only2D, bool forceRedraw) {
    CC_DRAW_CONTEXT context;
    fillContext(context);
    context.display = this;  // key: identifies this window

    if (!only2D) {
        context.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
        if (m_globalDBRoot) m_globalDBRoot->draw(context);
        if (m_winDBRoot)    m_winDBRoot->draw(context);
    }

    context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
    if (m_globalDBRoot) m_globalDBRoot->draw(context);
    if (m_winDBRoot)    m_winDBRoot->draw(context);

    // Trigger VTK rendering
    if (m_visualizer3D) {
        m_visualizer3D->getRenderWindow()->Render();
    }

    m_shouldBeRefreshed = false;
}
```

#### VTK pipeline init — shared vtkPolyData pattern

```cpp
void ecvGLView::initVtkPipeline(QMainWindow* parent, bool stereoMode) {
    m_vtkWidget = new QVTKWidgetCustom(parent, nullptr, stereoMode);

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    auto renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    auto interactorStyle =
        vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>::New();

    m_visualizer3D = std::make_shared<Visualization::VtkVis>(
        renderer, renderWindow, interactorStyle,
        m_title.toStdString(), false);

    m_vtkWidget->SetRenderWindow(renderWindow);
    m_visualizer3D->setupInteractor(m_vtkWidget->GetInteractor(),
                                     m_vtkWidget->GetRenderWindow());
    m_vtkWidget->initVtk(m_visualizer3D->getRenderWindowInteractor(), false);
    m_visualizer3D->initialize();

    ecvGenericGLDisplay::RegisterGLDisplay(m_vtkWidget, this);
}
```

#### QVTKWidgetCustom constructor adaptation

```cpp
// Current:
QVTKWidgetCustom(QMainWindow* win, ecvDisplayTools* tools, bool stereoMode);

// Add overload (backward compatible):
QVTKWidgetCustom(QMainWindow* win,
                  ecvGenericGLDisplay* display,  // nullable
                  bool stereoMode);
```

---

### Phase 8: VtkDisplayTools Routing + Per-View Properties

**Modified files**:
- `libs/VtkEngine/Visualization/VtkDisplayTools.h`
- `libs/VtkEngine/Visualization/VtkDisplayTools.cpp`

**Module**: VtkEngine | **Risk**: Medium

#### resolveVisualizer routing

```cpp
VtkVis* VtkDisplayTools::resolveVisualizer(ecvGenericGLDisplay* display) const {
    // Case 1: nullptr (legacy) → primary VtkVis
    if (!display) {
        return m_visualizer3D.get();
    }

    // Case 2: this singleton itself → primary VtkVis
    if (display == static_cast<const ecvGenericGLDisplay*>(this)) {
        return m_visualizer3D.get();
    }

    // Case 3: ecvGLView instance → its own VtkVis
    if (auto* glView = dynamic_cast<ecvGLView*>(display)) {
        VtkVis* vis = glView->getVisualizer3D();
        if (vis) return vis;
    }

    // Fallback
    return m_visualizer3D.get();
}
```

#### Modified draw dispatch

```cpp
void VtkDisplayTools::draw(const CC_DRAW_CONTEXT& context,
                           const ccHObject* obj) {
    VtkVis* vis = resolveVisualizer(context.display);

    ecvViewRepresentation* rep = nullptr;
    if (context.display) {
        rep = ecvRepresentationManager::instance().getRepresentation(
            const_cast<ccHObject*>(obj), context.display);
    }

    if (obj->isA(CV_TYPES::POINT_CLOUD)) {
        auto* cloud = ccHObjectCaster::ToPointCloud(const_cast<ccHObject*>(obj));
        if (cloud) drawPointCloud(context, cloud, vis, rep);
    } else if (obj->isKindOf(CV_TYPES::MESH)) {
        auto* mesh = ccHObjectCaster::ToGenericMesh(const_cast<ccHObject*>(obj));
        if (mesh) drawMesh(const_cast<CC_DRAW_CONTEXT&>(context), mesh, vis, rep);
    }
    // ... other entity types ...
}
```

#### Shared vtkPolyData optimization (ref: MeshLab)

When creating actors for a secondary view, reuse the `vtkPolyData` from the primary view's actor:

```cpp
void VtkDisplayTools::drawPointCloud(const CC_DRAW_CONTEXT& context,
                                      ccPointCloud* cloud,
                                      VtkVis* vis,
                                      ecvViewRepresentation* rep) {
    std::string viewID = CVTools::FromQString(context.viewID);

    if (!vis->contains(viewID)) {
        vtkPolyData* sharedData = nullptr;
        if (vis != m_visualizer3D.get()) {
            auto it = m_visualizer3D->getCloudActorMap()->find(viewID);
            if (it != m_visualizer3D->getCloudActorMap()->end()) {
                sharedData = vtkPolyData::SafeDownCast(
                    it->second.actor->GetMapper()->GetInput());
            }
        }

        if (sharedData) {
            vis->addPointCloudFromPolyData(sharedData, viewID);  // new method
        } else {
            // Normal Cc2Vtk conversion
            // ... existing addPointCloud logic ...
        }
    }

    // Apply per-view properties
    if (rep) {
        applyRepresentationProperties(vis, viewID, rep);
    }
}
```

#### Register VTK cleanup callback

```cpp
void VtkDisplayTools::registerVisualizer(QMainWindow* win, bool stereoMode) {
    // ... existing code ...

    // Register VTK actor cleanup with RepresentationManager
    ecvRepresentationManager::instance().setActorCleanupCallback(
        [this](ccHObject* entity, ecvGenericGLDisplay* view) {
            VtkVis* vis = resolveVisualizer(view);
            if (!vis) return;
            std::string viewID = std::to_string(entity->getUniqueID());
            vis->removeWidgets(viewID, 0);
            vis->removePointClouds(viewID, 0);
            vis->removeShapes(viewID, 0);
            vis->removeMesh(viewID, 0);
        });
}
```

---

### Phase 9: MainWindow MDI Management

**Modified files**:
- `app/MainWindow.h`
- `app/MainWindow.cpp`

**Module**: App | **Risk**: Medium

#### New methods

```cpp
// MainWindow.h
class ecvGenericGLDisplay;
class ecvGLView;

class MainWindow : public QMainWindow, ... {
public:
    // ── Multi-window support ──
    ecvGenericGLDisplay* getActiveGLView();
    ecvGLView* new3DView();
    int getGLViewCount() const;
    void refreshAll(bool only2D = false);
};
```

#### Implementation

```cpp
ecvGenericGLDisplay* MainWindow::getActiveGLView() {
    QWidget* widget = getActiveWindow();
    if (!widget) return nullptr;
    ecvGenericGLDisplay* display = ecvGenericGLDisplay::FromWidget(widget);
    if (display) return display;
    return ecvDisplayTools::TheInstance();  // fallback
}

ecvGLView* MainWindow::new3DView() {
    if (!m_mdiArea) return nullptr;

    auto* view = ecvGLView::Create(this);
    if (!view) return nullptr;

    QWidget* widget = view->asWidget();
    widget->setMinimumSize(400, 300);

    // Share the global scene DB
    view->setSceneDB(m_ccRoot ? m_ccRoot->getRootEntity() : nullptr);

    QMdiSubWindow* subWin = m_mdiArea->addSubWindow(widget);
    subWin->setAttribute(Qt::WA_DeleteOnClose);
    subWin->setWindowTitle(view->getTitle());

    connect(view, &ecvGLView::aboutToClose, this,
            [](ecvGLView* v) {
                ecvRepresentationManager::instance().removeRepresentationsForView(v);
                ecvViewManager::instance().unregisterView(v);
            });

    widget->showMaximized();
    return view;
}

void MainWindow::on3DViewActivated(QMdiSubWindow* mdiWin) {
    if (!mdiWin) return;
    QWidget* screen = mdiWin->widget();
    if (screen) {
        auto* display = ecvGenericGLDisplay::FromWidget(screen);
        if (display) {
            ecvViewManager::instance().setActiveView(display);
        }
        // ... existing fullscreen sync ...
    }
}

void MainWindow::addToDB(ccHObject* obj, ...) {
    // ... existing code ...

    // Multi-window: associate new entity with active window
    if (!obj->getDisplay()) {
        ecvGenericGLDisplay* activeView = getActiveGLView();
        if (activeView) {
            obj->setDisplay_recursive(activeView);
        }
    }

    // ... existing code ...
}

void MainWindow::refreshAll(bool only2D) {
    ecvViewManager::instance().refreshAll(only2D);
}
```

---

### Phase 10: Compilation Verification + Call-Site Migration

**Module**: All | **Risk**: Low

1. Full project build verification (CV_DB_LIB → QVTK_ENGINE_LIB → ACloudViewer + plugins).
2. Migrate existing `refreshAll()` calls to `ecvViewManager::instance().refreshAll()`.
3. Verify plugins compile without modification (they should — all changes are additive).
4. Run basic functional test: load point cloud, create second view, verify independent cameras.

---

## 7. Draw Chain Walkthrough

### Current (Single Window)

```
ecvDisplayTools::Draw3D()
  → GetContext(CONTEXT)                    // fills dimensions, GUI params
  → m_globalDBRoot->draw(CONTEXT)
    → ccHObject::draw(CONTEXT)
      → visible check
      → context.viewID = getViewId()
      → drawMeOnly(CONTEXT)
        → ecvDisplayTools::Draw(ctx, this)  // static
          → TheInstance()->draw(ctx, obj)    // virtual → VtkDisplayTools
            → m_visualizer3D->addPointCloud(viewID, ...)
```

### After Refactoring (Multi-Window)

```
ecvGLView_2::redraw()
  → fillContext(ctx)
  → ctx.display = this                     // ★ identifies current window
  → m_globalDBRoot->draw(ctx)
    → ccHObject::draw(ctx)
      → isDisplayedIn(ctx.display)          // ★ per-window filter
      → [optional] rep visibility check     // ★ representation override
      → context.viewID = getViewId()
      → drawMeOnly(ctx)
        → ecvDisplayTools::Draw(ctx, this)  // static (unchanged)
          → TheInstance()->draw(ctx, obj)    // virtual → VtkDisplayTools
            → resolveVisualizer(ctx.display) // ★ routes to ecvGLView_2's VtkVis
            → [optional] get Representation  // ★ per-view properties
            → vis2->addPointCloud(viewID, ...)  // uses shared vtkPolyData
```

### Key Changes in the Chain

| Step | Module | Change | Impact |
|------|--------|--------|--------|
| `fillContext` / `GetContext` | CV_db | Sets `ctx.display` | Low |
| `ccHObject::draw()` | CV_db | Adds `isDisplayedIn()` + rep check | Medium |
| `ccPointCloud::drawMeOnly()` | CV_db | **NO CHANGE** | None |
| `ecvDisplayTools::Draw()` static | CV_db | **NO CHANGE** | None |
| `VtkDisplayTools::draw()` | VtkEngine | Adds `resolveVisualizer()` + rep | Medium |
| `drawPointCloud()` etc. | VtkEngine | Accepts `VtkVis*` + `rep` params | Medium |

---

## 8. Backward Compatibility

### CC_DRAW_CONTEXT::display == nullptr

When `display` is `nullptr` (all existing code paths):

```cpp
bool ccHObject::isDisplayedIn(const ecvGenericGLDisplay* display) const {
    return (display == nullptr ||          // ← legacy code doesn't set display
            m_currentDisplay == nullptr ||  // ← entity has no binding
            m_currentDisplay == display);
}
// Result: true — ALL entities drawn — behavior unchanged
```

### Plugin API Compatibility

| Plugin Pattern | Compatible? | Notes |
|---------------|-------------|-------|
| `ecvDisplayTools::RedrawDisplay()` | ✅ Yes | Internally redraws active view |
| `ecvDisplayTools::GetViewportParameters()` | ✅ Yes | Returns active view's params |
| `ecvDisplayTools::GetContext()` | ✅ Yes | Adds `ctx.display = activeView` |
| `ecvRedrawScope` | ✅ Yes | Still calls `RedrawDisplay()` |
| `refreshAll()` | ✅ Enhanced | Now refreshes ALL windows |
| `ecvDisplayTools::SetSceneDB()` | ✅ Yes | Still sets primary window's scene DB |

### Migration Path

```
Phase 1-4 (zero risk):
  New files only. No existing signatures changed.
  ctx.display defaults to nullptr → no filtering.
  m_currentDisplay defaults to nullptr → no filtering.

Phase 5-6 (low risk):
  ccHObject::draw() adds isDisplayedIn check (nullptr → true).
  ecvDisplayTools inherits ecvGenericGLDisplay (additive).

Phase 7-8 (medium risk):
  New ecvGLView class. VtkDisplayTools routing.
  QVTKWidgetCustom constructor overload (original preserved).

Phase 9 (medium risk):
  MainWindow new methods. on3DViewActivated update.

Phase 10 (low risk):
  Full build verification. Optional call-site migration.
```

---

## 9. Cross-Platform Checklist

- [ ] All CV_db new classes use `CV_DB_LIB_API` export macro
- [ ] All VtkEngine new classes use `QVTK_ENGINE_LIB_API` export macro
- [ ] `Q_OBJECT` classes covered by CMake `AUTOMOC`
- [ ] `std::optional` only as struct members, not as function parameters crossing DLL boundaries
- [ ] `std::function` callback defined in `.cpp`, not exposed in header template instantiation
- [ ] `QHash` `qHash` specialization for `QPair<ptr,ptr>` in `.cpp`
- [ ] No platform-specific code (#ifdef) unless absolutely necessary
- [ ] Tested: Linux (primary), Windows (MSVC), macOS (Clang)

---

## 10. Performance & Memory Checklist

- [ ] Shared `vtkPolyData` between views (avoid geometry duplication)
- [ ] Lazy Representation creation (only when multi-window is used)
- [ ] `QHash<QPair<ptr,ptr>>` for O(1) representation lookup
- [ ] Debounced batch refresh in `ecvViewManager::redrawAll()`
- [ ] Weak reference (`QPointer`) for view lifecycle monitoring
- [ ] No memory leaks: Representation cleanup on entity deletion AND view closure
- [ ] No dangling pointers: `ecvGenericGLDisplay::UnregisterGLDisplay()` in destructors
- [ ] Actor cleanup callback avoids cross-module VTK header inclusion

---

## 11. VtkVis viewport Parameter Decision

### Current Status

- `int viewport = 0` appears in **~80 method signatures** in VtkVis.
- `createViewPort()` is **never called** anywhere in the codebase.
- The viewport parameter is **dead code** inherited from PCL.

### Decision: Keep for Future Split-View

The `viewport` parameter is **preserved** for potential future use as intra-window split-view (multiple `vtkRenderer` instances within a single `vtkRenderWindow`). This is orthogonal to multi-window support (separate `vtkRenderWindow` per window).

Multi-window rendering uses **separate `VtkVis` instances**, not the viewport mechanism:

```
Multi-window (this plan):    One VtkVis per window, each with one renderer
Future split-view (later):   One VtkVis, multiple renderers via createViewPort()
```

---

## 12. File Inventory

### New Files

| File | Module | Description |
|------|--------|-------------|
| `libs/CV_db/include/ecvGenericGLDisplay.h` | CV_db | Per-window display interface |
| `libs/CV_db/src/ecvGenericGLDisplay.cpp` | CV_db | Static registry implementation |
| `libs/CV_db/include/ecvViewManager.h` | CV_db | Active view management |
| `libs/CV_db/src/ecvViewManager.cpp` | CV_db | Implementation |
| `libs/CV_db/include/ecvViewRepresentation.h` | CV_db | Per-(entity, view) state |
| `libs/CV_db/src/ecvViewRepresentation.cpp` | CV_db | Implementation |
| `libs/CV_db/include/ecvRepresentationManager.h` | CV_db | Representation registry |
| `libs/CV_db/src/ecvRepresentationManager.cpp` | CV_db | Implementation |
| `libs/VtkEngine/Visualization/ecvGLView.h` | VtkEngine | Per-window VTK view |
| `libs/VtkEngine/Visualization/ecvGLView.cpp` | VtkEngine | Implementation |

### Modified Files

| File | Module | Change Scope |
|------|--------|-------------|
| `libs/CV_db/include/ecvDrawContext.h` | CV_db | Add `display` pointer |
| `libs/CV_db/include/ecvDrawableObject.h` | CV_db | Add `m_currentDisplay` + methods |
| `libs/CV_db/src/ecvDrawableObject.cpp` | CV_db | Implement display methods |
| `libs/CV_db/include/ecvHObject.h` | CV_db | Add filtering methods |
| `libs/CV_db/src/ecvHObject.cpp` | CV_db | Modify `draw()`, implement filtering |
| `libs/CV_db/include/ecvDisplayTools.h` | CV_db | Inherit `ecvGenericGLDisplay` |
| `libs/CV_db/src/ecvDisplayTools.cpp` | CV_db | `Init()`, `GetContext()` integration |
| `libs/CV_db/include/CMakeLists.txt` | CV_db | Add new headers |
| `libs/CV_db/src/CMakeLists.txt` | CV_db | Add new sources |
| `libs/VtkEngine/Visualization/VtkDisplayTools.h` | VtkEngine | Add `resolveVisualizer`, modify draw methods |
| `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` | VtkEngine | Routing + per-view properties |
| `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h` | VtkEngine | Constructor overload |
| `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` | VtkEngine | Constructor implementation |
| `app/MainWindow.h` | App | Add multi-window methods |
| `app/MainWindow.cpp` | App | MDI management, view activation |

### Unchanged Files (Verified)

| File | Reason |
|------|--------|
| `libs/CV_db/src/ecvPointCloud.cpp` | `drawMeOnly()` calls `ecvDisplayTools::Draw()` — no change needed |
| `libs/CV_db/src/ecvMesh.cpp` | Same pattern as point cloud |
| All plugin `.cpp` files | Use `ecvDisplayTools` static API — backward compatible |
| `libs/VtkEngine/Visualization/VtkVis.h` | viewport parameter preserved, no changes |

---

## Appendix: Representation Lifecycle

```
┌── Entity Loaded ──────────────────────────────────────────────────┐
│ MainWindow::addToDB(entity)                                        │
│   → entity->setDisplay(activeView)                                 │
│   → RepresentationManager::ensureRepresentation(entity, activeView)│
└────────────────────────────────────────────────────────────────────┘

┌── Entity Shown in Additional View ────────────────────────────────┐
│ User action: "Show in View 2"                                      │
│   → entity->setDisplay(nullptr)         // unbind from single view │
│   → RepresentationManager::ensureRepresentation(entity, view2)     │
└────────────────────────────────────────────────────────────────────┘

┌── View Closed ────────────────────────────────────────────────────┐
│ ecvGLView::aboutToClose →                                          │
│   → RepresentationManager::removeRepresentationsForView(view)      │
│     → actorCleanup callback → VtkVis::removeActors(viewID)         │
│   → ecvViewManager::unregisterView(view)                           │
└────────────────────────────────────────────────────────────────────┘

┌── Entity Deleted ─────────────────────────────────────────────────┐
│ ccHObject destructor:                                              │
│   → RepresentationManager::removeRepresentationsForEntity(entity)  │
│     → actorCleanup callback → each view's VtkVis::removeActors()   │
└────────────────────────────────────────────────────────────────────┘
```
