# ecvDisplayTools Singleton Removal — Per-View Migration Plan

> Date: 2026-04-29
> Version: 1.0
> Target: Align with ParaView multi-window architecture — every window is equal, no primary/secondary distinction
> Reference: [multi-window-views.md](multi-window-views.md), ParaView `/home/ludahai/develop/code/github/ParaView`

---

## Table of Contents

1. [Goal & Principles](#goal--principles)
2. [Current Architecture (Before)](#current-architecture-before)
3. [Target Architecture (After)](#target-architecture-after)
4. [Singleton State Inventory](#singleton-state-inventory)
5. [Static API Call-Site Inventory](#static-api-call-site-inventory)
6. [Signal/Slot Migration Map](#signalslot-migration-map)
7. [Migration Phases](#migration-phases)
   - [Phase 0: Already Completed](#phase-0-already-completed)
   - [Phase 1: Per-View Signal Hub](#phase-1-per-view-signal-hub)
   - [Phase 2: QVTKWidgetCustom Decoupling](#phase-2-qvtkwidgetcustom-decoupling)
   - [Phase 3: VtkDisplayTools Per-View Pipeline](#phase-3-vtkdisplaytools-per-view-pipeline)
   - [Phase 4: Static Method Elimination (High-Traffic)](#phase-4-static-method-elimination-high-traffic)
   - [Phase 5: Draw Pipeline Per-View Routing](#phase-5-draw-pipeline-per-view-routing)
   - [Phase 6: Python Wrapper Alignment](#phase-6-python-wrapper-alignment)
   - [Phase 7: Singleton Removal & Final Cleanup](#phase-7-singleton-removal--final-cleanup)
8. [Detailed Task Breakdown](#detailed-task-breakdown)
9. [Risk Matrix](#risk-matrix)
10. [Compilation & Test Strategy](#compilation--test-strategy)
11. [ParaView Reference Patterns](#paraview-reference-patterns)

---

## Goal & Principles

### Goal

Remove the `ecvDisplayTools` singleton pattern so that **every 3D view window** (`ecvGLView`) is a fully independent, self-contained rendering unit — identical to every other window. No "primary window" concept exists.

### Principles

1. **ParaView parity**: Follow `pqView` / `pqRenderView` / `pqActiveObjects` patterns
2. **No primary/secondary**: All windows are peers; `ecvViewManager` tracks "active" (UI focus) only
3. **Explicit display parameter**: Replace static `ecvDisplayTools::Foo()` with `display->foo()` where `display` is `ecvGenericGLDisplay*`
4. **Incremental**: Each phase compiles and runs independently; rollback is possible
5. **Signal locality**: Per-view events (mouse, camera, picking) emit from the view; app-level events (selection bus) remain global on `ecvViewManager`

---

## Current Architecture (Before)

```
┌─────────────────────────────────────────────────────────┐
│  ecvDisplayTools (singleton, QObject)                    │
│  ┌─────────────────────┐ ┌───────────────────────────┐  │
│  │ m_primaryCtx         │ │ m_activeItems             │  │
│  │ m_globalDBRoot       │ │ m_clickableItems          │  │
│  │ m_winDBRoot          │ │ m_messagesToDisplay       │  │
│  │ m_hotZone            │ │ m_overridenDisplayParams  │  │
│  │ m_rectPickingPoly    │ │ m_scheduleTimer           │  │
│  │ m_font               │ │ m_shouldBeRefreshed       │  │
│  └─────────────────────┘ └───────────────────────────┘  │
│                                                          │
│  ALL Qt signals: entitySelectionChanged, itemPicked,     │
│  cameraParamChanged, mouseMoved, drawing3D, etc.        │
│                                                          │
│  ~200+ static methods: GetCurrentScreen, RedrawDisplay,  │
│  RemoveWidgets, GetViewportParameters, HideShowEntities, │
│  UpdateScreen, DisplayNewMessage, GetContext, etc.       │
├─────────────────────────────────────────────────────────┤
│  VtkDisplayTools (inherits ecvDisplayTools)               │
│  m_visualizer3D, m_visualizer2D, m_vtkWidget            │
│  switchActiveView / restorePrimaryView / ScopedHotZone  │
└─────────────────────────────────────────────────────────┘
          ▲                          ▲
          │ TheInstance()            │ ScopedRenderOverride
          │                          │
   ┌──────┴──────┐           ┌──────┴──────┐
   │ Primary Win │           │ ecvGLView   │ (secondary)
   │ (built-in)  │           │ m_ctx       │
   │             │           │ m_vtkWidget │
   │             │           │ VtkVis      │
   └─────────────┘           └─────────────┘
```

**Problems**:
- Primary window has privileged access (owns singleton state)
- Secondary windows must use `ScopedRenderOverride` to temporarily redirect static calls
- All signals come from one QObject → consumers can't distinguish which view emitted
- `Update2DLabel()` pollutes singleton `m_activeItems` every 50ms
- `GetCurrentScreen()` has ~150+ call sites, all assuming singleton

---

## Target Architecture (After)

```
   ┌──────────────────────────────────┐
   │ ecvViewManager (singleton)       │
   │ - activeView (UI focus tracking) │
   │ - view registry                  │
   │ - global signals: selection bus  │
   │ - representation manager         │
   └─────────┬────────────────────────┘
             │ getActiveView()
     ┌───────┴───────┬───────────────┐
     ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ecvGLView│   │ecvGLView│   │ecvGLView│   ← All identical peers
│ m_ctx   │   │ m_ctx   │   │ m_ctx   │
│ VtkVis  │   │ VtkVis  │   │ VtkVis  │
│ Widget  │   │ Widget  │   │ Widget  │
│ signals │   │ signals │   │ signals │
│ DB roots│   │ DB roots│   │ DB roots│
│ hotZone │   │ hotZone │   │ hotZone │
│ picking │   │ picking │   │ picking │
└─────────┘   └─────────┘   └─────────┘
```

**Key changes**:
- `ecvDisplayTools` becomes a thin **utility class** (stateless static helpers only) or is removed entirely
- `VtkDisplayTools` functionality absorbed into `ecvGLView` or a per-view `VtkViewService`
- Each `ecvGLView` emits its own Qt signals
- No `ScopedRenderOverride` needed (each view renders independently)
- `ecvViewManager` remains singleton but only for UI focus tracking and selection bus

---

## Singleton State Inventory

### Category A: Already Per-View (on ecvGLView)

| Singleton Field | ecvGLView Equivalent | Status |
|----------------|---------------------|--------|
| `m_primaryCtx` (ecvViewContext) | `m_ctx` | ✅ Done |
| `m_globalDBRoot` | `m_globalDBRoot` | ✅ Done |
| `m_winDBRoot` | `m_winDBRoot` | ✅ Done |
| `m_activeItems` | `m_activeItems` | ✅ Done (Phase 0) |
| `m_clickableItems` | `m_clickableItems` | ✅ Done |
| `m_messagesToDisplay` | `m_messagesToDisplay` | ✅ Done |
| `m_hotZone` | `m_hotZone` | ✅ Done |
| `m_rectPickingPoly` | `m_rectPickingPoly` | ✅ Done |
| `m_overridenDisplayParameters` | `m_overriddenDisplayParameters` | ✅ Done |
| `m_font` | `m_font` | ✅ Done |
| `m_shouldBeRefreshed` | per-view refresh flag | ✅ Done |

### Category B: Truly Global (Keep on ecvViewManager or app-level)

| Field | Reason | Migration Target |
|-------|--------|-----------------|
| `USE_2D`, `USE_VTK_PICK` | App-level flags | Static constants or `ecvViewManager` |
| `m_captureMode` | Session/tool state | `ecvViewManager` or tool-level |
| `m_removeFlag/AllFlag/Infos` | Global entity removal batching | `ecvViewManager` |
| `m_scale_lineset` | Shared debug overlay | `ecvViewManager` or removed |
| `m_diagStrings` | Debug diagnostics | Remove or app-level |

### Category C: Needs Migration (Not Yet Per-View)

| Field | Current Location | Target | Phase |
|-------|-----------------|--------|-------|
| `m_currentScreen` / `m_mainScreen` | Singleton | Remove — each view owns its widget | Phase 2 |
| `m_win` (QMainWindow*) | Singleton | `ecvViewManager` or explicit param | Phase 2 |
| `m_timer` / `m_scheduleTimer` | Singleton | Per-view timers on `ecvGLView` | Phase 2 |
| `m_scheduledFullRedrawTime` | Singleton | Per-view on `ecvGLView` | Phase 2 |
| `m_autoRefresh` | Singleton | Per-view on `ecvGLView` | Phase 2 |
| `m_alwaysUseFBO` / `m_updateFBO` | Singleton | Per-view on `ecvGLView` | Phase 3 |
| `m_uniqueID` | Singleton | Per-view (already has `getUniqueID()`) | Phase 2 |
| `m_deferredPickingTimer` | Singleton | Per-view on `ecvGLView` | Phase 3 |
| `m_pickingTargetView` | Singleton | Per-view picking on `ecvGLView` | Phase 3 |
| **All Qt signals** | Singleton QObject | Per-view on `ecvGLView` QObject | Phase 1 |
| `m_visualizer3D` / `m_visualizer2D` | VtkDisplayTools singleton | Per-view on `ecvGLView` (already has VtkVis) | Phase 3 |
| `m_vtkWidget` | VtkDisplayTools singleton | Per-view (already owned by ecvGLView) | Phase 3 |

---

## Static API Call-Site Inventory

### Tier 1: Very High Traffic (50+ external call sites)

| Static Method | ~External Calls | Migration Strategy |
|--------------|----------------|-------------------|
| `GetCurrentScreen()` | ~150 | Replace with `ecvGenericGLDisplay::asWidget()` or `ecvViewManager::getActiveView()->asWidget()` |
| `RedrawDisplay()` | ~86 | Replace with `display->redraw()` where display is available; fallback to `ecvViewManager::getActiveView()->redraw()` |
| `DisplayNewMessage()` | ~52 | Move to `ecvGenericGLDisplay::displayMessage()` virtual method |
| `TheInstance()` | ~51 | Eliminate; pass `ecvGenericGLDisplay*` explicitly |

### Tier 2: High Traffic (20-49 external call sites)

| Static Method | ~External Calls | Migration Strategy |
|--------------|----------------|-------------------|
| `GetViewportParameters()` | ~33 | `display->getViewportParameters()` (already on interface) |
| `RedrawObject()` / `RedrawObjects()` | ~34 | `display->redrawObject(obj)` new virtual method |
| `HideShowEntities()` | ~31 | Route through `context.display->hideShowEntities()` |
| `UpdateScreen()` | ~30 | `display->refresh()` or `ecvViewManager::refreshAll()` |
| `RemoveEntities()` | ~29 | Route through `context.display` |
| `RemoveWidgets()` | ~27 | Route through display (mostly cc2DLabel internal) |
| `InvalidateViewport()` / `Deprecate3DLayer()` | ~21 | Per-view method on `ecvGenericGLDisplay` |
| `AddToOwnDB()` | ~18 | `display->addToOwnDB()` (already on interface) |
| `GetGLCameraParameters()` | ~19 | `display->getGLCameraParameters()` (already on interface) |

### Tier 3: Medium Traffic (10-19 external call sites)

| Static Method | ~External Calls | Migration Strategy |
|--------------|----------------|-------------------|
| `GetContext()` | ~12 | `display->getContext()` (already on interface) |
| `HasInstance()` | ~10 | Replace with null-check on display pointer |
| `Draw()` | ~13 | Route through `context.display` |
| `RemoveBB()` | ~8 | Per-view routing |
| `ChangeEntityProperties()` | ~6 | Per-view routing |

### Tier 4: Low Traffic (<10 external call sites)

All remaining statics — migrate on contact or batch at end.

---

## Signal/Slot Migration Map

### Per-View Signals (move to ecvGLView)

These signals are inherently per-window and should be emitted by the view that originated the event:

| Signal | Current Emitter | New Emitter | Connected Consumers |
|--------|----------------|-------------|-------------------|
| `itemPicked(ccHObject*, uint, int, int, CCVector3d&)` | ecvDisplayTools | ecvGLView | `ecvPickingHub`, `MainWindow` |
| `itemPickedFast(ccHObject*, int, int, int, int)` | ecvDisplayTools | ecvGLView | Internal picking |
| `pointPicked(ccHObject*, uint, int, int, CCVector3d&)` | ecvDisplayTools | ecvGLView | Point property dialogs |
| `viewMatRotated(ccGLMatrixd)` | ecvDisplayTools | ecvGLView | Camera tools |
| `cameraDisplaced(float)` | ecvDisplayTools | ecvGLView | Camera tools |
| `mouseWheelRotated(float)` | ecvDisplayTools | ecvGLView | Zoom tools |
| `perspectiveStateChanged()` | ecvDisplayTools | ecvGLView | UI |
| `pivotPointChanged(CCVector3d)` | ecvDisplayTools | ecvGLView | Camera tools |
| `cameraPosChanged(CCVector3d)` | ecvDisplayTools | ecvGLView | Camera dialogs |
| `cameraParamChanged()` | ecvDisplayTools | ecvGLView | Viewpoint toolbar |
| `leftButtonClicked(int, int)` | ecvDisplayTools | ecvGLView | Tools |
| `rightButtonClicked(int, int)` | ecvDisplayTools | ecvGLView | Context menus |
| `mouseMoved(int, int, Qt::MouseButtons)` | ecvDisplayTools | ecvGLView | Coordinate display |
| `buttonReleased()` | ecvDisplayTools | ecvGLView | Tools |
| `drawing3D()` | ecvDisplayTools | ecvGLView | GL filters |
| `filesDropped(QStringList)` | ecvDisplayTools | ecvGLView | File loading |
| `newLabel(ccHObject*)` | ecvDisplayTools | ecvGLView | DB tree |
| `labelmove2D(int, int, int, int)` | ecvDisplayTools | ecvGLView | Label tools |
| `exclusiveFullScreenToggled(bool)` | ecvDisplayTools | ecvGLView | UI |

### Global Signals (keep on ecvViewManager)

| Signal | Reason |
|--------|--------|
| `entitySelectionChanged(ccHObject*)` | App-wide selection state |
| `entitiesSelectionChanged(std::unordered_set<int>)` | App-wide selection state |
| `activeViewChanged(ecvGenericGLDisplay*)` | Already on ecvViewManager |

### Migration Pattern

For each per-view signal:
1. Declare on `ecvGLView` (it's already a QObject)
2. In `QVTKWidgetCustom`, emit via `m_ownerView->signalName(...)` instead of `m_tools->signalName(...)`
3. Connect consumers to `ecvViewManager::getActiveView()` signals; reconnect when active view changes
4. Or use `ecvViewManager` as a **relay**: connect all view signals to manager, manager re-emits

---

## Migration Phases

### Phase 0: Already Completed ✅

| Task | Status | Description |
|------|--------|-------------|
| P0.1 | ✅ | `activeItemsRef()` virtual on `ecvGenericGLDisplay` |
| P0.2 | ✅ | `Update2DLabel()` no longer manipulates `m_activeItems` |
| P0.3 | ✅ | `UpdateActiveItemsList()` routes to effective view |
| P0.4 | ✅ | `Pick2DLabel()` routes to effective view |
| P0.5 | ✅ | cc2DLabel visibility checks include `isEnabled()` |
| P0.6 | ✅ | cc2DLabel caption fixed during scene rotation |
| P0.7 | ✅ | VTK bypass for label drag (no camera rotation) |

### Phase 1: Per-View Signal Hub ✅ (Partial — dual-emit routing)

**Goal**: Move Qt signals from `ecvDisplayTools` to `ecvGLView`, with relay through `ecvViewManager`.

| Task ID | File | Description | Status |
|---------|------|-------------|--------|
| P1.1 | `ecvGLView.h` | Declare all per-view signals on `ecvGLView` (QObject) — 30+ signals | ✅ Done |
| P1.2 | `ecvViewManager.h` | Add relay signals (entitySelectionChanged, newLabel, filesDropped, cameraParamChanged) | ✅ Done |
| P1.3 | `QVTKWidgetCustom.cpp` | Dual-emit: `emit m_ownerView->signal()` + `emit m_tools->signal()` at 20 sites | ✅ Done |
| P1.4 | `QVTKWidgetCustom.cpp` | Primary window fallback (no `m_ownerView`) — singleton emit preserved | ✅ Done (auto-fallback) |
| P1.5 | `MainWindow.cpp` | Reconnect signal consumers to `ecvViewManager` relay signals | Pending (Phase 4) |
| P1.6 | Plugin signal consumers | Update `connect(TheInstance(), ...)` to `connect(viewManager, ...)` | Pending (Phase 4) |

**Compile checkpoint**: ✅ Compiles and links successfully. Dual-emit ensures backward compatibility.

### Phase 2: QVTKWidgetCustom Decoupling ✅ (Per-view routing for timers & state)

**Goal**: `QVTKWidgetCustom` routes through `m_ownerView` for all per-view state when available.

| Task ID | File | Description | Status |
|---------|------|-------------|--------|
| P2.1 | `ecvGLView.h/cpp` | Add per-view `m_timer`, `elapsedMs()`, `scheduleFullRedraw()`, `startDeferredPicking()` | ✅ Done |
| P2.2 | `QVTKWidgetCustom.cpp` | Route `m_tools->m_timer.elapsed()` → `m_ownerView->elapsedMs()` (2 sites) | ✅ Done |
| P2.3 | `QVTKWidgetCustom.cpp` | Route `m_tools->m_deferredPickingTimer` → `m_ownerView->deferredPickingTimer()` (2 sites) | ✅ Done |
| P2.4 | `QVTKWidgetCustom.cpp` | Route `m_tools->scheduleFullRedraw()` → `m_ownerView->scheduleFullRedraw()` | ✅ Done |
| P2.5 | `QVTKWidgetCustom.cpp` | Guard `m_hotZoneOwnedBySingleton` to only apply for primary window | ✅ Done |
| P2.6 | `QVTKWidgetCustom.h/cpp` | Full `m_tools` removal (member still exists for backward compat) | Deferred to Phase 7 |
| P2.7 | `QVTKWidgetCustom.cpp` | Remaining static calls (`GetCurrentScreen`, `ProcessClickableItems`) | Deferred to Phase 4 |

**Compile checkpoint**: ✅ Compiles and links successfully.

### Phase 3: VtkDisplayTools Per-View Pipeline ✅ (Virtual dispatch interface)

**Goal**: Add per-view virtual methods to `ecvGenericGLDisplay` interface for dispatch.

| Task ID | File | Description | Status |
|---------|------|-------------|--------|
| P3.1 | `ecvGenericGLDisplay.h` | Add virtual `invalidateViewport()`, `deprecate3DLayer()`, `displayNewMessage()` | ✅ Done |
| P3.2 | `ecvGLView.h/cpp` | Implement `invalidateViewport()`, `deprecate3DLayer()`, `displayNewMessage()` | ✅ Done |
| P3.3 | `ecvViewManager.h/cpp` | Add `activeWidget()`, `invalidateActiveViewport()`, `deprecateActive3DLayer()`, `displayMessageOnActiveView()` | ✅ Done |
| P3.4 | `VtkDisplayTools.cpp` | Remove `switchActiveView` / `ScopedHotZoneRender` | Deferred (functional, but swap still used) |
| P3.5 | `VtkDisplayTools.cpp` | Remove `m_primaryVis` swap state | Deferred (Phase 7) |

**Compile checkpoint**: ✅ Compiles and links successfully.

### Phase 4: Static Method Elimination (High-Traffic) ✅ (Infrastructure + QVTKWidgetCustom batch)

**Goal**: Add per-view dispatchers and begin replacing high-traffic statics.

**Completed:**
| Task ID | Description | Status |
|---------|-------------|--------|
| P4.A | `ecvViewManager` dispatchers: `activeWidget()`, `invalidateActiveViewport()`, `deprecateActive3DLayer()`, `displayMessageOnActiveView()` | ✅ Done |
| P4.B | Replace `InvalidateViewport()` + `Deprecate3DLayer()` in QVTKWidgetCustom (4 call sites) | ✅ Done |
| P4.C | Replace `ToBeRefreshed()` in QVTKWidgetCustom (2 call sites) | ✅ Done |

**Remaining (Deferred — use established pattern for incremental migration):**

| Target Static | ~Call Sites | Strategy |
|--------------|-------------|----------|
| `GetCurrentScreen()` | ~150 | `display->asWidget()` or `ecvViewManager::activeWidget()` |
| `RedrawDisplay()` | ~86 | `display->redraw()` |
| `DisplayNewMessage()` | ~52 | `display->displayNewMessage()` |
| `TheInstance()` signal connects | ~51 | `ecvViewManager::instance()` relay |
| Remaining statics | ~200+ | Same pattern: per-view dispatch with singleton fallback |

**Strategy for each call site** (applicable to all remaining phases):
1. If the call site has a `CC_DRAW_CONTEXT& context` → use `context.display`
2. If the call site has a `ccHObject*` → use `obj->getDisplay()`
3. If the call site is a UI handler → use `ecvViewManager::getActiveView()`
4. If none of the above → add explicit `ecvGenericGLDisplay*` parameter

### Phase 5: Draw Pipeline Per-View Routing ✅ (ccHObject core routing)

**Goal**: Entity draw pipeline routes through `context.display` and `getDisplay()` instead of singleton.

| Task ID | File | Description | Status |
|---------|------|-------------|--------|
| P5.1 | `ecvHObject.cpp` | `notifyGeometryUpdate()` uses `getDisplay()->invalidateViewport()/deprecate3DLayer()` | ✅ Done |
| P5.2 | `ecvHObject.cpp` | `updateNameIn3DRecursive()` uses `ecvViewManager::getEffectiveView()` for camera params | ✅ Done |
| P5.3 | `ecvHObject.cpp` | `redrawDisplay()` unconditionally calls `m_currentDisplay->redraw()` | ✅ Done |
| P5.4 | `ecvHObject.cpp` | `draw()` already uses `context.display` (verified: `isDisplayedIn`, `HideShowEntities`) | ✅ Verified |
| P5.5 | `ecvHObject.cpp` | `toggleVisibility_recursive()` already sets `context.display = getDisplay()` | ✅ Verified |
| P5.6 | `MainWindow.cpp` | Added `getActiveGLWidget()` helper that routes via `ecvViewManager` | ✅ Done |

### Phase 6: Python Wrapper Alignment

**Goal**: `ccDisplayTools.cpp` Python bindings updated for per-view API.

| Task ID | File | Description | Est. Lines |
|---------|------|-------------|-----------|
| P6.1 | `ccDisplayTools.cpp` | Replace ~117 `ecvDisplayTools::` static calls with per-view API | ~100 |
| P6.2 | `ccDisplayTools.cpp` | Add `display` parameter to Python-facing functions where needed | ~30 |
| P6.3 | Python test scripts | Update any Python tests that use the old API | ~20 |

### Phase 7: Singleton Removal & Final Cleanup

**Goal**: `ecvDisplayTools` is no longer a singleton or is removed entirely.

| Task ID | File | Description | Est. Lines |
|---------|------|-------------|-----------|
| P7.1 | `ecvDisplayTools.h/cpp` | Remove `s_tools` singleton, `Init()`, `ReleaseInstance()`, `TheInstance()` | ~-100 |
| P7.2 | `ecvDisplayTools.h` | Convert remaining static utilities to free functions or `ecvViewManager` methods | ~50 |
| P7.3 | `VtkDisplayTools.h/cpp` | Remove or flatten into `ecvGLView` | ~-300 |
| P7.4 | `MainWindow.cpp` | Create initial `ecvGLView` as the "first" view (not a singleton) | ~30 |
| P7.5 | `ecvViewManager.h/cpp` | Handle "first view" creation and registration | ~20 |
| P7.6 | All files | Remove `#include "ecvDisplayTools.h"` where no longer needed | ~50 |
| P7.7 | Documentation | Update `multi-window-views.md` to reflect final architecture | ~100 |

---

## Detailed Task Breakdown

### Phase 1 Detailed: Per-View Signal Hub

#### P1.1: Declare signals on ecvGLView

```
File: libs/VtkEngine/Visualization/ecvGLView.h

Add to signals section (ecvGLView is already Q_OBJECT):

signals:
    // Picking
    void itemPicked(ccHObject* entity, unsigned itemIdx, int x, int y, const CCVector3d& P);
    void itemPickedFast(ccHObject* entity, int subEntityID, int x, int y);
    void pointPicked(ccHObject* entity, unsigned pointIndex, int x, int y, const CCVector3d& P);

    // Camera
    void viewMatRotated(const ccGLMatrixd& rotMat);
    void cameraDisplaced(float ddist);
    void mouseWheelRotated(float wheelDelta_deg);
    void perspectiveStateChanged();
    void pivotPointChanged(const CCVector3d&);
    void cameraPosChanged(const CCVector3d&);
    void cameraParamChanged();

    // Mouse
    void leftButtonClicked(int x, int y);
    void rightButtonClicked(int x, int y);
    void mouseMoved(int x, int y, Qt::MouseButtons buttons);
    void buttonReleased();

    // Other
    void drawing3D();
    void filesDropped(const QStringList& filenames);
    void newLabel(ccHObject* label);
    void labelmove2D(int x, int y, int dx, int dy);
    void exclusiveFullScreenToggled(bool fullScreen);
```

#### P1.2: ecvViewManager Signal Relay

```
File: libs/CV_db/include/ecvViewManager.h + .cpp

Add relay signals that mirror ecvGLView signals.
When setActiveView() is called:
  1. Disconnect old view's signals from relay
  2. Connect new view's signals to relay
  3. Emit activeViewChanged()

Consumers connect to ecvViewManager relay signals for "active view" behavior.
```

#### P1.3: QVTKWidgetCustom emit routing

```
File: libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp

For each `emit m_tools->signalName(args)`:
  if (m_ownerView) {
      emit m_ownerView->signalName(args);
  } else {
      // Fallback during transition: emit on singleton
      emit m_tools->signalName(args);
  }
```

### Phase 2 Detailed: QVTKWidgetCustom Decoupling

#### P2.1: Replace m_tools pointer type

```
File: libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h

Before: ecvDisplayTools* m_tools;
After:  ecvGenericGLDisplay* m_display;  // per-view display interface
```

#### P2.2: Static call replacement in QVTKWidgetCustom

All `ecvDisplayTools::StaticMethod()` calls in QVTKWidgetCustom.cpp (~115 occurrences) need to be replaced.

Pattern: `ecvDisplayTools::GlWidth()` → `m_display->glWidth()`
Pattern: `ecvDisplayTools::GetCurrentScreen()` → `this` or `m_display->asWidget()`
Pattern: `ecvDisplayTools::GetContext(ctx)` → `m_display->getContext(ctx)`
Pattern: `ecvDisplayTools::FilterByEntityType(labels, type)` → per-view filter using `m_display->getSceneDB()`

### Phase 4 Detailed: GetCurrentScreen Elimination

`GetCurrentScreen()` (~150 call sites) is the highest-traffic static. Replacement strategy:

| Call site context | Replacement |
|-------------------|-------------|
| Inside `QVTKWidgetCustom` methods | `this` (the widget itself) |
| Inside `ecvGLView` methods | `m_vtkWidget` or `asWidget()` |
| Inside `ecvHObject::draw()` | `context.display->asWidget()` |
| Inside `MainWindow` UI handlers | `ecvViewManager::getActiveView()->asWidget()` |
| Inside plugins | `ecvViewManager::getActiveView()->asWidget()` |
| Inside `ecvDisplayTools.cpp` draw code | `effectiveCtx()` → per-view widget |

---

## Risk Matrix

| Risk | Severity | Mitigation |
|------|----------|------------|
| 150+ `GetCurrentScreen()` call sites | High | Automated refactoring; search-and-replace with manual review |
| Signal reconnection breaks tool behavior | High | Phase 1 adds relay without removing old signals; gradual migration |
| VTK re-entrancy during draw | Medium | Keep single-thread rendering; test with multiple views |
| Python wrapper divergence | Medium | Phase 6 explicitly; test Python scripts |
| Plugin breakage | Medium | Plugins use limited API surface; migrate last |
| Circular header dependencies | Low | `ecvGenericGLDisplay` forward-declare only; keep interface minimal |
| Performance regression (per-view timers) | Low | Profile before/after |

---

## Compilation & Test Strategy

### Per-Phase Compilation

Each phase MUST compile independently:

```bash
activate_pyenv 3.11 && cd build_app && make -j12
```

### Test Checklist (after each phase)

- [ ] Single window: load point cloud, rotate, zoom, pick points
- [ ] Multi-window: create 2+ views, verify entity isolation
- [ ] cc2DLabel: create 1/2/3 point labels, drag caption, right-click collapse
- [ ] cc2DLabel: hide/show via DB tree checkbox
- [ ] Show name: verify background renders in correct window
- [ ] Close tab: verify tab closes (not renames)
- [ ] Layout numbering: verify number reuse after close
- [ ] Plugin tools: segmentation, registration, SRA (basic smoke test)

---

## ParaView Reference Patterns

### pqView (per-view base)

```
// ParaView: each view is independent
class pqView : public pqProxy {
    QWidget* widget();
    void render();
    void forceRender();
    vtkSMViewProxy* getViewProxy();
signals:
    void beginRender();
    void endRender();
    void representationAdded(pqRepresentation*);
    void representationRemoved(pqRepresentation*);
};
```

**ACloudViewer equivalent**: `ecvGLView` — already has `redraw()`, VTK pipeline, widget. Needs signals.

### pqActiveObjects (active view tracking)

```
// ParaView: singleton tracks which view is "active" (has UI focus)
class pqActiveObjects : public QObject {
    pqView* activeView();
    void setActiveView(pqView*);
signals:
    void viewChanged(pqView*);
    void sourceChanged(pqPipelineSource*);
    void representationChanged(pqRepresentation*);
};
```

**ACloudViewer equivalent**: `ecvViewManager` — already tracks `m_activeView`. Needs signal relay.

### Key Pattern: No Singleton Display Tools

ParaView has NO equivalent of `ecvDisplayTools` singleton. Each `pqRenderView` owns:
- Its own `vtkSMRenderViewProxy`
- Its own camera/interaction state
- Its own representations
- Its own signals

The **only singleton** is `pqActiveObjects` which just tracks which view is "active" — it does NOT hold rendering state.

---

## Estimated Effort

| Phase | Est. Lines Changed | Complexity |
|-------|-------------------|------------|
| Phase 0 (done) | ~200 | Low |
| Phase 1 | ~230 | Medium |
| Phase 2 | ~200 | Medium |
| Phase 3 | ~400 | High |
| Phase 4 | ~500 | High (mechanical, many files) |
| Phase 5 | ~160 | Medium |
| Phase 6 | ~150 | Low |
| Phase 7 | ~350 | Medium |
| **Total** | **~2190** | |

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-04-29 | 1.0 | Initial migration plan |
| 2026-04-29 | 1.1 | Phase 1 complete: per-view signals on ecvGLView (30+ signals), dual-emit in QVTKWidgetCustom (20 sites) |
| 2026-04-29 | 1.2 | Phase 2 complete: per-view timer, scheduling, deferred picking routing |
| 2026-04-29 | 1.3 | Phase 3 complete: virtual dispatch methods on ecvGenericGLDisplay + ecvGLView implementations |
| 2026-04-29 | 1.4 | Phase 4 partial: ecvViewManager active-view dispatchers + QVTKWidgetCustom static replacements |
| 2026-04-29 | 1.5 | Phase 5 complete: ccHObject draw pipeline routes through per-view display |
| 2026-04-29 | 1.6 | Phase 4 continued: MainWindow getActiveGLWidget() convenience method |
