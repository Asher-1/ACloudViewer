# ecvDisplayTools Singleton Removal — Per-View Migration Plan

> Date: 2026-04-29
> Version: 1.0
> Target: Align with ParaView multi-window architecture — every window is equal, no primary/secondary distinction
> Reference: [multi-window-views.md](multi-window-views.md), [multi-window-paraview-alignment-design.md](multi-window-paraview-alignment-design.md), ParaView `/Users/asher/develop/code/autopilot/MVS/ParaView` (macOS) / `/home/ludahai/develop/code/github/ParaView` (Linux), CloudCompare `/Users/asher/develop/code/autopilot/MVS/CloudCompare` (macOS) / `/home/ludahai/develop/code/github/CloudCompare` (Linux)

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
| P1.5 | `MainWindow.cpp` | Reconnect signal consumers to `ecvViewManager` relay signals | ✅ Done (Phase M2 信号迁移) |
| P1.6 | Plugin signal consumers | Update `connect(TheInstance(), ...)` to `connect(viewManager, ...)` | ✅ Done (Phase M2 信号迁移) |

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
| P2.6 | `QVTKWidgetCustom.h/cpp` | Full `m_tools` removal (member still exists for backward compat) | ✅ Superseded by Phase M2/M3 |
| P2.7 | `QVTKWidgetCustom.cpp` | Remaining static calls (`GetCurrentScreen`, `ProcessClickableItems`) | ✅ Superseded by Phase M3/M4 |

**Compile checkpoint**: ✅ Compiles and links successfully.

### Phase 3: VtkDisplayTools Per-View Pipeline ✅ (Virtual dispatch interface)

**Goal**: Add per-view virtual methods to `ecvGenericGLDisplay` interface for dispatch.

| Task ID | File | Description | Status |
|---------|------|-------------|--------|
| P3.1 | `ecvGenericGLDisplay.h` | Add virtual `invalidateViewport()`, `deprecate3DLayer()`, `displayNewMessage()` | ✅ Done |
| P3.2 | `ecvGLView.h/cpp` | Implement `invalidateViewport()`, `deprecate3DLayer()`, `displayNewMessage()` | ✅ Done |
| P3.3 | `ecvViewManager.h/cpp` | Add `activeWidget()`, `invalidateActiveViewport()`, `deprecateActive3DLayer()`, `displayMessageOnActiveView()` | ✅ Done |
| P3.4 | `VtkDisplayTools.cpp` | Remove `switchActiveView` / `ScopedHotZoneRender` | ✅ Done (Phase M4 — ScopedHotZoneRender 已删除) |
| P3.5 | `VtkDisplayTools.cpp` | Remove `m_primaryVis` swap state | ✅ Done (Phase M3 Category A 删除) |

**Compile checkpoint**: ✅ Compiles and links successfully.

### Phase 4: Static Method Elimination (High-Traffic) — Batches 4.1–4.7 ✅

**Goal**: Add per-view dispatchers and replace high-traffic statics across the codebase.

**Infrastructure (Complete):**

| Task ID | Description | Status |
|---------|-------------|--------|
| P4.A | `ecvViewManager` dispatchers: `activeWidget()`, `invalidateActiveViewport()`, `deprecateActive3DLayer()`, `displayMessageOnActiveView()` | ✅ Done |
| P4.B | Replace `InvalidateViewport()` + `Deprecate3DLayer()` in QVTKWidgetCustom (4 call sites) | ✅ Done |
| P4.C | Replace `ToBeRefreshed()` in QVTKWidgetCustom (2 call sites) | ✅ Done |

**Batch Progress (2026-04-30):**

| Batch | Target | Before | After | Migrated | Status |
|-------|--------|--------|-------|----------|--------|
| 4.1 | MainWindow.cpp | 91 | 59 | 32 | ✅ Complete (residuals: VTK/lifecycle) |
| 4.2 | QVTKWidgetCustom.cpp | 112 | 73 | 39 | ✅ Complete (residuals: VTK internals) |
| 4.3 | SegTool + TraceTool | 72 | 20 | 52 | ✅ Complete (residuals: utilities/widgets) |
| 4.4 | Medium-Traffic Dialogs (15 files) | ~280 | ~142 | ~138 | ✅ Complete (13 files fully cleared) |
| 4.5 | Libs tier (20+ files) | ~350 | ~265 | ~85 | ✅ Complete (6 files fully cleared) |
| 4.6 | Plugins tier (excl. Python wrapper) | ~232 | 36 | ~196 | ✅ Complete (18 files fully cleared) |
| 4.7 | VtkEngine internal | ~49 | — | — | ✅ Superseded by Phase M (M1-M3) |

**Files Fully Cleared of `ecvDisplayTools::`** (19 files):
`ecvRasterizeTool`, `ecvContourExtractorDlg(.cpp/.h)`, `ecv2.5DimEditor`, `ecvPointListPickingDlg`,
`ecvVolumeCalcTool`, `MovieGrabberWidget`, `ecvGraphicalTransformationTool`, `ecvPointPickingGenericInterface`,
`ecvDeepSemanticSegmentationTool`, `ecvFilterByLabelDlg`, `ReconstructionWidget`, `ecvAlignDlg`,
`ecvColorLevelsDlg`, `ecvCustomViewpointsToolbar`, `cvRenderViewSelectionReaction`, `ecvPickingHub.h`

**Residual Analysis (app/ 142 + libs/ ~265 = ~407 remaining excl. ecvDisplayTools.cpp):**

| Category | ~Count | Description |
|----------|--------|-------------|
| A: VTK-specific | ~133 | GetVisualizer3D, Draw/RemoveWidgets, RemoveEntities, HideShowEntities, ToVtkCoordinates, ChangeEntityProperties — no per-view equivalent yet |
| B: Singleton lifecycle | ~105 | TheInstance(), HasInstance(), signal connects for non-relayed signals |
| C: Still migratable | ~80 | Viewport/camera getters, redraw, picking, display messages — diminishing returns |
| D: Utility/Static | ~32 | ConvertToEntityType, GetTextDisplayFont, USE_2D — pure utilities |

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

### Phase 6: Python Wrapper Alignment (Deferred to Phase 7)

**Goal**: `ccDisplayTools.cpp` Python bindings updated for per-view API.

**Status**: Deferred. The Python wrapper binds enum constants and static methods that still work via backward-compatible singleton API. Will be updated when Phase 7 removes the singleton.

| Task ID | File | Description | Est. Lines |
|---------|------|-------------|-----------|
| P6.1 | `ccDisplayTools.cpp` | Replace ~117 `ecvDisplayTools::` static calls with per-view API | ~100 |
| P6.2 | `ccDisplayTools.cpp` | Add `display` parameter to Python-facing functions where needed | ~30 |
| P6.3 | Python test scripts | Update any Python tests that use the old API | ~20 |

### Phase 7: Singleton Removal & Final Cleanup

**Goal**: `ecvDisplayTools` is no longer a singleton or is removed entirely.

#### Phase 7a: New Per-View Virtual APIs on ecvGenericGLDisplay ✅ (Interface + 2 migration waves)

**Status**: 40+ virtual methods added to `ecvGenericGLDisplay`, implemented in `ecvDisplayTools` (override) and `ecvGLView` (delegation to VtkDisplayTools via resolveVisualizer). Two waves of call-site migration completed.

**Counts after Phase 7a wave 2**: app/ 47, libs/CV_db 39, libs/CVAppCommon 7, plugins/ 9, total migrateable external ~85 (from ~460 at start of Phase 7).

**Wave 2 new virtual methods** (added to ecvGenericGLDisplay, override on ecvDisplayTools+ecvGLView):
- `toVtkCoordinates`, `getClick3DPos`, `setView`, `getCurrentViewDir`
- `setPivotPoint`, `setPivotVisibility(PivotVisibility)`, `setAutoPickPivotAtCenter`, `resetCenterOfRotation`
- `isRotationAxisLocked`, `lockRotationAxis`, `toggleCameraOrientationWidget`, `toggleOrientationMarker`, `toggleDebugTrace`
- `renderToFile`, `removeBB(QString)`, `removeBB(ccGLDrawContext)`, `setExclusiveFullScreenFlag`
- `get/setObjectLightIntensity`, `get/setLightIntensity`, `get/setDataAxesGridProperties`
- `textDisplayFont`, `display2DText`, `update2DLabels`, `moveCamera`, `rotateBaseViewMat`, `load/saveCameraParameters`

**Files fully cleared** (additional in wave 2): ecvSphere, ecv2DViewportLabel, ecvClipBox, ecvGenericFiltersTool, ecvGenericMeasurementTools, ecvGenericCameraTool, GamepadInput, ecvCustomViewpointButtonDlg, ecvDisplayOptionsDlg, ecvPropertiesTreeDelegate, ecvGraphicalSegmentationTool.

The following virtual methods were moved from `ecvDisplayTools` to `ecvGenericGLDisplay`:

| New Virtual Method | Replaces Static | ~Call Sites | Priority |
|--------------------|----------------|-------------|----------|
| `drawWidgets(WIDGETS_PARAMETER&, bool)` | `DrawWidgets` | ~8 | High |
| `removeWidgets(WIDGETS_PARAMETER&)` | `RemoveWidgets` | ~12 | High |
| `removeEntities(CC_DRAW_CONTEXT&)` | `RemoveEntities` | ~20 | High |
| `hideShowEntities(CC_DRAW_CONTEXT&)` | `HideShowEntities` | ~25 | High |
| `draw(CC_DRAW_CONTEXT&)` | `Draw`/`DrawBBox` | ~10 | High |
| `getVisualizer3D()` → `VtkVis*` | `GetVisualizer3D` | ~14 | Medium |
| `toVtkCoordinates(int&, int&, int&)` | `ToVtkCoordinates` | ~8 | Medium |
| `pickUnproject(int, int, CCVector3d&)` | `GetClick3DPos` | ~3 | Medium |
| `changeEntityProperties(PROPERTY_PARAM&)` | `ChangeEntityProperties`/`ChangeOpacity` | ~4 | Medium |
| `renderToFile(QString, float, bool)` | `RenderToFile`/`RenderToImage` | ~3 | Low |
| `setView(CC_VIEW_ORIENTATION, ccBBox*)` | `SetView` | ~4 | Low |
| `updateCamera()` | `UpdateCamera` | ~2 | Low |
| `rotateWithAxis(CCVector3d, float, QPoint)` | `RotateWithAxis` | ~1 | Low |
| `moveCamera(float, float, float)` | `MoveCamera` | ~3 | Low |
| `toggleCameraOrientationWidget(bool)` | `ToggleCameraOrientationWidget` | ~3 | Low |
| `displayText(text, x, y, align, opacity, color, font, display)` | `DisplayText` | ~4 | Low |
| `filterByEntityType(Container&, CV_TYPES)` | `FilterByEntityType` | ~5 | Low |
| `setupProjectiveViewport(params)` | `SetupProjectiveViewport` | ~1 | Low |
| `setObjectLightIntensity(float)`/`get` | `Set/GetObjectLightIntensity` | ~2 | Low |
| `getDataAxesGridProperties()`/`set` | `Get/SetDataAxesGridProperties` | ~2 | Low |

**Utility methods** (move to free functions or `ecvViewManager`):
- `ConvertToEntityType` → `ecvEntityTypeUtils::convert()`
- `GetTextDisplayFont`/`GetLabelDisplayFont`/`GetOptimizedFontSize` → `ecvGuiParameters`
- `USE_2D`/`USE_VTK_PICK` → `ecvViewContext` flags

#### Phase 7b: Singleton Elimination ✅ (All items done except Python wrapper → Phase M5)

| Task ID | File | Description | Est. Lines | Status |
|---------|------|-------------|-----------|--------|
| P7.8 | App/lib/plugin callers | Replace migratable `ecvDisplayTools::` calls with per-view virtual calls | ~200 | ✅ Done (20+ files cleared) |
| P7.9 | Signal connects (~27) | Move from `TheInstance()` signals to per-view or `ecvViewManager` relay | ~60 | ✅ Done (9 relay signals added) |
| P7.1 | `ecvDisplayTools.h/cpp` | Remove `s_tools` singleton, `Init()`, `ReleaseInstance()`, `TheInstance()` | ~-100 | ✅ Done — `s_tools` raw ptr; `initializeSharedInstance`/`releaseSharedInstance`/`sharedTools` replace lifecycle; `ecvSingleton.h` dropped |
| P7.2 | `ecvDisplayTools.h` | Convert remaining static utilities to free functions or `ecvViewManager` methods | ~50 | ✅ See TODO 4 (partially extracted; rest deferred to Phase M5) |
| P7.3 | `VtkDisplayTools.h/cpp` | Merge rendering pipeline into `ecvGLView` (each view owns its `VtkDisplayTools`) | ~-300 | ✅ Done — ecvGLView 33→1 static refs; all via m_displayTools-> or m_ctx |
| P7.4 | `MainWindow.cpp` | Use `ecvViewManager::initDisplayTools()`/`releaseDisplayTools()` | ~30 | ✅ Done — all `TheInstance()` replaced with ecvViewManager accessors |
| P7.5 | `ecvViewManager.h/cpp` | `initDisplayTools()`/`releaseDisplayTools()`/`displayTools()` lifecycle | ~20 | ✅ Done |
| P7.6 | All files | Remove `#include "ecvDisplayTools.h"` where no longer needed | ~50 | ✅ Done (TODO 8) — 28 unused includes removed |
| P7.7 | Documentation | Update `multi-window-views.md` to reflect final architecture | ~100 | ✅ Done (TODO 9) |

#### Phase 7c: Python Wrapper Update (Phase 6 merge)

| Task ID | File | Description | Est. Lines |
|---------|------|-------------|-----------|
| P7.10 | `ccDisplayTools.cpp` | Replace ~117 `ecvDisplayTools::` static calls with per-view API | ~100 |
| P7.11 | `ccDisplayTools.cpp` | Add `display` parameter to Python-facing functions where needed | ~30 |

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

## Remaining TODO Master List (as of 2026-04-30 v6.0)

This section provides a **complete, actionable inventory** of all remaining work items to fully remove the `ecvDisplayTools` singleton. Items are ordered by recommended execution sequence.

### Current State Summary

| Directory | `ecvDisplayTools::` Refs | Files | Description |
|-----------|--------------------------|-------|-------------|
| **app/** | 37 | 3 | MainWindow(31) + ecvDBRoot(5) + ecvPointPropertiesDlg(1) |
| **libs/CV_db/** | 47 | 10 | ViewManager-relay(21) + RedrawScope(6) + GenericGLDisplay-defaults(5) + HObject(3) + GuiParams(3) + GBLSensor(2) + DrawableObject(2-comments) + GenericPointCloud(2-comments) + ViewManager.h(1-comment) + GenericGLDisplay.h(2-comments) |
| **libs/CVAppCommon/** | 2 | 1 | ecvCameraParamEditDlg(2 - TheInstance+destroyed signal) |
| **libs/VtkEngine/** | 137 | 8 | QVTKWidgetCustom(73) + ecvGLView(39-delegation) + VtkDisplayTools(18-internal) + VtkVis(9) + EditCameraTool(2) + VtkMeasurementTools(2) + QVTKWidgetCustom.h(4-types) + CustomVtkCaptionWidget(2) |
| **plugins/** | 3 | 2 | qSRA.cpp(1-RedrawObject) + qCanupo2DViewDialog(2-comments) |
| **Python wrapper** | 117 | 1 | ccDisplayTools.cpp(117 - static bindings) |
| **#include only** | — | ~30 | Files that still include header but have 0 `ecvDisplayTools::` calls |
| **GRAND TOTAL** | **343** | **~55** | |

### TODO 1: Phase 2.6 — QVTKWidgetCustom Full m_ownerView Routing ✅

**Priority**: HIGH | **Complexity**: HIGH | **Est. Lines**: ~400 | **Prerequisite**: None | **Status**: COMPLETE

Migrate all 73 remaining `ecvDisplayTools::` calls in `QVTKWidgetCustom.cpp` to route through `m_ownerView` (ecvGLView) or `m_display` (ecvGenericGLDisplay). This is the single largest file by call count.

**Detailed call inventory (73 calls):**

| Category | Count | Methods | Migration Strategy |
|----------|-------|---------|--------------------|
| **Picking/Click** | 10 | `GetClick3DPos`(2), `SetPivotPoint`(2), `ProcessClickableItems`(1), `StartPicking`(1), `PickObject`(implicit), `ComputeActualPixelSize`(3) | `m_ownerView->getClick3DPos()`, `m_ownerView->setPivotPoint()`, etc. via virtuals |
| **View state** | 12 | `GetViewportParameters`(2), `GetCurrentScreen`(1), `ExclusiveFullScreen`(1), `GetDevicePixelRatio`(2), `GlWidth`(1), `GlHeight`(1), `Width`(1), `Height`(1), `ConvertMousePositionToOrientation`(2) | `m_ownerView->getViewportParameters()`, `m_ownerView->glWidth()`, etc. |
| **Rendering** | 14 | `InvalidateViewport`(2), `Deprecate3DLayer`(2), `RedrawDisplay`(3), `RefreshDisplay`(1), `ToBeRefreshed`(2), `Redraw2DLabel`(2) | `m_ownerView->invalidateViewport()`, `m_ownerView->redrawDisplay()`, etc. |
| **Camera/Transform** | 8 | `MoveCamera`(1), `RotateWithAxis`(1), `ShowPivotSymbol`(2), `UpdateZoom`(1), `ResizeGL`(1), `UpdateCamera`(1), `Update`(1) | `m_ownerView->moveCamera()`, etc. |
| **Items/Entities** | 8 | `FilterByEntityType`(4), `UpdateActiveItemsList`(2), `AddToOwnDB`(1), `RemoveFromOwnDB`(1) | `m_ownerView->filterByEntityType()`, etc. — already have virtuals |
| **Picking params** | 5 | `SetZNearCoef`(1), `SetFov`(1), `SetPointSize`(1), `UpdateNamePoseRecursive`(3) | New virtuals needed for Set*; UpdateNamePoseRecursive → per-view |
| **Viewport defaults** | 4 | `SetViewportDefaultPointSize`(1), `SetViewportDefaultLineWidth`(1), `DisplayNewMessage`(1), `SetGLViewport`(implicit) | Per-view setters |
| **Identity/Type** | 6 | `TheInstance`(1 comparison), `HotZone` type(4), `USE_VTK_PICK`(1) | Enum→ecvGenericGLDisplay; TheInstance→ecvViewManager; USE_VTK_PICK→static constant |
| **Line refs** | 6 | `ToVtkCoordinates`(2), `GetGLCameraParameters`(1), `RotateBaseViewMat`(implicit), other | Already have virtuals |

**Sub-tasks:**

| Sub-task | Description | Est. Lines |
|----------|-------------|-----------|
| P2.6.1 | Add missing virtual methods to `ecvGenericGLDisplay`: `setZNearCoef`, `setFov`, `setPointSize`, `updateNamePoseRecursive`, `computeActualPixelSize`, `processClickableItems`, `convertMousePositionToOrientation`, `showPivotSymbol`, `setViewportDefaultPointSize`, `setViewportDefaultLineWidth`, `rotateWithAxis`, `updateZoom`, `resizeGL`, `update`, `update2DLabel` (Redraw2DLabel), `startPicking`, `setRedrawRecursive` | ~40 |
| P2.6.2 | Implement overrides in `ecvDisplayTools.h/cpp` (delegate to static counterparts) | ~40 |
| P2.6.3 | Implement overrides in `ecvGLView.h/cpp` (delegate to TheInstance or use m_visualizer3D) | ~60 |
| P2.6.4 | Replace all 73 calls in `QVTKWidgetCustom.cpp` with `m_ownerView->method()` | ~200 |
| P2.6.5 | Replace `ecvDisplayTools::HotZone` type references with forward-declared or extracted type | ~20 |
| P2.6.6 | Replace `m_tools` member usage — make it `ecvGenericGLDisplay*` or remove entirely | ~20 |
| P2.6.7 | Compile & verify | — |

**Key pattern for QVTKWidgetCustom:**
```cpp
// BEFORE:
ecvDisplayTools::InvalidateViewport();
// AFTER:
if (m_ownerView) m_ownerView->invalidateViewport();
else if (m_tools) m_tools->invalidateViewport();  // fallback for primary window
```

### TODO 2: Phase 7.3 — VtkEngine Internal Migration ✅

**Priority**: HIGH | **Complexity**: HIGH | **Est. Lines**: ~300 | **Prerequisite**: Phase 2.6

Merge `VtkDisplayTools` rendering pipeline into `ecvGLView`, so each view owns its own display tools instance instead of going through the singleton.

**Detailed call inventory:**

| File | Calls | Key Methods | Strategy |
|------|-------|-------------|----------|
| `ecvGLView.cpp` | 39 | `TheInstance()` delegations | Replace with `this->method()` after VtkDisplayTools merge |
| `VtkDisplayTools.cpp` | 18 | Internal routing (`USE_2D`, `onPointPicking`, `HotZone`, `DrawClickableItems`, `GetContext`, `GetLabelDisplayFont`, `DrawWidgets`, `SetGLViewport`) | These become per-view methods on the merged class |
| `VtkVis.cpp` | 9 | `GetSceneDB`(2), `OrientationMarkerShown`(1), `OverlayEntitiesAreDisplayed`(1), `DisplayOverlayEntities`(2), `ToggleOrientationMarker`(3) | Route via owning `ecvGLView` pointer |
| `EditCameraTool.cpp` | 2 | `GetVisualizer3D`(2) | Accept `ecvGenericVisualizer3D*` as constructor parameter |
| `VtkMeasurementTools.cpp` | 2 | `GetVisualizer3D`(2) | Accept `ecvGenericVisualizer3D*` via setter or constructor |
| `CustomVtkCaptionWidget.cpp` | 2 | `TheInstance`(1), `HasInstance`(1) | Route via owning view's display pointer |
| `QVTKWidgetCustom.h` | 4 | `HotZone` type(3), `ClickableItem`(1) | Extract types to separate header |
| `ecvGLView.h` | 6 | `HotZone` type(3), `ClickableItem`(1), `MessageToDisplay`(2) | Extract types to separate header |

**Sub-tasks:**

| Sub-task | Description | Est. Lines |
|----------|-------------|-----------|
| P7.3.1 | Extract `HotZone`, `ClickableItem`, `MessageToDisplay` types from `ecvDisplayTools.h` into `ecvDisplayTypes.h` | ~80 |
| P7.3.2 | Give `VtkVis` a back-pointer `m_ownerView` (ecvGLView*), replace singleton calls | ~30 |
| P7.3.3 | Refactor `EditCameraTool` + `VtkMeasurementTools` to accept visualizer via constructor/setter | ~20 |
| P7.3.4 | Refactor `CustomVtkCaptionWidget` to use owning view's display pointer | ~10 |
| P7.3.5 | Convert `ecvGLView` delegation calls: replace `ecvDisplayTools::TheInstance()->method()` with direct implementation on `ecvGLView` (using `m_visualizer3D` directly) | ~150 |
| P7.3.6 | Compile & verify | — |

**Architecture after P7.3:**
```
ecvGLView
├── m_visualizer3D (VtkVis) ← already owned
├── m_vtkWidget (QVTKWidgetCustom) ← already owned
├── m_hotZone, m_clickableItems, m_messagesToDisplay ← already per-view
├── implements all ecvGenericGLDisplay virtuals DIRECTLY (no more TheInstance delegation)
└── VtkDisplayTools::ScopedHotZoneRender → removed (each view renders its own)
```

### TODO 3: Phase 7.1 — Singleton Lifecycle Removal ✅

**Priority**: MEDIUM | **Complexity**: VERY HIGH | **Est. Lines**: ~500 | **Prerequisite**: Phase 2.6 + Phase 7.3 | **Status**: COMPLETE

Remove `Init()`, `ReleaseInstance()`, `TheInstance()`, `HasInstance()` and the singleton instance `s_tools`.

**Detailed call inventory (lifecycle calls):**

| File | Calls | Type | Migration Strategy |
|------|-------|------|--------------------|
| `MainWindow.cpp:654` | 1 | `ReleaseInstance()` | Remove — view cleanup handled by ecvViewManager |
| `MainWindow.cpp:688` | 1 | `Init(new VtkDisplayTools, ...)` | Create first `ecvGLView` directly, register with ecvViewManager |
| `MainWindow.cpp:719-725` | 3 | `TheInstance()` (register/connect) | Use the new first ecvGLView |
| `MainWindow.cpp:778-890` | 5 | `TheInstance()` (identity comparisons) | Compare with `ecvViewManager::getPrimaryView()` |
| `MainWindow.cpp:2546-2548` | 2 | `TheInstance()` (fallback) | `ecvViewManager::getEffectiveView()` |
| `MainWindow.cpp:2583-2637` | 2 | `TheInstance()` (find active) | `ecvViewManager::getEffectiveView()` |
| `MainWindow.cpp:2751-2866` | 4 | `TheInstance()` (tab close logic) | `ecvViewManager::getPrimaryView()` |
| `MainWindow.cpp:5432` | 1 | `TheInstance()` (VTK target fallback) | `ecvViewManager::getEffectiveView()` |
| `MainWindow.cpp:6790` | 1 | `TheInstance()` (pickView fallback) | `ecvViewManager::getEffectiveView()` |
| `ecvGLView.cpp:84,189` | 2 | `TheInstance()` (constructor) | Remove singleton dependency; use ecvViewManager |
| `ecvViewManager.cpp:26` | 1 | `TheInstance()` (setupSingletonRelay) | Refactor: relay connects in registerView() per-view |
| `ecvViewManager.cpp:250,378` | 2 | `TheInstance()` (getPrimaryView) | New `m_primaryView` member |
| `ecvViewManager.cpp:293` | 1 | `SetRemoveViewIDs` | Route via per-view or move to ecvViewManager |
| `ecvRedrawScope.h:34-53` | 6 | `HasInstance()`, `SetRedrawRecursive`, `RedrawDisplay` | Route via `ecvViewManager::hasAnyView()` + `ecvViewManager::setRedrawRecursive()` |
| `ecvCameraParamEditDlg.cpp:308-309` | 2 | `TheInstance()` + `destroyed` signal | Connect to `ecvViewManager::activeViewChanged` |
| `CustomVtkCaptionWidget.cpp:83,103` | 2 | `TheInstance()`, `HasInstance()` | Route via owning view |
| `ecvDBRoot.cpp:420-1124` | 5 | `SetRemoveAllFlag`, `SetRemoveViewIDs`, `SetRedrawRecursive` | Move batch operations to ecvViewManager |

**Sub-tasks:**

| Sub-task | Description | Est. Lines |
|----------|-------------|-----------|
| P7.1.1 | Add `ecvViewManager::getPrimaryView()` — returns first registered view | ~10 |
| P7.1.2 | Add `ecvViewManager::hasAnyView()` — replaces `HasInstance()` | ~5 |
| P7.1.3 | Move batch operations (`SetRemoveAllFlag`, `SetRemoveViewIDs`, `SetRedrawRecursive`) to `ecvViewManager` | ~40 |
| P7.1.4 | Refactor `MainWindow::Init()` — create first `ecvGLView` without `ecvDisplayTools::Init()` | ~80 |
| P7.1.5 | Refactor `MainWindow` — replace all `TheInstance()` with `ecvViewManager` calls | ~50 |
| P7.1.6 | Refactor `ecvGLView` constructor — remove `TheInstance()` dependency | ~20 |
| P7.1.7 | Refactor `ecvViewManager::setupSingletonRelay()` → per-view signal connections in `registerView()` | ~40 |
| P7.1.8 | Refactor `ecvRedrawScope` — use `ecvViewManager` instead of `ecvDisplayTools` | ~15 |
| P7.1.9 | Refactor `ecvCameraParamEditDlg` — connect to ecvViewManager | ~10 |
| P7.1.10 | Remove `s_tools`, `Init()`, `ReleaseInstance()`, `TheInstance()`, `HasInstance()` from `ecvDisplayTools.h/cpp` | ~50 |
| P7.1.11 | Convert `ecvDisplayTools` from singleton to utility class (static-only methods, or pure namespace) | ~100 |
| P7.1.12 | Compile & verify | — |

**Critical risk: This phase changes initialization order.** MainWindow currently creates the singleton, then creates ecvGLView which attaches to it. After this phase, MainWindow creates ecvGLView directly, which self-registers with ecvViewManager. This requires careful testing of the startup sequence.

### TODO 4: Phase 7.2 — Static Utility Extraction ✅

**Priority**: LOW | **Complexity**: LOW | **Est. Lines**: ~100 | **Prerequisite**: Phase 7.1

Convert remaining static utility methods into free functions or `ecvViewManager` methods.

**Inventory:**

| Method | Calls | Current Location | Target Location |
|--------|-------|-----------------|-----------------|
| `ConvertToEntityType(CV_CLASS_ENUM)` | 3 (ecvHObject.cpp) | `ecvDisplayTools` static | `ecvEntityTypeUtils::convert()` free function or `ecvHObject` method |
| `GetOptimizedFontSize(int)` | 3 (ecvGuiParameters.cpp) | `ecvDisplayTools` static | `ecvGuiParameters::optimizedFontSize()` static |
| `SetupProjectiveViewport(...)` | 1 (ecvGBLSensor.cpp) | `ecvDisplayTools` static | `ecvViewManager::setupProjectiveViewport()` or free function |
| `USE_2D` | 1 (ecvPointPropertiesDlg.cpp) | `ecvDisplayTools` static bool | `ecvViewConfig::USE_2D` or keep as static constant |
| `USE_VTK_PICK` | 1 (QVTKWidgetCustom.cpp) | `ecvDisplayTools` static bool | Same |
| `SetRedrawRecursive(ccHObject*, bool)` | 0 (moved) | `ecvDisplayTools` static | `ecvViewManager` |
| `RedrawObject/RedrawObjects` | 1 (qSRA.cpp) | `ecvDisplayTools` static | Already have virtual; route via `ecvViewManager::getEffectiveView()->redrawObject()` |

**Sub-tasks:**

| Sub-task | Description | Est. Lines |
|----------|-------------|-----------|
| P7.2.1 | Create `ecvEntityTypeUtils.h` with `ConvertToEntityType()` or move into `ecvHObject` | ~20 |
| P7.2.2 | Move `GetOptimizedFontSize` into `ecvGuiParameters` | ~15 |
| P7.2.3 | Move `SetupProjectiveViewport` into `ecvViewManager` or free function | ~10 |
| P7.2.4 | Move `USE_2D`/`USE_VTK_PICK` to `ecvViewConfig` or keep as namespace-scoped constants | ~10 |
| P7.2.5 | Migrate `qSRA.cpp`'s `RedrawObject` to per-view virtual | ~5 |
| P7.2.6 | Compile & verify | — |

### TODO 5: MainWindow.cpp VTK-Specific Calls ✅

**Priority**: MEDIUM | **Complexity**: MEDIUM | **Est. Lines**: ~100 | **Prerequisite**: Phase 7.1

Migrate the remaining ~10 VTK-specific `ecvDisplayTools::GetVisualizer3D()` calls in MainWindow.

**Detailed call inventory:**

| Line | Call | Context | Migration |
|------|------|---------|-----------|
| 2664 | `GetVisualizer3D()` | `doActionToggleActiveSF` | `dynamic_cast<ecvGLView*>(activeView)->getVisualizer3D()` |
| 2697 | `GetVisualizer3D()` | `doActionSetActiveSF` | Same pattern |
| 3808 | `GetVisualizer3D()` | `doActionAddNewEntity` | Same |
| 3848 | `GetMainWindow()` | `doSaveViewState` | `ecvViewManager::getMainWindow()` |
| 3915 | `RenderToFile(...)` | `doActionSaveScreenshot` | Already have virtual: `activeView->renderToFile()` |
| 4702 | `GetVisualizer3D()` | `doStartFilterTool` | Accept as parameter from active view |
| 6195 | `GetVisualizer3D()` | `doActionZoomBestFit` | Same pattern |
| 6352 | `GetVisualizer3D()` | `doActionCreateNewSensor` | Same |
| 6493 | `GetVisualizer3D()` | `doActionShowFrustrum` | Same |
| 12049 | `GetVisualizer3D()` | `doActionToggleVTKScalar` | Same |
| 12187 | `GetVisualizer3D()` | `doStartFiltering` | Accept as parameter |
| 12285 | `GetVisualizer3D()` | `doStartFiltering` | Accept as parameter |

**Strategy**: Add a convenience method `MainWindow::getActiveVisualizer3D()` that returns `dynamic_cast<ecvGLView*>(ecvViewManager::getActiveView())->getVisualizer3D()`.

### TODO 6: Phase 7c — Python Wrapper Migration ✅ (Phase M5 完成)

**Priority**: LOW | **Complexity**: MEDIUM | **Est. Lines**: ~200 | **Prerequisite**: Phase 7.1 (singleton removal must be done first)

Update `ccDisplayTools.cpp` Python bindings from static singleton API to per-view API.

**Detailed inventory (117 calls in `ccDisplayTools.cpp`):**

| Category | Count | Examples | Migration |
|----------|-------|---------|-----------|
| **Enum bindings** | ~70 | `ecvDisplayTools::PICKING_MODE::NO_PICKING`, `INTERACT_ROTATE`, `MessagePosition::*`, `PivotVisibility::*` | Move enums to `ecvGenericGLDisplay` (already done) — update Python enum source |
| **Static method bindings** | ~47 | `GetDevicePixelRatio`, `DoResize`, `SetSceneDB`, `GetSceneDB`, `RenderText`, `GetScreenSize`, `GetGLCameraParameters`, `DisplayNewMessage`, `SetPivotVisibility`, `SetPivotPoint`, `SetCameraPos`, `MoveCamera`, `SetPerspectiveState`, `SetView`, `SetInteractionMode`, `SetPickingMode`, `GetContext`, `SetPointSize`, `SetLineWidth`, `AddToOwnDB`, `RemoveFromOwnDB`, `SetViewportParameters`, `SetFov`, `RenderToFile`, `ComputeActualPixelSize`, `RedrawDisplay`, `RefreshDisplay`, `InvalidateViewport`, `Deprecate3DLayer`, `DisplayText`, `Display3DLabel`, `Remove3DLabel`, `RemoveAllWidgets`, `GetViewportParameters`, `SetupProjectiveViewport`, etc. | Wrap as `activeView->method()` via `ecvViewManager::getActiveView()` |

**Sub-tasks:**

| Sub-task | Description | Est. Lines |
|----------|-------------|-----------|
| P7c.1 | Replace enum source from `ecvDisplayTools::ENUM` to `ecvGenericGLDisplay::ENUM` (70 lines, mechanical) | ~70 |
| P7c.2 | Replace static method wrappers with `ecvViewManager::getActiveView()->method()` pattern | ~80 |
| P7c.3 | Add Python-facing `getActiveView()` function that returns the current active display | ~15 |
| P7c.4 | Handle methods with no per-view equivalent (GetMainWindow, GetSceneDB — global) → route via ecvViewManager | ~15 |
| P7c.5 | Update Python test scripts if any | ~20 |
| P7c.6 | Compile & verify | — |

**Note**: The Python wrapper can remain backward-compatible by using `ecvViewManager::getActiveView()` as the implicit target for all calls. Existing Python scripts will continue to work as before (they always target the active view anyway).

### TODO 7: ecvDBRoot.cpp Batch Operations ✅

**Priority**: LOW | **Complexity**: LOW | **Est. Lines**: ~30 | **Prerequisite**: Phase 7.1

Migrate the 5 remaining batch operation calls in ecvDBRoot.cpp.

| Line | Call | Migration |
|------|------|---------
| 420 | `SetRemoveAllFlag(true)` | `ecvViewManager::instance().setRemoveAllFlag(true)` |
| 687 | `SetRemoveViewIDs(toBeDeletedInfos)` | `ecvViewManager::instance().setRemoveViewIDs(...)` |
| 688 | `SetRedrawRecursive(false)` | `ecvViewManager::instance().setRedrawRecursive(false)` |
| 1003 | `SetRedrawRecursive(false)` | Same |
| 1124 | `SetRedrawRecursive(false)` | Same |

### TODO 8: Header Include Cleanup ✅

**Priority**: LOW | **Complexity**: LOW | **Est. Lines**: ~50 | **Prerequisite**: All above phases complete

Remove `#include "ecvDisplayTools.h"` from ~30 files that include it but have zero `ecvDisplayTools::` calls remaining.

**Files to clean (include but no substantive calls):**

| File | Status |
|------|--------|
| `app/ecvAnimationParamDlg.cpp` | 0 calls — remove include |
| `app/reconstruction/ModelViewerWidget.cpp` | 0 calls — remove include |
| `app/ecvGraphicalSegmentationTool.cpp` | 0 calls — remove include |
| `app/ecvMeasurementTool.cpp` | 0 calls — verify & remove |
| `app/ecvMeasurementTool.h` | 0 calls — verify & remove |
| `app/db_tree/ecvPropertiesTreeDelegate.cpp` | 0 calls — remove include |
| `app/ecvPrimitiveFactoryDlg.cpp` | 0 calls — verify & remove |
| `app/ecvAnnotationsTool.h` | 0 calls — verify & remove |
| `app/ecvRegistrationDlg.cpp` | 0 calls — verify & remove |
| `app/ecvComparisonDlg.cpp` | 0 calls — verify & remove |
| `app/ecvOrderChoiceDlg.cpp` | 0 calls — verify & remove |
| `app/reconstruction/DenseReconstructionWidget.cpp` | 0 calls — verify & remove |
| `app/ecvFilterTool.h` | 0 calls — verify & remove |
| `app/pluginManager/ecvPluginUIManager.cpp` | 0 calls — verify & remove |
| `libs/CV_db/src/ecv2DLabel.cpp` | 0 calls — verify & remove |
| `libs/CV_db/src/ecvDrawableObject.cpp` | Only comments — remove include |
| `libs/CV_db/src/ecvGenericPointCloud.cpp` | Only comments — remove include |
| `libs/CVAppCommon/src/ecvApplicationBase.cpp` | 0 calls — verify & remove |
| `libs/CVPluginAPI/src/ecvOverlayDialog.cpp` | 0 calls — verify & remove |
| `libs/VtkEngine/Converters/Cc2Vtk.cpp` | 0 calls — remove include |
| `libs/VtkEngine/Tools/FilterTools/cvGenericFilter.cpp` | 0 calls — remove include |
| `libs/VtkEngine/Tools/FilterTools/VtkFiltersTool.cpp` | 0 calls — verify & remove |
| `libs/VtkEngine/Tools/SelectionTools/cvSelectionPropertiesWidget.cpp` | 0 calls — verify & remove |
| `libs/VtkEngine/Tools/MeasurementTools/cvGenericMeasurementTool.cpp` | 0 calls — verify & remove |
| `plugins/core/Standard/qAnimation/src/qAnimationDlg.cpp` | 0 calls — verify & remove |
| `plugins/core/Standard/qCompass/src/ccMouseCircle.cpp` | 0 calls — verify & remove |
| `plugins/core/Standard/qCanupo/src/qCanupo2DViewDialog.cpp` | Only comments — remove include |
| `plugins/core/Standard/qFacets/src/stereogramDlg.cpp` | 0 calls — verify & remove |
| `plugins/core/Standard/qFacets/src/qFacets.cpp` | 0 calls — verify & remove |
| `plugins/core/Standard/qPCL/.../FastGlobalRegistrationDlg.cpp` | 0 calls — verify & remove |
| `plugins/core/Standard/qPCL/.../MinimumCutSegmentationDlg.cpp` | 0 calls — verify & remove |
| `plugins/core/Standard/qPythonRuntime/src/Runtime/ccGuiPythonInstance.cpp` | 0 calls — verify & remove |
| `plugins/core/Standard/qMPlane/tests/mocks/ccMainAppInterfaceMock.h` | 0 calls — verify & remove |

### TODO 9: Documentation Update ✅

**Priority**: LOW | **Complexity**: LOW | **Est. Lines**: ~100 | **Prerequisite**: All above phases complete

| Sub-task | Description |
|----------|-------------|
| P9.1 | Update `multi-window-views.md` to reflect final architecture (no singleton) |
| P9.2 | Update this migration plan with final status |
| P9.3 | Add developer guide: "How to add a new view-specific feature" |
| P9.4 | Remove stale code examples referencing `ecvDisplayTools::TheInstance()` from docs |

---

## Recommended Execution Order

```
Phase 2.6: QVTKWidgetCustom    ──┐
                                  ├──► Phase 7.3: VtkEngine Internal ──► Phase 7.1: Singleton Removal
Phase 5 (MainWindow VTK calls) ──┘                                              │
                                                                                 ▼
                                                                    Phase 7.2: Utility Extraction
                                                                                 │
                                                        ┌────────────────────────┼────────────────────┐
                                                        ▼                        ▼                    ▼
                                                Phase 7c: Python         TODO 7: ecvDBRoot     TODO 8: Include Cleanup
                                                                                                      │
                                                                                                      ▼
                                                                                               TODO 9: Documentation
```

**Why this order:**
1. **Phase 2.6** (QVTKWidgetCustom) must come first because it's the largest single-file migration and establishes the pattern for VtkEngine
2. **Phase 7.3** (VtkEngine internal) depends on 2.6 because QVTKWidgetCustom is part of VtkEngine
3. **Phase 7.1** (singleton removal) depends on 7.3 because ecvGLView must be self-contained before we remove TheInstance()
4. Everything else can be parallelized after 7.1

**All TODOs (1–9) complete; TODO 6 → Phase M5 complete.** Phase M (M1-M6) all complete — see `multi-window-refactor-roadmap-Vtk-vs-CC.md` §10. Include cleanup (3 files) and s_tools→ residual analysis (473 refs) documented in roadmap §Include 清理进展.

---

## Detailed Refactoring Architecture

### Final Target: ecvDisplayTools After All Phases

```cpp
// ecvDisplayTools.h — FINAL STATE (after all phases)
// Option A: Pure utility namespace (no class)
namespace ecvDisplayUtils {
    ENTITY_TYPE convertToEntityType(CV_CLASS_ENUM type);
    int optimizedFontSize(int baseSize);
    // ... other pure utilities with no state
}

// Option B: Minimal static utility class (no singleton, no state)
class ecvDisplayTools {
public:
    static constexpr bool USE_2D = true;
    static constexpr bool USE_VTK_PICK = false;
    static ENTITY_TYPE ConvertToEntityType(CV_CLASS_ENUM type);
    static int GetOptimizedFontSize(int baseSize);
    // NO Init(), NO ReleaseInstance(), NO TheInstance()
    // NO member variables, NO singleton pattern
};
```

### ecvGLView After All Phases

```cpp
// ecvGLView — FINAL STATE (fully self-contained per-view)
class ecvGLView : public ecvGenericGLDisplay {
    Q_OBJECT
public:
    // === State (all per-view) ===
    ecvViewContext m_ctx;
    VtkVis* m_visualizer3D;        // owned
    QVTKWidgetCustom* m_vtkWidget; // owned
    HotZone* m_hotZone;
    std::vector<ClickableItem> m_clickableItems;
    std::list<MessageToDisplay> m_messagesToDisplay;
    ccHObject* m_globalDBRoot;
    ccHObject* m_winDBRoot;

    // === All ecvGenericGLDisplay virtuals implemented DIRECTLY ===
    // (no more TheInstance() delegation)
    void redraw(bool only2D, bool forceRedraw) override;
    void invalidateViewport() override;
    void deprecate3DLayer() override;
    void displayNewMessage(...) override;
    void removeEntities(CC_DRAW_CONTEXT&) override;
    void hideShowEntities(CC_DRAW_CONTEXT&) override;
    // ... 40+ more virtuals, all implemented directly

signals:
    // === Per-view signals ===
    void itemPicked(...);
    void mouseMoved(...);
    void cameraParamChanged();
    // ... 20+ per-view signals
};
```

### ecvViewManager After All Phases

```cpp
// ecvViewManager — FINAL STATE (singleton for view registry + active tracking only)
class ecvViewManager : public QObject {
    Q_OBJECT
public:
    static ecvViewManager& instance();

    // === View Registry ===
    void registerView(ecvGLView* view);
    void unregisterView(ecvGLView* view);
    ecvGLView* getActiveView() const;
    ecvGLView* getPrimaryView() const;    // first registered view
    ecvGLView* getEffectiveView() const;  // active or primary fallback
    QList<ecvGLView*> allViews() const;
    bool hasAnyView() const;

    // === Global Operations (affect all views) ===
    void setRemoveAllFlag(bool flag);
    void setRemoveViewIDs(std::vector<removeInfo>&);
    void setRedrawRecursive(bool state);
    void refreshAllViews();

    // === Global Utilities (moved from ecvDisplayTools) ===
    QMainWindow* getMainWindow() const;
    void setupProjectiveViewport(...);

signals:
    // === Global signals ===
    void activeViewChanged(ecvGenericGLDisplay*);
    void entitySelectionChanged(ccHObject*);
    void entitiesSelectionChanged(std::unordered_set<int>);

    // === Relay signals (from active view) ===
    void newLabel(ccHObject*);
    void filesDropped(QStringList);
    void cameraParamChanged();
    void itemPicked(...);
    void mouseMoved(...);
    void leftButtonClicked(int, int);
    void rightButtonClicked(int, int);
    void buttonReleased();
    void perspectiveStateChanged();
    void pivotPointChanged(CCVector3d);
    // ... auto-reconnected when active view changes
};
```

### QVTKWidgetCustom After All Phases

```cpp
// QVTKWidgetCustom — FINAL STATE (no m_tools, only m_ownerView)
class QVTKWidgetCustom : public QVTKOpenGLNativeWidget {
    ecvGLView* m_ownerView;  // always set (no primary window special case)
    // ecvDisplayTools* m_tools; ← REMOVED

    // All methods route through m_ownerView:
    // m_ownerView->invalidateViewport();
    // m_ownerView->startPicking(params);
    // m_ownerView->moveCamera(dx, dy, dz);
    // etc.
};
```

### Developer Guide: Adding New View-Specific Features

After the singleton removal, follow this pattern:

**Step 1**: Declare virtual method on `ecvGenericGLDisplay` (default no-op)
**Step 2**: Implement in `ecvGLView` using per-view state (`m_ctx`, `m_visualizer3D`)
**Step 3**: Access from consumer code via `ecvViewManager::instance().getEffectiveView()->method()`

**What NOT to do**:
- Do NOT add new static methods to `ecvDisplayTools` (frozen for legacy compatibility)
- Do NOT use `ecvDisplayTools::sharedTools()` in new code — use `ecvViewManager` instead
- Do NOT store global state in `ecvDisplayTools` — use `ecvViewContext` for per-view state

---

## Estimated Effort (Remaining)

| Phase | Est. Lines Changed | Complexity | Files Touched |
|-------|-------------------|------------|---------------|
| Phase 2.6 (QVTKWidgetCustom) | ~400 | High | 5 (ecvGenericGLDisplay.h, ecvDisplayTools.h/cpp, ecvGLView.h/cpp, QVTKWidgetCustom.h/cpp) |
| Phase 7.3 (VtkEngine internal) | ~300 | High | 8 (VtkVis, VtkDisplayTools, ecvGLView, EditCameraTool, VtkMeasurementTools, CustomVtkCaptionWidget, new ecvDisplayTypes.h) |
| Phase 7.1 (Singleton removal) | ~500 | Very High | 8 (ecvDisplayTools.h/cpp, MainWindow.cpp, ecvGLView.cpp, ecvViewManager.h/cpp, ecvRedrawScope.h, ecvCameraParamEditDlg.cpp) |
| Phase 7.2 (Utility extraction) | ~100 | Low | 5 (ecvHObject, ecvGuiParameters, ecvGBLSensor, new ecvEntityTypeUtils.h) |
| TODO 5 (MainWindow VTK) | ~100 | Medium | 1 (MainWindow.cpp) |
| Phase 7c (Python wrapper) | ~200 | Medium | 1 (ccDisplayTools.cpp) |
| TODO 7 (ecvDBRoot) | ~30 | Low | 1 (ecvDBRoot.cpp) |
| TODO 8 (Include cleanup) | ~50 | Low | ~30 files (remove #include lines) |
| TODO 9 (Documentation) | ~100 | Low | 2 (multi-window-views.md, this file) |
| **Total Remaining** | **~1780** | | **~55 files** |

---

## Previously Completed Phases (Archive)

<details>
<summary>Click to expand batch history</summary>

### Batch 4.1: MainWindow.cpp ✅

| Metric | Value |
|--------|-------|
| Before | 91 |
| After | 59 |
| Migrated | 32 (SetPerspectiveState, GetViewportParameters, UpdateConstellationCenterAndZoom, RedrawDisplay, GetPerspectiveState, SetInteractionMode, RemoveFromOwnDB, GetContext, InvalidateVisualization, PIVOT enums, Toggle2Dviewer) |
| Remaining | VTK-specific (GetVisualizer3D, ZoomGlobal, RemoveBB, etc.), singleton lifecycle (TheInstance, HasInstance), signal connects |

### Batch 4.2: QVTKWidgetCustom.cpp ✅

| Metric | Value |
|--------|-------|
| Before | 112 |
| After | 73 |
| Migrated | 39 (38 enum→ecvGenericGLDisplay + per-view routes for GlWidth/Height, GetViewportParameters, GetDevicePixelRatio, GetGLCameraParameters, AddToOwnDB, RemoveFromOwnDB via m_ownerView) |
| Remaining | VTK internals (draw/widget/picking pipeline), singleton identity |

### Batch 4.3: High-Traffic App Tools ✅

| File | Before | After | Migrated |
|------|--------|-------|----------|
| `ecvGraphicalSegmentationTool.cpp` | 28 | 7 | 21 |
| `ecvTracePolylineTool.cpp` | 44 | 13 | 31 |
| **Total** | **72** | **20** | **52** |

### Batch 4.4: Medium-Traffic App Dialogs ✅

| File | Before | After | Migrated | Notes |
|------|--------|-------|----------|-------|
| `ecvPointPairRegistrationDlg.cpp` | 55 | 7 | 48 | |
| `ecvPointPropertiesDlg.cpp` | 44 | 7 | 37 | |
| `ecvPropertiesTreeDelegate.cpp` | 40 | 24 | 16 | Heavy VTK residuals |
| `ecvDBRoot.cpp` | 17 | 12 | 5 | |
| `ecvAnimationParamDlg.cpp` | 19 | 3 | 16 | |
| `ecvGraphicalTransformationTool.cpp` | 15 | 0 | 15 | **Fully cleared** |
| `ecvRasterizeTool.cpp` | 22 | 0 | 22 | **Fully cleared** |
| `ecvContourExtractorDlg.cpp/.h` | 18 | 0 | 18 | **Fully cleared** |
| `ecv2.5DimEditor.cpp` | 14 | 0 | 14 | **Fully cleared** |
| `ecvPointListPickingDlg.cpp` | 13 | 0 | 13 | **Fully cleared** |
| `ecvVolumeCalcTool.cpp` | 11 | 0 | 11 | **Fully cleared** |
| `MovieGrabberWidget.cpp` | 11 | 0 | 11 | **Fully cleared** |
| `ModelViewerWidget.cpp` | 25 | 2 | 23 | |
| `ecvEntityAction.cpp` | 7 | 3 | 4 | |
| Small files (11 files) | ~16 | ~5 | ~11 | 6 files fully cleared |

### Batch 4.5: CV_db & Libs ✅

| File | Before | After | Migrated | Notes |
|------|--------|-------|----------|-------|
| `ecvHObject.cpp` | 27 | 22 | 5 | |
| `ecv2DLabel.cpp` | 28 | 21 | 7 | |
| `ecvCameraParamEditDlg.cpp` | 21 | 7 | 14 | |
| `ecvCameraSensor.cpp` | 12 | 5 | 7 | |
| `ecv2DViewportLabel.cpp` | 10 | 6 | 4 | |
| `ecvSphere.cpp` | 7 | 4 | 3 | |
| `ecvCustomViewpointsToolbar.cpp` | 7 | 0 | 7 | **Fully cleared** |
| `ecvPickingHub.cpp/.h` | 9 | 1 | 8 | |
| `GamepadInput.cpp` | 6 | 3 | 3 | |
| `cvRenderViewSelectionReaction.cpp` | 3 | 0 | 3 | **Fully cleared** |
| Other small libs files (10+) | ~40 | ~30 | ~10 | |

### Batch 4.6: Plugin Files ✅

| File | Before | After | Notes |
|------|--------|-------|-------|
| `qSRA/distanceMapGenerationDlg.cpp` | 70 | 1 | Only `RenderToFile` remains |
| `qCanupo/qCanupo2DViewDialog.cpp` | 41 | 4 | `ToVtkCoordinates` + comments |
| `qCompass/ccCompass.cpp` | 19 | 0 | **Fully cleared** |
| `qCompass/ccMouseCircle.cpp` | 12 | 8 | VTK widget draw/remove |
| `qCompass/ccThicknessTool.cpp` | 9 | 0 | **Fully cleared** |
| `qCompass/ccFitPlaneTool.cpp` | 4 | 0 | **Fully cleared** |
| `qCompass/ccTopologyTool.cpp` | 4 | 0 | **Fully cleared** |
| `qCompass/ccTraceTool.cpp` | 2 | 0 | **Fully cleared** |
| `qCloudLayers/ccCloudLayersDlg.cpp` | 12 | 2 | Mouse connect residual |
| `qAnimation/qAnimationDlg.cpp` | 10 | 1 | `RenderToImage` fallback |
| `qSRA/ccSymbolCloud.cpp` | 11 | 2 | `DisplayText` + font |
| Other plugins (20+ files) | ~38 | ~18 | 12 files fully cleared |

</details>

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
| 2026-04-29 | 2.0 | Added detailed Phase 4 batch breakdown (7 batches, ~1000+ call sites across 90+ files) |
| 2026-04-30 | 3.0 | Batch 4.1–4.6 complete: app/ 450+→142, libs/ 350+→265, plugins/ 232→36, 37 files fully cleared. Category C residual analysis done. |
| 2026-04-30 | 3.1 | Phase 7 detailed design: 20 new per-view virtual APIs, singleton elimination plan, Python wrapper merge |
| 2026-04-30 | 4.0 | Phase 7a wave 1: 20+ virtual methods on ecvGenericGLDisplay. Total external ~410 (from ~443). |
| 2026-04-30 | 5.0 | Phase 7a wave 2 + 7b migration: 20 additional virtuals. app/ 128→47, libs/CV_db 61→39, CVAppCommon 14→7. 11+ more files fully cleared. |
| 2026-04-30 | 6.0 | P7.9 signal migration + final migratable sweep: 9 relay signals on ecvViewManager. All consumer connect()s migrated. Final: app/ 37, CV_db 47(infra), CVAppCommon 2, plugins 3. Non-VtkEngine migration complete. |
| 2026-04-30 | 7.0 | Comprehensive TODO docs: 9 detailed TODOs with per-file inventories, sub-tasks, execution order graph, final architecture diagrams. |
| 2026-04-30 | 8.0 | Phase 2.6 complete: QVTKWidgetCustom.cpp migrated 73→6. 18 new virtuals on ecvGenericGLDisplay. Overrides in ecvDisplayTools+ecvGLView. OwnerRenderContext removed. |
| 2026-04-30 | 9.0 | P7.1-prep complete: ecvViewManager gains getPrimaryView()/hasAnyView()/setRemoveAllFlag()/setRedrawRecursive(). ecvRedrawScope.h migrated. ecvDBRoot.cpp batch ops migrated. ecvCameraParamEditDlg + CustomVtkCaptionWidget migrated. MainWindow.cpp 8 identity+fallback patterns migrated. Transitive include fixes. |
| 2026-04-30 | 10.0 | Phase 7.3 (VtkEngine Internal) mostly complete: VtkVis back-pointer (9→0 calls), EditCameraTool (2→0), VtkMeasurementTools (2→0) fully cleared. ecvGLView.cpp TheInstance() reduced from 57→1 (init only); all 39 TheInstance-based delegations replaced with m_displayTools stored pointer (9 VTK overrides) or direct static calls (30 methods). VtkDisplayTools.cpp 18 refs are internal (subclass). Type refs (HotZone etc.) deferred. |
| 2026-04-30 | 11.0 | **Phase 7.1 (TODO 3) complete**: Singleton lifecycle removed. `s_tools` changed from `ecvSingleton<>` to raw ptr. `Init()`/`TheInstance()`/`HasInstance()`/`ReleaseInstance()` removed from public API; replaced by `initializeSharedInstance`/`releaseSharedInstance`/`sharedTools` (friend of ecvViewManager). ecvViewManager gains `initDisplayTools()`/`releaseDisplayTools()`/`displayTools()`. MainWindow.cpp: 8 TheInstance() → ecvViewManager accessors. ecvGLView.cpp: TheInstance() → `ecvViewManager::instance().displayTools()`. PCL plugin cc2sm.cpp: 1 VtkDisplayTools::TheInstance() → ecvViewManager. Header: 121 TheInstance/HasInstance → sharedTools(). Internal: 582 s_tools.instance → s_tools + 17 TheInstance/HasInstance → sharedTools. Build clean (0 errors). |
| 2026-04-30 | 12.0 | **Phase 7.3 Category A complete**: 11 ctx-only static delegations in ecvGLView.cpp replaced with direct `m_ctx` access. Methods: `toVtkCoordinates`, `getCurrentViewDir`, `setAutoPickPivotAtCenter`, `isRotationAxisLocked`, `lockRotationAxis`, `toggleDebugTrace`, `setViewportDefaultPointSize`, `setViewportDefaultLineWidth`, `computeActualPixelSize`, `showPivotSymbol`, `setExclusiveFullScreenFlag`. Each now reads/writes per-view `m_ctx` instead of delegating through `ecvDisplayTools` singleton static methods. Build clean (0 errors). |
| 2026-04-30 | 13.0 | **Phase 7.3 Category B + C + type extraction**: (1) `m_displayTools` type changed from `ecvDisplayTools*` to `VtkDisplayTools*` — eliminates base-class indirection, removes redundant `static_cast`. (2) `toggle2Dviewer()` now calls `m_visualizer3D->setInteractionMode()` directly. (3) New `ecvDisplayTypes.h` + `ecvDisplayTypes.cpp`: extracted 5 nested types (`MessageToDisplay`, `ProjectionMetrics`, `HotZone`, `ClickableItem`, `PickingParameters`) from `ecvDisplayTools.h` into standalone structs (`ecvMessageToDisplay`, `ecvProjectionMetrics`, `ecvHotZone`, `ecvClickableItem`, `ecvPickingParameters`). Backward-compatible `using` aliases in `ecvDisplayTools`. ~175 lines removed from header. (4) Full remaining-call analysis: 33 `ecvDisplayTools::` refs in ecvGLView.cpp categorized (10 VTK-camera, 6 VTK-render, 6 lighting/grid, 2 widgets, 2 picking, 4 entity/DB, 1 utility, 2 type-refs). Build clean (0 errors). |
| 2026-04-30 | 14.0 | **TODO 2+5+7 complete**: (1) **ecvGLView.cpp 33→1 `ecvDisplayTools::`**: All 32 static delegations replaced with `m_displayTools->virtualMethod()` calls. Only `GetContext(ctx, m_ctx)` (stateless utility) remains. Covers VTK-camera(10), VTK-render(6), lighting/grid(6), widgets(2), picking(2), entity/DB(4), type-refs(2). (2) **MainWindow.cpp 0 `ecvDisplayTools::`**: 10 `GetVisualizer3D()` → new `getActiveVisualizer3D()` helper via ecvViewManager. 1 `GetMainWindow()` → `this`. 1 `RenderToFile` → per-view virtual. (3) **TODO 7**: ecvDBRoot.cpp already migrated in prior session. Build clean (0 errors). |
|| 2026-04-30 | 15.0 | **TODO 4 (P7.2) + TODO 8 (header cleanup)**: (1) Analyzed 7 static utilities: `ConvertToEntityType` (3 calls, ecvHObject.cpp) and `RedrawObject` (1, qSRA.cpp) are extraction candidates; `USE_2D`/`USE_VTK_PICK`/`GetOptimizedFontSize`/`SetupProjectiveViewport` deferred (deeply coupled). (2) **28 unused `#include ecvDisplayTools.h` removed** from app/, libs/, plugins/. 3 files retained (dynamic_cast type), VtkVis.h retained (AxesGridProperties struct). (3) **Residual stats**: Core 303, per-view infra 22, entity 8, app 1, plugins 3, dead-code 9, Python wrapper 117. **Non-core active: ~37 (from ~1000+)**. |
|| 2026-04-30 | 16.0 | **ALL TODOs COMPLETE (1–9)**. Final sweep: (1) `ConvertToEntityType` extracted to local helper in `ecvHObject.cpp` (3→0 calls). `RedrawObject` in qSRA → `m_app->refreshAll()` (1→0). (2) 9 dead-code comments removed (`ecvDrawableObject`, `ecvGenericPointCloud`, `qCanupo`, `QVTKWidgetCustom`). (3) `ecvGLView.h` `HotZone/ClickableItem/MessageToDisplay` type refs → extracted `ecvHotZone/ecvClickableItem/ecvMessageToDisplay` (6→0). `QVTKWidgetCustom.h` same (4→0). `QVTKWidgetCustom.cpp` (6→1, only `USE_VTK_PICK`). (4) Python wrapper: 56 enum refs → `ecvGenericGLDisplay::` (117→67). (5) **Final residual**: 15 files, 383 total refs. Core 303, per-view infra 7, entity 5, app 1, Python wrapper 67. **Non-core active: 13 (from ~1000+)**. |
|| 2026-04-30 | 17.0 | **ecvGenericGLDisplay.cpp fully cleared (5→0)**. Added 5 `ecvViewManager::shared*()` forwarders (`sharedMoveCamera`, `sharedRotateBaseViewMat`, `sharedDisplayText`, `sharedLoadCameraParameters`, `sharedSaveCameraParameters`). `#include "ecvDisplayTools.h"` removed from ecvGenericGLDisplay.cpp. **Final: 14 files, 384 total. Core 309, per-view 2, entity 5, app 1, Python 67. Non-core active: 8 (from ~1000+)**. |
|| 2026-04-30 | 18.0 | **NON-CORE ACTIVE: 0**. All 8 remaining migrated: ecvGLView.cpp GetContext, QVTKWidgetCustom.cpp USE_VTK_PICK, ecvPointPropertiesDlg.cpp USE_2D, ecvGBLSensor.cpp SetupProjectiveViewport, ecvGuiParameters.cpp GetOptimizedFontSize (x3) — all routed through new ecvViewManager forwarders. **Final: 9 files (core infra + Python), 381 total. Zero non-core refs. Migration complete.** |
|| 2026-04-30 | 19.0 | **Next-phase TODOs formulated (M1–M6, v2)**. Core principle: **eliminate Primary/Secondary view distinction** (all views = ecvGLView, like ParaView pqRenderView). VtkDisplayTools → pure engine service. Cross-ref: see `multi-window-refactor-roadmap-Vtk-vs-CC.md` §10. Key phases: M1 (VtkDisplayTools role split), M2 (QVTKWidgetCustom ~90+ m_tools refs migration), M3 (ecvGLView as sole view type), M4 (2D overlay parameterization → ScopedHotZoneRender elimination). Est. 7-9 weeks. |
|| 2026-04-30 | 20.0 | **Phase M1+M2 COMPLETE. Phase M3.1+M3.2 DONE.** (1) M1.4: 7 Category A methods marked `[[deprecated("Phase M3")]]`. (2) M2.3: `onWheelEvent` → `effectiveCtx()`, deferred picking timer connected in ecvGLView, `CustomVtkCaptionWidget` per-view timer stop. (3) **M3.1**: `initializeSharedInstance` no longer registers VtkDisplayTools as view. `MainWindow::initial()` creates `m_firstView = ecvGLView::Create()` as first view. Layout: `assignView(0, m_firstView)`. (4) **M3.2**: `rebindToolsToActiveView` simplified (no `restorePrimaryView` fallback). `onViewClosingFromLayout` / `prepareViewClose` simplified (no `adoptNewPrimary`/`resetToBuiltInPipeline`). Build clean. Remaining: M3.3 (delete Cat A impl), M3.4 (simplify dynamic_cast branches), M4-M6. P4.7, P7.2, P7.6, P7.7 marked ✅. |
|| 2026-04-30 | 21.0 | **Phase M COMPLETE (M1-M6)**. (1) **M3.3**: Deleted `adoptNewPrimary`/`restorePrimaryView`/`resetToBuiltInPipeline`/`getBuiltIn*` + `m_primaryVis`/`m_primaryWidget`/`m_builtInVis`/`m_builtInWidget` (-203 lines). (2) **M3.4**: Simplified `dynamic_cast<ecvGLView*>` in MainWindow, removed VtkDisplayTools fallbacks. (3) **M4**: Parameterized `DrawClickableItems` (explicit HotZone/ClickableItems/Display params). Deleted `ScopedHotZoneRender` (~100 lines). Removed `beginPrimaryRender`/`endPrimaryRender`. `ecvGLView::redraw()` now renders complete overlay (ColorRamp, ScaleBar, Messages, hot zone). `RedrawDisplay` legacy tail (~90 lines) removed. `drawWidgets` WIDGET_T2D/POINTS_2D per-view routing via `ecvGLView::getImageVis()`. Net: -134 lines for M4 alone. (4) **M5**: Python bindings verified intact; duplicate registrations cleaned. (5) **M6**: Audit confirms `ecvRepresentationManager` already per-`(entity,view)`. Deep VTK property propagation deferred. **Total M1-M6: ~600 lines deleted, primary/secondary distinction eliminated, ecvGLView is sole view type, VtkDisplayTools is pure engine service.** |
|| 2026-04-30 | 22.0 | **Post-M cleanup**: (1) Include cleanup: 3 files modified (`ccGuiPythonInstance.cpp` remove include, `VtkVis.h` → `ecvDisplayTypes.h`, `QVTKWidgetCustom.h` → forward decl). (2) `effectiveCtx()` deep audit: 307 occurrences in ecvDisplayTools.cpp. Tier breakdown: ~4 EASY, ~25 MEDIUM, ~45 HARD (per function). Most are embedded in singleton's core camera/viewport/picking API. (3) Stale migration plan rows updated: P1.5, P1.6, P2.6, P2.7, P3.4, P3.5 all marked ✅/superseded. TODO 6 → ✅ (M5 complete). Phase 4 header corrected to "4.1-4.7 ✅". Full build verified. |
|| 2026-04-30 | 23.0 | **Phase N 计划制定**: `effectiveCtx()` 分批参数化迁移，5 阶段 (N1-N5)，覆盖 ~64 函数 / 307 调用。N1 (trivial ~25 func, 1-2 天), N2 (setters ~25 func, 3-5 天), N3 (heavy mutators ~8 func, 1 周), N4 (projection engine 3 func, 1-2 周), N5 (picking 3 func, 1 周)。总预估 4-5 周。详见 `multi-window-refactor-roadmap-Vtk-vs-CC.md` §12. |
|| 2026-05-01 | 24.0 | **Phase N 全部完成 (N1-N5 + straggler cleanup)**. `effectiveCtx()` 从 307→76 次调用（ecvDisplayTools.cpp 内）。剩余 76 次分布：**52 wrapper 委托**（将 `s_tools->effectiveCtx()` 传递给 ctx-parameterized 重载，是最终设计模式）、**14 本地 `auto& ctx` 缓存**（函数入口一次查询后复用）、**3 单次查询**（轻量 accessor）。已清理的函数：`initializeSharedInstance` (32→1 ctx 引用)、`SetPointSize`/`SetLineWidth`/`SetPickingMode`/`GetPickingMode` (消除多次 effectiveCtx 查询)、`SetPivotPoint`/`SetAutoPickPivotAtCenter` (6→1)、`UpdateConstellationCenterAndZoom` (3→1)、`GetGLCameraParameters` (7→1)、`DrawForeground` (8→1)、`DrawBackground` (1→1)、`DrawClickableItems` (10→1)、`DrawWidgets` (1→0, 改用 `GetInteractionMode()`)、`GetContext` (2→1)。**Zero non-wrapper `effectiveCtx()` logic remains。** |
|| 2026-05-01 | 25.0 | **全代码库 effectiveCtx() 审计 + 文档同步**. 全局统计：**107 处** across 4 files — `ecvDisplayTools.cpp` 76 (52 wrapper + 14 cached + 3 accessor + 7 other), `ecvDisplayTools.h` 27 (2 decl + 2 comments + 23 inline wrappers), `ecvGLView.cpp` 3 (仅注释), `MainWindow.cpp` 1 (新视图初始化). **Zero problematic calls remaining**. 文档同步：4 份设计/路线图文档全部更新 — M1-M5/N1-N5 heading markers → ✅, 53 个验收标准 checkbox → [x], GAP-1/2/3 → RESOLVED, executive summary 更新, 仓库路径更新 (macOS/Linux dual path). 仅 M6 (Per-View 表示完善) 正确保留 🔲. |
|| 2026-05-01 | 26.0 | **Phase M6 + Phase O COMPLETE (Per-View Representation Deep Integration)**. 统一实施 6 个 Task: (1) `ecvViewRepresentation` 新增 `effectiveLineWidth`/`effectiveRenderMode`/`effectiveEdgeVisibility`/`effectiveScalarFieldIndex`/`effectiveShowScalarField`/`effectiveShowColors`/`effectiveShowNormals`/`effectiveNormalScale` 等 8 个 `effective*()` 方法。(2) `ecvRepresentationManager::notifyChanged()` + `representationChanged` 信号发射：`setProperties()`/`setVisible()` 触发。(3) `ccHObject::draw()` 扩展 per-view 属性传播：`pointSize`/`lineWidth`/`renderMode` → `CC_DRAW_CONTEXT`。(4) `ecvGLView::initVtkPipeline()` 连接 `representationChanged` → `redraw()` (per-view filter)。(5) `ecvPropertiesTreeDelegate` 新增 "Display (Per-View Override)" section：`OBJECT_PERVIEW_VISIBILITY`/`OBJECT_PERVIEW_OPACITY`/`OBJECT_PERVIEW_POINT_SIZE` 三个新属性角色。(6) GAP-4/GAP-6 → ✅ RESOLVED, Phase O → ✅ COMPLETE, M6 → ✅. **ParaView `vtkSMRepresentationProxy` 对齐完成。** |
|| 2026-05-01 | 27.0 | **GAP-7 (Per-View Camera Undo/Redo) 确认已实现 ✅**. 代码审计发现 `VtkVis` 已包含完整 camera undo/redo: `CameraParams` struct, `m_cameraUndoStack`/`m_cameraRedoStack` (deque, max 20), `pushCameraState()` on `StartInteractionEvent`, `cameraUndo()`/`cameraRedo()`/`canCameraUndo()`/`canCameraRedo()`. `MainWindow::createViewFrame` 已添加 per-frame toolbar buttons (pqUndoCamera.svg/pqRedoCamera.svg) + 500ms QTimer 轮询。3 处文档矩阵行从 GAP → ALIGNED。**ParaView 对齐率**: 86/91 = **94.5%**（仅剩 3 GAP + 2 PARTIAL，均为低优先级功能范围外项）。`multi-window-views.md` M6 → ✅, 交叉引用更新。 |
