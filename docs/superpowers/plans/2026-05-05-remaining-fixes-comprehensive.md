# Comprehensive Remaining Fixes — Multi-Window Architecture

**Date:** 2026-05-05
**Status:** Active — post session fixes for gray-window and label rendering

---

## Completed Fixes (This Session)

### Fix A: Window Turns Gray After Tab Switch
**Root cause:** `VtkDisplayTools::switchActiveView()` destroyed the old `QVTKWidgetCustom` (hide → unparent → deleteLater), a leftover single-window assumption.

**Changes:**
| File | Change |
|------|--------|
| `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` | Removed destructive widget teardown in `switchActiveView()` |
| `app/MainWindow.cpp` | Added `QTimer::singleShot(0, ...)` redraw in `rebindToolsToActiveView()` to force render on newly active view |

### Fix B: cc2DLabel Caption Not Displayed / Wrong Position
**Root cause:** VTK camera matrices (`projMatd`, `viewMatd`) not synced to `ecvViewContext` before the 2D label projection pass.

**Changes:**
| File | Change |
|------|--------|
| `libs/VtkEngine/Visualization/ecvGLView.cpp` | Call `syncVtkCameraToContext()` before 3D+2D foreground passes in `redraw()` |
| `libs/VtkEngine/Visualization/ecvGLView.cpp` | Defensive sync in `getGLCameraParameters()` when validity flags are false |
| `libs/CV_db/include/ecvViewContext.h` | Added `validModelviewMatrix` / `validProjectionMatrix` tracking flags |

---

## Remaining Fixes — Prioritized

### Priority 1: Critical Multi-Window Bugs

#### Fix C: Static Mouse Orientation Cross-View Trackball Bug
**Source:** Phase 3, Task 1
**Risk:** Dragging in View A then View B reuses A's last orientation → incorrect rotation
**Files:** `QVTKWidgetCustom.cpp`, `QVTKWidgetCustom.h`
**Change:** Move `s_lastMouseOrientation` from `static` local to per-widget member `m_lastMouseOrientation`
**Effort:** Small (< 30 min)

#### Fix D: `removeEntities` Cross-View Fan-Out
**Source:** Phase 3, Task 4
**Risk:** Deleting a 2D overlay in View A removes it from ALL views' VtkVis pipelines
**Files:** `VtkDisplayTools.cpp`
**Change:** Guard propagation to only 3D scene entities; 2D overlays stay per-view
**Effort:** Small (< 30 min)

#### Fix E: `ExclusiveFullScreen()` Targets Wrong View
**Source:** Phase 3, Task 3
**Risk:** Fullscreen toggle in View B may read/write View A's state via `resolveViewContext()`
**Files:** `ecvDisplayTools.h`, `ecvDisplayTools.cpp`
**Change:** Add view-explicit overloads, update `DrawClickableItems` to use them
**Effort:** Small (< 30 min)

### Priority 2: Cleanup & Correctness

#### Fix F: Remove `m_hotZoneOwnedBySingleton` Flag
**Source:** Phase 3, Task 2
**Risk:** Stale ownership semantics; each view already owns its hot zone
**Files:** `ecvDisplayTools.h`, `ecvDisplayTools.cpp`
**Change:** Remove flag declaration, remove usage in `DrawClickableItems` and destructor
**Effort:** Small (< 15 min)

#### Fix G: Migrate `USE_2D` / `USE_VTK_PICK` to Per-View
**Source:** Phase 3, Task 5
**Risk:** Class-level globals prevent per-view 2D/picking configuration
**Files:** `ecvDisplayTools.h`, `ecvDisplayTools.cpp`, `ecvViewContext.h`, `ecvGLView.cpp`
**Change:** Add flags to `ecvViewContext`, initialize from global defaults
**Effort:** Medium (< 1 hr)

#### Fix H: Update Legacy Singleton Comments
**Source:** Phase 3, Task 6
**Risk:** Misleading documentation for future developers
**Files:** `ecvDisplayTools.h`, `ecvGenericGLDisplay.h`, `ecvDisplayTools.cpp`
**Change:** Replace "singleton" references with accurate descriptions
**Effort:** Small (< 15 min)

### Priority 3: Architecture Gaps (Deferred — Future Phases)

| Gap | Description | Call Sites | Effort | Notes |
|-----|-------------|-----------|--------|-------|
| **S** | `primaryDT()` signal emission loses view attribution | ~30+ | Large | Needs per-view signal relay |
| **T** | `GetInstance()` static accessor lacks view param | ~10 | Medium | Add view-parameter overloads |
| **U** | `ScopedRenderOverride` thread safety | 6 | Medium | Replace with explicit context passing |
| **V** | Implicit `resolveViewContext()` | ~115 | Large | Gradual migration to explicit-context APIs |
| **W** | Picking fallback chain | 3 | Small | Replace global fallback with assert |
| **X** | Centralized DB roots | 47 | Very Large | Future per-view visibility model |

---

## Execution Order

```
Phase (session 1):
  [x] Fix A — gray window (DONE)
  [x] Fix B — label rendering (DONE)

Phase (session 2 — Priority 1):
  [x] Fix C — static mouse orientation (already applied)
  [x] Fix D — removeEntities guard (already applied)
  [x] Fix E — ExclusiveFullScreen per-view overloads (DONE)

Phase (session 2 — Priority 2):
  [x] Fix F — remove hotZone singleton flag (already removed)
  [ ] Fix G — USE_2D/USE_VTK_PICK per-view
  [x] Fix H — update comments (DONE)

Phase 4 (future — Priority 3):
  [ ] Gap S — signal attribution
  [ ] Gap T — GetInstance per-view
  [ ] Gap U — ScopedRenderOverride safety
  [ ] Gap V — resolveViewContext migration
  [ ] Gap W — picking fallback
  [ ] Gap X — per-view DB roots
```

---

## Verification Checklist

For each fix, verify:
1. `cd build_app && make -j48` exits 0
2. Open multi-tab layout, switch between tabs — no gray windows
3. Add cc2DLabel in View A — caption renders at correct position
4. Rotate camera in View A, switch to View B — View B trackball independent
5. Delete entity in View A — View B unaffected for 2D overlays
6. Fullscreen toggle in View A — only View A affected

---

## Files Modified (Cumulative from Session)

| File | Status |
|------|--------|
| `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` | Modified — switchActiveView cleanup |
| `libs/VtkEngine/Visualization/ecvGLView.cpp` | Modified — syncVtkCameraToContext in redraw, defensive getGLCameraParameters |
| `libs/CV_db/include/ecvViewContext.h` | Modified — matrix validity flags |
| `app/MainWindow.cpp` | Modified — forced redraw on view activation |
| `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` | Pending — mouse orientation fix |
| `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.h` | Pending — mouse orientation member |
