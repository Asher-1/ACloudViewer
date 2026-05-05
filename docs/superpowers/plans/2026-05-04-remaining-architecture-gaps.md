# Remaining Architecture Gaps — Post-Singleton Removal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document remaining architectural gaps after full singleton removal from ecvDisplayTools

**Architecture:** The singleton (`s_tools`, `sharedTools()`, `effectiveCtx()`, `m_primaryCtx`) has been fully removed. ecvViewManager now owns the ecvDisplayTools instance and manages view lifecycle. Each ecvGLView has its own ecvViewContext. ScopedRenderOverride ensures per-view correctness during rendering.

**Status:** Analysis complete — 2026-05-04. Build verified (make -j48 exits 0).

---

## Priority Matrix

| Gap | Description | Call Sites | Priority | Effort |
|-----|-------------|-----------|----------|--------|
| **S** | `primaryDT()` signal/member access | ~30+ | HIGH | Large |
| **T** | `GetInstance()` static accessor | ~10 in header | MEDIUM | Medium |
| **U** | `ScopedRenderOverride` thread safety | 6 sites | MEDIUM | Medium |
| **V** | Implicit `resolveViewContext()` | ~115 | LOW-MEDIUM | Large |
| **W** | Picking fallback chain | 3 sites | LOW | Small |
| **X** | Centralized DB roots | 47 in DT | LOW | Very Large |

---

## Gap S: `primaryDT()` Soft-Singleton Helper (~30+ call sites)

**File:** `libs/CV_db/src/ecvDisplayTools.cpp:60`

```cpp
static ecvDisplayTools* primaryDT() {
    return ecvViewManager::instance().displayTools();
}
```

Used for:
- **Signal emission** (`emit primaryDT()->cameraParamChanged()`, `emit primaryDT()->entitySelectionChanged(...)`, etc.) — ~15 call sites
- **DB root access** (`primaryDT()->m_globalDBRoot`, `primaryDT()->m_winDBRoot`) — ~12 call sites
- **Member access** (`primaryDT()->m_win`, `primaryDT()->m_captureMode`, `primaryDT()->m_pickingTargetView`)

**ParaView comparison:** In ParaView, signals are emitted from the specific `pqRenderView` being manipulated, not from a shared instance. This gap means camera changes in View B emit signals from the primary display tools instance, losing view-source attribution.

**Fix approach:** Move signal emission to per-view instances or route through ecvViewManager with view identification.

---

## Gap T: `ecvGenericDisplayTools::GetInstance()` Static Accessor

**Files:**
- `libs/CV_db/include/ecvGenericDisplayTools.h:42`
- `libs/CV_db/src/ecvGenericDisplayTools.cpp:15`

```cpp
ecvGenericDisplayTools* ecvGenericDisplayTools::GetInstance() {
    return static_cast<ecvGenericDisplayTools*>(
            ecvViewManager::instance().displayTools());
}
```

Called from static helper methods (`GetPerspectiveState()`, `ToWorldPoint()`, `ToDisplayPoint()`). These methods have no view parameter and always resolve via the global manager.

**Impact:** Any code calling `ecvGenericDisplayTools::ToWorldPoint()` implicitly uses whichever view the manager currently considers "active."

**Fix approach:** Add view-parameter overloads to frequently called static methods.

---

## Gap U: `ScopedRenderOverride` Thread-Unsafe Stack Override

**File:** `libs/CV_db/include/ecvViewManager.h:59`

```cpp
class ScopedRenderOverride {
public:
    explicit ScopedRenderOverride(ecvGenericGLDisplay* view)
        : m_saved(ecvViewManager::instance().m_renderingView) {
        ecvViewManager::instance().m_renderingView = view;
    }
    ~ScopedRenderOverride() {
        ecvViewManager::instance().m_renderingView = m_saved;
    }
};
```

Used in 6 sites: `ecvGLView::redraw()`, `ecvViewManager::refreshAll()/redrawAll()`, `ecvGenericGLDisplay::moveCamera()/rotateBaseViewMat()/load/saveCameraParameters()`.

**Risk:** Global mutable pointer override — not thread-safe and fragile if nested overrides occur.

**Fix approach:** Replace with explicit view-context passing or thread-local storage.

---

## Gap V: Implicit `resolveViewContext()` (~115 call sites)

**Files:** `libs/CV_db/src/ecvDisplayTools.cpp` (~80), `libs/CV_db/include/ecvDisplayTools.h` (~25)

Many static methods have two overloads:
```cpp
static void SetFov(ecvViewContext& ctx, float fov);  // explicit
static void SetFov(float fov);                         // implicit — uses resolveViewContext()
```

The no-arg versions silently depend on the current active/rendering view state.

**Fix approach:** The explicit-context overloads already exist. Gradually migrate callers to use the explicit versions and deprecate no-arg wrappers.

---

## Gap W: `m_pickingTargetView` Fallback Chain

**File:** `libs/CV_db/src/ecvDisplayTools.cpp`

Picking uses a 3-tier fallback:
1. `m_pickingTargetView` (if set)
2. `ecvViewManager::instance().getEffectiveView()`
3. `primaryDT()` (global fallback)

**Fix approach:** Replace step 3 with explicit error/assert.

---

## Gap X: Centralized DB Roots

**File:** `libs/CV_db/src/ecvDisplayTools.cpp` (47 references)

`m_globalDBRoot` / `m_winDBRoot` owned by ecvDisplayTools, shared across all views.

**ParaView comparison:** ParaView supports per-view pipeline visibility via `pqRepresentation`. Shared DB root is correct for multi-camera-view use case, but blocks per-view filtering.

**Fix approach:** Future feature — move to per-view visibility model when needed.

---

## Conclusion

The singleton pattern has been **fully removed**. What remains are:
- **Convenience accessors** (`primaryDT()`, `GetInstance()`) that route through `ecvViewManager`
- **Implicit view resolution** via `ScopedRenderOverride` + `resolveViewContext()`
- **Signal attribution gaps** where events don't carry view-source info

For the current "multi-camera views of the same scene" use case, the architecture is **functionally correct**. Gaps S and T should be addressed if the project needs per-view signal attribution or per-view pipelines in the future.
