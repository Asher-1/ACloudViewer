# Phase 2: ecvDisplayTools Singleton Complete Removal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fully eliminate `s_tools` singleton and `m_primaryCtx` fallback so every render window is truly equivalent — no "primary view" concept remains.

**Architecture:** Replace the `s_tools` singleton with per-view ownership of `ecvDisplayTools` state. Each `ecvGLView` already has `m_ctx` (per-view context); we extend this so all rendering, picking, and UI helpers route through the view-local context instead of `sharedTools()->effectiveCtx()`. Legacy static API callers are migrated incrementally to either accept an explicit `ecvViewContext&` parameter or resolve context from `ecvViewManager::getEffectiveView()`.

**Tech Stack:** C++17, Qt 5/6, VTK, CMake

**Current State (from audit):**
- `TheInstance()` removed; `s_tools` + `sharedTools()` remain
- `effectiveCtx()` resolves via `ecvViewManager::getEffectiveView()` -> `viewContext()`, falling back to `m_primaryCtx`
- ~70+ `effectiveCtx()` call sites in `ecvDisplayTools.cpp`
- ~25+ in `ecvDisplayTools.h` inline methods
- Per-view `ecvGLView::m_ctx` exists and works during `ScopedRenderOverride`
- `QVTKWidgetCustom::curCtx()` falls back to `m_tools->m_primaryCtx` when `m_ownerView` is null
- `CustomVtkCaptionWidget` directly writes `tools->m_primaryCtx.widgetClicked`

---

## File Map

| File | Responsibility | Action |
|------|---------------|--------|
| `libs/CV_db/include/ecvDisplayTools.h` | Static API + `m_primaryCtx` + `sharedTools()` | Major refactor: remove `m_primaryCtx`, migrate static methods |
| `libs/CV_db/src/ecvDisplayTools.cpp` | `s_tools`, `effectiveCtx()`, 70+ call sites | Major refactor: route all through explicit view context |
| `libs/CV_db/include/ecvViewManager.h` | `getEffectiveView()`, `getPrimaryView()` | Remove `getPrimaryView()`, add `resolveViewContext()` helper |
| `libs/CV_db/src/ecvViewManager.cpp` | View management | Remove primary-view special cases |
| `libs/VtkEngine/Visualization/ecvGLView.h` | Per-view context `m_ctx` | Ensure `viewContext()` is always non-null |
| `libs/VtkEngine/Visualization/ecvGLView.cpp` | Rendering pipeline | Update `ScopedRenderOverride` usage |
| `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` | `curCtx()` fallback | Remove `m_primaryCtx` fallback |
| `libs/VtkEngine/VTKExtensions/Widgets/CustomVtkCaptionWidget.cpp` | `m_primaryCtx.widgetClicked` | Route through view context |
| `app/MainWindow.cpp` | `copyPrimaryViewConfig`, `effectiveCtx()` fallback | Remove primary-view fallback |

---

### Task 1: Add `resolveViewContext()` helper to ecvViewManager ✅

**Files:**
- Modify: `libs/CV_db/include/ecvViewManager.h`
- Modify: `libs/CV_db/src/ecvViewManager.cpp`

- [ ] **Step 1: Add `resolveViewContext()` declaration**

In `libs/CV_db/include/ecvViewManager.h`, add after the `getEffectiveView()` declaration:

```cpp
    /// Returns a guaranteed non-null view context: effective view -> active view -> first view.
    /// Asserts in debug if no view exists at all.
    ecvViewContext& resolveViewContext();
    const ecvViewContext& resolveViewContext() const;
```

- [ ] **Step 2: Implement `resolveViewContext()`**

In `libs/CV_db/src/ecvViewManager.cpp`:

```cpp
ecvViewContext& ecvViewManager::resolveViewContext() {
    auto* view = getEffectiveView();
    if (view && view->viewContext()) return *view->viewContext();

    auto* active = getActiveView();
    if (active && active->viewContext()) return *active->viewContext();

    if (!m_views.isEmpty()) {
        auto* first = m_views.first();
        if (first && first->viewContext()) return *first->viewContext();
    }

    Q_ASSERT_X(false, "resolveViewContext", "No view context available");
    // Unreachable in normal usage, but satisfy compiler:
    static ecvViewContext s_emergency;
    return s_emergency;
}

const ecvViewContext& ecvViewManager::resolveViewContext() const {
    auto* view = getEffectiveView();
    if (view && view->viewContext()) return *view->viewContext();

    auto* active = getActiveView();
    if (active && active->viewContext()) return *active->viewContext();

    if (!m_views.isEmpty()) {
        auto* first = m_views.first();
        if (first && first->viewContext()) return *first->viewContext();
    }

    Q_ASSERT_X(false, "resolveViewContext", "No view context available");
    static ecvViewContext s_emergency;
    return s_emergency;
}
```

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target CV_db_LIB -j$(nproc) 2>&1 | tail -20`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvViewManager.h libs/CV_db/src/ecvViewManager.cpp
git commit -m "refactor: add resolveViewContext() helper for guaranteed context resolution"
```

---

### Task 2: Remove `m_primaryCtx` fallback from `effectiveCtx()` ✅

**Files:**
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`
- Modify: `libs/CV_db/include/ecvDisplayTools.h`

- [ ] **Step 1: Redirect `effectiveCtx()` to use `resolveViewContext()`**

In `libs/CV_db/src/ecvDisplayTools.cpp`, replace the `effectiveCtx()` implementations:

```cpp
ecvViewContext& ecvDisplayTools::effectiveCtx() {
    return ecvViewManager::instance().resolveViewContext();
}

const ecvViewContext& ecvDisplayTools::effectiveCtx() const {
    return ecvViewManager::instance().resolveViewContext();
}
```

- [ ] **Step 2: Mark `m_primaryCtx` as deprecated**

In `libs/CV_db/include/ecvDisplayTools.h`, add deprecation attribute:

```cpp
    [[deprecated("Use ecvViewManager::resolveViewContext() instead")]]
    ecvViewContext m_primaryCtx;
```

- [ ] **Step 3: Build and fix any compile warnings**

Run: `cmake --build build -j$(nproc) 2>&1 | grep -i "deprecat" | head -20`
Expected: Deprecation warnings from remaining `m_primaryCtx` direct access sites

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/src/ecvDisplayTools.cpp libs/CV_db/include/ecvDisplayTools.h
git commit -m "refactor: redirect effectiveCtx() through resolveViewContext(), deprecate m_primaryCtx"
```

---

### Task 3: Fix `QVTKWidgetCustom::curCtx()` primary fallback ✅

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp`

- [ ] **Step 1: Replace `m_primaryCtx` fallback in `curCtx()`**

In `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp`, find `curCtx()` (~line 96–104) and replace:

```cpp
ecvViewContext& QVTKWidgetCustom::curCtx() {
    if (m_ownerView && m_ownerView->viewContext()) {
        return *m_ownerView->viewContext();
    }
    return ecvViewManager::instance().resolveViewContext();
}
```

- [ ] **Step 2: Build and verify**

Run: `cmake --build build --target QVTKWidgetCustom -j$(nproc) 2>&1 | tail -10`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp
git commit -m "refactor: remove m_primaryCtx fallback from QVTKWidgetCustom::curCtx()"
```

---

### Task 4: Fix `CustomVtkCaptionWidget` direct `m_primaryCtx` write ✅

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/CustomVtkCaptionWidget.cpp`

- [ ] **Step 1: Route `widgetClicked` through view context**

At line ~96 of `CustomVtkCaptionWidget.cpp`, replace:

```cpp
// Before:
tools->m_primaryCtx.widgetClicked = true;

// After:
auto* view = ecvViewManager::instance().getEffectiveView();
if (view && view->viewContext()) {
    view->viewContext()->widgetClicked = true;
} else {
    ecvViewManager::instance().resolveViewContext().widgetClicked = true;
}
```

- [ ] **Step 2: Build and verify**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -10`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/CustomVtkCaptionWidget.cpp
git commit -m "refactor: route widgetClicked through view context instead of m_primaryCtx"
```

---

### Task 5: Remove `getPrimaryView()` and primary-view special cases ✅

**Files:**
- Modify: `libs/CV_db/include/ecvViewManager.h`
- Modify: `libs/CV_db/src/ecvViewManager.cpp`
- Modify: `app/MainWindow.cpp`

- [ ] **Step 1: Deprecate `getPrimaryView()`**

In `libs/CV_db/include/ecvViewManager.h`:

```cpp
    [[deprecated("All views are equivalent; use getActiveView() instead")]]
    ecvGenericGLDisplay* getPrimaryView() const;
```

- [ ] **Step 2: Remove `includePrimary` parameter from `redrawAll()`**

In `libs/CV_db/include/ecvViewManager.h`, change:

```cpp
    // Before:
    void redrawAll(bool only2D = false, bool includePrimary = true);
    // After:
    void redrawAll(bool only2D = false);
```

In `libs/CV_db/src/ecvViewManager.cpp`, update implementation to always redraw all views:

```cpp
void ecvViewManager::redrawAll(bool only2D) {
    for (auto* view : m_views) {
        if (view) view->redraw(only2D);
    }
}
```

- [ ] **Step 3: Fix all `redrawAll` callers that pass `includePrimary`**

Search codebase for `redrawAll(` with two arguments and remove the second argument.

- [ ] **Step 4: Rename `copyPrimaryViewConfig` to `copyActiveViewConfig`**

In `app/MainWindow.cpp`, rename the method and remove the `effectiveCtx()` fallback:

```cpp
void MainWindow::copyActiveViewConfig(ecvGLView* view) {
    auto* sourceView = ecvViewManager::instance().getActiveView();
    if (!sourceView || !sourceView->viewContext()) return;
    const ecvViewContext& srcCtx = *sourceView->viewContext();
    view->context() = srcCtx;
}
```

- [ ] **Step 5: Build and verify**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -20`
Expected: BUILD SUCCEEDED (possibly with deprecation warnings for `getPrimaryView()`)

- [ ] **Step 6: Commit**

```bash
git add libs/CV_db/include/ecvViewManager.h libs/CV_db/src/ecvViewManager.cpp app/MainWindow.cpp
git commit -m "refactor: remove primary-view special cases, all views are equivalent"
```

---

### Task 6: Migrate `ecvDisplayTools` static helpers (batch 1 — viewport/screen) (DEFERRED)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

This task migrates the viewport/screen-related static helpers to accept an explicit `ecvGenericGLDisplay*` parameter or use `resolveViewContext()` internally. The key methods to migrate:

- [ ] **Step 1: Migrate `viewportHeightFor()`, `Width()`, `Height()`, `GlWidth()`, `GlHeight()`, `size()`**

For each, add a view-parameterized overload that uses `view->viewContext()` directly, and update the existing static version to call `resolveViewContext()` instead of `effectiveCtx()`:

```cpp
// Example for Width():
static int Width(const ecvGenericGLDisplay* view) {
    if (view && view->viewContext()) return view->viewContext()->glW;
    return 0;
}
static int Width() {
    return ecvViewManager::instance().resolveViewContext().glW;
}
```

- [ ] **Step 2: Build and verify**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -20`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "refactor: migrate viewport/screen static helpers to view-parameterized API"
```

---

### Task 7: Migrate `ecvDisplayTools` static helpers (batch 2 — camera/interaction)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

This task migrates camera and interaction state helpers: `SetZoom`, `GetZoom`, `SetPivotPoint`, `SetCameraPos`, `GetCameraCenter`, `SetInteractionMode`, `GetInteractionMode`, `SetPerspectiveState`, `GetPerspectiveState`, etc.

- [ ] **Step 1: For each camera/interaction method, replace `effectiveCtx()` with `resolveViewContext()`**

The methods already have `ecvViewContext&` overloads in many cases. Ensure the parameter-less versions call `resolveViewContext()` and add `[[deprecated]]` annotations to encourage callers to use the parameterized versions.

- [ ] **Step 2: Build and verify**

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: migrate camera/interaction static helpers to resolveViewContext()"
```

---

### Task 8: Migrate `ecvDisplayTools` static helpers (batch 3 — drawing/rendering)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

Migrate drawing helpers: `Draw`, `DrawBBox`, `DisplayText`, `RedrawDisplay`, `RefreshDisplay`, hot-zone helpers, and widget/axes rendering methods.

- [ ] **Step 1: Replace all `sharedTools()->` calls in drawing methods with view-resolved calls**

Each drawing method should either:
- Accept an `ecvGenericGLDisplay*` parameter, or
- Use `ecvViewManager::instance().resolveViewContext()` internally

- [ ] **Step 2: Build and verify**

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: migrate drawing/rendering helpers to view-resolved context"
```

---

### Task 9: Remove `m_primaryCtx` member and `sharedTools()` accessor

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h`
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp`

- [ ] **Step 1: Remove `m_primaryCtx` member variable**

Delete the `m_primaryCtx` declaration from the class. Fix all remaining direct accesses.

- [ ] **Step 2: Remove `viewContext()` from `ecvDisplayTools` returning `&m_primaryCtx`**

The base `ecvDisplayTools::viewContext()` should return `nullptr` (not a valid view — it's a utility, not a view).

- [ ] **Step 3: Audit remaining `sharedTools()` call sites**

If all methods now route through `resolveViewContext()` or accept explicit parameters, `sharedTools()` is only needed for non-context state (timers, fonts, DB roots, etc.). Plan to either keep `sharedTools()` for utility state or distribute that state to `ecvViewManager`.

- [ ] **Step 4: Build and fix all errors**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: remove m_primaryCtx, ecvDisplayTools no longer pretends to be a view"
```

---

### Task 10: Final verification and documentation update

**Files:**
- Modify: `docs/user-guide/singleton-removal-migration-plan.md`
- Modify: `docs/user-guide/multi-window-paraview-alignment-design.md`

- [ ] **Step 1: Full build verification**

Run: `cmake --build build -j$(nproc) 2>&1 | tail -20`
Expected: BUILD SUCCEEDED with zero warnings related to singleton/primary-view patterns

- [ ] **Step 2: Grep verification — no remaining singleton patterns**

```bash
rg "m_primaryCtx" libs/ app/ --count
rg "sharedTools\(\)" libs/CV_db/include/ecvDisplayTools.h --count
rg "getPrimaryView" libs/ app/ --count
```

Expected: Zero matches for `m_primaryCtx` (except deprecated declaration if kept for ABI); `sharedTools()` limited to utility-only access; `getPrimaryView()` deprecated or removed.

- [ ] **Step 3: Update documentation**

Update `singleton-removal-migration-plan.md` to mark Phase 2 as complete. Update `multi-window-paraview-alignment-design.md` to reflect full window equivalence.

- [ ] **Step 4: Commit**

```bash
git commit -m "docs: mark Phase 2 singleton removal complete"
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| 70+ `effectiveCtx()` call sites — regression risk | Tasks 6-8 batch migrates by category; each batch is independently buildable |
| `s_tools` still needed for non-context state (timers, DB, fonts) | Task 9 audits and may keep `sharedTools()` for utility state only |
| Plugin compatibility — plugins may depend on static API | Deprecation warnings first; remove in next major version |
| Runtime crash when no views exist (startup/shutdown) | `resolveViewContext()` has Q_ASSERT + emergency fallback |

## Estimated Effort

- Tasks 1-4: ~2 hours (mechanical, low risk)
- Task 5: ~1 hour (rename + remove special cases)
- Tasks 6-8: ~4-6 hours (bulk migration, needs careful testing)
- Task 9: ~2 hours (final cleanup)
- Task 10: ~30 minutes (verification + docs)

**Total: ~10-12 hours of focused implementation**
