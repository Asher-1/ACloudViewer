# Multi-Window Lifecycle Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Eliminate all remaining crash and desync bugs in the multi-window view lifecycle (creation, activation, destruction).

**Architecture:** Harden `MainWindow`, `ecvViewManager`, `VtkDisplayTools` lifecycle paths. No new classes.

**Tech Stack:** C++ / Qt 5 / VTK / CMake

---

## Task 1: Handle MDI deactivation (nullptr) in on3DViewActivated

**Files:**
- Modify: `app/MainWindow.cpp` — `on3DViewActivated` method

**Problem:** When Qt emits `subWindowActivated(nullptr)` (e.g. all subwindows minimized or focus leaves MDI), the handler returns immediately without clearing or reconciling `ecvViewManager`'s active view. Tools remain bound to a stale view.

**Fix:**
- [ ] When `mdiWin == nullptr` and not `m_closing`, call `ecvViewManager::instance().setActiveView(nullptr)` to reconcile state.
- [ ] After setting active view to nullptr, call `rebindToolsToActiveView(nullptr)` or handle the null case by disabling overlays.

**Acceptance:** `on3DViewActivated(nullptr)` clears the active view in the manager and does not leave tools bound to a phantom view.

---

## Task 2: Handle activeViewChanged(nullptr) signal

**Files:**
- Modify: `app/MainWindow.cpp` — the `activeViewChanged` lambda connection (around line 851)

**Problem:** The lambda ignores `newActive == nullptr`. When the last view is unregistered, tools are not explicitly cleared, potentially leaving stale pointers.

**Fix:**
- [ ] Remove the early return when `newActive == nullptr`.
- [ ] When `newActive` is nullptr, call `restorePrimaryView()` or clear overlay bindings safely.

**Acceptance:** After last view unregistered, tools are in a clean/default state, no dangling bindings.

---

## Task 3: ecvViewManager::unregisterView — prefer MDI-active replacement

**Files:**
- Modify: `libs/CV_db/src/ecvViewManager.cpp` — `unregisterView` method

**Problem:** When the active view is unregistered, the replacement is always `m_views.first()` (registration order), which may not be the user-focused MDI tab.

**Fix:**
- [ ] After removing the view from `m_views`, if `m_activeView` was the removed view, attempt to find the MDI-active `ecvGLView` via the signal or use the last-activated view if tracked.
- [ ] If no better candidate, fall back to `m_views.first()` or nullptr.
- [ ] Emit `activeViewChanged` for the replacement so downstream rebinds correctly.

**Acceptance:** After unregistering the active view, the new active view matches the user's focused MDI tab (or is null if none remain).

---

## Task 4: Remove duplicate registerView in new3DView

**Files:**
- Modify: `app/MainWindow.cpp` — `new3DView` method (around line 2969)

**Problem:** `ecvGLView::Create` already calls `registerView` and `RegisterGLDisplay`. `new3DView` calls both again — harmless but confusing and redundant.

**Fix:**
- [ ] Remove the duplicate `ecvViewManager::instance().registerView(view)` call in `new3DView`.
- [ ] Remove the duplicate `ecvGenericGLDisplay::RegisterGLDisplay(view->asWidget(), view)` call in `new3DView`.

**Acceptance:** Each view is registered exactly once (in `ecvGLView::Create`).

---

## Task 5: Harden rebindToolsToActiveView for null/degraded states

**Files:**
- Modify: `app/MainWindow.cpp` — `rebindToolsToActiveView` method

**Problem:** If `GetCurrentScreen()` is null (e.g. after last view closed), the function returns early at line ~2502, leaving overlays and property panels not re-linked.

**Fix:**
- [ ] When `rebindToolsToActiveView(nullptr)` is called (or display is null), explicitly hide/detach overlay tools (clipping box, section extraction, etc.) instead of silently returning.
- [ ] When `GetCurrentScreen()` is null, skip screen-dependent operations but still update non-screen state (selection, properties panel clearing).

**Acceptance:** Calling `rebindToolsToActiveView(nullptr)` leaves the UI in a clean state: overlays hidden, properties cleared, no dangling widget references.

---

## Task 6: Verify build and run smoke test

**Files:** None (verification only)

- [ ] `cmake --build build_app --target CloudViewerApp -j24` succeeds
- [ ] Verify all modified files compile without warnings related to our changes

**Acceptance:** Build succeeds with exit code 0 for CloudViewerApp target.
