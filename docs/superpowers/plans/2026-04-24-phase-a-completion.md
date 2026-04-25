# Phase A Completion: Final `TheInstance()->m_*` Routing in Header

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the last 2 per-view `TheInstance()->m_*` reads in `ecvDisplayTools.h` through `effectiveCtx()`, completing Phase A.

**Architecture:** After Phase A Tasks 1-4, 13 `TheInstance()->m_*` reads remain in the header. Of those, 11 are **global state** (correctly on the singleton: `m_win`, `m_globalDBRoot`, `m_winDBRoot`, `m_currentScreen`, `m_mainScreen`, `m_removeAllFlag`, `m_font`) or **Phase B targets** (`m_updateFBO`). Only 2 are **per-view state** that should use `effectiveCtx()`: `m_validProjectionMatrix` (line 2154) and `m_validModelviewMatrix` (line 2157).

**Tech Stack:** C++17, Qt 5/6, VTK, CMake

---

## Remaining `TheInstance()->m_*` reads — categorized

| Line | Field | Category | Action |
|---|---|---|---|
| 997 | `m_currentScreen` | Global/App | Keep |
| 1012, 1020 | `m_mainScreen` | Global/App | Keep |
| 1027, 1034 | `m_win` | Global/App | Keep |
| 1073 | `m_winDBRoot` | Global/App | Keep |
| 1103 | `m_globalDBRoot` | Global/App | Keep |
| 1194 | `m_removeAllFlag` | Global/App | Keep |
| 2152 | `m_updateFBO` | Phase B target | Keep (address in Phase B) |
| **2154** | **`m_validProjectionMatrix`** | **Per-view** | **Migrate** |
| **2157** | **`m_validModelviewMatrix`** | **Per-view** | **Migrate** |
| 2324, 2341 | `m_font` | Global/App | Keep |

---

### Task 1: Route `InvalidateViewport` and `InvalidateVisualization` Through `effectiveCtx()`

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h:2152-2158`

- [ ] **Step 1: Read current code at lines 2152-2158**

```cpp
inline static void Deprecate3DLayer() { TheInstance()->m_updateFBO = true; }
inline static void InvalidateViewport() {
    TheInstance()->m_validProjectionMatrix = false;
}
inline static void InvalidateVisualization() {
    TheInstance()->m_validModelviewMatrix = false;
}
```

- [ ] **Step 2: Change `InvalidateViewport` to use `effectiveCtx()`**

```cpp
inline static void Deprecate3DLayer() { TheInstance()->m_updateFBO = true; }
inline static void InvalidateViewport() {
    TheInstance()->effectiveCtx().validProjectionMatrix = false;
}
inline static void InvalidateVisualization() {
    TheInstance()->effectiveCtx().validModelviewMatrix = false;
}
```

`ecvViewContext` already has `validProjectionMatrix` (line 45) and `validModelviewMatrix` (line 46).

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`

- [ ] **Step 4: Verify header has exactly 11 `TheInstance()->m_*` reads remaining**

Run: `rg 'TheInstance\(\)->m_' libs/CV_db/include/ecvDisplayTools.h | wc -l`

Expected: 11 (all global/app state or Phase B targets).

- [ ] **Step 5: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h
git commit -m "refactor(phase-a): route InvalidateViewport/Visualization through effectiveCtx

m_validProjectionMatrix and m_validModelviewMatrix are per-view state
already in ecvViewContext. Route through effectiveCtx() instead of
reading the singleton copy. Phase A header migration complete —
remaining 11 TheInstance()->m_* reads are global/app state."
```

---

### Task 2: Phase A Acceptance Verification

- [ ] **Step 1: Count per-view `TheInstance()->m_*` reads — should be 0**

Run: `rg 'TheInstance\(\)->m_(validProjection|validModelview|viewportParams|interactionFlags|pickingMode|bubbleView|pivotVis|clickableItems|showCursor|exclusiveFull|showDebug|rotationAxis|displayOverlay)' libs/CV_db/include/ecvDisplayTools.h`

Expected: No matches (all routed through `effectiveCtx()`).

- [ ] **Step 2: Count `effectiveCtx()` reads in header**

Run: `rg 'effectiveCtx\(\)' libs/CV_db/include/ecvDisplayTools.h | wc -l`

Expected: ~30 (up from 28).

- [ ] **Step 3: Count context-aware API overloads**

Run: `rg 'ecvViewContext' libs/CV_db/include/ecvDisplayTools.h | wc -l`

Expected: 25+ (11 context-aware APIs).

- [ ] **Step 4: Runtime regression test**

1. Load point cloud, rotate/zoom
2. Create second view — renders independently
3. Close second view — first keeps working
4. Click object in DB tree — highlights without crash

---

## Phase A Final Metrics

| Metric | Baseline | After Phase A |
|---|---|---|
| `TheInstance()->m_*` in header | 35 | **11** (all global/app) |
| `effectiveCtx()` reads in header | 0 | **~30** |
| Context-aware APIs | 6 | **11** |
| Per-view header reads | 35 | **0** |
