# Phase F: Advanced Features — Optional Enhancements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optional enhancements that build on the clean per-view architecture from Phases A-E. Each feature is independent and can be implemented in any order based on product priority.

**Architecture:** With each `ecvGLView` fully owning its state and no singleton per-view members, these features become straightforward extensions rather than fighting global state.

**Tech Stack:** C++17, Qt 5/6, VTK, CMake

**Prerequisite:** Phases A-E complete.

---

## Feature 1: Layout Persistence

**Inspiration:** ParaView `vtkSMViewLayoutProxy`

**Goal:** Save/restore the MDI window arrangement (positions, sizes, split ratios) to/from project files.

### Task F1.1: Define Layout Serialization Format

- [ ] **Step 1: Design JSON/XML schema for layout state**

```json
{
  "layout": {
    "type": "mdi_tabbed",
    "views": [
      { "id": 0, "geometry": [100, 100, 800, 600], "is_primary": true },
      { "id": 1, "geometry": [900, 100, 400, 300], "is_primary": false }
    ],
    "active_view_id": 0
  }
}
```

- [ ] **Step 2: Add `saveLayout()` / `restoreLayout()` to `ecvViewManager`**
- [ ] **Step 3: Hook into project save/load**

---

## Feature 2: Per-View Representation

**Inspiration:** ParaView `pqDataRepresentation`

**Goal:** The same `ccHObject` can have different visual properties (color, visibility, render mode) in different views.

### Task F2.1: Design Per-View Display Properties

- [ ] **Step 1: Add `PerViewDisplayProps` to `ecvGLView`**

```cpp
struct PerViewDisplayProps {
    bool visible = true;
    ecvColor::Rgb color;
    int renderMode = -1;  // -1 = use entity default
};

std::unordered_map<unsigned, PerViewDisplayProps> m_perViewProps;
```

- [ ] **Step 2: Modify `ccHObject::draw` to check per-view overrides**
- [ ] **Step 3: Add UI for per-view visibility toggle**

---

## Feature 3: Selection Link

**Inspiration:** ParaView `vtkSMSelectionLink`

**Goal:** Selecting an entity in one view optionally highlights it in all views (linked selection) or keeps selection per-view.

### Task F3.1: Implement Selection Sync Manager

- [ ] **Step 1: Add `ecvSelectionLink` class**

```cpp
class ecvSelectionLink {
    bool m_enabled = false;
    void onSelectionChanged(ecvGLView* source, const QSet<unsigned>& selectedIds);
};
```

- [ ] **Step 2: Connect to `ecvViewManager::selectionChanged` signal**
- [ ] **Step 3: Add UI toggle for linked/unlinked selection**

---

## Feature 4: Tab Multi-Layout

**Inspiration:** ParaView `pqTabbedMultiViewWidget`

**Goal:** Multiple tab pages, each with its own independent MDI layout.

### Task F4.1: Add Tab Container

- [ ] **Step 1: Replace `QMdiArea` with `QTabWidget` containing `QMdiArea` per tab**
- [ ] **Step 2: Each tab has its own `ecvViewManager` scope**
- [ ] **Step 3: Tab creation/deletion UI**

---

## Feature Priority Matrix

| Feature | User Value | Effort | Dependency |
|---|---|---|---|
| Layout persistence | High | Low | Phase E |
| Per-view representation | Medium | Medium | Phase E |
| Selection link | Low-Medium | Low | Phase E |
| Tab multi-layout | Medium | High | Phase E + Layout persistence |

**Recommendation:** Start with Layout Persistence (F1) — highest value/effort ratio.
