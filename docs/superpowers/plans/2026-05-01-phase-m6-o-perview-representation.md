# Phase M6 + O: Per-View Representation Deep Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propagate all `ecvViewRepresentation::Properties` overrides through the VTK actor pipeline, emit the `representationChanged` signal, and add a per-view display properties UI panel — completing the ParaView `vtkSMRepresentationProxy` alignment.

**Architecture:** `ecvViewRepresentation::Properties` (per-view optional overrides) → `ccHObject::draw()` merges into `CC_DRAW_CONTEXT` → `VtkDisplayTools` draw methods apply to VTK actors via `VtkVis`. A new `representationChanged` signal from `ecvRepresentationManager` triggers view refresh. The properties panel adds a "Display (per-view)" section for multi-view editing.

**Tech Stack:** C++ (Qt 5/6, VTK), `ecvViewRepresentation`, `ecvRepresentationManager`, `ecvDrawContext`, `VtkVis`, `ecvPropertiesTreeDelegate`

**References:**
- `docs/user-guide/multi-window-paraview-alignment-design.md` §4.4 GAP-4, §6.6 Phase O
- `docs/user-guide/multi-window-refactor-roadmap-Vtk-vs-CC.md` §TODO M6
- ParaView: `/Users/asher/develop/code/autopilot/MVS/ParaView`

---

## File Map

| File | Responsibility | Action |
|------|---------------|--------|
| `libs/CV_db/include/ecvViewRepresentation.h` | Per-view display state | Modify: add `effective*()` methods for all properties |
| `libs/CV_db/src/ecvViewRepresentation.cpp` | Effective value resolution | Modify: implement new effective methods |
| `libs/CV_db/src/ecvRepresentationManager.cpp` | Central registry | Modify: emit `representationChanged` on property updates |
| `libs/CV_db/include/ecvDrawContext.h` | Draw context struct | Modify: add `meshRenderingModeOverride` field |
| `libs/CV_db/src/ecvHObject.cpp` | Entity draw entry | Modify: propagate all per-view properties to context |
| `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` | VTK draw pipeline | Modify: use per-view overrides from context |
| `app/db_tree/ecvPropertiesTreeDelegate.cpp` | Properties panel UI | Modify: add per-view override controls |
| `app/db_tree/ecvPropertiesTreeDelegate.h` | Properties panel header | Modify: add per-view property role enums |

---

### Task 1: Add effective*() Methods to ecvViewRepresentation

**Files:**
- Modify: `libs/CV_db/include/ecvViewRepresentation.h`
- Modify: `libs/CV_db/src/ecvViewRepresentation.cpp`

Currently only `effectiveOpacity()` and `effectivePointSize()` exist. We need effective resolvers for all remaining `Properties` fields.

- [x] **Step 1: Add method declarations to header**

In `libs/CV_db/include/ecvViewRepresentation.h`, after the existing `effectivePointSize()` declaration (line ~70), add:

```cpp
    float effectiveLineWidth() const;
    RenderMode effectiveRenderMode() const;
    bool effectiveEdgeVisibility() const;
    int effectiveScalarFieldIndex() const;
    bool effectiveShowScalarField() const;
    bool effectiveShowColors() const;
    bool effectiveShowNormals() const;
    float effectiveNormalScale() const;
```

- [x] **Step 2: Implement effective methods in .cpp**

In `libs/CV_db/src/ecvViewRepresentation.cpp`, after the existing `effectivePointSize()` implementation, add:

```cpp
float ecvViewRepresentation::effectiveLineWidth() const {
    if (m_properties.lineWidth.has_value()) {
        return m_properties.lineWidth.value();
    }
    return 2.0f;  // ecvDrawContext default
}

ecvViewRepresentation::RenderMode
ecvViewRepresentation::effectiveRenderMode() const {
    if (m_properties.renderMode.has_value()) {
        return m_properties.renderMode.value();
    }
    return RenderMode::Inherit;
}

bool ecvViewRepresentation::effectiveEdgeVisibility() const {
    if (m_properties.edgeVisibility.has_value()) {
        return m_properties.edgeVisibility.value();
    }
    return false;
}

int ecvViewRepresentation::effectiveScalarFieldIndex() const {
    if (m_properties.scalarFieldIndex.has_value()) {
        return m_properties.scalarFieldIndex.value();
    }
    return -1;  // no override
}

bool ecvViewRepresentation::effectiveShowScalarField() const {
    if (m_properties.showScalarField.has_value()) {
        return m_properties.showScalarField.value();
    }
    if (m_entity) {
        return m_entity->sfShown();
    }
    return false;
}

bool ecvViewRepresentation::effectiveShowColors() const {
    if (m_properties.showColors.has_value()) {
        return m_properties.showColors.value();
    }
    if (m_entity) {
        return m_entity->colorsShown();
    }
    return true;
}

bool ecvViewRepresentation::effectiveShowNormals() const {
    if (m_properties.showNormals.has_value()) {
        return m_properties.showNormals.value();
    }
    if (m_entity) {
        return m_entity->normalsShown();
    }
    return false;
}

float ecvViewRepresentation::effectiveNormalScale() const {
    if (m_properties.normalScale.has_value()) {
        return m_properties.normalScale.value();
    }
    return 1.0f;
}
```

- [x] **Step 3: Add include for sfShown/colorsShown/normalsShown**

Verify that `ecvHObject.h` (already included as `ecvHObject.h`) provides `sfShown()`, `colorsShown()`, `normalsShown()`. If it's on `ccGenericPointCloud`, add the appropriate cast guard in the implementations above (similar to how `effectivePointSize()` already does `isKindOf(CV_TYPES::POINT_CLOUD)`).

---

### Task 2: Emit representationChanged Signal

**Files:**
- Modify: `libs/CV_db/src/ecvRepresentationManager.cpp`
- Modify: `libs/CV_db/src/ecvViewRepresentation.cpp`

The `representationChanged` signal is declared in `ecvRepresentationManager.h` (line 73) but never emitted anywhere.

- [x] **Step 1: Add notifyChanged method to ecvRepresentationManager**

In `libs/CV_db/include/ecvRepresentationManager.h`, add a public method:

```cpp
    void notifyChanged(ecvViewRepresentation* rep);
```

In `libs/CV_db/src/ecvRepresentationManager.cpp`, add:

```cpp
void ecvRepresentationManager::notifyChanged(ecvViewRepresentation* rep) {
    if (rep) {
        emit representationChanged(rep);
    }
}
```

- [x] **Step 2: Emit from setProperties**

In `libs/CV_db/src/ecvViewRepresentation.cpp`, modify `setProperties`:

```cpp
void ecvViewRepresentation::setProperties(const Properties& props) {
    m_properties = props;
    m_dirty = true;
    ecvRepresentationManager::instance().notifyChanged(this);
}
```

- [x] **Step 3: Emit from setVisible**

In `libs/CV_db/src/ecvViewRepresentation.cpp`, modify `setVisible`:

```cpp
void ecvViewRepresentation::setVisible(bool v) {
    m_visibilityOverride = v;
    m_dirty = true;
    ecvRepresentationManager::instance().notifyChanged(this);
}
```

- [x] **Step 4: Add include for ecvRepresentationManager.h in ecvViewRepresentation.cpp**

Add at the top of `libs/CV_db/src/ecvViewRepresentation.cpp`:

```cpp
#include "ecvRepresentationManager.h"
```

---

### Task 3: Propagate Per-View Properties Through Draw Context

**Files:**
- Modify: `libs/CV_db/src/ecvHObject.cpp:1582-1585`

Currently only `opacity` is propagated from the representation. We need to propagate `pointSize`, `lineWidth`, and `renderMode` as well.

- [x] **Step 1: Extend the per-view property propagation block**

In `libs/CV_db/src/ecvHObject.cpp`, replace the existing property propagation block (lines ~1582-1585):

```cpp
    context.visible = m_visible;
    context.opacity = (viewRep && viewRep->properties().opacity.has_value())
                              ? viewRep->effectiveOpacity()
                              : getOpacity();
```

with:

```cpp
    context.visible = m_visible;
    context.opacity = (viewRep && viewRep->properties().opacity.has_value())
                              ? viewRep->effectiveOpacity()
                              : getOpacity();

    if (viewRep && viewRep->properties().pointSize.has_value()) {
        context.defaultPointSize =
                static_cast<unsigned char>(viewRep->effectivePointSize());
    }
    if (viewRep && viewRep->properties().lineWidth.has_value()) {
        context.defaultLineWidth =
                static_cast<unsigned char>(viewRep->effectiveLineWidth());
        context.currentLineWidth = context.defaultLineWidth;
    }
    if (viewRep && viewRep->properties().renderMode.has_value()) {
        auto rm = viewRep->effectiveRenderMode();
        if (rm != ecvViewRepresentation::RenderMode::Inherit) {
            context.meshRenderingMode =
                    static_cast<MESH_RENDERING_MODE>(static_cast<int>(rm));
        }
    }
```

- [x] **Step 2: Mark representation clean after draw**

After `drawMeOnly(context)` (line ~1614), add:

```cpp
            drawMeOnly(context);

            if (viewRep && viewRep->isDirty()) {
                viewRep->setDirty(false);
            }
```

---

### Task 4: Connect representationChanged to View Refresh

**Files:**
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp`

When a representation changes, the owning view must refresh.

- [x] **Step 1: Find the ecvGLView constructor that connects signals**

In `libs/VtkEngine/Visualization/ecvGLView.cpp`, locate the constructor or `initializeGL`-equivalent where signals are connected.

- [x] **Step 2: Connect representationChanged to redraw**

Add a connection in the ecvGLView initialization code:

```cpp
connect(&ecvRepresentationManager::instance(),
        &ecvRepresentationManager::representationChanged,
        this, [this](ecvViewRepresentation* rep) {
            if (rep && rep->getView() == this) {
                redraw();
            }
        });
```

This ensures only the affected view redraws when its representation changes.

- [x] **Step 3: Add necessary includes**

Add at the top of `ecvGLView.cpp` if not already present:

```cpp
#include "ecvRepresentationManager.h"
#include "ecvViewRepresentation.h"
```

---

### Task 5: Properties Panel — Per-View Display Section

**Files:**
- Modify: `app/db_tree/ecvPropertiesTreeDelegate.h`
- Modify: `app/db_tree/ecvPropertiesTreeDelegate.cpp`

Add a "Display (per-view)" section to the properties panel that shows per-view override controls when multiple views exist.

- [x] **Step 1: Add property role enums**

In `app/db_tree/ecvPropertiesTreeDelegate.h`, locate the property role enum and add:

```cpp
    OBJECT_PERVIEW_OPACITY,
    OBJECT_PERVIEW_POINT_SIZE,
    OBJECT_PERVIEW_RENDER_MODE,
    OBJECT_PERVIEW_VISIBILITY,
```

- [x] **Step 2: Add fillWithPerViewProperties method declaration**

In `app/db_tree/ecvPropertiesTreeDelegate.h`, add:

```cpp
    void fillWithPerViewProperties();
```

- [x] **Step 3: Implement fillWithPerViewProperties**

In `app/db_tree/ecvPropertiesTreeDelegate.cpp`, add:

```cpp
void ccPropertiesTreeDelegate::fillWithPerViewProperties() {
    if (!m_currentObject || !m_currentObject->getDisplay()) return;

    auto* activeView = m_currentObject->getDisplay();
    auto* viewRep = ecvRepresentationManager::instance().getRepresentation(
            m_currentObject, activeView);

    if (!viewRep) return;

    addSeparator(tr("Display (per-view override)"));

    // Per-view visibility override
    appendRow(ITEM(tr("Visible (this view)")),
              CHECKABLE_ITEM(viewRep->isVisible(),
                             viewRep->hasVisibilityOverride()
                                     ? tr("Per-view override active")
                                     : tr("Using global visibility")));

    // Per-view opacity
    if (m_currentObject->isKindOf(CV_TYPES::POINT_CLOUD) ||
        m_currentObject->isKindOf(CV_TYPES::MESH)) {
        appendRow(
                ITEM(tr("Opacity (this view)")),
                PERSISTENT_EDITOR(OBJECT_PERVIEW_OPACITY), true);
    }

    // Per-view point size
    if (m_currentObject->isKindOf(CV_TYPES::POINT_CLOUD)) {
        appendRow(
                ITEM(tr("Point Size (this view)")),
                PERSISTENT_EDITOR(OBJECT_PERVIEW_POINT_SIZE), true);
    }
}
```

- [x] **Step 4: Call fillWithPerViewProperties from fillModel**

In `app/db_tree/ecvPropertiesTreeDelegate.cpp`, locate `fillModel()` and add the call after the existing `fillWithViewProperties()`:

```cpp
    fillWithViewProperties();
    fillWithPerViewProperties();  // NEW
```

- [x] **Step 5: Handle per-view property edits in updateItem**

In `app/db_tree/ecvPropertiesTreeDelegate.cpp`, locate the `updateItem` switch statement and add cases:

```cpp
case OBJECT_PERVIEW_OPACITY: {
    auto* activeView = m_currentObject->getDisplay();
    auto* rep = ecvRepresentationManager::instance().ensureRepresentation(
            m_currentObject, activeView);
    if (rep) {
        auto props = rep->properties();
        props.opacity = value.toFloat();
        rep->setProperties(props);
    }
} break;

case OBJECT_PERVIEW_POINT_SIZE: {
    auto* activeView = m_currentObject->getDisplay();
    auto* rep = ecvRepresentationManager::instance().ensureRepresentation(
            m_currentObject, activeView);
    if (rep) {
        auto props = rep->properties();
        props.pointSize = value.toFloat();
        rep->setProperties(props);
    }
} break;
```

- [x] **Step 6: Add includes**

Add at the top of `ecvPropertiesTreeDelegate.cpp`:

```cpp
#include "ecvRepresentationManager.h"
#include "ecvViewRepresentation.h"
```

---

### Task 6: Update Documentation

**Files:**
- Modify: `docs/user-guide/multi-window-refactor-roadmap-Vtk-vs-CC.md`
- Modify: `docs/user-guide/multi-window-paraview-alignment-design.md`
- Modify: `docs/user-guide/singleton-removal-migration-plan.md`

- [x] **Step 1: Update M6 status in roadmap**

In `docs/user-guide/multi-window-refactor-roadmap-Vtk-vs-CC.md`, change the M6 heading from:

```
### TODO M6: Per-View 表示完善 🔲 (架构已就绪，深化待续)
```

to:

```
### TODO M6: Per-View 表示完善 ✅ (2026-05-0X)
```

(Replace `0X` with actual completion date.)

- [x] **Step 2: Update Phase O status in alignment doc**

In `docs/user-guide/multi-window-paraview-alignment-design.md`, update §6.6 Phase O header and §2.4 Per-View Representation table status from **PARTIAL** to **ALIGNED**.

- [x] **Step 3: Update GAP-4 status**

In `docs/user-guide/multi-window-paraview-alignment-design.md`, update §4.4 GAP-4 to mark as **RESOLVED**.

- [x] **Step 4: Add changelog entry to singleton-removal-migration-plan.md**

Add a new version entry recording M6/Phase O completion with the specific changes made.

---

## Dependency Graph

```
Task 1 (effective methods)
    │
    ├──► Task 2 (representationChanged signal) ──► Task 4 (view refresh connect)
    │
    └──► Task 3 (draw context propagation)
                                                    ┌──► Task 6 (docs)
Task 5 (properties panel UI) ──────────────────────┘
```

Tasks 1-4 are sequential (each builds on the prior). Task 5 is independent of Tasks 2-4. Task 6 runs last.

## Compilation & Test Strategy

**DO NOT compile until all tasks are complete** (per user instruction — macOS build is slow).

After all code changes:

```bash
cd /Users/asher/develop/code/github/ACloudViewer
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j48
```

**Manual verification checklist:**
1. Open app → create split view (2 windows)
2. Load a point cloud → verify it appears in both views
3. In properties panel, change "Opacity (this view)" in one view → verify only that view updates
4. Change "Point Size (this view)" → verify per-view isolation
5. Close one view → verify no crash, representations cleaned up
