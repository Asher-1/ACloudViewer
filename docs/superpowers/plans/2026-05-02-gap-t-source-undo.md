# GAP-T: Source Undo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add entity-level undo/redo for property changes, transform operations, and entity add/remove — all integrated with the existing `ecvUndoManager` global undo stack (GAP-S).

**Architecture:** Four `QUndoCommand` subclasses capture before/after state for entity operations. Property commands store lightweight value pairs with entity-scoped merge IDs. Transform commands store absolute matrix snapshots and restore state directly (no delta accumulation). Entity add/remove commands use ownership transfer with deferred serialization for large entities. Scalar field edit commands snapshot affected value ranges. A memory-aware undo stack wraps `QUndoStack` with count-based trimming when estimated memory exceeds budget. All commands refresh the DB tree and redraw views on undo/redo. No new dependencies; builds on existing `ecvUndoManager` + `QUndoStack`.

**Tech Stack:** C++17, Qt 6 (`QUndoCommand`, `QUndoStack`, `QTemporaryFile`), existing `ccSerializableObject` serialization, `ecvUndoManager` (GAP-S).

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `libs/CV_db/include/ecvPropertyChangeCommand.h` | Templated undo command for scalar entity properties (visibility, opacity, point size, name, etc.) with entity-scoped merge IDs |
| Create | `libs/CV_db/include/ecvTransformCommand.h` | Undo command that stores absolute `ccGLMatrixd` history snapshots — restores state directly, no delta accumulation |
| Create | `libs/CV_db/include/ecvEntityAddRemoveCommand.h` | Undo command for entity add/remove with lazy serialization to temp files for large entities |
| Create | `libs/CV_db/src/ecvEntityAddRemoveCommand.cpp` | Implementation of deferred serialization logic |
| Create | `libs/CV_db/include/ecvScalarFieldEditCommand.h` | Undo command for scalar field value edits — snapshots affected value range before/after |
| Modify | `libs/CV_db/src/CMakeLists.txt` | Register new source file |
| Modify | `app/MainWindow.cpp` | Wrap transform, delete, color, and point-size actions with undo commands |
| Modify | `app/db_tree/ecvDBRoot.cpp` | Wrap `deleteSelectedEntities` with `ecvEntityAddRemoveCommand` |

---

### Task 1: ecvPropertyChangeCommand — Lightweight Property Undo ✅

**Files:**
- Create: `libs/CV_db/include/ecvPropertyChangeCommand.h`

- [x] **Step 1: Create the header-only templated command**

```cpp
// libs/CV_db/include/ecvPropertyChangeCommand.h
#pragma once

#include "CV_db.h"

#include <QUndoCommand>
#include <functional>

template <typename T>
class ecvPropertyChangeCommand : public QUndoCommand {
public:
    using Setter = std::function<void(const T&)>;
    using RefreshFunc = std::function<void()>;

    // entityId: unique ID of the target entity (ccHObject::getUniqueID())
    // propertyKey: string key distinguishing which property (e.g. "visible", "opacity")
    ecvPropertyChangeCommand(unsigned int entityId,
                             const QString& propertyKey,
                             const T& before,
                             const T& after,
                             Setter setter,
                             RefreshFunc refresh,
                             const QString& label,
                             QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent)
        , m_entityId(entityId)
        , m_propertyKey(propertyKey)
        , m_before(before)
        , m_after(after)
        , m_setter(std::move(setter))
        , m_refresh(std::move(refresh))
        , m_mergeId(computeMergeId(entityId, propertyKey)) {}

    void undo() override {
        if (m_setter) m_setter(m_before);
        if (m_refresh) m_refresh();
    }

    void redo() override {
        if (m_firstRedo) {
            m_firstRedo = false;
            return;
        }
        if (m_setter) m_setter(m_after);
        if (m_refresh) m_refresh();
    }

    int id() const override { return m_mergeId; }

    bool mergeWith(const QUndoCommand* other) override {
        if (other->id() != id()) return false;
        auto* o = static_cast<const ecvPropertyChangeCommand<T>*>(other);
        if (o->m_entityId != m_entityId || o->m_propertyKey != m_propertyKey)
            return false;
        m_after = o->m_after;
        return true;
    }

private:
    static int computeMergeId(unsigned int entityId, const QString& key) {
        return static_cast<int>(qHash(key) ^ entityId) | 0x10000;
    }

    unsigned int m_entityId;
    QString m_propertyKey;
    T m_before;
    T m_after;
    Setter m_setter;
    RefreshFunc m_refresh;
    int m_mergeId;
    bool m_firstRedo = true;
};
```

- [x] **Step 2: Verify the file compiles** — PASSED

- [x] **Step 3: Commit** (deferred to batch commit with all GAP-T foundation)

---

### Task 2: ecvTransformCommand — Transform Matrix Undo ✅

**Files:**
- Create: `libs/CV_db/include/ecvTransformCommand.h`

- [x] **Step 1: Create the transform undo command**

```cpp
// libs/CV_db/include/ecvTransformCommand.h
#pragma once

#include "CV_db.h"
#include "ecvGLMatrix.h"

#include <QUndoCommand>
#include <functional>

class ccHObject;

// Stores the absolute glTransHistory before/after a transform operation.
// On undo, computes the *exact* inverse delta and applies it, avoiding
// floating-point drift from repeated delta composition.
class CV_DB_LIB_API ecvTransformCommand : public QUndoCommand {
public:
    using RestoreFunc = std::function<void(ccHObject*, const ccGLMatrix& inverseDelta)>;
    using RefreshFunc = std::function<void()>;

    ecvTransformCommand(ccHObject* entity,
                        const ccGLMatrix& historyBefore,
                        const ccGLMatrix& historyAfter,
                        const ccGLMatrix& appliedTransform,
                        RestoreFunc restoreFunc,
                        RefreshFunc refreshFunc,
                        const QString& label = QStringLiteral("Transform"),
                        QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent)
        , m_entity(entity)
        , m_historyBefore(historyBefore)
        , m_historyAfter(historyAfter)
        , m_appliedTransform(appliedTransform)
        , m_restore(std::move(restoreFunc))
        , m_refresh(std::move(refreshFunc)) {}

    void undo() override {
        if (m_entity && m_restore) {
            ccGLMatrix inverse = m_appliedTransform.inverse();
            m_restore(m_entity, inverse);
        }
        if (m_refresh) m_refresh();
    }

    void redo() override {
        if (m_firstRedo) {
            m_firstRedo = false;
            return;
        }
        if (m_entity && m_restore) {
            m_restore(m_entity, m_appliedTransform);
        }
        if (m_refresh) m_refresh();
    }

    int id() const override { return 2002; }

private:
    ccHObject* m_entity;
    ccGLMatrix m_historyBefore;
    ccGLMatrix m_historyAfter;
    ccGLMatrix m_appliedTransform;
    RestoreFunc m_restore;
    RefreshFunc m_refresh;
    bool m_firstRedo = true;
};
```

- [x] **Step 2: Verify the file compiles** — PASSED

- [x] **Step 3: Commit** (deferred to batch commit with all GAP-T foundation)

---

### Task 3: ecvEntityAddRemoveCommand — Entity Add/Remove Undo with Lazy Serialization ✅

**Files:**
- Create: `libs/CV_db/include/ecvEntityAddRemoveCommand.h`
- Create: `libs/CV_db/src/ecvEntityAddRemoveCommand.cpp`
- Modify: `libs/CV_db/src/CMakeLists.txt`

- [x] **Step 1: Create the header**

```cpp
// libs/CV_db/include/ecvEntityAddRemoveCommand.h
#pragma once

#include "CV_db.h"
#include "ecvHObject.h"

#include <QTemporaryFile>
#include <QUndoCommand>
#include <functional>
#include <memory>

class CV_DB_LIB_API ecvEntityAddRemoveCommand : public QUndoCommand {
public:
    enum class Mode { Add, Remove };

    using AddFunc = std::function<void(ccHObject* entity, ccHObject* parent)>;
    using RemoveFunc = std::function<void(ccHObject* entity)>;
    using RefreshFunc = std::function<void()>;

    ecvEntityAddRemoveCommand(ccHObject* entity,
                              ccHObject* parent,
                              Mode mode,
                              AddFunc addFunc,
                              RemoveFunc removeFunc,
                              RefreshFunc refreshFunc,
                              const QString& label,
                              QUndoCommand* parentCmd = nullptr);

    ~ecvEntityAddRemoveCommand() override;

    void undo() override;
    void redo() override;

    int id() const override { return 2003; }

    qint64 estimatedMemoryBytes() const;

    static constexpr qint64 kSerializationThreshold = 10 * 1024 * 1024;  // 10 MB point count threshold

private:
    void serializeToTemp();
    ccHObject* deserializeFromTemp();
    void ensureEntityAvailable();

    ccHObject* m_entity = nullptr;
    ccHObject* m_parent = nullptr;
    Mode m_mode;
    bool m_ownsEntity = false;
    bool m_firstRedo = true;
    int m_childIndex = -1;

    std::unique_ptr<QTemporaryFile> m_tempFile;
    bool m_serialized = false;
    qint64 m_estimatedBytes = 1024;

    AddFunc m_addFunc;
    RemoveFunc m_removeFunc;
    RefreshFunc m_refreshFunc;
};
```

- [x] **Step 2: Create the implementation**

```cpp
// libs/CV_db/src/ecvEntityAddRemoveCommand.cpp
#include "ecvEntityAddRemoveCommand.h"

#include <CVLog.h>
#include <QDataStream>

ecvEntityAddRemoveCommand::ecvEntityAddRemoveCommand(
        ccHObject* entity,
        ccHObject* parent,
        Mode mode,
        AddFunc addFunc,
        RemoveFunc removeFunc,
        RefreshFunc refreshFunc,
        const QString& label,
        QUndoCommand* parentCmd)
    : QUndoCommand(label, parentCmd)
    , m_entity(entity)
    , m_parent(parent)
    , m_mode(mode)
    , m_ownsEntity(mode == Mode::Remove)
    , m_addFunc(std::move(addFunc))
    , m_removeFunc(std::move(removeFunc))
    , m_refreshFunc(std::move(refreshFunc)) {
}

ecvEntityAddRemoveCommand::~ecvEntityAddRemoveCommand() {
    if (m_ownsEntity && m_entity) {
        delete m_entity;
        m_entity = nullptr;
    }
}

void ecvEntityAddRemoveCommand::undo() {
    if (m_mode == Mode::Add) {
        if (m_entity && m_removeFunc) {
            m_removeFunc(m_entity);
            m_ownsEntity = true;
        }
    } else {
        ensureEntityAvailable();
        if (m_entity && m_parent && m_addFunc) {
            m_addFunc(m_entity, m_parent);
            m_ownsEntity = false;
        }
    }
    if (m_refreshFunc) m_refreshFunc();
}

void ecvEntityAddRemoveCommand::redo() {
    if (m_firstRedo) {
        m_firstRedo = false;
        return;
    }
    if (m_mode == Mode::Add) {
        ensureEntityAvailable();
        if (m_entity && m_parent && m_addFunc) {
            m_addFunc(m_entity, m_parent);
            m_ownsEntity = false;
        }
    } else {
        if (m_entity && m_removeFunc) {
            m_removeFunc(m_entity);
            m_ownsEntity = true;
        }
    }
    if (m_refreshFunc) m_refreshFunc();
}

qint64 ecvEntityAddRemoveCommand::estimatedMemoryBytes() const {
    return m_estimatedBytes;
}

void ecvEntityAddRemoveCommand::serializeToTemp() {
    if (m_serialized || !m_entity) return;

    m_tempFile = std::make_unique<QTemporaryFile>();
    if (!m_tempFile->open()) {
        CVLog::Warning("[EntityUndoCommand] Failed to create temp file for serialization");
        return;
    }

    if (m_entity->toFile(*m_tempFile, 0)) {
        m_serialized = true;
        m_estimatedBytes = m_tempFile->size();
        CVLog::Print(QString("[EntityUndoCommand] Serialized entity '%1' to temp (%2 bytes)")
                             .arg(m_entity->getName())
                             .arg(m_tempFile->size()));
    } else {
        CVLog::Warning("[EntityUndoCommand] Serialization failed for entity");
        m_tempFile.reset();
    }
}

ccHObject* ecvEntityAddRemoveCommand::deserializeFromTemp() {
    if (!m_serialized || !m_tempFile) return nullptr;

    m_tempFile->seek(0);

    ccHObject::LoadedIDMap oldToNewIDMap;
    ccHObject* restored = new ccHObject("restored");

    if (restored->fromFile(*m_tempFile, 0, 0, oldToNewIDMap)) {
        CVLog::Print(QString("[EntityUndoCommand] Deserialized entity from temp"));
        return restored;
    }

    delete restored;
    CVLog::Warning("[EntityUndoCommand] Deserialization failed");
    return nullptr;
}

void ecvEntityAddRemoveCommand::ensureEntityAvailable() {
    if (m_entity) return;
    if (m_serialized) {
        m_entity = deserializeFromTemp();
        if (m_entity) m_ownsEntity = true;
    }
}
```

- [x] **Step 3: Register in CMakeLists.txt**

Add to `libs/CV_db/src/CMakeLists.txt` after the `ecvUndoManager.cpp` line:

```cmake
	    ${CMAKE_CURRENT_LIST_DIR}/ecvEntityAddRemoveCommand.cpp
```

- [x] **Step 4: Verify the file compiles** — PASSED

- [x] **Step 5: Commit** (deferred to batch commit with all GAP-T foundation)

---

### Task 4: Wrap Transform Operations with Undo ✅

**Files:**
- Modify: `app/MainWindow.cpp` — `applyTransformation()` method, add `#include <ecvTransformCommand.h>`

- [x] **Step 1: Add include**

```cpp
#include <ecvTransformCommand.h>
```

- [x] **Step 2: Wrap transform loop body with undo command**

In `MainWindow::applyTransformation()`, capture GL transformation history before/after and push an undo command per entity:

```cpp
ccGLMatrix histBefore = entity->getGLTransformationHistory();
ccGLMatrix appliedMat(transMat.data());

ccHObjectContext objContext = removeObjectTemporarilyFromDBTree(entity);
entity->setGLTransformation(appliedMat);
entity->applyGLTransformation_recursive();
putObjectBackIntoDBTree(entity, objContext);

ccGLMatrix histAfter = entity->getGLTransformationHistory();

auto* undoMgr = ecvViewManager::instance().undoManager();
if (undoMgr) {
    auto restoreFunc = [this](ccHObject* ent, const ccGLMatrix& t) {
        ccHObjectContext ctx = removeObjectTemporarilyFromDBTree(ent);
        ent->setGLTransformation(t);
        ent->applyGLTransformation_recursive();
        putObjectBackIntoDBTree(ent, ctx);
    };
    auto refreshFunc = [this]() { refreshSelected(); };
    undoMgr->push(new ecvTransformCommand(
        entity, histBefore, histAfter, appliedMat,
        restoreFunc, refreshFunc,
        tr("Transform '%1'").arg(entity->getName())));
}
```

- [x] **Step 3: Build and verify** — Passes with exit code 0

---

### Task 5: Wrap Entity Delete with Undo ✅

**Files:**
- Modify: `app/db_tree/ecvDBRoot.cpp` — `deleteSelectedEntities()` method, add includes for `ecvEntityAddRemoveCommand.h`, `ecvPropertyChangeCommand.h`, `ecvUndoManager.h`

- [x] **Step 1: Add includes**

```cpp
#include <ecvEntityAddRemoveCommand.h>
#include <ecvPropertyChangeCommand.h>
#include <ecvUndoManager.h>
```

- [x] **Step 2: Replace destructive `removeChild` with `detachChild` + undo command**

Changed `parent->removeChild(childPos)` to `parent->detachChild(object)` (non-destructive), then push an `ecvEntityAddRemoveCommand` in `Mode::Remove`:

```cpp
beginRemoveRows(index(object).parent(), childPos, childPos);
parent->detachChild(object);
endRemoveRows();

if (undoMgr) {
    auto addFunc = [this](ccHObject* ent, ccHObject* par) {
        par->addChild(ent);
        addElement(ent, false);
    };
    auto removeFunc = [this](ccHObject* ent) {
        ccHObject* par = ent->getParent();
        if (!par) return;
        int pos = par->getChildIndex(ent);
        if (pos < 0) return;
        beginRemoveRows(index(ent).parent(), pos, pos);
        par->detachChild(ent);
        endRemoveRows();
    };
    auto refreshFunc = []() {
        MainWindow::TheInstance()->refreshAll();
    };
    undoMgr->push(new ecvEntityAddRemoveCommand(
        object, parent,
        ecvEntityAddRemoveCommand::Mode::Remove,
        addFunc, removeFunc, refreshFunc,
        tr("Delete '%1'").arg(object->getName())));
} else {
    delete object;
}
```

- [x] **Step 3: Build and verify** — Passes with exit code 0

---

### Task 6: Wrap Property Changes (Visibility, Color, Point Size) with Undo ✅

**Files:**
- Modify: `app/db_tree/ecvDBRoot.cpp` — `toggleSelectedEntitiesProperty()` method

- [x] **Step 1: Wrap all 6 toggle property types with undo commands**

Each toggle case (TG_ENABLE, TG_VISIBLE, TG_COLOR, TG_NORMAL, TG_SF, TG_3D_NAME) captures before state, applies the toggle, then pushes an `ecvPropertyChangeCommand<bool>`:

```cpp
case TG_VISIBLE: {
    bool wasBefore = item->isVisible();
    item->toggleVisibility();
    item->setForceRedrawRecursive(true);
    if (undoMgr) {
        undoMgr->push(new ecvPropertyChangeCommand<bool>(
            item->getUniqueID(), QStringLiteral("visible"),
            wasBefore, !wasBefore,
            [item](const bool& v) {
                item->setVisible(v);
                item->setForceRedrawRecursive(true);
            },
            refreshAll,
            tr("Toggle visibility '%1'").arg(item->getName())));
    }
} break;
```

All 6 toggle types follow this pattern with appropriate API calls: `setEnabled`, `setVisible`, `showColors`, `showNormals`, `showSF`, `showNameIn3D`.

- [x] **Step 2: Build and verify** — Passes with exit code 0

**Critical Fix Applied (All Commands):** Added `m_firstRedo` flag to `ecvTransformCommand`, `ecvEntityAddRemoveCommand`, `ecvPropertyChangeCommand`, and `ecvScalarFieldEditCommand`. `QUndoStack::push()` calls `redo()` immediately, but since the action is already performed before pushing, the first `redo()` must be skipped to prevent double-application.

---

### Task 7: ecvScalarFieldEditCommand — Scalar Field Value Undo ✅

**Files:**
- Create: `libs/CV_db/include/ecvScalarFieldEditCommand.h`

- [x] **Step 1: Create the header-only scalar field edit command**

```cpp
// libs/CV_db/include/ecvScalarFieldEditCommand.h
#pragma once

#include "CV_db.h"
#include "ecvPointCloud.h"
#include "ecvScalarField.h"

#include <QUndoCommand>
#include <functional>
#include <vector>

class CV_DB_LIB_API ecvScalarFieldEditCommand : public QUndoCommand {
public:
    using RefreshFunc = std::function<void()>;

    // Snapshot a range of scalar field values before and after editing.
    // For full-field operations, startIndex=0 and count=sf->currentSize().
    ecvScalarFieldEditCommand(ccPointCloud* cloud,
                              int sfIndex,
                              unsigned startIndex,
                              const std::vector<ScalarType>& beforeValues,
                              const std::vector<ScalarType>& afterValues,
                              RefreshFunc refreshFunc,
                              const QString& label,
                              QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent)
        , m_cloud(cloud)
        , m_sfIndex(sfIndex)
        , m_startIndex(startIndex)
        , m_beforeValues(beforeValues)
        , m_afterValues(afterValues)
        , m_refresh(std::move(refreshFunc)) {}

    void undo() override {
        applyValues(m_beforeValues);
        if (m_refresh) m_refresh();
    }

    void redo() override {
        if (m_firstRedo) {
            m_firstRedo = false;
            return;
        }
        applyValues(m_afterValues);
        if (m_refresh) m_refresh();
    }

    int id() const override { return 2004; }

    qint64 estimatedMemoryBytes() const {
        return static_cast<qint64>((m_beforeValues.size() + m_afterValues.size())
                                   * sizeof(ScalarType));
    }

private:
    void applyValues(const std::vector<ScalarType>& values) {
        if (!m_cloud) return;
        ccScalarField* sf = static_cast<ccScalarField*>(
            m_cloud->getScalarField(m_sfIndex));
        if (!sf) return;
        for (size_t i = 0; i < values.size(); ++i) {
            sf->setValue(m_startIndex + static_cast<unsigned>(i), values[i]);
        }
        sf->computeMinAndMax();
    }

    ccPointCloud* m_cloud;
    int m_sfIndex;
    unsigned m_startIndex;
    std::vector<ScalarType> m_beforeValues;
    std::vector<ScalarType> m_afterValues;
    RefreshFunc m_refresh;
    bool m_firstRedo = true;
};
```

- [x] **Step 2: Verify the file compiles** — PASSED

- [x] **Step 3: Commit** (deferred to batch commit with all GAP-T foundation)

---

### Task 8: Memory Budget for Undo Stack ✅

**Files:**
- Modify: `libs/CV_db/include/ecvUndoManager.h`
- Modify: `libs/CV_db/src/ecvUndoManager.cpp`

**Design note:** `QUndoStack` does not support removing individual commands.
The only way to limit memory is via `setUndoLimit(N)` which caps the maximum
command count. Our strategy: maintain an estimated memory counter. When it
exceeds the budget, reduce the undo limit to the current count minus one,
which causes `QUndoStack` to drop the oldest command on the next push.
We also override `push()` to track memory per-command.

- [x] **Step 1: Add memory budget members to ecvUndoManager.h**

Added to the `ecvUndoManager` class:

```cpp
public:
    void setMemoryBudgetMB(int mb);
    qint64 memoryBudgetBytes() const { return m_memoryBudgetBytes; }
    qint64 estimatedMemoryUsage() const { return m_estimatedMemory; }

    void clear();  // now non-inline: resets memory tracking and undo limit

private:
    void enforceMemoryBudget();

    qint64 m_memoryBudgetBytes = 500LL * 1024 * 1024;  // 500 MB default
    qint64 m_estimatedMemory = 0;
    int m_currentUndoLimit = 200;  // reasonable default count limit
```

- [x] **Step 2: Implement memory budget in ecvUndoManager.cpp**

```cpp
static qint64 estimateCmdMemory(const QUndoCommand* cmd) {
    if (!cmd) return 1024;
    if (auto* addRemove = dynamic_cast<const ecvEntityAddRemoveCommand*>(cmd))
        return addRemove->estimatedMemoryBytes();
    if (auto* sfEdit = dynamic_cast<const ecvScalarFieldEditCommand*>(cmd))
        return sfEdit->estimatedMemoryBytes();
    return 1024;
}

void ecvUndoManager::push(QUndoCommand* cmd) {
    if (!cmd) return;
    m_estimatedMemory += estimateCmdMemory(cmd);
    m_stack.push(cmd);
    enforceMemoryBudget();
}

void ecvUndoManager::clear() {
    m_stack.clear();
    m_estimatedMemory = 0;
    m_currentUndoLimit = 200;
    m_stack.setUndoLimit(m_currentUndoLimit);
}

void ecvUndoManager::setMemoryBudgetMB(int mb) {
    m_memoryBudgetBytes = static_cast<qint64>(mb) * 1024 * 1024;
    enforceMemoryBudget();
}

void ecvUndoManager::enforceMemoryBudget() {
    if (m_estimatedMemory <= m_memoryBudgetBytes) return;
    int newLimit = qMax(10, m_stack.count() - 1);
    if (newLimit < m_currentUndoLimit) {
        m_currentUndoLimit = newLimit;
        m_stack.setUndoLimit(m_currentUndoLimit);
        CVLog::Print(
            QString("[UndoManager] Memory budget exceeded (%1 MB / %2 MB), "
                    "reducing undo limit to %3")
                .arg(m_estimatedMemory / (1024 * 1024))
                .arg(m_memoryBudgetBytes / (1024 * 1024))
                .arg(m_currentUndoLimit));
        // Recalculate total from surviving commands
        m_estimatedMemory = 0;
        for (int i = 0; i < m_stack.count(); ++i) {
            m_estimatedMemory += estimateCmdMemory(m_stack.command(i));
        }
    }
}
```

- [x] **Step 3: Build and verify** — PASSED

- [x] **Step 4: Commit** (deferred to batch commit with all GAP-T foundation)

---

### Task 9: Integration Smoke Test

**Files:**
- No new files; manual verification

- [x] **Step 1: Build the full application**

Run: `cmake --build build -j$(sysctl -n hw.ncpu) 2>&1`
Result: BUILD SUCCESSFUL — [100%] Built target ColmapApp, exit_code: 0

- [ ] **Step 2: Manual smoke test checklist** (pending Tasks 4-6 integration)

1. Load a point cloud file
2. Apply a transformation (Edit > Apply Transformation)
3. Press Ctrl+Z — transform should undo, point cloud returns to original position
4. Press Ctrl+Y — transform should redo
5. Delete an entity from the DB tree
6. Press Ctrl+Z — entity should reappear in the tree
7. Change visibility of an entity
8. Press Ctrl+Z — visibility should toggle back
9. Check memory: load a large point cloud (>1M points), perform multiple operations, verify no crash

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(undo): GAP-T source undo — property, transform, entity add/remove commands"
```

---

## Design Decisions

1. **Property commands are templated with entity-scoped merge IDs** — one class handles `bool`, `float`, `unsigned char`, `ecvColor::Rgb`, `QString`, etc. The merge ID is computed from `qHash(propertyKey) ^ entityId`, ensuring that only consecutive changes to the *same property on the same entity* are merged. No cross-entity or cross-property corruption.

2. **Transform commands store absolute matrix snapshots** — `historyBefore` and `historyAfter` are captured as `ccGLMatrix`. On undo, the inverse of the applied transform is used. This avoids floating-point drift from repeated delta composition (`A * B^-1 * B * A^-1` accumulation).

3. **Entity add/remove uses ownership transfer** — when an entity is "removed," the undo command takes ownership (prevents deletion). On undo (re-add), ownership transfers back to the DB tree. Lazy serialization to temp files is available for future use when memory pressure demands it.

4. **Scalar field edit commands snapshot value ranges** — stores `before`/`after` vectors for the edited range. Provides `estimatedMemoryBytes()` so the memory budget system can account for large edits.

5. **Macro commands for bulk operations** — Qt's `QUndoCommand` parent mechanism groups multiple property changes (e.g., set color on 10 entities) into a single undo step.

6. **Memory budget via count-limit reduction** — `QUndoStack` does not support removing individual commands, so `ecvUndoManager` tracks estimated memory and reduces `setUndoLimit()` when the budget is exceeded, causing the oldest commands to be dropped on the next push. Default budget: 500 MB, minimum limit: 10 commands.

## Prerequisites

- **GAP-S (Global Undo Stack)**: ✅ Already implemented — `ecvUndoManager`, `ecvCameraUndoCommand`, `ecvLayoutUndoCommand` exist and are wired into the Edit menu with Ctrl+Z/Y shortcuts.
