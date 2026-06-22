# GAP-S: Global Undo Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify layout undo and camera undo into a single `QUndoStack` managed by `ecvUndoManager`, wired to Edit menu Ctrl+Z / Ctrl+Y, replacing the two separate internal stacks.

**Architecture:** Create `ecvUndoManager` singleton (owned by `ecvViewManager`) wrapping a `QUndoStack`. Two `QUndoCommand` subclasses (`ecvLayoutUndoCommand`, `ecvCameraUndoCommand`) capture before/after state. Existing `ecvViewLayoutProxy` internal stack and `VtkVis` camera deque are replaced by pushes to the global stack.

**Tech Stack:** Qt 5/6 (`QUndoStack`, `QUndoCommand`, `QAction`), C++17, CMake

---

## File Structure

| Operation | File | Responsibility |
|-----------|------|---------------|
| Create | `libs/CV_db/include/ecvUndoManager.h` | Singleton owning `QUndoStack*`, convenience push methods |
| Create | `libs/CV_db/src/ecvUndoManager.cpp` | Implementation |
| Create | `libs/CV_db/include/ecvLayoutUndoCommand.h` | `QUndoCommand` storing layout before/after snapshots |
| Create | `libs/CV_db/include/ecvCameraUndoCommand.h` | `QUndoCommand` storing camera before/after params |
| Modify | `libs/CV_db/src/CMakeLists.txt` | Add `ecvUndoManager.cpp` |
| Modify | `libs/CV_db/include/ecvViewManager.h` | Forward-declare and store `ecvUndoManager*` |
| Modify | `libs/CV_db/src/ecvViewManager.cpp` | Create/destroy `ecvUndoManager` |
| Modify | `libs/CV_db/include/ecvViewLayoutProxy.h` | Remove internal undo stack members; add `setUndoManager()` |
| Modify | `libs/CV_db/src/ecvViewLayoutProxy.cpp` | Route `beginUndoSet`/`endUndoSet` through global `ecvUndoManager` |
| Modify | `libs/VtkEngine/Visualization/VtkVis.h` | Add `setUndoManager()`; keep `CameraParams` public |
| Modify | `libs/VtkEngine/Visualization/VtkVis.cpp` | Route `pushCameraState` through global stack |
| Modify | `app/MainWindow.cpp` | Add Edit > Undo/Redo (Ctrl+Z/Y) wired to `ecvUndoManager` |

---

## Task 1: Create `ecvUndoManager` ✅

**Files:**
- Create: `libs/CV_db/include/ecvUndoManager.h`
- Create: `libs/CV_db/src/ecvUndoManager.cpp`
- Modify: `libs/CV_db/src/CMakeLists.txt`

- [x] **Step 1: Create the header**

```cpp
// libs/CV_db/include/ecvUndoManager.h
#pragma once

#include <QObject>
#include <QUndoStack>

class QAction;

class ecvUndoManager : public QObject {
    Q_OBJECT

public:
    explicit ecvUndoManager(QObject* parent = nullptr);
    ~ecvUndoManager() override;

    QUndoStack* undoStack() { return &m_stack; }
    const QUndoStack* undoStack() const { return &m_stack; }

    void push(QUndoCommand* cmd);

    bool canUndo() const { return m_stack.canUndo(); }
    bool canRedo() const { return m_stack.canRedo(); }
    QString undoText() const { return m_stack.undoText(); }
    QString redoText() const { return m_stack.redoText(); }

    void undo() { m_stack.undo(); }
    void redo() { m_stack.redo(); }
    void clear() { m_stack.clear(); }

    QAction* createUndoAction(QObject* parent,
                              const QString& prefix = QString()) const;
    QAction* createRedoAction(QObject* parent,
                              const QString& prefix = QString()) const;

    void setUndoLimit(int limit) { m_stack.setUndoLimit(limit); }

private:
    QUndoStack m_stack;
};
```

- [x] **Step 2: Create the implementation**

```cpp
// libs/CV_db/src/ecvUndoManager.cpp
#include "ecvUndoManager.h"

ecvUndoManager::ecvUndoManager(QObject* parent)
    : QObject(parent) {
    m_stack.setUndoLimit(64);
}

ecvUndoManager::~ecvUndoManager() = default;

void ecvUndoManager::push(QUndoCommand* cmd) {
    if (cmd) {
        m_stack.push(cmd);
    }
}

QAction* ecvUndoManager::createUndoAction(QObject* parent,
                                          const QString& prefix) const {
    return m_stack.createUndoAction(parent, prefix);
}

QAction* ecvUndoManager::createRedoAction(QObject* parent,
                                          const QString& prefix) const {
    return m_stack.createRedoAction(parent, prefix);
}
```

- [x] **Step 3: Add to CMake**

In `libs/CV_db/src/CMakeLists.txt`, add `ecvUndoManager.cpp` after the `ecvViewManager.cpp` line:

```cmake
	    ${CMAKE_CURRENT_LIST_DIR}/ecvViewManager.cpp
	    ${CMAKE_CURRENT_LIST_DIR}/ecvUndoManager.cpp
```

- [x] **Step 4: Build to verify compilation**

```bash
cd /Users/asher/develop/code/github/ACloudViewer/build_app
make -j48 CV_DB_LIB 2>&1 | grep -E "error:|Built target"
```

Expected: `Built target CV_DB_LIB` with no errors.

- [x] **Step 5: Commit**

```bash
git add libs/CV_db/include/ecvUndoManager.h libs/CV_db/src/ecvUndoManager.cpp libs/CV_db/src/CMakeLists.txt
git commit -m "feat(undo): add ecvUndoManager with QUndoStack"
```

---

## Task 2: Wire `ecvUndoManager` into `ecvViewManager` ✅

**Files:**
- Modify: `libs/CV_db/include/ecvViewManager.h`
- Modify: `libs/CV_db/src/ecvViewManager.cpp`

- [x] **Step 1: Add forward declaration and accessor to header**

In `libs/CV_db/include/ecvViewManager.h`, add the forward declaration near the top (after existing forward declarations):

```cpp
class ecvUndoManager;
```

Add a public accessor method (near `displayTools()` accessor):

```cpp
    ecvUndoManager* undoManager();
    const ecvUndoManager* undoManager() const;
```

Add a private member (near `m_displayTools`):

```cpp
    ecvUndoManager* m_undoManager = nullptr;
```

- [x] **Step 2: Create and destroy in `ecvViewManager.cpp`**

In `libs/CV_db/src/ecvViewManager.cpp`, add include:

```cpp
#include "ecvUndoManager.h"
```

In the constructor body (after `m_displayTools` creation if any, or at the end of the constructor):

```cpp
    m_undoManager = new ecvUndoManager(this);
```

Add accessor implementations:

```cpp
ecvUndoManager* ecvViewManager::undoManager() {
    return m_undoManager;
}

const ecvUndoManager* ecvViewManager::undoManager() const {
    return m_undoManager;
}
```

No explicit delete needed — `m_undoManager` is a `QObject` child, destroyed by Qt parent ownership.

- [x] **Step 3: Build to verify**

```bash
cd /Users/asher/develop/code/github/ACloudViewer/build_app
make -j48 CV_DB_LIB 2>&1 | grep -E "error:|Built target"
```

Expected: `Built target CV_DB_LIB` with no errors.

- [x] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvViewManager.h libs/CV_db/src/ecvViewManager.cpp
git commit -m "feat(undo): wire ecvUndoManager into ecvViewManager"
```

---

## Task 3: Create `ecvLayoutUndoCommand` ✅

**Files:**
- Create: `libs/CV_db/include/ecvLayoutUndoCommand.h`

This is a header-only `QUndoCommand` that captures a `ecvViewLayoutProxy` state snapshot before and after a mutation.

- [x] **Step 1: Create the command header**

`ecvViewLayoutProxy` currently has a private `Snapshot` struct. We need to expose a way for external commands to capture and restore state. The simplest approach: use `saveState()` / `loadState()` which work with `QJsonObject` and are already public.

```cpp
// libs/CV_db/include/ecvLayoutUndoCommand.h
#pragma once

#include <QJsonObject>
#include <QUndoCommand>

class ecvViewLayoutProxy;

class ecvLayoutUndoCommand : public QUndoCommand {
public:
    ecvLayoutUndoCommand(ecvViewLayoutProxy* layout,
                         const QJsonObject& beforeState,
                         const QJsonObject& afterState,
                         const QString& label,
                         QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent)
        , m_layout(layout)
        , m_before(beforeState)
        , m_after(afterState) {}

    void undo() override;
    void redo() override;

    int id() const override { return 1001; }

private:
    ecvViewLayoutProxy* m_layout;
    QJsonObject m_before;
    QJsonObject m_after;
};
```

- [x] **Step 2: Add implementation to `ecvUndoManager.cpp`**

Add to `libs/CV_db/src/ecvUndoManager.cpp` (or create a separate `.cpp` — but since it's tiny, co-locate):

```cpp
#include "ecvLayoutUndoCommand.h"
#include "ecvViewLayoutProxy.h"

void ecvLayoutUndoCommand::undo() {
    if (m_layout) {
        m_layout->loadState(m_before);
    }
}

void ecvLayoutUndoCommand::redo() {
    if (m_layout) {
        m_layout->loadState(m_after);
    }
}
```

- [x] **Step 3: Build to verify**

```bash
cd /Users/asher/develop/code/github/ACloudViewer/build_app
make -j48 CV_DB_LIB 2>&1 | grep -E "error:|Built target"
```

- [x] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvLayoutUndoCommand.h libs/CV_db/src/ecvUndoManager.cpp
git commit -m "feat(undo): add ecvLayoutUndoCommand"
```

---

## Task 4: Create `ecvCameraUndoCommand` ✅

**Files:**
- Create: `libs/CV_db/include/ecvCameraUndoCommand.h`

- [x] **Step 1: Create the command header**

`VtkVis::CameraParams` is the state struct. `VtkVis` has public `getCamera(int viewport)` and we need a way to apply params back. Currently `applyCameraState` is a file-local static in `VtkVis.cpp`. We'll add a public `setCameraState(const CameraParams&)` method in Task 6.

```cpp
// libs/CV_db/include/ecvCameraUndoCommand.h
#pragma once

#include <QUndoCommand>
#include <functional>

struct ecvCameraState {
    double pos[3] = {0, 0, 0};
    double focal[3] = {0, 0, 0};
    double view[3] = {0, 0, 1};
    double clip[2] = {0.01, 1000};
    double fovy = 0.8575;
    bool parallelProjection = false;
    double parallelScale = 1.0;
};

class ecvCameraUndoCommand : public QUndoCommand {
public:
    using ApplyFunc = std::function<void(const ecvCameraState&)>;

    ecvCameraUndoCommand(const ecvCameraState& before,
                         const ecvCameraState& after,
                         ApplyFunc applyFunc,
                         const QString& label = QStringLiteral("Camera"),
                         QUndoCommand* parent = nullptr)
        : QUndoCommand(label, parent)
        , m_before(before)
        , m_after(after)
        , m_apply(std::move(applyFunc)) {}

    void undo() override {
        if (m_apply) m_apply(m_before);
    }

    void redo() override {
        if (m_apply) m_apply(m_after);
    }

    int id() const override { return 1002; }

    bool mergeWith(const QUndoCommand* other) override {
        if (other->id() != id()) return false;
        auto* o = static_cast<const ecvCameraUndoCommand*>(other);
        m_after = o->m_after;
        return true;
    }

private:
    ecvCameraState m_before;
    ecvCameraState m_after;
    ApplyFunc m_apply;
};
```

The `mergeWith` override is important: rapid camera movements produce many intermediate states; merging collapses them into a single before→after command so Ctrl+Z doesn't step through 50 micro-rotations.

- [x] **Step 2: Build to verify header compiles**

```bash
cd /Users/asher/develop/code/github/ACloudViewer/build_app
make -j48 CV_DB_LIB 2>&1 | grep -E "error:|Built target"
```

- [x] **Step 3: Commit**

```bash
git add libs/CV_db/include/ecvCameraUndoCommand.h
git commit -m "feat(undo): add ecvCameraUndoCommand with merge support"
```

---

## Task 5: Migrate `ecvViewLayoutProxy` to use global stack ✅

**Files:**
- Modify: `libs/CV_db/include/ecvViewLayoutProxy.h`
- Modify: `libs/CV_db/src/ecvViewLayoutProxy.cpp`

The goal: `beginUndoSet()` captures a `saveState()` snapshot. `endUndoSet()` captures a second snapshot and pushes an `ecvLayoutUndoCommand` to the global `ecvUndoManager`. The internal `m_undoStack` / `m_redoStack` are removed.

- [x] **Step 1: Update the header**

In `libs/CV_db/include/ecvViewLayoutProxy.h`:

Remove these private members:
```cpp
    QList<Snapshot> m_undoStack;
    QList<Snapshot> m_redoStack;
```

Add forward declaration and setter:
```cpp
class ecvUndoManager;
```

Add public method:
```cpp
    void setUndoManager(ecvUndoManager* mgr);
```

Add private member:
```cpp
    ecvUndoManager* m_undoManager = nullptr;
    QJsonObject m_pendingBeforeState;
```

Remove `canUndo()`, `canRedo()`, `undoLabel()`, `redoLabel()`, `undo()`, `redo()` declarations — these are now handled by the global stack. Keep `beginUndoSet()` and `endUndoSet()`. Keep `undoRedoChanged()` signal.

Remove `static constexpr int MaxUndoDepth = 32;`.

Remove the old `Snapshot` struct and `takeSnapshot()` / `applySnapshot()` private helpers — replaced by `saveState()` / `loadState()` which already exist.

- [x] **Step 2: Update the implementation**

In `libs/CV_db/src/ecvViewLayoutProxy.cpp`:

Add includes:
```cpp
#include "ecvUndoManager.h"
#include "ecvLayoutUndoCommand.h"
```

Replace `beginUndoSet`:
```cpp
void ecvViewLayoutProxy::beginUndoSet(const QString& label) {
    if (m_undoNesting == 0) {
        m_pendingLabel = label;
        m_pendingBeforeState = saveState();
    }
    ++m_undoNesting;
}
```

Replace `endUndoSet`:
```cpp
void ecvViewLayoutProxy::endUndoSet() {
    if (m_undoNesting <= 0) return;
    --m_undoNesting;
    if (m_undoNesting == 0 && m_undoManager) {
        QJsonObject afterState = saveState();
        if (m_pendingBeforeState != afterState) {
            m_undoManager->push(new ecvLayoutUndoCommand(
                    this, m_pendingBeforeState, afterState, m_pendingLabel));
        }
    }
}
```

Add setter:
```cpp
void ecvViewLayoutProxy::setUndoManager(ecvUndoManager* mgr) {
    m_undoManager = mgr;
}
```

Remove the implementations of: `canUndo()`, `canRedo()`, `undoLabel()`, `redoLabel()`, `undo()`, `redo()`, `takeSnapshot()`, `applySnapshot()`.

Update `reset()` to not clear internal stacks (they no longer exist):
```cpp
void ecvViewLayoutProxy::reset() {
    m_tree.clear();
    m_tree.resize(1);
    m_maximizedCell = -1;
    m_undoNesting = 0;
    emit layoutChanged();
}
```

Add `m_pendingLabel` as a private `QString` member in the header.

- [x] **Step 3: Wire layout to undo manager in `ecvViewManager`**

In `libs/CV_db/src/ecvViewManager.cpp`, in `registerLayout()`:

```cpp
void ecvViewManager::registerLayout(ecvViewLayoutProxy* layout) {
    // ... existing code ...
    if (layout && m_undoManager) {
        layout->setUndoManager(m_undoManager);
    }
}
```

- [x] **Step 4: Build to verify**

```bash
cd /Users/asher/develop/code/github/ACloudViewer/build_app
make -j48 CV_DB_LIB 2>&1 | grep -E "error:|Built target"
```

Fix any compilation errors from removed declarations.

- [x] **Step 5: Commit**

```bash
git add libs/CV_db/include/ecvViewLayoutProxy.h libs/CV_db/src/ecvViewLayoutProxy.cpp libs/CV_db/src/ecvViewManager.cpp
git commit -m "refactor(undo): migrate layout undo to global ecvUndoManager"
```

---

## Task 6: Migrate `VtkVis` camera undo to use global stack ✅

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkVis.h`
- Modify: `libs/VtkEngine/Visualization/VtkVis.cpp`

- [x] **Step 1: Add `setUndoManager` and `applyCameraState` public API to header**

In `libs/VtkEngine/Visualization/VtkVis.h`:

Add forward declaration:
```cpp
class ecvUndoManager;
```

Add public methods:
```cpp
    void setUndoManager(ecvUndoManager* mgr);
    void applyCameraState(const CameraParams& state, int viewport = 0);
```

Add private member:
```cpp
    ecvUndoManager* m_undoManager = nullptr;
```

Keep existing `canCameraUndo()`, `canCameraRedo()`, `cameraUndo()`, `cameraRedo()` for per-view toolbar buttons (they fall back to the internal deque if no global manager is set).

- [x] **Step 2: Update implementation**

In `libs/VtkEngine/Visualization/VtkVis.cpp`:

Add includes:
```cpp
#include "ecvUndoManager.h"
#include "ecvCameraUndoCommand.h"
```

Add setter:
```cpp
void VtkVis::setUndoManager(ecvUndoManager* mgr) {
    m_undoManager = mgr;
}
```

Add public `applyCameraState`:
```cpp
void VtkVis::applyCameraState(const CameraParams& state, int viewport) {
    m_inCameraUndoRedo = true;
    vtkSmartPointer<vtkCamera> cam = getVtkCamera(viewport);
    if (cam) {
        cam->SetPosition(state.pos);
        cam->SetFocalPoint(state.focal);
        cam->SetViewUp(state.view);
        cam->SetClippingRange(state.clip);
        cam->SetViewAngle(state.fovy * 180.0 / M_PI);
        cam->SetParallelProjection(state.parallelProjection ? 1 : 0);
        cam->SetParallelScale(state.parallelScale);
        UpdateScreen();
    }
    m_inCameraUndoRedo = false;
}
```

Modify `pushCameraState()` to also push to global stack:
```cpp
void VtkVis::pushCameraState() {
    CameraParams state = getCamera(0);
    if (!m_cameraUndoStack.empty()) {
        const auto& top = m_cameraUndoStack.back();
        auto eq = [](const double* a, const double* b, int n) {
            for (int i = 0; i < n; ++i)
                if (std::abs(a[i] - b[i]) > 1e-12) return false;
            return true;
        };
        if (eq(top.pos, state.pos, 3) && eq(top.focal, state.focal, 3) &&
            eq(top.view, state.view, 3))
            return;
    }

    CameraParams beforeState = m_cameraUndoStack.empty()
                                       ? state
                                       : m_cameraUndoStack.back();

    m_cameraUndoStack.push_back(state);
    while (static_cast<int>(m_cameraUndoStack.size()) > CAMERA_STACK_DEPTH) {
        m_cameraUndoStack.pop_front();
    }
    m_cameraRedoStack.clear();

    if (m_undoManager) {
        ecvCameraState before, after;
        auto copy = [](const CameraParams& src, ecvCameraState& dst) {
            std::copy(std::begin(src.pos), std::end(src.pos), dst.pos);
            std::copy(std::begin(src.focal), std::end(src.focal), dst.focal);
            std::copy(std::begin(src.view), std::end(src.view), dst.view);
            std::copy(std::begin(src.clip), std::end(src.clip), dst.clip);
            dst.fovy = src.fovy;
            dst.parallelProjection = src.parallelProjection;
            dst.parallelScale = src.parallelScale;
        };
        copy(beforeState, before);
        copy(state, after);

        auto* self = this;
        m_undoManager->push(new ecvCameraUndoCommand(
                before, after,
                [self](const ecvCameraState& s) {
                    CameraParams p;
                    std::copy(std::begin(s.pos), std::end(s.pos), p.pos);
                    std::copy(std::begin(s.focal), std::end(s.focal), p.focal);
                    std::copy(std::begin(s.view), std::end(s.view), p.view);
                    std::copy(std::begin(s.clip), std::end(s.clip), p.clip);
                    p.fovy = s.fovy;
                    p.parallelProjection = s.parallelProjection;
                    p.parallelScale = s.parallelScale;
                    self->applyCameraState(p);
                },
                QStringLiteral("Camera")));
    }
}
```

- [x] **Step 3: Wire in `ecvViewManager` view registration**

In `libs/CV_db/src/ecvViewManager.cpp`, wherever new views are registered (in `registerView` or after `VtkVis` creation), call:

```cpp
if (auto* glView = dynamic_cast<ecvGLView*>(view)) {
    if (auto* vis = glView->getVisualizer3D()) {
        vis->setUndoManager(m_undoManager);
    }
}
```

This requires including `ecvGLView.h` and checking if `ecvViewManager.cpp` can see `VtkVis`. If not, add the wiring in `MainWindow.cpp` where views are created and `VtkVis` is accessible.

- [x] **Step 4: Build to verify**

```bash
cd /Users/asher/develop/code/github/ACloudViewer/build_app
make -j48 CV_DB_LIB VtkEngine ACloudViewer 2>&1 | grep -E "error:|Built target"
```

- [x] **Step 5: Commit**

```bash
git add libs/VtkEngine/Visualization/VtkVis.h libs/VtkEngine/Visualization/VtkVis.cpp libs/CV_db/src/ecvViewManager.cpp
git commit -m "refactor(undo): migrate camera undo to global ecvUndoManager"
```

---

## Task 7: Wire Edit menu Undo/Redo in MainWindow ✅

**Files:**
- Modify: `app/MainWindow.cpp`

- [x] **Step 1: Add global Undo/Redo to Edit menu**

Currently layout undo is in the Display menu with Ctrl+Shift+Z/Y. The global undo should use Edit menu with standard Ctrl+Z / Ctrl+Y.

In the `MainWindow` constructor, in the menu setup section (where the Edit menu is built), add:

```cpp
    // Global Undo / Redo — wired to ecvUndoManager
    auto* undoMgr = ecvViewManager::instance().undoManager();
    if (undoMgr) {
        QAction* globalUndoAct =
                undoMgr->createUndoAction(this, tr("Undo"));
        globalUndoAct->setShortcut(QKeySequence::Undo);
        globalUndoAct->setIcon(
                QIcon(":/Resources/images/svg/pqUndo.svg"));
        editMenu->addAction(globalUndoAct);

        QAction* globalRedoAct =
                undoMgr->createRedoAction(this, tr("Redo"));
        globalRedoAct->setShortcut(QKeySequence::Redo);
        globalRedoAct->setIcon(
                QIcon(":/Resources/images/svg/pqRedo.svg"));
        editMenu->addAction(globalRedoAct);

        editMenu->addSeparator();
    }
```

Add include at the top of `MainWindow.cpp`:
```cpp
#include "ecvUndoManager.h"
```

- [x] **Step 2: Remove old layout undo from Display menu**

Find and remove the `undoLayoutAct` / `redoLayoutAct` block (lines ~1509–1570) in the Display menu since layout undo is now handled by the global Edit > Undo.

Alternatively, keep both — the Display menu shortcuts become Ctrl+Shift+Z/Y for layout-specific undo (if desired). If removing, delete the entire block from `displayMenu->addSeparator()` through the `firstLayout` connection block.

Decision: **Remove** — a single undo entry point is cleaner.

- [x] **Step 3: Update per-view camera undo buttons to reflect global stack state**

The per-view camera undo/redo toolbar buttons (lines ~2822–2859) currently poll `VtkVis::canCameraUndo()` on a timer. They should continue to work — they call `VtkVis::cameraUndo()` / `cameraRedo()` which use the internal deque. This is intentional: the per-view buttons provide **view-scoped** camera undo, while Ctrl+Z provides **global** undo across all action types.

No changes needed for per-view buttons.

- [x] **Step 4: Build to verify**

```bash
cd /Users/asher/develop/code/github/ACloudViewer/build_app
make -j48 ACloudViewer 2>&1 | grep -E "error:|Built target"
```

- [x] **Step 5: Manual verification**

1. Launch ACloudViewer
2. Split a view (Ctrl+Shift+S or right-click)
3. Press Ctrl+Z → the split should undo
4. Press Ctrl+Shift+Z → should redo (if kept) or Ctrl+Y
5. Rotate camera in a view
6. Press Ctrl+Z → camera should snap back to pre-rotation state
7. Verify per-view camera undo buttons still work independently

- [x] **Step 6: Commit**

```bash
git add app/MainWindow.cpp
git commit -m "feat(undo): wire global Undo/Redo to Edit menu (Ctrl+Z/Y)"
```

---

## Task 8: Update alignment documentation ✅

**Files:**
- Modify: `docs/user-guide/multi-window-paraview-alignment-design.md`

- [x] **Step 1: Update GAP-S status**

Change GAP-S from the current plan text to:

```markdown
### GAP-S: Global Undo Stack — ✅ ALIGNED

**实现摘要**:
- `ecvUndoManager` 单例 (由 `ecvViewManager` 管理) 持有 `QUndoStack`
- `ecvLayoutUndoCommand`: 捕获 layout `saveState()`/`loadState()` 前后快照
- `ecvCameraUndoCommand`: 捕获 camera params 前后状态，支持 `mergeWith`
- `ecvViewLayoutProxy` 已迁移至全局栈 (移除内部 `m_undoStack`/`m_redoStack`)
- `VtkVis` camera undo 同时推送到全局栈
- `MainWindow` Edit 菜单: Ctrl+Z / Ctrl+Y 绑定全局 Undo/Redo
```

- [x] **Step 2: Update alignment rate**

Change from `89/91 = 97.8%` to `90/91 = 98.9%`.

Update the prediction table.

- [x] **Step 3: Commit**

```bash
git add docs/user-guide/multi-window-paraview-alignment-design.md
git commit -m "docs: update alignment to 98.9% after GAP-S completion"
```

---

## Self-Review Checklist

1. **Spec coverage**: All items from the GAP-S design (§7 of alignment doc) are covered:
   - ✅ `ecvUndoManager` singleton with `QUndoStack` → Task 1-2
   - ✅ `ecvCameraUndoCommand` wrapping VtkVis state → Task 4, 6
   - ✅ `ecvLayoutUndoCommand` wrapping layout memento → Task 3, 5
   - ✅ Edit menu Ctrl+Z/Y → Task 7
   - ✅ Layout proxy migration → Task 5
   - ✅ VtkVis camera migration → Task 6
   - ✅ Documentation update → Task 8

2. **Placeholder scan**: No TBDs, TODOs, or "fill in later" present. All code blocks contain complete implementations.

3. **Type consistency**:
   - `ecvCameraState` (in command header) mirrors `VtkVis::CameraParams` fields
   - `ecvLayoutUndoCommand` uses `QJsonObject` via `saveState()`/`loadState()` — consistent with existing `ecvViewLayoutProxy` API
   - `ecvUndoManager::push(QUndoCommand*)` is used consistently across Tasks 5 and 6
