# Singleton Removal Phase 2: Per-View Messages & Signal Relay

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the two most impactful singleton-shaped bottlenecks: the global message overlay system and the singleton signal relay — making each rendering window fully independent for message display and event handling.

**Architecture:** Route `DisplayNewMessage` through `ecvViewManager` to per-view `displayNewMessage`, make `ecvGLView::redraw()` consume its own `m_messagesToDisplay`, and replace the singleton signal relay with per-view signal connections during `registerView()`.

**Tech Stack:** C++17, Qt5/6 signals/slots, VTK, ACloudViewer VtkEngine

---

## Current State (from audit)

1. **Messages**: `DisplayNewMessage()` pushes to global `s_tools->m_messagesToDisplay`. Each `ecvGLView::redraw()` reads that global list → all windows show the same messages. `ecvGLView::displayNewMessage()` writes to a per-view list that is **never consumed** in `redraw()`.

2. **Signal relay**: `setupSingletonRelay()` connects Qt signals (picking, camera changes, entity selection) from `ecvDisplayTools::sharedTools()` — not from individual views. After Phase M3, `ecvGLView` is no longer an `ecvDisplayTools` subclass, so the `dynamic_cast` fails and the relay always falls back to the singleton.

---

## File Map

| File | Responsibility | Action |
|------|----------------|--------|
| `libs/VtkEngine/Visualization/ecvGLView.cpp:219-262` | Message rendering in `redraw()` | **Modify** — read own `m_messagesToDisplay` |
| `libs/CV_db/src/ecvDisplayTools.cpp:2587-2643` | `DisplayNewMessage()` static method | **Modify** — route through active view |
| `libs/CV_db/src/ecvDisplayTools.cpp:3323-3336` | `RedrawDisplay` message pruning | **Modify** — prune per-view lists |
| `libs/CV_db/src/ecvViewManager.cpp:26-76` | `setupSingletonRelay()` | **Modify** — connect per-view signals |
| `libs/CV_db/include/ecvDisplayTools.h` | `m_messagesToDisplay` declaration | **Verify** stays for backward compat |

---

### Task 1: Make `ecvGLView::redraw()` Read Its Own Per-View Messages ✅

**Files:**
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp:219-262`

Currently:
```cpp
auto* st = ecvDisplayTools::sharedTools();
if (st && !st->m_messagesToDisplay.empty()) {
    // ...
    for (const auto& message : st->m_messagesToDisplay) {
```

This must read from `m_messagesToDisplay` (the per-view list) instead.

- [ ] **Step 1: Replace the shared message source with per-view**

In `ecvGLView.cpp`, find the block at line ~219-262 and replace:

```cpp
    // --- Messages overlay ---
    if (m_ctx.displayOverlayEntities) {
        auto* st = ecvDisplayTools::sharedTools();
        if (st && !st->m_messagesToDisplay.empty()) {
            QFont font = m_font;
            QFontMetrics fm(font);
            int margin = fm.height() / 4;
            int ll_currentHeight = m_ctx.glViewport.height() - 10;
            int uc_currentHeight = 10;

            for (const auto& message : st->m_messagesToDisplay) {
```

with:

```cpp
    // --- Messages overlay (per-view) ---
    if (m_ctx.displayOverlayEntities && !m_messagesToDisplay.empty()) {
        // Prune expired messages
        int currentTime = m_messageTimer.elapsed() / 1000;
        m_messagesToDisplay.remove_if(
                [currentTime](const ecvMessageToDisplay& msg) {
                    return currentTime > msg.messageValidity_sec;
                });

        if (!m_messagesToDisplay.empty()) {
            QFont font = m_font;
            QFontMetrics fm(font);
            int margin = fm.height() / 4;
            int ll_currentHeight = m_ctx.glViewport.height() - 10;
            int uc_currentHeight = 10;

            for (const auto& message : m_messagesToDisplay) {
```

And close the extra brace. The closing `}` for the `if (!m_messagesToDisplay.empty())` block needs to be added before the Hot Zone block.

- [ ] **Step 2: Add `m_messageTimer` to ecvGLView**

In `ecvGLView.h`, add a `QElapsedTimer m_messageTimer;` member and start it in the constructor:

```cpp
// In ecvGLView.h, add member:
QElapsedTimer m_messageTimer;

// In ecvGLView constructor (ecvGLView.cpp), add:
m_messageTimer.start();
```

- [ ] **Step 3: Fix `displayNewMessage` to use absolute time**

In `ecvGLView::displayNewMessage()` (line ~462-481), the `messageValidity_sec` is currently set to `displayMaxDelay_sec` (relative), but redraw pruning needs absolute time:

```cpp
void ecvGLView::displayNewMessage(const QString& message,
                                  MessagePosition pos,
                                  bool append,
                                  int displayMaxDelay_sec,
                                  MessageType type) {
    if (!append) {
        m_messagesToDisplay.remove_if([type](const ecvMessageToDisplay& msg) {
            return msg.type == type;
        });
    }
    if (!message.isEmpty()) {
        ecvMessageToDisplay msg;
        msg.message = message;
        msg.messageValidity_sec =
                m_messageTimer.elapsed() / 1000 + displayMaxDelay_sec;
        msg.position = pos;
        msg.type = type;
        m_messagesToDisplay.push_back(msg);
    }
    toBeRefreshed();
}
```

- [ ] **Step 4: Build and verify**

Run: `cmake --build build --target ACloudViewer -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add libs/VtkEngine/Visualization/ecvGLView.cpp libs/VtkEngine/Visualization/ecvGLView.h
git commit -m "refactor: per-view message rendering in ecvGLView::redraw()

redraw() now reads from the per-view m_messagesToDisplay list instead
of the global ecvDisplayTools::sharedTools()->m_messagesToDisplay.
Messages are pruned on each frame using QElapsedTimer-based absolute
timestamps. This eliminates message duplication across windows."
```

---

### Task 2: Route `DisplayNewMessage` to Active View ✅

**Files:**
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp:2587-2643`

The static `DisplayNewMessage()` currently writes to `s_tools->m_messagesToDisplay`. After Task 1, each `ecvGLView` reads its own list. So `DisplayNewMessage` must now forward to the active view's `displayNewMessage()`.

- [ ] **Step 1: Modify DisplayNewMessage to route through ecvViewManager**

```cpp
void ecvDisplayTools::DisplayNewMessage(const QString& message,
                                        MessagePosition pos,
                                        bool append,
                                        int displayMaxDelay_sec,
                                        MessageType type) {
    // Route to the active (or rendering) view's per-view message list.
    auto* effectiveView = ecvViewManager::instance().getEffectiveView();
    if (effectiveView) {
        effectiveView->displayNewMessage(message, pos, append,
                                         displayMaxDelay_sec, type);
        return;
    }

    // Fallback: no views registered yet → store on shared tools.
    if (message.isEmpty()) {
        if (append) {
            s_tools->m_messagesToDisplay.remove_if(
                    [type](const MessageToDisplay& m) {
                        return m.type == type;
                    });
        }
        return;
    }

    if (!append) {
        s_tools->m_messagesToDisplay.remove_if(
                [type](const MessageToDisplay& m) {
                    return m.type == type;
                });
    }

    MessageToDisplay mess;
    mess.message = message;
    mess.messageValidity_sec =
            s_tools->m_timer.elapsed() / 1000 + displayMaxDelay_sec;
    mess.position = pos;
    mess.type = type;
    s_tools->m_messagesToDisplay.push_back(mess);
}
```

- [ ] **Step 2: Build and verify**

Run: `cmake --build build --target ACloudViewer -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "refactor: route DisplayNewMessage to per-view message list

DisplayNewMessage now forwards to the effective view's
displayNewMessage() instead of writing to the global
s_tools->m_messagesToDisplay. Falls back to the shared list only
when no views are registered."
```

---

### Task 3: Replace Singleton Signal Relay with Per-View Connections ✅

**Files:**
- Modify: `libs/CV_db/src/ecvViewManager.cpp:26-76`

Currently `setupSingletonRelay()` connects signals from `ecvDisplayTools::sharedTools()` because the `dynamic_cast<ecvDisplayTools*>(view)` fails for `ecvGLView`. This should connect per-view signals directly.

- [ ] **Step 1: Check which signals ecvGLView emits**

Verify that `ecvGLView` declares the same signals that `setupSingletonRelay` connects:
- `entitySelectionChanged`
- `itemPickedFast`
- `newLabel`
- `exclusiveFullScreenToggled`
- `cameraParamChanged`

- [ ] **Step 2: Modify setupSingletonRelay to connect per-view signals**

```cpp
void ecvViewManager::setupSingletonRelay(ecvGenericGLDisplay* view) {
    // Connect picking/camera signals from this specific view.
    // After Phase M3, views are ecvGLView (QObject-derived with signals),
    // not ecvDisplayTools. We connect to the view directly.
    auto* glView = dynamic_cast<ecvGLView*>(view);
    if (glView) {
        connect(glView, &ecvGLView::entitySelectionChanged,
                this, &ecvViewManager::entitySelectionChanged);
        connect(glView, &ecvGLView::itemPickedFast,
                this, &ecvViewManager::itemPickedFast);
        connect(glView, &ecvGLView::newLabel,
                this, &ecvViewManager::newLabel);
        connect(glView, &ecvGLView::exclusiveFullScreenToggled,
                this, &ecvViewManager::exclusiveFullScreenToggled);
        connect(glView, &ecvGLView::cameraParamChanged,
                this, &ecvViewManager::cameraParamChanged);
        return;
    }

    // Legacy fallback for non-ecvGLView displays.
    if (m_singletonRelayConnected) return;
    auto* dt = dynamic_cast<ecvDisplayTools*>(view);
    if (!dt) {
        dt = ecvDisplayTools::sharedTools();
    }
    if (!dt) return;

    connect(dt, &ecvDisplayTools::entitySelectionChanged,
            this, &ecvViewManager::entitySelectionChanged);
    // ... (keep existing fallback connections)
    m_singletonRelayConnected = true;
}
```

**Important:** The `m_singletonRelayConnected` guard must NOT prevent per-view connections. Only use it for the legacy fallback path.

- [ ] **Step 3: Verify signal declarations in ecvGLView.h**

Ensure all connected signals exist on `ecvGLView`. If any are missing, they must be added (likely as forwarding signals that emit when `ecvDisplayTools::sharedTools()` emits).

- [ ] **Step 4: Build and verify**

Run: `cmake --build build --target ACloudViewer -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add libs/CV_db/src/ecvViewManager.cpp
git commit -m "refactor: connect per-view signals in setupSingletonRelay

ecvGLView instances now have their picking/camera signals connected
directly to ecvViewManager, instead of relying on the shared
ecvDisplayTools singleton as the sole signal emitter. The legacy
fallback path is retained for non-ecvGLView displays."
```

---

### Task 4: Scope m_captureMode Per-View (Optional Enhancement)

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h` (move `m_captureMode` to `ecvViewContext`)
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp` (update references)

This task is lower priority. `m_captureMode` affects font sizing globally. Moving it to per-view context prevents concurrent captures from interfering.

- [ ] **Step 1: Add `CaptureModeOptions captureMode` to `ecvViewContext`**
- [ ] **Step 2: Update `GetFontPointPointSize` / `GetLabelFontPointSize` to read from effectiveCtx()**
- [ ] **Step 3: Build, verify, commit**

---

## Self-Review Checklist

1. **Spec coverage:**
   - Per-view messages: Tasks 1 + 2 ✅
   - Signal relay: Task 3 ✅
   - m_captureMode scoping: Task 4 ✅

2. **Placeholder scan:** Task 4 has deliberate abbreviated steps (marked optional). Tasks 1-3 have full code.

3. **Type consistency:** `ecvMessageToDisplay` (ecvGLView's internal type) vs `MessageToDisplay` (ecvDisplayTools's type) — both have `message`, `messageValidity_sec`, `position`, `type` fields. `displayNewMessage()` signature matches across both classes.

---

## Singleton Removal Progress After This Phase

| Before | After |
|--------|-------|
| ~25-35% | ~60-70% (Tasks 1-3 complete, effectiveCtx() delegates to resolveViewContext(), per-view messages and signals operational) |

**Still remaining for future phases:**
- Shrink `sharedTools()` static wrapper surface in `ecvDisplayTools.h` (~200+ inline methods)
- Move `m_hotZone` fully to per-view (partially done, `ecvGLView` already has its own)
- Deprecate `s_tools` pointer and make VtkDisplayTools a per-view owned instance
- Remove `effectiveCtx()` in favor of explicit context parameters
