# Phase B: Render Pipeline — Eliminate Singleton Swap in Drawing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `ecvGLView`'s draw pipeline self-contained — `getContext()` reads only from `m_ctx`, `ScopedHotZoneRender` swaps are reduced, `RedrawDisplay()` delegates to per-view `redraw()`, and the singleton draw path is deprecated.

**Architecture:** Phase B is split into **low-risk immediate work** (Tasks 1-5) and **deferred higher-risk work** (Tasks 2b, 3b, 3c). The low-risk tasks localize `getContext`, reduce `ScopedHotZoneRender` swap surface area, delegate `RedrawDisplay`, remove `ScopedRenderOverride` from wheel events, and deprecate singleton draw paths. The deferred tasks (full `ScopedHotZoneRender` elimination, overlay migration, `beginPrimaryRender` deletion) are pushed to Phase C/E where their prerequisites are met.

**Tech Stack:** C++17, Qt 5/6, VTK, CMake

---

## Current State (from audit)

| Mechanism | Location | Count | Notes |
|---|---|---|---|
| `ScopedVisSwap` | (removed) | 0 | Already eliminated |
| `ScopedHotZoneRender` | `VtkDisplayTools.h/cpp`, `ecvGLView.cpp` | 8 def + 1 usage | Swaps 8 singleton fields |
| `ScopedRenderOverride` | `QVTKWidgetCustom.cpp:774` | 1 usage | `wheelEvent` only |
| `beginPrimaryRender/endPrimaryRender` | `VtkDisplayTools.h/cpp` | 4 refs | Used by `RedrawDisplay` |
| `ecvGLView::getContext` | `ecvGLView.cpp:329` | Hybrid | Calls singleton `GetContext(context)` then overrides |

## Key files

| File | Role |
|---|---|
| `libs/VtkEngine/Visualization/ecvGLView.cpp` | Per-view `redraw()` and `getContext()` |
| `libs/VtkEngine/Visualization/VtkDisplayTools.h` | `ScopedHotZoneRender` class, `beginPrimaryRender` |
| `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` | `ScopedHotZoneRender` impl (lines 198-283), `beginPrimaryRender/endPrimaryRender` (285-330) |
| `libs/CV_db/src/ecvDisplayTools.cpp` | `RedrawDisplay` (lines 3211-3400), `GetContext` overloads (105-149, 2669-2740) |
| `libs/CV_db/include/ecvDisplayTools.h` | Static API declarations, context-aware `GetContext` at line 205 |
| `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` | `ScopedRenderOverride` in `wheelEvent` (line 774) |

---

### Task 1: Make `ecvGLView::getContext` Fully Local

**Rationale:** Currently `ecvGLView::getContext` (line 329) calls `ecvDisplayTools::GetContext(context)` — the singleton overload at line 2669. A context-aware overload `GetContext(context, viewCtx)` already exists at line 105 and fills `CC_DRAW_CONTEXT` purely from the passed `ecvViewContext`. We switch to using that overload.

**Risk:** Low — single method, pure read-path change.

**Files:**
- Modify: `libs/VtkEngine/Visualization/ecvGLView.cpp:329-343`

- [ ] **Step 1: Read current `ecvGLView::getContext`**

Current code at line 329:

```cpp
void ecvGLView::getContext(ccGLDrawContext& context) const {
    ecvDisplayTools::GetContext(context);  // singleton path (line 2669)
    context.display = const_cast<ecvGLView*>(this);
    if (m_vtkWidget) {
        context.glW = m_vtkWidget->width();
        context.glH = m_vtkWidget->height();
        context.devicePixelRatio =
                static_cast<float>(m_vtkWidget->devicePixelRatioF());
    }
    context.defaultPointSize =
            static_cast<unsigned char>(m_ctx.viewportParams.defaultPointSize);
    context.defaultLineWidth =
            static_cast<unsigned char>(m_ctx.viewportParams.defaultLineWidth);
    context.currentLineWidth = context.defaultLineWidth;
}
```

- [ ] **Step 2: Replace singleton call with context-aware overload**

Change to:

```cpp
void ecvGLView::getContext(ccGLDrawContext& context) const {
    ecvDisplayTools::GetContext(context, m_ctx);  // pure per-view path (line 105)
    context.display = const_cast<ecvGLView*>(this);
    if (m_vtkWidget) {
        context.glW = m_vtkWidget->width();
        context.glH = m_vtkWidget->height();
        context.devicePixelRatio =
                static_cast<float>(m_vtkWidget->devicePixelRatioF());
    }
    // defaultPointSize, defaultLineWidth, currentLineWidth are already
    // filled by the context-aware GetContext from m_ctx.viewportParams.
}
```

**Why the widget overrides remain:** The context-aware `GetContext(context, viewCtx)` reads `glW/H` from `viewCtx.glViewport` (which is the VTK viewport rect) and hardcodes `devicePixelRatio = 1.0f`. The widget provides the true widget dimensions and DPI ratio, so these 3 overrides are correct and intentional.

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`
Expected: Clean compile (the `GetContext(context, m_ctx)` overload is declared at `ecvDisplayTools.h:205`).

- [ ] **Step 4: Commit**

```bash
git add libs/VtkEngine/Visualization/ecvGLView.cpp
git commit -m "refactor(phase-b): ecvGLView::getContext reads from m_ctx not singleton

Replaced singleton GetContext(context) call with the context-aware
GetContext(context, m_ctx) overload. defaultPointSize, defaultLineWidth,
and currentLineWidth now come from m_ctx.viewportParams via the overload.
Widget glW/glH/devicePixelRatio overrides remain (widget is ground truth
for display dimensions)."
```

---

### Task 2a: Reduce `ScopedHotZoneRender` Swap Surface (Phase B scope)

**Rationale:** `ScopedHotZoneRender` (lines 201-283 in `VtkDisplayTools.cpp`) currently swaps 8 singleton fields:

| # | Field swapped | Already in `effectiveCtx()`? | Can remove swap? |
|---|---|---|---|
| 1 | `m_visualizer3D` | No (VTK pipeline) | **No** — needed by `DrawClickableItems` |
| 2 | `m_visualizer2D` | No (VTK pipeline) | **No** |
| 3 | `m_vtkWidget` | No (VTK pipeline) | **No** |
| 4 | `m_glViewport` | Partially | **No** — `SetGLViewport` writes to effectiveCtx |
| 5 | `m_hotZone` | No | **No** — `DrawClickableItems` reads `s_tools.instance->m_hotZone` |
| 6 | `m_clickableItemsVisible` | Yes (`effectiveCtx()`) | **Yes** — redundant after Phase A |
| 7 | `m_viewportParams.defaultPointSize` | Yes (`effectiveCtx()`) | **Yes** — redundant after Phase A |
| 8 | `m_viewportParams.defaultLineWidth` | Yes (`effectiveCtx()`) | **Yes** — redundant after Phase A |
| 9 | `m_bubbleViewModeEnabled` | Yes (`effectiveCtx()`) | **Yes** — redundant after Phase A |

After Phase A, `effectiveCtx()` already routes items 6-9 to the active view's `ecvViewContext`. The swap is redundant for these 4 fields.

**Risk:** Low — removing redundant swaps, not changing the draw path.

**Files:**
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.cpp:201-283`
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.h:457-470` (update saved-state member list)

- [ ] **Step 1: Read `ScopedHotZoneRender` constructor, draw, destructor**

Read `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` lines 198-283.

- [ ] **Step 2: Remove redundant saves/restores from constructor**

In the constructor (line 201), remove the saves of:
- `m_savedClickableVis` (line 214)
- `m_savedPtSize` (line 215)
- `m_savedLnWidth` (line 216)

And remove the writes to singleton of:
- `dt->m_clickableItemsVisible = ctx.clickableItemsVisible;` (line 250)
- `dt->m_viewportParams.defaultPointSize = ...` (lines 251-252)
- `dt->m_viewportParams.defaultLineWidth = ...` (lines 253-254)
- `dt->m_bubbleViewModeEnabled = ctx.bubbleViewModeEnabled;` (line 255)

New constructor body (changes only — VTK pipeline swaps remain):

```cpp
VtkDisplayTools::ScopedHotZoneRender::ScopedHotZoneRender(
        VtkDisplayTools* dt,
        VtkVisPtr vis,
        QVTKWidgetCustom* widget,
        ecvDisplayTools::HotZone*& hotZone,
        ecvViewContext& ctx,
        std::vector<ecvDisplayTools::ClickableItem>& clickableItems)
        : m_dt(dt),
          m_savedVis(dt->m_visualizer3D),
          m_saved2D(dt->m_visualizer2D),
          m_savedWidget(dt->m_vtkWidget),
          m_savedGLViewport(dt->m_glViewport),
          m_savedHz(dt->m_hotZone),
          m_savedItems(dt->m_clickableItems),
          m_hotZone(hotZone),
          m_ctx(ctx),
          m_clickableItems(clickableItems) {
    dt->m_visualizer3D = vis;
    dt->m_vtkWidget = widget;
    dt->SetCurrentScreen(widget);

    if (widget) {
        ecvDisplayTools::SetGLViewport(
                QRect(0, 0, widget->width(), widget->height()));
    }

    if (ecvDisplayTools::USE_2D && widget) {
        auto localVis = widget->localImageVis();
        if (!localVis && widget->getVtkRender()) {
            localVis = std::make_shared<ImageVis>("2Dviewer_hz", false);
            localVis->setRender(widget->getVtkRender());
            localVis->setupInteractor(widget->GetInteractor(),
                                      widget->GetRenderWindow());
            widget->setLocalImageVis(localVis);
        }
        if (localVis) {
            dt->m_visualizer2D = localVis;
        }
    }

    ++dt->m_scopedVisSwapDepth;

    if (!m_hotZone) {
        m_hotZone = new ecvDisplayTools::HotZone(widget);
    }
    dt->m_hotZone = m_hotZone;
    // Phase A: clickableItemsVisible, defaultPointSize, defaultLineWidth,
    // bubbleViewModeEnabled are read from effectiveCtx() — no swap needed.
}
```

- [ ] **Step 3: Simplify `draw()` — remove write-back of pointSize/lineWidth**

In `draw()` (line 258), the current code reads back `m_dt->m_viewportParams.defaultPointSize` into `m_ctx` after `DrawClickableItems`. Since the hot zone slider modifies the singleton value directly, we need to read the result back into `m_ctx` — but through `effectiveCtx()` which is already correct. Change:

```cpp
void VtkDisplayTools::ScopedHotZoneRender::draw() {
    int yStart = 0;
    ecvDisplayTools::DrawClickableItems(0, yStart);

    // Hot zone slider may have modified point size / line width.
    // Read results back from effectiveCtx() (which is the active view's m_ctx).
    m_clickableItems = m_dt->m_clickableItems;
}
```

- [ ] **Step 4: Simplify destructor — remove redundant restores**

In destructor (line 269), remove restores of:
- `m_dt->m_viewportParams.defaultPointSize = m_savedPtSize;` (line 272)
- `m_dt->m_viewportParams.defaultLineWidth = m_savedLnWidth;` (line 273)
- `m_dt->m_clickableItemsVisible = m_savedClickableVis;` (line 275)

New destructor:

```cpp
VtkDisplayTools::ScopedHotZoneRender::~ScopedHotZoneRender() {
    --m_dt->m_scopedVisSwapDepth;

    m_dt->m_hotZone = m_savedHz;
    m_dt->m_clickableItems = m_savedItems;

    m_dt->m_visualizer3D = m_savedVis;
    m_dt->m_visualizer2D = m_saved2D;
    m_dt->m_vtkWidget = m_savedWidget;
    m_dt->m_glViewport = m_savedGLViewport;
    m_dt->SetCurrentScreen(m_savedWidget);
}
```

- [ ] **Step 5: Update header — remove saved-state members no longer needed**

In `libs/VtkEngine/Visualization/VtkDisplayTools.h`, remove these member declarations from `ScopedHotZoneRender`:
- `bool m_savedClickableVis;`
- `float m_savedPtSize;`
- `float m_savedLnWidth;`

- [ ] **Step 6: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`

- [ ] **Step 7: Commit**

```bash
git add libs/VtkEngine/Visualization/VtkDisplayTools.cpp \
        libs/VtkEngine/Visualization/VtkDisplayTools.h
git commit -m "refactor(phase-b): reduce ScopedHotZoneRender swap surface

Remove redundant swaps of clickableItemsVisible, defaultPointSize,
defaultLineWidth, bubbleViewModeEnabled — these are already routed
through effectiveCtx() after Phase A. VTK pipeline swaps
(m_visualizer3D, m_vtkWidget, etc.) remain until Phase C."
```

---

### Task 3a: Make `RedrawDisplay` Delegate to Per-View `redraw()` (Progressive)

**Rationale:** `RedrawDisplay` (lines 3211-3400 in `ecvDisplayTools.cpp`) runs a full singleton-based draw pipeline: `beginPrimaryRender` → debug traces → `CheckIfRemove` → `DrawBackground` → `Draw3D` → `DrawForeground` → `UpdateScreen` → `endPrimaryRender`. This duplicates `ecvGLView::redraw`. 

**Progressive approach:** Rather than deleting the entire function, we insert a delegation path at the top. If registered views exist, each view's `redraw()` handles the actual drawing. The singleton path remains as a legacy fallback. Non-draw responsibilities (debug traces, message cleanup, capture mode check) remain in `RedrawDisplay` for now — they will move to per-view in Phase C/D.

**Risk:** Low-Medium — the delegation is additive; the fallback path is preserved.

**Files:**
- Modify: `libs/CV_db/src/ecvDisplayTools.cpp:3211-3400` (the `RedrawDisplay` function)
- Reference: `libs/CV_db/include/ecvViewManager.h` (for `views()` accessor)

- [ ] **Step 1: Verify `ecvViewManager::views()` accessor exists**

```bash
rg "views\(\)" libs/CV_db/include/ecvViewManager.h
```

Expected: A `const QList<ecvGenericGLDisplay*>& views()` or similar accessor.

- [ ] **Step 2: Refactor `RedrawDisplay` — add delegation path**

Insert the per-view delegation after the housekeeping (debug traces, message cleanup, `CheckIfRemove`) but **before** the singleton draw calls. The key insight: housekeeping is global (message timers etc.), but the actual background/3D/foreground draw should go through per-view `redraw()`.

New `RedrawDisplay`:

```cpp
void ecvDisplayTools::RedrawDisplay(bool only2D, bool forceRedraw) {
    if (!HasInstance()) return;

    // === Global housekeeping (stays in RedrawDisplay) ===

    // Debug traces cleanup
    if (s_tools.instance->effectiveCtx().showDebugTraces) {
        RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, DEBUG_LAYER_ID));
        if (!s_tools.instance->m_diagStrings.isEmpty()) {
            QStringList::iterator it = s_tools.instance->m_diagStrings.begin();
            while (it != s_tools.instance->m_diagStrings.end()) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, *it));
                it = s_tools.instance->m_diagStrings.erase(it);
            }
        }
        s_tools.instance->m_diagStrings
                << QString("only2D : %1").arg(only2D ? "true" : "false");
        s_tools.instance->m_diagStrings
                << QString("ForceRedraw : %1")
                           .arg(forceRedraw ? "true" : "false");
    } else {
        RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, DEBUG_LAYER_ID));
        if (!s_tools.instance->m_diagStrings.isEmpty()) {
            QStringList::iterator it = s_tools.instance->m_diagStrings.begin();
            while (it != s_tools.instance->m_diagStrings.end()) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, *it));
                it = s_tools.instance->m_diagStrings.erase(it);
            }
        }
    }

    CheckIfRemove();
    if (s_tools.instance->m_removeAllFlag) {
        Update();
        return;
    }

    SetFontPointSize(GetFontPointSize());

    if (!only2D) {
        Deprecate3DLayer();
    }

    // Clean outdated messages (global, not per-view)
    {
        std::list<MessageToDisplay>::iterator it =
                s_tools.instance->m_messagesToDisplay.begin();
        qint64 currentTime_sec = s_tools.instance->m_timer.elapsed() / 1000;
        while (it != s_tools.instance->m_messagesToDisplay.end()) {
            if (it->messageValidity_sec < currentTime_sec) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                                it->message));
                it = s_tools.instance->m_messagesToDisplay.erase(it);
            } else {
                ++it;
            }
        }
    }

    // === Per-view delegation (Phase B) ===
    auto& vm = ecvViewManager::instance();
    const auto& views = vm.views();
    if (!views.isEmpty()) {
        for (auto* viewDisplay : views) {
            viewDisplay->redraw(only2D, forceRedraw);
        }
        s_tools.instance->m_shouldBeRefreshed = false;
        return;
    }

    // === Legacy singleton draw path (fallback when no views registered) ===
    s_tools.instance->beginPrimaryRender();

    bool drawBackground = false;
    bool draw3DPass = false;
    bool drawForeground = true;
    bool draw3DCross = GetDisplayParameters().displayCross;

    if (s_tools.instance->m_updateFBO ||
        s_tools.instance->m_captureMode.enabled) {
        drawBackground = true;
        draw3DPass = true;
    }

    CC_DRAW_CONTEXT CONTEXT;
    GetContext(CONTEXT);
    CONTEXT.display = s_tools.instance;

    QRect originViewport = s_tools.instance->effectiveCtx().glViewport;
    bool modifiedViewport = false;

    if (drawBackground) {
        CONTEXT.clearColorLayer = true;
        CONTEXT.clearDepthLayer = true;
        DrawBackground(CONTEXT);
    }

    if (draw3DPass) {
        CONTEXT.forceRedraw = forceRedraw;
        Draw3D(CONTEXT);
    }

    // Display debug traces
    if (s_tools.instance->effectiveCtx().showDebugTraces) {
        if (!s_tools.instance->m_diagStrings.isEmpty()) {
            QFont font = GetTextDisplayFont();
            int font_size = font.pointSize();
            QFontMetrics fm(font);
            int x = s_tools.instance->effectiveCtx().glViewport.width() / 2 - 100;
            int margin = font_size / 2;
            int y = margin;
            {
                int height = (s_tools.instance->m_diagStrings.size() + 1) *
                             (fm.height() + margin);
                WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                        DEBUG_LAYER_ID);
                param.color = ecvColor::dark;
                param.color.a = 0.5f;
                param.rect = QRect(
                        x, s_tools.instance->effectiveCtx().glViewport.height() - y - height,
                        200, height);
                DrawWidgets(param, true);
            }
            y += margin;
            for (const QString& str : s_tools.instance->m_diagStrings) {
                RenderText(x + font_size, y + font_size, str, font,
                           ecvColor::yellow, DEBUG_LAYER_ID);
                y += fm.height() + margin;
            }
        }
    }

    if (modifiedViewport) {
        SetGLViewport(originViewport);
        CONTEXT.glW = originViewport.width();
        CONTEXT.glH = originViewport.height();
    }

    if (drawBackground || draw3DCross) {
        s_tools.instance->m_updateFBO = false;
    }

    if (drawForeground) {
        DrawForeground(CONTEXT);
    }

    s_tools.instance->m_shouldBeRefreshed = false;

    UpdateScreen();
    s_tools.instance->endPrimaryRender();
}
```

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`

- [ ] **Step 4: Runtime test**

1. Launch ACloudViewer, load a point cloud
2. Rotate/zoom — should render correctly
3. Create second view — both should render independently
4. Close a view — remaining view keeps rendering
5. Hot zone (point size slider) visible and functional in each view

- [ ] **Step 5: Commit**

```bash
git add libs/CV_db/src/ecvDisplayTools.cpp
git commit -m "refactor(phase-b): RedrawDisplay delegates to per-view redraw()

When registered views exist, RedrawDisplay iterates them and calls
each view's redraw() instead of running the singleton draw pipeline.
Global housekeeping (debug traces, message cleanup) stays in
RedrawDisplay. Legacy singleton path retained as fallback when
no views are registered."
```

---

### Task 4: Remove `ScopedRenderOverride` From `wheelEvent`

**Rationale:** `ScopedRenderOverride` in `QVTKWidgetCustom::wheelEvent` (line 774) sets `m_renderingView` so that `getEffectiveView()` returns the wheel target. After Phase A, all inline APIs read from `effectiveCtx()` which consults `getEffectiveView()`. But the wheel event fires on whichever widget has focus — that widget's view should already be the active view (set on mouse press). The override is redundant.

**Risk:** Low — wheel events only fire on the focused widget.

**Files:**
- Modify: `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp:774`

- [ ] **Step 1: Read current `wheelEvent` around line 774**

Read `libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp` around line 770-790.

- [ ] **Step 2: Verify wheel handler dependency on `ScopedRenderOverride`**

Check whether any calls inside the wheel handler read from `effectiveCtx()` or from the singleton directly. If all reads go through `effectiveCtx()` (which uses `getEffectiveView()` → `m_renderingView` or active view), and the active view is correct (set on mouse press), the override is safe to remove.

- [ ] **Step 3: Remove `ScopedRenderOverride` from `wheelEvent`**

Delete the line:
```cpp
ecvViewManager::ScopedRenderOverride wheelRenderScope(wheelDisplay);
```

If `wheelDisplay` is still needed elsewhere in the function, keep its computation but remove the RAII guard.

- [ ] **Step 4: Build and verify**

Run: `cmake --build build --target ACloudViewer 2>&1 | tail -30`

- [ ] **Step 5: Runtime test — wheel zoom in multi-window**

1. Open two views, load data in both
2. Wheel-zoom in Window 1 — only Window 1 zoom changes
3. Wheel-zoom in Window 2 — only Window 2 zoom changes
4. Scroll quickly between windows — no cross-contamination

- [ ] **Step 6: Commit**

```bash
git add libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp
git commit -m "refactor(phase-b): remove ScopedRenderOverride from wheelEvent

After Phase A routing, effectiveCtx() returns the correct per-view
state for the active view (set on mouse press). ScopedRenderOverride
in wheelEvent was a bridge mechanism that is now redundant."
```

---

### Task 5: Deprecate Singleton Draw Path

**Rationale:** After Task 3a makes `RedrawDisplay` delegate to per-view `redraw()`, the singleton draw functions (`Draw3D`, `DrawBackground`, `DrawForeground`) and `beginPrimaryRender/endPrimaryRender` are only used by the legacy fallback path. Marking them `[[deprecated]]` creates compiler warnings that surface any remaining call sites.

**Risk:** None — annotations only, no behavior change.

**Files:**
- Modify: `libs/CV_db/include/ecvDisplayTools.h` (add `[[deprecated]]` to `Draw3D`, `DrawBackground`, `DrawForeground`)
- Modify: `libs/VtkEngine/Visualization/VtkDisplayTools.h` (add `[[deprecated]]` to `beginPrimaryRender`, `endPrimaryRender`)

- [ ] **Step 1: Add deprecation to singleton draw functions**

In `libs/CV_db/include/ecvDisplayTools.h`, find and annotate:

```cpp
[[deprecated("Phase B: use per-view ecvGLView::redraw()")]]
static void Draw3D(CC_DRAW_CONTEXT& CONTEXT);

[[deprecated("Phase B: use per-view ecvGLView::redraw()")]]
static void DrawBackground(CC_DRAW_CONTEXT& CONTEXT);

[[deprecated("Phase B: use per-view ecvGLView::redraw()")]]
static void DrawForeground(CC_DRAW_CONTEXT& CONTEXT);
```

- [ ] **Step 2: Add deprecation to primary render guards**

In `libs/VtkEngine/Visualization/VtkDisplayTools.h`, find and annotate:

```cpp
[[deprecated("Phase B: use per-view ecvGLView::redraw()")]]
void beginPrimaryRender() override;

[[deprecated("Phase B: use per-view ecvGLView::redraw()")]]
void endPrimaryRender() override;
```

- [ ] **Step 3: Build — count deprecation warnings**

Run: `cmake --build build --target ACloudViewer 2>&1 | grep -c "deprecated"`

This tells us how many call sites still use the legacy draw path (expected: 1 — the fallback in `RedrawDisplay` itself).

- [ ] **Step 4: Commit**

```bash
git add libs/CV_db/include/ecvDisplayTools.h \
        libs/VtkEngine/Visualization/VtkDisplayTools.h
git commit -m "refactor(phase-b): deprecate singleton draw path

Mark Draw3D, DrawBackground, DrawForeground, beginPrimaryRender,
endPrimaryRender as [[deprecated]]. Per-view ecvGLView::redraw()
is the replacement path."
```

---

### Task 6: Phase B Acceptance Verification

- [ ] **Step 1: Verify `ecvGLView::getContext` has zero singleton reads**

Run: `rg "ecvDisplayTools::GetContext\(context\)" libs/VtkEngine/Visualization/ecvGLView.cpp`

Expected: No matches (should use `GetContext(context, m_ctx)` now).

- [ ] **Step 2: Verify `ScopedHotZoneRender` swap count reduced**

Run: `rg "m_saved" libs/VtkEngine/Visualization/VtkDisplayTools.cpp | wc -l`

Expected: Fewer saved members than before (removed 3: clickableVis, ptSize, lnWidth).

- [ ] **Step 3: Verify `RedrawDisplay` delegates to per-view `redraw()`**

Run: `rg "->redraw\(" libs/CV_db/src/ecvDisplayTools.cpp`

Expected: At least one match in `RedrawDisplay`.

- [ ] **Step 4: Verify `ScopedRenderOverride` removed from wheel**

Run: `rg "ScopedRenderOverride" libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp`

Expected: No matches.

- [ ] **Step 5: Verify deprecation annotations present**

Run: `rg "deprecated.*Phase B" libs/CV_db/include/ecvDisplayTools.h libs/VtkEngine/Visualization/VtkDisplayTools.h`

Expected: 5 matches (Draw3D, DrawBackground, DrawForeground, beginPrimaryRender, endPrimaryRender).

- [ ] **Step 6: Full runtime regression test**

1. Load point cloud, rotate/zoom — renders correctly
2. Create second view — both render independently
3. Close second view — first keeps working
4. Split view — both halves render
5. Wheel zoom in each view — isolated, no cross-contamination
6. Click object in DB tree — highlights without crash
7. Hot zone (point size slider) works in each view
8. Clickable items visible/functional in each view

---

## Deferred Work (NOT in Phase B)

| Task | Deferred to | Reason |
|---|---|---|
| **Task 2b:** Full `ScopedHotZoneRender` elimination | Phase C | `DrawClickableItems` reads `s_tools.instance->m_hotZone` and VTK objects; requires `QVTKWidgetCustom` owner-view refactoring first |
| **Task 3b:** Move debug traces, messages, capture mode to per-view | Phase C/D | These are global/primary-view responsibilities; moving them requires `ecvOverlayDialog` view binding |
| **Task 3c:** Delete `beginPrimaryRender/endPrimaryRender` | Phase E | Only after all draw paths route through per-view `redraw()` |
| **Full `ScopedRenderOverride` deletion** | Phase C | Other uses besides `wheelEvent` may exist in interaction pipeline |

---

## Phase B Completion Summary

| Acceptance Criterion (from roadmap) | Task | Status |
|---|---|---|
| `ecvGLView::getContext` does not call singleton `GetContext` | Task 1 | Ready |
| `ScopedHotZoneRender` swaps reduced (4 fewer fields) | Task 2a | Ready |
| `RedrawDisplay` delegates to per-view `redraw()` | Task 3a | Ready |
| `ScopedRenderOverride` removed from `wheelEvent` | Task 4 | Ready |
| Singleton draw path deprecated | Task 5 | Ready |
| Multi-window draw: no cross-window bleed, no flicker | Task 6 | Ready |

**Next phase:** Phase C — Interaction Pipeline (`QVTKWidgetCustom` de-singleton + full `ScopedHotZoneRender` elimination).
