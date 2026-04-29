# 审计：单例直接成员访问全量扫描

**扫描范围：** 仓库内 `*.cpp` / `*.h`
**扫描日期：** 2026-04-24
**扫描模式：** `TheInstance()->m_` / `s_tools.instance->m_` / `m_tools->m_` / `activeSecondaryView()`

---

## 1. 全量统计

| 模式 | 文件 | 访问次数 | 说明 |
|------|------|---------|------|
| `TheInstance()->m_*` | `libs/CV_db/include/ecvDisplayTools.h` | **35** | 内联 getter/setter |
| `TheInstance()->m_*` | `libs/CV_db/src/ecvDisplayTools.cpp` | **1** | `m_viewportParams.perspectiveView` |
| `TheInstance()->m_*` | `app/MainWindow.cpp` | **2** | `MainWindow::TheInstance()->m_mdiArea` |
| `s_tools.instance->m_*` | `libs/CV_db/src/ecvDisplayTools.cpp` | **527** | 核心单例直读 |
| `m_tools->m_*` | `libs/VtkEngine/.../QVTKWidgetCustom.cpp` | **163** | Widget 事件处理 |
| `activeSecondaryView()` | `libs/CV_db/src/ecvDisplayTools.cpp` | **20** | 缓解措施（write-through 委托点） |
| **总计** | | **748** | |

---

## 2. 优先级说明

| 优先级 | 含义 | 典型修复 |
|--------|------|----------|
| **P0** | 渲染/投影/相机/拾取与用户感知强相关；错视图会错位、串窗 | 改为 **context-aware API** 或 **ecvViewContext** 直读 |
| **P1** | 全局 UI / 主窗口结构 | 评估是否必须从 `MainWindow` 取 MDI |
| **P2** | 低频或开发调试用 | 迭代替换 |
| **—** | 语义上必须为全局单例（Scene DB / MainWindow） | **不修** |

---

## 3. `TheInstance()->m_*` 逐行清单（ecvDisplayTools.h）

### 3.1 窗口/屏幕（保留或迁移）

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| 937 | `m_currentScreen` | P2 | `GetCurrentScreen()` 已先尝试 `getEffectiveView()`；此行为 fallback |
| 952 | `m_mainScreen` | — | 应用级主屏幕指针，全局可接受 |
| 960 | `m_mainScreen` (setter) | — | 同上 |
| 967 | `m_win` | — | `QMainWindow*`，全局单例 |
| 974 | `m_win` (setter) | — | 同上 |

### 3.2 数据库（保留）

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| 1013 | `m_winDBRoot` | — | `GetOwnDB()` 已委托 `getEffectiveView()` |
| 1043 | `m_globalDBRoot` | — | **全局 Scene DB**，有意共享 |

### 3.3 视口/相机参数（P0 — 核心重构目标）

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| **1108** | `m_viewportParams.viewMat` | **P0** | `GetBaseViewMat()` 未走 `getEffectiveView()`；渲染/数学调用会错位 |
| **1444-1445** | `m_viewportParams.zNear/zFar` | **P0** | `SetCameraClip` 内联直写单例，无 write-through |
| **1466** | `m_viewportParams.fov_deg` | **P0** | `SetCameraFovy` 内联直写单例，无 write-through |
| **1931** | `m_viewportParams.perspectiveView` | **P0** | `getPerspectiveState` 间接读单例 |
| **2545** | `m_glViewport.width()` | P2 | `GlWidth()` 先尝试 effective view；此行为主视口 fallback |
| **2550** | `m_glViewport.height()` | P2 | `GlHeight()` 同上 |

### 3.4 渲染标志

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| 2073 | `m_updateFBO` | P2 | `Deprecate3DLayer()` 置 true |
| 2075 | `m_validProjectionMatrix` | P2 | `InvalidateViewport()` |
| 2078 | `m_validModelviewMatrix` | P2 | `InvalidateProjectionMatrix()` |

### 3.5 Bubble / Pivot / 显示选项

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| **2026** | `m_pivotVisibility` | **P0** | 枢轴可见性跨窗 |
| **2106** | `m_bubbleViewModeEnabled` | P0 | 气泡视图 per-view 但读单例 |
| 2114 | `m_showCursorCoordinates` (setter) | P2 | |
| 2119 | `m_showCursorCoordinates` | P2 | |
| **2130** | `m_autoPickPivotAtCenter` | P0 | 自动枢轴 |
| **2138** | `m_rotationAxisLocked` | P0 | 旋转锁定 per-view |
| 1876 | `m_exclusiveFullscreen` | P2 | |
| 1879 | `m_exclusiveFullscreen` (setter) | P2 | |
| 1128 | `m_removeAllFlag` (setter) | P2 | |

### 3.6 拾取/覆盖/字体

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| **2270** | `m_pickRadius` (setter) | **P0** | 拾取半径应 per-view |
| **2273** | `m_pickRadius` | **P0** | 同上 |
| **2282** | `m_displayOverlayEntities` | P0 | 覆盖层显示 |
| 2244 | `m_font` (setter) | P1 | |
| 2252 | `m_clickableItemsVisible` (setter) | P1 | |
| 2255 | `m_clickableItemsVisible` | P1 | |
| 2261 | `m_font` | P1 | |

### 3.7 调试

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| 2149 | `m_showDebugTraces` (setter) | P2 | |
| 2154 | `m_showDebugTraces` (toggle) | P2 | 同一行两次 `TheInstance()` |

---

## 4. `TheInstance()->m_*`（ecvDisplayTools.cpp）

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| **2227** | `m_viewportParams.perspectiveView` | **P0** | `ZoomCamera` 中读取，应改为 `GetViewportParameters()` |

---

## 5. `MainWindow::TheInstance()->m_*`

| 行号 | 表达式 | P | 说明 |
|------|--------|---|------|
| 2191 | `m_mdiArea->subWindowList()` | P1 | `GetRenderWindows`，考虑改为 `m_mdiArea` 成员 |
| 2205 | `m_mdiArea->subWindowList()` | P1 | `GetRenderWindow`，同上 |

---

## 6. `s_tools.instance->m_*` 分类统计（ecvDisplayTools.cpp，527 处）

| 类别 | 代表性成员 | 估计数量 | 重构阶段 |
|------|-----------|---------|---------|
| 视口/相机 | `m_viewportParams`, `m_viewMatd`, `m_projMatd`, `m_glViewport` | ~120 | B (绘制管线) |
| 交互/拾取 | `m_interactionFlags`, `m_pickingMode`, `m_pickRadius`, `m_activeItems` | ~60 | C (交互管线) |
| 鼠标/触摸 | `m_lastMousePos`, `m_mouseMoved`, `m_mouseButtonPressed` | ~40 | C (交互管线) |
| VTK 管线 | `m_visualizer3D`, `m_vtkWidget`, `m_visualizer2D` | ~30 | B (绘制管线) |
| 显示/HotZone | `m_hotZone`, `m_clickableItems`, `m_messagesToDisplay` | ~50 | C/D |
| Bubble/Pivot | `m_bubbleViewModeEnabled`, `m_pivotVisibility` | ~30 | C |
| 光照 | `m_sunLightPos`, `m_customLightPos` | ~15 | B |
| 场景/DB | `m_globalDBRoot`, `m_winDBRoot` | ~20 | 保留 |
| 渲染标志 | `m_updateFBO`, `m_captureMode`, `m_alwaysUseFBO` | ~25 | B |
| 定时器/调试 | `m_timer`, `m_deferredPickingTimer`, `m_showDebugTraces` | ~20 | D |
| 窗口/UI | `m_currentScreen`, `m_mainScreen`, `m_win`, `m_font` | ~15 | 保留/迁移 |
| 杂项 | `m_uniqueID`, `m_autoRefresh`, `m_overridenDisplayParameters` | ~50 | E |
| **信号/连接** | Signal emitter helpers | ~52 | E |

---

## 7. `m_tools->m_*` 分类统计（QVTKWidgetCustom.cpp，163 处）

| 类别 | 代表性成员 | 估计数量 | 重构阶段 |
|------|-----------|---------|---------|
| 鼠标状态 | `m_lastMousePos`, `m_mouseMoved`, `m_mouseButtonPressed` | ~30 | C |
| 交互标志 | `m_interactionFlags` | ~20 | C |
| 拾取 | `m_pickingMode`, `m_pickRadius`, `m_pickingModeLocked` | ~15 | C |
| 视口参数 | `m_viewportParams.*` | ~40 | C |
| Bubble/Pivot | `m_bubbleViewModeEnabled`, `m_pivotVisibility`, `m_pivotSymbolShown` | ~20 | C |
| 定时器 | `m_deferredPickingTimer`, `m_timer`, `m_lastClickTime_ticks` | ~12 | C/D |
| HotZone | `m_hotZone` | ~8 | C |
| 其他 | `m_widgetClicked`, `m_ignoreMouseReleaseEvent`, etc. | ~18 | C |

---

## 8. `activeSecondaryView()` 使用点（ecvDisplayTools.cpp，20 处）

这些是 **已有的缓解措施**，表示 write-through 或委托到副视图的地方：

| 函数 | 作用 | 完备性 |
|------|------|--------|
| `SetPointSize` | write-through 到副视图 `setViewportParameters` | 完备 |
| `SetLineWidth` | 同上 | 完备 |
| `SetDisplayParameters` | write-through 到副视图 | 完备 |
| `GetDisplayParameters` | 从副视图读取 | 完备 |
| `GetGLCameraParameters` | 委托到副视图 | 完备 |
| `GetContext` | pointSize/lineWidth + `CONTEXT.display` | **不完备：glW/glH 未委托** |
| `GetCurrentScreen` | fallback 到 getEffectiveView | 完备 |
| `GlWidth/GlHeight` | 先尝试 effective view | 完备（在 header 中） |
| `GetOwnDB` | 委托到 effective view | 完备 |
| 其余 ~10 处 | 各种 getter/setter | 部分完备 |

**未覆盖的关键 API（需要添加 activeSecondaryView 委托）：**

- `SetCameraClip` (行 1444-1445) — 直写 `TheInstance()->m_viewportParams`
- `SetCameraFovy` (行 1466) — 直写 `TheInstance()->m_viewportParams`
- `GetBaseViewMat` (行 1108) — 直读 `TheInstance()->m_viewportParams.viewMat`
- `getPerspectiveState` (行 1931) — 直读 `TheInstance()->m_viewportParams.perspectiveView`
- `SetViewportDefaultPointSize/LineWidth` — 仅写单例
- `Deprecate3DLayer / InvalidateViewport / InvalidateProjectionMatrix` — 仅写单例

---

## 9. push/pull 覆盖审计（ecvGLView.cpp L394-523）

### 已覆盖字段（~30）

| 类别 | 字段 |
|------|------|
| 交互/拾取 | `interactionFlags`, `pickingMode`, `pickingModeLocked`, `pickRadius` |
| 鼠标状态 | `lastMousePos`, `lastMouseMovePos`, `mouseMoved`, `mouseButtonPressed`, `ignoreMouseReleaseEvent`, `widgetClicked` |
| 触摸 | `touchInProgress`, `touchBaseDist` |
| HotZone | `hotZone`, `clickableItemsVisible`, `clickableItems` |
| 显示 | `displayOverlayEntities`, `exclusiveFullscreen`, `showCursorCoordinates`, `showDebugTraces` |
| Bubble | `bubbleViewModeEnabled`, `bubbleViewFov_deg` |
| Pivot | `pivotVisibility`, `autoPickPivotAtCenter`, `pivotSymbolShown` |
| 视口 | `viewportParams`（完整拷贝） |
| 旋转锁 | `rotationAxisLocked`, `lockedRotationAxis` |
| 拾取辅助 | `last_point_index`, `last_picked_id`, `rectPickingPoly`, `allowRectangularEntityPicking` |
| 光照 | `sunLightEnabled`, `customLightEnabled`, `customLightPos[4]` |
| 定时器 | `lastClickTime_ticks` |

### 未覆盖字段（已确认遗漏）

| 字段 | 风险 | 说明 |
|------|------|------|
| **`m_activeItems`** | 高 | 2D 交互器/标签 hover 列表，全局共享 |
| **`m_messagesToDisplay`** | 中 | 消息覆盖状态不随视图切换 |
| **`m_viewMatd` / `m_projMatd`** | 中 | CPU 相机矩阵镜像，VTK 更新后可能不同步 |
| `m_autoPivotCandidate` | 低 | |
| `m_diagStrings` | 低 | |
| `m_captureMode` | 低 | |
| `m_overridenDisplayParameters*` | 中 | 显示参数覆盖可能跨窗口 |
| `m_hotZone`（pull 方向） | 低 | push 有但 pull 无（通常 OK，view 持有对象） |

---

## 10. 推荐执行顺序（与重构路线图对齐）

### 阶段 A 前置（立即可做）

1. **P0 header 内联修复**：
   - `GetBaseViewMat` (L1108) → 改为 `GetViewportParameters().viewMat`
   - `SetCameraClip` (L1444-1445) → 添加 `activeSecondaryView()` write-through
   - `SetCameraFovy` (L1466) → 同上
   - `getPerspectiveState` (L1931) → 改为读 `GetViewportParameters().perspectiveView`

2. **ecvDisplayTools.cpp L2227** → 改为 `GetViewportParameters().perspectiveView`

3. **push/pull 补齐**：
   - 添加 `m_activeItems` 到 push/pull
   - 添加 `m_messagesToDisplay` 到 push/pull

### 阶段 A（定义 ecvViewContext）

4. 将 push/pull 的 ~30 个字段整合到 `ecvViewContext` 结构体

### 阶段 B-E

5. 按 `multi-window-refactor-roadmap-Vtk-vs-CC.md` 执行

---

## 11. 一键复扫命令

```bash
cd /path/to/ACloudViewer

# TheInstance 直读
grep -rn 'TheInstance()->m_' --include='*.cpp' --include='*.h' | grep -v 'MainWindow::TheInstance'

# 单例内部直读
grep -c 's_tools\.instance->m_' libs/CV_db/src/ecvDisplayTools.cpp

# Widget 单例直读
grep -c 'm_tools->m_' libs/VtkEngine/VTKExtensions/Widgets/QVTKWidgetCustom.cpp

# write-through 委托点
grep -c 'activeSecondaryView()' libs/CV_db/src/ecvDisplayTools.cpp
```

---

*本审计为静态扫描结果；动态路径（插件 dlopen、Python 绑定）请另做抽样。*
*更新日期：2026-04-24*
