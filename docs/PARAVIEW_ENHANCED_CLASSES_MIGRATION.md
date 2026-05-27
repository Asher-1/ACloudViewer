# ParaView 增强类迁移计划

> 目标：将 ParaView 中经过增强的 VTK 类迁移到 ACloudViewer，替代当前使用的标准 VTK 原生类。

## 概述

| Phase | ParaView 增强类 | 替代 VTK 原生类 | 影响文件数 | 优先级 | 状态 |
|-------|----------------|----------------|-----------|--------|------|
| 1 | `vtkGridAxesActor3D` + `vtkPVGridAxes3DActor` | `vtkCubeAxesActor` | 5 | P0 | **完成** |
| 2 | `vtkPVLODActor` | `vtkLODActor` | 34+ | P1 | **完成** |
| 3 | `vtkPVScalarBarActor` | `vtkScalarBarActor` | 15 | P1 | **已等价实现** (vtkScalarBarActorCustom) |
| 4 | `vtkPVAxesActor` | `vtkAxesActor` | 7 | P2 | **完成** |
| 5 | `vtkPVLight` | `vtkLight` | 5 | P3 | 跳过（ROI 低） |

### 已从 ParaView 迁移的类（无需重复迁移）

- `vtkPVCenterAxesActor` → 已在 `VTKExtensions/Views/`
- `vtkPVInteractorStyle` / `vtkPVImageInteractorStyle` → 已在 `VTKExtensions/InteractionStyle/`
- `vtkPVTrackball*`（Rotate/Zoom/Pan/Roll/MultiRotate 等）→ 已在 `VTKExtensions/InteractionStyle/`
- `vtkPVJoystickFly*` → 已在 `VTKExtensions/InteractionStyle/`
- `cvHardwareSelector`（参考 `vtkPVHardwareSelector` 重写）→ 已在 `Tools/SelectionTools/`
- `vtkPVPostFilter` / `vtkPVCompositeDataPipeline` 等 Core 类 → 已在 `VTKExtensions/Core/`

---

## Phase 1: vtkGridAxesActor3D + vtkPVGridAxes3DActor (P0)

### 目标
用 ParaView 的 GridAxes 模块完全替换 `vtkCubeAxesActor`，解决 Axes Grid 渲染不稳定问题。

### 源文件清单

| 文件 | ParaView 来源路径 | 目标路径 |
|------|------------------|---------|
| `vtkGridAxesHelper.h/.cxx` | `VTK/Rendering/GridAxes/` | `VTKExtensions/Views/GridAxes/` |
| `vtkGridAxesPlaneActor2D.h/.cxx` | `VTK/Rendering/GridAxes/` | `VTKExtensions/Views/GridAxes/` |
| `vtkGridAxesActor2D.h/.cxx` | `VTK/Rendering/GridAxes/` | `VTKExtensions/Views/GridAxes/` |
| `vtkGridAxesActor3D.h/.cxx` | `VTK/Rendering/GridAxes/` | `VTKExtensions/Views/GridAxes/` |
| `vtkPVGridAxes3DActor.h/.cxx` | `Remoting/Views/` | `VTKExtensions/Views/GridAxes/` |

### 依赖关系
```
vtkPVGridAxes3DActor
  └─ vtkGridAxesActor3D
       └─ vtkGridAxesActor2D
            ├─ vtkGridAxesPlaneActor2D
            │    └─ vtkGridAxesHelper
            └─ vtkGridAxesHelper
```

### VTK 模块依赖（全部已满足）
- `VTK::ChartsCore` → vtkAxis
- `VTK::RenderingContext2D` → vtkContextScene
- `VTK::CommonCore/DataModel/Math`
- `VTK::FiltersCore/General/Sources`

### 适配要点
1. 移除 `vtkRenderingGridAxesModule.h` 导出宏 → 替换为 `ECV_VTK_ENGINE_LIB_API`
2. 移除 `vtkRemotingViewsModule.h` 导出宏 → 替换为 `ECV_VTK_ENGINE_LIB_API`
3. 移除 `vtkWrappingHints.h` 相关宏（`VTK_MARSHALAUTO` 等）
4. CMakeLists.txt 添加 GridAxes source group

### 需替换的文件
- `vtkOrthoSliceViewWidget.cpp` — OrthoSlice 的 axes grid
- `vtkGLView.h/.cpp` — 主 3D 视图的 axes grid
- `VtkVis.h/.cpp` — 通用可视化工具的 axes grid

---

## Phase 2: vtkPVLODActor (P1)

### 目标
用 ParaView 简化版 LOD Actor 替换标准 `vtkLODActor`。

### 源文件
- `vtkPVLODActor.h/.cxx` → `VTKExtensions/Views/`

### 注意事项
- `vtkPVLODActor` 继承自 `vtkActor`（非 `vtkLODActor`），不是简单 drop-in 替换
- 需检查每个使用点是否依赖 `vtkLODActor` 特有 API（`SetNumberOfCloudPoints()` 等）
- 移除 OSPRay 条件编译引用（`vtkOSPRayActorNode.h`）
- 影响 34+ 文件，建议通过 `typedef` 别名逐步迁移

---

## Phase 3: vtkPVScalarBarActor (P1)

### 目标
增强 ScalarBar 的标签精度和刻度控制。

### 源文件
- `vtkPVScalarBarActor.h/.cxx` → 合并到现有 `vtkScalarBarActorCustom`

### 注意事项
- 依赖 `vtkScalarBarActorInternal.h`（VTK 内部头文件），需确认可用性
- ACloudViewer 已有 `vtkScalarBarActorCustom`，建议合并增强功能而非引入新类
- 增强功能：固定字体大小、自动标签精度、刻度线/子刻度线

---

## Phase 4: vtkPVAxesActor (P2)

### 目标
增强 Orientation Axes 的渲染质量。

### 源文件
- `vtkPVAxesActor.h/.cxx` → `VTKExtensions/Views/`

### 注意事项
- 继承自 `vtkProp3D`（完全独立实现，非 `vtkAxesActor` 子类）
- 自定义轴几何（圆柱/锥形/球形），文本跟随相机
- 影响 7 个文件

---

## Phase 5: vtkPVLight (P3, 跳过)

ROI 较低，`vtkPVLight` 仅增加命名光源类型，对当前功能影响有限。

---

## 迁移后修复

### 修复 1: OrthoSlice Axes Grid 视觉对齐 ParaView

**问题**：Axes Grid 在灰色背景上使用深灰/黑色字体和网格线，几乎不可见。

**修复**：
- 2D 视图背景从 (0.5, 0.5, 0.5) 改为 (0.2, 0.2, 0.2) 深色
- 网格线颜色从 (0.35) 改为 (0.75) 亮灰
- 标题文字从 (0.1) 改为 (1.0, 1.0, 1.0) 白色
- 标签文字从 (0.2) 改为 (0.9, 0.9, 0.9) 浅灰
- 切片轮廓线从黑色 (0.0) 改为亮色 (0.9)，确保在深色背景下可见

### 修复 2: OrthoSlice Axes Grid 标签/刻度不显示

**根因**：`vtkGridAxesActor3D` 默认 `LabelUniqueEdgesOnly=true`，在 2D 正交投影中，
两个对面（如 MIN_ZX | MAX_ZX）的边投影到相同屏幕位置 → edge_count==2 → 所有标签被隐藏。

**修复**（`vtkOrthoSliceViewWidget.cpp`）：
- 2D 视图 FaceMask 设为两个对面 + `FrontfaceCulling=true`（通过 `vtkProperty`），确保只渲染背面
- 2D 视图显式设置 `SetLabelUniqueEdgesOnly(false)`，确保 Top/Right Side/Front 所有四边都显示刻度标签
- 3D 视图保持 `MIN_XY | MIN_YZ | MIN_ZX` + 默认 `LabelUniqueEdgesOnly=true`

### 修复 2b: Axes Grid 显示效果全面对齐 ParaView

**问题**：
1. 内部网格线密集十字交叉，ParaView 无此效果
2. 2D 视图 grid 不自适应数据包络，标签超出视窗被裁剪
3. 3D 视图使用 0xff（全6面）导致线条过密，与 VTK 原生 cube-axes 差异大

**修复**（`vtkOrthoSliceViewWidget.cpp` + `.h`）：
- `SetGenerateGrid(false)` 关闭内部网格线，仅保留边框和刻度
- 3D 视图 FaceMask 从 `0xff` 改为 `MIN_XY | MIN_YZ | MIN_ZX`（标准 cube-axes）
- 网格线属性：移除 `SetOpacity(0.6)`，使用全不透明清晰边框
- `resetCameras()` 根据 `m_axesGridVisible` 状态选择缩放策略：
  - 显示 grid 时：以几何边界 ResetCamera + 1.35x 缩放，为标签留空间
  - 隐藏 grid 时：以切片数据边界紧凑显示 + 1.05x 缩放
- 3D 视窗 grid 可见时 Zoom 从 1.6 降至 1.2，防止 grid 超出视窗
- Toggle handler 改为直接调用 `resetCameras()` 统一相机管理

### 修复 3: Comparative 窗口切换 Crash

**根因**：
1. `m_sourceView` 使用裸指针，源视图销毁后指针悬空
2. `m_cameraLinkTimer`（16ms 间隔）在窗口隐藏后仍触发，访问无效 VTK 上下文
3. `QTimer::singleShot` 回调在 widget 隐藏后仍执行渲染操作
4. 析构函数未断开全局信号连接

**修复**：
- `m_sourceView` 改为 `QPointer<vtkGLView>`，自动感知对象销毁
- 添加 `hideEvent()` 停止 camera link timer
- 添加 `showEvent()` 恢复 camera link timer
- 所有 `QTimer::singleShot` 回调添加 `!isVisible()` 守护
- `onCameraLinkTick()` 添加 `m_closing || !isVisible()` 检查
- `syncCamerasFromFirst()` 添加完整的空指针链式检查
- 析构函数中断开 `ecvRepresentationManager` 和 `ecvViewManager` 的信号连接

**后续加固**（v2）：
- `installCameraLink()` 仅创建定时器，不立即启动，由 `showEvent()` 控制
- `safeRenderWindow()` 增加 `GetNeverRendered()` / `IsCurrent()` GL 上下文检查
- `forceRenderAllSubViews()` 增加 `!isVisible()` 前置守护

### 修复 4: VTK 日志级别过滤

**问题**：`vtkPolyDataPlaneCutter` 每次切片更新输出 INFO 日志，刷屏终端。

**修复**（`QVTKWidgetCustom.cpp`）：
- 在初始化时调用 `vtkLogger::SetStderrVerbosity(vtkLogger::VERBOSITY_WARNING)`
- 仅输出 WARNING 及以上级别日志

### 修复 5: 快捷键优先级与事件分发

**问题**：VTK widget 的 `event()` 方法先处理 VTK 快捷键并 `accept()` 消费事件，
`ecvKeySequences` 模态快捷键系统永远收不到按键。

**修复**（`QVTKWidgetCustom.cpp`）：
- 在 `QEvent::ShortcutOverride` 和 `QEvent::KeyPress` 中，查询 `ecvKeySequences::instance().active(seq)`
- 若存在已注册的模态快捷键，`ignore()` 让 Qt 全局快捷键系统分发
- 保证应用级和窗口级快捷键优先于 VTK 快捷键

### 修复 6: OrthoSlice 窗口无法缩小（调整后回弹）

**问题**：工具栏使用 `setFixedWidth` 固定宽度，导致 QLayout 计算出的最小宽度约 600px，
QSplitter（`setChildrenCollapsible(false)`）不允许子窗口缩小到该值以下，拖拽分隔条后自动回弹。

**修复**（`vtkOrthoSliceViewWidget.cpp`）：
- 所有工具栏控件 `setFixedWidth` 替换为 `setMaximumWidth`，允许自适应缩小
- 主布局设置 `setSizeConstraint(QLayout::SetNoConstraint)` 阻止自动最小尺寸
- 设置 `setMinimumSize(200, 150)` 合理下限
- `QVTKOpenGLNativeWidget` 设置 `setMinimumSize(0, 0)` 允许 VTK 渲染区缩小

### 修复 7: 分屏窗口点击激活/高亮失效

**问题**：`updateFrameHighlighting` 使用 `color` CSS 属性控制 `CentralWidgetFrame` 边框色，
但 Qt stylesheet 模式下 `color` 仅设置前景文本色，不可靠地映射到 `QFrame::Plain|Box` 的边框绘制。
导致点击分屏视图时边框颜色不变化、激活状态不可见。

**修复**（`ecvMultiViewWidget.cpp` / `MainWindow.cpp` / `ecvMultiViewFrameManager.cpp`）：
- 将 CSS 从 `{ color: rgb(...) }` 改为 `{ border: 2px solid rgb(...) }`
- 三处高亮函数统一使用 `border` 属性显式控制边框宽度、样式和颜色
- 非激活帧清除 stylesheet，恢复原生 QFrame 渲染

### 修复 8: Selection Tools Tooltips 不匹配裸键快捷键

**问题**：Selection tools 按钮 tooltips 仍显示 `Alt+C`、`Alt+D` 等组合键，
但实际注册的快捷键已改为裸键 `S`、`D`、`F`、`G`、`B`（对齐 ParaView）。

**修复**（`cvSelectionToolController.cpp`）：
- 所有 selection action 的 text 和 tooltip 对齐 ParaView 格式：
  - "Select Cells On (s)"、"Select Points On (d)"、"Select Cells Through (f)" 等
- Hover/Interactive 工具移除快捷键提示，与 ParaView 一致
- Hover 工具 tooltip 改为 "Use Ctrl-C/Cmd-C to copy the content to clipboard."

### 修复 9: Selection 高亮全局同步到所有视图

**问题**：在一个窗口选中 3D 网格的部分 faces 后，其他关联窗口不高亮显示选中区域。
ParaView 中 selection 是全局的，所有显示同一数据的窗口同步高亮。

**修复**（`cvSelectionHighlighter.cpp`）：
- `addActorToVisualizer()`：除主视图外，遍历 `ecvViewManager::getAllViews()` 获取所有 `vtkGLView`，
  将同一 highlight actor 添加到每个视图的 renderer
- `removeActorFromVisualizer()`：同步从所有视图的 renderer 移除 highlight actor
- `setHighlightsVisible()`：visibility 切换后刷新所有视图
- VTK actor 可安全添加到多个 renderer，无需 clone

### 修复 10: Comparative 视图 Bubble View 文本乱码

**问题**：Comparative 视图中只有一个子窗口的 hot zone 文本（"default point size"、
"default line width"）正常显示，其他三个子窗口的文本全部糊在一块（garbled）。

**根因**：`ImageVis` 使用 `vtkContext2D::DrawString()` 渲染文本，该路径依赖
`vtkFreeTypeTools` 全局单例的字体纹理缓存。当多个 `vtkRenderWindow`（各自拥有独立
OpenGL 上下文）并存时，`vtkOpenGLContextDevice2D` 的字体纹理在不同上下文间冲突，
导致文字渲染乱码。

**修复**（`VtkDisplayTools.cpp`）：
- `displayText()`：secondary views 使用 `VtkVis::addText()`（基于 `vtkTextActor`）
  替代 `ImageVis::addText()`（基于 `vtkContext2D`）。Actor ID 使用 `groupID#text`
  复合格式确保唯一性。
- `drawWidgets()` WIDGET_T2D：secondary views 同样路由到 `VtkVis::addText()`
- `removeEntities()` ECV_TEXT2D：新增 `removeBySubstring(prefix)` 清理 secondary
  views 的 vtkTextActor

**原理**：`vtkTextActor` 作为 VTK 标准 2D 覆盖 actor，每个 renderer 独立管理纹理
渲染，不受多窗口 `vtkFreeTypeTools` 字体缓存冲突影响。

### 修复 11: Orientation Marker 路由到主视图而非当前视图

**问题**：`vtkGLView::toggleOrientationMarker()` 通过共享的 `VtkDisplayTools` 单例
路由，始终操作主视图的 `m_visualizer3D`，而非调用者自身的 VtkVis。Comparative 等
多窗口视图的子窗口无法独立管理 orientation marker。

**修复**（`vtkGLView.cpp`）：
- `toggleOrientationMarker()` 改为直接调用 per-view `m_visualizer3D` 的
  `showPclMarkerAxes()` / `hidePclMarkerAxes()`，绕过 VtkDisplayTools 单例

### 修复 12: Comparative 子窗口加载 mesh 后文本被 mesh 颜色覆盖

**问题**：修复 10 解决了无 mesh 时的文本乱码，但加载 mesh 后文本再次被 mesh 颜色覆盖。

**根因**：每个子窗口的 `ImageVis`（`vtkContextActor`）仍在 renderer 中渲染。
`vtkContext2D` 渲染管线（`vtkOpenGLContextDevice2D`）在处理矩形/圆点等图元时，
mesh 加载后 OpenGL 纹理绑定和着色器状态被 mesh 渲染污染。
被污染的 GL 状态传递到后续 overlay pass 中 `vtkTextActor` 的渲染，导致文本纹理
采样到 mesh 的纹理数据。

**修复**：
- `vtkGLView` 新增 `disableContext2DOverlay()` 方法：清除并释放 `m_visualizer2D`
  （ImageVis），移除所有 `vtkContextActor` 以彻底消除 `vtkContext2D` 渲染管线
- `vtkComparativeViewWidget::createRenderSubViews()` 创建子视图后立即调用
  `view->disableContext2DOverlay()`，确保子窗口无 `vtkContext2D` 干扰
- `WIDGET_RECTANGLE_2D` / `WIDGET_POINTS_2D`：ImageVis 为 null 时自动跳过
- `vtkTextActor` 自动设置半透明深色背景（`BackgroundOpacity=0.7`）和阴影，
  替代原来由 ImageVis 绘制的背景矩形，确保文本可读性

**原理**：彻底从 Comparative 子窗口的 renderer 中移除 `vtkContextActor`，
消除 `vtkContext2D` → `vtkOpenGLContextDevice2D` 渲染路径对 OpenGL 状态的污染。
所有 2D overlay 元素统一通过 `vtkTextActor`/`vtkActor2D` 渲染，这些 VTK actor
独立管理各自的 GL 资源，不受多窗口纹理缓存冲突影响。

### 修复 13: Selection Highlight 跨视窗渲染安全

**问题**：在主窗口选中 3D 网格的部分 faces 后，selection highlight actor 被添加
到所有视图（包括 Comparative 子视图）。但 `cvSelectionHighlighter` 中使用
`rw->MakeCurrent(); rw->Render();` 直接刷新跨窗口的 `vtkRenderWindow`。
与修复 3 相同的根因：多个 GL 上下文间直接切换导致纹理句柄失效，在 Comparative
子视图的 `vtkTexturedActor2D` overlay 渲染时崩溃。

**修复**（`cvSelectionHighlighter.cpp`）：
- 新增 `scheduleAllViewsUpdate()` 静态辅助函数：遍历 `ecvViewManager::getAllViews()`，
  获取每个 `vtkGLView` 的 `QVTKWidgetCustom`，调用 `w->update()` 通过 Qt 事件循环
  安全地触发重绘
- 替换 `addActorToVisualizer()` 中的 `QTimer::singleShot` + `rw->Render()` 回调
- 替换 `removeActorFromVisualizer()` 中的 `QTimer::singleShot` + `rw->Render()` 回调
- 替换 `clearHighlights()`、`clearHighlight()`、`setHighlightsVisible()` 中的直接
  `renWin->Render()` 调用
- 替换 `setColor()`、`setOpacity()`、标签管理函数中的直接 `renWin->Render()` 调用
- 全文件零残留的直接 `Render()` 调用

**原理**：`QWidget::update()` 将重绘请求推入 Qt 事件队列，由 Qt 在正确的
GL 上下文中执行 `paintGL()` → `vtkRenderWindow::Render()`。这确保每个
`vtkRenderWindow` 始终在其所属的 GL 上下文中被渲染，消除跨上下文的纹理
句柄失效问题。

### 修复 14: Selection Highlight 同步到 OrthoSlice 视图

**问题**：OrthoSlice 的 4 个内部 renderer 不在 `ecvViewManager::getAllViews()` 中
注册（它们是单个 `QVTKOpenGLNativeWidget` 内的 viewport 分区），因此主视图的
selection highlight actor 不会传播到 OrthoSlice 的 3D 视窗。

**修复**：
- `cvSelectionHighlighter` 新增三个信号：`highlightActorAdded(vtkActor*)`、
  `highlightActorRemoved(vtkActor*)`、`highlightsCleared()`
- `vtkOrthoSliceViewWidget` 新增 `connectExternalHighlighter(QObject*)` 方法：
  接收 highlighter 并连接其信号，将 highlight actor 添加/移除到 `PERSPECTIVE_VIEW`
  renderer
- `ecvMultiViewWidget::createOrthoSliceForCell` 创建后立即调用
  `orthoView->connectExternalHighlighter(selCtrl->highlighter())`
- 2D 切面视图不同步 highlight（切面只显示横截面，3D 高亮在 2D 投影中无意义）

### 修复 15: vtkGLView 统一安全渲染（消除 Comparative 加载物体 crash）

**问题**：Comparative 子视图加载 mesh 物体后 crash，栈帧：
`safeRedraw` → `view->redraw()` → `getRenderWindow()->Render()` →
`vtkTexturedActor2D::RenderOverlay` → nvidia driver SIGSEGV。

**根因**：`vtkGLView::redraw()` 在场景图处理完成后直接调用
`m_visualizer3D->getRenderWindow()->Render()`。当 Comparative 对多个子视图依次
调用 `safeRedraw()` 时，每次 `Render()` 都在当前线程的 GL 上下文中执行，
但不保证是该子视图自己的 GL 上下文。`vtkTexturedActor2D` 的字体纹理在错误的
GL 上下文中查找导致 SIGSEGV。

**修复**（`vtkGLView.cpp`）：
- `redraw()` 末尾从 `rw->Render(); w->update()` 改为仅 `w->update()`
- `updateCamera()`、`updateScene()` 从 `rw->Render()` 改为 `w->update()`
- `resetCamera(bbox)` 和 `resetCamera()` 从 `rw->Render()` 改为 `w->update()`
- `zoomOnSelectedEntities()` 从 `rw->Render(); w->update()` 改为仅 `w->update()`

**原理**：`QWidget::update()` 将渲染请求推入 Qt 事件队列。Qt 在处理 paint event
时会自动调用 `makeCurrent()` 确保正确的 GL 上下文，然后执行 `paintGL()` →
`Render()`。这保证了每个子视图始终在自己的 GL 上下文中渲染。

### 修复 16: ImageVis 回调安全与 Render() 清理（消除 Comparative 打开即 crash）

**问题**：打开 Comparative 窗口后立即 SIGSEGV，栈帧：
`forceRenderAllSubViews()` → `w->show()` → Qt resize event →
`vtkRenderWindowInteractor::UpdateSize()` → `WindowResizeEvent` →
`ImageVis::onWindowResize()` → `ImageVis::updateImageScales()` → CRASH。

**根因**（三重缺陷）：
1. `ImageVis` 没有析构函数，不清理通过 `win->AddObserver()` 注册的
   `WindowResizeCallback`。当 `ImageVis` 被销毁后，render window 仍持有回调的
   引用，回调的 `ClientData`（raw `this*`）变为 dangling pointer
2. `ImageVis::onWindowResize()` 包含直接 `win_->Render()` 调用，在
   Comparative 多上下文环境中会触发 GL context cross-contamination
3. `ImageVis::setupInteractor()` 包含直接 `getRenderWindow()->Render()` 调用，
   在子视图初始化阶段即产生不安全的同步渲染

**修复**（`ImageVis.cpp` / `ImageVis.h`）：
- 新增 `~ImageVis()` 析构函数：设置 `m_disposed = true`，调用
  `win_->RemoveObserver(m_windowResizeCallback)` 清理 VTK observer
- 新增 `m_disposed` 标记：在 `WindowResizeCallback` 和 `onWindowResize()` 中
  检查，防止 disposed 后的野指针访问
- `onWindowResize()` 移除直接 `win_->Render()` 调用，仅执行图像缩放更新
- `setupInteractor()` 移除直接 `getRenderWindow()->Render()` 调用
- `updateImageScales()` 增加 `m_disposed` 前置检查

### 修复 17: 诊断日志清理

清除修复过程中添加的所有运行时诊断日志，保持代码整洁：
- `QVTKWidgetCustom.cpp`：移除裸键 ShortcutOverride 诊断和 ESC 按键日志
- `MainWindow.cpp`：移除 selection shortcut 的 5 条诊断日志，lambda 捕获列表
  不再包含 `keyStr`
- `vtkOrthoSliceViewWidget.cpp`：移除 gridAxes 属性值诊断日志
