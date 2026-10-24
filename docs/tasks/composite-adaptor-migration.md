# 任务文档：Composite 渲染 Adaptor 基类收敛（海量实体统一渲染通道）

> 状态：**待办**（排队于当前缺陷批次全部闭环之后）。
> 背景纪要：本会话已将相机（ccCameraSensor）与网格聚合为 composite 渲染的两个运行实例，
> 并沉淀了三条接入契约。本任务把"隐式契约"升级为"引擎侧显式 Adaptor 基类"，
> 使未来任何 ccHObject 派生类型（label、polyline、octree、点云子集、新传感器族…）
> 都能以最小成本接入聚合渲染，同时自动获得 DB 树显隐、属性栏、selection tools、
> 拾取解析的完整交互兼容。

## 0. 现状矩阵与必要性裁定（2026-10-07 实证审计）

### 0.1 类型分发与聚合覆盖（VtkDisplayTools::draw 的 if-else 链）

| ccHObject 类型 | draw 路径 | 渲染形态 | 是否已聚合 | 数量级（典型场景） |
|---|---|---|---|---|
| POINT_CLOUD (ccPointCloud) | drawPointCloud | 专用 actor（每云一 actor，VBO 直建） | **否（裁定不聚合）** | 个位数~几十，每云点数巨大 |
| MESH / SUB_MESH | drawMesh | composite 组 + 块替换（reconciliation signature 含 normals/colors/SF/materials） | **是** | 可大（多片导入/重建输出） |
| SENSOR（ccCameraSensor/GBLSensor） | drawSensor → aggregateCameraSensor | composite 组 + `#plane` 派生 actor + 回放期挂起通道切换 | **是** | 可大（lingbot 286 台） |
| POLY_LINE | drawPolyline → aggregateLineEntity | composite 组 + `#plw<width>` 独立族（2D 模式/light override 回退专用） | **是（2026-10-07 已落地）** | 可大（轨迹/标注线族） |
| LINESET | drawLines → aggregateLineEntity | 同上（与 polyline 同族共组，同宽度即同组） | **是（2026-10-07 已落地）** | 可大 |
| IMAGE (ccImage) | drawImage | 2D overlay（非 3D actor） | 否（语义外） | 少 |
| LABEL_2D (cc2DLabel) | QPainter overlay | 非 VTK 通道 | 否（语义外） | 少~中 |
| plain 容器（type=0） | 无自身几何 | 组级仅参与显隐级联 | N/A | 插件结果组 |

聚合运行实例 = **3 个**（MESH、SENSOR、LINE 家族）；`compositeRenderingSupported()` = VTK≥9 门控。

### 0.1b 线家族聚合契约（aggregateLineEntity，2026-10-07）

- 组族 `#plw<width>` 与 sensor `#lw<width>` **刻意不相交**：回放驱逐按 `"#lw"` 子串匹配，静态线在回放中保持聚合（Hybrid 隔离：动态回放通道 sensor 专属）；
- 签名字段：entity 指针 / 点或段数 / **closed**（改拓扑不改点数）/ 线宽（组属性）/ 打包解析色 / 标量色有无；几何编辑走 isRedraw()（与 mesh 同契约）；
- 回退专用：2D 模式（逐次经活动相机反投影，烘焙即冻结）、per-entity light override、容器 < kMinLeavesForComposite（正数计数缓存，与 sensor 同款）；
- 无标量色块用 `setLeafColor` 块级覆盖（专用 actor 的 SetColor 无 composite 等价）；组 actor Surface + LightingOff（与两条专用路径的终色语义对齐）；
- 必改配套：`drawPolygon/drawLines` 的 firstShow 补 `&& !isCompositeLeaf`（mesh/sensor 已有该防每帧重建检查，polyline/lineset 原缺）。

### 0.2 交互卡顿的主源已在本会话闭环（实测达标）

| 卡顿源 | 修复 | 实测 |
|---|---|---|
| 每实体每 pass 无条件 SetVisibility+Modified（579 实体/pass bounds 失效） | setLeafVisibilityIfChanged / hideShowActors 变更检测返回 | 交互 15-72ms（原 490-1104ms） |
| 选中文件夹每 pass 递归 BB（284ms） | HIERARCHY_OBJECT 跳过 BB | 选择 51ms |
| 强制 pass 全量重转换 | 状态变更走非强制 + O(changed) 选择重建 | 全不选 25ms |
| 每次点击强制 Render + 拾取路径日志 IO | deferred timer 去 Render；[cpuPick]/[pointPicking]/[convert] 清理/降级 | 拾取响应恢复即时 |

### 0.3 必要性裁定（Hybrid 渐进，非全量统一）

- **统一 composite 不消除 draw() 遍历**：每实体仍走 ccHObject::draw 做状态对账，聚合只折叠 VTK 侧 per-actor 成本——而该成本已被 O(changed) 变更检测压到稳态近零。**因此统一不是性能刚需，是结构收敛**。
- **增量性能收益**仅在"大量同质静态"的新类型面（如数百 polyline 段/大 lineset）——当前主场景（lingbot）的两个大户已聚合。
- **全量统一不可行**（实证约束）：per-block 属性面仅 color/visibility/opacity——点大小/光照 override/texture 无 per-block 等价（sensor 的 light override 已实证走"差异化回退专用 actor"）；动态位姿（live/回放）依赖挂起通道切换逃生门；多窗口下 registry per-VtkVis，块几何内存 × 窗口数。
- **裁定**：采用 **Adaptor 基类 + Hybrid 回退**——静态同质可烘焙类型接入聚合，差异化/动态类型声明回退专用 actor；不做"任意类型一律聚合"。

## 1. 目标

1. 引擎侧提供 `CompositeLeafAdaptor`（名称待定）基类/接口：新类型只需提供
   **几何回调**（每实体 → vtkPolyData + 渲染属性）即可进入聚合渲染；
2. 现有两实例（相机线框、网格块）迁移到该基类，行为逐字节不变（回归门保护）；
3. 交互全兼容作为硬验收标准：DB 树显隐（单个/文件夹/级联）、删除/添加、
   属性面板（Drawing scale/opacity/颜色）、selection tools（点/三角形/RECT/实体拾取）、
   多视图、保存加载。

## 2. 三条接入契约（Adaptor 必须内建，调用方不再手写）

| # | 契约 | 现状落点 | Adaptor 化后 |
|---|------|---------|--------------|
| C1 | `drawMeOnly` 输出 polydata 并 `setLeafBlock(leafId, groupId, data)` 注册 | `VtkVis::aggregateCameraSensor` / mesh 聚合分支 | Adaptor::buildGeometry(context) → 引擎统一注册（含 same-source key） |
| C2 | 派生 actor 遵守 `#后缀` 命名空间：`#plane`（近平面面片）、`#lwN`（线宽分组组键）、`-normal`；位姿 `applyGLTransform/clearGLTransform`、可见性 `hideShowEntities`、拾取 `getIdByActor` 后缀剥离三管道全覆盖 | 各管道内手工枚举后缀 | Adaptor 声明派生 actor 列表，引擎按声明自动覆盖三管道 |
| C3 | 状态经 `setLeafVisibility/Opacity/Color/WorldTransform`（变更检测、`setLeafBlock` 同组替换迁移属性） | 手工调用 | Adaptor::applyState(state) 统一入口 |

## 3. 迁移清单（逐类型，按风险升序）

1. **网格块**（已聚合）：迁移 + 回归门（test_composite_block_registry 7 用例扩展）；
2. **相机传感器族**（线框 + `#plane` + 骑行态 `applyViewport` 交互）：最复杂，
   需保持回放期通道切换（setSensorAggregationSuspended）、出生门、几何签名
   （CompositeLeafKey 必须含所有烘焙输入：pointCount/线宽/颜色/sensorScale）；
3. **线家族 polyline/lineset**（✅ 2026-10-07 已落地，见 §0.1b）：aggregateLineEntity 统一接入，`#plw` 独立族；
4. **label 2D**（当前专用 overlay）：评估聚合收益（数量级小则保留专用）；
5. **octree / 点云**：点云不建议聚合（逐点交互多、引擎 per-pass gate 已高效），
   文档化"不迁移"裁定及理由；
6. **新类型**：以 Adaptor 接入示例（一个最小 demo 类型 + gtest）。

## 4. 回归与验收

- `test_composite_block_registry` gtest：现有 7 用例 + 新增 Adaptor 契约用例
  （几何回调注册/派生 actor 三管道/状态应用/类型无关拆除）；
- RPC 运行时清单（已有驱动面）：文件夹显隐、回放同步、删除即时消失、
  滑杆隔离、`view.pickEntity` 实体拾取、像素定量 + 截图目视；
- 性能门：选择/勾选端到端延迟不劣于当前基线（选择 ~60ms、文件夹操作 ~15-70ms）；
- 兄弟插件编译探针（qDA3/qFreeSplatter/qJSON_RPC/qAnyLabeling）零错误。

## 5. 明确不做

- 点云聚合（见迁移清单 4）；
- 修改 viewID 后缀命名空间（`#plane`/`#lwN`/`-normal` 为既有数据格式）；
- 在缺陷批次未闭环前启动本任务。

## 6. 前置条件

- 当前全部渲染/交互缺陷闭环（见会话记录：物化首绘、回放同步、选中高亮、
  HiDPI 坐标契约、selection tools 拾取定位）；
- 工作区分批提交（引擎/插件/COLMAP fork 分离）。
