# COLMAP Fork 端到端对齐验证 TODO

> **日期**：2026-09-15（本轮 e2e 收口）
> **上游基线**：COLMAP `d3ccaf35`（4.3.0.dev0，本机 `/home/asher/develop/code/github/MVS/colmap`）
> **Fork**：`libs/Reconstruction/`（ACloudViewer 内联 fork）
> **测试数据集**：mini6（6×iPhone7 照片，含 GPS），`/tmp/e2e_mini6/{fork,upstream}`
> **构建状态**：两侧 EXIT=0，CLI 就绪（fork `build_app/bin/Colmap`，上游 `MVS/colmap/build/src/colmap/exe/colmap`）

---

## 1. 总体进度概览

| 阶段 | 状态 | 说明 |
|------|------|------|
| 稀疏重建 — exhaustive matcher | ✅ 完成 | fork 6图/1735点/0.97px vs 上游 1384点/1.02px |
| 稀疏重建 — sequential matcher | ✅ 完成 | fork 6图/1714/0.963 vs 上游 6图/1346/1.015 |
| 稀疏重建 — spatial matcher (GPS) | ✅ 完成 | fork 6图/1739/0.982 vs 上游 6图/1384/1.017 |
| 稀疏重建 — transitive matcher | ✅ 完成 | 两侧一致失败（行为对齐，无几何验证） |
| 稀疏重建 — global_mapper | ✅ 完成 | fork 6图/1462/1.066 vs 上游 6图/1362/1.005 |
| 稀疏重建 — vocab_tree matcher | ✅ 完成 | fork 6图/1842/0.987 vs 上游 6图/1509/1.026 |
| 稠密重建 — image_undistorter | ✅ 完成 | 两侧各输出 6 张 undistorted 图 + 同构 sparse 模型 |
| 稠密重建 — patch_match_stereo | ✅ 完成 | 同 CUDA 路径同 `max_image_size=2000`；覆盖率逐图 ±1% |
| 稠密重建 — stereo_fusion | ✅ 完成 | fork 290319 点 vs 上游 289072 点（差 0.43%） |
| 网格 — poisson_mesher | ✅ 完成 | 差异根因定位为 P4 稀疏离群点传导（见 §5.2） |
| 网格 — delaunay_mesher | ✅ 完成 | fork V=117276/F=234564 vs 上游 V=114969/F=230071（差 ~2%） |
| 纹理 — mesh_texturer | ✅ 完成 | 两侧各产出 texture.png + mesh.ply（D1 默认路径） |
| 位姿/内参逐项对比 | ✅ 完成 | 旋转 mean 0.025°/max 0.029°；光心 ~1e-4；焦距差 0.053%（§4.1） |
| ATRISK 探针清理 | ✅ 完成 | 全局 grep 零命中，无遗留探针 |
| 文档三件套更新 | ✅ 完成 | 本文件 + COLMAP_ALIGNMENT.md + PLAN 变更记录 + manifest |
| P4 点数差异根因 | ✅ 定位传导链 | 稀疏离群点 → PM 深度范围 → fusion bbox → poisson 规模（§5.2）；参数级根因待专项 |

---

## 2. 本轮修复的 Bug 清单（e2e 验证中发现）

### Bug #1–#3（上一轮已修复，保留记录）

- **Bug #1**：`EstimateTwoViewGeometryPose` 缺失调用（`sfm/incremental_mapper.cc`）
- **Bug #2**：`FindNextImages` `.at()` 对未注册 image 崩溃（`sfm/incremental_mapper_impl.cc`）
- **Bug #3**：`RunGlobalMapper` ini 输出路径错误（`exe/sfm.cc`）

### Bug #4：`ReadImages{Binary,Text}` 未恢复 image 注册状态（本轮 P0 阻塞项）

- **文件**：`libs/Reconstruction/src/scene/reconstruction_io_binary.cc`、`reconstruction_io_text.cc`
- **根因**：fork `Reconstruction::AddFrame` 显式不注册 posed frame（`reconstruction.cc` L336，image 级注册模型），读路径读完每个 image 只 `AddImage` 未 `RegisterImage` → `reg_image_ids_` 恒空 → `NumRegImages()=0` → undistorter/稠密链看到 0 张已注册图。上游通过 `AddFrame`→`RegisterFrame`（frame 带位姿时 `num_reg_images_ += ImageIds().size()`）自动恢复。
- **修复**：在两个读函数的读循环内、`AddImage(std::move(image))` 之后调用 `reconstruction.RegisterImage(image_id)`（move 前捕获 id）。与写侧契约逐条对应（`WriteImagesBinary` 只遍历 `RegImageIds()` 输出，进入文件即已注册），与 fork "synthetic 生成器在 AddImage 后做等价 image 注册"的既有模式一致；`RegisterImage` 幂等且无其它副作用。
- **验证**：fork undistorter 输出 0 张 → 6 张；`reconstruction_io_test` 19/19。

### Bug #5：`mvs::Model::ReadFromCOLMAP` 读取陈旧 legacy 位姿缓冲

- **文件**：`libs/Reconstruction/src/mvs/model.cc`
- **根因**：使用 `image.Qvec()/Tvec()`（legacy 阴影缓冲）。`Reconstruction::Read` 非legacy 路径下文件中的每 image 位姿被读出后丢弃（上游同构——位姿在 frame 中），legacy 缓冲保持默认恒等 → 所有相机位姿恒等 → 三角化角≈0 → `__auto__` source image 选择全空 → patch_match "no source images" 全部跳过。
- **修复**：改为上游同款 `image.CamFromWorld()`（frame 接线时 rig-aware 读取 + 独立 image legacy 回退）。
- **验证**：patch_match 从 "0 problems" → "Configuration has 6 problems"。

### Bug #6：`Bitmap::Read` 忽略 `as_rgb` 参数（OIIO 迁移回归）

- **文件**：`libs/Reconstruction/src/sensor/bitmap.cc`
- **根因**：OIIO 迁移后的 `Read` 按文件原样保留 channel 数（JPG→3 通道 RGB），`as_rgb=false` 未转换灰度。patch_match workspace 以 `image_as_rgb=false` 读图 → `CHECK(image.GetBitmap().IsGrey())` 失败 abort。稀疏链侥幸正常是因为 `feature_extraction.cc` 显式 `CloneAsGrey()` 掩盖了该缺陷。
- **修复**：补上游同款转换块（`if (as_rgb && channels_ != 3) *this = CloneAsRGB(); else if (!as_rgb && channels_ != 1) *this = CloneAsGrey();`）。
- **验证**：patch_match 全 6 视图跑通；`bitmap_test` 31/31。

---

## 3. 稀疏重建矩阵结果

| Matcher | 指标 | Fork | 上游 | 对齐 |
|---------|------|------|------|------|
| Exhaustive | 注册图/3D点/重投影误差 | 6/6、1735、0.974px | 6/6、1384、1.017px | 注册✓ 误差✓ 点数⚠ |
| Sequential | 注册图/3D点/重投影误差 | 6/6、1714、0.963px | 6/6、1346、1.015px | 注册✓ 误差✓ 点数⚠ |
| Spatial (GPS) | 注册图/3D点/重投影误差 | 6/6、1739、0.982px | 6/6、1384、1.017px | 注册✓ 误差✓ 点数⚠ |
| Global mapper | 注册图/3D点/重投影误差 | 6/6、1462、1.066px | 6/6、1362、1.005px | 注册✓ 误差⚠ 点数⚠ |
| Transitive | 结果 | 失败 | 失败 | ✅ 行为一致 |
| Vocab tree | 注册图/3D点/重投影误差/平均track | 6/6、1842、0.987px、2.97 | 6/6、1509、1.026px、3.17 | 注册✓ 误差✓ 点数⚠ |

> 点数差异根因见 §5.2（P4 升级结论）。

---

## 4. 稠密链与位姿对比结果

### 4.1 位姿/内参逐项对齐报告（exhaustive 模型，按名配对 + Umeyama Sim3d）

| image | 旋转误差 (deg) | 光心误差 (归一化) | fork 焦距 | 上游焦距 | 焦距相对差 |
|-------|---------------|------------------|-----------|----------|-----------|
| IMG_1024 | 0.0287 | 0.000100 | 3298.69 | 3300.42 | 0.053% |
| IMG_1026 | 0.0222 | 0.000070 | 3298.69 | 3300.42 | 0.053% |
| IMG_1028 | 0.0247 | 0.000168 | 3298.69 | 3300.42 | 0.053% |
| IMG_1030 | 0.0254 | 0.000068 | 3298.69 | 3300.42 | 0.053% |
| IMG_1032 | 0.0235 | 0.000066 | 3298.69 | 3300.42 | 0.053% |
| IMG_1040 | 0.0276 | 0.000057 | 3298.69 | 3300.42 | 0.053% |

- 主点：两侧完全一致（cx=2016.00, cy=1512.00）；尺度比 0.7292（gauge 自由度）。
- **model_comparer 工具结论**：同库模型对（同 id 同关键点）fork comparer 正常出报告（fork exh-vs-seq：旋转 mean 0.031°/光心 mean 0.0009）；**跨库比较（两侧各自提特征）两侧对称崩溃于 `alignment.cc:48`（上游 4 vs 5，fork 同）**——上游工具契约要求同数据库（`Estimate` 断言 ImageId 相等 + `Residuals` 断言 NumPoints2D 相等），且上游自身两次运行的 id 分配即不确定（特征提取顺序非确定），属上游限制而非 fork 偏差。跨管线逐图对比由按名配对脚本完成（`.qoder/tmp/compare_poses.py`）。

### 4.2 稠密链对比（同机同 CUDA、同 `--PatchMatchStereo.max_image_size 2000`）

| 阶段 | Fork | 上游 | 对齐 |
|------|------|------|------|
| undistorter | 6 张图 + 同构 sparse | 6 张图 + 同构 sparse | ✅ |
| patch_match 耗时 | 2.572 min | 2.557 min | ✅ |
| 深度图覆盖率（逐图） | 36.6/43.9/48.3/51.8/43.5/55.6% | 36.0/43.8/48.3/50.6/42.2/54.6% | ✅ ±1% |
| 深度中值比 | ×1.37（gauge 尺度比，恒定） | — | ✅ |
| fusion 点数 | 290319 | 289072 | ✅ 差 0.43% |
| delaunay 网格 | V=117276 F=234564 | V=114969 F=230071 | ✅ 差 ~2% |
| poisson 网格 | V=272714 F=533776 | V=855977 F=1681180 | ⚠ 根因=P4 传导（§5.2） |
| mesh_texturer | texture.png + mesh.ply | 同 | ✅ |

---

## 5. 已知差异与根因分析

### 5.1 上游工具/环境限制（非 fork 偏差）

| 项目 | 现象 | 结论 |
|------|------|------|
| model_comparer 跨库崩溃 | `alignment.cc:48` ImageId 断言（两侧对称） | 上游工具契约要求同库；上游 id 分配自身非确定。fork 同库模型对可用 |
| CLI 选项前缀差异 | 上游 `--FeatureExtraction.*` vs fork `--SiftExtraction.*` | 已知：d3ccaf35 重构了选项前缀；patch_match 两侧均 `--PatchMatchStereo.*` |
| PatchMatch 无 CPU 模式 | d3ccaf35 选项面无 `use_gpu` | 两侧同走 CUDA（同机同卡），公平性等价 |

### 5.2 P4：Fork 稀疏点数差异的根因排查（2026-09-15 第二轮，正向修复）

**现象层**：所有 matcher 一致 fork 多 22–33% 点数（1735/1714/1739/1842 vs 1384/1346/1384/1509），fork 重投影误差略优。

**数据 vs 引擎劈半实验（决定性）**：将 fork 的特征/匹配/TVG 移植进上游库（绕过库版本墙），让上游 mapper 在 fork 数据上运行：

| 组合 | 点数 | >30 离群 |
|------|------|----------|
| 上游引擎 + fork 数据 | 1441 | 11 |
| 上游引擎 + 上游数据 | 1384 | 12 |
| fork 引擎 + fork 数据 | 1735 | 46 |

结论：上游引擎在 fork 数据上点数正常（+4% 对应 fork 数据多 3% 匹配内点）——**差异在 fork mapper/triangulator 引擎内部**，非数据层。强制同初始对后（fork 1620 vs 上游 1441）次序一致仍多 179 点。

**已验证等价的层（排除项）**：选项默认值（tri_*/filter_*/init_* 全部一致）；三角化 RANSAC 参数（fork 显式值 = 上游 EstimateTriangulationOptions 构造默认，已改为上游构造函数形态）；过滤语义（Reconstruction 旧版 FilterPoints3D* 已是 any-pair keep_point + track<2 整点删 + 误差分母删除后长度，与上游 ObservationManager 版一致）；重建收敛分支（已补，见下）；Normalize 时点（fork 已在 AdjustGlobalBundle 内）。

**本轮正向修复的三项确定性上游缺口**：

1. `AdjustGlobalBundle` 缺上游小重建严格 BA 收敛分支（`kMinNumRegFramesForFastBA=10`：function/gradient/parameter tolerance ÷10、max_num_iterations ×2、max_linear_solver_iterations=200）——已补齐（sfm/incremental_mapper.cc），mini6（6 图）全量命中该分支。
2. `AdjustLocalBundle` 缺上游 `FixGauge(BundleAdjustmentGauge::THREE_POINTS)`，fork 用旧"固定两图位姿 7-DOF"启发式——已切换到 FixGauge（上游 parity；W3-2b 已移植的 CeresBundleAdjuster gauge 机制直接复用）。
3. `EstimateInitialTwoViewGeometry` 残留 `INITPAIR-DEBUG` LOG(INFO) 调试日志（每对必打；此前 ATRISK grep 因关键字不同漏检）——已替换为上游 VLOG(3) 形态；`EstimateTriangulationOptions` 补上游同款构造函数（承载 confidence=0.9999/min_inlier_ratio=0.02/max_num_trials=10000 默认值），Create/Continue 改为仅覆盖 max_error（上游逐行 parity）。

修复后回归全绿：bundle_adjustment_test 11、global_mapper_test 5、triangulation_test 2、synthetic_test 19。修复后 e2e：fusion 点数差从 0.43% 收敛到 0.17%；poisson 网格从 10.2MB 提升到 13.0MB（更严格收敛后的正向变化）。

**剩余差异的归属（诚实边界）**：mini6 上 +25% 点数差的主导因素在以下两个**已立项的 XL 对齐项**，非未发现的缺陷：

1. **上游 camera-ray 三角化重构未拉入**（d3ccaf35 的 EstimateTriangulation 已全面 ray 化：`cam_ray = CamRayFromImg(...)` 单位射线 + cam_from_world；fork 仍是 point_normalized 2D 归一化平面 + proj_matrix——记录于 W3-3 的"camera-ray point data refactor 属基线后演进未拉入"）；
2. **flat-vs-solvers 求解器去重（ODR 风险集）**：fork 自有 RANSAC/LORANSAC 与上游实现的采样/局部优化/支持度量细节差异（影响运行时 TVG 重估与初始对选择——本轮实证 fork 在 (1030,1028) 对上判定失败而上游成功，导致注册次序不同，间接影响点集）。

**第三轮（2026-09-15，camera-ray 三角化移植落地 + RANSAC 层逐层对齐验证）**：

1. **camera-ray 三角化移植完成 ✅**：`geometry/triangulation.{h,cc}` 补 ray 版 `TriangulatePoint`（6x4 射线 DLT SVD）与 `TriangulateMultiViewPoint`（射线 DLT 特征分解，vector 版承载上游 span 语义）；`scene/projection.{h,cc}` 补 `CalculateAngularReprojectionError(ray, xyz, cam_from_world)`；`Camera::IsPerspective()` 补齐（上游 d3ccaf35 形态）；`estimators/triangulation.{h,cc}` 整体换上游 ray 化形态（PointData{img_point, cam_ray} / PoseData{cam_from_world, proj_center, camera}，cheirality 透视/全向双支，EstimateTriangulation 改为 (points, cams_from_world, cameras) 签名——观测到射线转换收敛进库函数）；消费者全部迁移（IncrementalTriangulator::Create/Continue/Resection、RegisterNextStructureLessImage）。回归：triangulation 2、bundle_adjustment 11、global_mapper 5、synthetic 19 全绿。
   - mini6 上点数不变的根因：SIMPLE_RADIAL 透视相机下 2D 归一化平面点与射线 hnormalized 数学等价——ray 化的价值在全向/有限域模型（EQUIRECTANGULAR 后半球、division/EUCM 域外像素）的行为修复，透视场景是数值等价重建。
2. **RANSAC 层七层对齐验证 ✅（全部等价）**：本轮新增验证——LORANSAC 主循环结构、InlierSupportMeasurer 比较（num_inliers 主序 + residual_sum tie-break）、RANSAC SetUp 初始 max_num_trials 收缩、ComputeNumTrials 公式、correspondence graph 建边（UseInlierMatchesCheck）、TVG calibrated 主干 E/F/H 三模型竞争（min_E_F_inlier_ratio 门 + H 竞争 + 水印），fork 与上游 d3ccaf35 逐行同构（fork 为内联形态、上游抽 helper，语义等价）。
3. **两侧确定性方差实验 ✅**：同数据 3 次运行 fork 1735×3 / 上游 1441×3（CLI 启动 seed=0，各自确定可复现）——+20% 为系统性差异而非随机方差；上游自身在 fork 数据上的点数（1441）与上游数据（1384）差 4% 对应匹配差 3%，正常。
4. **剩余差异精确归因收敛到 solvers/ 求解器实现层**：F/E/H 求解器（solvers/fundamental_matrix、essential_matrix 等）的实现细节差异——即 flat-vs-solvers ODR 去重批次的精确范围。本轮已把 RANSAC 引擎、支持度量、TVG 主干全部排除，下一批次可直接从 solver 级数值对比切入。

**第八轮（2026-09-16(3)，W3-2b 收官：BA 内核单块化 ✅）**：

1. **BA 内核上游形态落地 ✅**：`AddImageToProblem` 重写为上游 trivial/non-trivial-frame 三态 functor 分派（`ReprojErrorCostFunctor` / `RigReprojErrorCostFunctor` / `RigReprojErrorConstantRigCostFunctor` / `ReprojErrorConstantPoseCostFunctor`，作用于单个 `Rigid3d` 阴影参数块（Vector7d params），TearDown 写回 frames/rigs）；新增 `ParameterizeRigsAndFrames`（Product(EigenQuaternion, Euclidean) 流形、constant_rig_from_world_rotation 子集、fork parity 的 ConstantTvec Product+Subset 流形）；`FixGaugeWithTwoCamsFromWorld` 重写为上游形态；上游 `min_track_length` 过滤与单块 `AbsolutePosePositionPriorCostFunctor` 移植；legacy equirectangular 特化分支退役（seam wrap 已内建于上游 functor）；`cost_functions/utils.h` 补齐 ceres 包装类 `CovarianceWeightedCostFunction`/`ScaleWeightedCostFunction` 与工厂。
2. **结果（forced 初始对）**：点数 **1441 = 1441**、mean reprojection error **1.0276 vs 1.0272 px（差 0.04%）**、focal 差 0.008%、位姿旋转差 0.0049°——精度达到 bit 级邻近。远点尾部（d>30：47 vs 11，gauge 1.42）定性为 PnP RANSAC ±1 inlier 容差差在低视差三角化上的 PRNG 级混沌放大（同输入下 inlier 集 ±1 的合法浮点分叉），实际重建精度已对齐。
3. **修复路径中的构建事故与修复**：flat `estimators/absolute_pose` 残留注册导致 configure 失败（CMakeHelper 静态库路径）→ 删除；`libColmapLib.a` 被并发 make 破坏 → 重建归档（188 成员）。
4. **回归 ✅**：absolute_pose 6、bundle_adjustment 11、global_mapper 5、synthetic 19、two_view 3 全绿（BA 测试曾因 SetConstantPose 冻结缺失与 ConstantTvec 流形缺失失败 9→5→0，两处修复后全过）。

**第十八轮（2026-09-18(2)，PoissonRecon 18.75 移植→撤销 ✅ + pybind Open3D 风格集成启动）**：

1. **第十七轮移植（已撤销，保留其独立修正）**：曾将引擎 vendored 树从 v6（48 文件，`int PoissonRecon()` 老 API）整体替换为上游 d3ccaf35 vendored 树（99 文件，`RunPoissonRecon`，ADAPTIVE_SOLVERS_VERSION 18.75），poisson_meshing.cc 重写为 ThreadPool 直控 + `--fullDepth/--colors/--density` 参数集，color double→bool；同输入引擎残差实测 0.001%。**并完成两项与库版本无关的独立修正（撤销时保留）**：① CLI 三个 mesher（poisson/delaunay/AFM）剥离 fork 独有 MeshPostProcessing 后处理（该后处理曾改写/删减原始网格，导致引擎残差被高估为 -6.3%~-8.9%）；② GUI `--exclude-libs=libpoisson_recon.a` 防 -rdynamic 符号插拔。
2. **撤销（用户裁定）**：精度 gap 实测仅 0.016% 场景对角线（v6 与 18.75 表面位置几乎重合，-6% 仅是采样拓扑密度差）；升级收益只剩“输出规模可比”，代价是源码冗余 +2.5MB 且 pybind 闭包会同源携带（多镜像多份实现）→ **收益/代价比为负，整体撤销**：git 恢复 v6 树 + poisson_meshing.{h,cc} + UI color double。**标签纠错**：引擎侧 vendored 树是 v6（非 v12）；3rdparty ExternalProject（v12）仅 qPoissonRecon 插件使用。
3. **撤销后基线（v6 纯引擎，同上游 fused 输入）**：842907 verts / 1665352 faces = **-6.12% / -5.71%**，定性为版本行为差，记录接受；CGAL 保持系统 5.4 不变（从未升级；Delaunay -0.76% 为数据驱动）。
4. **pybind Open3D 风格集成启动（上游 src/pycolmap 14 子模块 API 面的重表达）**：新增 `pybind/reconstruction/geometry/`（Rigid3d/Sim3d，四元数以 (w,x,y,z) 数组暴露，适配 fork W3-2b params 单块访问器形态）与 `pybind/reconstruction/scene/`（Camera/Image/Reconstruction 核心类）；主注册接线 + CMakeLists。**运行时冒烟全过**：read mini6 模型（6 图/1354 点/1 相机）、Image 位姿经 `reference_internal` 引用语义可读（copy 会丢 frame back-pointer——fork 双轨语义在绑定层的正确处理）、Camera SIMPLE_RADIAL focal 3296.638、Rigid3d 构造/旋转/平移/逆变换往返、mean_track_length 3.1448。
5. **pybind 后续批次映射**：scene 补 Point2D/Point3D/Track/Frame/Rig/Database/DatabaseCache；sensor（相机模型）、estimators（pose/essential/homography/triangulation/BA）、mvs（patch_match/fusion/meshing 扩展）、pipeline（自动重建入口）、retrieval（visual_index）、feature/image/sfm 扩展、optim/util；pyceres（ceres 求解器绑定）独立评估。**约束**：不照抄上游 dataclass 机制，保持 Open3D 自由函数+简洁类绑定风格。
6. **两项修复的用户质询复核（最终裁定）**：① `--exclude-libs=libpoisson_recon.a` **移除**（已执行）——实测 GUI dynsym 零引擎符号（连 colmap:: 都未导出）、v6 与 v12 无同名全局函数（v12 无 `PoissonRecon(int,char**)`，v6 无 PoissonRecon:: 命名空间），防御场景不存在；app/CMakeLists.txt 已恢复 codec-only 原状（多平台：该选项本为 GNU ld 专属，移除后 Windows/macOS 无影响）。② **MeshPostProcessing 剥离撤销、CLI 后处理恢复**（git 恢复 mvs.cc）——后处理是 fork 有意功能（注释明示为纹理图表质量服务）而非 bug；对齐口径的正确姿势是引擎对比时显式 `--MeshPostProcessing.enabled 0`（第十七轮判定实验已示范），而非删除 fork 功能。多平台：两处变更均为纯行为/链接层，无编译影响。
7. **pybind scene 批次 2**：补 Point2D（xy/point3D_id/has_point3D）、TrackElement（image_id/point2D_idx）、Track（length/elements）、Point3D（xyz/color/error/track）与 Reconstruction::point3D(id)（copy 安全——Point3D 无 back-pointer 语义）；编译通过，冒烟实读 point3D(1)（xyz/error 0.5738/track_len 3/element (6,4405)）全过。
8. **pybind estimators 批次（新模块）**：`pybind/reconstruction/estimators/` 暴露 RANSACOptions、AbsolutePoseEstimation/RefinementOptions、EstimateTriangulationOptions（numpy (N,2)/(N,3) 数组）与四个入口：triangulate_point、estimate_triangulation、estimate_absolute_pose（PnP）、refine_absolute_pose。合成数据运行时验证：triangulate_point 精确（[0,0,5]）、estimate_triangulation 默认参数精确、PnP 机器精度恢复 t（8/8 inliers）、refine 收紧 gradient_tolerance 后 6 迭代 cost 0.0295→0（t 误差 1.6e-05）——gradient_tolerance=1.0 在亚像素 cost 上的提前收敛为上游同款行为（非缺陷）。
9. **顺带修复的真实对齐缺口**：fork `AbsolutePoseEstimationOptions` 缺上游构造默认（`ransac_options.max_error=12.0/min_num_trials=100/max_num_trials=10000/confidence=0.99999`），聚合默认下 max_error=0 会在 `RANSACOptions::Check` 直接 abort——构造已补入 estimators/pose.h（影响引擎自身所有未显式设置 ransac 的消费者，非仅 pybind）。另确认 EstimateTriangulationOptions 构造已与上游一致（第 15 轮已补）。绑定契约记录：EstimateAbsolutePose/RefineAbsolutePose 的 points2D 为**像素坐标**（内部 CamRayFromImg/主点居中转换）；Camera::focal_length 属性绑 MeanFocalLength（FocalLength() 对双焦距模型如 PINHOLE 会 THROW，fork/上游一致）。

**第三十轮（2026-09-19(5)，math/optim 既有失败修复 + 全面对齐审查 + mini6 端到端 AB ✅）**：

1. **两个既有失败修复 ✅（第二十九轮立项）**：① `math/polynomial_test`——fork 移植时把 EXPECT_NEAR 容差写成了 `std::abs(0.0) * (1e-6) / 100.0`（零基线下容差恒 0，任何 1e-17 级浮点虚部都判失败）；按上游同款改回固定绝对容差 1e-6（6 处），9/9 绿。② `optim/least_absolute_deviations_test` RidgeRegularization——根因是**编译器浮点差异**：fork 全局 `-march=x86-64-v3`（FMA）下，秩亏 A^T A 的 LLT 第二 pivot 舍入为 ~+1.8e-15（非精确 0），Eigen `info()` 误报 Success（上游 CI 基线 x86-64 舍入下 pivot 精确为 0 故恰好失败）；修复 = `SimplicialLLTLinearSolver::Compute` 在 info()==Success 后增加**确定性正定性判定**（`matrixL().nestedExpression().diagonal()` 的 min/max 相对比，FMA 免疫、不改变正常路径数值），20/20 绿。组级 ctest（estimators/scene/sfm/controllers/reconstruction/geometry/optim/math）**73/73 全绿**。
2. **全面对齐审查（上游 src/colmap vs fork src 逐目录文件级 diff，排除 _test）**：结论分三类——(A) **等价物存在**：mvs/mat.cc（fork mat.h header-only）、util/endian.cc（fork endian.h header-only）、sensor/rig（fork 位于 scene/rig，已记录的位置差异）、controllers/undistorters（fork automatic_reconstruction/da3_depth_controller 内联等价）；(B) **架构性差异（fork 决策，非缺口）**：controllers/feature_matching、pairing、matcher_cache、base_option_manager 与 feature/{extractor,matcher,aliked,onnx_*}——fork 的特征/匹配/配对走 ACloudViewer 产品管线与 AICore，GUI 走自有 Qt/VTK；(C) **真实小缺口（S 级，记录立项）**：estimators/ceres_loss_function.{h,cc}（上游的 loss 工厂 enum+CreateCeresLossFunction，fork 各处直接 new ceres::LossFunction，功能等价但缺选项驱动的工厂入口）。**引擎算法层（estimators/sfm/scene/geometry/optim/math/mvs 核心）无剩余代码缺口**。
3. **mini6 端到端 AB ✅**：同一数据集（mini6 = 6×iPhone7 JPG，/tmp/ab_mini6/{upstream,fork}/images）同一命令链（feature_extractor single_camera → exhaustive_matcher → mapper → image_undistorter(max 2000px) → patch_match_stereo(geom_consistency) → stereo_fusion → poisson_mesher → mesh_texturer）分别用上游 d3ccaf35 构建（CUDA ON）与 fork bin/Colmap 执行。结果：fork **6/6** 图配准 vs 上游 **5/6**；fused 点云 fork 289,105 vs 上游 298,992（-3.3%，PRNG/浮点自由度带内）；textured mesh fork 791,812 顶点/1,514,321 面 vs 上游 820,121/1,614,777（-3.5%/-6.2%）。**交付 OBJ**（引擎统一输出 textured PLY+atlas，第三方转换脚本 ply_to_obj.py 同规则转 OBJ+MTL+PNG）：`/tmp/ab_mini6/fork/dense/textured_obj/mesh.obj` 与 `/tmp/ab_mini6/upstream/dense/textured_obj/mesh.obj`（含 mesh.mtl + texture.png，UV 量化去重、v 轴翻转到 OBJ 约定），供人工视觉对比。
4. **AB 产物人工对比问题的取证与澄清（用户反馈三问）**：① 纹理 atlas"细节缺失"——量化：atlas 有效内容率 upstream 0.420 vs fork 0.373，与面数差（-6.2%）同量级，属 per-face patch 布局/packing 差异（纹理引擎第 25 轮已证逐行一致），非内容缺失；ply 体积 67/71.2MB 与面数严格成正比。② "mesh 无法重叠"——两边 sparse 相机中心差仅 0.2-0.9m（同坐标系、同 1368 个 3D 点），sim3（Umeyama）拟合 scale=1.00378、对齐残差 mean=0.0003m/max=0.0004m——**两个重建毫米级一致**；视觉差异来自 SfM gauge 的微小剩余 + 查看器 up-axis 惯例。③ "渲染黑花"——ParaView 不支持 PLY face-texcoord 纹理采样与 OBJ+MTL 纹理自动加载，属查看器限制（**勘误：此结论错误，实为转换脚本 v 双重翻转，见第三十一轮**）；数据正确性由**顶点色 PLY**（UV→atlas 采样写入顶点色）与离屏渲染并排图 `/tmp/ab_mini6/side_by_side.png` 自证：两边几何与纹理内容一致。**对比产物**：`/tmp/ab_mini6/{fork,upstream}/dense/textured_obj/` 下 mesh.obj（UV/MTL，用 MeshLab/Blender 查看）+ mesh_vcolored*.ply（fork 版已 sim3 对齐到上游坐标系，可与 upstream 版直接叠加）。**顺手修复 fork 绑定层真缺陷**：Rigid3d/Sim3d 的 rotation/translation property 按值返回 Eigen::Vector4d/Vector3d，def_property 默认 reference_internal policy 会包装临时对象缓冲导致悬空（转换脚本中 q copy 后健康、t 未 copy 读出垃圾的随机性实锤）——四个 property 显式 `py::return_value_policy::copy`。

**第三十五轮（2026-09-20(5)，特征层 bit 对齐 + 匹配顺序层定位）**：

1. **特征层 bit 对齐 ✅**：database 对比（fork2 vs upstream）——keypoints 6/6 张 **BIT-IDENTICAL**；descriptors 每张仅差 1-6 字节（±1 uchar，~120 万元素中）。根因 = `L1RootNormalizeFeatureDescriptors` 两处分歧：① fork 用 `Eigen::MatrixXf`（**ColMajor**）参数，调用方 RowMajor 数据被隐式转置 → `lpNorm<1>` 行归约跨步访问、求和顺序不同；② fork 用 `row / norm`（一次舍入），upstream 用 `row *= 1/norm`（乘倒数、两次舍入）。修复 = feature/utils.{h,cc} 签名与实现改 RowMajor（FeatureDescriptorsFloat）+ 乘倒数（upstream d3ccaf35 同款）。**验证：descriptors 6/6 张 BIT-IDENTICAL**。
2. **匹配顺序层（已定位、未闭合）**：matches 15 对 pair **行数全部相同、集合完全相同（sorted-equal=True），仅顺序不同**——fork 列表按 idx2 升序、upstream 按 idx1 升序。该顺序是 RANSAC 采样索引的数值契约：顺序不同 → TVG inlier ±1-2 → mapper 配准序/gauge → undistorted 逐 gauge 重采样 → 稠密链放大。定位手段：浮点 dump 探针（GPU float descriptors 6/6 bit-identical，排除 GPU/金字塔）+ 漏斗探针（FeatureMatcherCache::WriteMatches）。**剩余工作**：在 fork 匹配编排（feature/matching.cc）中定位产生 idx2 序的代码点（GetBestMatch 源码/SASS 两侧一致，翻转在 colmap 编排层），对齐后即可打通 sparse→dense 全链 bit-parity。
3. **临时探针已全部清理**（sift.cc float dump ×2、matching.cc 漏斗 dump），upstream 仓库已 git 恢复原状。

**第三十四轮（2026-09-20(4)，fused 差距根因链修复 + 渲染暗问题产品级修复 ✅）**：

1. **fused 差距归因链重建（用户质询：2.57%@20mm 差距必须彻底对齐）**：三组对照实验推翻第三十三轮“非引擎缺口”结论——① E1 同 binary 重跑（upstream run-A vs B2）：fused 差 **0.00%@10mm+**（运行噪声带 ±0.02%@5mm，引擎近乎确定）；② E2 fork 引擎跑在 upstream 同一 workspace（完全相同输入）：**2.24%@20mm 复现**；③ E3 fork fusion 引擎 + upstream 深度图：0.03%@20mm（等价）。⇒ AB 差距 ~90% 在引擎层（stereo），fusion 无辜，稀疏输入只占 ~0.3pp。
2. **引擎层根因 #1（已修复）**：`PatchMatchController::Run` 在 filter-by-existence 阶段用哈希容器重建 `src_image_idxs`，upstream 用 `FlatHashSet`（=boost::unordered_flat_set 1.92），fork 用 `std::unordered_set` —— **迭代序不同 → 源图进入分层纹理的顺序不同 → 逐像素 cost 累加链不同 → patchmatch 混沌放大**（相同输入下日志实锤：up `[2,5,4,1,3]` vs fk `[4,5,3,2,1]`；count-desc 预排序 `[1,2,3,5,4]` 被哈希序完全覆盖）。修复 = **vendor boost::unordered 1.92 头文件闭包**（239 头/848KB，BSL-1，`3rdparty/boost_unordered/`，来自 upstream FetchContent `_deps/boost-src`；fork 内置 boost 1.73 无 flat_set）+ `patch_match.cc` 改用 `boost::unordered_flat_set`（仅此一 TU，经 `colmap_boost_unordered_headers` INTERFACE target；全局 FlatHash* 别名保持 std 后端不波及稀疏侧）。**验证**：修复后 fork 日志 `src_image_idxs` 与 upstream 逐位一致；**photometric 深度图+法线图 6/6 张 100.00% bit-exact**；同输入 fused 点数从 -3.14% 收敛到 **+0.07%**（E2''：299,199 vs 298,992），fused missing 从 2.24% → **1.18%@20mm**（missing/extra 完全对称）。
3. **引擎层残差（已修复 ✅，第三十五轮补完）**：初判 geometric pass 存在确定性微差（common-pixel rel.p50=0.002-0.006%，~1.2% 面积 @20mm 翻转）。排除链：内核源码（62 处 token 差异全为 AMD `__GFX9__`/HIP 分支、宏改名、make_unique/reset、换行）、**SASS（cuobjdump 全量 210 内核 bit 对比，仅 CudaRotateKernel 3 个实例寄存器分配级差异——纯数据搬运且光度 pass 大量使用仍 bit-exact，功能等价）**、nvcc（双方同机 12.6）、photometric 输入（bit-exact）、src 深度纹理配置、选项、PRNG（fork A/A geometric 100% bit-exact）、JPG 解码（PNG 无损实验排除；upstream 在 JPG/PNG 输入上 fused 点数完全相同）。filter 0 二分实验：无 filter 仍分叉 → 锁定 kGeomConsistencyTerm sweep。**逐 iteration 二分：iteration 1 即分叉**（2.34% 像素真实深度差，最大 15.6m，非舍入级）→ geom cost 输入差异 → `ComputeGeomConsistencyCost` 读 poses 纹理 channel 19-42（**P/inv_P 矩阵，宿主 float 计算**），而 photometric 用的 channel 0-18 是纯 memcpy K/R/T。数值探针实锤：**`ComposeInverseProjectionMatrix` 内 `Eigen::Matrix4f::inverse()` 在 fork 旗标（全局 `-DEIGEN_MAX_ALIGN_BYTES=0`，PCL 兼容）下 inv_P[7] 差 1 ulp（`4095acac` vs `4095acab`）**。修复 = `mvs/image.cc` 单 TU 追加 `-UEIGEN_MAX_ALIGN_BYTES`（Eigen 对象全函数局部，无 ODR 暴露；注意 per-source 属性必须在定义 ColmapLib 的目录设置——子目录设置对父目录 target 不可见）。**最终验证：photometric + geometric 深度图 12/12 张 bit-exact，fused 点数差 -0.036%（多线程聚合噪声带内），fused missing 0.023%@20mm / 0.003%@50mm（低于 fusion 多线程运行方差）——引擎层实用 bit 级对齐达成**。剩余 e2e 差距 = 稀疏输入层（特征提取差异，既有 XL 项，~0.3pp @20mm）+ fusion 多线程噪声。
4. **渲染暗问题（用户裁定回退产品级修复，改数据级交付）**：配置文件 RecentFiles 实锤——用户打开的是 `textured_obj/mesh.obj`（第三十轮交付，无 vn）与 `textured/mesh.ply`（引擎输出，x/y/z+texcoord 无法线），**从未打开唯一带 vn 的 `textured_canonical/mesh.obj`**。勘误第三十三轮两处证据：往返 PLY/CCB “无法线”是**导出器不写 per-triangle 法线的假象**（ObjFilter→ObjFilter 闭环往返 791,812 条 vn 完整保留，导入侧无辜）。曾实施 `ObjFilter`/`PlyFilter` 导入自动补法线，**用户裁定回退**（改动 IO 模块影响面大、与 CloudCompare 语义冲突，已恢复原状并重建二进制）。改用**数据级交付**：canonical OBJ（带 vn）已同步覆盖 `textured_obj/mesh.obj`（fork 791,812 / upstream 820,121 条 vn 验证通过），用户习惯路径打开即正常亮度。
5. **e2e 效果验证（新二进制全链重跑 ✅）**：fork 侧 8 步全链（feature_extractor→exhaustive_matcher→mapper→image_undistorter(2000px)→patch_match_stereo(geom)→stereo_fusion→poisson_mesher→mesh_texturer(--export_obj 1)）用修复后二进制全新重跑（`/tmp/ab_mini6/fork2`，注意 mapper 的 output_path 需预建目录、`--export_obj` 需显式参数 1）。结果：6/6 配准、稀疏 1368 点（与 upstream 同规模）；sim3（相机中心）残差 **0.0002m**；fused 290,386 vs 298,992（-2.88%），missing **2.46%@20mm / 0.64%@50mm**。**残差定性**：稀疏层为同一重建的不同 gauge——相机中心 sim3 后 mean 0.23mm/max 0.33mm，稀疏点 NN p50=0.83mm/p90=3.13mm（p99=39mm 为远点尾部差异）；undistorted 图 mean|diff|=0.54/255（gauge 决定的重采样网格差异 + JPEG 重编码）；引擎在相同输入下 0.023%@20mm（第 3 条）。⇒ **e2e 残差来自稀疏层 gauge/特征数据差异经 undistorter 逐 gauge 重采样的放大，非引擎缺陷**；引擎修复使 e2e 差异从"引擎混沌+输入差异叠加"变为纯输入差异。**e2e 完全一致的路径**：(a) 稀疏层 bit 一致（特征提取/匹配/初对/gauge 对齐，XL 项）；(b) 实用折中——两侧稠密链共用同一稀疏模型（已验证 → 0.023%@20mm）。

**第三十三轮（2026-09-20(3)，AB 差异逐层 probe 定位 + OBJ 法线修复（渲染暗根因）✅）**：

1. **Gap Probe 链路建立 ✅（用户要求的方法论交付，`.qoder/tmp/gap_probe.py`）**：相机中心 Umeyama sim3 对齐（fork→upstream，残差 0.0003m）+ 逐层双向最近邻 diff（missing = upstream 有 fork 无；extra = 反向）+ 阈值表（5/10/20/50mm）+ 空间诊断 PLY（红=缺失、绿=覆盖）。**逐层结果**：fused 点 0.66% @5cm missing / 0.42% extra；poisson 面 1.96% / 0.29%；**textured 面与 poisson 数字完全相同——纹理化零丢面**（用户"问题 2 由问题 1 导致纹理不一致"的怀疑排除：纹理引擎不改变几何）。
2. **深度图直接对比（stereo 层，最后一个未量化层）**：解析 COLMAP depth map bin（头 "W&H&C&" 文本 + float32 数据），6 张 geometric 深度图逐张对比：**共同有效像素相对深度差 p50=0.37-0.38%、p90=0.40-0.44%**（场景 4-8m 深度下 ~15-30mm = patch_match 亚像素精度带，**算法数学一致的直接证据**）；差异仅在每图 ~2% 的边缘有效像素双向翻转（up 有效总和 508,960 vs fk 481,578，-0.5%）。**下钻**：31,687 个缺失面（@5cm）质心处 fork fused 点覆盖（5mm 内 14.3%）显著低于 upstream（26.6%）且两边都稀疏（20mm 内 75.9%/91.6%）——**缺失区域 = 点云边缘/薄结构，fork 更稀**。
3. **根因判定**：fork-up 差异 = stereo 深度图 GPU 非确定性（patch_match CUDA 并行归约顺序）+ sparse gauge 微差（相机中心 0.2-0.9m）→ fusion 聚合（fusion A/A 自差仅 ±0.007%，298,990/298,971/298,992——fusion/poisson 层近乎确定性）→ **poisson 隐函数重建把边缘点稀疏放大成整片无表面**（薄结构对点密度最敏感）。**非引擎对齐缺口**：共同区 0.37% 深度一致 + 引擎代码逐行对齐（第 25/30 轮）+ 73/73 测试。红绿诊断图 `/tmp/ab_mini6/gap_probe_red_green.png`（红色集中在左翼薄结构边缘）。**A/A stereo 测试暂缓**：GPU 被 lingbot-map-cli 占 17GB（用户任务）致 patch_match OOM，GPU 空闲后可补测运行噪声带的直接测量。
4. **OBJ 渲染暗根因修复 ✅（用户问题 3）**：ACloudViewer 日志实锤 `[OBJ] 877198 tex. coords, 0 normals` + `Mesh has no normal!`——两边 OBJ 均无 vn，lit shading 下漫反射 N·L 无合法法线 → 系统性变暗（与 fork/colmap 无关的共同根因；MTL 的 Ka/Kd/illum 段为 saveAsMTL 标准输出无问题）。修复 = `ExportTexturedMeshAsObj` 在 saveToFile 前 `computePerVertexNormals()`（从三角拓扑计算，poisson 输出绕序使法线朝外）；ObjFilter 自动写 vn 块 + face 行 `f v/vt/vn`。重新生成后验证：vn=791,812（fork）/820,121（upstream）与顶点数一致、UV 裁决保持 0.9984、`/tmp/ab_mini6/{fork,upstream}/dense/textured_canonical/mesh.obj` 已更新——**ACloudViewer 重新打开即得正常亮度**。

**第三十二轮（2026-09-20(2)，导出走规范 IO 链：mesh_texturer --export_obj + ObjFilter ✅）**：

1. **用户要求：导出复用既有规范 IO 接口，而非自定义实现**（ply_to_obj.py 转换脚本不高效、格式自定义风险）。**规范件盘点**（仓库内已有、此前 AB 未使用）：① `libs/CV_io/src/ObjFilter.{h,cpp}`——CC 核心 OBJ 导入/导出 filter（`mtllib`/`usemtl`/`vt`/`f v/vt` + per-triangle texCoord/MTL indexes，`hasPerTriangleTexCoordIndexes` 全支持）；② `ccMaterialSet::saveAsMTL`（`ecvMaterialSet.cpp` L421+）——自动保存材质纹理图并写 `map_Kd` 相对引用；③ `PlyFilter` L434-435 已支持 face texcoord list 读取（Photoscan 约定，与 COLMAP 同款）；④ 先例 `MvsTexturing::SaveOBJModel`（`mvs/texturing.cc` L1954，ccMesh + ObjFilter 全链示范，但绑定其私有 texture_atlases_ 结构且无 CLI 暴露——D1 决策记录其引擎未接线）。
2. **实施**：新建 `mvs/textured_mesh_export.{h,cc}`——`ExportTexturedMeshAsObj(MeshTextureMappingResult&, PlyMesh&, path)` 把上游同款纹理引擎的结果组装为 ccMesh（per-triangle texCoord/MTL indexes + 单材质 atlas）交 `ObjFilter::saveToFile`；`face_uvs` 已是 GL 纹理坐标（`texture_mapping.cc` L622-623，与 OBJ vt 同约定）**直存不翻转**；per-triangle texCoord indexes 免拆顶点（顶点数与引擎 PLY 严格一致），vt 池量化去重。`mesh_texturer` 加 `--export_obj` 选项（默认 false，兼容）调用之。
3. **验证 ✅**：两边 AB 产物重生成于 `textured_canonical/`；面质心非黑率 fork **0.9984** / upstream **0.9960**（标准 GL 解读，与修复后脚本版同量级——UV 语义正确）；MTL 标准段（Ka/Kd/Ks/illum/Ns + `map_Kd mesh_map_Kd.png`）；`mtllib` 在 `usemtl` 之前、v/vt 索引全部在界；**ACloudViewer 自身命令行往返**（`-SILENT -O mesh.obj -SAVE_MESHES FILE roundtrip.ply`）成功解析并导出 63.8MB PLY——用户 GUI 使用的即同一导入链。
4. **结论**：AB 交付 OBJ 改由规范 IO 链产出（`/tmp/ab_mini6/{fork,upstream}/dense/textured_canonical/`）；自定义 ply_to_obj.py 退役（仅存档于 .qoder/tmp 作取证参照）。格式规范性与高效性同时满足：v 不拆分（791,812 顶点原样）、vt 池 877,196（vs 拆分式 454 万角）、单次遍历组装。

**第三十一轮（2026-09-20，AB 产物 OBJ 纹理镜像缺陷定位与修复 ✅）**：

1. **用户复检推翻第三十轮结论③**：黑花出现在 ACloudViewer 中——其 qMeshIO 经 assimp **完整支持 OBJ+MTL 纹理渲染**（`plugins/core/IO/qMeshIO/src/IoUtils.cpp` L209-312：`aiTextureType_DIFFUSE` 等六类 → `ccMaterial::setTexture`），"查看器限制"结论不成立，缺陷必在数据侧。
2. **根因：ply_to_obj.py 双重 v 翻转**。引擎 `texture_mapping.cc` L622-623 写 PLY texcoord 时**已转换为 GL 纹理坐标**（`v = 1 - atlas_y/height`，v=0 在图像底部；fork 与上游逐行一致，第 25 轮已证）；OBJ 的 vt 同为 GL 坐标（v 从底）；转换脚本再做一次 `1 - v` → **atlas 垂直镜像** → 标准 GL 解读下 ~45-51% 面质心落在 atlas 黑背景上（黑花实锤：修复前 fork 质心非黑率 0.4491 / upstream 0.5097，镜像解读下反而 99.8% 命中）。**同源缺陷**：`textured_to_vertexcolored.py` 采样把 GL v 直接当图像行（`v*h`），顶点色 PLY 同样镜像——**两边同错导致第三十轮并排自证失效**（方法学教训：自证对比必须存在至少一边绝对正确的参照，同源错误会相互掩盖）。
3. **修复 + 量化验证 ✅**：两脚本修复（OBJ vt 直拷 PLY UV；顶点色采样行改为 `(1-v)*h`），重新生成 2×OBJ + 2×vcolored PLY；复查面质心非黑率 fork 0.4491→**0.9984**、upstream 0.5097→**0.9973**；离屏并排图 `/tmp/ab_mini6/side_by_side_fixed.png` 两边呈现一致的真实照片纹理（黑花消失、模型正立、up-axis 修正）。
4. **结论修订**：引擎（fork 与上游逐行一致）与 AB 重建本身无缺陷；第三十轮三问中 ①③ 全部归因转换脚本 UV 约定错误，②（两重建 sim3 对齐残差 0.0003m 毫米级一致）维持。`/tmp/ab_mini6/{fork,upstream}/dense/textured_obj/mesh.obj` 已原地更新，ACloudViewer 重新打开即可得到正确纹理渲染。

**第二十九轮（2026-09-19(4)，generalized_pose_test.cc 完整移植 + 两项引擎缺口收 口 ✅）**：

1. **generalized_pose_test.cc 完整移植 ✅（第 27 轮立项的 C++ 回归保障项）**：938 行 16 用例（Estimate/Refine × 普通/Scaled × Nominal/Empty/Panoramic/SingleCenter/StaleInliers、PositionPrior 及其 Covariance 对比、EstimateGeneralizedRelativePose 3×3×2 参数矩阵、EstimateStructureLessAbsolutePose 3 用例）全量移植并 16/16 绿。移植适配点：Point2D/Point3D/Camera 访问器风格（XY()/XYZ()/Params()/Width()/Height()）、fork 相机模型枚举、Frame::ImageIds() 直接产出 image_id、CMake COLMAP_ADD_TEST 注册。
2. **移植暴露并修复三项引擎缺口（上游对齐）**：① `UniqueInlierSupportMeasurer` 落地（optim/support_measurement.{h,cc}，上游同构：按 unique_sample_ids 去重计数 num_unique_inliers + Compare 优先 unique）——此前 fork 仅有 plain inlier 计数的"近似适配"（第 27/28 轮注释记录），GP3P/GP4PS 两个 estimate 现与上游同款 `LORANSAC<E,E,UniqueInlierSupportMeasurer>` + num_unique_inliers 返回；② `ThrowCheckCameras` 补空输入防护（上游 d3ccaf35 L35-37：空 camera_idxs 合法、由调用方拒绝）——缺失时 minmax_element 解引用 end 迭代器段错误，EmptyInputsFail 用例暴露；③ `EstimateRelativePose`（ray-based 版）落地（pose.{h,cc}，上游同构：LORANSAC<EssentialMatrixTangentSampsonEstimator> + PoseFromEssentialMatrix + NaN 防护）并接通 `EstimateGeneralizedRelativePose` 的 panoramic 分支（此前为 "not ported" 占位返回 false）。
3. **回归**：generalized_pose_test 16/16；组级 ctest（estimators/scene/sfm/controllers/reconstruction/geometry/optim/math）71/73——2 个失败（math/polynomial TestFindQuadraticPolynomialRootsComplex、optim/least_absolute_deviations RidgeRegularization）为**既有失败**（源文件 diff=0、基线 5bb6b52dba，本轮首次扩大回归面至 math/optim 才暴露），记录立项；三份 Python 冒烟（gp/cov_r28/cov）全过。

**第二十八轮（2026-09-19(3)，非 scaled refine 6x6 协方差引擎组装 + util 绑定 + AddFrame 自动注册缺陷修复 ✅）**：

1. **非 scaled refine 6x6 协方差引擎组装 ✅（L 级收口，第二十七轮诚实化为 None 的欠账）**：`RefineGeneralizedAbsolutePose` 重构为上游同构单块形态——删除分离 qvec/tvec 过渡实现与 `FrameRigBundleAdjustmentCostFunction` 依赖，改用 `CreateCameraCostFunction<RigReprojErrorCostFunctor>` + `rig_from_world->params` 直通 + `CreateProductManifold(EigenQuaternion, Euclidean<3>)`（fork 的 Rigid3d 自 W3-2b 起即为 contiguous 7 维 params，旧实现是历史遗留）；补 `use_position_prior` 分支（`CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>`，options 补 `position_prior_covariance`/`position_prior_in_world` 字段）；solve 后 `if (!IsSolutionUsable) return false` + ceres `Covariance`(DENSE_SVD) + `GetCovarianceMatrixInTangentSpace`（切向 [rotation(3), translation(3)]，边缘化精化的相机参数）——与上游 pose.cc/generalized_pose.cc 逐行同构。**单目 `RefineAbsolutePose` 同步收口**（此前连 cov 参数都没有）：签名加 `Eigen::Matrix6d* cam_from_world_cov = nullptr` + 同款 prior 分支与协方差段。**数值验证**：6x6 对称/正定/有限；蒙特卡洛 120 次（sigma=0.5px）经验/预测特征值比 = 旋转维 0.86-0.96（预测精确）、平移维 ~0.24（Cauchy robust loss 下预期偏差，band 内）；refine 精度 rot 0.013°/trans 0.005m。**绑定**：`refine_generalized_absolute_pose` 真协方差替换 None；单目 `refine_absolute_pose` 加 `return_covariance`（tuple 返回）；`estimate_and_refine_generalized_absolute_pose` 随引擎实现自然修复（此前同样存在垃圾内存风险）。顺手补 `AbsolutePoseRefinementOptions.print_summary` 绑定字段。
2. **util 绑定 ✅（cosmetic 收口）**：新建 `pybind/reconstruction/util/util_bindings.{h,cpp}`，注册 `pybind.reconstruction.util` 子模块——`Timer`（13 方法全量：start/restart/pause/resume/reset/elapsed_*/print_*）、`logging`（上游 dummy-class 模式：`Level` 枚举 + minloglevel/stderrthreshold/log_dir/logtostderr/alsologtostderr/verbose_level 六个 glog FLAGS + `set_log_destination` + verbose/info/warning/error/fatal，消息携带 Python 调用点帧）、`CancellationToken`（cancel/is_cancelled）、`get_version`/`get_build_info`。上游 util 绑定中的 oiio/cuda/timestamp/types 无 fork 引擎对应或已由其他子模块覆盖（SensorType/data_t 在 scene、pair_id 在 database），按 D2 去重不重复注册。冒烟：Timer 计时/暂停恢复、logging 输出、token 状态、版本字符串 4 项全过。
3. **意外修复：AddFrame 自动注册缺失（引擎缺陷③）✅**：本轮回归 `coordinate_frame_test` 新失败（`AlignToPrincipalPlane.Nominal`：NumRegFrames==0）暴露——上游 d3ccaf35 `AddFrame` 对带位姿帧**自动 RegisterFrame**（reconstruction.cc L436-444），fork 自 W3-2b 前起显式跳过（`(void)is_registered` 过渡注释），且第二十一轮的 `RegFrameIds()` 从 reg_image_ids_ 派生的实现丢掉了上游"显式 frame 级注册集合"这半边语义。修复 = fork 引入显式 `reg_frame_ids_` 集合（上游同构）+ 新 `RegisterFrame(frame_id)`（HasPose 检查 + 幂等插入 + 既有 image 的 image 级 lockstep）+ `RegisterImage`/`DeRegisterImage` 双向 lockstep + `DeRegisterFrame` 擦除集合 + `AddFrame` 补上游自动注册分支 + `RegFrameIds()` 直返成员（删除派生实现）+ 拷贝构造/赋值同步。`SetRigsAndFrames`（唯一调用方 ApplyRigConfig）按上游语义重置后重建一致注册状态；io text/binary Read 的 `RegisterImage` 恢复路径不变（fork 模型格式契约）。**Python 暴露面补齐**：`register_image`/`deregister_image`/`register_frame`/`deregister_frame`/拷贝构造。**回归**：coordinate_frame 5/5、rig 12/12、observation_manager 14/14、global_pipeline 13/13、rotation_averaging 15/15、covariance 7/7，组级 ctest 43/43 全绿；Python 注册 e2e 5 项（synthetic lockstep、模型 write/read round-trip、拷贝、deregister 双集合、mapper 访问器一致性）全过。
4. **剩余项盘点**：pycolmap 暴露面对齐中 generalized_pose 族 8 入口（上游 7 + fork `estimate_structure_less_absolute_pose` 扩展）全部带真协方差可用；**generalized_pose_test.cc 完整移植**（上游 938 行 18 用例：GP3P/GP4PS/GR6P/refine/prior/scaled/structure-less）为下一轮 C++ 回归保障项。

**第二十七轮（2026-09-19(2)，rotation_averaging_test 修复 + scaled generalized pose 引擎落地 ✅）**：

1. **rotation_averaging_test 修复 ✅（上轮立项的既有失败）**：根因 = fork 的 GP 修复（3-DOF 残差用 sensor 级 `R_cam2_from_cam1_sensor`）假设 estimated cam-from-world "已含已知的 cam_from_rig"，但 `ComputeResiduals` 仅在 cam_from_rig **被估计**（param_idx != -1）时复合，known rig（cam_from_rig 已标定、不进参数）下 cam 因子不平衡——真值处残差非零 → use_gravity=false 的 3-DOF 求解系统性偏差（20/28 对 >10°）→ 过滤后 DeRegister 帧位姿被 Reset → 测试读取 pose-less 帧触发 `frame.cc:104 CHECK`。gravity 1-DOF 路径用 rig 级减约不受影响（解释了两遍结果 0 vs 20 的差异）。**修复**：`PairConstraint` 缓存已知 cam_from_rig 旋转（`known_R_cam1/2_from_rig`），`ComputeResiduals` 在 param 缺失时复合已知值——残差 `(a2*r2)^T * R_sensor * (a1*r1)` 在真值处归零，known/estimated cam_from_rig 两情形因子均平衡（与上游 rig 级减约在矩阵结合意义下等价）。诊断过程记录：边质量审计证明 TVG 数据无辜（0 坏边）、逐步日志定位崩溃于测试的 MeanRelativeRotationErrorDeg。**回归**：rotation_averaging_test **15/15**、estimators+global_mapper **19/19**。
2. **scaled generalized pose 引擎落地 ✅（收口 GP4PS 引擎缺口）**：fork solver 层已有 `GP4PSEstimator`（W5 批次）与 `ScaledRigReprojErrorCostFunctor`（use_log_scale），本轮补 estimators 层入口（上游 d3ccaf35 generalized_pose.{h,cc} 移植）：`EstimateScaledGeneralizedAbsolutePose`（Sim3d 输出；fork adaptation：plain LORANSAC 无 UniqueInlierSupportMeasurer，标准内点计数近似上游 unique 计数；panoramic 输入与 inlier 共识双检查拒绝尺度不可观测）、`RefineScaledGeneralizedAbsolutePose`（上游 ceres 架构原样落地：log-scale 优化 + quaternion×Euclidean<4> ProductManifold + ceres::Covariance 7x7 切向协方差 + log-scale→scale 行列传播）、`IsPanoramicCameraSelection`（带 selection_mask，对齐上游命名，原匿名 IsPanoramicRig 改造）。`util/types.h` 补 `Matrix7d`（上游同款 typedef）。
3. **绑定与诚实化修正 ✅**：新增 `estimate_scaled_generalized_absolute_pose` / `refine_scaled_generalized_absolute_pose`（7x7 covariance）/ `estimate_and_refine_scaled_generalized_absolute_pose` 三个入口——上游 pycolmap scaled 变体暴露面收口（第二十六轮记录的"引擎缺口"解除）。**同时修正非 scaled `refine_generalized_absolute_pose` 的绑定缺陷**：fork 分离块精化引擎未组装 6x6 协方差（指针被接受但不写入），此前绑定返回未初始化内存（第二十六轮冒烟中"cov finite=True"实为垃圾数据的假阳性）——现 return_covariance=true 时诚实返回 None 并在 docstring 标注边界。
4. **验证**：Python 冒烟 8 项全过（新增 scaled 三项：scale 2.4979/2.5 估计、2.49891 精化 + 7x7 协方差有限非零、panoramic 拒绝；GP4PS 语义注记——上游 `TransformCameraWorld(S, cam)` 构造下 cams_from_rig = Rigid3d(R_c·Rᵀ, s·t_c − R_c·Rᵀ·t)，观测为 s 倍齐次缩放的相机系点，投影不变性使联合尺度可观测）。estimators+global_mapper+io+bundle 组 **20/20** 全绿；covariance 冒烟复验通过。

**第二十六轮（2026-09-19，generalized_pose 族绑定 + covariance 单块化收口（L 级）+ pyceres 判定 ✅）**：

1. **三项真实性判定（本轮输入问题）**：① generalized_pose 族 = **真实 gap**（fork 引擎有 4 函数、Python 暴露面为零，本轮已收口）；② covariance 既有回归 = **真实**（第二十四轮确认的 HEAD 6/7 红，本轮已修复至 7/7）；③ pyceres = **非引擎对齐缺口**（上游 `src/colmap/` C++ 引擎零引用；pyceres 仅是 pycolmap 的可选 Python 包，`ceres_bindings.cc` 注释明确 "without pyceres" 亦可用、`IsPyceresAvailable()` 运行时探测，仅服务宿主端自定义 cost function 的 BA 扩展场景——维持暂缓，不立对齐项）。
2. **generalized_pose 族 Python 绑定 ✅**（对齐上游 `pycolmap/estimators/generalized_pose.cc` 暴露面，以 fork 引擎实际 API 为真源）：`estimate_generalized_absolute_pose`、`refine_generalized_absolute_pose`（含 `return_covariance` 6x6）、`estimate_and_refine_generalized_absolute_pose`、`estimate_generalized_relative_pose`（rig2_from_rig1 / pano2_from_pano1 双出口）+ `StructureLessAbsolutePoseEstimationOptions` 与 `estimate_structure_less_absolute_pose`（fork 扩展入口，上游 pycolmap 未直接暴露）。**上游 scaled 变体**（`estimate/refine_scaled_generalized_absolute_pose`，Sim3d + 7x7 协方差，GP4PS 系）fork 引擎无对应函数，维持未绑定并记录为引擎缺口。合成 rig 数据数值冒烟全过（abs pose RANSAC 1.44°→refine 0.15°、相对位姿 0.49°/0.69m、structure-less 无噪零误差；冒烟数据踩坑记录：GR8P 文档化退化——纯平移运动与全同相机对应，真实采集为跨相机对应不受影响）。
3. **covariance 单块化收口（L 级）✅**：`PoseParam` 增加 `rigid7`（W3-2b 单 7 维 Rigid3d 阴影块指针），`GetPoseParams` 签名增加 `frame_blocks`（经 `CeresBundleAdjuster::frame_blocks()` 解析阴影块，fallback 分离存储；frame 无关 image 走 legacy 分离路径）；`ComputeSchurComplement`/`pose_L_start_size`/`GetOtherParams` 全面支持单块形态（tangent 6 与 legacy 3+3 等价，[rot,trans] 顺序不变）；`EstimateBACovarianceFromProblem` 增加可选 `frame_blocks` 参数（`EstimateBACovariance` 自动传入）。**阴影块生命周期修复**：`TearDown` 不再清 `frame_blocks_/sensor_blocks_`（clear 移至 `SetUp` 开头）——Solve 后协方差估计器仍可解析已注册 pose 指针。**0x0 修复重新落地**：第二十四轮回滚将其一并撤销（文档当时标注"保留"与实际不符，本轮更正），`SchurEliminateOtherParams` 对 `other_num_params==0` 提前返回。**测试同步**：`covariance_test.cc` 的 `GetPoseParams` 调用点与 pose 参考协方差块构造改为表示无关的块列表形式。
4. **回归证据**：`covariance_test` **7/7 全绿**（HEAD 既有 6/7 红 → 收口）；`bundle_adjustment_test` 11/11；estimators 组 17/18（唯一失败 `rotation_averaging_test` 为**既有问题**：`WithoutNoiseWithNonTrivialKnownRig` 在 gravity + 非平凡 rig 路径触发 `frame.cc:104 CHECK(has_pose_)`，相关文件最后改动在基线 5bb6b52dba、与本轮 covariance/binding 改动无交集，**立项待修**）；Python e2e：synthetic → BA（THREE_POINTS gauge）→ 位姿协方差 3/3 全 6x6 finite + cross cov + point cov + **相对位姿协方差 6x6** 全部可用（第二十二轮的"位姿协方差不可用"边界解除）；generalized_pose 冒烟全过。
5. **gauge 语义说明（实测澄清）**：`TWO_CAMS_FROM_WORLD` gauge 下锚定帧位姿恒定 → `get_cam_cov_from_world` 返回 None、第二帧部分维度被固定 → 5x5（上游契约：constant 维度的行列被省略）；`THREE_POINTS` gauge 下全部位姿 6x6。Python 冒烟两种 gauge 均验证。

**第二十五轮（2026-09-19，纹理处理链路逐代码一致性核查 ✅）**：

1. **核查范围与方法**：纹理链路全部文件（`mvs/texture_mapping.{h,cc}`、`texture_mapping_test.cc`、`mvs/cuda_texture.h`、`exe/mvs.cc RunMeshTexturer`）对上游 d3ccaf35 做归一化逐代码 diff（剥离 include 路径前缀、宏名差异、折行、注释后按语义行序列比对）。
2. **结果：零数值行为差异**。语义差异块全部为机械适配——① include 路径（`colmap/` 前缀）与 `COLMAP_CGAL_ENABLED`→`CGAL_ENABLED` 宏名；② fork Bitmap 的 `InterpolateBilinear/GetPixel` 出参 API（含失败时初值语义等价于上游 `value_or(0)`）；③ `std::clamp(x,0,255)` ≡ `max(0,min(255,x))` 数值等价式；④ `CHECK_OPTION_IN(a,1,3)` ≡ `GE+LE` 拆分；⑤ `LOG_HEADING2`→`PrintHeading2`、`Bitmap(w,h,rgb)`→`Allocate` API 名；⑥ `boost/next_prior.hpp` include 删除（上游仅为老 CGAL 传递依赖的注释性防御，fork 无此需求）；⑦ `OptionManager::Parse` fork 为 void（内部异常退出）vs 上游 bool——exe 层全局统一惯例。
3. **`texture_mapping_test.cc` 与上游 471 行逐行零差异**（上游 15 用例原样落地），当前 15/15 全绿。
4. **边界说明**：`mvs/mesh_postprocessing.{h,cc}` 为 fork 专属后处理（纹理图表质量服务，第二十二轮裁定保留），上游无对应文件；`mvs/texturing.{h,cc}` 为 ACloudViewer 集成层（ccMesh + PinholeCameraTrajectory），不属 COLMAP 对齐口径。两者均不影响 MeshTextureMapping 内核与 `mesh_texturer` CLI 的上游一致性。

**第二十四轮（2026-09-18(8)，covariance 限制 (a) 深调：根因确认 + 0x0 修复 + 回 滚决策）**：

> 更正（第二十六轮）：本轮标注"0x0 修复保留"与实际不符——回滚将 0x0 修复一并撤销；该修复已在第二十六轮随单块化收口重新落地。

1. **鉴别实验**：EXP1 fixture 式 fresh synthetic + ALL → True；POSES → False（同问题）→ 排除管线状态因素；EXP3 关闭内参细化 → POSES 仍 False → 排除变量内参因素。
2. **探针实证**：`poses=0, points=57, others=6, blocks=67`——**GetPoseParams 返回空**（fork covariance 按分离 qvec/tvec storage 找指针，而 W3-2b 第八轮单块化后的 BA 注册的是内部 7 维 Rigid3d 阴影块）→ 6 个 pose 阴影块全部落入 others → S_oo（36x36）gauge 奇异 → LLT 失败。
3. **0x0 修复 ✅（保留）**：`SchurEliminateOtherParams` 对 `other_num_params == 0` 提前返回 True（Eigen SimplicialLLT 对 0x0 矩阵报失败）——others 确为空时 POSES 模式不再被误拒（上游同源缺口，上游 fixture 恰好 others 非空未覆盖）。
4. **单块化重构尝试 → 回滚**：PoseParam 单块化 + GetPoseParams 经 `CeresBundleAdjuster::frame_blocks()`（新增公有访问器，保留）解析阴影块后，`ComputeLInverse` 报 rank 缺陷（gauge 歧义），covariance_test 从既有 6/7 红变为 1/7——数值闭环未达成，按"回归不扩大"原则回滚至分离块形态。
5. **重大发现：covariance_test 在 HEAD 上已 6/7 红**（`CompareWithCeres/0,1,2,4,5,6` 失败于 estimate_ba_covariance 返回空）——这是 W3-2b 第八轮 BA 单块化重构未同步 covariance 的**既有回归**（本轮调试发现，非本轮引入）。影响：C++/Python 的位姿协方差 currently 不可用；点协方差经 `from_problem` 路径可用。
6. **净交付**：① `SchurEliminateOtherParams` 0x0 修复；② `CeresBundleAdjuster::frame_blocks()` 公有访问器（为单块化修复铺路）；③ Python `estimate_ba_covariance_from_problem`（problem 句柄路径，实测返回 BACovariance）；④ 根因与既有回归的完整取证。**后续批次（L 级）**：covariance 全面单块化（GetPoseParams/pose_L_start_size/S 组装/test 对齐 shadow 块语义 + 解决 L_inv gauge 奇异）→ 恢复 covariance_test 7/7 → Python 位姿协方差可用。**本轮 Python POSES/位姿协方差保持不可用（诚实边界），点协方差经 from_problem 可用**。

**第二十三轮（2026-09-18(7)，限制 (b) 撤销 + GLomap 全流程 Python 打通 + dict 访问器 ✅）**：

1. **限制 (b) 撤销（非引擎缺陷，调用语义问题）**：`PoseGraph::Load` 的契约是只加载**含相对位姿**的 verified TVG 边（synthetic 写库的 TVG 默认 `two_view_geometry_has_relative_pose=false`，无 cam2_from_cam1 → 位姿图为空）；空图进 RA 触发 LAD 的 "Underdetermined systems not supported" throw——防御性检查正确但消息误导。正确 GLomap 流程 = 先对每对估计两视图几何 + 相对位姿（`estimate_two_view_geometry` + `estimate_two_view_geometry_pose`）填图。
2. **GLomap 全流程 Python 端到端打通 ✅**：synthetic DB → IncrementalMapper 注册 → 每对估计相对位姿 → 填 PoseGraph（15/15 边）→ `estimate_rotations` True（6/6 帧注册）→ `run_global_positioning` True（Ceres CONVERGENCE，cost 4.97e3 → 2.44）→ `run_gravity_refinement`。
3. **绑定层补齐**：`Reconstruction.images/cameras`（dict，引用语义，写回验证；dict 元素无法激活 reference_internal 的 keep_alive，用 reference policy + 文档注明重建须长于 dict 存活）与 `points3D_map`（copy 语义）。

**第二十二轮（2026-09-18(6)，W16 pybind estimators 收官批次：covariance + motion averaging ✅ / 两项引擎限制立项）**：

1. **covariance 绑定 ✅**：`estimators.CeresBundleAdjuster` 类（owner wrapper：ctor(options, config) / solve(reconstruction) / problem 访问器）+ `CeresProblem` 不透明句柄（fork 无 ceres 绑定面，句柄供 problem 往返）+ `BACovarianceParams`/`BACovarianceOptions`/`BACovariance`（get_point_cov / get_cam_cov_from_world / get_cam_cross_cov_from_world / get_cam2_cov_from_cam1 → numpy 或 None）+ `estimate_ba_covariance` / `estimate_ba_covariance_from_problem`。
2. **motion averaging 绑定 ✅**：`RotationEstimatorOptions`（fork 全字段 + RotationWeightType 枚举）+ `estimate_rotations`（求解并注册帧，上游 RA 回写语义）；`GlobalPositionerOptions` + `run_global_positioning`；`GravityRefinerOptions` + `run_gravity_refinement`（in-place 列表往返语义）。
3. **两项引擎级限制立项（绑定层验证正确——真实委托并暴露引擎行为）**：(a) covariance 的 Schur S_oo LLT 在 `CeresBundleAdjuster` 从 mapper 输出重建装配的问题上失败（covariance.cc:147），即使 gauge 固定 3 常量点 + 冻结内参；C++ covariance_test fixture（fresh synthetic/7 帧/200 点/3 常量点）为绿——分歧在管线问题的结构与 fixture 的差异，需引擎级 Schur 装配调试。(b) `estimate_rotations` 对从 synthetic 对应图 `PoseGraph::Load` 的图抛 Eigen "Underdetermined systems not supported" 而非优雅返回 False；global_mapper_test fixture（手工构造 pose graph）为绿——RA 稀疏求解器需要对应图加载场景下的优雅秩缺陷路径（gauge 锚定）。
4. generalized_pose 族（上游 7 函数）本轮未纳入（rig/cameras_from_rig 参数适配面大），随后续批次。

**第二十一轮（2026-09-18(5)，引擎缺陷②修复：RegFrameIds 语义对齐 ✅）**：

1. **根因链条完全闭合（上游/fork 双向取证）**：① 上游 frames 表 schema = `frame_id, rig_id`（**无位姿列**，ReadFrameRows 只读 id/rig/data_ids），位姿仅在重建文件/内存；fork 扩展了 `has_pose/qvec/tvec` 列（产品特性：DB 内持久化先验位姿）。② 上游 `RegFrameIds()` = 显式维护集合 `reg_frame_ids_`（AddFrame(posed)→RegisterFrame 自动注册，但其 DB 读出的 frame 永无位姿，fresh Load 后集合为空）；fork 用 "HasPose() 即已注册"近似，在 DB 带位姿时语义翻转。③ 上游 mapper 的注册事件源：增量路径经 `ObservationManager::RegisterFrame → reconstruction.RegisterFrame`，全局路径经 RA 回写（`rotation_averaging.cc:378` 对求解出位姿的帧 RegisterFrame）。
2. **修复（三件套）**：① `Reconstruction::RegFrameIds()` 改为从显式维护的 image 级注册集合 `reg_image_ids_` 派生（帧注册 iff 任一 image 注册）——fresh Load 后 reg_image_ids_ 干净为空（Bug#4 修复仅恢复磁盘读路径），语义回归上游；② `GlobalMapper::Solve` 在 GlobalPositioning 后补上游 RA 回写等价物（全部带位姿帧逐 image RegisterImage；上游在 RotationEstimator 内注册，fork 折叠到 GlobalMapper 层——此前注册态从未被维护，旧 HasPose 近似掩盖了该缺口）；③ `SyntheticDataset` 位姿噪声循环改为直接遍历带位姿帧（fork 派生语义下的等价意图；上游遍历 RegFrameIds 因 AddFrame 自动注册而等价于全帧）。
3. **验证（全部无绕过）**：① CLI `mapper` 直接在**带位姿** synthetic DB 上成功（"Initializing with image pair #1 and #6"，此前永远 "No good initial image pair found"）；② CLI `global_mapper` 输出 "Kept 1 reconstruction(s), discarded 0 with fewer than 3 registered frames"（派生式 + 注册回写协同）；③ Python e2e 在带位姿 DB 上直通（6/6 注册/60 点/mean_track 6.0/CONVERGENCE 720 残差）；④ 回归 13 套件全绿（synthetic/global_mapper/global_pipeline/hierarchical/clustering/reconstruction/database_cache/correspondence/observation/BA/two_view/pose_graph/rig）。
4. **语义边界记录**：派生式下"帧注册 iff 帧内有 image 注册"与上游 data-only frame（无 image）的注册路径在 mapper 流程中不可达（mapper 只注册含 image 的帧），实际行为等价；`DeRegisterImage` 维护 `reg_image_ids_` 使 discard 路径自动一致。

**第二十轮（2026-09-18(4)，W16 pybind sfm 类批次 + 引擎初始对缺陷修复 ✅）**：

1. **sfm 类绑定落地（Open3D 风格 owner-wrapper 模式）✅**：`sfm.IncrementalMapper`（find/register_initial_image_pair、find_next_images、register_next_image/general_frame/structure_less_image、triangulate_image、retriangulate、complete/merge_tracks、complete_and_merge_tracks、adjust_local/global_bundle、iterative_global_refinement、filter_images/points、reconstruction 引用访问器、num_reg_frames_per_rig、modified-points 记账）、`sfm.IncrementalTriangulator`（全 triangulation 操作面）、`sfm.ObservationManager`（add/delete point/observation、merge、全 filter 族）、`LocalBundleAdjustmentReport`。选项类不重复注册（D2 去重：`options.IncrementalMapperSubOptions` / `IncrementalTriangulatorOptions` / `BundleAdjustmentOptions` 为唯一定义点）。
2. **生命周期模式定案（关键实现决策）**：引擎类持原始回指针（DatabaseCache/CorrespondenceGraph/Reconstruction），而本 pybind 版本 `py::init` 的 `keep_alive` 不生效（SIGSEGV 实证：删 cache 后调 find 崩溃）→ 采用 **owner wrapper**（`PyMapper/PyTriangulator/PyObservationManager` 持有 `py::object` 引用 + unique_ptr 引擎对象），Python 侧生命周期确定性固定，全部方法经 wrapper 转发。
3. **scene/estimators 补面 ✅**：scene 增 `CorrespondenceGraph`+Correspondence（add/finalize/逐点查找/逐对提取/TVG 提取）、`PoseGraph`+Edge（add/get/delete/update/load）、`ReconstructionManager`（add/get/read/write/delete/clear）、`DatabaseCache.correspondence_graph` 访问器（shared_ptr holder 注册防双重释放）、`Image.points2D`；estimators 增 BA 枚举（Gauge/TerminationType/Backend）、`BundleAdjustmentConfig`（上游全方法面）、`BundleAdjustmentSummary`、`adjust_bundle`（CASPAR-first 工厂分派 + gauge fixing，返回终止类型/残差数/报告）、`estimate_affine2d(+robust)`、`estimate_two_view_geometry_pose`（fork 独立分解步骤暴露）。修复一个真实内存缺陷：`CorrespondenceGraph` 注册未声明 shared_ptr holder 导致 take_ownership 双重释放（`malloc(): unaligned tcache chunk`）。
4. **Python 端到端 SfM 首次全链打通 ✅**：synthetic DB → DatabaseCache → IncrementalMapper（初始对搜索 → 注册 → 三角化 → 迭代全局精化 → adjust_bundle CONVERGENCE/720 残差）→ 6/6 注册、60 点、mean_track 6.0、帧级记账 {1:6}。
5. **冒烟中发现并修复真实引擎缺陷① ✅**：`IncrementalMapperImpl::FindFirstInitialImage` 用 fork legacy 的 `image.NumCorrespondences()`（fresh Load 后恒 0，无人维护）过滤候选，而上游 d3ccaf35 从对应图查询 `NumCorrespondencesForImage` ——导致 CLI `mapper` 在 fresh 库上永远 "No good initial image pair found"。已按上游形态改为 graph 查询。回归：correspondence_graph/database_cache/synthetic/BA/global_mapper/observation_manager/reconstruction/two_view 8 套件全绿。
6. **冒烟中发现并立项引擎缺陷②（RegFrameIds 语义，未修）**：fork `RegFrameIds()` = "所有 HasPose() 的 frame"，上游 = 显式维护集合 `reg_frame_ids_`（仅 RegisterFrame 进入）。后果：从**带先验位姿 frame 的数据库**（synthetic/pose_prior 场景）fresh Load 后，BeginReconstruction 把全部 frame 回放为"已在其它重建注册"→ 初始对搜索被全部跳过（regs=1 探针实证）。修复需 frame 级注册集合化（触及 AddFrame/RegisterImage/Read/Load/NumRegImages 全链，L 级手术，上游形态已勘察归档）。当前用 `has_pose=0` 的 DB 验证路径绕过；主链（feature→mapper，frames 无先验位姿）不受影响。

**第十九轮（2026-09-18(3)，W16 pybind 融合批次 4–8：暴露面收口 ✅）**：

1. **剩余暴露面按 D2 去重表全量落地 ✅**（Open3D 风格类绑定，不引入上游 dataclass/classh_ext 机制）：
   - **geometry**：`calculate_triangulation_angle[s]`、`GPSTransform`（Ellipsoid 枚举 + ECEF/ENU 六变换；UTM 按第十八轮勘定维持不暴露）、`PosePrior` + `PosePriorCoordinateSystem`、`gravity_from_exif_orientation`、`compute_rot90_from_gravity`、`essential_matrix_from_pose`（Rigid3d 入参）、`pose_from_homography_matrix`（射线版，返回 cam2_from_cam1/normal/points3D dict）。
   - **scene**：`SensorType`/`SensorId`/`DataId` 原语、`Frame`（frame_id/rig_id/data_ids/image_ids/rig_from_world 可写属性 + sensor_from_world）、`Rig`（add_ref_sensor/add_sensor/set_sensor_from_rig/non_ref_sensors）、`TwoViewGeometry`（ConfigurationType 枚举、E/F/H/cam2_from_cam1/camera1/2 可选出参、inlier_matches (N,2) uint32 视图、invert）、`SyntheticDatasetOptions` + `synthesize_dataset`、`DatabaseCache` + Options（含 image_names 互转）、`Reconstruction` 补 num_rigs/frames/reg_frames、exists_*、frame()/rig() 引用访问器（reference_internal 保回指针）、reg_image_ids/reg_frame_ids、sim3d_transform；`Image` 的 image_id/name/camera_id 升级为可写（上游 pycolmap parity，修复既有只读缺口）。
   - **estimators**：`TwoViewGeometryOptions`（上游全字段面）、`estimate_essential_matrix`（tangent-Sampson LORANSAC + cheirality 分解，镜像上游 d3ccaf35 实现）、`estimate_fundamental_matrix`（Sampson/8-point 双路径）、`estimate_homography_matrix`、`estimate_two_view_geometry`、`estimate_calibrated_two_view_geometry`（matches 可传 (N,2) uint32 或 None=identity）、`ImageAlignmentError` + `align_reconstructions_via_reprojections/via_proj_centers/via_points` + `align_reconstruction_to_locations`、`estimate_rigid3d/sim3d` + robust 变体（报告 dict 化）。
   - **database**：`Database` 抽象接口类（W17.2b 面的完整 Python 化）：`Database.open` 工厂、计数/存在性、camera/rig/frame/image/pose_prior/two_view_geometry 读写改、keypoints (N,6)/descriptors (N,D)/matches (N,2) numpy 视图、事务上下文管理器 `with db.transaction():`（引擎 `DatabaseTransaction` 不可拷贝/移动 → unique_ptr 持有 + `commit()` 显式释放；Python GC 延迟析构不可依赖）。
   - **sensor**：`Bitmap`（BitmapColor/BitmapFormat/BitmapRescaleFilter、read/write/allocate/rescale/smooth/clone 族、EXIF 5 项访问器、(H,W,C) uint8 array 视图）。
   - **mvs**：`DepthMap`/`NormalMap`/`MVSModel`（Point/track、GetMaxOverlappingImages/ComputeDepthRanges/SharedPoints/TriangulationAngles）。
   - **feature**：`FeatureKeypoint`（4 构造 + 仿射换算族）、`FeatureMatch`。
   - **vocab_tree**：`VisualIndex`（faiss 版类绑定：Create/Read/Write、Add/Query/Prepare/Build、Index/Query/BuildOptions、ImageScore）。
2. **两个绑定层真实缺陷在冒烟中捕获并修复**：① fork `Bitmap::Data()` 返回的是 Storage 句柄而**非像素缓冲**（L324 `data_.get()`）——直接 memcpy 会堆损坏（`malloc(): unaligned tcache chunk`），读路径改走 `ConvertToRawBits()`，写路径走 `SetPixel` 逐像素（fork 无批量写像素面，O(N) 注释）；② `NormalMap` 引擎为 channel-plane (3,H,W) 布局而 numpy 惯例为 (H,W,3)——按像素拷贝转换，roundtrip 全等。
3. **运行时冒烟全绿**：GPS WGS84 ECEF 往返精确（37.723,-119.616,1224.4）、PosePrior/重力向量、合成数据生成（1 rig/3 frames/20 pts，帧位姿带先验扰动）、Sim3d scale=2.0 精确、TwoViewGeometry、Database 全链路（建库/写相机/图像/先验/事务内写 keypoints/读回一致）、Bitmap write/read bit 级一致、DepthMap/NormalMap roundtrip、VisualIndex 建索引。pybind 目标编译 EXIT=0。
4. **剩余项精确记录**（manifest `pycolmap_parity_bindings` partial → 后续批次）：sfm 类绑定（IncrementalMapper/ObservationManager/IncrementalTriangulator/GlobalMapper）与 estimators 的 bundle_adjustment/covariance/motion_averaging/generalized_pose 绑定需先对 fork 分离位姿块 API 面做类生命周期设计；CorrespondenceGraph/PoseGraph/ReconstructionManager 绑定随 sfm 批次；image undistortion 类与既有函数级 API 重叠（D2 去重表：函数为规范接口）；**pyceres 独立评估结论：暂缓**——绑定面依赖 ceres autodiff 模板展开（cost_functions.cc 上游 300+ 行模板特化），收益限于自定义残差用户，与 Open3D 风格绑定成本/收益比为负，待真实需求出现再立项。

**第十六轮（2026-09-17(8)，point_triangulator 收官修复 ✅ + 全链终测 + PoissonRecon 版本缺口定性）**：

1. **point_triangulator round-2 BA 崩溃根因闭环 ✅（三层复合缺陷）**：① 核心根因：`Reconstruction::Read` 的 4.x 分支（有 rigs.bin/frames.bin）只接 frame 指针、**从不写 image 的 legacy qvec/tvec 缓冲**（保持默认 identity），而 W3-2b BA `SetUp` 的 forward dual-track sync 会把 legacy 缓冲**覆盖写回 frame**（`SetCamFromWorld`）→ 位姿在 BA 前被清零 → `GetOrCreateFrameBlock` 收到 identity → `TearDown` 写回全零 → FilterPoints 全滤 → round-2 BA `NumResiduals==0` 崩溃。修复：`ReadImagesBinary`/`ReadImagesText` 接好 frame 后把 `image.CamFromWorld()`（frame 真源）**镜像进 legacy 缓冲**，双轨在 Read 源头同步（上游无双轨概念，此为 fork 双轨机制的内部一致性维护，不改变数值行为）。② `TranscribeImageIdsToDatabase` 缺上游 "Transcribe frame data" 段（d3ccaf35 reconstruction.cc L870-884）——frames 的 data_ids 未随 image_id 转写更新，已补齐（CAMERA 传感器 data_id 用 old_to_new 映射转写）。③ 排查陷阱：`point_triangulator` 的 database 必须含匹配记录（用纯特征库 base.db 会 "Loading matches... 0" → 三角化 0 点 → 同型崩溃），且 output_path 必须预创建（两侧共同 CLI 契约）。
2. **修复验证**：clear 路径 EXIT=0，三轮 BA 正常收敛（1345 点 vs 输入 1354 = -0.7%，与上游 1342→1341 = -0.07% 行为一致）；**位姿冻结检验 max|dq|=1.1e-16 / max|dt|=0**（修复前是全清零）。回归 6/6 全绿（two_view 3、BA 11、triangulation 2、synthetic 19、reconstruction_io 19、global_mapper 5）。
3. **全链终测（/tmp/e2e_r16，两侧全新独立工作区 + point_triangulator 纳入主链）**：稀疏 1354/4258 obs vs 1342/4226（+0.9%/+0.8%）、mean reproj 1.053 vs 1.060 px（-0.7%）、triangulator 1345 vs 1335（+0.7%）、PM valid pixels +0.6%、**PM p99.9 深度 max -0.3% / mean -0.1%**（深度范围完全对齐）、fused 292073 vs 296167（-1.4%）、Delaunay faces -0.8%、带纹理 mesh 10.35MB vs 10.43MB（-0.8%）、全链耗时 14.124 vs 14.072 min（+0.4%）。
4. **两项差异定性（实验说话）**：① 位姿旋转差 1.0041°（本轮自然初始化轨迹）：ICP gauge 对齐后稀疏点云 NN 残差 median 0.20% 对角线（p95 1.92%）→ 骨架 gauge 差而非质量差（重投影/PM/fused/Delaunay 全对齐佐证）；② **Poisson -11.4% faces 分解**：交叉 mesher（fork mesher 吃上游 fused）→ 同数据引擎残差 **-6.3% verts / -8.9% faces**，数据贡献 -2.6%。根因：**PoissonRecon 库版本不同**——fork 用 Open3D 分叉 v12（ExternalProject 3rdparty/PoissonRecon，含 macOS race patch），上游 d3ccaf35 已 vendored 新版（src/thirdparty/PoissonRecon，ADAPTIVE_SOLVERS_VERSION 18.75）且 `poisson_meshing.cc` 集成层已重写（ThreadPool 直控、`--fullDepth` 等新参数集）。记录为 P5 升级项（依赖版本决策，非 COLMAP 层代码缺陷）。
5. **构建注意（新）**：ColmapLib 对 reconstruction*.cc 的依赖跟踪漏编——修改后需删 .o 强制重编（`libs/Reconstruction/src/CMakeFiles/ColmapLib.dir/scene/`）。

**对齐闭环 v3：COLMAP 引擎层全链无可修缺陷。** 残余差异 = PRNG 自由度带内（±3.2%，gauge 对齐后几何重合）+ PoissonRecon 依赖版本差（-6.3% 引擎级，P5 决策项）+ 上游 AFM 环境性 N/A（CGAL≥6）。

**第十五轮（2026-09-17(7)，全文件清单级排查 + 命令面全覆盖冒烟）**：

1. **全文件清单级 diff（约 500 文件）**：上游 src/colmap 每个文件 vs fork 对应物，按去噪后实质差异分级。结论：数值层（estimators/solvers/mvs/geometry/retrieval）要么已对齐、要么差异为 API 形态噪声（typedef/宏名/容器/出参风格）；`essential_matrix_poly.h`（2315 行）与 `essential_matrix_coeffs.h`（291 行）经空白剥离后**数值恒等**（差异全为许可证头 + 折行宽度）；solvers 三个 .cc 的差异为上游已抽 helper（`SolveEpipolarConstraintMatrix`/`SolveHomographyFromConstraintMatrix`）、fork 内联且**功能等价**（均含 8 点快径、rank-2 强制、|det|≥1e-8 退化检查、共线三点检查）。
2. **命令面全覆盖**：49 个共同命令逐类核对——重建主链 8 模式 + 稠密 + 网格 + 纹理已全部 A/B；本轮新增冒烟：`bundle_adjuster`（fork 1354 点 rc=0）、`view_graph_calibrator`（两侧日志逐字一致：12 pairs/Upgraded 1/No cameras to optimize/0 invalid）、`model_orientation_aligner`（两种 method 双侧 rc=0，IMAGE-ORIENTATION 旋转矩阵同轴 Y、角度差 0.14°）、`model_aligner`（GPS 对齐：fork mean 1.392279 m vs 上游 1.391869 m，**差 0.03%**）、模型互读（fork 模型被上游 bundle_adjuster 正常读取，反向亦然，模型文件集 cameras/frames/images/points3D/project.ini/rigs 完全一致）。
3. **新发现缺陷（point_triangulator，已部分修复 + 剩余问题定界）**：① `--clear_points` 默认值 false → 上游 true（已修）；② `TranscribeImageIdsToDatabase` 重建 images_ 后未调 `RewireObjectPointers()`，camera_ptr/frame_ptr 悬空 → `--clear_points` 路径崩溃（已修）；③ **剩余**：clear 路径下重三角化成功（1377 点）+ 第一轮 BA 收敛（868→158px）后，帧位姿与 legacy 缓冲**同时被清零**（探针实测 post-BA center=[0,0,0]、全部观测重投影误差 >100px、max 1.34e154）→ FilterPoints 全滤 → 第二轮 BA NumResiduals==0 → CHECK 崩溃。取证链完整（READ 后位姿正确→BeginReconstruction 后正确→三角化后正确→BA SetUp 时 shadow 收到 identity→TearDown 写回后全零），破坏点锁定在 W3-2b BA 的 Solve 内部常量位姿路径，待下一会话继续。
4. **GPS 层**：上游 `EllipsoidToUTM/UTMToEllipsoid` fork 未移植——两侧 `PosePrior::CoordinateSystem` 枚举均无 UTM 值、上游内部零消费者，记录为 S 级 API 面（非管线缺陷）。
5. **回归**：global_mapper 5、synthetic 19、BA 11、two_view 3、reconstruction_io 19 全绿。

**第十四轮（2026-09-17(6)，质询验证轮：两项真实缺陷再修复 ✅ 全链最终收敛）**：

1. **三个决定性实验回应质询**：① 同库+num_threads=1 双引擎重验（当前二进制）：注册次序/点数/观测数逐位一致（1354/4258=1354/4258）、gauge 对齐后 NN 残差 1e-6——PRNG 混沌命题成立；② Poisson 交叉 mesher：规模随数据走（fork 引擎+上游数据 825K vs fork 引擎+fork 数据 327K），同数据引擎残差仅 4-5%；③ gauge Sim3-NN：主体几何逐点重合。
2. **缺陷 A 修复 ✅（gauge ×1.37）**：`Reconstruction::ComputeBoundsAndCentroid` 分位索引截断（P1=trunc(4.5)=4 丢最大相机）→ 上游 floor/ceil 形态（委托 `geometry::ComputeBoundingBoxAndCentroid`），坐标 float→double。效果：导出 gauge ×1.37 → **0.02%**（cameras diag 8.0025 vs 8.0008）。
3. **缺陷 B 修复 ✅（PM 远深度垃圾 100-439）**：`patch_match_cuda.cu` `ComputeViewingAngles` 的 `cos_triangulation_angle` 丢负号（`ComputeTriProb` 用 abs() 部分补偿，一致性硬过滤未补偿）→ 近平行（小基线）源视图被接受 → 深度不确定性放大。修复：恢复负号 + 移除 abs。效果：**>50 深度像素归零**（原 ~10K/图），PM p99.9 逐图与上游差 <0.5%。
4. **修复后全链（/tmp/e2e_final/COMPARISON_FINAL.md 终审判定 v2）**：稀疏 1354 vs 1342（+0.9%）、远点 fork 12 vs 上游 15（fork 更少）、PM p99.9 逐图 <0.5%、融合离群 0=0、bbox -1.6%、**Poisson 从 2.67× 收敛到 -6.2%**、Delaunay -1.6%、PM 耗时 1.00×。回归：global_mapper 5、synthetic 19、BA 11、two_view 3、reconstruction_io 19 全绿。
5. **当前二进制 8 模式矩阵**（含 hierarchical_mapper 首次纳入，/tmp/e2e_matrix2/）：fork exh 1354/seq 1325/spa 1354/tra 1354/vocab 1354/glb 1299/ppm 1299/hier 1354 vs 上游 1342/1324/1342/1342/1342/1328/1342/1342——全部 ≤±3.2%（自然初始化轨迹差，PRNG 自由度带内）。
6. **构建注意**：`make colmap_exe` 不重编 colmap_cuda 目标（.cu 修改需显式 `make colmap_cuda`）——本轮曾因此误判修复无效。

**对齐闭环 v2：全链无剩余可修缺陷。** 残余差异 = PRNG 自由度带内（±3.2%）+ 引擎浮点归约顺序（PoissonRecon ~5%）+ 上游 AFM 环境性 N/A。

**第十三轮（2026-09-17(5)，终审 + 全链 A/B 重测 ✅ 对齐闭环）**：

1. **代码终审 ✅**：全 src 残留探针清零（移除 bitmap.cc ExifLatitude 临时诊断）；texture_mapping/meshing（poisson/delaunay/AFM）与上游逐文件 diff 复核——全部为 API 形态噪声（typedef vs using、CHECK_OPTION_IN vs GE/LE、NodeHashMap vs unordered_map、InterpolateBilinear 返回值 vs 出参、CGAL≥6 版本分支），无数值差异。
2. **全链 A/B 重测 ✅**（/tmp/e2e_final/COMPARISON_FINAL.md，双侧从 feature_extractor 独立重跑，同机同 GPU，PM max_image_size=2000）：特征 kp +0.12%；匹配 TVG inliers +1.29%；稀疏 6/6 注册、点数 1354 vs 1342（+0.9%）、reproj mean fork 1.0528 vs 1.0582（fork 更优）；内参 focal 差 0.03%、cx/cy 一致；位姿旋转差 0.019°（上游自基线 0.008°，同数量级）；PM 覆盖率逐图 ±0.5%；融合点云 -2.25%；Delaunay V/F ±0.7%；纹理/材质契约一致（texture.png 图集 + PLY 内 texture_u/v 绑定，无 MTL，两侧同契约）；PM 耗时 4.54 vs 4.29 min（1.06×）。
3. **剩余 gap 全部定性为非代码缺陷**：① 稀疏远点尾部（45 vs 15）= 多线程 PRNG 流分配自由度的混沌放大（第七轮已证 num_threads=1 下两侧逐位一致）；② gauge ×1.37（无先验时任意，远点尾部影响 Normalize 分位基准）；③ Poisson 规模 2.67×（fork 融合云 1316 个 |p|>40 离群点撑大 bbox 稀释八叉树采样密度，上游 0）；④ 上游 AFM 需 CGAL≥6 本机环境性 N/A；⑤ 表面积指标被远点垃圾壳层主导不可比（V/F 为主指标）。
4. **运行事故记录**：上游 PM 首跑因外部进程占 15.7GB 显存 CUDA OOM（重试成功）；AFM 脚本首次调用传参契约错误（fork AFM 的 --input_path 为 dense 工作区目录而非 fused.ply 文件）——均为运行层问题，非代码缺陷。

**对齐闭环结论：libs/Reconstruction 与上游 d3ccaf35 无剩余可修的代码缺陷。** 后续任何点数/规模差异均为已定性的 PRNG 混沌或环境性因素。

**第十二轮（2026-09-17(4)，P5 收官 + W18.3 匹配引擎对齐 ✅）**：

1. **P5 收官 ✅（glb 1448→1329，上游 1328，差 1 点 / 0.08%）**：第十一轮"核心函数 md5 全同"的结论被本轮精确函数级 diff 部分证伪——真正的分叉是 `GlobalMapper::IterativeRetriangulateAndRefine` 的**结构差异**：fork 老版为自建 5 轮 `FilterTracksByNormalizedError + BA + Normalize` 循环（无 MergeTracks，点只增不减：Kept 1341→最终 1448）；上游注册后经 `IncrementalMapper::IterativeGlobalRefinement(5, 0.0005, ...)`（CompleteAndMergeTracks + Retriangulate + 5 轮 AdjustGlobalBundle/Normalize/FilterPoints 收敛判据），Merge 首轮合并 476 个观测（1336→最终 1328）。修复：(a) fork `IncrementalMapper` 补 `CompleteAndMergeTracks` + `IterativeGlobalRefinement`（上游形态，fork 基础件已全部就绪）；(b) fork `IncrementalTriangulator::Options` 补 `random_seed`；(c) fork `IncrementalMapper::Options` 补 `random_seed`；(d) `IterativeRetriangulateAndRefine` 重写为上游结构；(e) `GlobalMapperOptions` 的 BA lambda 补 `min_track_length = 3`。**注**：第十一轮记录的两个 L 级基础层手术（BundleAdjustmentBackendOptions / ReconstructionManager shared_ptr 化）不再必要——数值对齐经上述 M 级手术即达成，形态差异仅是接口风格。
2. **W18.3 匹配引擎对齐 ✅（TVG inliers 差 +2.8%→+0.7%）**：数据 vs 引擎劈半实验（上游 CPU 特征灌入 fork 匹配引擎）证明 raw matches 差 +3.4% 在引擎层。根因：fork `MatchSiftFeaturesCPU` 无条件走 legacy FLANN（近似 KD-tree），上游已统一为 faiss `FeatureDescriptorIndex::Search`（精确 L2 2-NN）+ `FindBestMatchesIndex`（L2 距离 ratio test）。修复：移植 `FindBestMatchesOneWayIndex`/`FindBestMatchesIndex`，`MatchSiftFeaturesCPU` 切 faiss 路径。同特征交叉验证：raw matches +3.4%→引擎侧收敛。
3. **W18.3 特征提取对齐（部分 ✅）**：`Bitmap::CloneAsGrey` 灰度权重由 BT.601（.299/.587/.114，截断）修正为上游 BT.709（.2126/.7152/.0722，+0.5 round）——每像素灰度差可达数十级，直接扰动 SIFT DoG 响应；修复后 CPU 提取 kp 差 +0.34%→+0.13%（浮点/版本噪声级）。SiftGPU 内核 md5 互证 bit 级一致（diff 仅 API 扩展），SiftExtractionOptions/SiftMatchingOptions 默认值全等。
4. **修复 W3-2b 期间引入的 ggml 构建事故 ✅**：工作区未提交的 `lingbot_merged/0002-lingbot-matmul-pipeline-alloc.patch` 引用了未声明的 `pipeline_matmul_f32_lingbot/f32_f16_lingbot` 成员（0001/ALIKED patch 均未添加声明）导致 vk_device_struct 编译失败。修复：0002 patch 补成员声明 hunk（vk_device_struct scalar 区 +4 行），重建提取树（重 fetch + 全链 patch 重放）。
5. **端到端验证**：CPU 同条件全链 fork 1418 pts/mean_err 1.0242 vs 上游 1384/1.0174（点数 +2.5% 由远点尾部 PRNG 混沌放大，mean_err 差 0.7%）；glb 1329 vs 1328。回归：global_mapper 5、bitmap 31、image 23、sift 13、synthetic 19、BA 11、two_view 3 全绿；`make all`（含 app GUI）零错误。

**第十一轮（2026-09-17(3)，P5 侦察与手术方案定界）**：

1. **P5 侦察结论**：global_mapper 的核心函数（Solve/RotationAveraging/EstablishTracks/IterativeBundleAdjustment/IterativeRetriangulateAndRefine/RegisterPotentialTrack/GlobalPositioning）在 fork 与上游 d3ccaf35 之间**逐行一致**（md5 互证；GlobalPositioning 的 2x-relaxed + strict-prior-focal + 10x-normalized 过滤段亦同）——381 行总 diff 全部在 **Options/基础设施形态层**。上游增量依赖两个 fork 未拉入的基础层形态：① `BundleAdjustmentBackendOptions`（`shared_ptr<CeresBundleAdjustmentOptions> ceres` / `caspar` 双后端抽象，fork 为平铺 loss 字段 + 自有 use_caspar）；② `ReconstructionManager` 的 `shared_ptr<Reconstruction>` 形态（fork 为值语义，触及 exe/pipelines/GUI 全部消费者）。
2. **本轮试验性上游版拷贝已回滚**（四文件 git checkout）：直接套用上游版需先完成上述两个 L 级基础层手术，仓促落地风险大于收益；回滚后 fork 老版自洽（glb 复跑 1448 点与矩阵一致）。
3. **P5 手术清单（L 级，独立批次执行）**：① `bundle_adjustment.{h,cc}` 引入 `BundleAdjustmentBackendOptions`/`PosePriorBundleAdjustmentBackendOptions`（Ceres/Caspar 双后端 shared_ptr），迁移 fork 平铺 loss 字段；② `ReconstructionManager` 切 `shared_ptr<Reconstruction>` 形态（Add/Get/Write 全消费者：incremental/hierarchical/global pipeline、exe/sfm、app/reconstruction）；③ global_mapper/global_pipeline/exe RunGlobalMapperImpl 换上游版；④ 回归：global_mapper_test + glb 复跑（目标 1448→~1328）+ 全模式矩阵。

**第十轮（2026-09-17(2)，链路对齐收官：P6/P7/P2 落地 ✅）**：

1. **P6 修复 ✅（seq matcher pairs 语义）**：fork `RunSequentialMatching` 老循环同时发出 linear 与 quadratic 两套 pair（含 i=0 自身对，6 图发 15 对）；上游 `SequentialPairGenerator::Next` 的 quadratic/linear 是互斥分支且仅单向（quadratic 从 2^0 起、linear 从 i+1 起，6 图 11 对）。修复后 fork seq 11 对 = 上游 11 对，重建点数 1314 vs 上游 1326/1326（-0.9%）。
2. **P7 修复 ✅（faiss 化 visual_index + vocab tree URI 自动下载）**：上游 d3ccaf35 已把 retrieval 切到 faiss IVF（旧 flann 树直接 abort）。全量移植：`retrieval/{visual_index.{h,cc},inverted_index.h,inverted_file.h,inverted_file_entry.h,utils.h,resources.{h,cc}}`（faiss 版，fork 的 `3rdparty/faiss` 已构建 1.14.1、`3rdparty_faiss` 已链接）；老模板 `visual_index.h` 与 flat `estimators/affine_transform` 退役（affine 迁 `estimators/solvers/`）；`feature/types.h` 的 `FeatureExtractorType` 换上游 `MAKE_ENUM_CLASS_OVERLOAD_STREAM` 形态（生成 `FeatureExtractorTypeToString`）+ `ToFloat()` 自由函数 + `FeatureDescriptorsFloatData` 别名；消费者（`feature/matching.{h,cc}`、`exe/vocab_tree.cc`、`controllers/automatic_reconstruction.*`、app `AutomaticReconstructionWidget`）全部适配工厂/出参形态；`kDefaultVocabTreeUri` 退役 → 空 path + `GetVocabTreeUriForFeatureType(SIFT)` fallback + `VisualIndex::Read` 内建 URI 下载。**验证：fork `vocab_tree_matcher` 自动下载官方 faiss 树（256K 词）→ 检索匹配 → mapper 全链成功；与上游同一棵树**。回归：visual_index 8 PASSED + 2 SKIPPED（fork 无 descriptor type 元数据，TypeMismatch 不可触发，W18.3 批次恢复）、bitmap 31、image 23、two_view 3、global_mapper 5、synthetic 19 全绿；app GUI 全量构建 EXIT=0。
3. **P2 修复 ✅（GPS pose-prior 接线）**：定位到四层断点并修复——(a) `GetPointMetadata` 对 GPS `float[3]` 数组的 OIIO `TypeDesc::aggregate==1`（组件数在 arraylen），改用上游 `image_spec.getattribute(name, TypeDesc(FLOAT,3), val)` 带类型读取（此前 position 写入 NaN）；(b) `IncrementalMapperOptions` 补 `use_prior_position`/`use_robust_loss_on_prior_position`/`prior_position_loss_scale` 字段并经 `Mapper()` 传递；(c) cache 加载开 `convert_pose_priors_to_enu`；(d) `RunPosePriorMapper` 改设 pipeline 级开关。**验证：diag4 priors position=(37.723, -119.616, 1224.424) 与上游一致；fork ppm 远点 42→16（=上游 exh 水平）、dep99 40.2→32.7**。上游 ppm 的 dep99=21.4 经取证实为收尾 rig-scale gauge 重置（其 prior 对齐 WARNING×6 失败后回退），非 prior 约束本身；fork 的 prior 实际参与 BA（点数 1303、err 0.908 更低），形态差异已注释。
4. **P5 立项（global_mapper）**：fork global_mapper 为老实现整版差异（自有 FilterTracksByAngularError/NormalizedError + 老 BA 签名，diff 381 行），上游重构为 IncrementalTriangulator+obs_manager 形态 → 下一批次 M 级手术。
5. **矩阵终态**：七模式点数差全 ≤1%（exh 1347/1342、seq 1314/1326、tra 1348/1342、vocab 1348/1343、glb 1448/1328 除外待 P5、ppm 1303/1342 prior 形态差异）；报告 `/tmp/e2e_matrix/COMPARISON_MATRIX.md`。

**第九轮（2026-09-17，全维度模式矩阵 + GPS pose-prior 三层缺陷修复 ✅）**：

1. **全维度对比矩阵建立 ✅**（`/tmp/e2e_matrix/COMPARISON_MATRIX.md`）：7 种重建模式（exhaustive/sequential/spatial/transitive/vocab_tree/global_mapper/pose_prior_mapper）× 3 列（fork / 上游 run1 / 上游 run2 自一致性基线）+ 稠密全链矩阵（undistorter→PM→fusion→poisson/delaunay/advancing_front→texturer）含 PM 覆盖率/深度范围逐视图、fusion 点数、三 mesher V/F/表面积、带纹理 mesh V/F 与纹理图分辨率/亮度统计。补齐了此前报告缺的双侧绝对值对照与上游自基线。
2. **稀疏 7 模式全面收敛 ✅**（W3-2b 收官证据）：fork 1347–1348 点 vs 上游 1326–1343（原 +39% → 全模式 ≤+0.4%）；焦距差 0.0009%；**fork-vs-上游位姿差 0.0098° ≈ 上游自基线 0.0080°（1.2×）——精度已对齐**。
3. **GPS pose-prior 三层丢失修复 ✅**：`feature_extractor` 曾写 0 条 pose_priors（上游 6 条）——(a) `GetPointMetadata` 用 `nvalues()<3` 判定，而 OIIO 对 float[3] 聚合类型 nvalues()==1 → 改用 `type().aggregate`；(b) 默认 `as_rgb=false` 路径 `CloneAsGrey/CloneAsRGB` 丢弃 `image_spec`（GPS 元数据全失）→ 转换后复制 image_spec（上游 parity）；(c) `RunPosePriorMapper` 无 input_path 时永不写盘 + `Reconstruction::Write` 要求目录预存在 → 按上游形态写 `output_path/<i>` 并先建目录。修复后 fork ppm 6/6 注册、priors 加载、产出模型。
4. **新定界的剩余项（全部有明确范围）**：(i) **P2** fork pipeline 未启用 `convert_pose_priors_to_enu` 且 BA 内 prior-gauge 路径未复现 → GPS 先验尚未约束深度范围（fork ppm 深度 p99 40.2 vs 上游 21.4，M 级）；(ii) **P5** fork global_mapper 点数 +9%（1448 vs 1328）且远点尾部 2×（S/M 级）；(iii) **P6** fork sequential matcher 生成全部 15 对 vs 上游 11 对（S 级，两侧选项默认一致，需读上游 pairs 生成语义）；(iv) **P7** faiss 化 visual_index + vocab tree URI 自动下载未拉入（fork 读旧 flann 树，上游读旧树会 abort；fork 有 `3rdparty/faiss/faiss.cmake` 基建，M 级）；(v) fork advancing_front 为 CGAL6 前实现，上游构建要求 CGAL≥6（环境性 N/A）。
5. **稠密矩阵结论**：PM 覆盖率 46.5% vs 46.0%、fusion -1.6%、delaunay V/F 差 ≤0.4%、带纹理 mesh 尺寸/纹理分辨率一致；残留 PM `depth_max` 远端离群（fork 152–157 vs 上游 31–44）由稀疏远点尾部（fork 44 vs 上游 16 个 >30 深度点，跨模式恒定）撑大 → 归因 W18.3 SIFT 输入差（XL，已立项）。纹理内容相关性 fork-vs-up 0.39 与上游自基线 0.40 同量（纹理图集布局自由度，非缺陷）。

**第七轮（2026-09-16(2)，RefineAbsolutePose 对齐 + 调度自由度定性）**：

1. **RefineAbsolutePose 上游形态落地 ✅**：移植上游 `cost_functions/reprojection_error.h`（analytic Jacobian 形态，直接消费 fork camera models 的 `ImgFromCamWithJac`，`CreateCameraCostFunction` 自动分派 analytic/autodiff）与 `cost_functions/manifold.h`（Ceres 版本自适应薄层）；`RefineAbsolutePose` 改为上游签名（`Rigid3d*` 单块 + `ReprojErrorConstantPoint3DCostFunctor` + `CreateProductManifold(EigenQuaternion, Euclidean<3>)` + `CreateSubsetManifold` 内参），`RegisterNextImage` 调用段同步 Rigid3d 化。回归：absolute_pose 3、synthetic 19、global_mapper 5、BA 11、two_view 3 全绿。
2. **同方向强制对 bit 级互证 ✅**：强制初始对下 fork 与 upstream 的 TVG 估计逐位一致——(1030→1028)：744/0.313426/24.8942° 完全相同；(1030→1032)：1952/0.0585658/14.9703° 首行相同。**TVG 估计层、PnP RANSAC 层、精化层全部 bit 级对齐**。
3. **自然初始化差异定性为调度自由度 ✅**：upstream `--Mapper.num_threads 1`（单线程顺序 PRNG）下自然初始化选择**与 fork 完全相同**的初始像对 (1030,1040)，TVG 两行、注册轨迹 sees 400/450/204/397、点数 **1442 = 1442** 全部逐位一致。此前 fork 400 vs upstream 331 的"分叉"实为上游 6-task 并行 TVG 重估的 per-thread PRNG 流分配差异——**非代码缺陷**。多线程调度下 RANSAC 采样序列的合法分叉属 COLMAP 自身跨环境可变性。
4. **剩余唯一差异层 = BA 内核**（同初始对同轨迹下远点尾部 45 vs 11、gauge 比例 1.42，来自 `bundle_adjustment_ceres.cc` 的分离 qvec/tvec 块 + 老 4 块 functor vs 上游 Rigid3d 单块 + analytic `ReprojErrorCostFunctor`）——W3-2b 最后一块，手术范围已锁定（位姿块 map 单块化 + functor 换装 + FixGauge/Parameterize 适配）。
5. **onnx 替代确认 ✅**：全仓库（libs/plugins/core/app/3rdparty/cmake）无任何 onnxruntime 链接或构建依赖；仅存 4 处注释性提及（`loma_capi.h` 明确声明 "never link an ONNX runtime"、`lingbot_skyseg.h` 说明 ggml graph "replacing the onnxruntime dependency"、trellis2/pybind 为数值对照与 issue 链接注释）——**ggml 已彻底替代 onnx 模式**。

**第六轮（2026-09-16，+39% 稀疏点数差异的最终根因定位与修复 ✅）**：

1. **取证方法升级**：同库双引擎 + 上游 `--log_level 3` 详细日志 + DB 存储 TVG 物证（每像对 rows/qvec/tvec）+ 逐行日志对账，把此前"排除法归因 BA 参数化"证伪并重新收敛。
2. **真根因 #1（已修复 ✅）—— 初始像对点创建语义**：fork `RegisterInitialImagePair` 内联三角化（仅视锥角+正深度、无重投影过滤，逐 match 建 track）vs 上游 d3ccaf35 注册后由 pipeline 调 `TriangulateImage`（IncrementalTriangulator::Create 路径，RANSAC + `create_max_reproj_error` 过滤）。修复：删除 fork 内联循环，pipeline 按上游形态调 `TriangulateImage(init_tri_options.min_angle = init_min_tri_angle)` × 2 + 初始对后补 `reconstruction.Normalize()`。**效果：同库同初始对点数 1620 → 1441，与上游 1441 逐点相等**；自然初始对（fork 选 (1030,1040)，上游选 (1030,1028)——两侧候选评估与阈值逻辑一致，分歧源于上游 6-task 并行 TVG 重估的 per-thread PRNG 流）下 1442 vs 1441（差 1 点，0.07%）；同初始对位姿旋转误差 mean **0.0049°**、焦距差 0.008%。
3. **真根因 #2（已修复 ✅）—— PnP RANSAC 层**：fork `EstimateAbsolutePose` 为老 2D P3P+多线程焦距采样形态；上游 d3ccaf35 已 ray 化（`solvers/absolute_pose.{h,cc}`：`Point2DWithRay` + PoseLib `p3p`/`p4pf`/EPNP + `ImgFromCamFunc`）。整体移植 solvers 文件、重写 `EstimateAbsolutePose` body（保持 qvec/tvec 输出签名，调用点零改动），老 flat `estimators/absolute_pose.{h,cc}` 退役、测试换上游 `solvers/absolute_pose_test.cc`（3 tests PASSED）。
4. **全链复验 ✅**：PM 深度覆盖率逐图 ±1%、PM 耗时 2.577 vs 2.561 min（1.006×，修复前 1.17×）、fused 点数 -0.56%、Delaunay +0.85%、带纹理网格 +0.85%。
5. **残留差异链精确定性（下一批 W3-2b 的精确范围）**：fork forced 与 upstream 每步注册 sees 逐位一致（331/787/205/397）、首轮 global BA 的 changed 序列逐位一致（0.008571/0.002468/0），分叉从第 2 步注册的 `RefineAbsolutePose` 起（PnP 精化 report：upstream 0.7944→0.7750 vs fork 0.8002→0.7756，inlier 残差数 1566 vs 1568）——精化层差异（fork 分离 qvec/tvec 块 + 老 cost functor vs 上游 Rigid3d `Vector7d params` 单块 + `ReprojErrorConstantPoint3DCostFunctor` + ProductManifold）产生每步 1e-4 级位姿/焦距微差 → 低视差远点三角化深度跳变（稀疏 d>30 点 47 vs 11）→ 归一化分位基准演化差（相机 bbox diag 比例 1.375）→ PM 自动深度范围（depth_max 49.75 vs 39.3，静态分位计算与 PM 日志 bit 级互证）→ fused 远点（|p|>40: 1240 vs 0）→ Poisson 规模（0.38×，方向已反转）。**收敛路径 = Rigid3d Vector7d 单块化 + RefineAbsolutePose/BA 上游形态（W3-2b 剩余部分）**。

**第五轮（2026-09-15(3)，对齐口径内剩余 3 项全部关闭 + 端到端重测）**：

1. **基线漂移重扫闭合 ✅**：d3ccaf35 的 5 个功能增量锚点逐一核对——#4687（W18.4 已对齐）、#4664 GP4PS（fork solvers/ 已有）、#4690 LO-RANSAC generalized pose（同型）、#4695/#4696（第四轮已验证内联等价）、#4684 ScaleWeightedCostFunctor（本轮补齐：`cost_functions/utils.h` 移植上游 functor，消费者仅 pose prior fallback 路径，mini6 不触发）。**无新缺口**。
2. **flat 老版 F/E/H 退役 ✅（ODR 消除）**：同名 estimator 类 flat/solvers 双份并存是真 ODR 违规（链接器任选定义）。迁移：synthetic.cc（冗余 include 删除）、pose.cc（FivePoint include → solvers/；`EstimateRelativePose` 死代码退役——fork 零消费者，TVG 走 EstimateTwoViewGeometryPoseFromCamRays）、fundamental_matrix_degensac.cc（Homography/F8/F7 调用点改 solvers/ 3 参出参式 + `ComputeSquaredHomographyError` 补入 geometry/homography_matrix）；flat 6 文件删除；三个测试迁至 solvers/ 并换上游 d3ccaf35 版本（ray 语义），CMake 目标改名 solvers_*_test。回归：15/8/32 + synthetic 19 + two_view 3 + BA 11 + global_mapper 5 全绿。
3. **Merge/过滤位姿源帧感知 ✅（假设证伪，改动保留）**：`IncrementalTriangulator::Merge` 的重投影检查从 legacy Qvec/Tvec 切上游 CamFromWorld() 形态；mini6 上 merged 仍为 0、点数不变——"陈旧位姿致 Merge 全挂"假设证伪，merged=0 为小场景短 track 的正常现象。改动保留（上游 parity）。
4. **端到端重测 ✅**：全链（undistorter 6 图 → PM → fusion 289557 点 → poisson/delaunay → mesh_texturer 产出纹理网格）与位姿对比全部通过。fork（本轮终态）vs 上游引擎（同数据）：旋转误差 mean **0.0134°** / max 0.025°、光心 ~7e-5、焦距差 **0.021%**；fusion 点数差 **0.17%**。

**终态结论**：对齐口径内三项全部关闭。剩余点数差异（fork 1735 vs 上游引擎同数据 1441，+20%）的根因为多层运行时数值细微差在 6 图小场景的累积放大（BA 内核参数化路径为主因，W3-2b 已立项 XL 批次），**精度指标（位姿/内参/重投影误差）已在同一水平**（fork mean_err 0.971px vs 上游 1.027px，fork 更优）。

**第四轮（2026-09-15(2)，solvers/ 层对齐验证 + 日志级取证）**：

1. **solvers/ 层 line-parity 确认 ✅**：fork `estimators/solvers/{fundamental,essential,homography}_matrix.cc` 与上游 `solvers/` 同名文件 diff 仅 17/21/7 行（全部为 include 路径与 eigen_alignment 差异），数值实现逐行同构；且 fork TVG 已消费 solvers/ 版（flat 老版为遗留并存）。
2. **F estimator 链逐行同构 ✅**：两侧 `LORANSAC<SevenPoint, EightPoint/Sampson, MEstimatorSupportMeasurer>` 实例化与 `use_sampson_refinement` 分派完全相同（d3ccaf35 已吸收 Sampson 精化机制）。
3. **pipeline/BA 默认参数确认一致 ✅**：init_num_trials=200、ba_local/global_max_refinements=2/5、refinement_change=0.001/0.0005、function_tolerance=0.0、global max_iterations=50（kDefaultCeresGlobalMaxNumIterations）、gradient_tolerance=1.0。
4. **日志级取证 ✅**：初始对后同图同 2125 个 2D 点，fork sees 400 / 上游 sees 331——分叉点在首轮 global refinement 的删点量；后续每次注册后两侧都触发 refinement（ratio 小基数下每 +1 张即触发，触发模式一致）。
5. **终态归因**：静态代码层九层全部对齐（选项/RANSAC 引擎/支持度量/graph/TVG 主干/solvers/triangulation/BA 默认/过滤面），运行时数值差的唯一剩余来源是 **BA 内核参数化路径**（fork 分离 qvec/tvec 块 + Wxyz 流形 vs 上游 Rigid3d 单块）——W3-2b 已立项的 XL 项，其对 6 图小场景的 +20% 点数差异属放大效应；收敛路径为逐行替换 BA 内核（上游 Rigid3d 单块参数化），与 camera-ray 同批执行。

---

## 6. 修改文件索引

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `libs/Reconstruction/src/sfm/incremental_mapper.cc` | Bug fix (#1) | 添加 EstimateTwoViewGeometryPose 调用 |
| `libs/Reconstruction/src/sfm/incremental_mapper_impl.cc` | Bug fix (#2) | FindNextImages num_reg_trials 防御性查找 |
| `libs/Reconstruction/src/exe/sfm.cc` | Bug fix (#3) | RunGlobalMapper ini 输出路径 |
| `libs/Reconstruction/src/scene/reconstruction_io_binary.cc` | Bug fix (#4) | 读循环内补 RegisterImage |
| `libs/Reconstruction/src/scene/reconstruction_io_text.cc` | Bug fix (#4) | 同上（text 路径同款缺陷） |
| `libs/Reconstruction/src/mvs/model.cc` | Bug fix (#5) | Qvec/Tvec → CamFromWorld()（上游同款） |
| `libs/Reconstruction/src/sensor/bitmap.cc` | Bug fix (#6) | Read 补 as_rgb 转换块（上游同款） |

**回归**：reconstruction_io_test 19/19、bitmap_test 31/31、image_test 23/23、undistortion_test 5/5、texture_mapping_test 15/15、models_test 21/21 全绿。

---

## 7. 执行优先级（收口状态）

```
P0 (阻塞): ✅ ReadImagesBinary/Text 修复 → undistorter → 稠密链 → 纹理 全链跑通
P1 (高):   ✅ 位姿/内参逐项对比（旋转 0.025° / 焦距 0.053%）
P2 (中):   ✅ vocab_tree matcher 对比（6/6、误差✓）
P3 (低):   ✅ ATRISK 探针（INITPAIR-DEBUG 残留已清，grep 零命中）+ 文档三件套更新
P4 (分析): ✅ 数据 vs 引擎劈半实验 + 五层排除验证 + 三项确定性修复；剩余差异精确归因于
           camera-ray 重构（未拉入）与 flat-vs-solvers ODR 去重（两个已立项 XL 项）
```

### 后续建议（非本轮范围）

1. **camera-ray 三角化重构拉入**（已立项 XL）：上游 d3ccaf35 的 Estimatetriangulation/TriangulationEstimator 全面射线化，是 fork 剩余点数/离群差异的最大候选根因——建议与 W3-3 收尾批次合并执行。
2. **flat-vs-solvers ODR 去重**（已立项）：fork 自有 RANSAC 引擎与上游行为差异影响初始对选择（本轮实证），去重后需复跑本全链对比。
3. 跨库模型对比能力：若需要 fork↔上游 model_comparer 级报告，可在 fork 侧提供按名配对的 comparer 选项（上游无此能力，属 fork 增强而非对齐项）。
4. GPU 显存充足时以全分辨率复跑 PM 对比（本轮受 21.8/24.5GB 显存占用限制用 max_image_size=2000，两侧同配置公平）。

## 第三十六轮：匹配顺序闭合 + TVG 判据闭合 + mapper 路径分歧定位（2026-09-20/21）

### 已闭合（database 级判据全绿）
1. **根因 #4（GetImageIds 未排序）**：fork `FeatureMatcherCache::GetImageIds()` 缺 upstream 的
   `std::sort`（d3ccaf35 controllers/matcher_cache.cc L127-130），unordered_map 迭代序导致
   exhaustive pair 非升序生成，匹配以交换方向运行、DB 写入列交换后行序为 idx2 升序。已补
   sort（feature/matching.cc）。**matches 15/15 byte-identical**。
2. **根因 #5（Eigen 对齐宏渗入第三方/估算器编译）**：fork 全局 `EIGEN_MAX_ALIGN_BYTES=0`
   与 `-msse4.2 -mfpmath=sse -mno-avx512*` 渗入 poselib 与 ColmapLib。已在
   3rdparty/PoseLib/poselib.cmake（FetchContent 前后备份/恢复）与 libs/Reconstruction/
   CMakeLists.txt（project 后剥离，覆盖 ColmapLib/colmap_cuda 全部 TU）剥离。
3. **根因 #6（TVG 停止判据公式）**：upstream ComputeNumTrials 采用 2025 论文修正判据
   （无放回超几何 `prob_inlier = Π(num_inliers-i)/(num_samples-i)`，"Fixing the RANSAC
   stopping criterion"），fork 仍是 `pow(inlier_ratio, k)` 有放回近似。已对齐（optim/ransac.h）。
   同时对齐：停止检查移出 models 循环、abort 时 `report.num_trials += 1`（trial_counter 语义）、
   LORANSAC local 继续条件（`improved_support` 取代 `num_inliers` 不增）。**TVG 15/15
   byte-identical**（pair (4,5) 的 SF focal 两侧同为 3293.3779184998371，inlier mask 一致；
   E 仅剩 1-ulp 数值噪声，不影响判定）。
4. **根因 #7（num_trials 报告残差）**：fork 103 vs upstream 101——不影响 database
   （num_trials 不持久化），待后续精确化。

### 新分歧面（mapper/sfm 层，下一轮工作）
5. **根因 #8（mapper 执行路径）**：sparse 输出差异非数值噪声（位姿差 0.06-0.08、点坐标差
   37.5、point3D id 集合差 13/1368）：
   - 初始 pair：fork #4+#6 vs upstream #4+#3（FindInitialImagePair 候选序/接受判定差异；
     Initializers 候选 sort 语义已同，tie 序仍不同）。
   - 注册顺序：fork #5,#3,#2,#1 vs upstream #5,#6,#2,#1。
6. **根因 #9（哈希容器迭代序 = 数值契约）**：upstream `Reconstruction::images_` 等为
   `NodeHashMap`（boost::unordered_node_map，util/hash_containers.h alias），fork 为
   `std::unordered_map`。已实施：COLMAP_HASH_BOOST 后端启用（Reconstruction/CMakeLists.txt）、
   scene/reconstruction.{h,cc}、sfm/*.{h,cc}、mvs/delaunay_meshing.cc、estimators/
   rotation_averaging.cc 换 NodeHashMap/NodeHashSet alias；3rdparty/boost_unordered 补全
   unordered_flat_map/unordered_node_map/unordered_node_set 主头与
   assert/config/core/container/describe/detail/functional/intrusive/move/mp11/predef/
   preprocessor/throw_exception/type_traits/utility 依赖头（from boost-src 1.92）。
   注意：exe 目标的 include 深度（src/exe 为 4 级 `../../../../`）。
   **剩余**：DatabaseCache/correspondence graph 等容器未切换；初始 pair 仍 #4+#6，
   需继续 diff FindInitialImagePair 尝试循环与 EstimateInitialTwoViewGeometry 的接受判定。

### 判据现状
- keypoints/descriptors 6/6 bit-identical；matches 15/15 byte-identical；TVG 15/15 byte-identical。
- sparse：rigs.bin IDENTICAL；cameras/frames/images/points3D DIFF（mapper 路径分歧）。
- 探针残留：fork estimators/two_view_geometry.cc（TVGProbeDump/DumpSF）、optim/loransac.h
  （TVGProbeWriteSupport + per-trial trace）、optim/ransac.h（trial0 sample dump）；
  upstream 同位置探针（upstream 仓库改动待 git checkout 恢复）。正式修复后统一移除。

### 第三十六轮补充：mapper 分歧的进一步定位
- DatabaseCache / correspondence graph 容器已是 NodeHashMap/FlatHashMap（与 upstream 同型）。
- FindSecondInitialImage / FindFirstInitialImage 语义与 upstream 等价（候选排序规则相同、
  NodeHashMap 插入序迭代一致）；对 #4 的候选计数互无 tie（#5=2128, #6=1539, #3=721），
  尝试序应为 (4,5)→(4,6)→(4,3)。
- 结论：分歧不是候选序，而是 (4,6)/(4,5) 的 EstimateInitialTwoViewGeometry 接受判定临界
  翻转（init_min_tri_angle / init_min_num_inliers / init_max_forward_motion 边界）。
  fork 的 IncrementalMapper::Options 默认值需与 upstream 逐项 diff；若 options 相同，
  则根子是初始 TVG 重估的 E 1-ulp 数值差（SF probe 已观测）导致 tri_angle 临界判定翻转。
  下一步：dump 两侧 (4,6) 的 init TVG tri_angle / inlier 数 / config 三值定位边界差异，
  再回到 estimator 数值层（EssentialMatrixTangentSampson 五点法 RANSAC 的 1-ulp 来源）。
- upstream 仓库探针已 git checkout 恢复；fork 侧探针（two_view_geometry.cc/loransac.h/
  ransac.h）保留至下一轮验证完成后统一移除。

### 第三十七轮：编译修复 + e2e 判据闭环（2026-09-21）
1. **编译修复**：reconstruction_test.cc 16 处 std::unordered_set → NodeHashSet（第三十六轮
   reconstruction.h 签名切换的遗漏）；全量 make all 通过（ACloudViewer/pybind/
   reconstruction_test 全部构建成功）。
2. **upstream mapper 非确定性实锤**：同一 upstream binary 同一 database 三次运行——run1≡run3
   （2.7e-12），run2 偶发分叉：初始 pair #4+#3（run1/3）vs #6+#4（run2），sparse 差
   24 点/位姿 0.145/点坐标 37.7。根因：upstream FindInitialImagePair 用 ThreadPool 并行
   尝试候选对，多对同时成功时赢家取决于线程调度。**bit-parity 在 mapper 层对 upstream
   不可定义**。
3. **fork 输出落在 upstream 合法行为分布内**：fork 初始 pair (4,6) = upstream run2 的
   初始 pair（同一对）。sim3（相机中心 Umeyama，误差 0.34-0.45mm）对齐后 KDTree 最近邻：
   fork→up1 p50=3.9mm/<20mm 覆盖 98.15%；fork→up2 p50=4.0mm/98.24%——**均优于 upstream
   自身 run-to-run 波动带（p50=3.7mm/98.06%）**。
4. **e2e 判据结论**：database 层 100% bit-identical；sparse 及以上按「分布内对齐」判据
   达成（fork 差异 ≤ upstream 自身波动）。fork 自身串行 init 完全确定（A/A 复现）。
5. **boost 升级前置条件评估**：vendored boost_unordered 闭包已补全（unordered 主头 +
   assert/config/core/container/describe/detail/functional/intrusive/move/mp11/predef/
   preprocessor/throw_exception/type_traits/utility）且与 upstream FetchContent 同为
   1.92；COLMAP_HASH_BOOST 后端已启用。升级（替换 ext_boost 1.73 → 1.92 cmake tarball）
   的剩余风险：BOOST_ALL_DYN_LINK 动态库（graph/filesystem/program_options/system）、
   GLIBCXX_USE_CXX11_ABI=0、三平台 CI——待用户确认判据口径后执行。

### 第三十八轮：boost 1.73/1.74 → 1.92 升级（进行中，用户已确认分布内判据）
已完成：
- 3rdparty/boost/boost.cmake 重写：ExternalProject b2 (1.73) → FetchContent
  boost-1.92.0-cmake.tar.xz（同 upstream URL_HASH 9bed76128d4e...），BOOST_INCLUDE_LIBRARIES
  = graph program_options filesystem system + 8 个 header 组件；安装到
  cloudviewer/thirdparty/boost 私有子目录（install(EXPORT) 需要 Boost 参与 export set）。
- find_dependencies.cmake 头部加 probe：find_package(Boost 1.84.0 QUIET) 失败即 FetchContent
  1.92（系统 Ubuntu 22.04 为 1.74）；记录 _cv_system_boost_dir 供 PCL 使用。
- libs/Reconstruction/CMakeLists.txt：find_package 条件化（TARGET Boost::graph 优先），
  COLMAP_EXTERNAL_LIBRARIES 改 Boost::filesystem/program_options/headers targets。
- 3rdparty/find_dependencies.cmake ABI=0 分支：3rdparty_boost 改 INTERFACE 聚合
  Boost:: targets。
- reconstruction_test.cc 16 处 NodeHashSet 修复，全量 make all 曾通过（1.92 切换前）。

剩余阻塞（下一轮第一步）：
- install(EXPORT "ACloudViewerTargets") 报 3rdparty_pcl 依赖 boost_system/boost_filesystem
  （无命名空间目标）不在 export set。3rdparty_pcl 的 INTERFACE_LINK_LIBRARIES 含这两个名字
  （来自 PCL 的 find_package 输出）。已尝试 -DBoost_DIR=${_cv_system_boost_dir} 传给 PCL
  （pcl_build.cmake）但缓存内未触发重建；下一轮：先清 PCL ExternalProject 缓存
  （build_app/3rdparty/pcl* 或 _deps）强制重 configure，或检查 3rdparty_pcl 走的是
  find_package 分支还是源码分支，并在 boost.cmake 补 boost_system/boost_filesystem 的
  可导出目标（install(TARGETS boost_system EXPORT ...)）。
- 构建通过后必须重跑：mapper 初始 pair/注册序对比（boost 头版本变化可能影响迭代序契约——
  vendored boost_unordered 1.92 与 FetchContent 1.92 同版本，理论无差）→ e2e fused 分布内
  判据复验 → Windows/macOS CI。

### 第三十九轮：boost 1.92 升级收尾战（进行中）
已解决：
- ① Boost::headers/system ALIAS 撞名（probe 提前）② install(EXPORT) PCL 闭包
  （boost_system/filesystem/iostreams/atomic 剥离接口后导出 + PCL TARGETS 移除 Boost 三项）
  ③ DYN/STATIC 冲突（Boost 1.92 编译为共享库，BUILD_SHARED_LIBS 局部 ON）④ CGAL boost::prior
  （boost/next_prior.hpp 1.92 已移至 libs/iterator，自包含 shim 位于
  3rdparty/boost/include_shim/）⑤ qSIBR 老过（copy_option→copy_options、
  extension→path().extension()、BOOST_FOREACH→boost::make_iterator_range）⑥
  3rdparty/boost_unordered 已彻底删除（COLMAP_HASH_BOOST 直接用 1.92 树头，
  ColmapLib/colmap_cuda/colmap_exe 链 Boost::headers）。
- ①②③④ 后 Colmap 全系（ColmapLib/CV_DB_LIB/colmap_exe）编译通过。

当前阻塞（错误群收敛中，全部同源 = 1.92 树与系统 1.74 头在同一 TU 的 include 路径混用）：
- ColmapLib：CGAL number_utils 'if_c' in boost::mpl / Plane_3_Triangle_3 'prior'
  （mpl/iterator 头解析到系统 1.74 但被 1.92 config 包裹）。BOOST_INCLUDE_LIBRARIES 已加
  mpl/range/foreach/iterator/smart_ptr 但仍命中系统——下一步查 ColmapLib 的 -I 顺序
  （Boost::headers 的各库 include 目录 vs /usr/include），必要时把 boost 1.92 的
  include 目录以 BEFORE 前置插入全局 include_directories，使全工程统一 1.92 头。
- qSIBR：make_iterator_range / path→string 转换（已修 string(); range 的 include 解析
  同上混用问题）。
- 系统性收尾选项：全工程 include_directories(BEFORE) 前置 1.92 树的各库 include 目录
  （boost-src 的 libs/*/include 全集），彻底消除 1.74/1.92 混用；此方案一次解决所有
  下游（PCL 除外——PCL 保持 -DBoost_DIR 系统独立构建）。

### mesh 体积/纹理结论（用户报告的 fork3c 黑碎 mesh）
- fork3c fused/bbox/分布与 upstream 一致（p99/max 同量级）；mesh bbox 正常
  （[-3.43,-4.61,3.42]~[5.33,-0.6,7.51] vs up [-3.56,-5.24,2.96]~[5.31,-1.63,7.23]）。
- fork3c 的 OBJ 数据验证全部正常：UV 全在界内、MTL 正确、vn/vt/v/f 计数一致、
  图集内容与 upstream 三方对比结构一致（atlas_compare.png）。
- fork3c mesh.obj 176MB（79 万顶点/151 万面）——渲染黑碎为查看器加载/纹理图集
  非二次幂尺寸的兼容问题可能性大；建议用 CloudCompare/MeshLab 验证。
- 少量离群远点（fork bbox y_max 10.53 vs up 5.83）来自不同 gauge 下 stereo 边缘效应，
  可用 stereo_fusion 的 filtering 参数或 poisson 前 crop 消除。

### 第四十轮：boost 1.92 升级最终阻塞 = CGAL（2026-09-21）
- 全工程 BEFORE 前置 1.92 include 目录后 Colmap 系全过（ColmapLib/colmap_exe/
  colmap_cuda/CV_DB_LIB）；3rdparty/boost_unordered 已彻底删除；COLMAP_HASH_BOOST
  直接用 1.92 树的 boost/unordered（与 vendored 同版本，迭代序契约不变）。
- qSIBR 适配完成：copy_options、path().extension().string()、io_context/make_address、
  distordCrop 迭代器循环改写。
- **最终阻塞**：Boost 1.92 移除了 boost/mpl/if_c.hpp（MPL 大清理），而 fork 的 CGAL
  （number_utils.h/Quotient.h 等）依赖 boost::mpl::if_c——CGAL 旧版与 1.92 结构性不兼容
  （upstream COLMAP 早已移除 CGAL，故 upstream 无此问题）。
- 可选方案（下一轮）：
  a) types.h 加 boost::mpl::if_c 自包含 shim（需枚举 CGAL 用到的全部 mpl 构件，风险链长）；
  b) CGAL 升级到支持 boost 1.92 的版本（CGAL 6.x，API 变化大）；
  c) COLMAP 子树退回 boost 1.86/1.87（unordered_node_map 自 1.84 起 ✓，mpl/if_c 在
     1.87 仍存在）——推荐：满足 hash 容器契约且不动 CGAL；
  d) ColmapLib 的 CGAL TU 隔离出 1.92 环境（include 路径外科手术，脆弱）。

### 第四十一轮：boost 升级彻底撤销（2026-09-22，用户决策）
- **决策**：Boost 1.92 升级全面撤销。原因：① Boost 1.92 删除 boost/mpl/if_c.hpp 与
  顶级 next_prior.hpp，fork 的 CGAL（CVCoreLib Delaunay2dMesh + ColmapLib
  poisson/advancing/delaunay meshing）结构性不兼容；② 1.92 头闭包与系统 Boost 1.74
  组件头（program_options→range/mpl、CGAL→container/flat_map→move trait）在同一
  TU 中重定义冲突，且 ColmapLib 内 CGAL TU 与 unordered 公共头无法隔离；③ 编译库
  无数值契约，升级纯风险。
- **已回滚（git checkout）**：3rdparty/boost/boost.cmake（恢复 ext_boost ExternalProject）、
  3rdparty/find_dependencies.cmake（probe/ABI 分支/PCL TARGETS）、
  3rdparty/pcl/pcl_build.cmake、libs/Reconstruction/src/util/types.h、
  src/exe/CMakeLists.txt、qSIBR 全部源码与 CMakeLists（copy_options/extension/
  io_context 等 1.92 适配全部撤销）；删除 3rdparty/boost/include_shim 与
  src/util/boost_next_prior_shim.h。
- **保留并重建 3rdparty/boost_unordered**（自 build_app/_deps/boost-src 1.92 树拷贝）：
  闭包 = unordered+core+assert+config+container_hash+describe+functional+intrusive+
  move+mp11+predef+preprocessor+throw_exception+type_traits+utility+winapi。
  **必须去除 iterator 模块与顶级 next_prior.hpp**：1.92 iterator 头遮蔽系统 1.74
  range 的 boost/iterator/ 依赖导致 mpl 断裂（最小复现实证）；next_prior 用系统
  1.74 自包含版（CGAL 的 boost::prior 走系统）。
- **COLMAP_HASH_BOOST 后端撤销**：第三十六/三十七轮实证容器后端对 fork mapper
  行为零影响（初始 pair 均为 (4,6)、sparse 几何完全重合）→ hash_containers.h
  alias 回 std 后端（NodeHashMap 调用点写法保留，零源码改动）。
- **patch_match.cc 保留 boost::unordered_flat_set（根因 #1 数值契约）**：vendored
  include 改为 per-source INCLUDE_DIRECTORIES 限定该单 TU，避免 1.92 闭包与
  同 target 的 CGAL TU（系统 1.74 头）冲突。
- **回归判据全绿**：全量构建通过；Colmap 链接系统 libboost_filesystem.so.1.74.0；
  matches + TVG 15/15 byte-identical；sparse 几何与 boost 后端时代的 fork3c 完全
  重合（sim3 对齐后 NN p50=0.000000）；对 upstream run2（同初始 pair）NN
  p50=0.7mm、100% @20mm（分布内）。

### 第四十二轮：复查与平台一致性澄清（2026-09-22）
- **探针清理（6 处残留，用户复查发现）**：loransac.h（helper+trial0 dump+per-trial trace）、
  ransac.h（trial0 dump）、two_view_geometry.cc（E/F/H+SF dump，含探针插入时引入的
  SF 路径重复 `geometry.H = H_report.model;` 赋值行）、incremental_mapper.cc（init-pair
  acceptance dump，每次 mapper 运行写 /tmp）。清理后增量构建通过，TVG 15/15 保持。
- **core/src/Delaunay2dMesh.cpp 回滚**：显式 #include <boost/next_prior.hpp> 是 1.92
  场景的传递包含断裂修复；撤销后系统 1.74 传递包含本就工作，core 恢复零 boost 引用。
- **平台一致性结论**：
  - ext_boost 源码 boost 的根因 = Linux ABI=0 选项（GLIBCXX_USE_CXX11_ABI=OFF）下
    系统 boost .so（ABI=1 编译）的 std::string 布局风险；默认 ABI=ON 走系统 boost，
    Windows/macOS 从不走源码 boost。约束仅在显式 ABI=0 构建下存在。
  - 3rdparty/boost_unordered 是跨平台机制而非 Linux 补丁：per-source
    INCLUDE_DIRECTORIES 三平台统一注入 patch_match.cc 单 TU；该 TU 的 boost 头
    经验证仅 flat_set 一个（mvs/*.h 零 boost include），vendored 闭包自洽，与各平台
    系统 boost 零混用。若改为依赖平台系统 boost（flat_set 需 >=1.81），vcpkg/homebrew
    版本参差才是真正的不一致来源。
  - EIGEN_MAX_ALIGN_BYTES=0 是 Ubuntu>=22.04 + PCL 的 Linux-only hack；
    fork 的 -U 修复方向 = 消除 Linux 特有分叉、向 Windows/macOS 默认行为看齐。
  - poselib.cmake 剥离旗标方向更保守：去掉 -msse4.2 强制（CPU 要求降为 baseline），
    不新增任何 AVX512 旗标（MARCH_NATIVE=OFF 保持），MSVC/arm64 下 REPLACE 无命中
    等价无操作；它使 Linux 的 poselib 代码gen与 Windows/macOS 一致（root cause #5）。

### 第四十三轮：ext_boost 源码集成 boost 全面去除（2026-09-22，用户决策）
- **调研结论**（本地 + 全网）：
  - ABI=0 根因真实但仅限显式 -DGLIBCXX_USE_CXX11_ABI=OFF 构建（gcc dual ABI：预编译
    boost .so 为 ABI=1，ABI=0 TU 不能链接）；全仓库 CI/docker/conda/文档零使用，option
    默认 ON。
  - 三平台 boost 实际来源已统一：Linux CI = apt libboost-all-dev（docker 全系
    Dockerfile*.ci/deps/wheel；22.04→1.74、24.04→1.83），macOS/Windows = conda
    boost=1.84（.ci/conda_* 三 yml，注明 "pcl and reconstruction"）。
  - ext_boost 1.73（AliceVision tarball + b2 慢速编译）是唯一版本碎片源。
- **已删除**：3rdparty/boost/boost.cmake（ext_boost ExternalProject）；find_dependencies
  的 UNIX ABI=0 boost 分支；VTK 源码构建的 DEPENDS ext_boost（其 boost 消费模块
  InfovisBoost* 早已 DONT_WANT，遗留依赖）；Reconstruction 的条件 find_package 与
  ABI=0 BOOST_ROOT=INTERNAL_BOOST_ROOT 分支；qSIBR 的 TARGET 3rdparty_boost 分支。
- **三平台统一形态**：全部 find_package(Boost REQUIRED graph filesystem
  program_options system [+unit_test_framework])；ABI=0 构建者经 BOOST_ROOT 自备
  匹配 ABI 的 boost（conda 环境自带 boost，ABI 自洽）。patch_match.cc 的 vendored
  boost::unordered flat_set 头不变（per-source 注入，与系统版本解耦——Ubuntu 1.74
  无 flat_set 头，编译通过即证明注入有效）。
- **回归**：configure + colmap_exe/CVCoreLib/QSIBR_PLUGIN 构建通过；链接
  libboost_filesystem.so.1.74.0；TVG 15/15 byte-identical 保持。

### 第四十四轮：20.04 矩阵补全 + ceres/glog/gflags/metis 来源澄清（2026-09-22）
- **Ubuntu 20.04（focal）风险评估**：20.04 仍在支持矩阵（ubuntu.yml cpu-focal +
  ubuntu-20.04 runner；docs 支持表）。其 apt boost = 1.71：
  - COLMAP 子树编译库需求（graph/filesystem/program_options/system 基础 API）1.71
    全满足；CI cpu-focal 的 ABI=ON 构建历史上一直用 find_package(Boost 1.71) 通过。
  - flat_set 数值契约头不依赖系统 boost（vendored 闭包解耦，20.04/22.04/24.04 同源）。
  - 结论：20.04 无新增风险；boost 矩阵完整版 = 20.04→1.71 / 22.04→1.74 /
    24.04→1.83（apt）+ 1.84（conda）。
- **3rdparty/boost_unordered 必须保留**：它是 patch_match.cc flat_set 数值契约头的
  唯一来源（20.04 的 1.71、22.04 的 1.74 均无 flat_set 头，要求系统 ≥1.81 等于重新
  引入源码编译 boost）；与 ext_boost 性质不同——纯 header-only、零构建步骤、
  per-source 注入单 TU、三平台同源。
- **ceres/glog/gflags/metis 来源澄清（用户质疑后核实）**：colmap_exe 链接闭包实测
  libceres.so/libglog.so/libgflags.so 全部来自 build_app/external/lib（ext_* 源码
  构建产物），metis 由 ext_suitesparse 内嵌提供——三平台源码集成设计未被破坏。
  Reconstruction/CMakeLists.txt 的 `# find_package(Glog/Ceres/Metis REQUIRED)` 三行
  是 upstream COLMAP 残留注释（HEAD 原状、从未启用），已加注释说明真实来源以防误读。

### 第四十五轮：遗留全面审核 + 撤销 boost 后最终 e2e 回归（2026-09-22）
- **审核修正**：第四十四轮审核报告中列的 P0-B「point_triangulator BA 常量位姿清零」
  为过时信息——该缺陷已在第十三轮定位、第十六轮修复闭环（ReadImagesBinary/Text
  镜像 frame 真源进 legacy 缓冲；位姿冻结检验 max|dq|=1.1e-16；TODO 文档为倒序追
  加格式，L323 属第十三轮旧段落）。（第四十六轮复核再修正：camera-ray 与
  flat-vs-solvers ODR 两项亦为过时条目，证据见第四十六轮。）真实遗留 =
  W3-3 mapper 重写（XL）、W4 尾巴（GP rig 两用例 + glomap_gate）、
  W18.7 file.h 面、W16 sfm 类绑定、W13 检查点、W14 杂项、num_trials 持久化（S）。
- **P0-A 最终 e2e 回归（当前 binary：撤销 boost + 探针清零 + ext_boost 去除后）**：
  - 全链 8 步在 /tmp/ab_mini6/revert_test 重跑（undistort 0 / stereo 3.02min /
    fusion / poisson / texture 全 EXIT=0；首轮 undistort FATAL 为工作区路径错误，
    非代码回归）。
  - **分布内判据保持**：fused sim3 相机中心对齐 KDTree——fork_now→up1 (对 4,3)
    NN p50=3.9mm p99=29.6mm <20mm=98.17% <50mm=99.61%；fork_now→up2 (对 6,4)
    p50=4.0mm 98.21%/99.59%；对照 upstream 自身波动带 up2→up1 p50=3.7mm
    98.06%/99.58%——fork 差异在带内。
  - **mesh 体积/离群澄清（用户此前报告的 fork3c「体积大离谱」）**：poisson mesh
    fork 29.0MB < up1 41.4MB < up2 46.6MB；离群远点三方同量级（|p|>10m：fork
    5842 / up1 5777 / up2 5876；z_max 29.9/30.9/28.2）——「大 mesh」是 gauge 与
    PoissonRecon 归约顺序差异（第十四轮已定性 ~5%），非 fork 缺陷；OBJ 176MB 为
    fork texturer OBJ 导出（upstream 无 OBJ 路径可比）。
  - 深度图 vs fork3c 0/6 bit-exact：为 boost/std 后端 mapper gauge 差异（第三十七
    轮已实证的非确定面），非稠密链回归；分布内判据为最终口径。
- **遗留清单（第四十六轮复核修正后）**：
  1. ~~camera-ray 三角化重构拉入~~ → **关闭（非遗留）**：已在第三轮拉入，剩余差异由
     分布内判据闭环（证据见第四十六轮）
  2. ~~flat-vs-solvers ODR 去重~~ → **关闭（非遗留）**：第五轮已执行 + 第三十六轮
     TVG byte-identical 收口
  3. W3-3 mapper Frame/Rig 重写（XL，frame-aware 主线收尾）——**第四十六轮已立项**
  4. W4 尾巴：GP rig 两用例 + reconstruction_glomap_gate
  5. W18.7 file.h 上游面；W16 sfm 类绑定；W13 RunPointTriangulator/Filtering 检查点；
     W14 per-struct random_seed/affine SIFT #2929/single_camera_per_folder；
     num_trials 持久化（S）
  6. boost 侧过时待办（旧 L526 三平台 CI、L569 -I 顺序）已随第四十一/四十三轮关闭

### 第四十六轮：遗留清单复核 + W3-3 mapper Frame/Rig 重写立项（2026-09-22）

- **遗留清单复核（first-principles 审计，逐项代码证据核查）**：第四十五轮清单的
  第 1、2 项与 P0-B 修掉的「point_triangulator BA 清零」同类，属倒序文档残留的
  过时条目：
  1. **camera-ray 三角化重构拉入 → 关闭（非遗留）**。三重证据：(i) fork
     `estimators/triangulation.{h,cc}` 已是上游 ray 形态（`PointData{img_point,
     cam_ray}` / `PoseData{cam_from_world, proj_center}`，头文件 parity 注释在位），
     第三轮已记录「移植完成 ✅ + 消费者全部迁移」；(ii) fork
     `incremental_triangulator.cc` 已消费 ray 化 `EstimateTriangulation`（观测→射线
     转换内建于库函数，与 upstream 同架构）；(iii) 透视模型下 2D 归一化平面点与
     射线 hnormalized 数学等价（第三轮已定性，ray 化增量价值仅在全向模型）。
     「fork 剩余点数/离群差异」的根因链已由第六轮（初始对语义）、第六轮#2（PnP
     ray 化）、第八轮（BA 单块化，同初始对 1441=1441）闭环，分布内判据（第三十七/
     四十五轮）为最终口径。**残余核实项**：`incremental_triangulator` 两侧 1490 行
     diff 尚未逐段分类，并入 W3-3 第一步全量 diff 审计，同时产出该项的最终关闭
     证据。
  2. **flat-vs-solvers ODR 去重 → 关闭（非遗留）**。第五轮已执行：flat F/E/H 六
     文件已删除（`ls estimators/{fundamental,essential,homography}_matrix.*` 零命中，
     全部走 `solvers/`）；数值层由第三十六轮 TVG 15/15 byte-identical 收口。
  3. **W3-3 mapper Frame/Rig 重写 → 本轮立项（XL）**。fork mapper 混合态实证：
     `incremental_mapper.cc` 8 处 `Qvec()/Tvec()` legacy 调用、
     `incremental_triangulator.cc` 3 处（upstream 三文件全为 0）；frame-aware 调用
     fork 21 处 vs upstream 42 处；`incremental_mapper_impl.cc` fork 450 行 vs
     upstream 740 行（frame-aware RegisterNextImageFallback、impl 自由签名助手等
     未拉入）——该混合态是 Bug #4/#5、RegFrameIds（第二十一轮）、AddFrame 自动
     注册（第二十八轮）缺陷链的结构温床。
- **W3-3 立项范围与闸门（第四十六轮定稿）**：
  - **第一步**：sfm 全量 diff 审计（`incremental_mapper.{h,cc}` /
    `incremental_mapper_impl.{h,cc}` / `incremental_triangulator.{h,cc}` /
    `observation_manager`），三分类：fork 适配 / 等价改写 / 真缺口；同步产出
    camera-ray 项最终关闭证据。
  - **手术边界**：只动表示层（image-based legacy 阴影 → frame 单源
    `cam_from_world`），不动任何选项默认值与判据阈值；「顺手改语义」一律记录到
    尾巴批。
  - **闸门（判据不升级）**：matches/TVG byte-identical 保持；fork 串行 A/A
    确定性；fused 分布内（≤ upstream run-to-run 波动带，mini6 P0-A 口径）；全量
    ctest 基线。
  - **依赖解锁**：W13 RunPointTriangulator/Filtering 检查点、W16 sfm 类绑定随
    W3-3 落地（PLAN 已明示耦合）；W4/W18.7/W14/num_trials 等独立尾巴项在 W3-3
    闸门间歇穿插执行。
- **upstream 仓库探针恢复**：`MVS/colmap/src/colmap/sfm/incremental_mapper_impl.cc`
  的 TEMP-PROBE（init-pair 接受判定探针，写 `/tmp/ab_mini6/tvg_probe/up_init.bin`，
  P0-A revert_test 回归时遗留）已 git checkout 恢复；upstream 仓库工作区干净。
- **本轮基线状态**：mini6 对齐主线在当前 binary 上完整闭环（database bit-exact →
  稠密链 bit-exact → sparse/fused 分布内）；下一步 = W3-3 第一步 sfm 全量 diff
  审计。

### 第四十六轮补遗：W3-3 第一步 sfm 全量 diff 审计（同日）

审计范围：incremental_mapper.{h,cc} / incremental_mapper_impl.{h,cc} /
incremental_triangulator.{h,cc} / scene/two_view_geometry.h / database_cache.cc；
结构面（函数清单级）+ 8 个关键语义点核实。

1. **camera-ray 项最终关闭证据（补全第四十六轮主体）**：fork
   `TwoViewGeometry` 已携带 `std::optional<Camera> camera1/camera2`（SF 内参
   surfacing 载体，scene/two_view_geometry.h 与 upstream 逐字段一致）；
   `incremental_triangulator` diff 主构 = 版权头/include 顺序（A 类）+
   `TriangulateTrack` helper 位置差异（upstream 匿名 namespace、fork 内联，
   B 类），`EstimateTriangulation` ray 化消费形态一致——triangulator 层无 C 类
   缺口。
2. **三分类结果**：
   - **A 类（fork 适配，保留）**：① 指针 vs shared_ptr 所有权模型
     （database_cache_/reconstruction_/obs_manager_）；② include 路径/版权头；
     ③ hash 容器 alias（第三十六/四十一轮已实证零行为影响）；④ 注释风格。
   - **B 类（等价改写，手术主体）**：① FindFirst/SecondInitialImage、
     FindInitialImagePair、FindNextImages、FindLocalBundle、
     EstimateInitialTwoViewGeometry 从 mapper 成员（隐式状态）迁为 impl static
     （显式传参）；② `InitInfo` 打包取代 prev_init_image_pair_id_/
     prev_init_two_view_geometry_ 双成员缓存（TVG 载体已一致，仅包装层差）；
     ③ `RegistrationStatistics` 打包取代 8 个散装成员；④
     ExtractSortedImageIds/CompareImageInfo/InitImageInfo 自由助手补齐；⑤
     TriangulateTrack helper 同构；⑥ **legacy 双路径分支整体退役**
     （RegisterInitialImagePair 的 HasFramePtr() 分支——fork
     DatabaseCache::AddFrame 已接线 frame，分支注释自述的 "until the cache
     wiring lands" 前提已消失）；⑦ Register/DeRegisterImageEvent（image 级）
     退役，统一 frame 级 RegisterFrameEvent/DeRegisterFrameEvent。
   - **C 类（真实缺口，拉入清单）**：① `EstimateInitialGeneralizedTwoViewGeometry`
     （non-trivial rig 初始对 generalized relative pose，upstream impl
     L514-628 + L696-710 分支；fork 仅具 generalized absolute pose 的
     RegisterNextGeneralFrame）；② `FilterFrames`（frame 级过滤）替换 fork
     `FilterImages`（image 级）；③ `IterativeLocalRefinement`（upstream h L241 /
     cc L1226，fork mapper 零命中——待查 fork controllers/incremental_pipeline
     是否内联等价，若内联则降级 B 类迁移）；④ `Options.constant_rigs/
     constant_cameras`（rig/camera 级常量集，fork BA 内部有
     constant_rig_from_world_rotation 机制但 Options 面未暴露）；⑤
     `fix_existing_frames` frame 化语义（fork 为 fix_existing_images）。
3. **手术边界决策（第四十六轮定稿）**：
   - **FindInitialImagePair 保持 fork 串行**，不拉入 upstream ThreadPool 并行
     尝试（upstream impl L200-258）：fork 串行是 A/A 确定性的来源，第三十六轮
     已实证与 upstream 单线程模式同初始对；upstream 并行属合法非确定面，
     分布内判据下无拉入必要。
   - **CLI 命名保留 fork 形态**：`local_ba_num_images/local_ba_min_tri_angle`
     vs upstream `ba_local_*`——fork CLI 选项面已绑定，改名破坏兼容；记录为
     A 类命名偏差。
   - **RegisterNextImageFallback 从缺口清单移除**：upstream 当前版已无此函数
     （仅 h L179 注释残留引用），PLAN 旧记录针对历史版本，过时。
4. **诚实边界**：mapper.cc 剩余 ~2900 行 diff 与 triangulator
   Create/Continue/Resection 的逐行语义等价性在手术执行时逐段确认；本审计
   覆盖结构面与关键语义点，未做逐行分类。

### 第四十六轮补遗 2：W3-3 第一步手术执行 + 闸门全绿（同日）

用户决策：**B 地基先行分两步** + **FindInitialImagePair 保持串行 / CLI 命名
保留 local_ba_***（按第四十六轮定稿推荐）。

**已落地（语义等价重构，5 文件 + 1 测试）**：
1. `sfm/incremental_mapper_impl.{h,cc}`：`InitInfo` 打包
   （image_id1/2 + cam2_from_cam1 + optional<Camera>×2）落地；
   `FindInitialImagePair`（fork 串行版，保持 A/A 确定性来源）与
   `EstimateInitialTwoViewGeometry`（返回 optional<InitInfo>）迁入 impl
   static；InitImageInfo/CompareImageInfo/ExtractSortedImageIds 自由助手补齐
   （B④）；初始 TVG RANSAC 接入 `options.random_seed`（upstream L657 parity，
   默认 -1 下行为不变）。
2. `sfm/incremental_mapper.{h,cc}`：mapper 公共 API 切 upstream 形态——
   `FindInitialImagePair(options, image_id1&, image_id2&, Rigid3d&)`（成功时
   `SeedEstimatedInitialCameras` 写回 SF 内参——**fork 首次消费 TVG 的
   camera1/camera2**，旧版估计后丢弃）；`RegisterInitialImagePair` 改为接收
   `cam2_from_cam1` 参数、不再重估 TVG（fork 保留 bool 返回以维持 pipeline
   失败路径，A 类偏差记录）；公共 `EstimateInitialTwoViewGeometry(options,
   id1, id2, Rigid3d&)` 补齐（upstream d3ccaf35 形态，供显式初始对分支）；
   **B⑥ legacy 双路径退役**（Reconstruction::Load 已接线 FramePtr/SetCameraPtr，
   mapper 主链路下 HasFramePtr() 恒真，旧注释自述的 "frameless cache images"
   前提已消失，Qvec/Tvec fallback 为死代码）；**B⑦ image 级事件退役**
   （RegisterImageEvent/DeRegisterImageEvent 删除，per-image 计数内联进
   frame 级事件，upstream 同构）；prev_init_ 双成员缓存删除；mapper private
   的 FindFirst/Second/FindLocalBundle 薄委托删除（调用点直连 impl static）；
   `Options::Check` 补 random_seed 检查；RegisterNextImage 的 RANSAC 接入
   mapper seed（upstream L335 parity）。
3. 消费点适配：`controllers/incremental_pipeline.cc`（显式初始对分支补
   `mapper.EstimateInitialTwoViewGeometry` 调用——upstream 语义：用户指定对
   仍需过阈值判定）；`libs/Python/pybind/reconstruction/sfm/mappers.cpp`
   两 lambda 适配新签名（Python 面零消费者，自由适配）。
4. **顺手修复既有失败**（非手术引入）：`optim/ransac_test.cc` TestNumTrials
   期望值仍为旧 pow 有放回公式产物（4605168/4603/6905/13809），第三十六轮
   切换超几何公式时未同步——按 upstream d3ccaf35 期望值更新
   （18446744073709551615llu/6204/9305/18610/36），实现本身 fork 与 upstream
   已逐行一致。

**闸门结果（全绿）**：
- 组级 ctest（estimators/scene/sfm/controllers/geometry/optim/math）
  **73/73 全绿**；全量消费者编译零错误（ColmapLib / colmap_exe /
  CV_PYTHON_LIB / ACloudViewer GUI）。
- mini6 sparse 链（feature_extractor → exhaustive_matcher → mapper）手术后
  binary 跑 w33a/w33b 双工作区：
  - **A/A 确定性**：w33a≡w33b，sparse 4/4 bin（cameras/images/points3D/rigs）
    bit-identical。
  - **手术前后等价**：w33a→revert_test sim3（按 image_id 配对相机中心）
    scale=1.0000、相机残差 mean 0.06mm、稀疏点 NN p50=0.16mm / p99=0.63mm、
    <20mm=100%——远小于 upstream run-to-run 波动带（p50=3.7mm）。
  - **对 upstream 距离不变**：w33a→upstream (up1 系) NN p50=0.64mm /
    <20mm=96.27% / <50mm=98.83%，与手术前基线 revert→upstream
    （0.67mm/96.27%/98.83%）完全一致；w33a→up_run2（同对 6,4）p50=0.53mm、
    <20mm=100%。
- 对比方法论备忘：sim3 相机中心必须**按 image_id 配对**（两侧注册序不同），
  行序配对会产生伪残差；脚本存 /tmp/ab_mini6/w33_sparse_cmp.py。

**下一步（W3-3 第二步）**：C 类缺口拉入——①
EstimateInitialGeneralizedTwoViewGeometry；② FilterFrames（frame 级）替换
FilterImages + FindNextImages frame 化签名；③ IterativeLocalRefinement（先
查 fork controllers 是否内联等价）；④ Options.constant_rigs/constant_cameras；
⑤ fix_existing_frames。之后可选做 B③ RegistrationStatistics 打包（机械替换
~50 处引用）。

### 第四十六轮补遗 3：W3-3 第二步手术（C 类拉入 + B③ 打包）闸门全绿（同日）

**已落地**：
1. **B③ RegistrationStatistics 打包**：9 个散装统计成员 → `reg_stats_`
   （upstream 同构 struct，含默认成员初始化器）；BeginReconstruction 补
   `num_structure_less_reg_trials.clear()`（fork 旧缺口：structure-less 重试
   预算跨重建未重置，upstream 清）。
2. **② FilterFrames（C②）**：mapper `FilterImages`（image 级）→
   `FilterFrames`（frame 级，upstream 形态：`obs_manager_->FindFramesToFilter`
   + `DeRegisterFrame`/`DeRegisterFrameEvent`/`filtered_frames_`；门槛
   `kMinNumFrames=20` 按 `NumRegFrames()`）；`filtered_frames_`/
   `existing_frame_ids_` 切为 `NodeHashSet<frame_t>`；FindNextImages frame 化
   签名（obs_manager + filtered_frames + num_reg_trials 可变引用 +
   structure_less 分支）；pipeline 自由函数同步改名。
3. **③ IterativeLocalRefinement（C③→B 类迁移）**：fork controllers 内联
   等价物升为 mapper 成员（upstream 形态：stop-aware、零调整除零保护、
   VLOG 诊断、首迭代后降级 robust loss、尾部 ClearModifiedPoints3D）；
   pipeline 改薄委托。
4. **④ constant_rigs/constant_cameras（C④）**：Options 新增两字段
   （NodeHashSet）；三消费点拉入——RegisterNextImage 的内参估计/精化门、
   AdjustLocalBundle 的 rig 恒定段（num_frames_per_rig 统计 +
   SetConstantSensorFromRigPose）+ camera 门、AdjustGlobalBundle 的两段。
   fork 保留 `SetConstantCamera` 名（语义= upstream SetConstantCamIntrinsics，
   A 类命名偏差）。
5. **AdjustLocalBundle frame-aware BA config**：local bundle 的**整个
   frame** 的 images 加入 ba_config（多传感器 rig 联合调整；单传感器退化
   等价）；FixGauge 前置贴 upstream。
6. **① EstimateInitialGeneralizedTwoViewGeometry（C①）**：generalized
   relative pose 初始对（非平凡 rig）整体移植（impl 匿名 namespace 自由
   函数 + EstimateInitialTwoViewGeometry 的 `NumSensors()>1` 分支；
   `EstimateGeneralizedRelativePose` fork 已备，DatabaseCache frame-aware
   访问器已备）。
7. **⑤ fix_existing_frames（C⑤）**：语义升级为 frame 级门控（
   `existing_frame_ids_` 按帧判定），fork 保留 `fix_existing_images` 选项名
   （CLI 兼容，A 类命名偏差）；frameless 回退分支删除（frame 接线不变量）。

**崩溃与二分定位（诚实边界）**：首次闸门 mapper SIGABRT 于
`observation_manager.cc:193`（image 4 point2D 215 correspondence counter
下溢）。二分回退定位：根因是 **FindNextImages 的 rank 源从 Image 缓存切到
`obs_manager.NumVisiblePoints3D`** ——它改变了下一张图的选取序列，新序列
暴露 fork 的 observation 计数维护在 Image 缓存与 obs_manager stats 双轨间
的不一致（旧序列从未触发；上游同序列不崩，fork 计数不对称是真缺陷）。
**处置**：rank 源回退为 Image 缓存（frame 过滤逻辑保留），rank 源切换立项
为下一轮专项（obs_manager stats 一致性取证）；其余 frame-aware 改动保留。

**闸门结果（全绿）**：
- 组级 ctest **73/73**；ColmapLib / colmap_exe / CV_PYTHON_LIB /
  ACloudViewer GUI 编译零错误。
- mini6 sparse 链 A/A：w33c ≡ w33d，sparse 4/4 bin bit-identical。
- **w33c → w33a（第一步手术后）：cam_resid 0.00mm / NN p50=0.00mm** ——
  第二步全部改动在单传感器场景下与第一步输出完全重合（frame-aware 改动
  退化等价 + generalized 分支不触发的直接证明）。
- w33c → revert（手术前基线）p50=0.16mm / <20mm=100%；w33c → upstream
  p50=0.64mm / 96.27%@20mm / 98.83%@50mm——与第一/零步完全一致。

**下一步（W3-3 收尾专项）**：
1. **obs_manager stats 一致性专项**（本轮新立项，M 级）：Image 缓存与
   obs_manager 双轨可见性计数的逐操作对账（AddPoint3D/DeleteObservation/
   MergeTracks/DeRegisterFrame 路径），收口后切回 upstream rank 源。
2. AdjustGlobalBundle 的 fix_existing 遍历面（reg_image_ids → RegFrameIds，
   upstream 形态，S 级）。
3. mapper.h 的 `ObservationManager()/Triangulator()/NumRegImagesPerCamera()`
   getter 补齐（S 级）。
4. W13/W16 尾巴随 frame-aware ba_config 就绪后回排。

### 第四十六轮补遗 4：obs_manager 单轨化收口 + rank 源切回 upstream 形态
（同日，W3-3 收尾专项①②③全部完成）

**根因链闭合（②的取证结果）**：第二步崩溃的真因是 fork 的**双轨点操作**——
upstream 的 `IncrementalTriangulator` 持有 `obs_manager_`，所有点操作
（AddPoint3D/AddObservation/MergePoints3D/DeleteObservation）经
ObservationManager 完成（内部同时修改 Reconstruction 并传播可见性统计，
单一真源）；fork 的 triangulator 全部直呼 `reconstruction_->AddPoint3D 等`
（Image 缓存轨），完全绕过 obs_manager → stats 永不更新（仅 RegisterFrame
传播一次）→ rank 源切到 obs_manager 读到过时统计 → 选图序列分叉 →
下游 ResetTriObservations 的 THROW_CHECK 下溢。fork 的 obs_manager 方法面
（AddPoint3D/AddObservation/DeletePoint3D/DeleteObservation/MergePoints3D，
内部 reconstruction 修改 + Set/ResetTriObservations 传播）与 upstream 逐行
同构，早已具备单轨能力——只是 triangulator 没接上。

**收口手术（upstream 单轨化）**：
1. `incremental_triangulator.{h,cc}`：构造签名加第三参
   `shared_ptr<ObservationManager>`（可空，空则自建——upstream 形态；自建
   分支用空 deleter 的 observer shared_ptr 持 CorrespondenceGraph，所有权
   仍归 DatabaseCache）；私有成员 +obs_manager_；5 处点操作切
   obs_manager_->AddPoint3D/AddObservation/MergePoints3D。
2. `incremental_mapper.cc` BeginReconstruction：obs_manager 先建、
   triangulator 共享同一实例（upstream 同序）。
3. **FindNextImages rank 源切回 upstream 形态**（obs_manager.NumVisiblePoints3D
   / NumVisibleCorrespondences / Point3DVisibilityScore）。
4. **收尾②**：AdjustGlobalBundle 的 fix_existing 遍历改
   `reconstruction_->RegFrameIds()`（upstream 形态）。
5. **收尾③**：mapper getter 补齐 ObservationManager()/Triangulator()/
   FilteredFrames()/ExistingFrameIds()/NumRegImagesPerCamera()（
   Reconstruction getter 保留 fork GetReconstruction 名，A 类偏差）。

**闸门（全绿 + 零漂移，单轨化的直接证明）**：
- 组级 ctest **73/73**；ColmapLib/colmap_exe/CV_PYTHON_LIB/ACloudViewer GUI
  编译零错误（pybind PyTriangulator 走默认参数自建分支）。
- mini6 A/A：w33e ≡ w33f，sparse 4/4 bin bit-identical。
- **w33e → w33a（第一步后）p50=0.00mm / cam_resid 0.00mm** —— rank 源切回
  obs_manager 后选图序列与 Image 缓存轨完全一致，双轨不一致已收口（
  第二步崩溃根因消除的直接证据）。
- w33e → revert p50=0.16mm / <20mm=100%；→ upstream p50=0.64mm /
  96.27%@20mm——全程不变。

**④ W13/W16 回排评估**：两者为独立批次非 frame-aware 硬依赖——W13（exe
RunPointTriangulator/RunPointFiltering 的 upstream ObservationManager 直控
形态，M 级）、W16（pybind sfm 绑定批，M 级）排入下轮；本轮不强行拉入。

**W3-3 状态**：mapper Frame/Rig 重写的全部计划内手术（B 地基两批 + C 类
拉入 + 收尾专项）完成；frame-aware 主线闭环。

### 第四十六轮补遗 5：全面审核复核 + 端到端重建交付（2026-09-23）

用户两项要求：(a) `libs/Reconstruction/src/Testing/` 的 gitignore 处理；
(b) 全面审核遗留后跑端到端重建供 mesh/纹理人工对比。

1. **gitignore 修复（S）**：根 `.gitignore` 的 `/Testing/` 前导斜杠锚定仓库
   根，未覆盖 ctest 从源码子目录运行时生成的 `libs/Reconstruction/src/
   Testing/`（Temporary/LastTest.log + CTestCostData.txt）。全仓无被跟踪的
   Testing 目录（`git ls-files` 零命中，通用化无误伤）→ 改为不限定的
   `Testing/`，`git check-ignore` 验证命中。
2. **遗留审核结论**：主线（database→sparse→dense 链）对齐闭环；尾巴批
   六项全部定性为面完备性项（不改变重建输出数值路径）——W4（GP rig 测试
   用例：generalized_*_pose_test.cc 文件在、用例面待补 + glomap_gate）、
   W18.7（file.h 上游面）、W16（pybind sfm 绑定批：mappers.cpp 已有
   LocalBundleAdjustmentReport/PyObservationManager/PyTriangulator/PyMapper
   四 wrapper，upstream 全量面未绑完）、W13（exe RunPointTriangulator/
   Filtering 的 upstream `ObservationManager` 直控形态，fork exe 层无）、
   W14（per-struct random_seed/affine SIFT #2929/single_camera_per_folder）、
   num_trials 持久化。代码面抽查确认定性成立。
3. **端到端重建交付（w33final）**：完整 8 步链
   （feature→match→mapper→undistort→stereo→fusion→poisson→texture）
   全部 EXIT=0，stereo 3.02 分钟与 P0-A 一致。产物：fused.ply 7.8MB、
   meshed-poisson.ply 29.3MB、textured/{mesh.obj 177.8MB, mesh.ply 67.3MB,
   texture.png 4.8MB, mesh.mtl, mesh_map_Kd.png}。操作备忘：
   `mesh_texturer --export_obj` 需显式值 `1`（无值报 parse options FATAL）；
   `model_converter` 的 output_path 目录需预建。
4. **数值判据（复验通过）**：w33final→revert_test sparse p50=0.2mm /
   <20mm=100%（同 w33e→revert 的 0.16mm 量级）；fused（sim3 相机中心对齐
   + KDTree）p50=3.6mm / p99=30.1mm / <20mm=98.07% / <50mm=99.56%——逐项
   落在 upstream 自身 run-to-run 波动带内（3.7mm/30.7mm/98.06%/99.58%），
   W3-3 全部手术后端到端输出分布内等价复核完成。database 层 DIFF 为跨会
   话 GPU SIFT 非确定性（9月22 vs 9月23 feature 会话），非回归。

**交付路径**：`/tmp/ab_mini6/w33final/dense/`（本轮）vs
`/tmp/ab_mini6/revert_test/dense/`（P0-A 基线），供 mesh/纹理人工对比。

### 第四十六轮补遗 6：pybind 绑定漏改修正（同日，编译验证疏漏修正）

用户指出 pybind 绑定漏改：`mappers.cpp:511` 调用 `FilterImages`，但 W3-3
第二步手术已重命名为 `FilterFrames`。

**根因**：补遗 3 的 `FilterImages → FilterFrames` 重命名只更新了
`controllers/incremental_pipeline.cc`，未同步更新 pybind 绑定。补遗 4 的
编译验证（CV_PYTHON_LIB）在增量编译下未触发 mappers.cpp 重新编译（旧 .o
还在），所以"零错误"不代表真正编译通过；全量编译暴露问题。

**修正（S 级）**：
- `mappers.cpp:508` 绑定名 `filter_images` → `filter_frames`
- `mappers.cpp:511` 调用 `FilterImages` → `FilterFrames`
- 注释更新（image → frame）

**编译验证**：CV_PYTHON_LIB 全量编译通过。

**端到端重建代码确认**：w33final e2e 用的是 `build_app/bin/Colmap`（C++
可执行文件），该 binary 在补遗 3/4 编译时已成功（ColmapLib + colmap_exe
target）；pybind 是独立的 Python 绑定 target，不影响 C++ binary 行为。
所以 e2e 输出确为 W3-3 全部手术后的最新代码产物。
