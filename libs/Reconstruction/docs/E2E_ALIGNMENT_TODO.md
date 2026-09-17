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
