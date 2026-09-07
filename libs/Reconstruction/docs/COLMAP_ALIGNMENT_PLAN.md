# COLMAP 对齐与集成完整方案（Alignment & Integration Plan）

> 状态：已评审定稿（含产品决策 D1–D3）。执行状态跟踪见
> `libs/Reconstruction/colmap_alignment_manifest.json`（机器可读）与
> `libs/Reconstruction/docs/COLMAP_ALIGNMENT.md`（能力状态摘要）。
> 本文是排期与实施依据；两者完成后须回写 manifest，不在此文档维护逐项状态。

## 1. 目的、范围与决策记录

### 1.1 目的

将 `libs/Reconstruction/`（内联 COLMAP fork）的核心重建能力对齐到上游
`/home/asher/develop/code/github/MVS/colmap`（HEAD `dbb41680`，COLMAP
4.3.0.dev0，2026-09-04，与 manifest 的 `upstream_revision_checked` 一致），
并把上游 `pycolmap` Python 绑定融合进 `cloudViewer.reconstruction` 模块。

### 1.2 范围外（维持现状，随 manifest 记录）

| 项 | 处置 | 依据 |
|---|---|---|
| ONNX Runtime（ALIKED/LightGlue 路径） | 不引入，ggml LoMa/DeDoDe/DaD 路线保持 | manifest `loma_lightglue: partial`（CUDA DeDoDe-G token 精度门未闭合） |
| HIP PatchMatch | deferred，无 ROCm 硬件即不进发布矩阵 | manifest `hip_patchmatch: deferred` |
| FreeImage→OpenImageIO | 继续按平台门推进（属依赖替换，按对齐口径不计入） | manifest `freeimage_to_openimageio: partial` |
| DA3 系自研扩展（da3_*、photometric_* 选项等） | 全部保留，不受对齐影响 | 本地产品能力 |
| 目录结构差异（本地 `base/` 扁平 vs 上游分层） | 不做整体重构 | 见 2.3 策略 C |

### 1.3 决策记录（Decision Log）

| # | 决策 | 日期 | 内容 |
|---|---|---|---|
| D1 | mesh 纹理双路径 | 2026-09-05 | 保留本地 `image_texturer`（标 alternative），同时集成上游 `mesh_texturer` CLI 与工作流；**用户默认选择上游 `mesh_texturer` 方式**；两者共享同一 `MeshTextureMapping` 内核 |
| D2 | pycolmap 融合集成 | 2026-09-05 | 上游 pycolmap 绑定与既有 `cloudViewer.reconstruction` 模块（`libs/Python/pybind/reconstruction/`，包装 `src/pipelines/*` 函数级命令与 colmap 选项 struct）**融合**：保持单一命名空间与单一接口，新增上游类级 API、去除冗余包装；不另建 `cloudViewer.pycolmap` 平行包、不设 compat shim |
| D3 | 对齐落位策略 | 2026-09-04 | 策略 C：新增文件按上游目录落位，已有旧文件位置不动 |
| D4 | PBA 退役 | 2026-09-05 | 跟随 W2 立即退役（上游 c9729fa0 因 CUDA 12+ 已移除） |
| D5 | 依赖差异不作为对齐项 | 2026-09-04 | 依赖库/依赖方式/版本不计入；但改变求解算法集合的差异（如 PoseLib 求解器接入范围）计入 |
| D6 | 测试框架全面切换 googletest | 2026-09-06 | 上游 COLMAP 测试为 googletest 且本仓库其他模块同样使用；Reconstruction 全部 79 个测试由 Boost.Test 迁移为 gtest（`util/testing.h` 改为 gtest 包装、`COLMAP_ADD_TEST` 链接 `gtest_main`），此后上游测试文件可原样落地，不再做框架转换 |
| D7 | 执行进度快照 | 2026-09-06 | W1 ✅ / W2 ✅ / W5 ✅（PoseLib 升格硬依赖 ，与 solvers/poselib_utils 一并落地）/ W6 ✅（two_view_geometry 上游 1898 行重写版切换 + 消费者迁移 + parity 门）/ W9 partial（W3-1 对应图缓存已落地，下一步启用 synthetic）/ W3-1 ✅；W3-2a ✅（frame-aware 数据模型：Image frame_id/CameraPtr/FramePtr/CamFromWorld、Frame rig_ptr/SensorFromWorld、Rig NonRefSensors、Reconstruction 上游 Add* 装配 + WithTrivial 变体 + UpdatePoint3DErrors/RegFrameIds、Database pose_priors 持久化 + geometry/pose_prior + util/enum_utils 移植、OIIO ZLIB_ROOT 钉到 ext_zlib 1.3.1 修复本机 configure）；W3-2b mapper/BA frame-aware 与 W9 synthetic.cc 剩余 ~10 处符号适配未开始；W3-2/W4、W7–W8、W10–W16 未开始。全量构建 EXIT=0；ctest 73/77（失败均为 4 项预先存在的环境基线：polynomial、da3_depth_controller、sift GPU、gpu_mat Not Run） |
| D8 | 全量构建补验轮 | 2026-09-06 | 上轮的 ctest 73/77 是在部分测试目标链接失败的情况下对 stale 旧二进制测得的。全量重链暴露并修复：① fork `Image::DataId()` 实现错误（`sensor_t(CAMERA, ImageId()), 0` → 上游 `sensor_t(CAMERA, CameraId()), ImageId()`，image 1 恰好 camera==image id 掩盖缺陷）；② `Rig::NonRefSensors()` 引用签名与 fork qvec/tvec 存储不同构，改为按值物化；③ 补 `CalculateSquaredReprojectionError(Vector2d,Vector3d,Rigid3d,Camera)` 定义；④ `AddImageWithTrivialFrame(image)` 增加从 image legacy qvec/tvec 播种 frame 位姿；⑤ Crop/Merge 跨重建图像拷贝补 `ResetCameraPtr/ResetFramePtr` + trivial rig/frame 重建，Merge 恢复 camera-before-image 顺序；⑥ 六个 legacy 测试 fixture 迁移到 `AddCameraWithTrivialRig`+`AddImageWithTrivialFrame`。此后真实达成：全量构建 EXIT=0、ctest 73/77（仅 4 项环境基线）。**规则：ctest 结果只在全量构建 EXIT=0 之后报告** |

## 2. 对齐基线与比较方法

### 2.1 基线

- 上游：`/home/asher/develop/code/github/MVS/colmap` @ `dbb41680`（main，
  2026-09-04，仍为 4.3.0.dev0）。相对上一基线 `a395b826` 仅演进 1 个提交
  **#4673**（hash map 后端不再由构建机自动选择：STD 默认、BOOST 显式 opt-in、
  AUTO 警告；新增 `kHashMapBackend` 常量并写入 GetBuildInfo，
  `colmap-config.cmake` 预置后端使下游配置期报错而非静默重推导，pycolmap
  暴露 `__hash_map_backend__`）。该提交不触及 G1–G21 的任何锚点文件，缺口
  结论整体沿用；其后端固定语义转化为 W16 的一致性约束（W16 工作项 8 与
  §7 风险表）。
- 本地：`libs/Reconstruction/src`（182 头文件）+ `lib/` + `ColmapApp/`。
- 跟踪清单：`colmap_alignment_manifest.json` 为任务制清单（非完备覆盖清单），
  本方案的每一工作包完成后新增/更新 manifest 条目并附 parity gate。

### 2.2 比较口径（四层最小能力单元）

1. CLI 命令注册面（`exe/colmap.cc` 注册表 + `exe/*.h` Run 声明）；
2. Options struct 字段面（mapper / BA / 两视图 / SIFT / MVS / reader）；
3. 数据协议面（SQLite 表、迁移、IO 格式）；
4. 算法与求解器面（符号级 grep：缺失 / 死代码 / 已接线）。

### 2.3 落位策略 C（D3）

- 新增文件按上游目录名落位（`scene/`、`math/`、`geometry/`、`estimators/solvers/`、
  `pycolmap/`），include 适配为本地路径（`colmap/scene/x.h` → `scene/x.h` 或
  `base/x.h` 按本地既有映射）。
- 既有旧文件（`base/` 扁平布局）位置不动；重命名仅发生在命名冲突处（见 W15）。
- 收益：上游 `*_test.cc` 对新文件可近乎原样移植；旧文件差异保持已知、可控。

## 3. 缺口总清单（两轮扫描合并结论）

两轮证据级扫描（符号 grep + Options 字段 diff + 注册表比对 + 上游 git 交叉验证）
的合并结论如下（已随基线更新至 `dbb41680` 复核：Δ=1 基建提交 #4673，缺口
结论不变）；"W 列"为覆盖该缺口的工作包。

### 3.1 完全缺失（上游有、本地无实现或死代码）

| # | 缺口 | 关键上游锚点 | 本地证据 | W |
|---|---|---|---|---|
| G1 | GLomap 全局 SfM 栈（旋转平均 L1/L2、全局定位、视图图标定、pose_graph、motion-averaging 成本） | `sfm/global_mapper.h`、`estimators/rotation_averaging.*`、`global_positioning.*`、`view_graph_calibration.*`、`scene/pose_graph.*`、`cost_functions/motion_averaging.h`、`math/{connected_components,spanning_tree,union_find}` | 符号 grep 全零；本地 `controllers/global_mapper.h` 仅是多组件**增量**编排器 | W4 |
| G2 | GLomap CLI：`global_mapper` / `rotation_averager` / `view_graph_calibrator` | `exe/sfm.h` L80–86 | 本地注册表无 | W4 |
| G3 | 位姿先验栈：`pose_priors` 表、`geometry/pose_prior`、`cost_functions/pose_prior`、`gravity_refinement`、`pose_prior_mapper`、mapper `use_prior_position` 系选项 | `scene/database_sqlite.cc`、`geometry/pose_prior.h`、`estimators/gravity_refinement.h` | `PosePrior/pose_prior` grep 全零；本地 DB 无该表 | W7 |
| G4 | frame-aware 增量建图管线：`DatabaseCache`/对应图/mapper 的 Frame/Rig 感知、`RegisterNextImageFallback`、`incremental_mapper_impl`/`observation_manager` 拆分 | `sfm/incremental_mapper.h` L186–219、`sfm/incremental_mapper_impl.*`、`sfm/observation_manager.*` | `\b(rig|Rig|Frame|frame_t)\b` 在本地 mapper 与 cache **零命中**；DB 持久化层 Frame CRUD 已有，缺口在内存管线 | W3 |
| G5 | generalized pose 注册接线 | `estimators/generalized_pose.*`、mapper 消费 | 本地 `EstimateGeneralizedAbsolutePose` 除测试外零消费者（死代码） | W3/W5 |
| G6 | 两视图增量：DEGENSAC、`filter_stationary_matches`、`force_H_use`、`compute_relative_pose`、`relpose_shared_focal` | `estimators/fundamental_matrix_degensac.h`、`two_view_geometry.h`、`solvers/relpose_shared_focal.*` | 各符号 grep 零 | W6 |
| G7 | PoseLib 最小求解器统一接入（absolute/essential/generalized） | `estimators/solvers/poselib_utils.*` | 本地 PoseLib 仅用于 two_view 与 one-sided-focal 路径 | W5 |
| G8 | DB 版本迁移机制（`MakeDatabaseVersionNumber` + `Pre/PostMigrateTables`，零位姿/零 E-F-H→NULL 等） | `scene/database_sqlite.cc` | 本地仅写死 `PRAGMA user_version = 395`，无迁移 | W1 |
| G9 | 模型聚类与剪枝：`reconstruction_clustering`、`reconstruction_pruning`(`FindRedundantPoints3D`)、`model_clusterer` CLI、mapper `ba_global_ignore_redundant_points3D_min_coverage_gain` | `scene/reconstruction_{clustering,pruning}.*`、`sfm/incremental_mapper.h` L117 | 符号与选项 grep 零 | W8 |
| G10 | 描述符索引 `FeatureDescriptorIndex`（FAISS 可选） | `feature/index.*`、`controllers/matcher_cache.cc` | `3rdparty_faiss` 已链接但 src 无任何 faiss include（死链接） | W11 |
| G11 | CLI：`geometric_verifier` / `guided_geometric_verifier` / `rig_configurator` | `exe/feature.h`、`exe/database.h` | 本地无 | W11/W12 |
| G12 | BA 协方差 `EstimateBACovariance` | `estimators/covariance.h` | grep 零 | W14 |
| G13 | 合成场景生成 `synthetic` | `scene/synthetic.*` | grep 零 | W9 |
| G14 | 优雅退出（SIGINT → 取消并保存中间结果） | `exe/colmap.cc` `kSupportsGracefulShutdown`、`util/cancellation.h` | grep 零 | W13 |
| G15 | 逐项杂项：`as_rgb`、`force_covariant_extractor` + affine SIFT 方向修复 #2929、`single_camera_per_folder`/`camera_params`、per-struct `random_seed`、`ui/mesh_painter`、`util/timestamp` | 各对应文件 | 见 3.2 行为层 | W14 |
| G16 | pycolmap 类级绑定面：上游 `src/pycolmap/` 全树（Database/Reconstruction/Camera/Image/Point2D/Point3D/Rig/Frame/Rigid3d/Sim3d/estimators/optim/pipeline/mvs/retrieval 类与函数）。本地已有函数级 `cloudViewer.reconstruction` 模块（包装 `src/pipelines/*` 自由函数 + colmap 选项 struct），缺类级 API；集成方式为**融合去重**而非平行新增 | `src/pycolmap/`、`python/CMakeLists.txt`、本地 `libs/Python/pybind/reconstruction/` | W16 |

### 3.2 行为层差异（有实现但与上游不一致）

| # | 差异 | 上游 | 本地现状 | W |
|---|---|---|---|---|
| G17 | BA 选项面：缺 `refine_rig_from_world` / `refine_sensor_from_rig` / `constant_rig_from_world_rotation` / `min_track_length` / `backend` / `check_if_stopped` | `estimators/bundle_adjustment.h` L207/211 | 本地保留旧 `std::vector<CameraRig>` API + caspar_* 自有字段 | W2/W3 |
| G18 | mapper 选项：缺 `fix_existing_frames`（本地旧名 fix_existing_images）、`max_runtime_seconds`、`load_all_images`、`constant_rigs`/`constant_cameras`、`ba_local/global_backend`、per-struct `random_seed` | `controllers/incremental_pipeline.h` | 本地为旧字段名 + 全局 `--random_seed`（`util/option_manager.cc` L225） | W2/W3/W14 |
| G19 | PBA 后端残留（上游已移除） | c9729fa0 | 本地保留 `ba_global_use_pba` + `lib/PBA` + license_widget | W2 |
| G20 | model_aligner/comparer 增项：`ref_model_path`、`ref_is_gps`、`merge_image_and_ref_origins`、`ImageAlignmentError/AlignmentErrorSummary`、`max_proj_center_error` | `estimators/alignment.h`、`exe/model.cc` | 本地为旧版对齐流程，无误差统计输出 | W10 |
| G21 | 两视图选项面字段缺失 | `two_view_geometry.h` | 已有水印检测/sampson 细化；缺 3.1-G6 列项 | W6 |

### 3.3 已确认对齐、无需工作（避免虚增工作量）

`DatabaseCacheOptions`、`WorkspaceOptions`、`Poisson/Delaunay` 选项、
`StereoFusionOptions`（含 mask_path/use_cache）、`PatchMatchOptions`（本地另有多
个 photometric_* 自研选项）、Reconstruction 公有 API 方法面、IO 格式面
（BIN/TXT/NVM/Bundler/Recon3D/CAM/PLY/VRML）、guided matching、
LAD（least_absolute_deviations，已补）、Database Frame CRUD、
ImageReader mask（mask_path/camera_mask_path）、SIFT CPU/GPU、五类匹配器、
hierarchical mapper、undistort/rectify、retrieval、EQUIRECTANGULAR、
Rig/Frame v2 序列化、Caspar BA 核、tiny-Sampson F、6pt one-sided focal
（以上均有 manifest 或 grep 证据）。

**重要修正**：本地 `mvs/texture_mapping.{h,cc}` 并非自研，而是上游 47531f08
（#4202 "Add mesh texture mapping"）`MeshTextureMapping` 的回移植——struct/API
同名（`MeshTextureMappingOptions`/`MeshTextureMappingResult`），上游同样有
`texture_scale_factor`。本地 `mvs/texturing.{h,cc}` 是 ACloudViewer 集成层
（ccMesh + `PinholeCameraTrajectory` + `TextureView`），`image_texturer` 是其
CLI。因此 D1 的实现量比原估小：见 W15。

## 4. 工作分解（Work Breakdown）

估算尺码：S ≤ 1 人日级、M ≤ 1 周级、L ≤ 2–3 周级、XL ≥ 1 月级（含测试与 gate）。
每包含：内容 / 上游锚点 / 本地落点 / 测试与 gate / manifest 条目 / 依赖。

---

### P0 — 数据安全与后端面

#### [✅ 已完成（版本门 + 迁移 gate 绿）] W1 数据库版本迁移机制 [M]

- 内容：引入 `MakeDatabaseVersionNumber`/`GetDatabaseVersionNumber`；打开库时按
  `PRAGMA user_version` 门控迁移；`PreMigrateTables`/`PostMigrateTables`：
  零位姿→NULL、零 E/F/H→NULL、≤3.13 旧库升级路径。
- 上游锚点：`util/version.h`、`scene/database_sqlite.cc`（L2161 起迁移函数、
  L2193 起 user_version 门）。
- 本地落点：`src/base/database.cc`（替换 L1873 硬编码）、`src/util/version.h`。
  **迁移分支必须覆盖本地自有表**（`float_descriptors`/`rig_cameras`/
  `frame_images`），本地表保留、不参与上游迁移语义。
- 测试与 gate：移植上游 `database_test` 迁移用例 + 旧库 fixture；
  新 gate `reconstruction_db_migration_gate`（ctest -R）。
- manifest：`database_version_migration`（partial → implemented）。
- 依赖：无。**P0 最先执行**（W7 的 schema 扩展依赖迁移机制）。

#### [✅ 已完成] W2 BA 后端枚举化 + PBA 退役 [M]（D4）

- 内容：`BundleAdjustmentOptions` 增 `backend`(CERES|CASPAR) 与
  `check_if_stopped` 回调（W13 的前置）；mapper 选项面增
  `ba_local_backend`/`ba_global_backend`；移除 PBA：`optim/bundle_adjustment.cc`
  PBA 分支、`lib/PBA/`、`ba_global_use_pba`/`ba_global_pba_gpu_index`、
  `ui/license_widget.*`（PBA 许可页）、`pipelines/option_utils.cpp` 相关项、
  `ui/reconstruction_options_widget.cc`。
- 上游锚点：`estimators/bundle_adjustment.h` L207/L211、c9729fa0。
- 本地落点：`src/optim/bundle_adjustment.{h,cc}`、
  `src/controllers/incremental_mapper.{h,cc}`、上列 UI/pipeline 文件。
- 测试与 gate：`bundle_adjustment_test` + 既有
  `reconstruction_caspar_parity_gate` 回归。
- manifest：`ba_backend_surface`；PBA 移除写入 changelog。
- 依赖：无；W3/W13 受益。

---

### P1 — 核心算法栈

#### [⏸ partial（文件与清单就位，随 W3 启用）] W9 合成场景生成器 [S]（先行：W3/W4 的测试前置）

- 内容：移植 `scene/synthetic.{h,cc}`（SyntheticDatasetOptions、
  SynthesizeDataset）。
- 落点：`src/scene/synthetic.{h,cc}`（新建 scene/ 目录，策略 C）。
- 测试：上游 `synthetic_test`。
- manifest：`synthetic_dataset`。

#### [✅ 已完成（solvers/* 注册 + PoseLib 硬依赖；pose.cc ray-based 切换随 W3 收尾）] W5 最小求解器统一 [S]（先行于 W3d）

- 内容：移植 `estimators/solvers/poselib_utils.{h,cc}`；absolute/essential/
  generalized_absolute/generalized_relative 求解改走 PoseLib（与上游一致的
  算法集合）；one-sided-focal 路径不变。
- 落点：`src/estimators/solvers/`（新目录）；`PoseLib` 已链接
  （`RECONSTRUCTION_FETCH_POSELIB`）。
- 测试与 gate：并入既有 `reconstruction_camera_rig_parity_gate`
  （relpose_one_sided_focal_test 已在 gate 内）。
- manifest：`poselib_minimal_solvers`。

#### [✅ 已完成（上游本体切换 + parity 门；shared focal 用例随上游 suite）] W6 两视图几何增量 [M]

- 内容：`fundamental_matrix_degensac.{h,cc}`；TwoViewGeometryOptions 增
  `use_degensac`/`filter_stationary_matches`(+`stationary_matches_max_error`)/
  `force_H_use`/`compute_relative_pose`；`solvers/relpose_shared_focal.{h,cc}`
  并接入估计流程。
- 落点：`src/estimators/`（新文件按上游名）；`src/estimators/two_view_geometry.{h,cc}` 增字段。
- 测试与 gate：移植上游 `two_view_geometry_test`/`fundamental_matrix_test`
  相关用例，扩展现有 camera_rig gate。
- manifest：`two_view_geometry_increment`。

#### W3 frame-aware 增量建图管线 [XL]（W3-1 对应图缓存 ✅ 2026-09-06）

分四步独立提交，每步过 gate：

1. **cache 层**：`base/database_cache.{h,cc}` 装载 Frames/Rigs（pose_priors 位
   置在 W7 接入）；LRU 用既有 `util/cache.h`；内存估计：每 Frame 常量级记录 +
   现有 keypoints 描述符不变。
2. **对应图层**：`base/correspondence_graph.{h,cc}` 帧感知（按上游
   `scene/correspondence_graph.*` 移植数据结构与查找接口）。
3. **mapper 层**：按上游拆分移植 `sfm/incremental_mapper_impl.{h,cc}`、
   `sfm/observation_manager.{h,cc}`；`RegisterNextImage` +
   `RegisterNextImageFallback`；`num_reg_frames_per_rig`、
   `num_structure_less_reg_trials`、`fix_existing_frames`、
   `max_runtime_seconds`、`load_all_images`、`constant_rigs`/
   `constant_cameras`、`ba_global_ignore_redundant_points3D_min_coverage_gain`
   （接 W8 的 `FindRedundantPoints3D`）；注册路径接 generalized pose（激活
   W5 后的死代码）。
4. **BA 层**：Ceres 侧增 `refine_rig_from_world`/`refine_sensor_from_rig`/
   `constant_rig_from_world_rotation`/`min_track_length`（替换旧
   `std::vector<CameraRig>` API；`rig_bundle_adjuster` 命令随 W2 移除）；
   Caspar 因子矩阵从 11 组合扩到上游 15 组合（manifest caspar 条目的
   resolution 要求）。
- 测试与 gate：移植上游 `correspondence_graph_test`/`incremental_mapper_test`
  等；新 gate `reconstruction_frame_rig_pipeline_gate`（W9 合成 rig 数据 e2e）。
- manifest：`frame_aware_mapper`（分步 partial → implemented）。
- 依赖：W1（schema 稳定）、W2（BA 面）、W5（求解器）、W9（e2e fixture）。

#### W4 GLomap 全局 SfM 栈 [L]（与 W3 并行）

- 内容与落位（全部新文件，策略 C）：
  - `src/estimators/rotation_averaging.{h,cc}`（含 `_impl`）、
    `global_positioning.{h,cc}`、`view_graph_calibration.{h,cc}`；
  - `src/estimators/cost_functions/motion_averaging.h`（新建 cost_functions/ 子目录）；
  - `src/scene/pose_graph.{h,cc}`、`src/math/{connected_components,spanning_tree,union_find}.{h,cc}`；
  - `src/optim/sparse_cholesky.{h,cc}`（LAD 已在，二者配套）；
  - `src/sfm/global_mapper.{h,cc}`、`src/controllers/global_pipeline.{h,cc}`、
    `src/controllers/rotation_averaging.{h,cc}`。
- 编排决策：上游 `GlobalMapper` 已含多组件支持（d2da1944）；保留本地
  `GlobalMapperController` 作为 GUI/多组件入口，其 per-component 求解器切换为
  上游 GLomap 核；CLI 增 `global_mapper`/`rotation_averager`/
  `view_graph_calibrator`。
- 测试与 gate：移植上游 `rotation_averaging_test`/`global_positioning_test`/
  `view_graph_calibration_test`/`global_mapper_test`；新 gate
  `reconstruction_glomap_gate`（多组件 e2e 用 W9）。
- manifest：`global_mapper_glomap`（partial → implemented）。
- 依赖：W9（测试）；与 W3 无耦合。

#### W7 位姿先验栈 [L]（依赖 W1、W3-1）

- 内容：`pose_priors` 表 + 迁移（W1 机制）；`src/base/pose_prior.{h,cc}`
  （上游 `geometry/pose_prior.*`）；`cost_functions/pose_prior.h`；
  `gravity_refinement.{h,cc}`；mapper 选项 `use_prior_position`/
  `prior_position_loss_scale`/`use_robust_loss_on_prior_position`；
  `DatabaseCache` 装载先验；CLI `pose_prior_mapper`；
  `model_aligner` 补 `ref_is_gps`/`merge_image_and_ref_origins`（GPS→先验换算复用 `base/gps.h`）。
- 测试与 gate：上游 `pose_prior` 相关测试 + DB 迁移用例；新 gate
  `reconstruction_pose_prior_gate`。
- manifest：`pose_prior_mapping`。

---

### P2 — 工具 / CLI / UX / 杂项（相互独立，可并行分包）

#### W8 模型聚类与剪枝 [M]

`scene/reconstruction_clustering.*`、`scene/reconstruction_pruning.*`
（`FindRedundantPoints3D`）、`controllers/reconstruction_clustering.*`、
CLI `model_clusterer`；mapper 剪枝选项接线（G9 尾项）。
测试：移植 `reconstruction_clustering_test`/`reconstruction_pruning_test`；
gate 并入 `reconstruction_glomap_gate` 相邻目标或独立
`reconstruction_model_tools_gate`。manifest：`model_clustering_pruning`。

#### W10 模型对齐/比较增项 [S]

移植 `estimators/alignment.{h,cc}`；`exe/model.cc`：`RunModelAligner` 补
`ref_model_path`/`ref_is_gps`/`merge_image_and_ref_origins` 与
`ImageAlignmentError`/`AlignmentErrorSummary` 误差统计；`RunModelComparer` 补
`max_proj_center_error`。测试：`alignment_test`。manifest：`model_alignment`。

#### W11 匹配基础设施 [M]

`src/feature/index.{h,cc}` `FeatureDescriptorIndex`（消费已链接的
`3rdparty_faiss`；DEFAULT=flat 可退化，FAISS 可选启用）；matcher cache 接入
索引；CLI `geometric_verifier`/`guided_geometric_verifier`（对导入匹配做两视图
几何校验）。测试：移植 `feature/index_test`、`matcher_cache_test` 相关用例。
manifest：`feature_descriptor_index`、`geometric_verifier`。
注：`controllers/pairing.{h,cc}` 流式配对重构为可选低优先项，不阻塞。

#### W12 rig 配置 CLI [S]

`exe/database.cc` 增 `RunRigConfigurator`（复用既有 Database Frame/Rig CRUD）。
manifest：`rig_configurator`。

#### W13 优雅退出 [M]（依赖 W2 的 check_if_stopped）

`src/util/cancellation.h`（CancellationToken）+ `exe/colmap.cc`
`kSupportsGracefulShutdown` 标注长跑命令 + BA/mapper/matching 贯通取消点；
GUI（ACloudViewer app/reconstruction）停止按钮走同一 token。
manifest：`graceful_shutdown`。

#### W14 杂项批（各项独立 S）

- `ImageReaderOptions.as_rgb`（图像读取色彩策略）；
- SIFT：`force_covariant_extractor` 开关 + 上游 #2929 affine 方向检测修复；
- AutomaticReconstructionOptions：`single_camera_per_folder`/`camera_params`/
  per-struct `random_seed`（同时把全局 `--random_seed` 迁移语义处理好，数值门
  固定种子复跑出 before/after 报告）；
- `estimators/covariance.{h,cc}`（`EstimateBACovariance`）；
- `ui/mesh_painter.{h,cc}` + model_viewer 接线；
- `util/timestamp.h`。
- manifest：按条目各一条（`image_reader_as_rgb`、`covariant_sift`、
  `ba_covariance`、`mesh_painter` 等）。

---

### 新增工作包（产品决策落地）

#### W15 mesh 纹理双路径（D1）[M]

**事实基线**：本地 `mvs/texture_mapping.{h,cc}` 已是上游 47531f08 (#4202)
`MeshTextureMapping` 的回移植（同名 API）；`mvs/texturing.{h,cc}` 为
ACloudViewer 集成层（ccMesh + `PinholeCameraTrajectory`）；`image_texturer`
是该集成层的 CLI；上游 CLI 为 `mesh_texturer`（workspace + PLY + atlas 输出）。
本地 `.cc` 与上游存在 77 行 delta（含 include 适配），测试覆盖 4 vs 上游 15。

工作项：
1. `.cc` 77 行 delta 对账：逐块审查，以上游 `a395b826` 为基准回填算法行为，
   仅保留 include 路径等本地必要适配；结论记录进 manifest reason。
2. 移植上游 CLI：`exe/mvs.cc` `RunMeshTexturer`
   （workspace_path/input_path/output_path/output_type BIN|TXT +
   `AddMeshTextureMappingOptions`）；`util/option_manager.h` 增
   `AddMeshTextureMappingOptions`；`exe/mvs.h` + `exe/colmap.cc` 注册
   `mesh_texturer`。
3. 默认策略（D1）：`AutomaticReconstructionOptions` 增
   `texturing_type = {MESH_TEXTUREUR(默认), IMAGE_TEXTUREUR}`；
   `controllers/texturing_controller.cc` 按 enum 分派——默认走上游等价流
   （undistorted workspace + meshing 输出 PLY → `MeshTextureMapping` atlas），
   IMAGE_TEXTUREUR 保持现 `texturing.h` 集成流；dense_reconstruction_widget
   与 app/reconstruction 面板展示该选择。
4. 保留 `image_texturer`（同引擎、ACloudViewer 集成流）并标注 alternative。
5. 测试：`texture_mapping_test` 4→15 用例补齐（移植上游其余 11 个）；
   gate：`texture_mapping_test` 继续挂在 `reconstruction_oiio_parity_gate`，
   新增 `reconstruction_mesh_texturer_gate`（mesh_texturer e2e 小 fixture）。
6. manifest：新增 `mesh_texturer_dual_path`；更新既有 OIIO 条目中
   texture_mapping_test 的归属说明；`docs/COLMAP_ALIGNMENT.md` 增一行结论。

#### W16 pycolmap 融合进 `cloudViewer.reconstruction`（D2）[L]

**设计修正（2026-09-05 用户评审）**：不新建平行 Python 包；与既有绑定模块
**融合去重、合并接口**。

**既有面（本地事实）**：
- `libs/Python/pybind/reconstruction/`（子模块 `database/feature/gui/image/
  model/mvs/sfm/vocab_tree`），注册于 `cloudViewer_pybind.cpp` L55 → Python
  命名空间 `cloudViewer.reconstruction`；
- 绑定的是 `libs/Reconstruction/src/pipelines/*.{h,cpp}` 库级命令包装
  （`extract_feature`/`exhaustive_match`/`auto_reconstruction`/
  `bundle_adjustment`/`stereo_patch_match`/`stereo_fuse`/`mesh_delaunay`/
  `poisson_mesh`/`align_model`/`convert_model`/`undistort_image`/
  `build_vocab_tree`/`create_database` 等）+ colmap 选项 struct
  （`ImageReaderOptions`/`SiftExtractionOptions`/`SiftMatchingOptions`/
  `ExhaustiveMatchingOptions`…，见 `reconstruction_options.cpp`）。

**上游面**：`src/pycolmap/` 类级绑定（Database、Reconstruction、Camera、
Image、Point2D/Point3D、Track、Rig/Frame、Rigid3d/Sim3d、estimators、optim、
pipeline、mvs、retrieval）。

**融合原则**：
1. 单一命名空间 `cloudViewer.reconstruction`；不新建 `cloudViewer.pycolmap`、
   不设 compat shim。
2. 每个概念只允许一个规范绑定：既有函数名/选项类是规范接口（已公开、有
   用户），上游新增**类**与新增能力并入；与既有函数同义的上游函数不重复
   绑定，改为给既有函数补上游选项/返回结构。
3. 选项类去重：`reconstruction_options.cpp` 已绑定的 colmap 选项类是唯一
   绑定点；适配后的 pycolmap 源码引用它，不再重复 `py::class_`。
4. 实现后端统一：既有 `pipelines/*` 包装函数继续作为其 Python API 的实现；
   重叠函数以既有签名为规范、吸收上游新参数（W3/W14/W15 的新选项在此同步
   暴露）。
5. 绑定实现风格遵循本地模块约定（`cloudViewer` 命名空间、
   `docstring.cpp` 文案注册）。

**子模块映射与去重表**：

| 上游 pycolmap 子包 | 去向（pybind/reconstruction/） | 去重动作 |
|---|---|---|
| `scene/`（Camera/Image/Point2D/Point3D/Track/Rig/Frame/Reconstruction + IO） | 新增 `scene/` | 纯新增 |
| `scene/database`（Database 类 + row 类型） | `database/` | 新增类；既有 `create/clean/merge_database` 保留；`Database` 类成为底层标准接口，既有函数实现改走 `colmap::Database` |
| `geometry/`（Rigid3d/Sim3d/gps） | 新增 `geometry/` | 纯新增 |
| `sensor/`（Bitmap/Camera/Rig/specs） | 新增 `sensor/` | 纯新增（Bitmap 为 image/ 与 mvs/ 复用的基础类型） |
| `estimators/` + `optim/`（pose/E/F/H/相似变换/RANSAC 类型） | 新增 `estimators/` | 纯新增 |
| `feature/`（SIFT 配置/关键点/描述符 + extract/match） | `feature/` | 类并入；`extract_feature`/`*_match` 既有名保留为规范函数，实现统一到上游路径 |
| `image/`（undistortion/warp） | `image/` | `rectify_image`/`undistort_image` 等既有名保留；上游 Bitmap/undistortion 类并入 |
| `pipeline/feature` + `pipeline/sfm`（incremental options、Pipeline） | `sfm/` + `feature/` | 并入既有 `auto_reconstruction`/`bundle_adjustment` 的选项面；**不新增第二条重建入口** |
| `pipeline/mvs` + `pipeline/meshing` | `mvs/` | 既有 `stereo_patch_match`/`stereo_fuse`/`mesh_delaunay`/`poisson_mesh` 保留；新增 Workspace/DepthMap/NormalMap/PatchMatchOptions 类；W15 的 `mesh_texturer`/`image_texturer` 在此暴露 |
| `pipeline/images` + model 工具 | `model/` | `align/analyze/compare/convert/crop/merge/split_model` 既有名保留；上游图像访问器类并入 |
| `retrieval/`（VisualIndex） | `vocab_tree/` | `build_vocab_tree`/`retrieve_vocab_tree` 既有名保留；VisualIndex 类新增 |
| `gui` | `gui/` | ACloudViewer 专属，上游无对应，不动 |

工作项：
1. **CMake**：为 ColmapLib 建 INTERFACE/ALIAS 目标 `colmap::colmap`
   （include dirs + Ceres/glog/boost 链接闭包走既有 EXTERNAL_LIBRARIES）；
   `CV_PYTHON_LIB`（wheel 模块目标）条件性追加该链接与
   `libs/Reconstruction/src` include——条件 `BUILD_RECONSTRUCTION`；无重建时
   融合绑定整树排除（对齐上游 `MVS_ENABLED` 过滤模式）。
2. **分批合入**：按去重表将上游 `src/pycolmap/<pkg>/bindings.cc` 适配合入
   `pybind/reconstruction/<去向>/`（include 映射按 2.3；风格本地化）；批序
   scene/geometry/sensor/estimators 先行（不依赖 W3/W4），sfm/mvs/model 随
   W3/W8/W14/W15；每批过编译 + import 冒烟。
3. **接口合并**：既有函数补齐上游新参数（per-struct `random_seed`、
   `as_rgb`、`texturing_type` 等）；同义上游函数不落绑定；
   `reconstruction_options.cpp` 保持选项类唯一定义点。
4. **docstring/文档**：`docstring.cpp` 注册文案中注明与上游 pycolmap 的 API
   对应关系（便于直接使用上游文档/示例）。
5. **测试**：移植上游 pytest 并把 `import pycolmap` 改为
   `import cloudViewer.reconstruction`；删除本地刻意不暴露面的用例；新增
   绑定唯一性回归（import 时断言无重名/重复注册）。
6. **CI**：ubuntu Python job 增 pytest 步骤 + wheel 冒烟
   `import cloudViewer.reconstruction`。
7. **边界**：只包 ColmapLib C++ API；不触碰 `aicore_*` C ABI（既有
   `is_aicore_available` 等函数面保持）；DA3 深度融合不在 Python 面新增暴露。
8. **hash-map 后端一致性**（基线 #4673 语义带入）：随 W3/W4 端口代码引入的
   `util/hash_containers.h` 采用上游 STD 默认 + AUTO 警告语义；
   `colmap::colmap` 别名复刻 `colmap-config.cmake` 的后端预置（下游在配置期
   对不一致报错，而非静默重推导导致布局漂移）；融合模块暴露
   `__hash_map_backend__` 并在 wheel 冒烟中与 ColmapLib 构建值断言一致
   （移植上游 `main_test.py` 对应用例）；GetBuildInfo 携带后端
   （`version.cc.in` 增补随 W1 一并移植）。
- manifest：`pycolmap_bindings`（partial（骨架 + scene/geometry/sensor/
  estimators 批）→ implemented（去重表全部清零、pytest 绿））。
- 依赖：骨架（第 1–2 步）不阻塞于 W3/W4/W7；sfm/mvs/model 子包的选项暴露
  随 W3/W8/W14/W15。

---

## 5. 依赖拓扑与里程碑

```
W1 ──→ W7 ──→ W3(1,2)          W9 ──→ W3(4,e2e)、W4(gate)、W8
W2 ──→ W13、W3(4)、W15(默认项)      W5 ──→ W3(3)
W6 独立；W10/W11/W12/W14 独立；W16 骨架独立、子包随 W3/W4/W7/W8
W15 依赖 W2（AutomaticReconstruction 选项面变更方式），其余独立
```

| 里程碑 | 内容 | 出口条件 |
|---|---|---|
| M0 | W1 + W2 | 迁移 gate + BA backend gate 绿；PBA 从包内消失 |
| M1 | W9 + W5 + W6 | 合成器/求解器/两视图 gate 绿 |
| M2 | W3 | frame_rig_pipeline_gate 绿（rig 合成 e2e） |
| M3 | W4 + W7 + W8 + W15 | glomap/pose_prior/model_tools/mesh_texturer gate 绿 |
| M4 | W10–W14 + W16 | 全部 gate + pytest 套件绿；`reconstruction_alignment_all` 汇总目标绿 |

## 6. 验证体系

- **单包**：移植/新增 `*_test.cc`（策略 C 下新文件可原样 include）+ 各自
  `reconstruction_*_gate`（照 `reconstruction_camera_rig_parity_gate` 的
  `ctest -R` 模式注册进 `libs/Reconstruction/src/CMakeLists.txt`）。
- **汇总**：新增总目标 `reconstruction_alignment_all`（依赖全部 gate）。
- **数值门**：固定种子（W14 per-struct random_seed 落地后统一），行为变更项
  出 before/after 报告。
- **Python**：W16 pytest 套件 + wheel build 冒烟（`import
  cloudViewer.reconstruction`）+ 绑定唯一性回归（无重名/重复注册）+
  `__hash_map_backend__` 与 ColmapLib 构建后端一致性断言。
- **manifest 流转**：每包完成后新增/更新条目，遵守其 policy——仅
  `implemented` 是发布声明；`partial` 必须带 prerequisites 与 resolution。

## 7. 风险与回滚

| 风险 | 缓解 |
|---|---|
| W3 触碰 cache/对应图/mapper 三层联动 | 四步独立提交（§4 W3），每步过 gate；旧路径编译开关保留一个里程碑后删除 |
| Caspar 因子矩阵扩展（11→15 组合）回归 | W3-4 内同步扩 `reconstruction_caspar_parity_gate`；manifest caspar 条目 resolution 已预置此要求 |
| DB 迁移破坏本地自有表 | W1 迁移函数含本地分支 + 旧库 fixture 用例（DoD） |
| per-struct random_seed 改变默认数值行为 | W14 切换时全量 gate 固定种子复跑并留档 |
| PBA/`rig_bundle_adjuster` 移除影响既有工作流 | W2 发布说明 + CHANGELOG；无编译期保留（上游已删，本地长期维护成本更高） |
| pycolmap 与既有 `cloudViewer.reconstruction` 的重名冲突/双重注册 | 去重表驱动合入（§4 W16）；每批编译 + pytest；import 期绑定唯一性断言；选项类仅在 `reconstruction_options.cpp` 定义一次 |
| hash map 后端在 ColmapLib 与 Python 扩展间不一致（#4673 场景：布局漂移→内存损坏而非链接错误） | 后端固定 STD 默认；`colmap::colmap` 别名预置后端（配置期报错）；`kHashMapBackend` 进 GetBuildInfo、`__hash_map_backend__` 进 pybind 模块并双端断言（W16-8） |
| 与 AICore/DA3 链路冲突 | 本方案不触碰 `aicore_*` ABI 与 DA3 控制器；W16 明确不暴露 DA3 |
| 上游继续演进造成基线漂移 | manifest `upstream_revision_checked` 与本文档基线锁 `dbb41680`；后续升级走独立任务（重扫缺口 + A/B） |

## 8. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-04 | 初版：两轮缺口扫描结论（G1–G21）+ W1–W14 + 策略 C |
| 2026-09-05 | 产品决策 D1/D2/D4 定稿：新增 W15（mesh 纹理双路径，上游默认）、W16（pycolmap 集成）；修正 texture_mapping 归属为上游回移植；PBA 退役定案 |
| 2026-09-05 | W16 修正（用户评审）：改为与既有 `cloudViewer.reconstruction` 模块**融合去重、合并接口**（单一命名空间、既有函数名为规范接口、选项类唯一定义点），不新建 `cloudViewer.pycolmap` 平行包；G16/D2/风险表/验证体系同步修正 |
| 2026-09-05 | 对齐基线更新 `a395b826` → `dbb41680`（Δ=1：#4673 hash map 后端固定，STD 默认 + 预置 + 双端暴露）；G1–G21 结论复核沿用；W16 增补第 8 项 hash 后端一致性约束，§6/§7 同步 |
| 2026-09-07 | W9 落地：`scene/synthetic.{h,cc}` 注册进 ColmapLib，synthetic_test 18/18 + 上游 gps ENU 数值门（`base/gps_test`）绿；顺带修复 5 项 fork 缺陷（`Rig::SensorFromRig` 四元数顺序、GPS WGS84 扁率与 XYZToEll 收敛、pose_priors 表/语句缺失与 ReadPosePriorRow 列错位、`Frame::AddImageId` 在 image_id != camera_id 时的脏 data_id 重复、`Image::SetPoints2D` 不重算 num_points3D_）并补齐上游 Database API 名（`ExistsTwoViewGeometry`/`UpdateKeypoints`/`ReadTwoViewGeometries()`）与测试基建（`util/eigen_matchers.h`、`CreateTestDir`/`CreateDirIfNotExists`、COLMAP_ADD_TEST 链接 gmock）；W3-2b 前置条件就绪。全量构建 EXIT=0，ctest 77/79（余下 frame_test 与 caspar split-intrinsics focal=0 pp=0 两项与本工作无关，rms 逐位复现且 optim/ 与 HEAD 无 diff，记为预存基线问题） |
