# 测试对齐移植 TODO(批次 24)

> 基线:上游 MVS/colmap d3ccaf35 测试树 ↔ fork 测试树,按测试文件名差集
> 生成(2026-09-27)。C++:上游 166 / fork 101 / 缺 52;
> Python:上游 68 / fork 39 / 缺 28。
> 移植前须逐个核对主题是否已被 fork 合并文件覆盖,避免重复。

## P0 — 新式抽象配套测试(批次 23 新增面,最高优先)

| 上游测试 | 上游路径 | 覆盖对象 |
|---|---|---|
| `extractor_test` ✅已入 gate | `src/colmap/feature/` | 新式抽象薄壳(C++ 层) |
| `matcher_test` ✅已入 gate | `src/colmap/feature/` | 新式抽象薄壳(C++ 层) |
| `extract_features_test` ✅pytest | `src/pycolmap/pipeline/` | pybind 层(批次 20/23 面) |
| `extraction_test` ✅pytest | `src/pycolmap/feature/` | pybind 层(批次 20/23 面) |
| `matching_test` ✅pytest | `src/pycolmap/feature/` | pybind 层(批次 20/23 面) |

> 批次 24 第一批完成(2026-09-27):P0 全部落地。C++ 落 `feature/extractor_test.cc`
> / `feature/matcher_test.cc`(ctest `feature/extractor_test`、`feature/matcher_test`
> 通过);Python 落 `python/test/reconstruction/pipeline/test_extract_features.py`
> / `feature/test_extraction.py` / `feature/test_matching.py`(35 用例过)。
> 移植同时驱动 fork 表面对齐:`FeatureExtraction/MatchingOptions` 全选项组分配+深拷贝
> (上游 TypeOptions 语义)、`RequiresRGB`/`RequiresOpenGL`、`EffMaxImageSize` 类型默认值
> (SIFT 3200/其余 1600)、LoMa `Check()` 前置 `ResolveDefaultLomaModelPaths`(不下载)、
> ALIKED 工厂缺后端即抛(对齐上游 ONNX-off 语义)、pybind 补 `requires_rgb`/`num_octaves`。
> ONNX→ggml 替换面:上游 `onnx_matchers.h` 契约由 `learned_matchers.h` 承接,
> 嵌套 `BruteForceONNXMatchingOptions` 扁平化到 `AlikedMatchingOptions.min_cossim`。

## P1 — C++ 测试差集(52 项全列)

| # | 上游测试 | 上游路径 |
|---|---|---|
| 1 | `alignment_test` ✅已入 gate | `src/colmap/estimators/` |
| 2 | `alignment_test` ✅已入 gate(cost_functions 版) | `src/colmap/estimators/cost_functions/` |
| 3 | `aliked_test` ✅已入 gate(批次A:ggml引擎接线) | `src/colmap/feature/` |
| 4 | `automatic_reconstruction_test` ⚠裁定:fork 已重构为 DA3 控制器 | `src/colmap/controllers/` |
| 5 | `base_controller_test` ✅已入 gate | `src/colmap/util/` |
| 6 | `base_option_manager_test` ⚠裁定:fork 无 BaseOptionManager 面 | `src/colmap/controllers/` |
| 7 | `bundle_adjustment_ceres_test` ⚠裁定:fork 为 Caspar/Ceres 双后端面 | `src/colmap/estimators/` |
| 8 | `calibration_test` ✅已入 gate | `src/colmap/estimators/cost_functions/` |
| 9 | `cancellation_test` ✅已入 gate | `src/colmap/util/` |
| 10 | `delaunay_meshing_test` ✅已入 gate | `src/colmap/mvs/` |
| 11 | `eigen_matchers_test` ✅已入 gate | `src/colmap/util/` |
| 12 | `enum_utils_test` ✅已入 gate | `src/colmap/util/` |
| 13 | `extractor_test` ✅(并入 P0) | `src/colmap/feature/` |
| 14 | `feature_extraction_test` ⚠裁定:fork 保持 SiftFeatureExtractor 旧面 | `src/colmap/controllers/` |
| 15 | `feature_matching_test` ⚠裁定:fork 为 pairing 驱动引擎面 | `src/colmap/controllers/` |
| 16 | `feature_matching_utils_test` ⚠裁定:fork 无 FeatureMatcherCache 面 | `src/colmap/controllers/` |
| 17 | `fundamental_matrix_degensac_test` ✅已入 gate | `src/colmap/estimators/` |
| 18 | `fusion_test` ✅已入 gate | `src/colmap/mvs/` |
| 19 | `gravity_refinement_test` ✅已入 gate | `src/colmap/estimators/` |
| 20 | `hierarchical_pipeline_test` ⚠裁定:fork 为 HierarchicalMapperController 旧面 | `src/colmap/controllers/` |
| 21 | `image_reader_test` ✅已入 gate | `src/colmap/controllers/` |
| 22 | `incremental_mapper_test` ✅已入 gate(3 生命周期用例裁定) | `src/colmap/sfm/` |
| 23 | `incremental_pipeline_test` ⚠裁定:fork 无 IncrementalPipeline 类 | `src/colmap/controllers/` |
| 24 | `incremental_triangulator_test` ✅已入 gate(Retriangulate 用例裁定) | `src/colmap/sfm/` |
| 25 | `jacobian_test` ✅已入 gate | `src/colmap/sensor/models/` |
| 26 | `logging_test` ✅已入 gate | `src/colmap/util/` |
| 27 | `loma_test` ✅已入 gate(ggml 真推理) | `src/colmap/feature/` |
| 28 | `matcher_test` ✅(并入 P0) | `src/colmap/feature/` |
| 29 | `matcher_cache_test` ⚠裁定:fork 无 FeatureMatcherCache 面 | `src/colmap/controllers/` |
| 30 | `model_test` ✅已入 gate | `src/colmap/mvs/` |
| 31 | `motion_averaging_test` ✅已入 gate | `src/colmap/estimators/cost_functions/` |
| 32 | `onnx_matchers_test` ⚠裁定:不移植 | `src/colmap/feature/` |
| 33 | `onnx_utils_test` ⚠裁定:不移植 | `src/colmap/feature/` |
| 34 | `option_manager_test` ✅已入 gate | `src/colmap/controllers/` |
| 35 | `pairing_test` ⚠裁定:依赖 FeatureMatcherCache 面 | `src/colmap/controllers/` |
| 36 | `ply_test` ✅已入 gate | `src/colmap/util/` |
| 37 | `poisson_meshing_test` ✅已入 gate | `src/colmap/mvs/` |
| 38 | `poselib_utils_test` ⚠裁定:fork 构建未启用 PoseLib | `src/colmap/estimators/solvers/` |
| 39 | `quaternion_utils_test` ✅已入 gate | `src/colmap/estimators/cost_functions/` |
| 40 | `relpose_shared_focal_test` ✅已入 gate | `src/colmap/estimators/solvers/` |
| 41 | `reprojection_error_test` ⚠裁定:AnalyticalVersusAutoDiff SegFault 待查 | `src/colmap/estimators/cost_functions/` |
| 42 | `sampson_error_test` ✅已入 gate(RelPoseParams 用例裁定) | `src/colmap/estimators/cost_functions/` |
| 43 | `sfm_test` ⚠裁定:fork exe 层无 RunPointTriangulatorImpl | `src/colmap/exe/` |
| 44 | `sparse_cholesky_test` ✅已入 gate | `src/colmap/optim/` |
| 45 | `sprt_test` ✅已入 gate(暴露并修复 SPRT out-param bug) | `src/colmap/optim/` |
| 46 | `tiny_manifold_test` ✅已入 gate(SphereManifold<9> 用例裁定) | `src/colmap/estimators/cost_functions/` |
| 47 | `tiny_sampson_error_test` ✅已入 gate | `src/colmap/estimators/cost_functions/` |
| 48 | `tiny_solver_test` ✅已入 gate(NUMERICAL_FAILURE 已补齐) | `src/colmap/optim/` |
| 49 | `undistorters_test` ⚠裁定:fork 无 undistorters 控制器面 | `src/colmap/controllers/` |
| 50 | `version_test` ✅已入 gate | `src/colmap/util/` |
| 51 | `vote_and_verify_test` ✅已入 gate(eff_inlier_count 用例裁定) | `src/colmap/retrieval/` |
| 52 | `workspace_test` ✅已入 gate | `src/colmap/mvs/` |

> ONNX 引擎级测试裁定(批次 24):`onnx_matchers_test`/`onnx_utils_test` 是
> onnxruntime session 机制的运行时测试,fork 无 onnxruntime,其选项契约层已由
> `learned_matchers.h`/`aliked.h` 承接并经 P0 测试钉住;引擎级等价探针待 AICore
> ALIKED task 落地后补(与 `aliked.h` 头注释既有裁定一致)。

## P2 — Python 测试差集(28 项全列)

| # | 上游测试 | 上游路径 | fork 落位 |
|---|---|---|---|
| 1 | `affine_transform_test` ✅pytest | `src/pycolmap/estimators/` | `python/test/reconstruction/estimators/test_affine_transform.py` |
| 2 | `cancellation_test` ✅pytest | `src/pycolmap/util/` | `python/test/reconstruction/util/test_cancellation.py` |
| 3 | `dataclass_test` ⚠skip留因:fork 无 dataclass 协议面 | `src/pycolmap//` | `python/test/reconstruction/./test_dataclass.py` |
| 4 | `essential_matrix_test` ✅pytest | `src/pycolmap/estimators/` | `python/test/reconstruction/estimators/test_essential_matrix.py` |
| 5 | `essential_matrix_test` ✅pytest(from_pose 改名) | `src/pycolmap/geometry/` | `python/test/reconstruction/geometry/test_essential_matrix.py` |
| 6 | `extract_features_test` ✅pytest | `src/pycolmap/pipeline/` | `python/test/reconstruction/pipeline/test_extract_features.py` |
| 7 | `extraction_test` ✅pytest | `src/pycolmap/feature/` | `python/test/reconstruction/feature/test_extraction.py` |
| 8 | `fundamental_matrix_test` ✅pytest | `src/pycolmap/estimators/` | `python/test/reconstruction/estimators/test_fundamental_matrix.py` |
| 9 | `generalized_pose_test` ✅pytest | `src/pycolmap/estimators/` | `python/test/reconstruction/estimators/test_generalized_pose.py` |
| 10 | `global_mapper_test` ✅pytest | `src/pycolmap/sfm/` | `python/test/reconstruction/sfm/test_global_mapper.py` |
| 11 | `hierarchical_mapper_test` ✅pytest | `src/pycolmap/sfm/` | `python/test/reconstruction/sfm/test_hierarchical_mapper.py` |
| 12 | `homography_matrix_test` ✅pytest | `src/pycolmap/estimators/` | `python/test/reconstruction/estimators/test_homography_matrix.py` |
| 13 | `homography_matrix_test` ✅pytest(from_pose 改名) | `src/pycolmap/geometry/` | `python/test/reconstruction/geometry/test_homography_matrix.py` |
| 14 | `images_test` ✅pytest | `src/pycolmap/pipeline/` | `python/test/reconstruction/pipeline/test_images.py` |
| 15 | `incremental_mapper_test` ✅pytest | `src/pycolmap/sfm/` | `python/test/reconstruction/sfm/test_incremental_mapper.py` |
| 16 | `incremental_triangulator_test` ✅pytest | `src/pycolmap/sfm/` | `python/test/reconstruction/sfm/test_incremental_triangulator.py` |
| 17 | `logging_test` ✅pytest | `src/pycolmap/util/` | `python/test/reconstruction/util/test_logging.py` |
| 18 | `main_test` ✅pytest(get_version 适配) | `src/pycolmap//` | `python/test/reconstruction/./test_main.py` |
| 19 | `matching_test` ✅pytest | `src/pycolmap/feature/` | `python/test/reconstruction/feature/test_matching.py` |
| 20 | `meshing_test` ✅pytest | `src/pycolmap/pipeline/` | `python/test/reconstruction/pipeline/test_meshing.py` |
| 21 | `motion_averaging_test` ✅pytest | `src/pycolmap/estimators/` | `python/test/reconstruction/estimators/test_motion_averaging.py` |
| 22 | `mvs_test` ✅pytest | `src/pycolmap/pipeline/` | `python/test/reconstruction/pipeline/test_mvs.py` |
| 23 | `observation_manager_test` ✅pytest | `src/pycolmap/sfm/` | `python/test/reconstruction/sfm/test_observation_manager.py` |
| 24 | `pose_prior_test` ✅pytest | `src/pycolmap/geometry/` | `python/test/reconstruction/geometry/test_pose_prior.py` |
| 25 | `sfm_test` ✅pytest | `src/pycolmap/pipeline/` | `python/test/reconstruction/pipeline/test_sfm.py` |
| 26 | `similarity_transform_test` ✅pytest | `src/pycolmap/estimators/` | `python/test/reconstruction/estimators/test_similarity_transform.py` |
| 27 | `timer_test` ✅pytest | `src/pycolmap/util/` | `python/test/reconstruction/util/test_timer.py` |
| 28 | `timestamp_test` ⚠skip留因:fork 未绑定时间戳助手 | `src/pycolmap/util/` | `python/test/reconstruction/util/test_timestamp.py` |

## 移植约定(沿袭既有轮次)

- C++ 测试进 gate:`COLMAP_ADD_TEST` 自动收集;headless 排除项仍为
  `feature/sift_test` 与 `util/opengl_utils_test`(GL 依赖)。
- Python 测试落 `python/test/reconstruction/<上游目录>/test_<名>.py`,
  Open3D 风格(pytest + np.testing + conftest fixture);禁 `__init__.py`;
  basename 冲突沿用改名惯例。
- 引擎护栏:PINHOLE focal_length 标量 CHECK、faiss 全零词表 abort、
  glog CHECK 前置状态——违反前置条件的用例先补前置状态或 skip 留因。
- 每完成一批跑:`reconstruction_cpp_tests_gate`(104+ 项)+
  `pytest python/test/reconstruction/`。

## 批次 24 收尾(2026-09-27)

- 最终状态:C++ gate `ctest -R "^[a-z_0-9]+/[a-z_0-9]+$" -E "^(feature/sift_test|util/opengl_utils_test)$"`
  **149/149 通过**;`pytest python/test/reconstruction/` **395 passed / 3 skipped**(skip 即
  P2 dataclass/timestamp 两处留因 + 既有 skip)。P1/P2 中所有未入 gate 项均已带
  ⚠裁定 注明根因与恢复条件,不留静默缺口。
- 随移植修复的生产缺陷(均有测试断言背书):
  1. `SPRT::Evaluate` 将指针本身当作计数器覆写(`num_inliers = 0` 少解引用),out-param 恒为 0;
  2. `mvs::Image` 默认构造不初始化 width_/height_/K_/R_/T_,默认实例携带垃圾值;
  3. `OptionManager::AddAndRegisterDefaultOption` 对 `std::filesystem::path` 注册裸
     typed_value,写出的空值 config 无法回读;改为 string 中转 + notifier(并为
     boost::program_options 补 std::filesystem::path 的 validate 特化);
  4. `mvs::Model::GetImageIdx/GetImageName` 由 glog CHECK abort 改为 THROW(抛
     std::invalid_argument,对齐上游可恢复语义);
  5. `TinySolver` 补 upstream 的 NUMERICAL_FAILURE 状态(NaN 残差/Jacobian/参数/步长
     四处检测);
  6. `MAKE_ENUM(_CLASS)` 宏补生成 `XxxValues()/XxxStrings()`(含谓词重载,上游 parity);
  7. `logging.h` 补 `CHECK_OPTION_IN` 区间宏。
- 同名 target 冲突惯例:不同目录同名测试以 `<目录>_<名>` 注册(如
  `cost_functions_alignment_test`、`mvs_image_test`);ctest 名仍带目录前缀不冲突。
- Python 基名冲突沿用改名惯例:geometry 侧 `test_essential_matrix_from_pose.py` /
  `test_homography_matrix_from_pose.py`(estimator 侧保留原名)。
- 未移植项恢复条件汇总:ALIKED/ONNX 引擎面批次A已接线ggml引擎;controllers 新抽象
  (FeatureExtractorController/FeatureMatcherController/HierarchicalPipeline/
  AutomaticReconstruction/IncrementalPipeline/undistorters/FeatureMatcherCache/
  BaseOptionManager)待 fork 迁移到上游控制器抽象;PoseLib 待构建启用;dataclass
  协议/时间戳助手/pybind requires_rgb 之外的字段级绑定缺口随绑定收口补齐。

## 批次 A — 接线与补注册(2026-09-27)

- ALIKED/LightGlue feature 层接线完成:feature/aliked.{h,cc} 通过共享 aicore 模型目录
  (AICORE_MODEL_FAMILY_ALIKED/LIGHTGLUE)解析默认模型,SHA-256 校验 + 共享缓存下载
  (feature/loma.cc 模式);FeatureExtractor/FeatureMatcher::Create 的 ALIKED 分支从
  throw 改为真实 shell(提取 = aicore_aliked_extract_image_view;LightGlue =
  aicore_lightglue_run_match;BF = Eigen 余弦 + upstream ONNX 网络同语义)。
- 引擎测试落地:feature/aliked_test.cc(N16Rot 真推理,N32 未发布裁定 skip)、
  feature/onnx_matchers_test.cc(BF 6 用例 hermetic + LightGlue 2 真推理)、
  feature/onnx_utils_test.cc(session 契约;Identity/provider 用例无 fork 等价,留因)。
- estimators/solvers/poselib_utils_test.cc 补注册(2/2);exe/sfm.{h,cc} 拆出
  RunPointTriangulatorImpl/RunIncrementalMapperImpl/RunGlobalMapperImpl/
  RunHierarchicalMapperImpl(upstream 签名;取消探针 → .partial 写入语义),exe/sfm_test.cc
  5 用例移植(3 过;IncrementalMapper 2 用例 DISABLED 留因:fork 观测簿记 underflow 缺陷,
  与批次 24 EndReconstructionDiscard 同源)。
- 控制器面:IncrementalMapperController 补 SetCheckIfStoppedFunc/ShouldStop(5 检查点)。
- 修复 PoseLib 链接 PRIVATE→PUBLIC(poselib_utils.h 在接口中暴露 poselib 类型)。
- gate 154/154;pytest 395 passed/3 skipped;边界检查 PASS。

## 批次 B — BundleAdjustmentCeres Ceres 面对齐(2026-09-27)

- `estimators/bundle_adjustment_ceres.{h,cc}` 补 upstream d3ccaf35 的 Ceres-only 面:
  `CeresBundleAdjustmentOptions::CreateSolverOptions(config, problem)`(转发共享基类)、
  `CeresPosePriorBundleAdjustmentOptions` 别名(fork prior struct 为语义超集,注释留档)、
  `CeresBundleAdjuster::BindReconstruction()` + 无参 `Solve()` 返回 shared summary、
  `CreateDefaultCeresBundleAdjuster(options, config, reconstruction&)` 工厂、
  `PrintSolverSummary(summary, header)` 重载。fork 既有 `Solve(Reconstruction*)`
  入口与 Caspar 分派不动。
- **生产语义修复**:BA 取消回调由 `SOLVER_ABORT`(→USER_FAILURE)对齐为
  `SOLVER_TERMINATE_SUCCESSFULLY`(→USER_SUCCESS),取消=保留已接受参数更新的成功收尾
  (upstream 契约,有 bundle_adjustment_ceres_test.Cancellation 断言背书)。
- `estimators/bundle_adjustment_ceres_test.cc` 移植 2 个代表用例
  (NominalMultiCameraRig rig 多相机合成重建精度 + Cancellation);其余 upstream 用例为
  同一 rig/gauge/constant 矩阵的增量变体,fork bundle_adjustment_test 已覆盖共享
  BundleAdjuster 契约,机械跟进即可。
- gate 155/155。

## 批次 C — controllers 新抽象迁移(结构化计划,10 项)

按"接口对齐+适配层"策略逐模块执行,每模块完成即移植其测试并跑全量 gate;
fork 增强面(DA3、global_mapper、RotationAveraging、ViewGraphCalibration CLI)保留为
适配层消费者,行为保持为验收底线:

1. `feature_matching_utils.{h,cc}`(引擎无关 MatchFeatures 入口,复用批次 A 的
   ALIKED/LoMa shell)→ feature_matching_utils_test
2. `matcher_cache.{h,cc}`(DatabaseMatcherCache,复用 DatabaseCache 面)→ matcher_cache_test
3. `pairing`(librelay 风格 pairing)→ pairing_test
4. `undistorters.{h,cc}` → undistorters_test
5. `base_option_manager.{h,cc}`(OptionManager 保持 CLI exit 语义)→ base_option_manager_test
6. `feature_extraction`/`feature_matching` 控制器接口对齐 upstream(消费 1/2 的新面;
   app/reconstruction 与 exe/feature 调用点同批适配)→ 2 个测试
7. `incremental_pipeline`/`hierarchical_pipeline`/`automatic_reconstruction` 接口
   对齐(GUI 4 文件 + DA3 调用点同批适配)→ 3 个测试
- 前置修复建议:fork 观测簿记 underflow 缺陷(observation_manager deregistration,
  已 3 处测试 DISABLED 留因)——在批次 C 触及 controller 生命周期前先收敛该缺陷,
  可一并解锁 IncrementalMapper 2 个取消用例与 EndReconstructionDiscard。

## 批次 C 进度一 — matcher_cache 对齐完成(2026-09-28)

- **裁定修正**:fork 的 FeatureMatcherCache 此前裁定"无面"是错的——它一直存在于
  feature/matching.{h,cc}(旧位置),本轮把它的接口补齐到 upstream
  controllers/matcher_cache.h 形状:shared_ptr-owning 构造、GetTwoViewGeometry /
  ExistsTwoViewGeometry / UpdateTwoViewGeometry / MaxNumKeypoints /
  GetFeatureDescriptorIndexCache(LRU 惰性建索引)、AccessDatabase 放宽为非 const。
- controllers/matcher_cache_test.cc **19/19 全量移植**(gmock matcher、fork Camera
  访问器、ReadTwoViewGeometries 返回 map 的适配)。
- **三个数据库层生产缺陷修复**(全部有测试断言背书):
  1. `SqliteDatabase::DeleteInlierMatches` 删整行 TVG → 对齐 upstream 仅清 inlier
     数组并保留条目(单条 UPDATE,规避 fork Update 的 delete+write 再入);
  2. `SqliteDatabase::ExistsInlierMatches` 别名为"行存在" → 对齐 upstream
     "行存在且 inlier 非空";
  3. `SqliteDatabase::UpdateTwoViewGeometry` 误调 DeleteInlierMatches(清空后写回
     触发 pair_id UNIQUE 冲突) → 对齐 upstream 全行替换(Delete+Write)。
- `FeatureKeypoint` 补 operator==/!=(database 往返断言依赖)。
- **裁定留档**:fork 的 `util/sqlite3_utils.h` SQLite 错误处理保持 exit 而非 upstream
  的抛 std::runtime_error(fork 调用点未做异常安全审计);database_test 的
  "重复写同 pair 抛异常"契约随 W17.2b 数据库抽象层对齐,暂以注释留因。
- gate 156/156。
- 剩余顺序:C1(feature_matching_utils,依赖的 cache 面已就绪)→ pairing/undistorters/
  base_option_manager → 两个控制器 → 三管道。

## 批次 C 进度二 — feature_matching_utils 对齐完成(2026-09-28)

- 新增 `controllers/feature_matching_utils.{h,cc}`:FeatureMatcherWorker /
  FeatureMatcherController / GeometricVerifierController(upstream d3ccaf35 三类 +
  VerifierWorker/WriteVerifiedTwoViewGeometry),复用 C2 的 FeatureMatcherCache、
  批次 A 的引擎 shell 与 fork 的 JobQueue/Thread。
- fork 适配点:FeatureMatcher::Image 值语义(非拥有 shared_ptr 别名 cache 持有的
  camera;无 pose_prior 槽)、descriptors 走 packed-u8-as-float 桥、SIFT 共享
  descriptor-index 挂接(fork shell 自持 FLANN)与 RANSAC num_threads 断言不适用
  (fork 无并行 RANSAC 旋钮)。
- **packed 桥提取**:`ToPackedU8`/`FromPackedU8` 从 feature/matcher.cc 匿名 ns 提升
  为 feature/matching.h 的 inline 共享定义(两个真实消费者:matcher shell 与匹配
  控制器)。
- `GeometricVerifierOptions` 补 `use_existing_relative_pose` 字段。
- controllers/feature_matching_utils_test.cc **16/16 全量移植**。
- gate 157/157。

## 批次 C 进度三 — pairing 对齐完成(2026-09-28)

- **裁定修正**:fork 的 controllers/pairing.{h,cc} 与 upstream 结构一致(8 个
  PairingOptions + PairGenerator 抽象 + 7 个 Generator,含 shared_ptr<Database>
  与 shared_ptr<FeatureMatcherCache> 双构造),此前"无 pairing 面"的裁定再次过时——
  C3 实际缺口仅测试。
- controllers/pairing_test.cc 移植:16 用例中 **13 过**;3 个 DISABLED 留因
  (VocabTreePairGenerator 两个 + Sequential 的 LoopDetection)——fork retrieval 层
  vocab-tree 查询在共享 VisualIndex 上触发 null rwlock 段错误(gdb 实证
  pthread_rwlock_wrlock(0x20)),属 retrieval 层独立缺陷,修复待 retrieval 层
  reconcile 索引跨线程生命周期。
- gate 158/158。
- 剩余:C4 undistorters → C5 base_option_manager → C6 两个控制器 → C7 三管道。

## 批次 C 进度四 — C4 方案定稿(2026-09-28,执行待下轮)

- **裁定修正(第三次)**:fork 的 5 个 undistorter 全部已实现——image/undistortion.{h,cc}
  (1109 行:COLMAP/PMVS/CMPMVS/PureImage/StereoImageRectifier : Thread,CopyType 枚举),
  此前"无 undistorters 面"裁定再次过时。C4 真实缺口 = **位置迁移 + 签名升级 + 测试**:
  1. image/undistortion.{h,cc} 的 5 类迁至 controllers/undistorters.{h,cc},类名保持,
     签名升级为 upstream 形状(每类一个 Options struct 含 copy_type/
     num_patch_match_src_images/image_ids/jpeg_quality/num_threads;BaseController 基类;
     SetCheckIfStoppedFunc/ShouldStop 取消探针,照 IncrementalMapperController 模式——
     COLMAPUndistorter::Run/Undistort 的 IsStopped 检查点替换);
  2. jpeg_quality 经 Bitmap::SetMetaData("Compression","jpeg:Q")(fork Bitmap 无
     SetMetaData,需补)或裁定留因;
  3. 旧调用点同批改名:exe/image.cc、controllers/automatic_reconstruction、
     texturing_controller(共 3-4 处);
  4. 测试移植(8 用例,含 StopsPendingWork 取消用例——依赖新取消探针)。
- 预估:与 C6/C7 同型的一次迁移轮可完成 C4+C5(base_option_manager 缺口小)。

## 批次 C 进度五 — C4 undistorters 对齐完成(2026-09-28)

- fork 实现类就地改名 `*Impl`(COLMAP/PMVS/CMPMVS/PureImage/StereoImageRectifier,
  5 个调用点同步:exe/image.cc、da3_depth_controller、automatic_reconstruction、
  ui/dense_reconstruction_widget、ui/undistortion_widget)。
- 新增 `controllers/undistorters.{h,cc}`:upstream 签名壳(每类 Options struct +
  BaseController + 同步 Run 驱动 Impl Thread),零实现重复;COLMAPUndistorter 壳
  转发 SetCheckIfStoppedFunc。jpeg_quality 接受留因(fork Bitmap 无 metadata 钩子)。
- **两个生产行为修复**(测试背书):
  1. COLMAPUndistorterImpl 补 SetCheckIfStoppedFunc/ShouldStop(Run 提交循环检查点);
  2. 取消路径对齐 upstream StopsPendingWork 语义:ShouldStop 触发后等线程池收尾并
     **跳过重建重写与 MVS config 输出**(原实现继续全量输出,且 in-place
     UndistortReconstruction 与在飞 worker 存在尺寸竞争——Integration 时序侥幸掩盖)。
- controllers/undistorters_test.cc **8/8 全量移植**(含 StopsPendingWork)。
- gate 159/159。
- 剩余:C5 base_option_manager → C6 两控制器 → C7 三管道。

## 批次 C 进度六 — C5 base_option_manager 完成 + C6/C7 精确裁定(2026-09-28)

### C5 — BaseOptionManager 类层次重构(完成)

- 新增 `controllers/base_option_manager.{h,cc}`(upstream d3ccaf35 形状):通用选项面
  (desc_/五个类型化 registry/project-db-image paths/AddRequiredOption+AddDefaultOption+
  AddDefaultEnumOption+EnumOptionInfo/Parse(bool 返回)/Read(allow_unregistered)/ReRead/
  Write/Check/Reset/ResetOptions/RegisterOptionGroupOnce+HasOptionGroup)。
- **fork 偏差(留档)**:①基类构造不调用 AddRandom/AddLogOptions(派生覆盖必须保持权威,
  基类构造期虚分派会绑死基类版本并污染组守卫);②四个 Add*Options 为 virtual;
  ③log 组省略 log_severity/log_color(fork glog 无 FLAGS_minloglevel/色彩旗标)。
- `OptionManager : public BaseOptionManager`:删除重复面(desc_/registry/三 paths/
  RegisterOption/AddRequired/AddDefault 模板/Parse/ReRead/Write);派生保留 fork 旗标名
  (--log_to_stderr/--random_seed,CLI 兼容)覆盖 AddLogOptions/AddRandomOptions;
  AddAndRegister{Required,Default}Option 变为转发模板(200+ 调用点零改动);
  Reset/ResetOptions/Check/Read 改为 override 组合(base 先行);51 个 exe Parse 调用点
  升级为 `if (!options.Parse(...)) return EXIT_FAILURE;`(upstream 模式);
  MainWindow 改按值+move(基类 NON_COPYABLE);BundleAdjustmentController 的 options_
  改引用(upstream 同);2 处 size_t CLI 选项改 int(base RegisterOption 无 size_t)。
- **两个伴随缺陷修复**:①派生 added_*_ 布尔未初始化(删 ctor Reset() 后暴露)
  → 全部 `= false`;②派生类三 path shared_ptr 遮蔽基类同名成员(空指针段错误)
  → 删除遮蔽声明。
- 测试:`base_option_manager_test` **19/19**(含 enum 选项三态/allow_unregistered 两分支/
  project_path 回环);`option_manager_test` **14/14**(恢复批次 24 裁定的
  ParseUnknownArgumentsFails——Parse 返回 bool 后可进程内验证)。

### C6/C7 — 三管道(精确裁定,替换此前笼统"待迁移")

- `incremental_pipeline_test`(22 用例)与 `hierarchical_pipeline_test`(4 用例):
  接口对齐主体已具备(SetCheckIfStoppedFunc 批次 A 已入 IncrementalMapperController;
  upstream hierarchical 本无取消探针,fork 无缺口)。用例本体为端到端 SfM
  (合成数据→完整增量/分层重建→精度对齐断言),被 **observation_manager 观测簿记
  underflow** 硬阻塞(批次 24 实证同源:EndReconstructionDiscard/FullPipeline)。
  **恢复条件**:修复 incremental mapper 的 deregistration 簿记后,22+4 用例即可按
  既有模式移植。
- `automatic_reconstruction_test`(1 用例):fork 已重构为 DA3 增强面(Automatic-
  ReconstructionController::Options 语义不同)+ PatchMatch GPU 依赖——结构分叉裁定。
- fork 控制器保留 `Thread` 基类(GUI ThreadControlWidget 集成)为故意分叉,
  upstream BaseController+同步 Run 的形状不适用 GUI 路径。

### 本轮最终验证

- 全量构建:`make -k -j4 all` 通过(含 colmap_exe 与 ACloudViewer GUI)。
- C++ gate:`ctest -R "^[a-z_0-9]+/[a-z_0-9]+$" -E "^(feature/sift_test|util/opengl_utils_test)$"`
  **160/160 通过**(首跑 caspar 偶发,复跑全绿)。
- pytest:`python/test/reconstruction/` **395 passed / 3 skipped**;AICore 边界检查 PASS。

## 观测簿记 underflow 根因修复 + C6/C7 测试解锁(2026-09-29)

### 根因(插桩实证,observation_manager 本身无缺陷)

- `:626` 的 THROW_CHECK 是 upstream 同款防御断言,正确报告了配对失衡;真正的根因在
  **fork db round-trip 把 GT pose 写进 frames 表**(fork 扩展,upstream db 不存 pose):
  `IncrementalMapper::BeginReconstruction -> Reconstruction::Load(cache)` 时,
  `AddFrame` 的 upstream parity 规则"带 pose 即注册"把全部 cache 帧**假注册**
  (帧进 reg_frame_ids_,但 Load 的 frames-先-images-后循环顺序使 images 未注册,
  ObservationManager ctor 遍历空的 RegImageIds_ -> **可见对应计费恒为 0**);
  同时 `FindNextImages` 以 `HasPose()` 为"已注册"代理,所有帧被跳过 ->
  "No good initial image pair found"。EndReconstruction(discard) 一 deregister 即下溢。
- **修复**(incremental_mapper.cc BeginReconstruction):Load 前快照
  pre_existing_frames(continue 场景 Read 来的真注册帧),Load 后对**新引入帧**
  `DeRegisterFrame`(撤注册+ResetPose,恢复 upstream"从零注册,注册与计费由 SfM
  配对建立"语义;prior 走 db 的 pose_priors 表,不受影响);
  `EndReconstruction` 对 reconstruction_==nullptr 幂等返回(第二 end/reinit 周期,
  upstream fixture 的 nullable Reconstruction() 守卫在 fork 裸指针面的等价表达)。

### 解锁与移植(26+ 用例)

- `sfm/incremental_mapper_test`:恢复 upstream 用例 9 个(GettersAfterBeginReconstruction/
  EndReconstructionDiscard/EstimateInitialTwoViewGeometry/FullPipeline/
  ResetInitializationStats/ModifiedPoints3D/ObservationBookkeepingAfterInitAndFilter/
  ObservationBookkeepingAfterDiscardAndReinit/RegStatsResetBetweenReconstructions),
  **13/13 全过**(ModifiedPoints3D 裁定过时:fork 已有 Get/ClearModifiedPoints3D)。
- `controllers/incremental_pipeline_test` **首次移植**(22 用例):**7 个活跃用例全过**
  (WithoutNoise/WithPriorFocalLength/MultiReconstruction/ChainedMatches/
  IgnoreRedundantPoints3D/两个 Options 传播用例),13 个 DISABLED 留因:
  - 9 个 camera.cc:159 abort 组:fork synthetic 对 upstream 非平凡 rig(多相机每帧/
    球面/全景/未知传感器/prior 组合)未构造 camera 参数,恢复条件=fork synthetic
    对齐 upstream 的 per-rig camera 模型;
  - WithNoise:顺序相关的 tri 观察计费失衡(num_tri_corrs>num_total_corrs,单跑过,
    第二个簿记缺陷);
  - FixExistingFrames:fork fix_existing_images 路径内部 map::at;
  - DefaultMaxNumIterations:fork 无 -1 sentinel(默认即 25/50)。
- `exe/sfm_test` 3/3(WritesInterruptedReconstruction/DiscardsInterrupt... 恢复,
  后者因 fork controller 的取消探针短路顺序改为 DISABLED 留因)。
- `scene/synthetic_test` 的 MapperFrameWiring 用例随新语义适配(cache 帧加载后
  未注册/无 pose 为预期,接线断言保持)。

### 本轮最终验证

- 全量构建 + **C++ gate 161/161**;pytest 395/3;AICore 边界 PASS。
- hierarchical_pipeline_test 移植留待下轮(同 pipeline 模式)。

## 三缺陷批次 — T1 camera 守卫修复 + T2 定性 + T3 hierarchical 首移植(2026-09-29)

### T1 — 非透视相机的 absolute-pose 分流守卫(完成,解锁 Spherical)

- 根因(fprintf ret-addr 插桩实证):upstream d3ccaf35 的 RegisterNextImage 在
  abs_pose_options 段尾有 `if (!camera.IsPerspective()) { estimate_focal_length=false;
  refine_* = false; }` 守卫——非透视相机(Spherical/EQUIRECT)强制走
  EstimateAbsolutePose 的 **camera-ray P3P 分支**(CamRayFromImg, model-aware);
  fork 缺该守卫, spherical 相机带 estimate_focal_length=true 进 P4PF 分支,
  `PrincipalPointX/Y` 对空 principal-point 索引 `CHECK_EQ(idxs.size(),2)` abort。
- 修复:sfm/incremental_mapper.cc RegisterNextImage 补该守卫(upstream 逐字)。
- **WithoutNoiseSphericalCameras 解锁**,incremental_pipeline_test 活跃用例 7→8。

### T2 — tri 观察计费失衡(定性,修复留待下轮)

- T1 修复后,其余 9 个 pipeline DISABLED 与 hierarchical 4 用例**全部收敛到同一
  断言**:observation_manager.cc:233 `num_tri_corrs <= num_total_corrs`
  (SetObservationAsTriangulated 的重复对应守卫)。
- 特性:非平凡 rig(多相机每帧/全景)与 prior 组合下**必现**;WithNoise 为顺序相关
  (全量 PRNG 状态依赖,单跑过);Spherical 组已被 T1 修复分离出来,证明该缺陷与
  相机模型无关,而是**非平凡 rig 的 track/对应配对**路径(IncrementalTriangulator
  的 AddObservation 或 MergeTracks 对同一 (pair,corr) 重复计费)。
- 恢复条件:修好该配对缺陷后,pipeline 9 个 + hierarchical 4 个用例一次性解锁。

### T3 — hierarchical_pipeline_test 首次移植(22→4 用例,DISABLED 留因)

- 全量 4 用例移植(WithoutNoise/WithoutNoiseAndNonTrivialFrames/
  WithoutNoiseAndPanoramicNonTrivialFrames/MultiReconstruction),编译通过,
  运行被 T2 tri 失衡与 hierarchical_pipeline.cc:159 的 cache/cluster image-id
  一致性 map::at 阻塞,DISABLED 留因(恢复条件同 T2)。
- fork 适配:三分离 options(controller/clustering/mapper)+Thread Start/Wait、
  Get(0) 引用语义、MultiReconstruction 指针对比形态。

### 验证

- 全量构建 + **C++ gate 162/162**;pytest 395/3;AICore 边界 PASS。

## T2 tri 计费失衡根因修复(2026-09-29)

### 根因(插桩实证链)

- `observation_manager.cc:233` 的 THROW_CHECK 是 upstream 同款防御哨兵;Set/Reset/对应图/
  OM ctor 全部与 upstream 逐字等价(规范化 diff 实证)。
- **真根因 = 双轨簿记**:fork 的 `Reconstruction` 保留 upstream 已移除的私有
  `image_pair_stats_`(fork 遗留),其 `FilterPoints3D*`/`FilterObservationsWithNegativeDepth`
  直接 `DeletePoint3D` 只减**遗留账本**;`IncrementalMapper::AdjustLocalBundle` /
  `AdjustGlobalBundle` / `FilterPoints` 三处调用这些遗留过滤(upstream :1083/:1112/:1348
  全部走 `obs_manager_->FilterPoints3D*`),**ObservationManager 账本只增不减**——每次
  local BA 后过滤泄漏 +1,非平凡 rig 场景必然累积溢出。
- 插桩铁证:101 次 SET / **0 次 RST**;同一对应 (3,77)-(4,51) 先后计入 pt3d=70 与 pt3d=194;
  `REPOINT old=invalid` 证明清除未走 OM;CLEAR-SITE 实锤 `Reconstruction::DeletePoint3D`。
- 插桩调试教训再证:一轮 trace 零输出实为 camera.cc 插桩脚本截坏函数体致 BUILD=2 未察觉
  (见记忆),必须显式校验构建退出码。

### 修复(最小调用点对齐,不动 fork 遗留数据层)

1. `IncrementalMapper::AdjustLocalBundle`:`reconstruction_->FilterPoints3DInImages/FilterPoints3D`
   → `obs_manager_->FilterPoints3DInImages/FilterPoints3D`(upstream 逐字)。
2. `IncrementalMapper::AdjustGlobalBundle`:`FilterObservationsWithNegativeDepth` → OM 版。
3. `IncrementalMapper::FilterPoints`:`FilterAllPoints3D` → OM 版。
4. `controllers/bundle_adjustment.cc`:`ObservationManager(*reconstruction_).
   FilterObservationsWithNegativeDepth()`(upstream :32 逐字;临时 OM 无 graph,仅清数据)。

### 效果与遗留

- tri 守卫触发数:从每次必现 → **0**;pipeline 全量含 DISABLED 无 abort(RC=134 → 1)。
- 剩余 10 个 DISABLED(9 pipeline + 4 hierarchical 中的对应项)为**各自独立缺口**:
  非平凡 rig 点覆盖(231 vs 100)、模型分裂(Size 2 vs 1)、FixExistingFrames 测试体直调
  触发遗留 stats map::at、EffBa* 缺面——裁定注记已逐一更新为 post-T2 状态。
- **曾尝试一次性下线 Reconstruction 遗留簿记并回退**:该移除使 ChainedMatches 回归——
  fork db round-trip 世界的 pipeline 依赖遗留 SetUp bulk loop 为 db 载入状态维护 image 级
  可见性计数(Inc/Dec 探针实证 500/500 配对但读取为 0,图像对象被替换)。遗留层的退役须
  与 Load 语义(载入时是否建立可见性)一并设计,单删会破坏 pipeline。

### 验证

gate 162/162(100%);pytest 395/3;AICore 边界 PASS。
## T2 后续簇分诊修复 — graph 所有权 / EffBa* sentinel / UnknownSensor 守卫(2026-09-30)

T2 修复后对剩余 10 个 DISABLED 逐一重新分诊,本轮修复 3 个独立簇并解锁 3 个用例
(pipeline 活跃 8 → 11),另修 1 个生产级悬挂指针缺陷:

### 1. Reconstruction 悬挂 graph 回指(生产缺陷,插桩+try/catch 实证)

- `Reconstruction::correspondence_graph_` 是**裸指针**,指向 controller 成员
  DatabaseCache 内部的 graph;controller 析构后悬挂——FixExistingFrames 的测试体直调
  `DeRegisterFrame(1)` 走遗留 `ResetTriObservations` 解引用悬挂 graph,表现为
  `_Map_base::at`(flaky:释放内存被复用时可能"正常")。
- 修复:成员与 `SetUp` 形参改 `std::shared_ptr<const CorrespondenceGraph>`
  (DatabaseCache 侧本就是 shared_ptr;upstream 无回指,注释留档 fork 遗留读用),
  mapper :133 传 shared_ptr,3 个测试调用点包非拥有/共享别名。
- 注意:全量 `make all` 时 3 个测试编译错暴露了裸指针消费者——教训:改接口后必须
  `make -k all` 而非只编本目标。

### 2. EffBa* sentinel 迁移(upstream parity,解锁 2 用例)

- `ba_local/global_max_num_iterations` 默认 25/50 → **-1 sentinel**;
  新增 `EffBaLocalMaxNumIterations()/EffBaGlobalMaxNumIterations()`(upstream 逐字:
  >=0 直返,CASPAR→solver_iter_max,否则 kDefaultCeres 25/50,常量入匿名 ns);
  `Local/GlobalBundleAdjustment()` 工厂改用 Eff 解析(生产行为不变);
  `Check()` 的 `>= 0` 同步为 `>= -1`(sentinel 合法)。
- 解锁:`DefaultMaxNumIterationsUsesBackendDefaults`(此前因默认非 sentinel 的
  ASSERT_EQ(-1) 失败)+ `EffBaMaxNumIterations`(skip→真 port,fork 断言落在合并
  options 面无 per-backend 槽)。

### 3. UnknownSensorFromRig 优雅退出(upstream parity,解锁 1 用例)

- 控制器 trial 循环 Begin 后补 `HasUnknownSensorFromRig` 检查(upstream :99/:534
  逐字;LOG(ERROR)+EndReconstruction(discard)+Delete(idx)+break)→ Size()==0。
- 测试适配:fork 的 `NonRefSensors()` 返回**值**(内部是 optional<SensorPose> 非
  optional<Rigid3d>),upstream 返回引用——测试的 `.reset()` 打在临时副本上无效;
  改用 fork 既有的 `SetSensorFromRig(id, nullopt)`(inline 重载→ResetSensorFromRig)。

### 剩余留因(更新)

- **FixExistingFrames**:悬挂已修后暴露次级根因——测试直调 `DeleteObservation`
  递减 fork 遗留 `image_pair_stats_`,映射期新建 pair 无条目(operator[] 建 {0,0}
  后递减回绕为无符号巨值)触发遗留守卫 abort(reconstruction.cc:1534)。恢复条件:
  遗留 stats 层退役 + Load 重入语义一并设计(该批需为 db 载入状态建立 OM 侧可见性
  初始化,见"遗留层不可单删"契约)。
- **非平凡 rig 簇(4)与 Prior 观测簇(3)**:管线跑通无守卫触发,收敛为重建质量面
  差异(点覆盖 231 vs 100 / 模型分裂 / observations 2000 vs 1900)——两条线同
  上(遗留 stats 层 + Load 可见性)。

### 验证

gate 162/162(100%,活跃用例含 3 个新解锁);pytest 395/3;AICore 边界 PASS。
## bin/ 测试二进制治理 + 单账本迁移第一步(2026-09-30)

### bin/ 目录治理(完成,三平台兼容)

- 治理 CMake(`COLMAP_TEST_RUNTIME_DIR=bin/colmap_test`、`cloudviewer_set_bin_runtime_output_directory`=bin/plugin_test、
  AICore layout=bin/aicore_tests)早已在工作树,但 build_app 从未重配——bin 根残留 176 个旧测试二进制。
- 本轮重配重建:**164 个 reconstruction 测试 → bin/colmap_test/**,**13 个插件/helper 测试 → bin/plugin_test/**,
  AICore probe 归位 bin/aicore_tests/,9 个孤儿产物(对应 target 已改名/移除)删除;bin 根只剩主程序、工具、共享库与目录。
- **运行时解析修复**(归位暴露,全部实证):①COLMAP_ADD_TEST 补 POSIX `LD_LIBRARY_PATH` ENVIRONMENT + `BUILD_RPATH/INSTALL_RPATH`
  (Qt 从 Qt5::Core 动态解析 + external/lib)+ `--disable-new-dtags`(DT_RPATH 传递覆盖间接依赖,如 libAICore 的 opencv);
  ②新增共享 `cloudviewer_set_plugin_test_env[_offscreen]()`(注意:set_tests_properties 的 ENVIRONMENT 是覆盖语义,
  QT_QPA_PLATFORM 等必须合并进同一值);③aicore layout 同款 DT_RPATH + ENVIRONMENT(系统双 Qt 时保证加载构建所用 Qt)。
- 验证:gate 162/162(全部从 bin/colmap_test/ 运行);插件/helper 12 项全过;pytest 395/3(test_parity_batches 的
  ImportError 是并行会话新文件读 shell 环境所致,非本任务范围);AICore 边界 PASS。提交 a64e3b86e1/c11da90498。

### 单账本迁移第一步(读点切换)

- 实证:fork mapper 注册门槛(:326)与控制器 "Image sees" 打印读 **Image 层遗留计数**(OM-era 三角化不维护它,恒过时);
  upstream 读 OM。mapper 补 upstream `ObservationManager()` 访问器,两处读点切换到 `obs_manager_->NumVisiblePoints3D`。
  提交 202b243bad。遗留层剥离(数据+SetUp bulk loop+Image 层计数器)为下一步,预计解锁 FixExistingFrames/非平凡 rig/Prior 簇。

## loma/aliked ggml 替代 ONNX 的一致性评估(2026-09-30)

### 已确认(有 gate 证据)

- **模型层数值 parity**:AICore 已建立 reference-gate 体系,golden 明确来自 pinned upstream ONNX 图输出
  (test_loma_descriptor_reference_gate.cpp 头注释:"fixture contains ... from the pinned upstream ONNX graphs")。
  实测:`test_aliked_capi_parity_q8` 真推理 21.3s **通过**;`test_loma_default_triplet / test_loma_matcher_e2e /
  test_loma_published_model_gate / test_lightglue_aliked_e2e` 4/4 通过;`test_loma_g_backend_parity`(CPU/GPU 后端间
  parity)在注册清单。loma 的 detector/descriptor/matcher/b128_e2e reference gate 因 fixture 资产未下载而 skip(合规,
  资产由 aicore-validate-all 机制提供,下载后即可补全数值证据)。
- **端到端功能**:批次 A 接线后 feature/ 层 aliked_test(N16Rot 真推理)、onnx_matchers_test(LightGlue 真推理)在
  reconstruction gate 内运行。

### 未确认(诚实结论)

- **端到端重建精度一致性**(onnx 管线 vs ggml 管线在同一真实数据集上的 track 数/重投影/配准指标对照):未做。
- **速度一致性**(受控 A/B:同硬件、同模型、同输入、同采样协议):未做,且仓库无 onnxruntime 运行时,无法直接 A/B;
  需一次性基线(上游 colmap 构建 onnx 管线或 pycolmap-onnx 环境)后按 gate 的 A/B 规程执行。
- 结论:**模型层精度一致性有 gate 证据(loma 部分待 fixture 下载补全);端到端重建精度与速度一致性未确认,
  是独立的验证工程,不应宣称已对齐**。

## 遗留层剥离 + Load 可见性协同(2026-09-30/10-01)

### 剥离内容(单账本收敛完成)

前置:读点切换(202b243bad)消除了"遗留层不可单删"契约的前提——Image 层计数的
读者已全部切 OM;本轮正式退役 upstream d3ccaf35 已删除的 fork 遗留簿记层:
- scene/reconstruction.{h,cc}:ImagePairStat/image_pair_stats_/SetObservationAsTriangulated/
  ResetTriObservations/SetUp/Get·HasCorrespondenceGraph/NumImagePairs·ImagePair·ImagePairs·
  ExistsImagePair 访问器、FilterPoints3DInImages·FilterAllPoints3D(全部调用点已切 OM)、
  exe/sfm.cc 的孤儿 FilterAllPoints3D 调用点(切 OM + FilterPoints3DWithShortTracks,
  对齐 upstream point_filtering,顺带消除手工 DeletePoint3D 绕过账本);graph 所有权
  移交 OM(OM 构造持 shared_ptr,mapper 注释更新)。
- scene/image.{h,cc}:num_visible_points3D_/num_correspondences_have_point3D_/point3D_visibility_pyramid_
  三成员、SetUp/TearDown/Inc·Dec/NumVisiblePoints3D/Point3DVisibilityScore/IsPoint3DVisible/
  kNumPoint3DVisibilityPyramidLevels 全部删除(生产零调用,OM 持有唯一账本;exe/image.cc
  打印切 mapper.ObservationManager())。
- image_test.cc:TestNumVisiblePoints3D/TestPoint3DVisibilityScore 删除(OM 对应物已在
  observation_manager_test.cc :486+/:661+ 覆盖);两个 BA 测试的 reconstruction.SetUp() 调用删除。
- incremental_triangulator.cc Retriangulate 改读 obs_manager_->ImagePairs()(upstream parity)。

### 教训:静态库不解析符号,误删静默通过

回填脚本("return num_filtered;"→RegFrameIds 锚点)曾把 FilterPoints3D 尾部之后的
约 500 行正常函数(FilterImages/Compute*/Read·Write IO 全家/FilterPoints3DWith* 等 20 个)
一并删除——ColmapLib(静态库 archive)**不做符号解析**,编译零报错静默通过,直到测试
目标链接才以 undefined reference 暴露。修复:从 git HEAD 提取 :1048-1503 回填 20 个函数,
再以"定义集合 diff 恰好等于有意删除清单"做对账收口(MISSING=FilterAllPoints3D/
FilterPoints3DInImages/Set/Reset/SetUp,ADDED=空)。**净删对账应成为大范围删除的标准收口步骤。**

### 验证(解锁 3 个用例)

- FixExistingFrames:回绕 abort 消失(legacy guard 随层退役),正式解锁(去 DISABLED)。
- WithNoise、PriorBasedSfMWithRandomSeedStability:跑通,解锁。
- 非平凡 rig 簇(3)+Prior 簇(3)+GPSPrior(1) 共 7 个保持 DISABLED,失败形态为
  数值型(reconstruction_manager Size 2!=1 模型分裂/点覆盖差距),非簿记 abort——
  与遗留层无关的深层重建质量差距,留因注记已是准确描述。
- image_test 21、OM 14、BA 11+5、alignment 9 单测全绿;gate 162/162(100%)。

## loma reference gate fixture 补全(数值证据落地,2026-10-01)

- 机制核查:reference gate(test_loma_{detector,descriptor,g_token,matcher,b128_e2e}_reference_gate)
  依赖 AICORE_TEST_LOMA_* 环境变量提供 GGUF+fixture,validate_all/manifest 无对应场景行,
  catalog 亦无 fixture 条目——默认全 SKIP,数值证据缺口是结构性的。
- fixture 生成链路(仓库已有工具,此前从未跑通):loma_sources.json(pinned ONNX URL+sha256,
  github davnords/storage loma release)→ pip onnxruntime/onnx → 下载 detector_B/descriptor_g/
  matcher_B 三个 ONNX(sha256 全校验)→ export_loma_descriptor/detector_reference.py
  (784×784 固定图 examples/test_data/image/00000.png)→ export_loma_g_trace_model+
  g_token_reference → matcher_reference(seed 20260903)。产物在
  ~/cloudViewer_data/extract/loma_references/。
- **数值证据(G 线 4/4 PASS)**:detector gate(集合级配对)、descriptor gate(10.65s,
  严格逐索引)、g_token gate(7.79s)、matcher gate;score 集合误差 ~2.5e-8——
  ggml 图与 pinned upstream ONNX 数值一致。
- **detector gate 设计修复**:原实现按输出索引逐点比对(0.02px),对 top-k 检测器
  在近平局处要求两种实现排序完全一致——数学上不成立(分数一致仅证概率图一致,
  不证 top-k 次序)。改为贪心最近邻配对(≤2px)+配对分数误差,precision/recall≥0.995。
  固定输入的 descriptor/token gate 严格逐索引仍通过,佐证失败确为次序平局而非数值错误。
- 留项:b128_e2e gate 需 B 线 GGUF(descriptor_dedode_b/matcher_B128,catalog 有
  URL/digest,下载后按同流程生成 fixture 即可);reference 场景仍未纳入 validate_all
  矩阵(需 manifest 场景行+input_assets 登记,属独立批次)。

## onnx 端到端 A/B 基线规划(待执行)

目标:证明 ggml 接线在**重建管线末端**与 onnx 参考等价(此前仅有模型层证据)。

1. **基线环境**(一次性):pyenv/venv + onnxruntime(≥1.23,已装)+ pinned ONNX
   (loma_sources.json;aliked/lightglue 对应来源待登记入同格式 sources.json)+
   固定图像集(examples/test_data 与 loma gate 的 784×784 图)。
2. **精度 A/B 三层**:
   - L1 模型层:reference gate(已完成,本轮补全数值证据);
   - L2 特征层:同一图像集上 onnx 管线 vs libAICore 产出对比(descriptor 余弦相似度
     ≥0.999(f32)、match 正确率、对内点数),脚本落 core/AICore/tools/(复用
     export_*_reference.py 的加载代码);
   - L3 重建层:同一 COLMAP 重建流程(ACloudViewer point_triangulator / reconstruct
     auto)分别喂 L2 两路特征,比较 registered 数/重投影误差/track 长度/点数。
3. **速度 A/B**:validate_all 协议(同主机/同输入/warmup+p50/p95/≥3 重复进程);
   onnxruntime CPU 为基线,ggml 按 backend(CUDA/Vulkan/CPU)分别对照;仅在同一
   backend/量化对内声明倍数。
4. **门禁**:L3 registration 数差 ≤5% 且重投影同量级;不满足→回到 L2 定位。
5. **依赖**:B 线 GGUF 下载、aliked/lightglue 的 sources.json 登记、manifest 场景行
   (L2/L3 probe 进 validate_all 矩阵)——与上面留项合并为一个独立验证批次。

## 批次 D:controller 选项面 frame 级对齐 + 分裂根因取证(2026-10-01)

### 选项面移植(upstream parity 补齐)

对照 upstream IncrementalPipelineOptions 全字段 diff,fork 缺失面移植到
IncrementalMapperOptions:
- `ba_refine_sensor_from_rig`(默认 true):显式传入 Local/Global BA options——此前
  fork 依赖 BA options 默认值 true 行为等价,但两变体测试与 CLI 面缺失;现与
  upstream :178/:221 同路径,并注册 `Mapper.ba_refine_sensor_from_rig`;
- `constant_rigs`/`constant_cameras`(NodeHashSet,与 mapper Options 同型):传入
  mapper Options;BA config 消费链(SetConstantSensorFromRigPose/SetConstantCamera,
  mapper.cc :622-:799)fork 已完整存在;
- `random_seed`:传入 mapper(triangulator);fork mapper 已消费于 abs_pose RANSAC
  (:425)与 CHECK(:97)。
- 命名滞后项(fix_existing_frames vs fork fix_existing_images、ba_global_frames_ratio/freq
  vs images_ratio/freq、snapshot_frames_freq vs snapshot_images_freq)为 frame 级改名
  遗留,语义等价,另批处理(CLI 面已绑定旧名,改名牵连 exe/app)。

### 分裂根因取证(两个假设证伪,新证据链)

- 假设 1 证伪:"fork 无 per-sensor-from-rig BA refinement"——批次 B 已对齐 BA 层
  (bundle_adjustment.h :205 + ceres rig 参数块),controller knob 移植后行为不变;
- 假设 2 证伪:synthetic per-rig 构造差异——fork synthetic.cc 的 sensor_from_rig
  生成与 upstream 逐字一致(Z 轴旋转 clamp + 平移高斯);
- 新证据(WithoutNoiseAndWithNonTrivialFrames 实测日志):
  ①首个 local BA **initial cost ~80px**(无噪声数据,upstream 应为亚像素级起点);
  ②Iterations 201/No convergence 后 final 0.0099px,但紧随的过滤阶段
  **丢弃 99/100 观测** → 模型死亡 → 开新模型 = fragmentation 机理;
  ③ConstantRigs 用固定 rig 后分裂更重(Size 3)。
- 结论:根因收敛到 **two-view 初始化/三角化在非平凡 rig 下的系统性错误**
  (初始位姿或三角化点严重偏离,BA 参数化仍能拉回低残差但过滤阶段按真实几何
  阈值剔除),下一批需插桩对比 init 两图的 relative pose/三角化点 vs GT。
- 验证:gate 162/162;三用例保持 DISABLED,注记更新为本证据链。

## 批次 E:非平凡 rig 分裂根因修复 + 尺度锚定移植(2026-10-01)

### 三轮插桩取证(假设逐层证伪到实锤)

1. init 位姿写入探针(RegisterInitialImagePair 尾):5 次 init 全部
   cam1_is_identity_err=0、cam2 rot_err≤1.7e-6°、t_err≤8.9e-16 ——
   **SetCamFromWorld/Frame 分解组合序无缺陷**(假设证伪);
2. filter 分类探针(OM FilterAllPoints3D):**reproj=99、tri_angle=0**
   ——先前 min_tri_angle_deg=85.94 的读数是诊断打印自身二次换算的误读
   (OM 内部已 DegToRad,传入即度);reproj 是唯一杀手;
3. 三角化点级探针(TriangulateImage 后、BA 前):重投影 mean 1.6e-05px、
   focal=1280 正常 —— **三角化也完美**;但同一 run 的 BA report 报
   initial cost 80px —— **BA 参数块初始值 ≠ reconstruction 当前几何**。

### 根因实锤:BA SetUp 的反向 dual-track 同步

`CeresBundleAdjuster::SetUp` 的 W3-2b step4 把 **image 的 legacy qvec/tvec
镜像反写回 frame**(SetCamFromWorld(image.Qvec(), image.Tvec()))。fork 的
frame 级注册下,同一 frame 的**非 ref image** 的镜像从未在 init/Register
时被重写(仍持 synthetic load 时的 GT 位姿),反写即用 GT 世界系的位姿覆盖
init 估计世界系的位姿 —— **世界系撕裂** → BA initial 80px、撕裂起点下
BA 不收敛到全局一致 → filter 丢 99 → 雪崩。upstream 无此段(参数块直接
用 frame storage 就地优化,零写回零同步)。全部失败用例恰为
num_cameras_per_rig≥2(存在非 ref image),与现象完全吻合。

修复(bundle_adjustment_ceres.cc):frame 权威位姿存在时绝不接受镜像反写,
仅允许给无位姿 frame 播种(`frame.HasPose()` 守卫)。

### 尺度锚定移植

修复后模型不再分裂但尺度漂移 20%(scale 0.80)——upstream 的
`AlignReconstructionToPriorsOrRigScale`(pipeline 收尾:prior 对齐失败/未启
用时 fallback 到 AlignReconstructionToOrigRigScales,fork 的依赖函数
alignment.cc 已有但从未在 pipeline 调用)移植到 fork 的两个 keep 分支
(ShouldStop + 正常完成)。

### 效果与剩余留因(更新)

- **结构性缺陷清零**:不再有分裂(Size 2/3)、不再有 abort、BA initial
  cost 80px→8.6e-06px、scale 0.80→阈值内;
- 7 个 DISABLED 的失败形态全部变为 **seed-dependent 观测计数方差**
  (2600 vs 2800 / 2601 / 1901 vs 2000,跨 trial 波动,RANSAC seed=-1);
- 3 个用例曾短暂解锁但实测 flaky(ConstantRigs 1/3 挂、Panoramic 2/2 挂)
  ——按 gate 稳定原则回退 DISABLED,注记更新为方差形态;
- gate 162/162 ×2 次 + pipeline 默认跑法 ×3 全绿;
- **下一批入口**:trial 间 ~200 观测的过滤方差取证(统计化:多 trial 的
  filter 命中分布 vs upstream 同分布),而非继续逐层静态 diff——静态面已
  全部对齐。

## 批次 F:过滤方差统计化取证 + doc 清理(2026-10-01)

### 方差取证(独立进程 vs 全量跑的对照实验)

- **独立进程 ×20**(每买 shell 循环新进程跑 DISABLED_WithoutNoiseAndWithNonTrivialFrames):
  **20/20 全部 OK**(2800/2800)——单用例在干净全局态下完全稳定;
- 全量顺序跑时挂的用例每次不同(ConstantRigs/Panoramic/WithoutNoise 轮流挂,
  观测 2600/2601/1901 不等)——**方差源 = 进程内全局 PRNG 残留态**:前序用例
  消耗 PRNG 序列 → 后续用例的合成数据/RANSAC 采样不同 → 部分序列下
  2 帧的观测被 reproj 过滤清除;
- upstream 同样共享 PRNG 残留态(两边的 synthetic.cc 均为
  `if (PRNG == nullptr) SetPRNGSeed();`,fork :366 逐字存在——先前
  "fork 丢 SetPRNGSeed" 的判断是 grep 相对路径在错误 cwd 下零命中的误报)
  ——**upstream 对任意 PRNG 序列产的数据鲁棒,fork 在部分序列下丢 2 帧观测**,
  这是剩余的真实质量差距;
- **附带发现**:gtest_repeat≥2 时第 2 迭代必崩 `_Map_base::at`(跨迭代
  全局态污染,独立缺陷,单跑从不复现)——下一批候选之一;
- 结论:方差取证完成定性——**不是簿记/几何 bug,是特定数据序列下局部 BA
  不收敛(12.6→0.525px No convergence 案例)致 2 帧位姿劣化被 filter 清除**。
  下一批入口:固定 PRNG 态复现 2600 案例(临时 debug target 扫 SetPRNGSeed(k))
  + BA 前后帧级位姿对比,定位不收敛帧与 upstream 的行为差。

### 批次 G 补充:viewer 裁定回滚 + 方差二轮定性(2026-10-01)

- **viewer/pycolmap 删除裁定回滚**(用户质询命中):upstream 的 3D Viewer
  是纯前端静态页(viewer_src TS → vite → 嵌入文档站),**不是 CLI 能力**——
  upstream 全仓也无 `viewer` 命令(与 fork 命令面一致);pycolmap 文档是
  python 对齐 TODO 的目标态描述。全部恢复,doc 树回到与 upstream 逐字节
  一致(对账 UP/FORK/DIFF 全 NONE);仅保留 3 个真 FORK-only 删除
  (Colmap.desktop 旧启动脚本变体 + 2 张旧 png)。
- **方差二轮定性**(前缀组合扫描,零编译):同二进制、同用例顺序、同 PRNG
  前缀下 PriorBasedSfMWithoutNoise 10/10 PASS(含完整 13 前序组合)——
  PRNG 假设降级;全量挂仅发生在 ctest -j4 并行下 → **方差 = ceres 多线程
  位级抖动 × fork 在部分 PRNG 序列下贴近 filter 阈值边缘(质量余量不足)**;
  upstream 同为全核但质量余量大,故稳定。
- 新增 controllers/pipeline_rng_scan_test.cc(TEMP-DIAG,DISABLED_,不入
  gate,gate 命令需 -E controllers/pipeline_rng_scan_test):扫
  SetPRNGSeed(k) + options.random_seed=k 记录观测数;编译暂被并行会话的
  ggml patch 迭代阻断(ext_ggml 源与 patch 不同步),ggml 恢复后执行。
- 教训:删除同源快照件前必须验证 upstream 侧真相,而非只验证 fork 侧
  缺失——对齐工程中「fork 没有 X」不蕴含「应删除 X 的文档」,先判定 X
  在 upstream 的形态(能力 vs 文档资产)。

## 批次 H:三角化器第二处镜像读取面实锤与修复(2026-10-01)

- 扫描基建落地:ggml 收敛后 pipeline_rng_scan_test 编译成功,k=1..30 扫描:
  修复 E 后结构性稳固(models=1/14 帧全注册 ×30),但发现**新指纹**:
  seed 4→304 点、11→203、13/14→200(GT=100)——**同一物理点被拆成多个
  point3D**(track 碎化,直方图双峰 len=16×95 + len=6×195,模式 28=16+6+6)。
- SeedDetail 探针(seed=4,按 2D 观测集合配对 GT↔fork track):
  **gt_split=100/100**——每个 GT 点的观测集都被 fork 分到 ≥2 个 point3D;
  未对齐 XYZ 距离 1.3~2.9 系坐标系差异(首次实验设计缺陷:须先 Sim3 对齐)。
- Continue 角误差插桩:被拒候选全部来自 ref_img=24,**角误差 37°~50° vs
  阈值 5°**——非边缘拒绝,是 **image 24 的 legacy qvec/tvec 镜像 stale**
  (注册后 local BA 窗口外优化改变点/位姿,镜像只在 config 内帧刷新;
  upstream 用 CamFromWorld() 实时组合永不过期)。
- 修复(upstream parity):
  ①scene/projection.{h,cc} 移植 `CalculateAngularReprojectionError(Rigid3d)`;
  ②incremental_triangulator.cc Continue 改读 `image->CamFromWorld()`;
  ③incremental_triangulator.cc Complete 的 reproj 过滤同步改
  `image.CamFromWorld()`(upstream Merge/Complete 同款)。
- 效果:**SeedDetail gt_split 100→0、gt_fully_single 0→100**;gate 162/162;
  但修复改变管线轨迹后暴露下一层:2600×4、2400×1、seed 14 分裂回归(models=2)
  + seed 13/14/15 段大量 **ceres "Linear solver failure: dense Cholesky
  factorization"**(BA 无效步,分裂/观测丢失的直接机理候选)。
- 下一批入口:①Cholesky failure 的触发条件(哪个 BA/哪些参数块——norm
  退化/零列?与 upstream 对照);②剩余 2600/2400 的 seed 分段追踪;③
  gtest_repeat at 崩溃仍待独立排查。

### doc 目录对账与清理(与 upstream d3ccaf35 逐文件对账)

- 对账结果:69 个共同文件**逐字节一致**(fork doc = upstream doc 快照);
  FORK-only 遗留 3 件(Colmap.desktop 大小写变体、2 张 upstream 已换 webp
  的旧 png)→ 删除;
- "本地对不上"裁定(逐一核实 fork 实际能力面):fork Colmap exe 无
  `viewer` 命令(与 upstream cli.rst 一致均无)、无 web viewer 前端、
  python 模块名为 `pybind`(cloudViewer 包)非 pycolmap → 删除
  viewer.rst/viewer_src//tests/(playwright)/vite/package/playwright/tsconfig
  /pycolmap/ 全家及 _static/viewer-licenses.txt;
- 同步更新引用:index.rst 去掉 PyCOLMAP grid card 与 viewer/pycolmap
  toctree 行、conf.py html_sidebars 去 viewer 键(带 fork 注记)、
  gui.rst 去掉 :doc:3D Viewer 断链段、install.rst 去掉 PyCOLMAP :ref: 断链;
- 保留:其余 rst(与 upstream 逐字一致且描述的 CLI/GUI/格式/相机/rig
  能力 fork 全部存在,实测 Colmap help 命令面与 cli.rst 对齐)、
  COLMAP.desktop/colmap.1/sphinx 基建。

## 批次 I:第三处镜像读取面实锤修复 + 三缺陷裁定 + 全仓对齐扫描(2026-10-01)

### 缺陷 1:观测缺口与分裂回归 → 根因修复(第三处镜像消费)

批次 H 修复改变管线轨迹后,PRNG 扫描(k=1..30,单线程确定性验证)暴露
8/30 seed 退化:2600/2400 观测缺口、seed 5/14/28 分裂(models=2)。
分段追踪(管线四断点 + OM filter 分类 + 双账本口径 + local BA 前后
per-image RMS)逐层收窄:

- 缺口是**帧级全灭**:撕点在 image #5/#26 注册后的 local BA——
  abs-pose 8.5e-06px 完美 + Continued 100,ceres initial_cost 却 211px;
- **量级矛盾**直指参数块初值:track 扩展(`AddPointToProblem` 对 config 外
  image)建 constant-pose residual 时用 **stale 双轨镜像 image.Qvec()/Tvec()**,
  upstream d3ccaf35 用**实时 rig-aware 组合**(ref: RigFromWorld;
  非 ref: SensorFromRig*RigFromWorld)→ 起点数百 px → ceres 发散
  (worst RMS 达 1e154px)→ filter 屠杀 → 计数缺口/模型死亡;
- **修复**(bundle_adjustment_ceres.cc):track 扩展 constant pose 改读
  `image.CamFromWorld()`——与批次 E(SetUp 反写)/H(Continue/Complete)
  构成镜像缺陷三连修,全部同族(镜像窗口外 stale);
- **效果**:扫描 30/30 全部 models=1/14 帧/100 点/2800 观测,与 upstream
  d3ccaf35 同实验(同参数 up 探针 30/30 完美)**完全对齐**;
- **解锁 4 个 DISABLED**:WithoutNoiseAndWithNonTrivialFrames(两变体)、
  WithNonTrivialFramesAndConstantRigsAndCameras、
  WithoutNoiseAndWithPanoramicNonTrivialFrames——3 次重复全绿(17 PASSED)。

### 缺陷 2:Cholesky failure → 良性瞬态裁定

单线程确定性下 Cholesky failure 仅出现在 **seed 26/27(完美 seed)**,
与异常 seed 零关联——ceres trust-region 中间迭代的病态步瞬态(内部防御
路径工作正常,终态 Convergence)。批次 H 的「13-15 段大量失败」是日志
切段污染造成的误关联。**非缺陷**。

### 缺陷 3:gtest_repeat at 崩溃 → 机制取证 + 防御加固

- 崩溃点:ExtractColors WARNING 之后的下一 map at()(二进制 stripped,
  原子帧无符号);机制 = 跨迭代全局态(残留 PRNG 序列)产生
  point2D→point3D 悬挂引用;
- 30-seed 单线程扫描(含 ExtractColors 全开)**零悬挂**——需 repeat 特有
  序列才触发;
- 加固:ExtractColorsForImage 对悬挂引用防御性 skip(不再 at() 崩);
  DeletePoint3D 清理空循环残骸;
- **留因**:悬挂引用的原子产生点未定位(需 -g 重编 repeat 复现);非 gate
  面(gate 不用 repeat)。

### 全仓上游对齐扫描(逐文件 diff + 函数签名对账)

- 605 upstream 文件:575 内容 diff(含版权头差异)、30 upstream-only、
  67 fork-only;**P0 = 0**(编译/链接/gate 全绿);
- upstream-only 等价裁定:scene/rig(→fork scene/rig.h)、matcher_cache
  (→feature/matching 内联)、ceres_loss_function(→bundle_adjustment_ceres.h
  内联)、solvers/utils(→estimators/utils)、onnx_*(→AICore ggml 有意替代)、
  util/testing 等基建(fork 自有测试基建);
- **P1 清单(真缺失,行为面)**:
  1. **prior-position BA 残差路径**(CreatePositionPriorCostFunction/
     AddImagePosePriorToProblem/PosePriorBundleAdjuster ceres 版)——fork
     prior 只做收尾 Sim3 对齐(批次 E),upstream 在管线内 BA 约束——
     **3 个 Prior 用例 1900/2000 缺口的根因候选**;
  2. **two_view_geometry degenerate-check 家族**(upstream 23 处
     OrDegenerate/Check*,fork 0)——鲁棒性路径;
  3. SolveWithGpuFallback(GPU BA fallback,fork 无 GPU BA 面);
  4. SolveTriangulationDlt/LeftJacobianFromAngleAxis(数值路径升级);
  5. MaybeMarkWatermark 细节/RescaleToMaxImageSize/feature 面杂项。
- P2:其余(等价改名/平台特定 Win/CUDA/util 下载面/ui 自有实现/
  fork-only 扩展 67 文件有意保留)。

### 附带发现与基建状态

- caspar_test 的 SplitIntrinsic 子用例存在 **GPU warmup 顺序依赖**
  (单跑/并行坏值 63.9px,全量顺序跑过)——先前批次遗留 flaky,与本轮
  无关(修复路径未触及该 fixture);rerun 即过,gate 默认顺序满足;
- pipeline_rng_scan_test(TEMP-DIAG,gate 排除)保留:seed 扫描 + SeedDetail
  track 配对 + 悬挂引用审计三个探针,num_threads=1 固定确定性;
- upstream 对照实验基建:upstream build 树可编 main() 探针(用完即删,
  本次已清理)——后续对齐裁定可复用此通道。

## 批次 J:prior gauge/Normalize 语义对齐 + 冗余点第二段 BA 移植(2026-10-02)

### 全仓扫描 P1 清单更正

批次 I 函数名对账把「prior-position BA 残差路径」列为缺失系**假阳性**——
fork 用不同接口组织完整实现了该路径:`CeresBundleAdjuster::SetPosePriors`
注入 + SetUp 内 AlignReconstructionToPosePriors 播种(对齐 upstream
PosePriorBundleAdjuster 的 BA 前对齐)+ AddImageToProblem 注入
CreatePosePriorLossFunction 位置残差。从 P1 清单移除。

### 真实差异:AdjustGlobalBundle 的 gauge/prior/Normalize 语义

逐段 diff upstream d3ccaf35 IncrementalMapper::AdjustGlobalBundle 发现
三处 fork 行为差(3 个 Prior 用例 1900/2000 缺口的根因):

1. **双重 gauge 破坏 prior 语义**:fork 旧式无条件
   `SetConstantPose(reg0)+SetConstantTvec(reg1,{0})` 在 use_prior_position
   分支也生效——prior 拉住的位姿被硬固定对抗;upstream 非 prior 分支用
   `FixGauge(TWO_CAMS_FROM_WORLD)`(两相机启发式更稳、步数更少),prior
   分支**不 FixGauge**(prior 自身定 gauge)。fork 的 FixGauge ceres 侧
   基建(FixGaugeWithTwoCamsFromWorld/ThreePoints)早已完整,只差接线;
2. **AdjustGlobalBundle 内无条件 Normalize()**:prior 锚定的 metric 尺度
   每次 global BA 后被重缩放破坏——upstream 无此 Normalize(仅在 pipeline
   init 序列与 IterativeGlobalRefinement 的 !use_prior_position 条件下);
3. **冗余点第二段 BA 缺失**:upstream 第一 solve IgnorePoint 冗余点后,
   再以全帧 SetConstantRigFromWorldPose+ConstantCamera+
   ConstantSensorFromRigPose 的 config **专门第二 solve 优化冗余点**;
   fork 忽略后不管(冗余点永不优化;synthetic 均匀覆盖下冗余集近似空
   故 30/30 未暴露,真实数据必差)。

### 修复与验证

- mapper.cc AdjustGlobalBundle 重构:非 prior→FixGauge(TWO_CAMS_FROM_WORLD);
  prior→SetPosePriors(补 alignment_ransac_options.random_seed 传递)不
  FixGauge;删除 mapper 内 Normalize;移植冗余点第二段 BA(check_if_stopped
  守卫 + 重建 config);
- **3 个 Prior 用例全部翻绿**(PriorBasedSfMWithoutNoise/WithNoise/
  GPSPriorBasedSfMWithNoise)——pipeline 22 用例 21 PASSED+1 SKIPPED
  0 FAILED,3 次重复稳定;
- scan 30/30 保持完美(non-prior gauge 方式改变无回归);
- gate 162/162(排除面新增 caspar_test,见下)。

### caspar_test GPU 环境敏感升级记录

SplitIntrinsic 子用例的 parity 数值在 GPU 被并行负载占用时系统性恶化
(Caspar rms 0.29→1.00,params 差 63.9px,容差 5e-2),HEAD 版同样复现
——**非本工程回归**,是 Caspar GPU 后端在 GPU 竞争下归约序漂移使
parity 容差边缘爆。gate 在有并行 GPU 负载的环境需 -E 排除
bundle_adjustment_caspar_test(与 sift_test/opengl_utils_test 同类);
无并行 GPU 负载的环境(如 CI)仍应纳入。

## 批次 K:two_view_geometry 模块级移植 + db camera1/camera2 序列化(2026-10-02)

### 范围(test-first 裁定)

two_view_geometry 是 upstream 重构面(估计主流程重写 + degenerate-check
防御族 + fisheye/panoramic/watermark/shared·one-sided-focal 扩展)。
fork test 仅 4 用例 vs upstream 32+——按 test-first 原则整模块移植:

- **生产 9 文件**:two_view_geometry.{h,cc}、solvers/{homography,essential,
  fundamental}_matrix.{h,cc}、solvers/utils.{h,cc}(新)、
  cost_functions/{sampson_error,tiny_manifold,tiny_sampson_error,
  reprojection_error}.h——全部 upstream 逐字(include 映射 colmap/ 前缀
  剥除)——移植前逐一验证 fork 无特有语义(符号集对比仅版权头差异);
- **Camera API 适配**:upstream 的 public 成员风格(camera.width/
  has_prior_focal_length)→ fork getter/setter 风格(Width()/
  HasPriorFocalLength())——生产 19 处 + test 23 处,赋值形态改 Set 系;
  poselib::Camera 的小写成员须区分保留(两方向转换函数各按其类型);
- **db camera1/camera2 序列化移植**(MaybeDecomposeRelativePoses 用例
  揭示的反向缺口):two_view_geometries 表加 camera1/camera2 BLOB 列
  (新表 schema + PostMigrateTables 对旧库 ALTER ADD COLUMN 迁移)、
  INSERT/SELECT 语句扩展、CameraToBlob/CameraFromBlob/
  Write·ReadOptionalCamera 辅助(fork getter 版)、write/read 函数体
  接线(SINGLE 与 vector 两路、SELECT * 列序 camera1=10/camera2=11)。

### 验证

- two_view_geometry_test **32/32**(含 degenerate 族/fisheye/panoramic/
  watermark/SharedFocal/OneSidedFocal/MaybeDecomposeRelativePoses);
- database_test 22/22、scene_two_view_geometry_test 2/2;
- gate 162 测试中 160 绿;2 个 GPU 环境性失败与移植无关:
  bundle_adjustment_caspar_test(前批已记录)与 feature/loma_test
  (外部进程占 GPU 17GB 致 Vulkan 推理失败,AICore 路径与本模块
  零代码交集)。

### 依赖闭合链(级联记录)

two_view_geometry → solvers/{homography,essential,fundamental}_matrix
(RelPoseParams/CheiralityEstimator/闭式解)→ cost_functions/{sampson_error
(RelPoseParams 定义),tiny_manifold(2-sphere 断言更新),tiny_sampson_error}
→ solvers/utils(SolveEpipolarConstraintMatrix/RaysFromCamRaysWithJac)。
CMake 补 solvers/utils.cc;poselib_utils.cc 双方向 Camera 转换的成员
风格区分(colmap getter vs poselib 小写成员)。

## 批次 L:caspar 顺序依赖修复 + P1 清单再校正(2026-10-06)

### caspar_test SplitIntrinsic 顺序依赖:机理与修复

SplitIntrinsic 的失败模式重新定性:不是 GPU 资源竞争(单跑时 GPU 空闲
反而必挂、全量顺序跑反而过),而是 **Caspar CUDA solver 的初始化顺序
依赖**——前置用例(Merged variants)建立的 solver/CUDA 状态让后续
split-variant 求解走稳定 kernel 路径;冷启动首解落在不同 kernel 路径,
距 ceres 参考 ~60px(容差 5e-2)。

- **修复**:SplitIntrinsic 测试头加 merged-variant warmup solve(建立与
  全量跑相同的 solver 状态;parity 断言本身不变,不掩盖回归);
- 单跑 ×3 全绿、全量 4/4 ×2——顺序面闭环;
- **-j4 并行面仍是环境约束**:并行的 AICore Vulkan 测试与 CUDA caspar
  跨进程竞争无测试内解,gate 在并行 GPU 负载环境维持 -E 排除
  bundle_adjustment_caspar_test(独立/独占 GPU 的 CI 环境可纳入)。

### P1 清单再校正

- **SolveWithGpuFallback → 假阳性**:upstream 该函数是 GPU BA 求解失败
  时的回退防御;fork 的 ceres BA 为纯 CPU 求解(use_gpu 默认 false 且
  管线无 GPU BA 模式),无 fallback 需求。从 P1 移除;
- **feature/types 矩阵转换族**(Keypoints·Matches From/ToMatrix):
  upstream 自身无消费者(仅 types.cc 内定义),孤立 API 面——P2,待
  upstream 消费面出现时随 feature 批次移植;
- feature/utils 的 BitmapToCHW/HWCToCHW:ONNX 布局转换,AICore ggml
  路径已替代——维持既有裁定。

### P1 剩余清单(更新后)

1. ~~prior-position BA 残差路径~~(批次 J 更正:假阳性);
2. ~~SolveWithGpuFallback~~(本批更正:假阳性);
3. ~~two_view_geometry degenerate-check 家族~~(批次 K 随模块移植完成);
4. **剩余**:SolveTriangulationDlt/Left·RightJacobianFromAngleAxis 等数值
   路径升级(低风险独立移植)、MaybeMarkWatermark 细节面、
   RescaleToMaxImageSize、feature 面残差——均 P2 级,随对应模块批次处理。

## 批次 M:数值路径增量移植(DLT 三角化统一 + SO(3) 雅可比族)(2026-10-06)

### 数值路径对账(假阳性甄别后的真差异)

- **SolveTriangulationDlt/TriangulationDltTerm**:upstream 的多视 DLT
  求解器与其数学等价物在 fork 已存在(cam_rays 版手写展开同算法),真实
  差异仅两点:①fork 的 cam_rays 版缺 `eigen_solver.info() != Success`
  失败检查;②fork 的**矩阵版**(points 参数)是旧式 Vector3d 返回且无
  失败语义(且无消费者——幽灵 API);
- **CalculateAngleBetweenVectors**:fork 已有实现与 .h 声明(本批重复
  移植即重定义报错暴露)——**假阳性**,已回退重复;
- **pose 雅可比族**(QuaternionFromAngleAxis/Left·RightJacobianFromAngle
  Axis + kSmallAngleThreshold 小角阈值):真缺失,纯增量。

### 移植(增量,不破坏 fork 消费面)

- triangulation.cc:匿名区加 TriangulationDltTerm/SolveTriangulationDlt;
  两个 TriangulateMultiViewPoint 重载改用共享 DLT 求解器(矩阵版签名
  Vector3d→bool 对齐 upstream——该重载无 fork 消费者,签名变更安全);
- pose.{h,cc}:追加雅可比族(实现逐字 upstream 依赖 fork 既有
  CrossProductMatrix);
- .h 的重复 CalculateAngleBetweenVectors 声明为历史遗留(两处同签名
  声明无害但冗余),留待后续清理。

### 验证

pose_test 21/21、triangulation_test 2/2、gate 161/161 全绿(排除面:
sift/opengl/caspar/loma——均为 GPU 环境性)。

## 终局盘点:全仓对齐残余量化 + 测试缺口清单(2026-10-06)

### 统计口径演进(三轮)

1. 文件级(含版权头):575/605 diff——被版权头污染,无意义;
2. 剥头 + 行首空白归一:559 diff——仍被折行位置与**结构重组**
   (fork 的 inline 方法挪位、方法分组)污染,假阳性主导;
3. **token 流归一**(去 include 路径/注释/预处理器,保标识符与字面量):
   141/576 逐字同源 + 435 文件级差异——但抽样显示差异大头是
   **fork 适配层惯用语**(LOG 风格/CHECK 宏/fork 特有功能接线)与
   结构重组,文件级口径在重组模块上必然全 diff;
4. **测试用例集合差(套件名归一)是唯一有意义的验收度量**:
   - 套件名敏感口径:808 缺失(被 fork 类名映射假阳性放大,
     如 IncrementalMapperController vs IncrementalPipeline);
   - **用例名级口径:517 个 upstream 测试用例 fork 缺失**(94 个
     test 文件),这是全面对齐的真实工作清单。

### 缺口分布(top)

sensor/bitmap 67(fork bitmap 重写,upstream 新 API 测试全套)、
scene/reconstruction 45、util/threading 30、scene/camera 21、
BA ceres 20、caspar 17、scene/image 17、triangulation 13、
warp 13、misc 13、calibration 族(single_view_calibrator 13/
single_view_calibration 11/anycalib 9/ray_fitting 6——**upstream
新 calibration 模块整族未移植**)、onnx_test 8(排除)。

### 本批落地(+19 用例)

- pose.{h,cc}:补 AngleAxisToRotationMatrix(小角分支);
- pose_test +11(QuaternionFromAngleAxis 三态/Left·Right Jacobian
  五态/AngleAxisToRotationMatrix·YAxisAngle Roundtrip)——**32/32**;
- triangulation_test +8(CalculateAngleBetweenVectors 边界族)——
  **10/10**;
- gate 161/161(GPU 环境性排除面不变)。

### 剩余工作分批计划(按缺口规模)

1. calibration 模块整族(anycalib/ray_fitting/single_view_calibrator,
   含生产代码 + 测试 ~50 用例);
2. sensor/bitmap(fork 重写面,需 fork API 适配的 67 用例);
3. scene/reconstruction 45 + threading 30 + camera 21 + image 17;
4. BA ceres/caspar 测试补全(37);
5. feature/mvs/util/exe 残余;
6. onnx_test 与 onnx_matchers 全族:按裁定排除(AICore ggml 替代)。

## 批次 O:calibration 模块整族移植(2026-10-07)

### 缺口精确化(假阳性甄别先行)

upstream calibration 生态名义 51 用例,对账后 fork **已有** view_graph_calibration
(6/6,token sim 0.986-1.000)与 estimators/cost_functions/calibration(.h sim 1.000,
2/2)——517 清单中此两项为假阳性;真实缺口 = `calibration/` 目录三件套
(anycalib 12 + single_view_calibrator 13 + ray_fitting 7 = 32) +
`controllers/single_view_calibration`(11)= **43 用例**。

### 移植内容(逐字 + 最小适配)

- **生产 7 文件**:calibration/{anycalib,single_view_calibrator,ray_fitting}.{h,cc}
  + resources.h;controllers/single_view_calibration.{h,cc}。onnx 推理块
  (`#ifdef COLMAP_ONNX_ENABLED`)双侧保留(裁定:ggml/AICore 替代 onnx),
  `CreateAnyCalibCalibrator` 无 onnx 构建走 throw 分支;预处理
  `PrepareAnyCalibInput`/坐标变换在 onnx 块外全量可用;
- **依赖闭包增量**:sensor/bitmap.{h,cc} 补 `Rot90(int)`/`Crop`/
  `Pitch`/`RowMajorData`(upstream parity,OIIO-free 手写与 fork Storage
  布局对齐);scene/camera.h 补 3 个 `kDefault{Min,Max}FocalLengthRatio/
  MaxExtraParam` 常量;util/file.h 补 `THROW_CHECK_DIR_EXISTS`;util/misc.h
  补 `LOG_HEADING1/2`;
- **Camera API 适配**(成员→getter/setter):7 处生产(anycalib 4 +
  single_view_calibrator 13 处读写)+ 测试正则族(写形态 Set*/Params() 引用
  赋值/指针 `->` 形态/下标 `cameras[0]` 形态,括号闭合两轮);
- **行为对齐发现**:`Camera::CreateFromModelId` fork 多
  `SetPriorFocalLength(true)`(upstream 无)——审计 3 处生产消费者无 prior
  依赖后删除对齐;
- **测试门控**(onnx 裁定):anycalib_test 3 个 TEST_F 的 SetUpTestSuite
  try/catch + GTEST_SKIP;controllers IntegrationTestWithModel 沿用
  upstream 自带 COLMAP_ANYCALIB_MODEL_PATH 环境变量门控。

### ceres 2.2 NormalPrior 兼容缺陷(fork 环境实锤)

upstream `ray_fitting.cc` 用非方阵 stiffness(1×num_params)构造
`ceres::NormalPrior`:ceres 2.1 声明参数块大小 = `A_.cols()`(正确),ceres 2.2
改为 `b_.rows()`(非方阵时 ≠ 实际参数块 num_params)→ Problem 析构
NormalPrior 析构 double free(gdb 栈实锤)。fork 修复:行等价
`FocalPriorCost` 模板 functor(全参数块 + 运行时 focal idx,数学恒等:
residual_i = scale·(x[idx_i] − prior_i));首轮实现把 scaled_prior 多乘
scale(初值残差 2.11e11 量级指纹当场定位)修正为 prior 本身。

### 验证

- ray_fitting_test **31/31**(TEST_P 全透视模型参数化展开);
- single_view_calibrator_test **13/13**;anycalib_test **9 PASS + 3 SKIP**;
- single_view_calibration_test **10 PASS + 1 SKIP**(upstream 环境门控);
- 全量 gate **165/165**(161 + 4 个新测试目标),0 FAILED。
- 新增用例数:落地 43 用例(66 个含参数化展开),517 缺口清单余 ~474。
- 教训:①移植 functor 的代数必须与 upstream 逐项核对(初值 cost 量级
  指纹当场暴露 scale 双乘);②fork Camera getter 适配正则需覆盖
  `.x`/`->x`/`arr[0].x` 三种形态且写形态自动补右括号;③"fork 缺
  upstream 新模块"判断前先 token-sim 对账同名目录树(本轮 8/51 假阳性)。

## 批次 P:reconstruction_test 首次移植 + Normalize/ComputeBoundingBox/Crop 签名对齐(2026-10-07)

### 统计口径第三次修正:Test 前缀假阳性

用例名对账发现 fork 测试惯用 `TestCameraId` 而 upstream 用裸 `CameraId`——
归一化(剥 Test 前缀)后缺口从 464 修正为 **359**(消除 105 假阳性)。
另:misc_test 13 个「缺口」是**粒度风格差异**(fork 一个 TestCSVToVector
大用例覆盖全分支,upstream 拆 13 个细用例)——语义等价,P2 降级。

### 生产签名对齐(3 个函数,upstream parity)

- `Reconstruction::Normalize` → `Sim3d Normalize(bool fixed_scale=false,
  double extent=10.0, double min_percentile=0.1, double max_percentile=0.9,
  bool use_images=true)`:fork 原为 void 且无 fixed_scale 语义;8 处消费者
  全为语句调用零破坏;early-exit 返回单位 Sim3d;
- `ComputeBoundingBox` → `Eigen::AlignedBox3d (min_percentile=0.0,
  max_percentile=1.0, use_images=false)`:fork 原返回 pair 且无 use_images;
- `Crop` → `Reconstruction Crop(const Eigen::AlignedBox3d&)`:fork 原接受
  pair;exe/model.cc 全链路同步(bounding_box/bounds 容器、WriteBoundingBox、
  ComputeEqualPartsBounds、extent 计算 first/second → min()/max()/center())。

### reconstruction_test 首次移植(0 → 1455 行,48 用例)

fork 该文件为空占位。移植适配面:Camera/Rig/Frame/Image/Point3D/Point2D 的
成员→getter 映射(camera_id×24/track×9/xyz+color×10/point3D_id×4/error→
SetError×3/xy×1)、`CameraId() = 2` 写形态改 SetCameraId、
`AddSensor(sensor_id)` 补 fork 的 optional pose 双参、RegFrameIds NodeHashSet
无下标改 begin() 解引用、ExtractColorsForAllImages 去多线程参( fork 单线程)。

### 34/48 通过 + 14 DISABLED 留因(全部定性为 fork 深度行为差异)

- **注册语义类(9)**:fork 的 db round-trip 扩展使「frame 有 pose 即注册」
  (AddImageWithTrivialFrame/SetRigsAndFrames 的 num_reg_frames=24 vs 0、
  DeRegisterFrame/RegImageIds/FindCommonRegImageIds/ComputeNumObservations/
  ComputeMeanTrackLength/ComputeMeanObservationsPerRegImage/Normalize 的
  early-exit 还在读 reg_image_ids_ 旧账本);对齐方向 = fork 的注册语义
  与 upstream「结构创建 ≠ 注册」分离——这是 fork 核心语义面,需专项批次;
- **错误消息类(2)**:AddFrame/AddImage 的 fork 用 map::at 抛
  "_Map_base::at" 而非 upstream 的 "Rig with ID 1 does not exist" 显式
  THROW_CHECK——低风险,下一批补显式检查;
- **合并颜色类(2)**:MergePoints3D/Colors 的合并颜色选择语义差异;
- **Crop(1)**:断言细节待查。
- 验证:reconstruction_test 34 PASSED + 14 DISABLED;全量 gate **165/165**。
- 缺口清单余 ~311(359 − 48;其中 14 转 DISABLED 留因)。

## 批次 Q:注册簿记 upstream 收敛 + 13/14 用例解锁(2026-10-07)

### 根因分治(批次 P 的 14 个 DISABLED 逐类取证)

四类根因逐个证伪/实锤:
1. **错误消息(3 处)**:fork 的 10 个实体访问器(Camera/Image/Rig/Frame/Point3D
   ×const/非 const)全部裸 `map::at`(抛 `_Map_base::at` 无上下文);upstream
   是 try/catch out_of_range → StringPrintf("X with ID %d does not exist")。
2. **注册簿记(8 个用例的共同根因)**:fork 用「image 级 reg_image_ids_ 缓存 +
   条件 lockstep」,upstream 用「frame 级 reg_frame_ids_ (vector) + 无条件
   num_reg_images_ 计数器 + 派生式 RegImageIds()(带 THROW_CHECK 一致性检查)」。
   fork RegisterFrame 的 lockstep 在 image 不存在时跳过计数 → 注册帧后
   NumRegImages=0;DeRegisterFrame 以 HasPose 为早退条件且不必然清观测。
3. **Merge 颜色**:fork 无黑排除(把未提取的黑色卷入加权平均);
   upstream 黑=未提取,排除后按 track 长度加权。
4. **Crop**:fork 用 legacy TrivialFrame+SetRegistered(false) 结构(框外帧
   保留 pose);upstream 复制全结构→框内点支持帧→框外帧 DeRegister(丢 pose)。

### 生产收敛(全部 upstream parity)

- reconstruction.h:10 访问器 try/catch+StringPrintf;NumRegImages() 改
  num_reg_images_ 计数器;RegImageIds() 改派生式(遍历 reg_frame_ids_ +
  THROW_CHECK ExistsImage);成员删 reg_image_ids_ 加 num_reg_images_。
- reconstruction.cc:RegisterFrame 无条件计数(+ImageIds().size(),幂等守卫内);
  DeRegisterFrame 以 reg_frame_ids_ 成员为早退条件、逐 image 无条件 --计数
  +清观测(单账本路径 DeRegisterImage);RegisterImage/DeRegisterImage 降级为
  fork legacy 标志辅助(计数器唯一维护点=frame 级);MergePoints3D 黑排除+
  加权;Crop 重写为 upstream 结构(全结构复制→清观测→框内点→框外帧
  DeRegisterFrame);Normalize 早退条件 NumRegFrames()<2;单参
  AddImageWithTrivialFrame 删 legacy 播种(把默认 identity 镜像当 pose 假注册);
  两参重载改 RegisterFrame(upstream parity);拷贝 ctor/operator= 同步。
- sfm/observation_manager.cc:OM::RegisterFrame 尾部透传改 frame 级单次
  (upstream 同款);OM::DeRegisterFrame 已是 frame 级 ✓。
- 消费者帧级化:da3_depth_controller(删冗余 RegisterImage)、
  rotation_averaging、global_mapper(帧级 RegisterFrame)。
- fork 消费点 9 处改派生式(ComputeBoundsAndCentroid/FindCommonRegImageIds/
  FilterObservationsWithNegativeDepth/ComputeNumObservations/
  ComputeMeanObservationsPerRegImage/ExtractColorsForAllImages);
  TranscribeImageIdsToDatabase 的 reg ids 重映射行删除(frame data_ids
  已转录,派生式自动跟随)。

### 测试适配与结果

- reconstruction_test:13/14 解锁(AddFrame/AddImage/AddImageWithTrivialFrame/
  RegImageIds/Merge×2/SetRigsAndFramesResetsNumRegImages/Normalize/
  FindCommonRegImageIds/ComputeNumObservations/ComputeMeanTrackLength/
  ComputeMeanObservationsPerRegImage/DeRegisterFrame),47 PASSED +
  1 DISABLED;Normalize 的 in-place 位姿分量编辑改 SetRigFromWorld
  (fork RigFromWorld() 返回组装值,引用编辑是 no-op)。
- 留因 1:SetRigsAndFrames——注册流程已对齐(重放后 num_reg_frames=0 双方
  一致),但 fork db round-trip 持久化帧位姿,重放帧带 db 精度位姿 vs 原始
  内存位姿,ReconstructionEq/序列化对比见位级差(upstream db 不存 pose 无此面)。
  留待 db round-trip 精度专项。
- BA×2/undistortion 测试构造改两参显式 pose(旧构造依赖已删除的播种)。
- gate 165/165;caspar SplitIntrinsic 复跑 3/3 过(此前失败为 GPU 环境瞬态)。

### 遗留

- SetRigsAndFrames(db round-trip 位姿精度专项);
- exe/image.cc 的 FilterImages/DeRegisterImage 为 fork 特有工具语义
  (计数器为 frame 级权威,图像级 deregister 后计数不减——工具随后丢弃
  重建对象,边界可接受,已在注释注明)。

## 批次 R:SetRigsAndFrames 位级 round-trip + camera/threading 测试移植(2026-10-07)

### SetRigsAndFrames:fork db round-trip 的位级验证解锁

批 Q 的序列化 diff 取证推翻「精度损失」假设:fork frames 表 pose 为
double BLOB(memcpy 位级无损),真正的差异是**结构性的**——colmap db
schema 只存 rigs/frames(含 fork 的 pose)/images,**不存 points3D 与观测**
(它们是重建内存态)。upstream 测试期望(重放 == 注销后状态)成立是因为
upstream DeRegisterFrame 也删光了点;fork 的重放从 db 恢复 pose 与注册,
恢复的是「注销前」状态。fork 版断言改写为更强的形式:重放后 24 帧全部
HasPose 且与 synthetic 内存位姿**位级相等**(Rigid3d operator==),
num_reg_frames/num_reg_images 恢复为 24——这是 fork db round-trip 的
回归保护门。用例解锁,reconstruction_test 48/48 全绿。

### camera_test:+10 用例(33/33 全绿)

- 新移植:Equals/Print/SensorId/CreateFromModelId/CreateFromModelName/
  ImgFromCamWithoutCheiralityCheck/CamRayFromImgWithJac(+Unprojectable)/
  Spherical/IsPerspectivePinhole;
- 生产增量:camera operator<<(upstream 逐字,Camera API 适配);
  CalibrationMatrix 的 LOG(FATAL) 改 THROW_CHECK(IsPerspective())
  (可捕获,Spherical 探针不再 abort);
- util/string + util/misc:SetFullPrecTextStream(classic locale +
  precision(17))移植,VectorToCSV 从 std::to_string 定点 6 位改为
  ostream 全精度格式(upstream parity;ParamsToString 输出 "1, 0.5, 0.5")。

### threading_test:+8 用例(34/34 全绿)

- 生产移植:AggregateException 聚合异常类 + ThreadPool 的
  finished_task_checkers_ 机制(AddTask 返回 shared_future 并携带
  checker 闭包;Wait/Stop 先重放 checker 聚合任务异常再返回);
- 新移植:Future/Wait/Stop/FutureAndWait PropagatesException 家族 +
  CatchAggregateException/AggregateExceptionMessage + Nominal;
- 消费者适配:undistortion.cc 的 futures 容器 std::future → shared_future。

### gate 假绿教训(重要流程缺陷)

批 Q 收口的 gate 跑 `make ColmapLib` 后直接 ctest——**只重编了静态库,
未重链依赖它的测试二进制**(global_mapper_test/incremental_triangulator_test/
global_pipeline_test 等仍链接旧库副本)→ 批 Q 注册簿记收敛的行为涟漪被
旧二进制掩盖,165/165 为假绿。批次 R 全量重链后暴露 3 个真实失败:

- incremental_triangulator_test.CompleteImage:观测 100 vs upstream 80——
  fork 的 Continue 对被删观测的补回行为与 upstream 不同(upstream 该场景
  Continue 全拒);
- global_mapper_test(4 用例)/global_pipeline_test:retriangulation 先
  DeleteAllPoints2DAndPoints3D(连 2D 坐标一起清)再 TriangulateImage
  (遍历已清空的 2D)→ 零产出。该模块为 fork 专有(upstream 无
  global_mapper),属 fork 设计缺陷而非 upstream 对齐缺口。

两者均为批 Q 注册收敛的涟漪,需要专门批次插桩取证(含 upstream 真值
对照),本轮已从 gate 排除并记录。教训:**gate 前必须全量重链所有测试
二进制**(make -k ColmapLib 不够),或直接以 ctest 触发的隐式重链为准。

### 当前缺口

3 个批 Q 涟漪测试(含 8 个用例)回填 gate 待专项批次;其余测试移植面
(bitmap_test 52,OIIO 重写面)按批次 O/P 的甄别方法推进。

## 批次 S:批 Q 涟漪修复回填 gate + retriangulation 语义修复(2026-10-08)

### 修复 1:retriangulation 清空粒度缺陷(fork 专有模块)

global_mapper 的 IterativeRetriangulateAndRefine 先
DeleteAllPoints2DAndPoints3D(连 2D 坐标一起清)再 TriangulateImage
(遍历已清空的 2D)→ 零输入零产出。新增
Reconstruction::DeleteAllPoints3D(清点 + 清 2D→3D 关联,**保留 2D
keypoint 坐标**——三角化的输入数据),retriangulation 前置改用它。
调用方约束:跨调用不得持有 OM(新 OM 在 BeginReconstruction 重建)。

### 修复 2:三角化器门禁 IsRegistered → HasPose(upstream parity)

批 Q 注册收敛后 image 的 legacy registered 标志维护窗口
(EndReconstruction(discard) 清除)与位姿事实脱节——retriangulation 的
重建帧 pose 在而标志为 false(插桩实证 reg=0/num2d=60)。upstream 的
TriangulateImage/CompleteImage/Retriangulate 门禁全部是 `HasPose()`;
fork 三处对齐。量级矛盾定位法再次命中(同一重建两处读数 60 vs 0)。

### 结果

- global_mapper_test:WithoutNoise/WithNoiseAndOutliers/
  WithNonTrivialKnownRig 3 用例修复回填(global_pipeline_test 13/13 回填);
- incremental_triangulator_test 回填(7 PASS + CompleteImage 留因 DISABLED:
  Continue 对被删观测的补回语义与 upstream Find/Create 过滤不同——
  100 vs 80,upstream Find 对已三角化对应的过滤语义需专项对照);
- global_mapper_test 仅剩 WithoutNoiseWithNonTrivialUnknownRig 留因
  DISABLED:retriangulation 在 trivial rig 下已修复,但 non-trivial rig
  的合成 matches 是帧对粒度(C(14,2)=91 条 ref-ref 边),非 ref 相机的
  2D 在对应图中无对应(Find 返回 corrs=0)——需要
  DatabaseCache/CorrespondenceGraph 的 rig 感知展开,属核心基建专项;
- **gate 165/165 全绿**(全部测试二进制已重链;排除面仅
  caspar/loma/sift/opengl 四个 GPU/环境性用例)。

### 遗留(下一批入口)

- CompleteImage 的 Continue 补回语义(upstream Find 对已三角化对应的
  过滤对照);
- NonTrivialUnknownRig 的对应图 rig 感知展开(核心基建专项);
- bitmap_test 52(OIIO 重写面)。

批次 T:注册簿记容器对齐 + legacy 标志退役 + 两个 DISABLED 解锁(2026-10-08)

### 修复 1:reg_frame_ids_ 容器对齐 upstream(std::vector,插入序)

批 Q 把 reg_frame_ids_ 定为 NodeHashSet——枚举序是 hash 序,而
upstream d3ccaf35 是 std::vector<frame_t>(std::find 幂等 + push_back,
**注册插入序**)。RegImageIds()/RegFrameIds() 的枚举序是可观察 API 面:
CompleteImage 用例 `.at(0)` 恰好敏感于该序——fork hash 序下 .at(0)=image 5
而全部被删观测在 image 1(synthetic track 构造序的首图),CompleteImage
对非删除图运行 Complete(track) 补回全部 20 个观测(100 vs 80);
插入序下 .at(0)=image 1=被删图,观测全无 3D 点,Find 报
num_triangulated>0 不补回——恰 upstream 的 80。探针实证:
reg_image_ids 5,4,3,2,1 → 修复后 1,2,3,4,5;obs_after 100 → 80。
**批 S 的「Continue 补回语义差异」假说被证伪**(fork Continue/Create 与
upstream 逐行一致,upstream Find 也不过滤已三角化对应,仅计数)。
reg_frame_ids_ 改 std::vector,RegFrameIds() 返回 const 引用;
遍历中修改的站点(incremental_mapper BeginReconstruction、
rotation_averaging)改为显式拷贝。

### 修复 2:legacy per-image registered 标志门禁退役(HasPose 对齐)

upstream d3ccaf35 的 Image **没有** registered 标志(注册态=帧位姿事实)。
fork 的 lockstep 维护(RegisterFrame 内对已存在 image SetRegistered)
有顺序缺口:**Load 按 rigs→frames→images 添加,AddFrame 自动注册时
image 尚不存在 → SetRegistered 被跳过;后续 RegisterFrame 幂等跳过 →
标志永远 false**(reg-dbg 桩实证:frame1 already=0 first_img_exists=0)。
retriangulation 时 Find/Merge/Complete 内部残留的 IsRegistered() 门禁
把全部对应过滤(corrs=41 → corrs_data=0)→ TriangulateImage 零产出、
Retriangulate 把观测重建为 45 个 2-观测碎点(REC 95 点 vs GT 50)。
修复:①AddImage 末尾对已注册帧补设标志(lockstep 补全);②三角化器
Find/Merge/Complete 与 incremental_mapper 的 corr 门禁、
incremental_mapper_impl 的 FindLocalBundle 断言全部对齐 upstream 的
HasPose()。批 S 留因的「帧对粒度 matches」假说也被证伪:对应图已覆盖
全部 42 图(探针 images_with_corr=42/42,每图 50/60 观测有对应)。

### 结果

- incremental_triangulator_test:8/8(CompleteImage 解锁,期望 80 精确);
- global_mapper_test:5/5(WithoutNoiseWithNonTrivialUnknownRig 解锁——
  REC 与 GT 完全一致:50 点 × 42 观测 = 2100 观测精确,sensor_from_rig
  误差 1e-6 度);
- hierarchical_pipeline_test 两个 non-trivial 用例留因更新:失败形态
  前移到簇合并——MergeReconstruction/AddImageWithTrivialFrame 假设
  1 帧 1 图("Frame 2 already exists"),non-trivial rig 的合并需要
  rig-aware MergeReconstruction(真正的 rig 感知基建专项,下一批入口);
- **gate 162/163**(ctest 163 行,唯一失败 caspar parity 为并行 GPU
  92% 负载下的已知环境敏感项,与本批改动无关);
- 附带:修复 test_qanylabeling_pointcloud 链接错误(并行会话给
  PointCloudDocument 集成 cuboid sidecar 时漏把 PointCloudCuboid.cpp
  加进测试源列表)。

### 遗留(下一批入口)

- rig-aware MergeReconstruction(hierarchical 簇合并,解锁 2 个
  hierarchical 用例——真正的 rig 感知合并基建);
- bitmap_test 移植:实测缺口 64 用例(非 52;upstream 67 vs fork 31),
  需先补齐 fork Bitmap 的 API 面(Clone/CloneAsGrey/CloneAsRGB/
  MetaData/Exif 族/Interpolate/Rescale/Rot90/RowMajorData/Thumbnail
  等),独立子批;
- exe/image.cc 的 image 级过滤(IsRegistered)在磁盘模型(标志 false)
  下失效——对齐 upstream 的 FindFramesToFilter 帧级路径。

## 批次 U:rig-aware MergeReconstruction(2026-10-09)

- `Reconstruction::Merge` 不再为每个缺失 image 调
  `AddImageWithTrivialFrame`。它按源 frame 复制完整 rig 的全部 cameras、
  rig、frame data ids 与单一 frame pose，再逐 image 接线；因此一个 frame
  可安全承载多个相机图像。
- 相似变换先应用到 frame 的 `rig_from_world`，随后只做 frame→image 的
  legacy qvec/tvec 镜像同步。移除了合并尾部的二次重投影过滤：一对一轨迹
  冲突已在合并阶段解决，额外过滤会按簇局部优化噪声丢弃正确观测。
- `hierarchical_pipeline_test` 的
  `WithoutNoiseAndNonTrivialFrames` 与
  `WithoutNoiseAndPanoramicNonTrivialFrames` 已解除 DISABLED 并经 ctest
  通过。剩余入口：Bitmap API/64 个用例子批，以及 `exe/image.cc` 的帧级过滤。

## 批次 V:hierarchical 误差收尾与 image registrator 位姿门禁(2026-10-09)

- hierarchical cluster merge/filter 会暂存 normalized/angular residual；最终
  `ReconstructionManager` 发布前现在逐模型调用 `UpdatePoint3DErrors()`，保证
  `Point3D::Error()` 回到像素 reprojection error 契约。
  `HierarchicalPipeline.WithoutNoise` 已解除 DISABLED，并验证发布值与显式
  重算值一致。
- `RunImageRegistrator` 的已注册判据从 legacy `IsRegistered()` 改为
  `HasPose()`。指定 `Mapper.image_list_path` 时，cache 现在额外装入所有
  已有 pose 的图；否则 `BeginReconstruction()` 重载已有模型时会访问被选择集
  排除的 image/cache 映射并抛 `_Map_base::at`。同时对齐 upstream 的
  `load_all_images=true` 与 image-list 选项解析。新增注册的 `exe_image_test`
  已实测 **2/2 通过**（含断连已注册图回归）。
- L3 双端 gate 已落地为 opt-in `reconstruction_l3_alignment_gate`：要求固定
  图像清单 SHA-256、固定 mapper seed、候选二进制、d3ccaf35 源 checkout 与
  二进制、校准阈值 JSON；脚本导出 TXT 模型，使用公共注册图 camera center 的
  Sim3 规范化后比较注册图/点/观测/轨长/重投影 RMS、位姿与对称点云 NN 距离，
  输出 verdict 与完整 provenance JSON。缺少 fixture、pinned upstream build、
  harness 或阈值时配置/运行失败，不得报绿；本机尚未具备该执行前提，因而没有
  L3 通过声明。
- 仍待：Bitmap API/64 个用例、Caspar 冷启动/并行 GPU 稳定性矩阵，以及在具备
  d3ccaf35 build 与固定 fixture 后实际运行 L3 并把阈值校准结果回填。

## 批次 W:a2bd15c2 全面审核——默认值/管道/选项面缺口回填(2026-10-09)

审核方法:对 fork 与 upstream `a2bd15c2` 做 struct 默认值 diff(按 basename+struct
键配对)、全文件 token ratio(596 文件)与 option_manager 选项名集合 comm。结论:
"基线 d3ccaf35" 实为部分对齐,大量 pre-d3ccaf35 的 upstream 特性仍缺。本批回填:

- **PoseLib pin**:`3rdparty/PoseLib/poselib.cmake` fa7280fe → 1bb8881f(upstream
  d7ffb9c4 同步;含 gen_relpose_5p1pt gamma>0 + 第六点 cheirality 修复
  PoseLib#214)。`solvers_generalized_relative_pose_test` GR5P1P 2 用例由红转绿
  (17/17),`generalized_pose_test` 16/16。
- **fork BA 缺陷**:`bundle_adjustment_ceres.cc` AddImageToProblem 的 compose_rig
  仅在 `refine_sensor_from_rig` 为真时置位,非参考传感器冻结时用裸 rig 位姿重投影
  (upstream AddImageWithNonTrivialFrame 始终 compose,flag 只决定 sensor block
  是否变量)。去掉该条件后 `global_mapper_test` NonTrivialKnownRig/UnknownRig
  (retriangulation 后 0 点)翻绿;`hierarchical_pipeline_test` 对齐 upstream
  `ba_refine_sensor_from_rig=false` 设置,全二进制运行 3/3(此前仅单独运行通过
  ——测试间 PRNG 依赖暴露的合并脆弱性)。
- **增量管道**:`IncrementalMapperOptions` 新增
  `structure_less_registration_fallback=true / structure_less_registration_only=false
  / max_runtime_seconds=-1`;Check 加 `!only || fallback`;Run 启动
  `total_run_timer_`,各检查点 `CheckReachedMaxRuntime()`;注册循环重写为 upstream
  do-while + `structure_less_flags` 结构(结构优先、失败回退 structure-less;去掉
  双重 TriangulateImage)。option_manager 注册三项 `Mapper.*`。
- **默认值**:`AbsolutePoseRefinementOptions` refine_focal_length/refine_extra_params
  /print_summary → false(upstream 2024-09;mapper 全部显式设置);
  `BundleAdjustmentOptions::refine_points3D=true` 并由 Ceres ParameterizePoints 与
  Caspar point_variable 消费;`UndistortCameraOptions::max_cam_point_norm`
  (THROW_CHECK_NE 0,4 处边界采样 squaredNorm 过滤)+ `undistortion_test
  MaxCamPointNorm`;option_manager 注册 `BundleAdjustment.refine_rig_from_world/
  refine_sensor_from_rig/refine_points3D/constant_rig_from_world_rotation/
  min_track_length`、`GlobalMapper.tri_*/ra_*/ba_refine_rig_from_world/
  ba_refine_points3D/ba_min_track_length`。
- **VoteAndVerify**:fork 为 2018 旧版(`FeatureGeometryMatch::geometries2` 向量、
  固定 6 级、无 local_optimization/eff_inlier_count、num_transformations=30)。
  同步 a2bd15c2 `vote_and_verify.{h,cc}` + `geometry.h` 单 `geometry2`
  (消费者仅 visual_index.cc 一处);`vote_and_verify_test` 去裁定、NoEffectiveInliers
  解锁,retrieval/ 5/5。
- **PatchMatch #4154/#3857**:`PatchMatchOptions::num_threads=-1` + 共享
  `io_thread_pool_` 并行读入(workspace cache 加 `cache_mutex_`,evict 竞态按
  `cache_.Exists` 守卫);`Workspace::Load` 改为"尺寸不符即缩放"(#3857)。注册
  `PatchMatchStereo.num_threads`、automatic_reconstruction 透传、Dense GUI 与
  pybind 字段。
- **Hierarchical**:`Options::num_threads=-1` 总预算按 worker 分配(upstream
  语义:`Mapper.num_threads>0` 时 WARNING 并覆盖);CLI `--num_threads`、pybind。
- **SIFT**:`SiftExtractionOptions::force_covariant_extractor`(controller 两处
  分发 + CPU/GPU 提取 CHECK)、`SiftMatchingOptions::cpu_brute_force_matcher`
  (`MatchSiftFeaturesCPU` 分流到 BruteForce);option_manager 与 pybind 注册。
- **Rig verification**:新增 `Database::ReadNumMatches`(SQLite `SELECT pair_id,
  rows`),`controllers/feature_matching.cc` 移植 `RigVerification`(帧对统计 →
  `EstimateRigTwoViewGeometries` → 覆写 per-pair 几何),`GeometricVerifierOptions::
  rig_verification` 接入 `CreateGeometricVerifier`。新增
  `controllers_feature_matching_test`(Nominal / RigVerificationWithNonTrivialFrames
  15 CALIBRATED_RIG / WithTrivialFrames 跳过)**3/3 首次即过**。
- **pybind 面**:`GlobalPositionerOptions` generate_scales →
  initialize_scales_from_geometry + fix_first_scale/experimental_observation_stddev/
  uncalibrated_observation_weight;`Reconstruction.crop` 主签名改 `AlignedBox3d`
  (与 `compute_bounding_box` 返回类型一致,保留 pair 重载),两处 python 测试改
  upstream 语义。
- **Gate**:全量 `cmake --build build_app` 通过;C++ `ctest -R "_test$"`
  **171/172**(唯一失败 `feature/sift_test` CPUvsGPU 98/99 为账本既定 GPU 排除项);
  pytest `python/test/reconstruction/` **395 passed / 3 skipped**。

### 审核发现的结构性遗留(下一批入口,按优先级)

1. **feature 模块三层并存**(`feature/sift.cc` ratio 0.22、`controllers/
   feature_matching.cc` 0.27):CLI 走 legacy `ExhaustiveFeatureMatcher`(flat
   `SiftMatchingOptions`),parity 层 `Create*FeatureMatcher` 仍收
   `SiftMatchingOptions`,`FeatureMatcherController`(`FeatureMatchingOptions`)
   仅测试消费。upstream 形态为 `FeatureMatchingOptions + TwoViewGeometryOptions`
   贯穿 CLI/GUI/pybind,`FeatureMatching.rig_verification` 在 matcher 尾部触发。
   fork 亦缺 CLI `geometric_verifier` / `guided_geometric_verifier` 命令。
2. **option_manager 命名滞后**(fork 322 vs upstream 380 选项):
   `BundleAdjustment.use_gpu`→`BundleAdjustmentCeres.*`、`Mapper.ba_global_images_freq`
   →`ba_global_frames_freq`、`SiftMatching.max_error`→`TwoViewGeometry.*`、
   `HierarchicalMapper.*` 整组。方案:upstream 名为主名 + fork 旧名 `AddOptionAlias`。
3. 批3b(SKEWED_PINHOLE / CamFromImg autodiff / feature cache)、批4(Ceres
   rotation averaging / IMU)、Bitmap 64 用例、`StereoFusionOptions.mask_path` 已有。

## 批次 X:feature 三层合一 + option_manager 无别名全量改名(2026-10-10)

批次 W 遗留 1/2 的结构性收口。用户明确要求**彻底对齐、不保留旧名兼容层**:所有
模块(C++ 库/CLI/GUI/pybind/测试)统一切到 upstream `a2bd15c2` 命名,批次 W 预案
的 `AddOptionAlias` 路线作废(现 option_manager.cc 中 `AddOptionAlias` 为 0 处)。

### F1 feature 模块(三层 → 一层)

- **F1a `feature/`**:`sift.{h,cc}` 对齐 upstream 结构(`SiftExtractionOptions` 仅
  SIFT 字段;`SiftMatchingOptions` 仅 max_ratio/max_distance/cross_check/
  cpu_brute_force_matcher/lightglue;`CreateSiftFeatureExtractor/Matcher`),共享字段
  上移到 `FeatureExtractionOptions`(max_image_size/num_threads/use_gpu/gpu_index +
  `shared_ptr` sift/loma/aliked)与 `FeatureMatchingOptions`(num_threads/use_gpu/
  gpu_index/max_num_matches/guided_matching/rig_verification + sift/loma/aliked);
  RANSAC 阈值归 `TwoViewGeometryOptions`。新增 `lightglue.{h,cc}`、`model_cache.{h,cc}`、
  `learned_matchers.cc`(ggml brute-force,替代 upstream ONNX 版)。
- **F1b**:`feature/matching.{h,cc}` 删除,`FeatureMatcherCache` 迁至
  `controllers/matcher_cache.{h,cc}`(upstream 位置)。
- **F1c `controllers/`**:`feature_extraction.cc`、`feature_matching.cc`、
  `feature_matching_utils.{h,cc}`(`FeatureMatcherController` 成为唯一匹配执行层)、
  `pairing.h` 对齐 upstream;legacy `ExhaustiveFeatureMatcher` 等 Thread 子类退役,
  `Create*FeatureMatcher(pairing, FeatureMatchingOptions, TwoViewGeometryOptions,
  database_path)` 为唯一工厂形态,`rig_verification` 在 matcher 尾部触发。
- **F1d 消费者**:`exe/feature.cc` 重写(含新增 CLI `geometric_verifier` /
  `guided_geometric_verifier`,`exe/colmap.cc` 注册);`automatic_reconstruction.cc`、
  `ui/feature_matching_widget.cc`;ACloudViewer `app/reconstruction/
  {FeatureExtraction,FeatureMatching,BundleAdjustment,ReconstructionOptions,
  AutomaticReconstruction}Widget` 改用新选项面;pybind 见下。

### P3 选项结构 + option_manager

- **P3a**:`IncrementalPipelineOptions`(upstream 名;`IncrementalMapper::Options`
  为其 `mapper` 子对象)、`HierarchicalPipelineOptions{clustering_options,
  incremental_options,...}`、`BundleAdjustmentOptions::{ceres,caspar}` shared_ptr 子选项
  (`CeresBundleAdjustmentOptions` 承接 use_gpu/gpu_index/solver_options/loss_*;
  `CasparBundleAdjustmentOptions` 承接 solver_iter_max/pcg_iter_max/...)。
- **P3b option_manager**:成员 `feature_extraction/feature_matching/two_view_geometry/
  *_pairing/bundle_adjustment/mapper/global_mapper/hierarchical_mapper`;
  `AddFeatureExtractionOptions/AddFeatureMatchingOptions/AddTwoViewGeometryOptions/
  Add*PairingOptions`;选项组 `FeatureExtraction.* / SiftExtraction.* /
  FeatureMatching.* / SiftMatching.* / TwoViewGeometry.* / BundleAdjustmentCeres.* /
  BundleAdjustmentCaspar.* / Mapper.ba_global_frames_freq / HierarchicalMapper.*`。
  **选项名集合核验**(正则抽取 `Add*Option("…")`):fork 365 vs upstream 345;
  upstream-only 仅 9 项 `Loma*.use_bf16/*_bf16/use_fast_resize`(ONNX 专属,fork 以
  ggml `device` 字段替代,设计排除);fork-only 为 `Texturing.*`(13)、
  `MeshPostProcessing.*`(13)自有功能 + `Loma/Aliked*.device`(3)。

### pybind 面(libs/Python/pybind/reconstruction)

- `reconstruction_options.cpp`:Sift/BA/Ceres/Mapper 段按新 struct 重写;
  `CeresBundleAdjustmentOptions`、`BundleAdjustmentOptions`、`SiftExtractionOptions`、
  `SiftMatchingOptions`、`CasparBundleAdjustmentOptions`、Loma/Aliked 选项全部
  `shared_ptr` holder(与 C++ 成员类型一致,否则 pybind 拒绝);修正历史 sed 误改
  (`IncrementalMapper::Options` 的 python 名恢复 `IncrementalMapperOptions`,曾与
  `IncrementalPipelineOptions` 重复注册);删除六个旧 `*MatchingOptions` binder。
- `feature/feature.cpp`:`exhaustive_match/sequential_match/...(database_path,
  matching_options=FeatureMatchingOptions(), geometry_options=TwoViewGeometryOptions(),
  pairing_options=*PairingOptions())`;新增 `verify_geometry` /
  `verify_guided_geometry`;`FeatureMatcher.match/match_guided` 改 upstream 签名
  (描述子按 const& 传入,fork 矩阵类型无法做 shared_ptr holder);
  `pybind_feature_methods` **必须在 pipeline binder 之后调用**(默认参数依赖
  `*PairingOptions` 已注册)。
- `pipeline/match_features.cpp`:`MatchFeatures<Pairing, Factory>` 模板改
  `FeatureMatchingOptions`;新增 `guided_geometric_verification`、
  `TransitivePairingOptions`(fork-only)绑定;`GeometricVerifierOptions` 补
  `rig_verification/use_existing_relative_pose`。
- `estimators.cpp`:删除残留别名 `CeresPosePriorBundleAdjustmentOptions`(它覆盖了
  真实类导致 pytest TypeError)。

### 测试适配(全部链接+运行通过)

- 从 upstream 同步并机械适配回 fork 矩阵描述子契约:`sift_test`(+camera
  `Params()/Width()/Height()`)、`feature/utils_test`(`HWCToCHW(data.data(),…)`)。
- ggml 后端 API 适配:`loma_test`、`aliked_test`、`onnx_matchers_test`(packed-u8
  描述子、`make_shared` keypoints/descriptors、裸 `Camera*`、`brute_force./lightglue.`
  子字段、`ProvisionDefaultLomaModels(type,&opts)`、
  `DefaultLightGlueModelPath(LightGlueFeatureKind::ALIKED)`)。
- 选项结构:`incremental_pipeline_test`(两 MaxNumIterations 用例换 upstream 原文)、
  `bundle_adjustment_caspar_test`(`ceres->`/`caspar->`)、`hierarchical_pipeline_test`
  (`HierarchicalPipelineOptions`)、`matcher_test`、`index_test`、
  `matcher_cache_test`/`feature_matching_utils_test`(include 迁移)。
- `option_manager_test` **整体重新同步 a2bd15c2**(15 用例)+ 保留 fork 适配(mask_path
  非空写回)+ fork-only `ParseCasparSolverOptions`(`bundle_adjustment->caspar->*`);
  删除 `ParseLatestUpstreamFeatureOptionAliases`(别名层不存在)。
- `ransac_test TestOptions`:`max_num_trials` 断言随 upstream 改 `int::max`。

### 本批顺带修复的生产缺陷

- `learned_matchers.cc BruteForceGGMLMatchingOptions::Check`:`max_ratio` 由 `GT 0`
  改 upstream `IN(0,1)`——原 Check 与自身 matcher 语义(`max_ratio==0` 即禁用 ratio
  test)矛盾,`onnx_matchers_test` 4 用例因此抛 Check 失败。
- `loma_test MatcherCreationAcrossVariants` → `MatcherCpuCreation`:原 fork 扩展到
  LOMA_G 导致单测冷缓存触发 1.3GB gguf 下载(ctest 600s 超时);按 upstream
  `Bf16MatcherCpuFallback` 收窄为 LOMA_B。

### Gate

- 全量 `make -k`(ACloudViewer + 全部 colmap_test)**0 error**。
- C++ `ctest -R "_test$"` **173/173**(含此前账本排除的 `feature/sift_test`:CPUvsGPU
  本机通过;4 个 `TypeMismatch` 用例标 `DISABLED_`,见 F2)。
- pytest `python/test/reconstruction/` **395 passed / 3 skipped**(`test_extraction.py`
  改 `feature.Normalization`/`requires_opengl()`;`test_parity_batches.py` 改
  `bao.ceres/caspar`、`o.get_mapper()` 语义)。
- 过程注记:并行会话曾使 `3rdparty/ggml/patches` 补丁损坏(ext_ggml patch step 失败,
  阻塞所有依赖 AICore 的 make 目标),期间以直接调用各 target `build.make` 绕过;
  本批收尾时已恢复,最终 gate 走常规 make。

### 遗留(下一批入口,按优先级)

1. **F2 typed FeatureDescriptors（已完成）**: `FeatureDescriptors{type, data}`
   已成为唯一描述子契约；数据库迁入 `descriptors.type` 并删除旧
   `float_descriptors` 运行时路径。`sift_test` 的 4 个类型错配用例、
   VisualIndex 类型门控及 pybind typed holder 路径已由 C++/Python gate 覆盖。
2. **RANSAC 并行迁移（已完成）**: `RANSACOptions::num_threads`、每 worker
   独立 sampler/estimator/support state、种子派生和受锁的最佳模型归约已对齐；
   `ParallelSimilarityTransform` 与 `ReproducibilityWithRandomSeed` 已入
   `ransac_test`。
3. 批次 W 遗留 3 不变:批3b(SKEWED_PINHOLE / CamFromImg autodiff / feature cache)、
   批4(Ceres rotation averaging / IMU)、Bitmap 64 用例。

## 批次 Y:Bitmap/相机/IMU/Ceres RA 移植 + Caspar 稳定性矩阵 + LORANSAC 种子回归修复(2026-10-11)

基线:本地 upstream `a2bd15c2fbfa79ca3812f8b081708da5e47d5402`。

### 本批落地

- **Bitmap/OIIO**:metadata Set/Get/Clone、JPEG quality 实际写入、OIIO 线程局部
  错误清理、linear colorspace 读写往返、灰度 `GetPixel` 三通道语义;删除
  `BitmapFormat`/`BitmapRescaleFilter` 旧写入契约(改 `Bitmap::RescaleFilter`,
  pybind 同步)。EXIF orientation 按基线 `oiio:reorient=0` 保留 tag、不自动旋转。
- **SKEWED_PINHOLE**:model id 18、运行时分发、投影/反投影、Jet autodiff、解析
  Jacobian;数据库、文本/二进制模型与 pybind 往返用例。
- **IMU**:`sensor/imu.{h,cc}`、`scene/imu.{h,cc}`、`imu_t`/`kInvalidImuId`,
  测试 `sensor/imu_test`、`scene/scene_imu_test`。
- **Ceres rotation averaging**:`estimators/rotation_averaging_ceres.{h,cc}`
  (WXYZ manifold、MST 初值、连通性校验、match-count 重权、calibrated rig 外参常量)。
- **LORANSAC 种子回归(根因修复)**:先前 RANSAC 并行迁移把 `SetPRNGSeed` 从
  `RANSAC` 构造函数移入 `RANSAC::Estimate`,而 `LORANSAC` 覆写 `Estimate`,
  导致 LO 路径上 `random_seed` 失效,`two_view_geometry_test` 的 3 个
  `*Deterministic` 用例稳定失败。`LORANSAC::Estimate` 按 upstream 移植为
  per-worker 状态 + 种子派生 + 受锁归约(保留 fork 双形 estimator 适配),并补
  upstream 的"内点集不足 LocalEstimator::kMinNumSamples 即停止递归"守卫。
  test-first:新增 `LORANSAC.ReproducibilityWithRandomSeed` 在旧头文件下失败、
  修复后通过;另移植 `ParallelSimilarityTransform`、
  `RecursiveRefinementStopsBelowMinimumSampleCount`。
  上文批次 X 遗留第 2 条"RANSAC 并行迁移已完成"据此更正:当时 LORANSAC 未覆盖。

### Caspar 稳定性矩阵(实跑证据,撤销专项排除)

- 环境:HEAD `601806861b`(工作树 dirty)、`bundle_adjustment_caspar_test`
  sha256 `513cbf39…a2b389e1`、RTX 4090 / driver 580.95.05、Release、f32 kernels;
  Ceres CPU `num_threads=1 max_iter=100`,Caspar `solver_iter_max=100`。
- 冷启动:5 个用例各自独立进程 ×10 → 0/50 失败;完整二进制顺序子进程 ×10 → 0/10;
  并发 4×3 轮 → 0/12;并发 8×3 轮 → 0/24。合计 **106 次 0 失败**。
- 误差分布:SplitIntrinsic 三族逐次位级一致(pp-only Caspar rms 2.559e-4、
  focal-only 4.850e-4、全固定 1.00104 vs Ceres 1.00027);pinhole parity 的
  Caspar rms 跨次在 5.4e-5…3.5e-4 漂移(GPU 归约序),门限
  `ceres*1.25+1e-3≈1.06e-3`,余量约 3×。
- 结论:批次 W 前记录的 SplitIntrinsic 63.9px 失败签名在当前构建不可复现;
  上文"有并行 GPU 负载需 -E 排除 bundle_adjustment_caspar_test"的裁定**撤销**,
  默认 gate 纳入。矩阵脚本未入库(一次性证据采集,非 gate)。

### Gate

- 全量 `cmake --build build_app -j8` 通过。
- C++ `ctest -R "_test$" -j4` **176/176**,零排除(含 caspar_test)。
- pytest `python/test/reconstruction/` **397 passed / 3 skipped**。

### 遗留(实测用例数 fork vs upstream,未完成)

1. `sensor/bitmap_test` 47 vs 75。
2. `estimators/rotation_averaging_ceres_test` 5 vs 11:fork 缺 upstream 的
   `EstimatesUncalibratedRigs`、`RefinesCalibrationThroughPreparedProblem`
   (prepared-problem API)、`MissingSensorRotation`、
   `CalibratedRigWithDisconnectedImageGraph`、`RejectsUnconfiguredRotationBlocks`、
   `AddsIndividualRelativeRotationResidual`、`SelectsMstOrSuppliedInitialization`
   等行为面——属真实生产缺口。
3. `sensor/imu_test` 4 vs 16、`scene/imu_test` 2 vs 5:upstream 无预积分用例;
   差异是 fork 把 upstream 用例合并为粗粒度用例,`Iterator`/`Copy`/
   `ConstructFromVector`/`DuplicateThrows`/`InsertVector` 等未按同名单独断言,
   需逐名补齐以便符号级审计。
4. ~~CLI-Anything `reconstruct sparse` mapper 参数透传与 L3 上游基线~~ → 见下节
   (透传完成;L3 默认配置 PASS、带 mapper options 配置 FAIL,根因已定位)。
5. **数据库 schema 与 upstream 双向不互通**(见下节,真实生产缺口,未修)。

## 批次 Y-L3:CLI `reconstruct sparse` 透传 + L3 上游基线(2026-10-11)

### 透传契约(harness `8c7ce21` + 本地未提交测试)

- harness 已有 `_parse_mapper_options`(重复键/非 KEY=VALUE/空键/`-` 前缀键/
  `--random-seed` 与 `Mapper.random_seed` 并存 → 运行前拒绝)与 backend
  `_mapper_option_args`(逐字编码 `--KEY VALUE`),incremental 与 hierarchical
  同一路径;本批新增 CLI 层 `TestReconstructSparse`(7 例,两种 method 的透传
  与 5 类歧义输入 `assert_not_called`)。harness 四套件 `/usr/bin/python3`
  **433 passed**(pyenv 3.12 缺 `mcp` SDK 会 INTERNALERROR,不能作 gate 解释器)。
- 二进制侧实测:`build_app/bin/Colmap` 的 `mapper` 与 `hierarchical_mapper`
  均注册 71 个 `Mapper.*` 键与 `Mapper.random_seed`;未知键 exit 1
  `unrecognised option`,harness `_run` 转 `ColmapError`(fail-closed)。
- L3 runner 调用的是 `~/.local` 非 editable 安装的 harness 3.1.0(与仓库
  `acloudviewer_cli.py` 不同:缺 `ai` 命令组等);AST 比对
  `_parse_mapper_options`/`reconstruct_sparse` 两函数**逐字相同**,故 L3 证据对
  sparse 透传路径有效。
- 快速参考 `agent-integration/docs/CLI-QUICK-REFERENCE.md` 增 3b 可复现示例。

### L3 运行(fixture mini6,seed 20261009,SIFT CPU,max_features 2048,SIMPLE_RADIAL)

- 输入 digest(sha256 前 16):IMG_1024 `6a4e1b819c25bfd7`、IMG_1026
  `28116672a98a0038`、IMG_1028 `e75598533533028c`、IMG_1030 `5f98950105c1ac6f`、
  IMG_1032 `cf8a9ac377293be8`、IMG_1040 `9ea83fe99843d48f`;无模型资产(SIFT)。
- 二进制 sha256:candidate `e4b8771b378bf38d…`(rev `601806861b` dirty),
  upstream `680fb50f0c78686e…`(`a2bd15c2`,工作树干净)。

| 运行 | 配置 | reg | pts | obs | mtl | rms px | Sim3 center p95 | rot p95° | nn p95 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| l3-gate-3 cand | 默认 | 6 | 410 | 1255 | 3.0610 | 1.2929 | 1.6e-4 | 0.0096 | 0.0260 | PASS |
| l3-gate-3 up | 默认 | 6 | 410 | 1253 | 3.0561 | 1.2886 | | | | |
| l3-gate-4 cand | `num_threads=1`,`min_num_matches=20` | 6 | 406 | 1241 | 3.0567 | 1.2937 | 4.1e-4 | 0.0131 | 0.0275 | **FAIL** |
| l3-gate-4 up | 同上 | 6 | 406 | 1237 | 3.0468 | 1.2845 | | | | |

- 阶段秒(fe/match/map):gate-3 cand 23.23/1.29/1.07、up 22.69/1.72/0.73;
  gate-4 cand 23.25/1.36/1.02、up 22.14/1.50/1.04。
- 透传端到端证据:upstream `sparse/0/project.ini` 记录 `num_threads=1`、
  `min_num_matches=20`、`random_seed=20261009`。candidate **不写 project.ini**
  (upstream mapper 写出)——行为差异,待审计。
- gate-4 唯一违例:`mean_track_length` 相对差 0.00323 > 阈值 0.00208。pts 相同,
  故 mtl 差 ≡ obs 差 4/1237;阈值文件中 obs 阈值 0.0263 而 mtl 阈值 0.00208,
  二者由 gate-2 观测差标定(棘轮),彼此不自洽。**未改阈值**,FAIL 如实保留。

### gate-4 差异定位(逐级)

- keypoints:6 图逐字节相同。descriptors:仅 IMG_1032 第 2048 行第 9 元素
  38 vs 39(1 LSB),每个二进制跨运行逐字节稳定;该元素翻转 (1028,1032) 对
  1 个内点(55 vs 54),级联为 4 个观测差。image_id 按完成序分配(upstream
  跨运行亦变化),非差异源。
- 源码:fork `feature/sift.cc`、`feature/utils.cc` 相对 upstream 仅 include/
  宏名/`type` 赋值位置差异;VLFeat 源仅 host.h 宏行序、random.c 局部变量名差异;
  upstream 额外链接 `OpenMP::OpenMP_C`(fork C_FLAGS 已含 `-fopenmp`)。
- 假设 FMA(fork C++ `-march=x86-64-v3`,upstream 无 `-march`)**证伪**:仅
  sift.cc/utils.cc 以 `-ffp-contract=off` 重编并重链,descriptors 与默认
  candidate 逐字节相同、与 upstream 仍差同一元素;随后恢复,Colmap sha 回到
  `e4b8771b…`。剩余差异在源码以下(VLFeat C 目标码或解码库;fork Colmap
  额外动态加载系统 `libjpeg.so.8`,两者均静态含 build_app 的 OIIO 3.1.17),
  未进一步二分,不属上游源码契约。
- mapper 隔离实验(两二进制喂同一 DB)被下述 schema 缺口阻断。

### 数据库 schema 双向不互通(真实缺口)

- fork 打开 upstream 写出的 DB:`SQL logic error`(`database_sqlite.cc` 预编译
  `INSERT INTO images(... prior_qw ... prior_tz)`,upstream images 表无这些列)。
- upstream 打开 fork 写出的 DB:`no such column: corr_data_id` 后 abort(134)。
- schema 差异(sqlite `.schema` 实测):fork 多 `frame_images`、`rig_cameras`
  表;`images.prior_q*/t*`;`rigs.ref_camera_id`;`rig_sensors.qvec/tvec`
  (upstream `sensor_from_rig` BLOB);`frames.has_pose/qvec/tvec`;
  `pose_priors(data_id,sensor_id,sensor_type,…)`(upstream `corr_*` + 唯一索引);
  `descriptors` 列序(upstream `type` 在第 2 列);upstream 用 UNIQUE INDEX 而
  fork 用表内 PRIMARY KEY/UNIQUE。upstream 迁移只覆盖 upstream 自身历史 schema
  (inlier_matches 改名、`pose_priors.image_id`、two_view_geometries 补列、
  `descriptors.type`、identity→NULL 位姿),不含 fork-legacy 表。
- fork vs upstream `database_sqlite.cc` 差异行 4435;库外仅 `scene/rig.cc` 引用
  legacy 名。

## 批次 Y-DB/L3 收口:upstream schema 迁移、双向互通与阈值重标定(2026-10-11)

本节以本轮实跑结果**取代**上节“数据库 schema 双向不互通/mapper 隔离被阻断”
的旧结论。基线仍为 upstream
`a2bd15c2fbfa79ca3812f8b081708da5e47d5402`。

### DB test-first 失败集与最小移植

- 生产代码修复前，新增 schema/互通测试中 6 个 database 用例失败；另有
  `SynthesizeDataset.Nominal` 与 `Reconstruction.SetRigsAndFrames` 失败。
  `TestOpenUpstreamDatabase` 为 `SQL logic error`；
  `TestRigWritesUpstreamSensorEncoding` 为 `frame_images` 外键约束失败；
  two-view 缺失字段被写为零 blob 而非 NULL。
- fresh schema 按 upstream 4.3.0 对齐：`rigs/rig_sensors/frames/frame_data/images/
  pose_priors`、索引、FK、`sensor_from_rig` 7-double 小端 blob、two-view NULL
  语义与 `user_version=4030000`。descriptor 旧库沿用 upstream 自身的追加
  `type` 迁移，故物理列序可不同但按列名双向互通。
- fork-legacy 一次性迁移按表结构检测，不依赖私有新版本号；事务中保留
  rigs/sensors/frames/frame_data/images/pose priors/features/matches，丢弃 upstream
  不持久化的 frame pose 与 image prior。`legacy_alter_table=ON` 且临时关闭 FK，
  避免 SQLite 3.32.3 在 RENAME/DROP 时改写引用或级联删数据；失败抛错并回滚。
- 覆盖用例：`TestFreshSchemaMatchesUpstream`、`TestOpenUpstreamDatabase`、
  `TestRigWritesUpstreamSensorEncoding`、`TestTwoViewGeometryAbsentFieldsWriteNull`、
  `TestMigrateForkLegacyRigFrameSchema`、`TestMigrateImageKeyedPosePriors`。
  最后一项与生产移植同时落地，用于覆盖从代码审计发现的旧 upstream
  `pose_priors(image_id,...)` 改名后未回填风险，不声明为先红后绿证据。
- 最终审计补出 `ClearAllTables` 漏清 pose priors：先在 `TestMerge` 写入 prior，
  实跑得到 `NumPosePriors()==1` 的失败；随后复用 `ClearPosePriors()` 并将其
  const 性与本 fork 其余 clear API 对齐。`database_test` 最终 **29/29**。

### 双二进制 DB 互通实测

- case A：旧 fork DB(4030001)复制件 → 新 fork mapper 自动迁移 → upstream
  mapper；两段均 exit 0、均生成 sparse/0。迁移后 6 images、6 keypoints、
  6 descriptors、15 matches、15 two-view、0 FK violation，版本 4030000。
- case B：upstream DB(4030000)复制件 → 新 fork mapper → upstream mapper；两段
  均 exit 0、均生成 sparse/0，数据计数与 FK 检查不变。
- 两个方向均不再出现 `SQL logic error`、`no such column: corr_data_id` 或
  constraint 错误。旧 descriptor 表仅有列物理顺序差异，无 SQL/数据行为差异。

### RunMapper 与 L3 收口

- `RunMapper` 已按 upstream 写出 `project.ini`；`fix_existing_frames` 在归一化
  Sim3 下求解、导出前做逆 Sim3 回变换，新增测试先暴露缺口后修复。
- DB 移植后以 mini6、SIFT CPU、seed 20261009、`Mapper.num_threads=1`、
  `Mapper.min_num_matches=20` 连跑 3 次 calibration；所有非时间指标逐位一致。
  candidate 均为 reg=6、pts=406、obs=1241、mtl=3.0566502463、rms=1.2936505193；
  upstream 均为 6、406、1237、3.0467980296、1.2844693154。
- Sim3 后三次均为 center p95=0.0004135192351、rotation p95=0.01313247384°、
  symmetric point-NN p95=0.02749293910。阈值文件据三次实测最大值重标定；
  reg 与 points 维持 0 差异，obs/mtl/rms 阈值分别为 0.0032336297494、
  0.0032336297494、0.0071478577161。
- 最终正式 L3：candidate sha256 `7dc2349cc9f7a08d…`，upstream sha256
  `680fb50f0c78686e…`，`verdict=PASS`、`violations=[]`；输入 6 图完整 sha256
  记录于 `build_app/Testing/l3-db-port-final/comparison.json`。

### 回归与最终目录审计

- 公共头变化后全量 `cmake --build build_app -j8` exit 0；C++
  `ctest -R '_test$' -j4` **176/176**；Python reconstruction **397 passed / 3 skipped**。
  首轮全量 gate 的 `loma_test` 曾失败，单进程 6 次及与 Caspar/ALIKED 三轮并发
  均通过，完整 gate 重跑 176/176，判为非 DB 回归的瞬态外部后端失败。
- IMU 测试已按 upstream 细分到 `sensor/imu_test` **16/16**、
  `scene/scene_imu_test` **5/5**；fork 保留大写 accessor 命名适配。
- CMake 注册审计：本批新增/修改测试均由既有 `database_test`、`imu_test`、
  `scene_imu_test` 目标覆盖，无孤立测试文件或漏注册 target。
- 仍保留且不在本批扩张修复的差异：`rotation_averaging_ceres_test` 5 vs upstream
  11（prepared-problem/未标定 rig 等行为面）；`bitmap_test` 47 vs 75（当前生产
  契约已由现有色彩/metadata/JPEG/crop/rotation 用例覆盖，但测试名未逐一同步）；
  DB 管理 GUI 的 image prior 编辑不再持久化（upstream schema 无 image prior 列）；
  `SQLITE3_CALL` 仍以进程退出而非异常报告约束错误，故重复唯一键契约不能用
  同进程 `EXPECT_THROW`；`Database` 抽象 API 形态仍非 upstream 逐字同构。

## Batch Z: deterministic pipeline and pose-prior BA convergence (2026-10-11)

### Root-cause fixes

- `TwoGroupSampler` now declares both `is_randomized_sampler` and
  `is_parallel_safe_sampler`; `ProgressiveSampler` declares
  `is_randomized_sampler`. Seeded `(LO)RANSAC` therefore derives its sample
  stream from `RANSACOptions::random_seed` instead of ambient process PRNG
  state. The regression test runs 1/4-thread configurations and proves that
  different ambient seeds produce the same seeded sample sequence.
- `random_seed` now reaches initial generalized two-view geometry, next-frame
  absolute pose, triangulator complete/create, global retriangulation,
  automatic reconstruction, and the `sfm` CLI option surface.
- `IncrementalTriangulator` was reconciled with upstream traversal and state:
  BFS completion with a visited set, canonical image-pair merge trials,
  owner-stable scratch storage, cached track length before moves, key-based
  modified-point erasure, and `FlatHash*` containers across the mapper surface.
- Pose-prior BA now filters priors to configured camera images, requires at
  least three positions, robustly aligns or falls back to the two-camera
  gauge, normalizes at fixed scale, restores metric coordinates on every
  normalized exit, and rejects the unsupported Caspar+prior combination.
  Priors are added after parameterization with covariance/fallback-stddev
  whitening. Non-reference cameras use
  `AbsoluteRigPosePositionPriorCostFunctor` over sensor and rig blocks rather
  than incorrectly constraining the rig center. The covariance wrapper was
  also restored to upstream's post-autodiff residual/Jacobian whitening order.
- Upstream pose-prior tests were adapted for alignment outliers, insufficient
  priors, missing covariance, constant non-reference sensor extrinsics,
  robust optimization outliers, and Caspar rejection; all 8 Ceres BA tests
  pass.
- Triangulation now checks SVD status, computes ray angles through the clamped
  normalized dot product, and uses corrected observations in the legacy batch
  optimal API. A new rotated-pose test exposes a latent upstream bug in the
  otherwise unconsumed optimal API: `FindOptimalImageObservations` requires
  the transpose of the essential-matrix convention returned by
  `EssentialMatrixFromPose`. The fork intentionally fixes that defect; all 23
  triangulation tests pass.
- Fresh databases again stamp upstream revision `4030000`; legacy fork table
  layouts remain schema-detected and migrated without changing the fresh
  schema version. The stale insufficient-frame mapper death test now asserts
  upstream's normal `false` return.

### Full-precision fork/upstream scan

- The temporary probe ran five pipeline cases over seeds 0..29, one process
  per row and one mapper thread: 150/150 rows passed on both fork and upstream,
  with 0 crashes and identical model counts plus
  `(frames, images, points, observations)` for every row.
- The earlier `%.6f` statement was too strong. At `%.17g`, none of the five
  cases is bit-identical for every row. Maximum relative differences
  `(mean-reprojection-error, id-ordered point sum)` were:
  `hier_multi=(1.15e-6,6.37e-9)`, `inc_multi=(5.55e-9,6.20e-14)`,
  `inc_nontrivial=(1.05e-2,2.34e-6)`,
  `inc_prior_nonoise=(4.20e-15,3.58e-13)`, and
  `inc_prior_nontrivial=(1.49e-9,9.59e-13)`.
- A controlled upstream-vs-upstream rebuild with only
  `-ffp-contract=off` changed every case and reached
  `inc_nontrivial=(1.05e-2,2.35e-6)`, matching the scale of the largest
  fork/upstream residual. Therefore `%.17g` identity is not a valid
  cross-build acceptance criterion; structural identity, shared pass verdicts,
  and the real-data numeric gate are the supported evidence. All dump/scan
  probes and their two CMake test registrations were removed after capture.

### Final gates and L3 evidence

- Full build: `make -j20`, exit 0.
- C++: `ctest -R '_test$' -j4`, 174/174 passed after removing the two temporary
  scan targets. Python reconstruction: 397 passed, 3 skipped.
- L3 mini6 used seed 20261009, SIFT CPU, 2048 features, SIMPLE_RADIAL,
  `Mapper.num_threads=1`, and `Mapper.min_num_matches=20`. Three recalibration
  runs were non-time bit-stable, so the center-p95 threshold was updated from
  `0.000413519235057` to the new intentional-angle result
  `0.000414072916584` (a 0.13% shift); all other thresholds stayed unchanged.
- Final report: `build_app/Testing/l3-colmap-alignment-final-v3/comparison.json`,
  `verdict=PASS`, `violations=[]`. Candidate/upstream were respectively:
  registered images `6/6`, sparse points `406/406`, observations `1241/1237`,
  mean track length `3.0566502463/3.0467980296`, and reprojection RMS
  `1.2936503027/1.2844693154` px. Sim3-normalized p95 metrics were camera center
  `0.000414072916584`, rotation `0.01311550755 deg`, and symmetric point NN
  `0.02732701861`. Candidate/baseline SHA-256 start with
  `02e81db28a7f6bc2` / `680fb50f0c78686e`.

### Remaining alignment gaps

- The audit does not claim source- or test-inventory identity. Remaining
  non-gated surfaces include automatic-reconstruction mapper selection,
  relative-pose/Tangent-Sampson refinements, the historical upstream test-case
  inventory gap, and the intentional Graclus/METIS dependency split.
- These are separate API/coverage/dependency work items. No known divergence
  remains in the fixed-seed L3 structural outcome or its documented numeric
  acceptance envelope.

## Mapper, relative-pose, test-inventory, and partitioner closure (2026-10-12)

### First-principles decision record

The irreducible contracts are: one `--mapper` value must select exactly one
sparse pipeline; one relative-pose hypothesis must be scored in pixel units
with the same tangent-Sampson formula as upstream; one test inventory row must
identify one observable behavior; and one graph partition call must use a
header/library pair with one `idx_t` ABI. No new abstraction layer was added.

| Dimension | Selected minimal path | Rejected alternative | Reason |
|---|---|---|---|
| Mapper ownership | `AutomaticReconstructionController::Options::mapper` owns selection; the controller owns each pipeline lifetime | A mapper factory/registry | Three closed choices and one caller do not justify another owner |
| Relative pose | Existing `geometry`, `estimators`, and solver owners | A fork-only estimator facade | Upstream APIs already express the required rays, Jacobians, model, and error |
| Test inventory | File/test-name audit plus compiled behavioral gates | Treat equal file counts as coverage | File presence does not prove that a `TEST` behavior exists |
| Partitioner | Keep Graclus until one ABI-consistent METIS target exists | Include bundled 64-bit `metis.h` and link system 32-bit `libmetis.so.5` | The mixed ABI returns corrupt labels while reporting success |
| Chosen trade-off | Minimal direct dispatch and upstream-owned math; defer only the unsafe dependency swap | Broad reconstruction rewrite | Preserves ownership/lifetime and has executable evidence |

### Automatic reconstruction mapper selection

- Added `INCREMENTAL`, `HIERARCHICAL`, and `GLOBAL` to the controller options,
  CLI `--mapper`, CLI-Anything backend/Click command, and MCP schema. The
  controller directly owns each synchronous mapper lifetime; GLOBAL borrows the
  caller-owned `ReconstructionManager` through a no-delete shared alias only
  for the duration of `GlobalPipeline::Run()`.
- DA3 sparse mode rejects hierarchical/global rather than silently selecting a
  different path. CLI-Anything selects native COLMAP sparse mode when a
  non-incremental mapper is requested.
- `automatic_reconstruction_test` proves incremental/hierarchical scene parity
  and GLOBAL dispatch/model output. GLOBAL deliberately does not assert every
  frame is retained; upstream excludes it from the exact synthetic-scene gate
  for the same filtering behavior.

### Relative pose and tangent-Sampson

- Ported the upstream one-sided/shared-focal and generalized-absolute local
  refinements, ray/Jacobian `RefineRelativePose` and `RefineEssentialMatrix`,
  cheirality scoring, exact-zero denominator rule, and explicit
  `SquaredPixelGradientNorm` contraction.
- Removed the superseded point-only refine APIs and an unbuilt duplicate
  `estimators/relpose_one_sided_focal.{cc,h,_test.cc}` implementation. The
  production owner is now exclusively `estimators/solvers/`.
- Imported the upstream tangent-Sampson tests. Under the fork's
  `-march=x86-64-v3`, the algebraic pinhole identity differs by
  `2.2194168936644525e-14`; upstream built with the same flags has the identical
  result and passes its `1e-14` bound only with `-ffp-contract=off`. The fork
  therefore uses a `1e-13` bound, not a formula change.
- Registered the previously orphaned `reprojection_error_test`. Upstream's
  Ceres `GradientChecker` test crashes when a Ceres library built with Eigen
  16-byte alignment fills a result destroyed by a `-march=x86-64-v3` 32-byte
  caller. The fork compares analytical and autodiff residuals/Jacobians in the
  same translation unit instead; all 19 cases pass.

### Test inventory (upstream a2bd15c2 baseline)

The audit counts `TEST`, `TEST_F`, `TEST_P`, and `TYPED_TEST` declarations and
matches normalized suite/case names, with explicit aliases for moved files. It
is a coverage inventory, not a semantic proof. Current totals are upstream
`177 files / 1780 declarations`, fork `178 files`, with `1472` upstream cases
matched and `308` still unmatched. This batch added/registered solver tests for
generalized absolute pose, similarity transform, and solver utilities, plus
low-coupling upstream cases for homography refinement, tiny manifolds/solver,
alignment, motion averaging, Sim3 matching, texture sampling, polynomial
roots, reprojection errors, normalization, reconstruction reload/crop, pose,
and file extensions.

Files entirely absent, ranked by impact:

| Impact | Missing upstream file | Cases | Module / reason |
|---|---|---:|---|
| High | `inertial/preintegration_test.cc` | 16 | IMU integration and small-rate stability; owning module is absent |
| High | `inertial/preintegration_cost_test.cc` | 7 | IMU optimization residual; owning module is absent |
| Medium | `controllers/feature_extraction_test.cc` | 3 | extractor/importer controller integration |
| Medium | `feature/lightglue_test.cc` | 8 | optional LightGlue/ONNX inference surface |
| Low | `util/onnx_test.cc` | 9 | optional ONNX utility surface absent in this fork |

Remaining partial files, ordered by unmatched declaration count:

- High: `sensor/bitmap_test.cc` (50),
  `estimators/bundle_adjustment_test.cc` (21),
  `estimators/bundle_adjustment_caspar_test.cc` (17),
  `scene/database_test.cc` (17), `estimators/bundle_adjustment_ceres_test.cc`
  (15), `scene/frame_test.cc` (12), `scene/image_test.cc` (11), and
  `estimators/rotation_averaging_ceres_test.cc` (10).
- Medium: `util/misc_test.cc` (14), `geometry/pose_test.cc` (12),
  `util/cache_test.cc` (12), `geometry/gps_test.cc` (8),
  `feature/types_test.cc` (5), `mvs/advancing_front_meshing_test.cc` (5),
  `scene/track_test.cc` (5), `image/undistortion_test.cc` (4),
  `image/warp_test.cc` (4), `util/string_test.cc` (4), `feature/loma_test.cc`
  (3), `feature/onnx_matchers_test.cc` (3), `sensor/rig_test.cc` (3),
  `geometry/homography_matrix_test.cc` (2), `mvs/consistency_graph_test.cc`
  (2), `scene/rig_test.cc` (2), `sfm/incremental_mapper_test.cc` (2), and
  `sfm/incremental_triangulator_test.cc` (2).
- Low: `math/random_test.cc` (2), `optim/support_measurement_test.cc` (2),
  `scene/point2d_test.cc` (2), `scene/point3d_test.cc` (2),
  `scene/projection_test.cc` (2), `util/opengl_utils_test.cc` (2),
  `controllers/bundle_adjustment_test.cc` (1),
  `controllers/feature_matching_test.cc` (1),
  `estimators/gravity_refinement_test.cc` (1), `math/math_test.cc` (1),
  `mvs/fusion_test.cc` (1), `sensor/models_test.cc` (1),
  `sfm/observation_manager_test.cc` (1), and `util/types_test.cc` (1).

### Graclus / METIS evidence and convergence plan

- Fork runtime owner: `math/graph_cut.cc` calls Graclus `MLKKM_PartGraphKway`;
  `lib/Graclus` is built and linked as `graclus`. Upstream calls METIS 5.1
  `METIS_PartGraphKway` through `find_package(Metis)`.
- The difference is algorithmic, not merely a build flag: Graclus performs
  multilevel kernel k-means normalized cut, while METIS performs weighted k-way
  edge-cut partitioning. It already changes an exact hierarchical
  `scene_clustering_test` partition, although partitioner-independent coverage,
  size, connectivity, and L3 reconstruction gates pass.
- The current dependency graph is unsafe for a direct source replacement:
  `build_app/external/include/metis.h` declares `IDXTYPEWIDTH=64`, whereas the
  process loads Ubuntu METIS 5.1 with 32-bit `idx_t` through Ceres/CHOLMOD.
  A probe compiled against the 64-bit header and linked to the 32-bit library
  returned success but produced `edgecut=-4294967295` and corrupt partition
  labels. Graclus also exports METIS-4-named symbols into `ACloudViewer`, so
  keeping it while adding METIS 5 increases symbol-interposition risk.
- Safe convergence is one owned migration: build/provide one METIS 5.1 target
  with `IDXTYPEWIDTH=32` on every platform; make its generated header and binary
  inseparable; compile `graph_cut.cc` against that target; remove Graclus from
  `COLMAP_INTERNAL_LIBRARIES` and source collection; then require upstream
  `graph_cut_test` and exact `scene_clustering_test` partitions, ASan/UBSan,
  `ldd`/symbol checks, all 179 C++ tests, and L3. Do not add a runtime
  Graclus/METIS option: it would create two semantics for one owner.

### Additional upstream fixes captured during the audit

- Ported: optimal-triangulation convention `70dea881`; crop error preservation
  `c0a195a1`; reconstruction bookkeeping reset `826a1e55`; equal-percentile
  bounding boxes `9b8efa9a`; unified small-angle cutoff `75b60a7f`; stable
  small-angle rotation Jacobian `c60355a1`; case-insensitive file-extension
  regression coverage `e7177347` (implementation was already equivalent).
- `4a6d0cbe` polynomial coefficient-scale preservation remains a separate
  numeric migration; this batch added only the a2bd15c2 polynomial inventory.
  `9e653da3` (Ceres rotation averager), `f1ea91d0` (multiple calibrations per
  camera), and `4490d95a` (inertial RK4) are architectural/absent-module work,
  not safe leaf patches. Dependency/tooling/docs-only commits do not affect the
  reconstruction runtime gate.

### Verification

- Full build: `make -j16`, exit 0.
- C++: `ctest -R '_test$' --output-on-failure -j4 --timeout 900`, 179/179.
- Python reconstruction: 397 passed, 3 skipped. CLI-Anything: 616 passed.
- Fixed-seed L3: `build_app/Testing/l3-colmap-alignment-gap-audit-final/comparison.json`,
  `verdict=PASS`, `violations=[]`; candidate/upstream retain 6/6 registered
  images and 406/406 sparse points.
- `git diff --check` passes; no `.orig`, `.rej`, or temporary dump marker remains.
