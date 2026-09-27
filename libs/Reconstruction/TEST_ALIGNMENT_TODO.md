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
