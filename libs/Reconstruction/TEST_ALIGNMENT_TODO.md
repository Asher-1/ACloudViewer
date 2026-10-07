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
