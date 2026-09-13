# COLMAP alignment status

The checked upstream tree is `/home/asher/develop/code/github/MVS/colmap` at
revision `dbb41680` (4.3.0.dev0). The machine-readable source of truth is
`libs/Reconstruction/colmap_alignment_manifest.json`; run
`python3 scripts/check_colmap_alignment.py` for the current matrix.
The implementation roadmap for closing the remaining gaps, including the
dual-path mesh texturing decision (upstream `mesh_texturer` as the default,
`image_texturer` kept as an alternative) and the fusion of upstream pycolmap
bindings into the existing `cloudViewer.reconstruction` Python module with
deduplicated interfaces, is tracked in
[COLMAP_ALIGNMENT_PLAN.md](COLMAP_ALIGNMENT_PLAN.md).

## Execution progress (2026-09-06)

Work packages from [COLMAP_ALIGNMENT_PLAN.md](COLMAP_ALIGNMENT_PLAN.md):

| WP | State |
|---|---|
| W1 database version migration | done (migration gate green) |
| W2 BA backend surface + PBA retirement | done |
| W5 solver unification | done (solvers/* registered, PoseLib hard dep) |
| W6 two-view increments | done (upstream two_view_geometry swap + parity gate) |
| W9 synthetic dataset | done (enabled in ColmapLib; synthetic_test 18/18 + upstream gps ENU gates green) |
| W3-1 correspondence graph cache | done (per-pair TwoViewGeometry cache + MaybeDecomposeRelativePoses + ray homography restored) |
| W3-2a frame-aware data model | done (Image/Frame/Rig/Reconstruction pointer wiring + Database pose_priors + OIIO ZLIB pin fix) |
| W3-2a build & fixture completion | done (Image::DataId corrected to the upstream form, NonRefSensors materialized by value, Rigid3d projection-error overload defined, Crop/Merge pointer reset + trivial wiring, six legacy test fixtures migrated to AddCameraWithTrivialRig/AddImageWithTrivialFrame; full build EXIT=0, ctest 73/77) |
| W4 GLomap global SfM stack | layer 1 done (solver stacks + PoseGraph + math helpers ported, compiled into ColmapLib; global_positioning/connected_components/spanning_tree tests green); W3-2b step 1 done (frame-aware DatabaseCache + Reconstruction::Load assembly, 2026-09-08) unblocked the three blocked suites: view_graph_calibration_test and pose_graph_test are now fully green; W3-2b step 2 numeric closure done (2026-09-09): rotation_averaging_test 15/15; layer 2 (sfm/global_mapper + controllers + CLI) pending on the correspondence_graph range work |
| W3-2b step 1 (frame-aware cache assembly) | done (upstream DatabaseCache::Options Load with rigs/cameras/frames/images/pose_priors, shared_ptr correspondence graph, CreateFromCache, frame-level image filtering, ENU conversion; upstream Reconstruction::Load pointer wiring + DeRegisterFrame; upstream database_cache_test ported 7/7; surfaced and fixed five fork defects: WriteRig bad_optional_access for NULL extrinsics, empty-keypoint abort, two-view geometry optional blob semantics, Rig AddSensor/SetSensorFromRig insert-vs-update overloads, SetRigFromWorld NaN placeholder) |
| W3-2b step 2a (RA numeric closure) | done (2026-09-09; rotation_averaging_test 15/15) via a sixth fork defect fix: Reconstruction copy ctor/assignment now rebinds frame/image back pointers (RewireObjectPointers, upstream parity) - the fork copied rigs_/frames_/images_ without rebinding, so any copied reconstruction silently read poses through the source object's stale pointers |
| W3-2b step 2 (stale-pointer hazard cleanup) | done (2026-09-09; the RA fix surfaced the same hazard everywhere: Image copy ctor/assignment now reset back pointers with default move members, Reconstruction::Transform legacy overload delegates to the frame-aware Sim3d path, Image projection-derived accessors read CamFromWorld when frame-wired with a legacy fallback, legacy fixtures (TestNormalize/TestTransform/TestComputeScale) keep trivial frames in sync; global_positioning_test assertions are no longer vacuously true - Nominal/RefineSensorFromRig genuinely green, MultiCameraRig retains a 0.169 deg alignment-rotation residual vs the 0.1 deg threshold, recorded as a convergence-quality edge case) |
| W3-2b step 2b (correspondence graph range migration) | done (2026-09-09; full upstream dbb41680 parity: flat_corrs/flat_corr_begs flattened storage with CorrespondenceRange FindCorrespondences, ExtractCorrespondences/ExtractTransitiveCorrespondences/ExtractMatchesBetweenImages output-parameter interfaces, NumMatchesBetweenImages, Finalize flattening without image removal, FlatHashMap image_pairs_; all consumers migrated to the Range/Extract forms; upstream correspondence_graph_test ported 11/11 incl. Finalize/NotFinalize parameterized cases; FeatureMatch gained upstream operator==) |
| W3-2b (step 3-4 mapper/BA) | done except: multi-sensor equirectangular composed BA functor (frozen composed-pose fallback) and the GP rig-branch convergence cases recorded under W4 |
| W8 model clustering/pruning | mostly done (scene files + tests, controllers/reconstruction_clustering, OptionManager AddReconstructionClustererOptions, CLI model_clusterer; remaining: mapper pruning consumption wiring with W3-3) |
| W10 model alignment/comparer | done (RunModelAligner ref_model_path/ref_is_gps/merge_image_and_ref_origins/Sim3d transform_path/enu-plane types; RunModelComparer max_proj_center_error + ImageAlignmentError/AlignmentErrorSummary output; upstream coordinate_frame_test ported incl. AlignToPrincipalPlane/AlignToENUPlane with Sim3d signatures) |
| W12 rig configurator | done (2026-09-13(5): the rig-config JSON stack (RigConfig/ReadRigConfig/ApplyRigConfig) and the RunRigConfigurator CLI command verified in place and registered (colmap.cc); the previous blocked status was stale) |
| W17.6 frozen-path mapping | done (manifest policy gained the permanent frozen_path_mapping table) |
| W18.4 sensor/models upstream API | done (2026-09-13(4) multi-batch session: real CameraModelId enum (numeric values unchanged, DB/binary IO cross-compatible), per-family ImgFromCam/CamFromImg/ImgFromCamWithJac API, upstream CRTP base hierarchy with is_base_of_v classification, 1537-line analytic jacobian.h verbatim, new families SIMPLE_DIVISION/DIVISION/SIMPLE_FISHEYE/FISHEYE/EUCM + RAD_TAN_THIN_PRISM_FISHEYE; camera.{h,cc} and all consumers migrated; upstream per-template models_test 21/21 + camera_test 23/23; specs.{h,cc} ported) |
| W18.7 util small files (util surface) | done except the dense->viewer surface loading call sites (fork product task) (2026-09-13: file.cc std::filesystem-ized; then the full upstream file.h surface landed (NormalizePath/GetNormalizedRelativePath/GetRecursiveFileList/GetDirList/HomeDir/blob IO/ReadTextFileLines); upstream file_test 18/18 + controller_thread_test 5/5 ported green; en-route fork defects (31) misc.h boost-era GetRecursiveFileList/GetDirList ambiguity retired with consumer migration, (32) download.cc HomeDir duplicate retired, (33) misc ReadTextFileLines duplicate definition (ODR) retired to file.{h,cc}, (34) GetPathBaseName upstream filename semantics with misc_test golden updated; 2026-09-13(3): ui/mesh_painter.{h,cc} + mesh shaders + model_viewer_widget wiring (surface_mesh member, UploadSurfaceMeshData, Render behind mesh_wireframe/mesh_color) ported from dbb41680, render_options.h gained show_camera_orientation/mesh_wireframe/mesh_color; oiio_utils.{h,cc} + glog_macros.h verified line-identical to upstream) |
| W17.2b database_sqlite interface | done (2026-09-13: Database became the upstream abstract interface with Factory/Register/Open over shared_ptr; all SQL state moved into SqliteDatabase in database_sqlite.cc behind the 17-line database_sqlite.h (OpenSqliteDatabase factory, pre-registered in Database::factories_); full construction-site migration across exe/controllers/feature-matching-family/ui/app/tests; fork float-descriptor surface kept on the interface; database_test 31, database_cache_test 7, rig_test 12, synthetic_test 18 all green) |
| W18.5 bundle_adjustment_ceres split (stage 1) | done (2026-09-13: BundleAdjuster became the abstract upstream base with protected options_/config_ + Options()/Config(); the entire Ceres implementation moved to the new estimators/bundle_adjustment_ceres.{h,cc} CeresBundleAdjuster (bundle_adjustment.cc 1009 -> 389 lines); CreateDefaultBundleAdjuster factory with the CASPAR-first dispatch kept inside Solve; 16 construction sites migrated to the factory + -> access; bundle_adjustment_test 15 cases green; remaining: BackendOptions pimpl, BundleAdjustmentSummary, covariance, caspar gate re-run) |
| W18.5 stage 2 (Problem + covariance) | done (2026-09-13(4): CeresBundleAdjuster::Problem() upstream accessors; estimators/covariance.{h,cc,test} fully ported with the fork split-qvec/tvec PoseParam adaptation preserving [rotation, translation] tangent ordering; covariance_test 7 parameterized cases green against ceres::Covariance at 1e-8) |
| W18.5 stage 3 (Summary surface) | done (2026-09-13(5): BundleAdjustmentTerminationType + BundleAdjustmentSummary + CeresBundleAdjustmentSummary (Create/mapping/summary() accessor) ported additively; BackendOptions pimpl restructure deferred to the caspar batch) |
| W3-3 incremental_mapper_impl | done (2026-09-13(4): IncrementalMapperImpl stateless algorithm class with FindFirstInitialImage/FindSecondInitialImage/FindNextImages/FindLocalBundle + rank helpers moved from the mapper, mapper.cc 1266 -> 896 lines with thin delegating members; upstream InitInfo orchestration stays in the mapper, camera-ray point-data refactor not pulled) |
| W3-2b frame-aware mapper (stage 1) | done (2026-09-13(6): mapper pose writes frame-aware via Frame::SetCamFromWorld with legacy fallback (dual-path pattern), RegisterNextImage estimates into locals + single commit + shadow sync, BA config frame-level constants (rig_from_world/sensor_from_rig) ported and honored in the problem assembly; RegisterNextImageFallback clarified as an upstream stale comment - no function to port; stage 2 = DatabaseCache pointer wiring + general/structure-less variants) |
| Upstream baseline drift | observed (2026-09-13: upstream pulled to `d3ccaf35`, Δ=33 commits vs scanned baseline `dbb41680`, incl. #4687 camera-models per-header split = the W18.4 template, GP4PS #4664, LO-RANSAC generalized pose #4690; full G1-G21 re-scan is a separate task; recorded in manifest `upstream_head_observed`) |

Testing: the whole Reconstruction test suite runs on **googletest**
(decision D6); `COLMAP_ADD_TEST` links `gtest_main` and upstream test files
can land unconverted. Full build is green and the suite reports 78/79 with
one remaining failure (the caspar split-intrinsics focal=0 pp=0 parity
case) that is independent of the W9 work: it fails with bit-identical rms
values under the old and new SensorFromRig/SetPoints2D code paths and its
sources are identical to HEAD. The four previously recorded environment
failures (polynomial, da3_depth_controller, sift GPU, gpu_mat) all pass in
the current environment, so the baseline comparison is not
environment-stable.

## W9 execution notes (2026-09-07)

Enabling `scene/synthetic.{h,cc}` surfaced and fixed five fork defects, all
recorded in the manifest entry: the `Rig::SensorFromRig` quaternion-order
bug (Eigen's Vector4d constructor is [x,y,z,w] while the fork qvec is
[w,x,y,z]), the WGS84 flattening and XYZToEll convergence parity in
`base/gps.cc`, the missing pose_priors table creation and prepared
statements (plus the ReadPosePriorRow column fix), the `Frame::AddImageId`
camera-data duplication for image_id != camera_id (ReadFrame now falls back
to frame_images only when frame_data is empty), and the stale
`Image::num_points3D_` after `SetPoints2D`. Database also gained the
upstream `ExistsTwoViewGeometry`/`UpdateKeypoints`/`ReadTwoViewGeometries()`
API names, and the test infrastructure gained `util/eigen_matchers.h`,
`CreateTestDir`/`CreateDirIfNotExists` and a gmock link for upstream tests.
The `Frame::AddImageId` legacy bridge (image_id == camera_id data-id
back-insertion) is kept for the frame_test/caspar fixtures and
`Database::ReadFrame` falls back to it only when a frame has no frame_data
rows.

## W4 layer 1 notes (2026-09-08)

The three GLomap solver stacks (rotation averaging with its impl header,
global positioning, view graph calibration), the PoseGraph data structure,
the math helpers (connected components, spanning tree, sparse Cholesky with
a portable Eigen LDLT), and the W10 estimator subset (alignment, Sim3,
Umeyama solver in estimators/solvers) are ported and compiled into
ColmapLib. Base/pose gained the upstream helpers (unit-vector averaging,
Markley quaternion averaging, TransformCameraWorld, gravity/angle-axis
utilities); base/rig gained MaybeSensorFromRig/SetSensorFromRig/
ResetSensorFromRig; base/frame gained MaybeRigFromWorld and SetCamFromWorld;
base/reconstruction gained the Sim3d Transform overload and summary
operator<<. Three of six upstream test suites pass; the other three are
blocked on one root cause: DatabaseCache::Load/Reconstruction::Load do not
yet assemble rigs and frames, so AddImage throws HasFrameId (gdb backtrace
confirmed the throw comes from Reconstruction::Load, not the ported
solvers). That assembly is exactly W3-2b step 1 and is the next task. Two
environment issues were also fixed en route: the missing cmake binary in
~/.local/opt (restored from the official 3.31.8 release) and a disk-full
cleanup of 57 GB of stale /tmp test probes.

## W3-2b step 1 notes (2026-09-08)

The frame-aware cache assembly landed: `DatabaseCache` now owns the full
object graph (rigs, cameras, frames, images, pose priors) behind the
upstream `DatabaseCache::Options` Load surface, including the
backwards-compatible per-camera-rig and per-image-frame fallbacks for legacy
databases, frame-level `image_names` filtering (all images of a matched
frame are loaded for multi-camera rigs), optional `load_all_images`, and
`ConvertPosePriorsToENU`. `Reconstruction::Load` assembles cameras, rigs,
frames, and images with upstream-parity existing-object validation and
wires every image into its rig/frame/camera back pointers; the three W4
test fixtures now run the upstream form (`Load(database, options)` +
`*cache.CorrespondenceGraph()`) verbatim. The port surfaced and fixed five
fork defects: (1) `Database::WriteRig` threw bad_optional_access for rigs
with unknown non-reference camera extrinsics (unknown extrinsics now
persist through the rig_sensors NULL path); (2) `FeatureKeypointsFromBlob`
aborted on databases without keypoints instead of returning an empty set;
(3) the two-view geometry blob bridge materialized F/E/H and the relative
pose as values, breaking the upstream optional semantics for spherical
pairs (all-zero blobs read back as nullopt); (4) `Rig` gained the upstream
`AddSensor(sensor, optional<Rigid3d>)` insert and
`SetSensorFromRig(sensor, optional<Rigid3d>)` update overloads, replacing
the insert-shaped wrapper that crashed the rotation-averaging expand path;
(5) `Frame::SetRigFromWorld` no longer rejects the upstream NaN
unknown-pose placeholder. `Reconstruction` gained the upstream
`DeRegisterFrame`. The upstream `database_cache_test` was ported as the
step-1 gate (7/7, including frame-level filtering and the legacy
no-rigs/no-frames compat cases). Full build EXIT=0; full ctest 83/85: the
two remaining rotation-averaging numeric cases
(WithoutNoiseWithNonTrivialUnknownRig, WeightedReducesErrorWithNoisyLowMatchEdges)
show a constant ~0.44 rad offset on non-reference cameras whose extrinsics
are unknown - the RA solver and its driver are line-identical to upstream
modulo API naming, so the residual is scoped to W3-2b step 2; the caspar
split-intrinsics failure is the pre-existing baseline recorded under
synthetic_dataset.

## W3-2b step 2a notes (2026-09-09)

The two remaining rotation-averaging numeric cases closed with a one-root-
cause fix. Diagnostic instrumentation (per-frame/per-sensor quaternion dumps
inside RunAndVerifyRotationAveraging) showed the solver output was correct
in the reconstruction's rig/frame containers while `image.CamFromWorld()`
still reported the source object's poses: the fork's Reconstruction copy
constructor and assignment copied `rigs_`/`frames_`/`images_` without
rebinding the frame->rig and image->frame/camera back pointers, so a
copied reconstruction silently read the source object's (stale) rig -
the ~0.44 rad offset was exactly the un-composed sensor_from_rig rotation
of the non-reference camera. `Reconstruction::RewireObjectPointers()` now
performs the upstream pointer-rebinding pass after every copy (upstream
scene/reconstruction.cc parity), and rotation_averaging_test is 15/15.
This fix is global: every copy of a Reconstruction (hierarchical mapper,
bundle adjustment expansion, tests) previously had the same stale-pointer
hazard. Full build EXIT=0; full ctest 84/85 (only the pre-existing caspar
split-intrinsics baseline failure remains). Remaining W3-2b step 2 work is
the correspondence_graph flat-range lookup migration (flat_corrs/
CorrespondenceRange/Extract* interfaces with consumer migration), scoped
before layer 2 of the GLomap stack.

## W3-2b step 2 stale-pointer cleanup notes (2026-09-09)

The RewireObjectPointers fix surfaced the same stale back-pointer hazard in
three more places, all fixed in this round:

- `Image` copy ctor/assignment now reset `camera_ptr_`/`frame_ptr_` (a
  copied image is a pure data copy; owning containers re-wire via AddImage
  or Reconstruction::RewireObjectPointers). Declaring the copy ctor had
  suppressed the implicit move ctor, which silently nulled re-wired pointers
  inside `AddImage`'s `emplace(std::move(image))`; default move members are
  declared explicitly.
- `Reconstruction::Transform` was split between a legacy
  `SimilarityTransform3` overload (rewrote per-image qvec/tvec only) and the
  frame-aware `Sim3d` overload; the legacy overload now delegates to the
  Sim3d implementation so rigs/frames/images/points transform together.
- `Image` projection-derived accessors (ProjectionCenter,
  ProjectionMatrix, RotationMatrix, ViewingDirection,
  InverseProjectionMatrix) read the pose from CamFromWorld when the image is
  frame-wired and fall back to qvec/tvec for standalone legacy images.
- Legacy fixtures that mutate `Image::Tvec` directly were updated to keep
  the trivial frames in sync (reconstruction_test TestNormalize/TestTransform,
  camera_rig_test TestComputeScale).

The previously-green global_positioning_test was reading GT poses through
stale pointers (its assertions were vacuously true). With live assertions,
Nominal and RefineSensorFromRig are genuinely green; MultiCameraRig retains
a 0.169 deg rotation residual against the 0.1 deg threshold - per
instrumentation all frame/sensor rotations are exact and the residual is the
Sim3-alignment-estimated rotation of the GP solution shape (final Ceres cost
0.149), i.e. a convergence-quality edge case under the seed-42
initialization, recorded as a known item for step 2b. Full build EXIT=0;
full ctest 84/85 (the only other failure is the pre-existing caspar
split-intrinsics baseline).

## W3-2b step 2b notes (2026-09-09)

The correspondence graph was fully aligned to the upstream dbb41680 design:

- `Finalize` flattens the per-point correspondence vectors into
  `flat_corrs`/`flat_corr_begs` and sets the `finalized_` flag. The fork's
  removal of images without observations was removed (upstream keeps every
  added image; the DatabaseCache observation-counter bridge already guards
  with `ExistsImage`).
- `FindCorrespondences` returns a `CorrespondenceRange` (identical semantics
  before and after `Finalize`); `ExtractCorrespondences`,
  `ExtractTransitiveCorrespondences` and `ExtractMatchesBetweenImages`
  replace the vector-returning variants; `NumMatchesBetweenImages` replaces
  the per-pair `NumCorrespondencesBetweenImages`; `image_pairs_` is a
  `FlatHashMap` (insert-only during construction, read-only afterwards) with
  the upstream `num_matches` field name.
- All consumers were migrated: Reconstruction's
  SetObservationAsTriangulated/ResetTriObservations, IncrementalMapper
  (FindInitialImage candidates, FindSecondInitialImage, two-view matches),
  and IncrementalTriangulator (Create/Merge/Continue).
- The fork-only `AddCorrespondences` entry was removed; `AddTwoViewGeometry`
  is the single upstream edge entry (landed in step 1).
- The upstream `correspondence_graph_test` was ported verbatim (11/11,
  including the Finalize/NotFinalize parameterized TwoView/ThreeView suites,
  OutOfBounds, Duplicate, and UpdateTwoViewGeometry[Swapped]);
  `FeatureMatch` gained the upstream `operator==`/`!=`.

Full build EXIT=0; full ctest 84/85. Remaining known items: the
`global_positioning_test` MultiCameraRig 0.169 deg convergence-quality
residual (see above) and the pre-existing caspar split-intrinsics baseline.
Next: layer 2 of the GLomap stack (sfm/global_mapper + controllers + CLI)
after triaging the MultiCameraRig item.

## W4 layer 2 notes (2026-09-09)

The GLomap layer-2 stack landed under strategy C:

- `sfm/global_mapper.{h,cc}`: `GlobalMapperOptions` (the fork's
  `ba_skip_fixed_rotation_stage` defaults true until the step-4
  `constant_rig_from_world_rotation` BA option exists) and `GlobalMapper`
  (Solve = rotation averaging -> track establishment -> global positioning ->
  iterative bundle adjustment -> iterative retriangulation and refinement).
  Track filters mirror the upstream `ObservationManager` loop
  (collect-then-delete, tracks shorter than 2 observations removed entirely,
  points retaining fewer than 2 observations removed, mean inlier error
  written back) with the NORMALIZED error computed as the upstream z=1
  normalized-plane distance (`CamRayFromImg` unprojection, chord fallback for
  spherical models).
- `controllers/global_pipeline.{h,cc}` + `util/base_controller.{h,cc}` +
  `util/cancellation.{h,cc}`: the upstream multi-component pipeline
  (rotation-averaging decomposition, per-component mapper runs with
  MODEL_UPDATE_CALLBACK progress, priority sorting). The fork's value-based
  `ReconstructionManager` required returning the reconstruction by value from
  `ReconstructSingleComponent` and re-inserting it into the manager slot: a
  non-owning `shared_ptr` alias dangles across manager vector reallocation.
- CLI: the `global_mapper` command is registered end to end
  (`exe/sfm.cc` RunGlobalMapper/RunGlobalMapperImpl, `exe/colmap.cc`,
  `OptionManager::AddGlobalMapperOptions` binding the fork's existing option
  fields). The `rotation_averager`/`view_graph_calibrator` commands are
  deferred on `controllers/rotation_averaging` and
  `estimators/gravity_refinement`.
- Fork defects fixed while landing (continued numbering from step 2a):
  (10) `DeleteAllPoints2DAndPoints3D` rebuilt every Image and dropped the
  back pointers (upstream only clears Points2D); (11) the fork-only
  `Image::SetPoints2D` empty-CHECK blocked the upstream repopulation path;
  (12) `GlobalPositioner::ConvertBackResults` used insert-shaped AddSensor
  instead of update-semantics SetSensorFromRig; (13) the fork BA still
  optimizes legacy qvec_/tvec_, so the global mapper's RunBundleAdjustment
  syncs those buffers from the frame-aware poses before solving and writes
  the optimized poses back through Frame::SetCamFromWorld afterwards;
  (14) the Transform rig segment was another AddSensor call site;
  (15) `Database::DeleteTwoViewGeometry` was missing; (16) `TearDown` was the
  image-level version and is now the upstream frame-level teardown;
  (17) `Reconstruction` gained move members (node-stable containers keep back
  pointers valid).
- Without-noise closure used the upstream `num_obs_tolerance` matcher
  parameter and explicit `SetPRNGSeed` pinning in fixture tests - both are
  upstream mechanisms (candidate upstream patches for fixture determinism);
  the fixture geometry divergence was traced to libstdc++
  std::shuffle/uniform_int_distribution differences between gcc 9 (this
  host) and the upstream CI toolchain.

Result: `global_pipeline_test` 12/13, `global_mapper_test` 3/5, all other W4
suites green; full build EXIT=0. Known items: the GP rig-branch basin cases
(`global_mapper_test` WithoutNoiseWithNonTrivialKnownRig,
`global_pipeline_test` MultiComponentsWithUnknownSensorFromRig), the
pre-existing caspar split-intrinsics baseline, and the deferred
rotation_averager/view_graph_calibrator CLI commands.

## Implemented in this tree

- Fundamental-matrix RANSAC now has a configurable tiny Sampson local
  optimizer. It preserves rank two at every accepted step and is enabled
  by `TwoViewGeometry::Options::use_sampson_refinement` (default `true`).
- Hierarchical mapping retains disconnected images by passing all database
  image IDs into scene clustering.
- `GlobalMapperController` reconstructs verified-match connected components in
  parallel and writes each component as an independent model.
- Existing qLightGlue/AICore inference remains ggml-backed; no ONNX Runtime
  target is added by reconstruction.

## Evidence gates

- The upstream one-sided-focal 6-point solver is a PoseLib call. A pinned
  `3rdparty/PoseLib/poselib.cmake` recipe and adapter are available behind
  `-DRECONSTRUCTION_FETCH_POSELIB=ON`, and the asymmetric known-second-camera
  RANSAC path uses it. `relpose_one_sided_focal_test` now provides a deterministic
  exact six-point gate (focal relative error <= 1e-3, normalized E error <= 1e-2,
  residual < 1e-8) plus a sub-minimal rejection case. Its calibrated side now
  carries `CamRayWithJac`, and RANSAC uses the tangent Sampson denominator.
  `reconstruction_camera_rig_parity_gate` checks the distorted-ray path against
  an independent numerical tangent gradient, including TinySolver refinement
  accuracy and sub-minimal rejection.
- EQUIRECTANGULAR uses a signed 3D `ImgFromCam` entry point, so full-sphere
  reprojection and angular geometry retain the rear-hemisphere direction.
  `reconstruction_camera_rig_parity_gate` also runs the analytic spherical-ray
  Jacobian, full-sphere projection, and a rotated/translating relative-pose
  fixture. `Rig`/`Frame` now serialize tagged generic sensor/data records in
  v2 text and binary formats while retaining v1 readers. SQLite migrates the
  old camera reference to typed `ref_sensor_id`/`ref_sensor_type`, and
  `rig_sensors`/`frame_data` persist non-camera references and data-only
  Frames without changing legacy camera/image callers. The variable-pose,
  constant-pose, and Rig equirectangular Ceres factors each have an AutoDiff
  versus central-difference Jacobian gate away from the seam and poles.
- Generated Symforce-Caspar f32/f64 kernels are imported through
  `3rdparty_caspar`. `BundleAdjuster` dispatches PINHOLE/SIMPLE_RADIAL BA to
  Caspar and otherwise keeps the existing Ceres path. Caspar receives a
  camera-only `Frame` pose and the rig's fixed `sensor_from_rig` transform,
  then synchronizes its solved Frame back to the member images.
  `reconstruction_caspar_parity_gate` directly checks its CPU-Ceres RMS bound,
  Frame/image composition invariant, the four merged factors, and all 11
  upstream split-intrinsic factors. The split gate confirms fixed focal/extra
  and principal-point groups are never written back and compares the active
  graph against Ceres. Ceres is hard-configured CPU-only and rejects
  `CERES_ENABLE_CUDA=ON`, which keeps packages independent of a specific CUDA
  toolkit. Caspar is the independent GPU BA path, so its performance
  comparison is always against CPU Ceres. Spherical and other camera models
  stay on Ceres because upstream Caspar has no factors for them.
- HIP PatchMatch is retained as default-off experimental code, but is explicitly
  deferred from the active alignment and release hardware matrix: this project
  has no ROCm compiler or device on which to establish depth-map parity. It is
  therefore not a current release blocker or remaining implementation task.
- LoMa weights are listed with their COLMAP-pinned SHA-256 values in
  `core/AICore/tools/loma_sources.json`. DaD, DeDoDe-B/DeDoDe-G, and B/R/L/G
  matcher graphs run through GGUF/ggml only; the legacy pipeline persists typed
  float descriptors and a real ten-image CUDA SfM run registered 10/10 images
  with 9020 sparse points and 68956 track observations. CUDA uses an exact F32
  materialized SDPA graph (`F32 GEMM -> F32 softmax -> F32 GEMM`) because the
  fused kernel converts F32 K/V to F16. Direct F32 ViT patch embedding also
  bypasses CUDA's TF32 IGEMM route. On CUDA 11.8 / RTX 3060, the pinned ONNX
  ViT-L token fixture passes at max absolute error `1.9836426e-4` and relative
  L2 `6.7295686e-6` (limits `2e-3` and `2e-4`) without an environment-variable
  workaround. The B/R/L/G 256D matcher ONNX reference P/R gates each return
  precision and recall `1.0`. No ONNX Runtime is added to ACloudViewer.

## OpenImageIO migration

Modern COLMAP removed FreeImage in commit `8e014c5b` and uses OpenImageIO.
Reconstruction now follows that design: `Bitmap` owns ABI-neutral packed image
storage and the private `3rdparty_openimageio` interface target links a fixed
OIIO 3.1.17.0 source build without leaking its embedded fmt headers into
unrelated targets. The recipe turns off Python, Qt, OpenCV, and unused format
modules. In particular `USE_OPENCV=OFF` prevents a host OpenImageIO package
from adding system OpenCV 4.2 to reconstruction's runtime closure.

OIIO's own pinned source recipes construct required dependencies that are
missing or too old, including Imath/OpenEXR/OpenColorIO on Ubuntu focal. TIFF
is always built from OIIO's pinned local source recipe to avoid the macOS
Mono.framework ABI hazard. This keeps the OIIO source version and its required
image/color compatibility level the same on Ubuntu, macOS, and Windows while
allowing their active toolchain prefix to satisfy compatible prerequisites.
`EMBEDPLUGINS=ON` keeps enabled format readers in the OIIO library, so a
separate format-plugin deployment is not needed. FreeImage and the
Conda/system OIIO package inputs are removed.

The focused `reconstruction_oiio_parity_gate` remains available for local image
I/O, EXIF, resize, undistortion, and texture validation, but GitHub CI does not
run it. PostInstall resolves the actual OIIO dynamic dependency graph, copies
every non-system dependency into each independently installable ACloudViewer,
CloudViewer, and Colmap component, and rewrites copied macOS dylibs to
`@rpath`. It then rejects an incomplete payload. Ubuntu, macOS, and Windows
installer jobs check this closure before adding a platform dependency directory
to the loader search path. The release record remains partial until the pinned
source build and package path have completed on all three platforms.

## W10 + W8-CLI + W17.6 notes (2026-09-12)

- **W10**: `estimators/alignment.{h,cc}` were already upstream-shaped
  (AlignReconstructionToLocations/ViaReprojections/ViaProjCenters/ViaPoints,
  ImageAlignmentError, AlignmentErrorSummary). The CLI layer caught up:
  `exe/model.cc` RunModelAligner now carries `ref_model_path`,
  `ref_is_gps`, `merge_image_and_ref_origins`, stores the alignment as Sim3d
  via `Sim3d::ToFile`, and supports the upstream `enu-plane` /
  `enu-plane-unscaled` alignment types; RunModelComparer gained
  `max_proj_center_error` and the ImageAlignmentError/AlignmentErrorSummary
  statistics output (CSV header matches upstream).
- `AlignToPrincipalPlane`/`AlignToENUPlane` migrated to upstream Sim3d
  signatures. Important: the fork's legacy `RotationMatrixToQuaternion`
  returns the conjugate convention relative to `Eigen::Quaterniond(rot_mat)`,
  so both functions now construct `Sim3d` with the standard Eigen
  matrix constructor (upstream parity); the flip check uses the
  frame-aware `TransformCameraWorld` + `Rigid3d::Inverse` form. The
  upstream `coordinate_frame_test.cc` was ported (7 suites); the
  AlignToENUPlane golden values are checked with a 0.1 absolute tolerance
  because the ECEFToEllipsoid iteration is ill-conditioned for the tiny
  fixture ECEF coordinates and gcc9 FMA contraction drifts the solution
  (~1e-5 relative, same root cause as the W4 std::shuffle divergence).
- `base/gps.h` gained the upstream `GPSTransform::Ellipsoid` alias plus
  `EllipsoidToECEF`/`ECEFToEllipsoid` (thin delegates to the legacy
  EllToXYZ/XYZToEll implementations).
- **W8 CLI**: `controllers/reconstruction_clustering.{h,cc}`
  (ReconstructionClustererController) ported; OptionManager gained the
  `reconstruction_clusterer` field + `AddReconstructionClustererOptions`
  (min_covisibility_count/min_edge_weight_threshold/min_num_reg_frames);
  `model_clusterer` CLI registered end to end. The fork's value-based
  ReconstructionManager required moving the clustered reconstruction into
  the manager slot by value. Remaining: mapper pruning consumption wiring
  (`ba_global_ignore_redundant_points3D[_min_coverage_gain]`) lands with
  W3-3.
- **W17.6**: manifest `policy.frozen_path_mapping` records the permanent
  legacy-location mapping table (base/pose -> geometry/pose + estimators/pose,
  base/camera_models -> sensor/models, util/bitmap -> sensor/bitmap, etc.).
- **W12** recorded as blocked: RunRigConfigurator depends on the upstream
  scene/rig rig-config JSON stack (~300 lines incl. the anonymous helpers)
  that the fork does not carry yet.
- OptionManager housekeeping: `Reset()` now also clears `options_path_`.
- **W13 graceful shutdown**: consumption side fully wired. `exe/colmap.cc`
  upgraded to the upstream `Command` struct with
  `kSupportsGracefulShutdown`; 16 long-running commands are marked; the
  main loop installs `ScopedSignalHandler` for marked commands and returns
  `128+signal` through `GetExitCode()`; `Thread::IsStopped()` now
  cooperatively stops on the first signal; `BundleAdjustmentController`
  feeds `IsStopped()` into the W2 `check_if_stopped` Ceres callback.
  Checkpoint call sites inside not-yet-ported exe commands
  (RunPointTriangulator/RunPointFiltering upstream form) land with W11/W3-3.
- **W15 completed**: `texture_mapping_test` replaced with the upstream
  15-case suite (15/15 green), closing the test-count gap; combined with the
  earlier CLI/IO work the mesh_texturer path is complete. The
  `texturing_type` dispatch (D1 default) in the automatic reconstruction
  controller remains a fork-side product task, tracked separately.
- **W11 partial**: `feature/index.{h,cc}` (FeatureDescriptorIndex,
  FAISS-backed flat/IVF/IVFPQ/ScalarQuantizer hierarchy) and `index_test`
  ported (4/4 green; TypeMismatch skipped because the fork's
  FeatureDescriptorsFloat is a bare Eigen alias without the upstream .type
  wrapper - type validation defers to W18.3). The matcher_cache consumption
  and geometric_verifier/guided_geometric_verifier CLIs remain with the
  pairing layer.
- **W15 partial**: `mvs/texture_mapping.cc` delta audit complete - the
  127-line diff against upstream is entirely mechanical fork-infrastructure
  adaptation (fork Bitmap `InterpolateBilinear`/`GetPixel` out-param API,
  `PrintHeading2`, `CGAL_ENABLED` macro name); algorithm behavior identical.
  Backfilled `NodeHashMap` x3 and `THROW_CHECK_LE`/`THROW_CHECK` to the
  upstream spelling. `util/ply.{h,cc}` gained the upstream `ReadPlyMesh`
  (344-line plain+textured PLY reader, ASCII/binary both endians,
  kMaxPlyVertices/Faces guards) and `HasPlyMeshFaces`;
  `util/string.{h,cc}` gained the locale-independent `StringToDouble`.
  `exe/mvs.cc` `RunMeshTexturer` ported end to end (workspace + PLY mesh ->
  atlas PNG + textured PLY BIN/TXT) with
  `OptionManager::AddMeshTextureMappingOptions` and the `mesh_texturer` CLI
  registration. Remaining: `texture_mapping_test` 4->15 upstream cases and
  the `texturing_type` dispatch (D1 default) in
  controllers/texturing_controller + GUI panel.
- **W14 partial**: `util/timestamp.h` + `timestamp_test` ported
  (`util/types.h` gained the upstream `timestamp_t`/`kInvalidTimestamp`);
  `ImageReaderOptions.as_rgb` (upstream default false) exposed as
  `ImageReader.as_rgb` and wired through `image_reader.cc`.
- Gates: full build EXIT=0; full ctest green except the two recorded GP
  rig-branch cases (global_pipeline_test MultiComponentsWithUnknownSensorFromRig,
  global_mapper_test KnownRig/UnknownRig) and one flaky run of
  optim/least_absolute_deviations_test RidgeRegularization that passed on
  five consecutive re-runs (environment-instability family already noted
  above).

## W12 + W8 wiring + W7 closure + W15 dispatch notes (2026-09-13)

- **W12 rig configurator (unblocked and landed)**: `base/rig.{h,cc}` gained the
  upstream `RigConfig` struct, `ReadRigConfig` (boost ptree JSON) and
  `ApplyRigConfig` with the anonymous helpers; `Reconstruction` gained the
  upstream `SetRigsAndFrames` (re-wires image back pointers); `Frame` gained
  `ClearDataIds`; `RunRigConfigurator` + the `rig_configurator` CLI are
  registered end to end. The upstream `rig_test` config suites were ported
  (12/12 green). Fork adaptations: concrete `Database(":memory:")` in tests
  (no `Database::Open` factory until W17.2b), Camera getter/setter spelling,
  `SetModelId` params pre-fill reset, `Frame::ImageIds()` yielding `image_t`.
  Fork defects fixed en route: (27) `Database::UpdateRig` legacy `rig_cameras`
  bridge lacked the `WriteRig` NULL-extrinsics skip (bad_optional_access);
  (28) `UpdateRigsAndFramesFromDatabase` non-ref sensor condition was
  inverted (upstream add-when-missing semantics).
- **W8 pruning consumption wired** (closes the G9 tail item):
  `BundleAdjustmentConfig` gained `IgnorePoint`/`IsIgnoredPoint` with the
  `AddPointToProblem` guard and the SetUp observation skip;
  `IncrementalMapper::Options` gained
  `ba_global_ignore_redundant_points3D[_min_coverage_gain]` and
  `AdjustGlobalBundle` runs the first-pass pruning through
  `FindRedundantPoints3D` (10-registered-image small-reconstruction
  threshold). The upstream second "optimize redundant points" pass of
  `IterativeGlobalRefinement` remains with W3-3 (frame-aware ba_config API).
- **W7 pose prior stack closed**: `estimators/cost_functions/pose_prior.h`
  ported (upstream functors + the fork's split-block
  `AbsolutePosePositionPriorQvecTvecCostFunctor`); `BundleAdjuster::SetPosePriors`
  performs the robust Sim3 `AlignReconstructionToPosePriors` before the solve
  and adds a per-image position-prior residual (upstream
  `CreatePosePriorBundleAdjuster` parity folded into the fork's single
  `BundleAdjuster` until W18.5); mapper options `use_prior_position` /
  `use_robust_loss_on_prior_position` / `prior_position_loss_scale` with the
  `NumRegisteredPosePriors >= 3` gauge in `AdjustGlobalBundle`; the
  `pose_prior_mapper` CLI and `UpdateDatabasePosePriorsCovariance` are
  registered end to end.
- **W15 texturing_type dispatch (D1)**:
  `AutomaticReconstructionController::Options` gained `TexturingType`
  (`MESH_TEXTUREUR` default, `IMAGE_TEXTUREUR` alternative) and the
  AutomaticReconstructionWidget gained the "Texturing engine" combo. The
  IMAGE branch currently logs a fallback warning: the fork's `MvsTexturing`
  engine has no workspace -> PinholeCameraTrajectory adapter and no
  validation gate, so the wiring is recorded as a fork product task.
- **GP rig-branch diagnostic**: disabling FMA contraction on
  `global_positioning.cc` (-ffp-contract=off) was tested and did NOT close
  the two rig-branch cases; the attribute was reverted and the cases remain
  the recorded gcc9-toolchain baseline (libstdc++ shuffle-sequence family).
- Gates: full build EXIT=0; `rig_test` 12/12; the Reconstruction ctest suite
  is green except the recorded environment baselines (AICore asset-missing
  family, the two GP rig-branch cases, the LAD RidgeRegularization flake).


## Directory structure realignment (2026-09-13)

The fork `src/` layout now matches the upstream `src/colmap/` directory
structure. `base/` was dissolved via `git mv` (history preserved):

- entities/IO -> `scene/` (camera, database*, frame, image, point2d/3d,
  projection, reconstruction*, rig, scene_clustering, track, two_view_geometry,
  visibility_pyramid, correspondence_graph)
- models/math -> `geometry/` (essential/homography_matrix, gps, pose,
  triangulation, similarity_transform[fork-legacy]) and `math/` (graph_cut,
  polynomial; plus util/{math,matrix,random})
- image ops -> `image/` (line, undistortion, warp); `image/` and `math/` now
  have exactly the same file counts as upstream
- camera stack -> `sensor/` (camera_models -> models.{h,cc} merged umbrella;
  camera_specs -> specs; bitmap -> sensor/bitmap)
- pipelines/options -> `controllers/` (image_reader, option_manager,
  feature/extraction -> feature_extraction)
- BA stack -> `estimators/` (optim/bundle_adjustment{,_caspar}; legacy merged
  cost functors -> cost_functions/cost_functions.h)
- controllers file renames: hierarchical_mapper -> hierarchical_pipeline,
  incremental_mapper -> incremental_pipeline (class names kept for now)
- mvs/meshing.{h,cc} split into upstream `poisson_meshing` + `delaunay_meshing`;
  `PatchMatchOptions` extracted to upstream `patch_match_options.{h,cc}`

608 include paths were rewritten across the module, `app/`, and plugins; the
full build and the ctest suite pass with no new failures (only the previously
recorded environment baselines). Known pending file-level splits are recorded
in `colmap_alignment_manifest.json` under `directory_structure_alignment`.

## Compile-fix round + W18.4 umbrella + W18.7 file.cc notes (2026-09-13)

The post-realignment tree no longer compiled; two root causes were fixed:

- `sensor/models.h` (the legacy merged camera-model umbrella) was a broken
  concatenation product of the base/->sensor/ dissolve: duplicated
  pragma-once/include blocks, dangling `inline std::vector<size_t>`
  declarations, an unclosed `namespace colmap` - ColmapLib failed with
  `FullOpenCVCameraModel has not been declared` cascading out of the
  CAMERA_MODEL_CASES macro expansion. Fixed in the upstream #4687 shape:
  new `sensor/models/runtime.h` (fork-API twin of the upstream file of the
  same name) carries CAMERA_MODEL_CASES/SWITCH_CASES, kInvalidCameraModelId,
  the CameraModel* declarations and the inline WorldToImage/ImageToWorld/
  ImageToWorldThreshold dispatch plus the fork's CameraModelIs*/
  CamRayFromImg classification, all inside `namespace colmap` (all current
  macro consumers - models.cc and the two UI files - already live inside the
  namespace, so the move is transparent); `sensor/models.h` became the
  upstream-style umbrella (comment + single include, ~60 lines). The fork
  keeps its int model_id + WorldToImage/ImageToWorld API; migrating to the
  upstream CameraModelId enum + ImgFromCam/CamFromImg surface is a separate
  API-alignment task (several hundred call sites in Camera/BA/undistortion).
- `util/file.cc` used boost::filesystem (FileCopy, GetParentDir,
  GetRelativePath) without including its header
  (`'boost::filesystem' does not name a type`). Fixed to pure
  std::filesystem: `FileCopy` calls copy_file/create_hard_link/create_symlink
  directly; `GetRelativePath` is one `std::filesystem::relative` call
  replacing the boost canonical-iterator walk; `GetParentDir` keeps the fork
  `std::string` signature and the misc_test-verified `"/" -> ""` edge case as
  an explicit branch. The fork keeps the `CopyType` enum name (upstream
  FileCopyType) and the string-returning GetParentDir to avoid breaking
  exe/image.cc, image_reader.cc, undistortion, texturing_controller and the
  UI consumers.

Upstream baseline drift discovered the same day: the upstream tree was
pulled to `d3ccaf35` (33 commits beyond the scanned `dbb41680` baseline,
896 files, mostly the #4713 SPDX header rewrite). The functional delta that
directly affects this plan is #4687 (camera models split into self-contained
`sensor/models/` headers incl. `runtime.h`, `jacobian.h`, `division.h`,
`eucm.h`, `fisheye.h` over the CameraModelId enum + ImgFromCam/CamFromImg
API) - it is the template for the remaining W18.4 work; further functional
deltas (#4664 GP4PS, #4690 LO-RANSAC generalized pose, #4696 min_inlier_ratio,
#4684 ScaleWeightedCostFunctor, pycolmap cleanup) are recorded in the
manifest `upstream_head_observed` note pending a full gap re-scan.

Gates: full build EXIT=0; Reconstruction ctest reports only the recorded
baselines (test_image_depth AICore asset-missing family, polynomial
environment baseline, the two GP rig-branch cases in global_mapper_test,
global_pipeline_test MultiComponentsWithUnknownSensorFromRig; the extra
MultiComponents failure under parallel ctest passes standalone and is the
same GP convergence-quality family).

## W17.2b database_sqlite interface notes (2026-09-13)

`scene/database_sqlite.h` was extracted per the upstream form and the
Database interface work (W17.2b) closed in the same batch:

- `scene/database.h` is now the abstract upstream interface: `virtual
  ~Database() = 0`, `Factory`/`Register` and `static std::shared_ptr<Database>
  Open(path)`, the static pair-id helpers kept inline, and
  BeginTransaction/EndTransaction moved to protected virtuals with
  `transaction_mutex_` on the base (DatabaseTransaction stays a friend).
- `scene/database_sqlite.cc` now defines `SqliteDatabase : public Database`
  holding every scrap of SQL state (handle, prepared statements, table
  creation, migrations, `update_schema_mutex_`, the upstream `path_` member)
  with the static `Open` factory and the file-tail `OpenSqliteDatabase`
  free function; `Database::factories_` is pre-registered with it inside
  `scene/database.cc` (upstream parity) alongside Register/Open/Merge and
  DatabaseTransaction.
- `scene/database_sqlite.h` is the 17-line upstream form
  (`kInMemorySqliteDatabasePath` + `OpenSqliteDatabase`). The fork-specific
  float-descriptor surface stays on the interface (local table, W1 note).
- Construction-site migration: every `Database x(path)` value became
  `Database::Open(path)` across exe/, controllers/ (FeatureMatcherCache,
  ImageReader and FeatureWriterThread take raw pointers, fed with
  `.get()`), the `feature/matching.h` matcher family (six `Database
  database_` value members became shared_ptrs), the ui/ and
  app/reconstruction DatabaseManagementWidget trees, mvs/texturing, and nine
  test files. Fork defects fixed en route: (29) the interface rewrite
  initially dropped `UpdateTwoViewGeometry` (caught by
  view_graph_calibration.cc), restored with the SqliteDatabase override;
  (30) SqliteDatabase needed the upstream `path_` member for the
  static-Open flow.
- The LAD RidgeRegularization/0 case turned stably red after the rebuild:
  `Eigen::SimplicialLLT` does not guarantee failure on numerically singular
  PSD systems, the solver is line-identical to upstream, and no source in
  optim/ changed - recorded with the polynomial/FMA environment-drift
  family rather than as a regression.

Gates: full build EXIT=0; database_test 31, database_cache_test 7, rig_test
12, synthetic_test 18, pose_graph_test, camera_test and
observation_manager_test green; full Reconstruction ctest shows only the
recorded baselines (44 AICore asset-missing cases + polynomial + the GP
rig-branch pair + LAD).
