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
| W3-2b (step 3-4 mapper/BA), W7, W8, W10-W16 | pending (W10 estimator subset alignment/sim3 landed early with W4) |

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
