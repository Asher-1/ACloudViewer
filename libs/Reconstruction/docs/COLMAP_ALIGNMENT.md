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
| W3-2b (frame-aware mapper/BA), W4, W7, W8, W10-W16 | pending |

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
