# COLMAP alignment status

The checked upstream tree is `/home/ludahai/develop/code/github/colmap` at
revision `a395b826`. The machine-readable source of truth is
`libs/Reconstruction/colmap_alignment_manifest.json`; run
`python3 scripts/check_colmap_alignment.py` for the current matrix.

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
  `reconstruction_caspar_parity_gate` directly checks its Ceres RMS bound,
  Frame/image composition invariant, the four merged factors, and all 11
  upstream split-intrinsic factors. The split gate confirms fixed focal/extra
  and principal-point groups are never written back and compares the active
  graph against Ceres. The current Ceres build reports no cuDSS and falls back
  to CPU sparse solving, so this is not a Ceres-CUDA performance result. It
  remains partial only until a representative-scene Ceres-CUDA A/B gate is
  complete. Spherical and other camera models stay on Ceres because upstream
  Caspar has no factors for them. The factor implementation is complete; the
  missing evidence is a representative Ceres-CUDA versus Caspar timing A/B.
- HIP PatchMatch is retained as default-off experimental code, but is explicitly
  deferred from the active alignment and release hardware matrix: this project
  has no ROCm compiler or device on which to establish depth-map parity. It is
  therefore not a current release blocker or remaining implementation task.
- LoMa weights are listed with their COLMAP-pinned SHA-256 values in
  `core/AICore/tools/loma_sources.json`. DaD, DeDoDe-B/DeDoDe-G, and B/R/L/G
  matcher graphs run through GGUF/ggml only; the legacy pipeline persists typed
  float descriptors and a real ten-image CUDA SfM run registered 10/10 images
  with 9020 sparse points and 68956 track observations. This does not establish
  strict CUDA DeDoDe-G numeric equivalence: CUDA uses the exact F32 materialized
  SDPA graph (`F32 GEMM -> F32 softmax -> F32 GEMM`) because the fused kernel
  converts F32 K/V to F16. The strict ViT-L token gate remains the acceptance
  evidence. The upstream ONNX CUDA comparator also cannot start locally without
  `libcublasLt.so.12`; no ONNX Runtime is added to ACloudViewer.

## OpenImageIO migration

Modern COLMAP removed FreeImage in commit `8e014c5b` and uses OpenImageIO.
Reconstruction now follows that design: `Bitmap` owns ABI-neutral packed image
storage and the private `3rdparty_openimageio` interface target links a fixed
OIIO 3.1.17.0 source build without leaking its embedded fmt headers into
unrelated targets. The recipe turns off Python, Qt, OpenCV, and unused format
modules. In particular `USE_OPENCV=OFF` prevents a host OpenImageIO package
from adding system OpenCV 4.2 to reconstruction's runtime closure.

OIIO's own pinned source recipes construct required dependencies that are
missing or too old, including Imath/OpenEXR/OpenColorIO on Ubuntu focal. This
keeps the OIIO source version and its required image/color compatibility level
the same on Ubuntu, macOS, and Windows while allowing their active toolchain
prefix to satisfy compatible prerequisites. `EMBEDPLUGINS=ON` keeps enabled
format readers in the OIIO library, so a separate format-plugin deployment is
not needed. FreeImage and the Conda/system OIIO package inputs are removed.

The focused `reconstruction_oiio_parity_gate` remains available for local image
I/O, EXIF, resize, undistortion, and texture validation, but GitHub CI does not
run it. Package creation still fails if the final `.run`, `.exe`, or `.app`
payload lacks either `OpenImageIO` shared runtime library. The release record
remains partial until the pinned source build and package path have completed
on all three platforms.
