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

## Deliberately gated items

- The upstream one-sided-focal 6-point solver is a PoseLib call. A pinned
  `3rdparty/PoseLib/poselib.cmake` recipe and adapter are available behind
  `-DRECONSTRUCTION_FETCH_POSELIB=ON`, and the asymmetric known-second-camera
  RANSAC path uses it. `relpose_one_sided_focal_test` now provides a deterministic
  exact six-point gate (focal relative error <= 1e-3, normalized E error <= 1e-2,
  residual < 1e-8) plus a sub-minimal rejection case. The manifest remains
  partial until calibrated-side unprojection Jacobians are carried through the
  public camera path for distorted and spherical models.
- Caspar BA and HIP PatchMatch require new kernels, toolchain detection, and
  CUDA/ROCm parity benchmarks. The current tree therefore continues to expose
  its tested Ceres-CUDA/PBA and CUDA PatchMatch paths.
- LoMa requires a LoMa model graph. LightGlue is available through ggml, but no
  LoMa GGUF graph is present, so no ONNX dependency is introduced as a stopgap.

## OpenImageIO migration

Modern COLMAP removed FreeImage in commit `8e014c5b` and uses OpenImageIO.
Reconstruction now follows that design: `Bitmap` owns ABI-neutral packed image
storage and `3rdparty/openimageio` resolves the OIIO target. The pinned Conda
package is available for `linux-64`, `osx-64`, `osx-arm64`, and `win-64`; its
Linux glibc >= 2.17 requirement covers Ubuntu 20.04, 22.04, and 24.04.

The implementation removes FreeImage from the source/build graph. The release
gate remains partial until the declared Ubuntu/macOS/Windows image, EXIF,
resize, undistortion, and texture parity matrix has run.
