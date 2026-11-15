# Mip-Splatting Differential Gaussian Rasterizer

This directory contains the **unmodified CUDA rasterizer core** used by the
official SAM 3D Objects Python renderer when its `inria` backend is installed.
It is imported from the following immutable Mip-Splatting source snapshot:

- Repository: `https://github.com/autonomousvision/mip-splatting`
- Commit: `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`
- Source directory:
  `submodules/diff-gaussian-rasterization`
- Imported files: `cuda_rasterizer/`, `LICENSE.md`, and the matching GLM 0.9.9.9
  headers under `third_party/glm/glm/`.

The non-build GLM documentation, test, CMake, and utility directories are not
copied. They are not included by the rasterizer and omitting them keeps this
repository source-only dependency to approximately 3.3 MiB.

The rasterizer and this integration are restricted to research/evaluation use
by the included Inria Gaussian-Splatting license. `LICENSE.md` must accompany
redistribution. This directory is not ggml source and is not modified by the
ggml patch mechanism.

The `sam3d_mip_gaussian_rasterizer` CMake target builds these files unchanged
and exposes their pure C++/CUDA `CudaRasterizer::Rasterizer` API. It does not
use PyTorch, Python, nvdiffrast, or cuDNN.

## Local deltas

One deliberate deviation from the snapshot, kept as small as possible so the
files remain diffable against the upstream commit:

- `cuda_rasterizer/forward.cu` `computeCov2D`: the mip-splatting determinant
  compensation (`coef = sqrt(det_0/det_1)`, including its zero-alpha kill for
  tiny determinants) is replaced by a constant 1. The official SAM 3D Objects
  scene render path is the **gsplat** backend (`render_frames`,
  `backend="gsplat"`) with `antialiased=False`: it applies the default
  `eps2d = 0.3` screen-space low-pass and no compensation. The native wrapper
  passes `kernel_size = 0.3` so the cov2d dilation matches gsplat exactly.
  `backward.cu` keeps the upstream form: it is never used by the inference
  runtime.

A follow-up delta extends the gsplat parity to the tile cover: the
`preprocessCUDA` bounding rectangle now uses gsplat's opacity-aware
elliptical bbox (`extend = min(3.33, sqrt(2*ln(opacity/ALPHA_THRESHOLD)))`,
per-axis radii, opacity < ALPHA_THRESHOLD culled before tiling, radii stored
per-axis as `radii[2*idx]/radii[2*idx+1]`), and the compositing loop clamps
alpha at 0.999 with `__expf`, exactly like gsplat's classic path. Without
these deltas the native render was systematically darker than the official
reference (small/medium splats lose alpha), which dominated the renderer
share of the scene benchmark gap.
