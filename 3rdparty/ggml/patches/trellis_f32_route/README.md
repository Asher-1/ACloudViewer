# trellis_f32_route — upstream ggml-vulkan-f32-route.patch port status

Source: upstream `trellis-ggml/patches/ggml-vulkan-f32-route.patch`
(ggml submodule pinned at the same v0.18.1 / 90951f99 as this repository).

## What is applied (0001-ggml-f32-prec-route-cuda-only.patch)

The CUDA section, verbatim from upstream: in `ggml_cuda_mul_mat`,
`GGML_PREC_F32` (set by trellis2's `ggml_mul_mat_set_prec` sites) bypasses
the f16-MMA `mmvf`/`mmf` kernel selection and falls through to cuBLAS.
ggml's cuBLAS handles use `CUBLAS_COMPUTE_32F` (no TF32), so this matches
the CPU backend's f32-accumulate semantics. Validated: `core/AICore/src/tasks/trellis/tools/trellis_backend_ab.py`
coarse q8 (4 steps, seed 42) geometry hash `7bc0501eb018` reproduces across runs.

## What is deferred (0001-ggml-f32-prec-route.patch — full port, NOT in manifest)

The Vulkan sections: coopmat2 fp32 pipeline registration
(`matmul_f32_f32_fp32[_aligned]`, `matmul_f32_f16_fp32[_aligned]`,
`matmul_f16_f32_fp32_f32acc` with the patch's own warptile spec constants)
and the `prec_f32_route` dispatch changes (F16→F32 dequant copy + pure fp32
pipeline).

**Repro of the failure on our validation stack** (RTX 4090, driver
550.144.03, Ubuntu 22.04): with the full port, `aicore_trellis_generate_ex`
coarse q8 returns an empty mesh ("no occupied voxels at iso 0") on Vulkan —
with `GGML_VK_DISABLE_F16=1` AND with it unset. No GGML_ASSERT fires
(pipelines are created and dispatched), so the wrong output comes from the
executed pipelines, most plausibly the patch's warptile spec-constant
combinations on the shader variants produced by this build's
vulkan-shaders-gen configuration. Upstream presumably validated on a
different driver/shader-gen configuration.

With only the CUDA hunk, Vulkan runs the base v0.18.1 dispatch and the
coarse q8 e2e is functional and fast (9.9 s vs CUDA 18.6 s on the same
machine; geometry within the q8 chaotic-sampler envelope of the CUDA run —
with q8 weights the SS-flow CFG sampler amplifies ~1e-4 kernel differences
into different voxel sets, per the upstream README, so cross-backend voxel
identity is not expected on the q8 chain).

Revisit the Vulkan hunks on the next ggml upgrade: re-apply the full port
from the upstream repo, rebuild, and re-run
`core/AICore/src/tasks/trellis/tools/trellis_backend_ab.py --device cuda,vulkan` (a valid mesh on both
backends is the pass criterion).
