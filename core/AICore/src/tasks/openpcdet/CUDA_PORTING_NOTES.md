# OpenPCDet CUDA performance work — change notes

This records all changes from the 2026-10 CUDA work: 0023/0024 ggml patches,
task-side CUDA extensions and 0025/0026 follow-ups. Each item identifies an
**upstream port** or **intentional deviation**. Upstream reference:
[Asher-1/OpenPCDet-GGML](https://github.com/Asher-1/OpenPCDet-GGML)
`cpp_ggml/src/extensions/` and `cpp_ggml/patches/0001-native-sparse-gather.patch`.

## Results overview

| Metric | Before | After | Upstream, same host |
|------|--------|--------|----------|
| centerpoint CUDA e2e | 255.6ms | **83–91ms** (desktop-load dependent) | 85.8ms measured (earlier 79ms baseline not reproducible at current load) |
| kitti_pointpillar CUDA e2e | 95.8ms | **46–50ms** | ~40.5ms |
| centerpoint accuracy (307/307) | 0.99859 | **0.99908** | — |
| kitti accuracy (24/24) | 0.99976 | **0.99982** | — |

Known gap: ~5ms versus upstream is concentrated in generic dense-graph replay
(∝node count, supported by the kitti cross-comparison). Diagnosis requires
CUPTI/nsys infrastructure unavailable on this host, so it remains follow-up work.
Sparse execution (23ms) and GPU rulebook generation (4ms) match upstream.

## ggml patches (registered in cuda-manifest.yaml / cuda-vulkan-manifest.yaml)

| Patch | Content | Origin |
|-------|---------|--------|
| 0021 | Gate `opcpd_conv` F32 conv through TF32 igemm (only f16/q8 conv routing remains after 0022) | Repository-owned gate |
| 0022 | Byte-for-byte TF32 conv2d kernel chain (winograd/TF32 mma/FMA fallback, `opcpd_exact` namespace) | **Upstream port**, conv2d.cu section of `0001-native-sparse-gather.patch` |
| 0023 | Three `ggml_sparse_gather` / `ggml_sparse_scatter_add(_n)` ops appended to the enum; `GGML_MAX_SRC` 10→30, `GGML_MAX_OP_PARAMS` 64→256; CPU implementation + CUDA kernels + dispatch | **Upstream semantics** plus repository CPU implementation (upstream CPU uses native branches) |
| 0024 | Fused `ggml_bn_fold_relu` (CUDA float4 fast path + CPU fallback) | **Upstream port**, bn-fold-relu.cu |
| 0025 | Per-op CUDA event timing (`GGML_CUDA_PERF_OP=1`, off by default, diagnostics only) | **Upstream port** |
| 0026 | Phase-decomposed SECONDFPN deblock GEMM (gather + cublasSgemm + scatter, diagnostic gate) | **Upstream port**, conv2d-transpose.cu `conv2d_transpose_gemm_path`; repository node-name opt-in gate |

## Task-side CUDA extensions (`extensions/`, compile-time `AICORE_OPENPCDET_CUDA_EXT`)

| Component | Origin | Notes |
|-----------|--------|-------|
| `sparse_rulebook_cuda.{h,cu}` | **Upstream port** | GPU rulebook generation (30→4ms) |
| `vfe_scatter_cuda.{h,cu}` | **Upstream port + intentional deviation** below | Pillar VFE CUDA kernel |
| `height_compression_cuda.{h,cu}` | **Upstream port** | BEV height compression D2D |
| `openpcdet_cuda_ext.cu` | **Repository-owned** | Three components in one translation unit; deviation #4 |
| `remap_dense_indices_cuda` (inside ext) | **Repository-owned** | D2D -1→zero-row remap + padding clear; fixes uninitialized indices after capacity rebuild |
| `sparse_ops` compact lowering | **Upstream port + intentional deviation** below | Fused sparse-gather graph integration |
| Graph topology (`voxel_backbone_runner` / `pointpillar` / `sparse_backbone` / `second`) | **Repository-owned** | D2D chains, `spatial_device_ready`, capacity tensors and fused VFE/BN wiring |

## Intentional deviations from upstream

0. **focal_multimodal projection inverse fix (repository bug, neither port nor deviation)**:
   `inv[10]` in `voxelrcnn.cpp` `invert_4x4` (4×4 adjugate inverse) had a
   transcription error: `-m[12]*m[2]*m[5]` should be `-m[12]*m[3]*m[5]`.
   Calibration translation corrupted the inverse → misaligned `lidar_to_camera`
   projection → broken `conv_focal_multimodal` fusion → collapsed scores and
   2 detections (0.39/0.31), versus upstream C++ 8. The 1-line fix restored
   f32 CPU **8/8 matched mean_iou=1.000000** in scene mode. f16/q8_0 returned
   to 8/8, matching or exceeding upstream's own quantization gaps. Residual
   f32 CUDA 0.9978 is topology drift from ggml mul_mat noise (~2e-7) amplified
   by threshold-sensitive mask_kernel sites. CPU taps at 114 layers are
   bit-identical; tolerance rationale is in [UPSTREAM_ALIGNMENT.md](UPSTREAM_ALIGNMENT.md).

1. **Clear VFE BEV canvas**: upstream's persistent `gpu_pillar_vfe_scatter`
   canvas is never cleared. Changing pillar layouts scatter over stale data,
   a cross-frame ghost hidden by repeated-same-frame benchmarks. The port
   clears with `cudaMemsetAsync` before scatter, an intentional deviation.
2. **Quantized sparse weights fall back to dense lowering**: q8 weights cannot
   be reordered by kernel slices. Preserve dense lowering + device-side remap
   upload (absent upstream); f16 compact sparse GEMM retains strict fp32 precision.
3. **Fix `is_cuda()` device-name test**: `device.find("cuda")` never matched
   leased `CUDA0`, disabling persistent sparse-graph caching and rebuilding
   every frame. Use a `ggml_backend_name` prefix. This restores upstream's
   cache behavior, not an arithmetic deviation.
4. **One translation unit for three extensions**: separate libAICore units
   repeatedly failed HeightCompression/VFE launches with
   `cudaErrorInvalidDeviceFunction`. nm/cuobjdump confirmed registration data;
   the same objects linked independently worked, implicating multiple fatbins.
   Combining into `openpcdet_cuda_ext.cu` removed the failure, with target-level
   `CUDA_SEPARABLE_COMPILATION OFF`. rdc=true failed CUB device kernels; the
   inherited aliked-era setting was removed.
5. **Node-name deconv opt-in**: upstream globally dispatches
   `conv2d_transpose_gemm_path`; this repository follows 0021/0022 node-name
   opt-in (`opcpd_deconv`). Only OpenPCDet F32 deblock changes; f16 (SAM3),
   pristine mapany and other tasks keep their routes.
6. **Do not port `ggml_conv_2d_direct_packed` (evidence veto)**: upstream
   `bev_backbone.cpp` limits packed kernels to quantized weights. All 50 f32
   centerpoint conv nodes measured by `OPCPD_CONVDBG` use wino/f16-s2/fast-b4,
   exactly matching 0022, not direct_packed. Remaining causes are naive CT2D
   1x1/s1 (fixed by 0026) and unresolved graph replay.
7. **CUDA output hash change**: 0026 changes 1x1/s1 deblock from naive serial
   to upstream GEMM reduction order, changing output_hash (`e1e7…→6e52…`).
   Per-box IoU gates pass; CPU/Vulkan do not use this route and remain bit-identical.

## Final audit and remaining items (2026-10-29, synchronized with alignment audit)

**Integration audit (all six items pass)**: all 54 models are integrated and
measured SHA256 entries match `asset_digests.h`. Six structured-accuracy manifest
scenarios jointly cover all 54 rows. catalog-dump, level1-2-5 (111 passed),
contracts, coverage 100% and `--strict` boundaries pass. See integration
verification in [UPSTREAM_ALIGNMENT.md](UPSTREAM_ALIGNMENT.md).

**Remaining items (all classified, none untriaged)**:

| Item | Disposition | Rationale |
|---|---|---|
| focal f32 CUDA 0.99816 after Option C | Focal chain bit identity, unit memcmp 0 differing bytes | First residual divergence is **upstream** of the focal chain, in semseg. Backend noise feeds sensitive thresholds, outside Option C; see DETERMINISTIC_FOCAL_DESIGN.md validation |
| pointrcnn_iou 0.9947 | Inherited upstream unresolved item | REPORT declares "strict parity pending" 0.9944 |
| dynpillar/second_multihead/transfusion/pointrcnn f32 CUDA 0.997–0.999 | Tolerance | Equal detection counts at upstream's declared floor (0.9981–0.9988) |
| q8_0 nuScenes center heads | No assessment baseline | No upstream quantized baselines; aligned f16 and KITTI anchor q8_0 support implementation correctness |
| graph replay ~5–10ms | Performance gap | Upstream Voxel CenterPoint best 76ms vs AICore 83–91ms, same host; CUPTI/nsys required |
| bevfusion GPU ~10% | Optional performance follow-up | Hybrid decoder/head scalar GEMM not ported |
| 8/18 stale upstream baseline/cuda files | Upstream-tree issue | Read-only, not changed here; use fresh references |

## focal f32 CUDA residual analysis and candidate patches

**Noise chain**: task graph decomposition differs from upstream. parta2 compares
upstream SPARSE_GATHER/SCATTER_ADD+MUL_MAT×732 with AICore GET_ROWS+MUL_MAT+CONCAT.
Different MUL_MAT tiles/split-K trees produce ~2e-7 per-element noise. Both are
self-consistent against CPU, not incorrect. focal `conv_imp` sigmoid crosses a
**hard 0.5 threshold** to select voxels; 2945 sites are within <3e-5, closest
8e-9. Noise flips selections → feature topology drifts → tiny box-coordinate
changes. This is **hard-threshold amplification of continuous noise**, a
weight-dependent sensitivity; upstream CUDA itself drifts to 0.9997 vs CPU.

**Options**:

| Option | Assessment |
|---|---|
| A. Align ggml mul_mat reduction/decomposition | Rejected: affects all task dispatch; upstream cuda vs cpu is also 0.9997. Reproducing its noise is the wrong fix |
| B. Mask hysteresis/dual path | Rejected: changes upstream arithmetic semantics |
| C. Full focal-chain determinism (**feasible, bounded**): main-conv output feeds next-layer imp_features, so conv_imp alone is insufficient. Replace `run_sparse_conv` mul_mat with ordered K-dimension sparse dot products, no split-K/tile reorder; focal opt-in for 3 FocalSparseConv layers (~85% of model computation, absolute seconds; other 17 models unaffected) | Benefit: focal CUDA ≈ CPU bit identity beyond 0.9997. Cost: one deterministic kernel plus three-backend regressions, moderate effort |
| D. Keep 0.9978 | Current disposition; detection-level 8/8 matching meets the gate |

Recommendation at that stage: implement C if release-level CUDA/CPU bit-identity
evidence is required; otherwise retain D. Kernel pseudocode, fmaf/expf rounding,
files, gates and budget are in [DETERMINISTIC_FOCAL_DESIGN.md](DETERMINISTIC_FOCAL_DESIGN.md).
On 2026-10-29, three repeated focal f32 CUDA runs were **8/8 matched,
mean_iou=0.9978088, identical between runs**. Worst per-box IoU was 0.99602;
no missing/extra detections, parallel score distribution. **Detection-level
qualification passes**; residual coordinates differ at decimal places 3–4.

## Validation matrix (2026-10-29, after focal fix)

- `validate_all.py --tasks openpcdet`: **CPU 54/54 / CUDA 54/54 / Vulkan
  54/54 all PASS**. `openpcdet-focal-scene` covers the corrected multimodal
  path with calibration; all nine cells, three quantizations × three backends, pass.
- Full 18-checkpoint × 3-quantization × 3-backend = 162-cell audit:
  [UPSTREAM_ALIGNMENT.md](UPSTREAM_ALIGNMENT.md). CPU f32 matches 15/15
  (IoU≈1.0); 8/18 upstream baseline/cuda files are stale-build outliers.
  Fresh references align parta2/pp_multihead/centerpoint. Tolerance cells
  agree in counts at upstream's declared floor. The only code fix in that
  audit is the focal projection inverse.
- pointrcnn CUDA versus same-input upstream: `kitti_pointrcnn` 11/11 matched,
  mean_iou_bev=0.99924; `kitti_pointrcnn_iou` 12/12 matched, 0.99862.
- `test_openpcdet_capi_contract`, `test_qopenpcdet_helpers`, C-API coverage
  946/946 (100%) and `check_plugin_aicore_boundaries.py --strict` pass.

## Related manifest changes

- `openpcdet-kitti` excludes `kitti_pointrcnn*.gguf`, which requires fixed
  16384-point pre-sampling rather than generic inputs.
- Add `openpcdet-pointrcnn` using
  `openpcdet_inputs/kitti_000008_sampled16384.bin`, seeded from 000008.
  `openpcdet_input_pointrcnn.zip` is published in cloudViewer_downloads
  `OpenPCDet-GGUF`, with SHA registered in input_assets.

## Option C implementation (2026-10-29, focal-chain determinism)

[DETERMINISTIC_FOCAL_DESIGN.md](DETERMINISTIC_FOCAL_DESIGN.md) is implemented:
focal models (`openpcdet.voxelrcnn` + `focal.multimodal`) default on, overridable
by GGUF `focal.deterministic`. The 8 conv_imp/main sparse convolutions use
shared deterministic kernels, bypassing ggml mul_mat. Other layers and the
other 17 models are unchanged. No ggml patch or public ABI change.

**Files**: `extensions/deterministic_sparse_conv.{h,cpp}` provides shared
det_sparse_conv_cell/rows, det_expf/det_sigmoid, neighbor expansion and std::fma;
OpenMP partitions only output cells. `extensions/openpcdet_cuda_ext.cu` appends
`det_sparse_conv_kernel` and host wrapper in one translation unit:
grid=(n/128,Cout/8), serial K-outer/C-inner fmaf, no shared/warp/atomic reduction.
`models/voxelrcnn.cpp` adds the deterministic branch, retaining the original
f16/q8 graph as control. `gguf_loader.{h,cpp}` adds `hparams::focal_deterministic`;
`extensions/focal_sparse_conv.cpp` uses det_sigmoid; AICore/tests adds whitebox gates.

**Implementation details established during execution**:
1. Native ggml weights are o-major `[C_out, K*C_in]`,
   flat = o*(K*Cin)+k*Cin+c; c-inner access is contiguous. The initial
   mathematical-row-order misread was caught by the ggml control, with O(1) error.
2. det_expf uses explicit det_fma and 6th-order Taylor
   (1 + r + r²P(r), P = 1/2 + r·q), two-part ln2 range reduction and |x|>87
   saturation, preventing 2^k·(1+e) from exceeding FLT_MAX. Exact hex-float
   constants require float.hex() round trips: 23-bit mantissas vs 24-bit hex
   notation differ by a nibble position and are easy to transcribe incorrectly.
3. CUDA uploads host features/rulebooks/weights per layer into thread_local
   persistent VRAM buffers, copies results to host and applies det_relu,
   numerically equivalent to graph backend relu.

**Validation**:
- `test_openpcdet_det_sparse_conv` whitebox links the same
  `openpcdet_cuda_ext.cu` into AICore_test. Five shapes (C_in 16/64 ×
  C_out 16/27/32/64, K=27, n_out=1024, -1 neighbors and empty rows) give
  CPU(std::fma, OpenMP)/CUDA **memcmp=0 differing bytes**. ggml dense-lowering
  difference ≤3.2e-5 is within the 2e-6 relative gate scaled by cell Σ|term|;
  absolute order noise grows with K·Cin. det_expf vs double exp maximum relative
  error is 2.2e-7. `ctest -R test_openpcdet_det_sparse_conv` passes.
- E2e focal f32 scene, KITTI 000008: **8/8 matched** on all three backends vs
  cpp_cpu_f32. cpu mean_iou=0.9999998, center L2 2.3e-7 m; ordered reduction
  matches the upstream C++ CPU loop to the last digit, versus former 1.000000.
  cuda 0.99816 (before 0.99781), vulkan 0.97442.
- Regression: kitti_pointpillar f32 output_hash is byte-identical on all three
  backends (5ba9…/fece…/025d…); kitti_voxelrcnn_car f32 cuda detection JSON is
  byte-identical; `test_openpcdet_capi_contract` passes.
- E2e median latency, same host/load: CUDA 1412→~1409ms (Δ≈0; uploads offset
  kernel savings, within +15–25ms budget); CPU 2318→3748ms (+1.4s from serial
  scalar fma, acceptable for this non-real-time model); Vulkan 1133→3304ms
  because deterministic kernels also fall back to CPU.

**Unmet gate: three-backend e2e output_hash equality was not achieved and cannot
be achieved within Option C**. Bisection over 114 cpu/cuda taps establishes:
1. First divergence is **upstream** of the focal chain: semseg
   `voxelrcnn_image_features` differs by 0.0285 (backend arithmetic including
   bilinear interpolation). Plain conv_input/conv1_0 mul_mat differs by
   2.4e-6/3.8e-6, amplified to 2.3e-5 through imp, versus nearest threshold
   margin 8e-9. conv1.1 / conv_focal_multimodal selections still flip:
   `focal_mainconv_in_coords` has 577544 vs 577516 rows.
2. **Even kitti_voxelrcnn_car without a focal chain lacks three-backend
   detection-JSON bit identity**: BEV/dense-head/RoI differences reach output.
   Raw-box-bit output_hash equality requires whole-graph semseg/BEV/RoI
   determinism, beyond Option C. Delivered benefits are kernel bit identity,
   removal of focal-convolution reduction noise from flip budgets and CPU
   alignment with upstream at 1 ULP.
Optional, unimplemented follow-up: extend deterministic kernels to all focal
backbone run_sparse_conv calls (conv_input/conv1_0/conv2_pre/conv3_pre/conv4/conv_out).
This could remove LiDAR-only threshold flips; scene remains semseg-dependent.
Whole-graph upstream/downstream determinism is a separate project.

**Additional taps-only fix**: `run_focal_backbone`
`focal_mm_mainconv_{bias,weight}` taps directly dereferenced `tensor->data`.
GPU weights live in device memory, causing cuda+taps segfaults (latent on CPU).
Use `ggml_backend_tensor_get` into host buffers before passing to the sink.
