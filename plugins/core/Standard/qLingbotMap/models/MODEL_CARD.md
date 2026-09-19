# Model Card — LingBot-Map (Geometric Context Transformer)

| | |
|---|---|
| Task | Streaming 3D reconstruction: per-frame depth + confidence, camera pose (c2w 4×4), intrinsics; persistent KV cache across frames of a stream |
| Model | GCT: DINOv2-style patch encoder + 24 streaming transformer blocks (anchor context, pose-reference window, trajectory memory) + iterative CameraCausalHead + four-scale DPT depth head |
| Paper | Geometric Context Transformer for Streaming 3D Reconstruction, [arXiv:2604.14141](https://arxiv.org/abs/2604.14141) |
| Source runtime | [lingbot-map-ggml](https://github.com/Asher-1/lingbot-map-ggml) `cpp_ggml` (in-tree port under `core/AICore/src/tasks/lingbot/`) |
| Weights | https://huggingface.co/Asher-1/lingbot-map-gguf |
| Digests | Pinned in `core/AICore/include/aicore/asset_digests.h` (HF LFS SHA-256) |
| Input | Ordered RGB image sequence; official crop preprocessing (width = image_size, height snapped to the patch grid, center crop) |
| Output | Depth (meters) + depth confidence per processed pixel, c2w 4×4, [fx, fy, cx, cy] |

## Published files (map role)

| File | Size | Notes |
|---|---|---|
| `lingbot-map-f16.gguf` | 2.31 GiB | **recommended** (the upstream GUI default): full-alignment format — pose 1.72e-04 / depth 4.72e-04 vs the official fp32 checkpoint over the 286-frame courthouse stream, two orders of magnitude below the official bf16 deployment self-noise |
| `lingbot-map-q8.gguf` | 1.21 GiB | memory-saving deployment: half the weight memory; carries the documented q8 quantization loss (pose 1.31e-03 / depth 3.16e-03 over 286 frames — one order of magnitude looser than f16) and is no faster per-frame |
| `lingbot-map-f32.gguf` | 4.63 GiB | exact reference (deepest alignment, pose ~1.3e-04 / depth 1.35e-04); same per-frame speed as f16, 2× the weight memory |
| `lingbot-map-q4.gguf` | 703 MiB | experimental (not upstream-validated — no accuracy data) |
| `lingbot-map-long-q8.gguf` | 1.21 GiB | **long sequences**: architecture-identical to `lingbot-map-q8` (same tensor names/shapes; graph and options apply unchanged) |
| `lingbot-map-long-f16.gguf` | 2.31 GiB | long-sequence variant of the f16 format |
| `lingbot-map-long-f32.gguf` | 4.63 GiB | long-sequence variant of the f32 format |

## Published files (skyseg role, optional native sky masking)

| File | Size | Notes |
|---|---|---|
| `lingbot-map-skyseg-f16.gguf` | 88 MiB | **recommended**: 100% mask agreement with the official onnxruntime path over the 8-frame courthouse stream |
| `lingbot-map-skyseg-q8_0.gguf` | 47 MiB | quantized deployment |
| `lingbot-map-skyseg-f32.gguf` | 176 MiB | exact reference |

## Numerical contract

- Attention runs the exact strict path (hand-built F32 attention with
  `GGML_PREC_F32` matmuls and a persistent F16 KV cache, scale=8 / window=64
  release profile); the upstream flash-attention fast path is intentionally
  not part of this integration.
- Vulkan precision is enforced per graph node: matmul outputs are named
  `lingbot_scalar_*` and routed to the exact scalar pipelines (owner-scoped,
  no process-global state; see the `lingbot_merged` ggml patch).
- Upstream alignment evidence (286-frame courthouse stream, official
  checkpoint): q8 CUDA pose RMSE 6.14e-05 / depth 9.75e-05; f16 vs the
  official PyTorch pipeline pose 1.72e-04 / depth 4.72e-04. Reproduce with
  `bash cpp_ggml/scripts/run_e2e.sh cuda q8` in the source repo.
- **Long-sequence windowed alignment**: upstream `scripts/verify_windowed.py`
  cross-checks the numpy windowed orchestration against the official PyTorch
  `inference_windowed` at `keyframe_interval=1` (per-window raw pose 7.9e-05
  / depth 5.3e-04; stitch 6.3e-05 / 4.3e-04); the 286-frame long-model
  matrix (`scripts/run_long_matrix.sh`, validation report section 9) lands
  strict pose 3.4–6.9e-05 across f16/q8/f32 on CUDA and Vulkan. The plugin
  ports the same math (`LingbotWindowStitcher`) and gates it numerically
  against the official golden values with `test_lingbot_window_stitch`.
- **Preprocessing: bit-exact.** `aicore_lingbot_preprocess_image` replicates
  Pillow 12.2.0's 8bpc BICUBIC pipeline exactly (fixed-point coefficients,
  rounding biases, clip8, per-axis skip semantics) - verified max_abs_diff
  = 0.0 against PIL on five shape classes (identity, landscape downscale,
  portrait downscale+crop, upscale, deep-crop). The official scene datasets
  are pre-cropped to 518-wide, so their frames hit the identity path.
- **Engine A/B (measured, RTX 4090, q8 GGUF, Oxford 2 frames vs the official
  fp32 PyTorch checkpoint, KV 1/4)**: element-wise LBF3 dumps compared with
  `core/AICore/tests/lingbot/compare_lingbot_parity.py`.
  - CPU isolation: AICore CPU matches upstream CPU against the fp32
    reference (depth RMSE f0 9.6 vs 9.4e-3, f1 2.1 vs 1.0e-3 - same
    magnitude, both dominated by the q8 weight noise) - the port itself is
    numerically faithful.
  - CUDA: AICore keeps CUDA and CPU consistent with each other (depth RMSE
    1.09e-2 vs 9.6e-3), whereas upstream CUDA diverges from upstream CPU
    (2.1e-3 vs 9.4e-3) because the upstream CUDA GEMM path (F16-MMA MMF
    kernels, pre-fix Q8_1 semantics) differs from its CPU path. ACloudViewer
   's patch chain (q41_mmq_parity, trellis_f32_route, cuda_strict_f32)
    aligns CUDA to the CPU-referenced Q8_1/F32 semantics - a deliberate,
    previously validated correction - so long-stream depth values may
    diverge from the upstream CLI at the q8 noise level (RMSE ~3e-3) while
    both engines remain inside the upstream-published q8 accuracy band
    (depth 3.16e-3 vs the PyTorch mirror).
  - Pose: frame-0 c2w/pose_enc agree to 1.5e-5 mean between engines; the
    divergence on later frames stays at the q8 weight-noise level.
- **Speed A/B (same input, same profile)**: upstream `lingbot-map-bench`
  median 270.2 ms/frame (CUDA0); AICore steady-state streaming 273 ms/frame
  (30-frame stream, pure-inference wall) - parity within ~1%. The first
  frame carries graph-build warmup. Reproduce with
  `tests/lingbot/dump_lingbot_frames` + `lingbot-map-bench`.
- **2026-09-17 re-audit (8-frame courthouse stream, kv strict 1/4 both
  sides, LBF3 element-wise)**: f32 CPU depth RMSE 7e-06 / c2w max 2.3e-06
  (graph math bit-level equivalent); q8 CPU depth RMSE 1.9–6.6e-03 (inside
  the q8 weight-noise band); Vulkan deltas match each engine's own
  Vulkan-vs-CPU spread (0.34–0.40 depth max, triangle closes). Streaming
  per-frame wall: CUDA 296 vs upstream 350 ms, Vulkan 315 vs 425 ms.
- **Correction (2026-09-17 late, controlled re-measurement after clearing
  stale test processes)**: the previously reported "f32 Vulkan ~2.5x
  slower" was a measurement artifact with two root causes — (1) the
  single-frame bench compares different semantics (upstream
  `lingbot-map-bench` runs stateless with `disable_kv_cache=true`, the
  AICore counterpart rebuilds its resident cache every reset), and (2)
  two stale test processes from the audit itself (an hung
  `dump_lingbot_frames` at 2.5 GiB + 100% GPU utilization, and a leftover
  upstream `lingbot-map-cli` at 11.9 GiB) were competing for the GPU and
  masquerading as "other processes' VRAM pressure" (the upstream-side
  OOMs were caused by the leftover CLI's own 11.9 GiB). After killing
  them (GPU 16.2→1.8 GiB, util 100→13%), a clean direct A/B lands:
  **f32 Vulkan depth RMSE 3.3e-04 / c2w 3.3e-05 vs the upstream CLI, and
  streaming 314 vs 367 ms/frame (AICore 15% faster); q8 Vulkan depth
  RMSE 5.2e-04**. No gap in any tier or backend; the candidate dot2
  pipeline patch was abandoned (bit-identical outputs, no measurable
  gain). Discipline: wrap long engine tests in `timeout` and check
  `nvidia-smi --query-compute-apps` before interpreting any timing.

## License

Apache-2.0 (LingBot-Map upstream). GGUF deployment published under the same
license.
