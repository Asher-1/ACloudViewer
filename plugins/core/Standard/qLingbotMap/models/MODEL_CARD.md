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
- **AICore integration A/B (measured, RTX 4090, q8 GGUF, 3 frames, KV 1/4)**:
  element-wise LBF3 dumps from both engines compared with
  `core/AICore/tests/lingbot/compare_lingbot_parity.py`. Engine-to-engine
  depth differences (mean 3.7e-3) sit at or below each engine's own
  backend-to-backend noise (upstream CUDA-vs-CPU baseline mean 5.1e-3;
  AICore 3.8e-3 — the integration is *more* backend-consistent, thanks to
  the strict-F32 cuBLAS routing). Against the official fp32 PyTorch
  checkpoint the depth RMSE stays in the q8 quantization-noise band for all
  four engine/backend rows (upstream CLI 0.68-5.4e-3, AICore 1.3-6.1e-3),
  i.e. no accuracy regression from the port.
- **Speed A/B (same input, same profile)**: upstream `lingbot-map-bench`
  median 270.2 ms/frame (CUDA0); AICore steady-state streaming 273 ms/frame
  (30-frame stream, pure-inference wall) - parity within ~1%. The first
  frame carries graph-build warmup. Reproduce with
  `tests/lingbot/dump_lingbot_frames` + `lingbot-map-bench`.

## License

Apache-2.0 (LingBot-Map upstream). GGUF deployment published under the same
license.
