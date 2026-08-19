# qRFDetr Model Card

## Model

| Field        | Value                                                                  |
|--------------|------------------------------------------------------------------------|
| Architecture | RF-DETR (real-time DETR with deformable attention, Roboflow)           |
| Task         | Open-vocabulary object detection (COCO 80 classes) + optional instance segmentation |
| Input        | RGB image letterboxed to 640×640 (configurable at export)              |
| Output       | Detections JSON (class_id / class_name / score / box) + per-detection PNG masks (seg variants) |
| License      | Models: [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) (see [rf-detr.cpp](https://github.com/mudler/rf-detr.cpp) for conversion details) |
| Source       | [rf-detr.cpp](https://github.com/mudler/rf-detr.cpp) → `RF-DETR-GGUF` release |

## Files

**11 variants × 4 quantizations = 44 GGUF files** in the
[RF-DETR-GGUF release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RF-DETR-GGUF).
Filename pattern: `rfdetr-<variant>-<quant>.gguf`.

Sizes below are the f16 assets as published (measured from the release); every
variant ships in all four quantizations.

| Variant       | f16 size (approx.) | Notes                                 |
|---------------|--------------------|---------------------------------------|
| nano          | ~63 MB             | Detection, fastest                    |
| small         | ~67 MB             | Detection                             |
| base          | ~67 MB             | Detection (default)                   |
| medium        | ~70 MB             | Detection                             |
| large         | ~72 MB             | Detection, highest accuracy           |
| seg-nano      | ~71 MB             | Detection + instance segmentation     |
| seg-small     | ~72 MB             | Detection + instance segmentation     |
| seg-medium    | ~75 MB             | Detection + instance segmentation     |
| seg-large     | ~76 MB             | Detection + instance segmentation     |
| seg-xlarge    | ~80 MB             | Detection + instance segmentation     |
| seg-2xlarge   | ~82 MB             | Detection + instance segmentation     |

Quantizations (relative to the f16 size):

| Quant | Size vs f16 | Notes                                            |
|-------|-------------|--------------------------------------------------|
| f32   | ~1.85×      | float32 reference                                |
| f16   | 1×          | half precision (**default**)                     |
| q8_0  | ~0.60×      | 8-bit quantization, good accuracy/size trade     |
| q4_K  | ~0.49×      | 4-bit K-quant, smallest download                 |

## Download

Mirror hosted by ACloudViewer:

`https://github.com/Asher-1/cloudViewer_downloads/releases/download/RF-DETR-GGUF/rfdetr-<variant>-<quant>.gguf`

(e.g. `rfdetr-base-f16.gguf`, `rfdetr-seg-2xlarge-q4_K.gguf`). The model cache
directory is `rfdetr_models/` (see `aicore_rfdetr_model_cache_dir`).

## Inference benchmarks

Measured **2026-08-19** by the upstream
[rf-detr.cpp](https://github.com/mudler/rf-detr.cpp) benchmark
(`benchmarks/BACKENDS.md`): AMD Ryzen 9 5950X (32 threads) + NVIDIA RTX 3060
(CUDA 12.4), ggml v0.18.0, PyTorch 2.5.1+cu124 + rfdetr 1.7.0 as reference.
Median ms/image, end-to-end (preprocess + forward + postprocess, image load
excluded); CPU at T=8.

| Variant     | PyTorch F32 | PyTorch F16 opt | CUDA f16 | Fastest ggml (config) | Vulkan f16 | CPU f32 @ T=8 |
|-------------|------------:|----------------:|---------:|-----------------------|-----------:|--------------:|
| nano        |        20.1 |            11.8 |     13.0 | **13.0** (CUDA q8_0)  |       21.0 |         180.7 |
| small       |        24.8 |            14.2 |     21.9 | **20.7** (CUDA q8_0)  |       28.8 |         345.1 |
| base        |        30.5 |            13.8 |     29.4 | **27.0** (CUDA q8_0)  |       32.5 |         491.4 |
| medium      |        29.6 |            15.1 |     27.2 | **26.5** (CUDA q4_K)  |       33.2 |         452.4 |
| large       |        36.5 |            17.3 |     40.0 | **38.7** (CUDA q8_0)  |       42.9 |         841.5 |
| seg-nano    |        27.2 |            15.7 |     36.8 | **36.2** (CUDA q4_K)  |       42.9 |         301.2 |
| seg-small   |        30.7 |            17.4 |     43.4 | **41.7** (CUDA q8_0)  |       48.4 |         417.2 |
| seg-medium  |        39.5 |            20.5 |     56.8 | **55.8** (CUDA q8_0)  |       56.6 |         567.8 |
| seg-large   |        47.9 |            23.4 |     72.5 | **68.3** (Vulkan q8_0)|       69.7 |         828.9 |
| seg-xlarge  |        82.1 |            37.3 |    122.0 | **109.4** (Vulkan q8_0)|     109.6 |        1464.7 |
| seg-2xlarge |       132.8 |            55.0 |    207.8 | **196.7** (Vulkan f16)|      196.7 |        2876.0 |

Key findings:

- **GPU vs CPU:** CUDA f16 ÷ CPU f16 = 0.06×–0.07× (14–17× faster); Vulkan
  f16 ÷ CPU f16 = 0.07×–0.09× (11–14× faster).
- **ggml vs PyTorch:** CUDA f16 runs at 0.65×–1.56× the PyTorch F32 eager
  time — detection variants reach parity (0.65×–1.10×) while seg variants
  pay 1.35×–1.56×; Vulkan f16 is 1.04×–1.58×. PyTorch's official
  `optimize_for_inference()` arm (jit trace + fp16) stays ahead
  (11.8–55.0 ms).
- **Quant accuracy** (recall vs the CPU f32 reference, class + IoU ≥ 0.5):
  f32 and f16 100% across the board, q8_0 100%, q4_K 80%–92% — prefer q8_0,
  not q4_K, when shrinking below f16.
- **CPU threads** (base variant, 16C/32T test CPU): f32 bottoms at 377 ms
  @ T=12, f16 at 312 ms @ T=24, q8_0 at 247 ms @ T=24; T=32 (SMT) collapses
  f32 to 5520 ms — keep the thread count at or below the physical cores.
- **f16 is the recommended CPU default** (separate Ryzen 9 9950X3D
  measurement, upstream `BENCHMARK.md`): fastest CPU variant tested —
  137.2 ms vs 142.8 (f32) / 148.0 (q8_0) / 149.5 (PyTorch) at T=8 — and
  1.86× smaller than f32 (64.2 MB vs 119.2 MB), with 56/56 detections
  matching f32 at IoU ≥ 0.95 (max |Δscore| 0.006). This is why f16 is the
  default quantization in the table above.

## Backends

- CUDA / Vulkan / Metal / CPU — all ggml backends are supported through
  AICore's unified device resolution (`auto` picks the best available GPU).
- Thread count is configurable (0 = auto).
