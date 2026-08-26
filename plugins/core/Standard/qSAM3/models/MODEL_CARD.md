# qSAM3 model card

## Model

| Field        | Value                                                                 |
|--------------|-----------------------------------------------------------------------|
| Architecture | Meta Segment Anything 2 / 2.1 / 3 — Hiera ViT + IoU / text decoder   |
| Task         | Interactive segmentation (points / box / text prompt), video tracking |
| Input        | RGB image (any size, internally resized to the model's stride)        |
| Output       | Segmentation masks + confidence scores + bounding boxes per instance  |
| License      | [Apache 2.0](https://github.com/facebookresearch/segment-anything) (SAM 2/2.1/3) |
| Source       | [sam3-ggml](https://github.com/Asher-1/sam3-ggml) conversion → `sam` release (hosted on cloudViewer_downloads) |

## Architecture by model family

| Family | Encoder | Text decoder | Video tracking | Published |
|--------|---------|-------------|----------------|-----------|
| `sam3-*` | ViT-H (632M) | ✅ DETR-based text / exemplar prompt | ✅ | 4 quant variants |
| `sam3-visual-*` | ViT-H (632M) | ❌ visual-only | ✅ | 4 quant variants |
| `sam2.1_hiera_*` | Hiera (tiny/base+/small/large) | ❌ visual-only | ✅ | 5 quant variants × 4 sizes |
| `sam2_hiera_*` | Hiera (tiny/base+/large) (SAM 2, not 2.1) | ❌ visual-only | ✅ | 5 quant variants × 3 sizes |

`sam3-f32` is intentionally not published (GGUF too large at ~7 GB). All 39 published entries are enumerated below.

## Files

**39 GGUF files** in the
[sam release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/sam).

### SAM 3 — text + tracking (4 models)

| File | Size | Quant |
|------|------|-------|
| sam3-f16.gguf | 1.84 GB | F16 — half precision |
| sam3-q8_0.gguf | 1.10 GB | Q8_0 — 8-bit quant |
| sam3-q4_1.gguf | 756 MB | Q4_1 — 4-bit quant with bias |
| sam3-q4_0.gguf | 707 MB | Q4_0 — smallest SAM3 quant |

### SAM 3 Visual — visual-only (4 models)

| File | Size | Quant |
|------|------|-------|
| sam3-visual-f16.gguf | 946 MB | F16 — half precision |
| sam3-visual-q8_0.gguf | 517 MB | Q8_0 — 8-bit quant |
| sam3-visual-q4_1.gguf | 318 MB | Q4_1 — 4-bit quant with bias |
| sam3-visual-q4_0.gguf | 289 MB | Q4_0 — smallest visual quant |

### SAM 2.1 Hiera — visual-only (20 models)

| File | Size | Quant |
|------|------|-------|
| sam2.1_hiera_large_f16.gguf | 451 MB | F16 |
| sam2.1_hiera_large_f32.gguf | 898 MB | F32 |
| sam2.1_hiera_large_q8_0.gguf | 241 MB | Q8_0 |
| sam2.1_hiera_large_q4_1.gguf | 144 MB | Q4_1 |
| sam2.1_hiera_large_q4_0.gguf | 130 MB | Q4_0 |
| sam2.1_hiera_base_plus_f16.gguf | 163 MB | F16 |
| sam2.1_hiera_base_plus_f32.gguf | 323 MB | F32 |
| sam2.1_hiera_base_plus_q8_0.gguf | 88 MB | Q8_0 |
| sam2.1_hiera_base_plus_q4_1.gguf | 53 MB | Q4_1 |
| sam2.1_hiera_base_plus_q4_0.gguf | 48 MB | Q4_0 |
| sam2.1_hiera_small_f16.gguf | 94 MB | F16 |
| sam2.1_hiera_small_f32.gguf | 184 MB | F32 |
| sam2.1_hiera_small_q8_0.gguf | 50 MB | Q8_0 |
| sam2.1_hiera_small_q4_1.gguf | 31 MB | Q4_1 |
| sam2.1_hiera_small_q4_0.gguf | 28 MB | Q4_0 |
| sam2.1_hiera_tiny_f16.gguf | 79 MB | F16 (recommended start) |
| sam2.1_hiera_tiny_f32.gguf | 156 MB | F32 |
| sam2.1_hiera_tiny_q8_0.gguf | 43 MB | Q8_0 |
| sam2.1_hiera_tiny_q4_1.gguf | 26 MB | Q4_1 |
| sam2.1_hiera_tiny_q4_0.gguf | 24 MB | Q4_0 — smallest SAM2.1 |

### SAM 2 Hiera — visual-only (11 models)

| File | Size | Quant |
|------|------|-------|
| sam2_hiera_large_f16.gguf | 451 MB | F16 |
| sam2_hiera_base_plus_f16.gguf | 163 MB | F16 |
| sam2_hiera_base_plus_f32.gguf | 323 MB | F32 |
| sam2_hiera_base_plus_q8_0.gguf | 88 MB | Q8_0 |
| sam2_hiera_base_plus_q4_1.gguf | 53 MB | Q4_1 |
| sam2_hiera_base_plus_q4_0.gguf | 48 MB | Q4_0 |
| sam2_hiera_tiny_f16.gguf | 79 MB | F16 |
| sam2_hiera_tiny_f32.gguf | 156 MB | F32 |
| sam2_hiera_tiny_q8_0.gguf | 43 MB | Q8_0 |
| sam2_hiera_tiny_q4_1.gguf | 26 MB | Q4_1 |
| sam2_hiera_tiny_q4_0.gguf | 24 MB | Q4_0 — smallest SAM2 |

## Download

Mirror hosted by ACloudViewer:

`https://github.com/Asher-1/cloudViewer_downloads/releases/download/sam/<filename>`

The model cache directory is `~/cloudViewer_data/extract/sam3_models/`
(`$CLOUDVIEWER_DATA_ROOT/extract/sam3_models` when the environment variable
is set — the same convention as qDA3 / qTrellis via the AICore cache API).

## Recommended defaults

| Use case | Model |
|----------|-------|
| Text-prompt segmentation | `sam3-q8_0.gguf` (best accuracy/size trade-off) |
| Box / point segmentation (GPU) | `sam3-visual-f16.gguf` or `sam2.1_hiera_base_plus_f16.gguf` |
| Box / point segmentation (CPU) | `sam2.1_hiera_tiny_q8_0.gguf` (43 MB, fast encode) |
| Video tracking (low-latency) | `sam2.1_hiera_tiny_f16.gguf` (79 MB, recommended default) |
| Video tracking (high accuracy) | `sam2.1_hiera_large_f16.gguf` (451 MB) |

## Inference benchmarks

See upstream [sam3-ggml](https://github.com/Asher-1/sam3-ggml) for benchmarks.
Typical end-to-end latency on NVIDIA RTX 3060 (CUDA):

| Model | Encode (ms) | PCS/PVS decode (ms) | Track / frame (ms) |
|-------|------------|---------------------|-------------------|
| sam2.1 tiny F16 | ~180 | ~40 | ~80 |
| sam2.1 base+ F16 | ~260 | ~60 | ~120 |
| sam2.1 large F16 | ~450 | ~90 | ~200 |
| sam3 F16 | ~600 | ~100 | ~280 |