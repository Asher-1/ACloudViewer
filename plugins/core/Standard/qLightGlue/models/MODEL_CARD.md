# LightGlue / ALIKED GGUF Models

GGUF weights for **qLightGlue** — SIFT LightGlue and ALIKED LightGlue matcher families.

Inference runs through AICore (ggml): CPU / CUDA / Vulkan / Metal.

## Two families at a glance

| Family | Extractor | Matcher / model | Quant variants |
|--------|-----------|-----------------|----------------|
| **SIFT LightGlue** | OpenCV RootSIFT (C++) | `sift-lightglue-*.gguf` | F16, Q8_0, F32 |
| **ALIKED LightGlue** | `aliked-n16rot-*.gguf` (AICore) | `aliked-lightglue-*.gguf` | F16, Q8_0, F32 |

Each LightGlue family uses **matcher-only** GGUF files for the transformer stage. SIFT features come from OpenCV; ALIKED features come from a separate extractor GGUF (same quant suffix as the matcher, e.g. `aliked-lightglue-f16.gguf` + `aliked-n16rot-f16.gguf`).

## Model roles (LightGlue tab)

| GGUF prefix | Role | In qLightGlue GUI |
|-------------|------|-------------------|
| `sift-lightglue-*` | LightGlue matcher for SIFT descriptors | Matcher after OpenCV RootSIFT |
| `aliked-lightglue-*` | LightGlue matcher for ALIKED descriptors | Matcher after AICore ALIKED extract |
| `aliked-n16rot-*` | ALIKED CNN (keypoints + 128-D desc) | Extractor stage (AICore C API) |

## End-to-end matching (qLightGlue plugin)

| Matcher GGUF | Feature extractor | Status |
|--------------|-------------------|--------|
| `sift-lightglue-*.gguf` | OpenCV RootSIFT (C++) | **Supported** — no Python/ONNX |
| `aliked-lightglue-*.gguf` | AICore GGML (`aliked-n16rot-*.gguf`) | **Supported** — native C++ extractor + matcher |

COLMAP reference (runtime, no Python):

- SIFT features: classical C++ (VLFeat / OpenCV)
- ALIKED features: ONNX Runtime (`aliked-n16rot.onnx`) in COLMAP; **GGML extractor in ACloudViewer**
- LightGlue matcher: ONNX in COLMAP → **GGML matcher** in ACloudViewer

### ALIKED GGML pipeline (AICore)

AICore ships a native ALIKED extractor with PyTorch parity and optional CUDA VRAM pipeline (ported from [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML)).

| Optimization | Effect |
|--------------|--------|
| DCN fused weights + workspace cache | No per-frame `cudaMalloc` / H→D for deform conv |
| Score head single fused GGML graph | 4-layer conv+SELU chain → one `ggml_backend_graph_compute` |
| GPU DKD + SDDH | NMS / block Top-K / descriptor head stay on device |
| CUDA pool / upsample / crop | Removes GGML ping-pong on small spatial ops |

**Parity (1024 px long edge, 1024 keypoints, RTX 3060):** kpt median ≈ **0.003 px**, descriptor cosine median ≈ **0.9996** vs PyTorch ALIKED.

## Download — LightGlue matchers

Release: [cloudViewer_downloads / LightGlue](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/LightGlue)

Cache: `~/cloudViewer_data/extract/lightglue_models/` (or `aicore_lightglue_model_cache_dir()`)

| Download | Matcher for | Quant | Notes |
|----------|-------------|-------|-------|
| [`sift-lightglue-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f16.gguf) | SIFT | F16 | **default SIFT matching** |
| [`sift-lightglue-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-q8_0.gguf) | SIFT | Q8_0 | smaller; ~93% recall vs F32 |
| [`sift-lightglue-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f32.gguf) | SIFT | F32 | reference |
| [`aliked-lightglue-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-f16.gguf) | ALIKED | F16 | **default ALIKED matching** |
| [`aliked-lightglue-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-q8_0.gguf) | ALIKED | Q8_0 | matcher only |
| [`aliked-lightglue-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-f32.gguf) | ALIKED | F32 | reference |

## Download — ALIKED extractor

Cache: `~/cloudViewer_data/extract/lightglue_models/` (same directory as matcher models)

Place the extractor GGUF with the **same quant suffix** as the selected matcher (e.g. F16 matcher → F16 extractor). Build from [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML) or copy prebuilt artifacts into cache.

| File | Quant | Size (approx.) | Notes |
|------|-------|----------------|-------|
| `aliked-n16rot-f32.gguf` | F32 | ~2.7 MB | parity reference |
| `aliked-n16rot-f16.gguf` | F16 | ~1.4 MB | recommended for CUDA |
| `aliked-n16rot-q8_0.gguf` | Q8_0 | ~0.7 MB | smallest, good recall |

GGUF key: `aliked` · 128-D descriptors · default resize long edge 1024 · DKD + SDDH postprocess metadata embedded.

Convert extractor GGUF locally: `python scripts/convert_aliked_to_gguf.py models/aliked-n16rot-f32.gguf`

## Matcher architecture

- GGUF key: `lightglue`
- Inputs: keypoints, row-major descriptors, image sizes; SIFT also uses scale + orientation (radians)
- Outputs: mutual match index pairs + scores
- Backends: CPU / CUDA / Vulkan / Metal via AICore ggml

## Dev-only fixtures

`scripts/extract_aliked_features.py` generates LGINP01 test fixtures for AICore contract tests — **not used by the GUI plugin**.
