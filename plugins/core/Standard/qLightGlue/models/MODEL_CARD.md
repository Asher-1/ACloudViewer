# LightGlue GGUF Models

GGUF weights for the **LightGlue matcher** in AICore (ggml inference).

These files contain **only the LightGlue transformer matcher** — not ALIKED/SIFT feature extractors. Upstream [LightGlue-GGML cpp/README.md](https://github.com/Asher-1/LightGlue-GGML/blob/main/cpp/README.md) documents this as matcher-only, mirroring COLMAP's split between extractors and `LightGlueONNXFeatureMatcher`.

## End-to-end matching (qLightGlue plugin)

| Matcher GGUF | Feature extractor | Status |
|--------------|-------------------|--------|
| `sift-lightglue-*.gguf` | OpenCV RootSIFT (C++) | **Supported** — no Python/ONNX |
| `aliked-lightglue-*.gguf` | ALIKED CNN | **Matcher only** — needs ONNX extractor (COLMAP: `aliked-n16rot.onnx`) |

COLMAP reference (runtime, no Python):

- SIFT features: classical C++ (VLFeat / OpenCV)
- ALIKED features: ONNX Runtime
- LightGlue matcher: ONNX → we replace this stage with **GGML**

## Download

Release: [cloudViewer_downloads / LightGlue](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/LightGlue)

Cache: `~/cloudViewer_data/extract/lightglue_models/` (or `aicore_lightglue_model_cache_dir()`)

| Download | Matcher for | Quant | Notes |
|----------|-------------|-------|-------|
| [`sift-lightglue-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f16.gguf) | SIFT | F16 | **default for plugin matching** |
| [`sift-lightglue-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-q8_0.gguf) | SIFT | Q8_0 | smaller; ~93% recall vs F32 |
| [`sift-lightglue-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f32.gguf) | SIFT | F32 | reference |
| [`aliked-lightglue-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-f16.gguf) | ALIKED | F16 | matcher only until ONNX extractor lands |
| [`aliked-lightglue-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-q8_0.gguf) | ALIKED | Q8_0 | matcher only |
| [`aliked-lightglue-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-f32.gguf) | ALIKED | F32 | reference |

## Architecture

- GGUF key: `lightglue`
- Inputs: keypoints, row-major descriptors, image sizes; SIFT also uses scale + orientation (radians)
- Outputs: mutual match index pairs + scores
- Backends: CPU / CUDA / Vulkan via AICore ggml

## Dev-only fixtures

`scripts/extract_aliked_features.py` generates LGINP01 test fixtures for AICore contract tests — **not used by the GUI plugin**.
