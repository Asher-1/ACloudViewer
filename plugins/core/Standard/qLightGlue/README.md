# qLightGlue

LightGlue sparse feature matching for ACloudViewer — **native C++ end-to-end** for the SIFT path:

```
Image pair → OpenCV RootSIFT → AICore GGML LightGlue matcher → matches + visualization
```

Same architectural split as [COLMAP](https://github.com/colmap/colmap) and [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML): **feature extraction** and **LightGlue matching** are separate stages. GGUF weights are **matcher-only**.

## Features

- Two-image matching from DB tree or disk (dual preview panels)
- Built-in GGUF model download & cache (SIFT / ALIKED matcher weights)
- **SIFT LightGlue**: OpenCV RootSIFT + GGML — no Python, no ONNX at runtime
- Match visualization entity in DB tree (green keypoint lines)
- Export matches as JSON; Model Info mode for any GGUF

## Build

```bash
cmake -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
  -DBUILD_OPENCV=ON \
  ..
make -j4 QLIGHTGLUE_PLUGIN
```

| Option | Role |
|--------|------|
| `AICore_ENABLED` | GGML LightGlue matcher in `libAICore.so` |
| `PLUGIN_STANDARD_QLIGHTGLUE` | This plugin |
| `BUILD_OPENCV=ON` | **Required** for interactive SIFT feature extraction |

## Usage

1. **Plugins → LightGlue Feature Matching**
2. Select **SIFT F16 (recommended)** — downloads on first Run if missing
3. Pick two images → **Run**

### Pipeline (COLMAP-aligned)

| Stage | SIFT path (supported) | ALIKED path (planned) |
|-------|----------------------|------------------------|
| Feature extraction | OpenCV RootSIFT (C++) | COLMAP uses **ONNX** (`aliked-n16rot.onnx`) — not Python |
| LightGlue matcher | **GGML** (`sift-lightglue-*.gguf`) | **GGML** (`aliked-lightglue-*.gguf`) |

COLMAP does **not** use Python LightGlue at runtime. Neither does this plugin.

ALIKED GGUF models can be loaded for **Model Info**; interactive image matching with ALIKED requires a future native ONNX extractor (same approach as COLMAP).

## Models

See [models/MODEL_CARD.md](models/MODEL_CARD.md). Default for matching:

[sift-lightglue-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f16.gguf)

## References

- [LightGlue (ICCV 2023)](https://github.com/cvg/LightGlue)
- [LightGlue-GGML matcher-only C++](https://github.com/Asher-1/LightGlue-GGML)
- [COLMAP ONNX feature pipeline](https://github.com/colmap/colmap/tree/main/src/colmap/feature)
