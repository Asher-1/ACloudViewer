# qLightGlue

Sparse feature matching for ACloudViewer — **SIFT / ALIKED LightGlue** on shared **`libAICore.so`**:

```
Image pair ──► SIFT LightGlue   : OpenCV RootSIFT → sift-lightglue-*.gguf
            ──► ALIKED LightGlue: AICore ALIKED   → aliked-lightglue-*.gguf
```

Same architectural split as [COLMAP](https://github.com/colmap/colmap) and [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML): **feature extraction** and **matching** are separate stages.

## Features

- Two-image matching from DB tree or disk (dual preview panels, click-to-zoom thumbnails)
- **Two matcher families**, each with F16 / Q8_0 / F32 variants:
  - **SIFT LightGlue** — OpenCV RootSIFT + GGML matcher
  - **ALIKED LightGlue** — AICore GGML extractor + GGML matcher
- Built-in GGUF download & cache
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
| `AICore_ENABLED` | GGML inference in `libAICore.so` (LightGlue, ALIKED) |
| `PLUGIN_STANDARD_QLIGHTGLUE` | This plugin |
| `BUILD_OPENCV=ON` | **Required** for SIFT LightGlue extraction |

## Usage

1. **Plugins → LightGlue Feature Matching**
2. Select model variant — **F16 (recommended)**; downloads on first Run if missing
3. Pick two images → **Run**

### Model families

| Family | Extractor | Model GGUF | Cache directory |
|--------|-----------|------------|-----------------|
| SIFT LightGlue | OpenCV RootSIFT | `sift-lightglue-{f16,q8_0,f32}.gguf` | `~/cloudViewer_data/extract/lightglue_models/` |
| ALIKED LightGlue | `aliked-n16rot-{f16,f32}.gguf` | `aliked-lightglue-{f16,q8_0,f32}.gguf` | `lightglue_models/` + `aliked_models/` |

### Pipeline (COLMAP-aligned)

| Stage | SIFT path | ALIKED path |
|-------|-----------|-------------|
| Feature extraction | OpenCV RootSIFT (C++) | AICore GGML (`aliked-n16rot-*.gguf`) |
| LightGlue matcher | **GGML** (`sift-lightglue-*.gguf`) | **GGML** (`aliked-lightglue-*.gguf`) |

## Models

| Card | Content |
|------|---------|
| [models/MODEL_CARD.md](models/MODEL_CARD.md) | SIFT + ALIKED LightGlue matchers and ALIKED extractor |

Default matching models:

- SIFT: [sift-lightglue-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f16.gguf)
- ALIKED: [aliked-lightglue-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-f16.gguf) + `aliked-n16rot-f16.gguf`

User guide: [docs/guides/plugins/qLightGlue.md](../../../../docs/guides/plugins/qLightGlue.md)

## References

- [LightGlue (ICCV 2023)](https://github.com/cvg/LightGlue)
- [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML)
