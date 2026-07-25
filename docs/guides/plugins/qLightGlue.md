# qLightGlue — LightGlue Feature Matching Plugin

Run [LightGlue](https://github.com/cvg/LightGlue) **GGUF models** in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml), derived from [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML)) for sparse feature matching between image pairs.

## Architecture

```
GUI (LightGlue dialog) ──► OpenCV RootSIFT extraction ──► libAICore (lightglue_capi) ──► ggml matcher
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| Plugin | `plugins/core/Standard/qLightGlue/` |
| Feature extraction | OpenCV RootSIFT (built-in, C++) |
| Matcher | GGML LightGlue (GGUF weights) |

Same architectural split as [COLMAP](https://github.com/colmap/colmap): **feature extraction** (OpenCV) and **LightGlue matching** (ggml) are separate stages. GGUF weights are **matcher-only**.

## Enable and build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
  -DBUILD_OPENCV=ON \
  .

cmake --build build_app --target QLIGHTGLUE_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `AICore_ENABLED` | Build `libAICore.so` (shared with qDA3, qFreeSplatter) |
| `PLUGIN_STANDARD_QLIGHTGLUE` | This plugin |
| `BUILD_OPENCV=ON` | **Required** for SIFT feature extraction |
| `AICore_USE_VULKAN` / `AICore_USE_METAL` | Linux/Windows: Vulkan ON; macOS: Metal ON (Vulkan unsupported) |

Example outputs: `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQLIGHTGLUE_PLUGIN.so`.

## GUI usage

**Menu:** Plugins → **LightGlue Feature Matching**

1. Select image(s) in the DB tree, or use **Browse** / **Add Folder** to pick files.
2. Choose a **Model** — use **SIFT F16 (recommended)**. Downloads on first run if missing.
3. Set **Device** (`Auto` / Metal / Vulkan (Linux/Windows) / CUDA / CPU), thread count, and minimum match score.
4. Assign two images to **Slot 1** and **Slot 2** (auto-assigned from selection or file pool).
5. Click **Run**; match results appear as keypoint line visualization in the DB tree.

### Modes

| Mode | Output |
|------|--------|
| Match | Keypoint matches between two images → DB tree visualization + JSON export |
| Model Info | GGUF model metadata and architecture details |

### Pipeline (COLMAP-aligned)

| Stage | SIFT path (supported) | ALIKED path (planned) |
|-------|----------------------|------------------------|
| Feature extraction | OpenCV RootSIFT (C++) | ONNX runtime (future) |
| LightGlue matcher | **GGML** (`sift-lightglue-*.gguf`) | **GGML** (`aliked-lightglue-*.gguf`) |

### Inference device (Auto)

| Platform | Auto priority |
|----------|---------------|
| macOS | Metal → CPU |
| Linux / Windows | Vulkan → CPU |

### Model cache

| Platform | Default directory |
|----------|-------------------|
| Linux | `$HOME/cloudViewer_data/extract/lightglue_models` |
| Windows | `%USERPROFILE%\cloudViewer_data\extract\lightglue_models` |
| Override | `CLOUDVIEWER_DATA_ROOT` → `<root>/extract/lightglue_models` |

Default model: [sift-lightglue-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f16.gguf)

## DB tree integration

- **Input:** Select two `ccImage` entities in the DB tree; they auto-populate the input slots.
- **Output:** Match visualization entity (green keypoint lines) added as child of first image.
- **Export:** JSON match file with keypoint coordinates and confidence scores.
- **DB source images** panel: collapsible list of available images from the DB tree.

## Further reading

- Full plugin README (pipeline, COLMAP alignment): [`plugins/core/Standard/qLightGlue/README.md`](../../../plugins/core/Standard/qLightGlue/README.md)
- Model card: [`plugins/core/Standard/qLightGlue/models/MODEL_CARD.md`](../../../plugins/core/Standard/qLightGlue/models/MODEL_CARD.md)
- [LightGlue (ICCV 2023)](https://github.com/cvg/LightGlue) · [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML)
