# qLightGlue — LightGlue Feature Matching Plugin

Sparse feature matching in ACloudViewer via **SIFT / ALIKED LightGlue** on shared **`libAICore.so`** ([ggml](https://github.com/ggml-org/ggml)) — no Python or PyTorch at runtime.

## Model families

Each family ships **F16 (recommended)**, **Q8_0 (smaller)**, and **F32 (reference)** quantizations.

| Family | Pipeline | Extractor | GGUF weights | Default cache |
|--------|----------|-----------|--------------|---------------|
| **SIFT LightGlue** | RootSIFT → LightGlue matcher | OpenCV RootSIFT (C++) | `sift-lightglue-*.gguf` | `~/cloudViewer_data/extract/lightglue_models/` |
| **ALIKED LightGlue** | ALIKED → LightGlue matcher | AICore GGML (`aliked-n16rot-*.gguf`) | `aliked-lightglue-*.gguf` + matching extractor | `~/cloudViewer_data/extract/lightglue_models/` |

Default downloads (first **Run**):

| Family | Default GGUF |
|--------|----------------|
| SIFT LightGlue | [sift-lightglue-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/sift-lightglue-f16.gguf) |
| ALIKED LightGlue | [aliked-lightglue-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/LightGlue/aliked-lightglue-f16.gguf) + `aliked-n16rot-f16.gguf` (same quant suffix as matcher) |

## Architecture

```
                    ┌─ OpenCV RootSIFT ──► sift-lightglue-*.gguf ────────┐
Image pair ─────────┤─ AICore ALIKED ──► aliked-lightglue-*.gguf ────────┼──► matches + DB visualization
                              libAICore.so (ggml)
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| Plugin | `plugins/core/Standard/qLightGlue/` |
| SIFT extraction | OpenCV RootSIFT (`BUILD_OPENCV=ON`) |
| ALIKED extraction | `aicore_aliked_*` |
| LightGlue matcher | `aicore_lightglue_*` |

LightGlue paths follow the same **extract → match** split as [COLMAP](https://github.com/colmap/colmap).

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
| `AICore_ENABLED` | Build `libAICore.so` (shared with qDA3, qDeepLSD, qFreeSplatter) |
| `PLUGIN_STANDARD_QLIGHTGLUE` | This plugin |
| `BUILD_OPENCV=ON` | **Required** for SIFT LightGlue feature extraction |
| `AICore_USE_VULKAN` / `AICore_USE_METAL` | Linux/Windows: Vulkan ON; macOS: Metal ON |

Example outputs: `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQLIGHTGLUE_PLUGIN.so`.

## GUI usage

**Menu:** Plugins → **LightGlue Feature Matching**

1. Select image(s) in the DB tree, or use **Browse** / **Add Folder**.
2. Pick a **Model** variant (F16 recommended); missing GGUF files download on first **Run**.
3. Set **Device** (`Auto` / Metal / Vulkan / CUDA / CPU), threads, and minimum match score.
4. Assign two images to **Slot 1** and **Slot 2**; click **Run**.

### Modes

| Mode | Output |
|------|--------|
| Match | Keypoint matches between two images → DB tree visualization + JSON export |
| Model Info | GGUF metadata for any selected model |

### Per-family notes

| Family | When to use | Notes |
|--------|-------------|-------|
| **SIFT LightGlue** | General sparse matching, COLMAP-compatible SIFT | Fast GPU matching; no extra extractor download |
| **ALIKED LightGlue** | Learned local features + LightGlue | Needs **both** `aliked-lightglue-*.gguf` and matching `aliked-n16rot-*.gguf` in cache |

### Inference device (Auto)

| Platform | Auto priority |
|----------|---------------|
| macOS | Metal → CPU |
| Linux / Windows | Vulkan → CPU (CUDA → Vulkan → CPU when `AICore_USE_CUDA=ON`) |

### Model cache

| Content | Default directory |
|---------|-------------------|
| LightGlue matchers (`sift-*`, `aliked-lightglue-*`) + ALIKED extractors (`aliked-n16rot-*`) | `$HOME/cloudViewer_data/extract/lightglue_models` |
| Override root | `CLOUDVIEWER_DATA_ROOT` → `<root>/extract/<subdir>/` |

## DB tree integration

- **Input:** Select two `ccImage` entities; they auto-populate input slots.
- **Output:** Match visualization (green keypoint lines) added under the first image.
- **Export:** JSON with keypoint coordinates and scores.
- **Preview:** Thumbnails support click-to-zoom.

## Further reading

- Plugin README: [`plugins/core/Standard/qLightGlue/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qLightGlue/README.md)
- LightGlue / ALIKED models: [`plugins/core/Standard/qLightGlue/models/MODEL_CARD.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qLightGlue/models/MODEL_CARD.md)
- [LightGlue (ICCV 2023)](https://github.com/cvg/LightGlue) · [LightGlue-GGML](https://github.com/Asher-1/LightGlue-GGML)
