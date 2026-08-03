# qFaceDetect — Face Detection & Recognition Plugin

Run [face-detect.cpp](https://github.com/mudler/face-detect.cpp) **GGUF model packs** in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for SCRFD/YuNet face detection, ArcFace/SFace recognition, age/gender analysis, and identity verification.

## Architecture

```
GUI (FaceDetect dialog) ──► libAICore (facedetect_capi) ──► GGML CNN
                              ├── detect  → boxes + 5 landmarks
                              ├── analyze → age + gender
                              └── verify  → cosine distance + optional anti-spoof
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| GGML face-detect engine | `core/AICore/src/facedetect/` |
| Plugin | `plugins/core/Standard/qFaceDetect/` |

## Enable and build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QFACEDETECT=ON \
  .

cmake --build build_app --target QFACEDETECT_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `AICore_ENABLED` | Build `libAICore.so` (shared with qDA3, qDeepLSD, qLightGlue, qFreeSplatter) |
| `PLUGIN_STANDARD_QFACEDETECT` | This plugin |

Example outputs: `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQFACEDETECT_PLUGIN.so`.

Requires system **libjpeg** (e.g. `libjpeg-dev` on Ubuntu).

## GUI usage

**Menu:** Plugins → **Face Detect**

1. Choose a **model pack** (buffalo_l recommended; yunet-sface for Apache-2.0 / commercial use).
2. Select **Mode**: Detect, Analyze, or Verify.
3. Set **Device** (`Auto` / CUDA / Vulkan / CPU) and **Threads**.
4. Pick input image(s) from disk or the DB tree (click the thumbnail to enlarge).
5. Click **Run** — models download from Hugging Face on first use.

### Modes

| Mode | Output |
|------|--------|
| **Detect** | Face boxes + 5 SCRFD keypoints drawn on source image |
| **Analyze** | Detect + predicted age and gender (M/F) per face |
| **Verify** | Cosine distance between primary faces in two images; optional MiniFASNet anti-spoof veto |

Default verify threshold: **0.35** (insightface buffalo convention).

### Model packs

Official weights: [cloudViewer_downloads qFaceDetect release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/qFaceDetect) — **seven F16 publish files** (selective quant — Gemm heads F16, conv backbones F32). Upstream: [mudler/face-detect-gguf](https://huggingface.co/mudler/face-detect-gguf).

| Pack | Best for |
|------|----------|
| **buffalo_l** | Default — SCRFD + ArcFace 512-d |
| **buffalo_m** / **buffalo_s** | Smaller buffalo variants |
| **buffalo_sc** | Compact detect + recognize only |
| **antelopev2** | Highest accuracy (R100) |
| **yunet-sface** | **Apache-2.0** commercial use |
| **landmarks-2d106-1k3d68** | Dense landmarks only (not for this dialog — see MODEL_CARD) |

See [MODEL_CARD.md](../../../plugins/core/Standard/qFaceDetect/models/MODEL_CARD.md) for download links and licensing.

### DB export options

| Checkbox | Default | Output |
|----------|---------|--------|
| Add annotated ccImage to DB tree | On | `ccImage` with boxes, landmarks, and labels (Detect/Analyze) |

Verify mode logs cosine distance and match verdict to the dialog; no image export.

### Inference device (Auto)

| Platform | Auto priority |
|----------|---------------|
| macOS | Metal → CPU |
| Linux / Windows | CUDA → Vulkan → CPU (when CUDA backend is built) |

### Model cache

| Platform | Default directory |
|----------|-------------------|
| Linux | `$HOME/cloudViewer_data/extract/facedetect_models` |
| Windows | `%USERPROFILE%\cloudViewer_data\extract\facedetect_models` |
| Override | `CLOUDVIEWER_DATA_ROOT` → `<root>/extract/facedetect_models` |

Default download: [buffalo_l.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qFaceDetect/buffalo_l.gguf)

## C API (brief)

Header: `core/AICore/include/aicore/facedetect_capi.h`

```c
#include "aicore/facedetect_capi.h"

aicore_facedetect_options* opts = aicore_facedetect_options_new();
aicore_facedetect_options_set_device(opts, "auto");
aicore_facedetect_ctx* ctx =
    aicore_facedetect_load_opts("buffalo_l.gguf", opts);

char* json = aicore_facedetect_detect_path_json(ctx, "photo.jpg");
/* {"faces":[{"score":…,"box":[…],"landmarks":[[x,y],…]}, …]} */
aicore_facedetect_free_string(json);
aicore_facedetect_free(ctx);
aicore_facedetect_options_free(opts);
```

## Further reading

- Developer README (build targets): [`plugins/core/Standard/qFaceDetect/README.md`](../../../plugins/core/Standard/qFaceDetect/README.md)
- [face-detect.cpp](https://github.com/mudler/face-detect.cpp)
- [Asher-1/Face_AI](https://github.com/Asher-1/Face_AI) — related InsightFace REST API (ONNX/TensorRT)
- [insightface](https://github.com/deepinsight/insightface) (original models)
