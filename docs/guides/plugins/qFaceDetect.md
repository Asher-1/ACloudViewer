# qFaceDetect — Face Detection & Recognition Plugin

Run [face-detect.cpp](https://github.com/mudler/face-detect.cpp) **GGUF model packs** in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for SCRFD/YuNet face detection, ArcFace/SFace recognition, age/gender analysis, and identity verification.

![qFaceDetect registry and recognition workflow](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFaceDetect/images/qFaceDetect.png?raw=1)

## Architecture

```
GUI (FaceDetect dialog) ──► libAICore (facedetect_capi) ──► GGML CNN
                              ├── detect_image  → typed boxes + 5 landmarks
                              ├── analyze_image → typed age + gender + embedding
                              ├── dense_landmarks_image → typed 2D / 3D points
                              └── verify_images  → typed distance + anti-spoof result
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| GGML face-detect engine | `core/AICore/src/tasks/facedetect/` |
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

Image decode uses Qt's built-in codecs (JPEG/PNG via the Qt image plugins); no direct system libjpeg dependency.
Decoded pixels are passed to AICore as borrowed, stride-aware
`aicore_image_view` values. Detect, Analyze, Dense Landmarks and Verify use
typed results in the interactive path; path and JSON functions remain
compatibility entry points.

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

![qFaceDetect live multi-face recognition](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFaceDetect/images/qFaceDetect_video.png?raw=1)

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

See [MODEL_CARD.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFaceDetect/models/MODEL_CARD.md) for download links and licensing.

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

int detect_faces(const uint8_t *pixels, int32_t width, int32_t height,
                 size_t row_stride)
{
    aicore_facedetect_options *opts = aicore_facedetect_options_new();
    aicore_facedetect_options_set_device(opts, "auto");
    aicore_facedetect_ctx *ctx =
        aicore_facedetect_load_opts("buffalo_l.gguf", opts);
    aicore_image_view image = {
        pixels, width, height, row_stride, AICORE_IMAGE_RGB8
    };

    int rc = aicore_facedetect_detect_image(ctx, &image);
    if (rc == 0) {
        size_t count = aicore_facedetect_detection_count(ctx);
        for (size_t i = 0; i < count; ++i) {
            aicore_facedetect_detection face;
            aicore_facedetect_detection_at(ctx, i, &face);
            /* consume face.score, face.x1... and face.landmarks_xy10 */
        }
    }

    aicore_facedetect_free(ctx);
    aicore_facedetect_options_free(opts);
    return rc;
}
```

The caller retains ownership of `pixels`; the view is borrowed only for the
call. Analyze, dense landmarks and verification follow the same image-view and
typed-result ownership rules. See the header for their accessors and
`aicore_facedetect_last_pipeline_timings`.

## Further reading

- Developer README (build targets): [`plugins/core/Standard/qFaceDetect/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFaceDetect/README.md)
- [face-detect.cpp](https://github.com/mudler/face-detect.cpp)
- [Asher-1/Face_AI](https://github.com/Asher-1/Face_AI) — related InsightFace REST API (ONNX/TensorRT)
- [insightface](https://github.com/deepinsight/insightface) (original models)
