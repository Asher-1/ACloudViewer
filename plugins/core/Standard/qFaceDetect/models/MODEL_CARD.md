# Face Detect GGUF models

Published packs for [face-detect.cpp](https://github.com/mudler/face-detect.cpp). Related InsightFace ecosystem reference: [Asher-1/Face_AI](https://github.com/Asher-1/Face_AI).

**Download source (ACloudViewer default):** [cloudViewer_downloads — qFaceDetect release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/qFaceDetect)

Upstream origin: [mudler/face-detect-gguf](https://huggingface.co/mudler/face-detect-gguf) (same F16 selective-quant recipe).

## Official quantization

Each published GGUF is the **F16 publish variant** (selective quant — Gemm heads F16, conv backbones F32). See [face-detect.cpp](https://github.com/mudler/face-detect.cpp) conversion docs.

## Pack index

| Pack | Download | Size (approx) | Detector | Recognizer | Dim | License | Plugin default |
|------|----------|---------------|----------|------------|-----|---------|----------------|
| **buffalo_l** | [`buffalo_l.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qFaceDetect/buffalo_l.gguf) | ~170 MiB | SCRFD det_10g | ArcFace ResNet50 | 512 | Non-commercial | **yes** |
| **buffalo_m** | [`buffalo_m.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qFaceDetect/buffalo_m.gguf) | ~157 MiB | SCRFD | ArcFace ResNet50 | 512 | Non-commercial | |
| **buffalo_s** | [`buffalo_s.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qFaceDetect/buffalo_s.gguf) | ~17 MiB | SCRFD | ArcFace ResNet50 | 512 | Non-commercial | |
| **buffalo_sc** | [`buffalo_sc.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qFaceDetect/buffalo_sc.gguf) | ~12 MiB | SCRFD det_500m | MobileFaceNet | 512 | Non-commercial | |
| **antelopev2** | [`antelopev2.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qFaceDetect/antelopev2.gguf) | ~253 MiB | SCRFD-10G | ArcFace R100 glint360k | 512 | Non-commercial | |
| **yunet-sface** | [`yunet-sface.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qFaceDetect/yunet-sface.gguf) | ~25 MiB | YuNet | SFace | 128 | **Apache-2.0** | commercial use |
| **landmarks** | [`landmarks-2d106-1k3d68.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qFaceDetect/landmarks-2d106-1k3d68.gguf) | ~133 MiB | — (heads only) | — | — | Non-commercial | not for dialog |

`buffalo_l` includes genderage + MiniFASNet anti-spoof when present in the pack.

## Recommended packs

| Use case | Pack |
|----------|------|
| Default (detect / analyze / verify) | **buffalo_l** |
| Highest accuracy | **antelopev2** |
| Smallest buffalo | **buffalo_s** or **buffalo_sc** |
| Commercial / Apache-2.0 | **yunet-sface** |

## Custom GGUF

Use **Custom…** in the dialog to load a locally converted GGUF. Convert locally: [face-detect.cpp docs/conversion.md](https://github.com/mudler/face-detect.cpp/blob/main/docs/conversion.md).

## landmarks-2d106-1k3d68 (engine-only)

Dense 106-pt 2D + 68-pt 3D landmark heads only — **not** usable alone in the qFaceDetect dialog.

## Licensing

- **buffalo_***, **antelopev2**, **landmarks**: insightface **non-commercial** license.
- **yunet-sface**: **Apache-2.0** — commercial-friendly alternative.

Catalog in code: `aicore_facedetect_model_at()` in `core/AICore/include/aicore/facedetect_capi.h`.
