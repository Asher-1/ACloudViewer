# FreeSplatter GGUF Models

GGUF weights for the **qFreeSplatter** plugin (AICore / FreeSplatter inference).

## Download (CloudViewer)

All models are hosted on [cloudViewer_downloads](https://github.com/Asher-1/cloudViewer_downloads) release [**3dgs**](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/3dgs).

Base URL:

```
https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/
```

The plugin dialog and automatic download use the same URLs. Each input image is center-cropped and resized to **512×512**; precision variants share the same Transformer backbone and differ only in quantization format and numeric precision.

### Scene models (2-view scene reconstruction)

For **2** overlapping scene photos (indoor or outdoor). The plugin uniformly downsamples extra inputs to 2 views.

| Download | Quant | Size | Relative speed | Relative quality | Peak RAM (est.) | Notes |
|------|------|------|----------|----------|----------------|------|
| [`freesplatter-scene-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-q8_0.gguf) | Q8_0 | **324.5 MB** | fastest | good (near-lossless quant) | ~0.8–1.2 GB | **default recommended**; smallest size, fastest load, suitable for CPU / integrated GPU |
| [`freesplatter-scene-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f16.gguf) | F16 | **595.8 MB** | moderate | very good | ~1.2–1.8 GB | upstream FreeSplatter recommended scene precision; balanced quality / speed |
| [`freesplatter-scene-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f32.gguf) | F32 | **1.15 GB** | slowest | best (reference precision) | ~2.0–3.0 GB | full-precision reference model; slower CPU inference, suitable for benchmarking / offline batch |

### Object models (multi-view object reconstruction)

For **3 or more** photos taken around a single object (the plugin uses up to 16 views; extras are uniformly downsampled).

| Download | Quant | Size | Relative speed | Relative quality | Peak RAM (est.) | Notes |
|------|------|------|----------|----------|----------------|------|
| [`freesplatter-object-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-q8_0.gguf) | Q8_0 | **324.5 MB** | fastest | good | ~0.8–1.2 GB × views | **default recommended**; total time scales linearly with view count |
| [`freesplatter-object-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f16.gguf) | F16 | **595.8 MB** | moderate | very good | ~1.2–1.8 GB × views | more stable object detail; suitable for 4–8 views |
| [`freesplatter-object-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f32.gguf) | F32 | **1.15 GB** | slowest | best | ~2.0–3.0 GB × views | full precision; multi-view + F32 works best with CUDA / Vulkan |

> **Size note:** sizes above are exact GitHub Release asset byte counts (published 2026-07-16). Scene and Object files of the same precision have identical size; they differ in embedded head config (`gaussian_channels`, `sh_residual`, `use_2dgs`).
>
> **Performance note:** FreeSplatter has no published standardized benchmark yet; relative speed / memory are empirical rankings on the same device and view count. **CUDA / Vulkan / Metal** backends can significantly reduce inference time. When GPU and GUI share the same card, close the SIBR viewer before inference to avoid VRAM contention.

### Selection guide

| Scenario | Recommended model (click to download) |
|------|---------------------|
| Quick try / laptop CPU | [`freesplatter-scene-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-q8_0.gguf) or [`freesplatter-object-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-q8_0.gguf) |
| Everyday quality (Scene, 2 views) | [`freesplatter-scene-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f16.gguf) |
| Everyday quality (Object, multi-view) | [`freesplatter-object-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-q8_0.gguf) (≤8 views) or [`freesplatter-object-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f16.gguf) |
| Best quality / benchmarking | [`freesplatter-scene-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f32.gguf) / [`freesplatter-object-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f32.gguf) + GPU backend |

Manual download example:

```bash
curl -L -O https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-q8_0.gguf
```

## Model Architecture

All models share the same transformer backbone:
- Patch tokenizer: 8x8 conv, 1024 output channels
- 24-layer self-attention transformer (1024 embd, 16 heads, 64 head_dim)
- Gaussian head: unpatchify to 23 channels per pixel

### Gaussian Channel Layout (23ch, scene mode)

| Channel | Field |
|---------|-------|
| 0-2     | xyz position |
| 3-14    | SH coefficients (degree 1, 4x3) |
| 15      | opacity (sigmoid-activated) |
| 16-18   | scale (activated) |
| 19-22   | rotation quaternion (w,x,y,z, normalized) |

## Hyperparameters

- Image size: 512×512
- Patch size: 8×8
- SH degree: 1
- Scale activation: sigmoid in [scale_min_act, scale_max_act]
