# FreeSplatter GGUF Models

GGUF weights for the **qFreeSplatter** plugin (AICore / FreeSplatter inference).

## Download (CloudViewer)

All models are hosted on [cloudViewer_downloads](https://github.com/Asher-1/cloudViewer_downloads) release [**3dgs**](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/3dgs).

Base URL:

```
https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/
```

The plugin dialog and automatic download use the same URLs. Each input image is center-cropped and resized to **512×512**; precision variants share the same Transformer backbone and differ only in quantization format, numeric precision, and embedded head config.

### Model catalog (release tag `3dgs`)

| Asset | Variant | Quant | Size |
|-------|---------|-------|------|
| `freesplatter-scene-q8_0.gguf` | Scene | Q8_0 | 324.5 MB |
| `freesplatter-scene-f16.gguf` | Scene | F16 | 595.8 MB |
| `freesplatter-scene-f32.gguf` | Scene | F32 | 1.15 GB |
| `freesplatter-object-2dgs-q8_0.gguf` | Object-2DGS | Q8_0 | 324.5 MB |
| `freesplatter-object-2dgs-f16.gguf` | Object-2DGS | F16 | 595.8 MB |
| `freesplatter-object-2dgs-f32.gguf` | Object-2DGS | F32 | 1.15 GB |
| `freesplatter-object-q8_0.gguf` | Object-3DGS (legacy) | Q8_0 | 324.5 MB |
| `freesplatter-object-f16.gguf` | Object-3DGS (legacy) | F16 | 595.8 MB |
| `freesplatter-object-f32.gguf` | Object-3DGS (legacy) | F32 | 1.15 GB |

Object-2DGS **F32** was added to the release set on **2026-07-27** (converted from upstream `freesplatter-object-2dgs.safetensors`; the initial 2026-07-16 batch shipped only Q8_0 + F16 for that variant to keep download size down).

### Scene models (2-view scene reconstruction)

For **2** overlapping scene photos (indoor or outdoor). The plugin uniformly downsamples extra inputs to 2 views.

| Download | Quant | Size | Relative speed | Relative quality | Peak RAM (est.) | Notes |
|------|------|------|----------|----------|----------------|------|
| [`freesplatter-scene-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-q8_0.gguf) | Q8_0 | **324.5 MB** | fastest | good (near-lossless quant) | ~0.8–1.2 GB | **default recommended**; smallest size, fastest load, suitable for CPU / integrated GPU |
| [`freesplatter-scene-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f16.gguf) | F16 | **595.8 MB** | moderate | very good | ~1.2–1.8 GB | upstream FreeSplatter recommended scene precision; balanced quality / speed |
| [`freesplatter-scene-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f32.gguf) | F32 | **1.15 GB** | slowest | best (reference precision) | ~2.0–3.0 GB | full-precision reference model; slower CPU inference, suitable for benchmarking / offline batch |

### Object-2DGS models (recommended, multi-view object reconstruction)

For **3 or more** photos taken around a **single foreground object** with a clean background (background removal strongly recommended). Uses **2D Gaussian surfels** (`use_2dgs=true`, 22 channels): oriented flat disks instead of full 3D ellipsoids.

| Download | Quant | Size | Relative speed | Relative quality | Peak RAM (est.) | Notes |
|------|------|------|----------|----------|----------------|------|
| [`freesplatter-object-2dgs-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-q8_0.gguf) | Q8_0 | **324.5 MB** | fastest | good | ~0.8–1.2 GB × views | **default recommended** for objects; plugin Auto view cap = **24** (trained up to 32) |
| [`freesplatter-object-2dgs-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-f16.gguf) | F16 | **595.8 MB** | moderate | very good | ~1.2–1.8 GB × views | best everyday object quality; use **8–24 views** for thin / detailed geometry |
| [`freesplatter-object-2dgs-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-f32.gguf) | F32 | **1.15 GB** | slowest | best (reference precision) | ~2.0–3.0 GB × views | full-precision reference / benchmarking; slower CPU inference; plugin lists as **Object-2DGS F32** |

**When to prefer Object-2DGS**

- Product photos, sculptures, figurines, mechanical parts with thin edges
- You can capture **8–24** views around the object
- You need **cleaner surfaces** and **fewer floaters** around silhouettes
- Background is removed or uniform (white / green screen)

**Trade-offs**

- More views → better quality but **O(N²) compute** (24 views ≈ 2× cost of 16)
- Exports to SIBR PLY by expanding the 2D scale to a thin 3D ellipsoid (third scale axis fixed small)

### Object-3DGS models (legacy, deprecated)

Legacy **full 3D ellipsoid** object head (`use_2dgs=false`, 23 channels). Kept for backward compatibility; the plugin marks these **deprecated** — prefer Object-2DGS for new work.

| Download | Quant | Size | Relative speed | Relative quality | Peak RAM (est.) | Notes |
|------|------|------|----------|----------|----------------|------|
| [`freesplatter-object-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-q8_0.gguf) | Q8_0 | **324.5 MB** | fastest | good | ~0.8–1.2 GB × views | quick tests only; plugin Auto view cap = **16** |
| [`freesplatter-object-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f16.gguf) | F16 | **595.8 MB** | moderate | very good | ~1.2–1.8 GB × views | OK for **3–8 views** when 2DGS is unavailable |
| [`freesplatter-object-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f32.gguf) | F32 | **1.15 GB** | slowest | best | ~2.0–3.0 GB × views | reference / benchmarking only |

**When Object-3DGS might still be used**

- Very few views (**3–6**) and you need a fast sanity check
- Comparing against older FreeSplatter object checkpoints or published 3DGS baselines

**Trade-offs vs Object-2DGS**

- Faster per-view with fewer inputs, but **more floaters** and **blobbier thin structures**
- Does not benefit from the extra views (16+) that 2DGS was trained for

### Object-2DGS vs Object-3DGS (summary)

| | **Object-2DGS** (recommended) | **Object-3DGS** (deprecated) |
|---|---|---|
| Gaussian type | 2D oriented surfels (2 scale axes) | Full 3D ellipsoids (3 scale axes) |
| Channels | 22 (`gaussian_channels=22`) | 23 |
| GGUF flag | `use_2dgs=true` | `use_2dgs=false` |
| Min views | 3 | 3 |
| Plugin Auto views | **24** (eval); trained up to **32** | **16** |
| Best view count | **8–24** | **3–8** |
| Surface quality | Sharper surfaces, fewer floaters | More floaters on thin / edge geometry |
| Speed (same view count) | Similar per view; higher Auto cap → slower total | Lower Auto cap → faster total run |
| Input prep | Background removal **strongly recommended** | Same |
| Typical use | Products, props, scanned objects, multi-view capture | Legacy / quick low-view tests |

> **Size note:** Q8_0 / F16 / F32 tiers share identical byte size across Scene, Object-2DGS, and Object-3DGS at the same quant level (same transformer backbone; head differs only in `gaussian_channels`, `sh_residual`, `use_2dgs`). Published sizes: Q8_0 **324.5 MB**, F16 **595.8 MB**, F32 **1.15 GB** (Object-2DGS F32 verified **1,231,517,312** bytes, 2026-07-27).
>
> **Performance note:** FreeSplatter has no published standardized benchmark yet; relative speed / memory are empirical rankings on the same device and view count. **CUDA / Vulkan / Metal** backends can significantly reduce inference time. When GPU and GUI share the same card, close the SIBR viewer before inference to avoid VRAM contention.

### Selection guide

| Scenario | Recommended model (click to download) |
|------|---------------------|
| Quick try / laptop CPU (scene) | [`freesplatter-scene-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-q8_0.gguf) |
| Quick try / laptop CPU (object) | [`freesplatter-object-2dgs-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-q8_0.gguf) with **Remove BG** |
| Everyday quality (Scene, 2 views) | [`freesplatter-scene-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f16.gguf) |
| Everyday quality (Object, multi-view) | [`freesplatter-object-2dgs-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-q8_0.gguf) (≤16 views) or [`freesplatter-object-2dgs-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-f16.gguf) (8–24 views) |
| Thin / detailed object (many views) | [`freesplatter-object-2dgs-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-f16.gguf), **16–24 views**, background removed |
| Legacy 3DGS object / ≤8 views only | [`freesplatter-object-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-q8_0.gguf) |
| Best quality / benchmarking | [`freesplatter-scene-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f32.gguf) / [`freesplatter-object-2dgs-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-f32.gguf) / [`freesplatter-object-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f32.gguf) (legacy 3DGS) + GPU backend |

Manual download example:

```bash
# Object-2DGS (recommended)
curl -L -O https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-q8_0.gguf

# Object-2DGS F32 (reference precision)
curl -L -O https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-2dgs-f32.gguf
```

### Convert from upstream (maintainers)

GGUF files are produced by [`scripts/convert.py`](../scripts/convert.py) from TencentARC checkpoints on [Hugging Face](https://huggingface.co/TencentARC/FreeSplatter). Needs `torch`, `safetensors`, `gguf` only (no FreeSplatter repo).

| Upstream checkpoint | `--variant` | Head |
|---------------------|---------------|------|
| `freesplatter-scene.safetensors` | `scene` | 23ch, `sh_residual=true` |
| `freesplatter-object.safetensors` | `object` | 23ch, legacy 3DGS |
| `freesplatter-object-2dgs.safetensors` | `object-2dgs` | 22ch, `use_2dgs=true` |

```bash
cd plugins/core/Standard/qFreeSplatter
mkdir -p .cache

# Download upstream (example: Object-2DGS)
curl -L -o .cache/freesplatter-object-2dgs.safetensors \
  https://huggingface.co/TencentARC/FreeSplatter/resolve/main/freesplatter-object-2dgs.safetensors

# Convert all precision tiers
for outtype in f32 f16 q8_0; do
  python3 scripts/convert.py \
    .cache/freesplatter-object-2dgs.safetensors \
    .cache/freesplatter-object-2dgs-${outtype}.gguf \
    --variant object-2dgs --outtype "$outtype"
done
```

Upload to [cloudViewer_downloads `3dgs`](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/3dgs) (requires `gh` auth):

```bash
gh release upload 3dgs .cache/freesplatter-object-2dgs-f32.gguf \
  -R Asher-1/cloudViewer_downloads --clobber
```

## Model Architecture

All models share the same transformer backbone:
- Patch tokenizer: 8×8 conv, 1024 output channels
- 24-layer self-attention transformer (1024 embd, 16 heads, 64 head_dim)
- Gaussian head: unpatchify to **22** (Object-2DGS) or **23** (Scene / Object-3DGS) channels per pixel

Variant metadata in GGUF (`free-splatter.*` keys):

| Variant | `gaussian_channels` | `use_2dgs` | `sh_residual` |
|---------|---------------------|------------|---------------|
| Scene | 23 | false | true |
| Object-3DGS | 23 | false | false |
| Object-2DGS | 22 | true | false |

### Gaussian Channel Layout (23ch, Scene / Object-3DGS)

| Channel | Field |
|---------|-------|
| 0-2     | xyz position |
| 3-14    | SH coefficients (degree 1, 4×3) |
| 15      | opacity (sigmoid-activated) |
| 16-18   | scale (activated, 3 axes) |
| 19-22   | rotation quaternion (w,x,y,z, normalized) |

### Gaussian Channel Layout (22ch, Object-2DGS)

| Channel | Field |
|---------|-------|
| 0-2     | xyz position |
| 3-14    | SH coefficients (degree 1, 4×3) |
| 15      | opacity (sigmoid-activated) |
| 16-17   | scale (activated, **2 axes** — in-plane surfel extent) |
| 18-21   | rotation quaternion (w,x,y,z, normalized) |

PLY export pads the missing third scale axis to a small constant so SIBR / standard 3DGS viewers can render the surfels.

## Hyperparameters

- Image size: 512×512
- Patch size: 8×8
- SH degree: 1
- Scale activation: sigmoid in [scale_min_act, scale_max_act]
