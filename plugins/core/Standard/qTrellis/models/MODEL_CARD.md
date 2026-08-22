# qTrellis Model Card — TRELLIS.2 image-to-3D

## Model

| Field       | Value                                                                 |
|-------------|-----------------------------------------------------------------------|
| Architecture | TRELLIS.2 image-to-3D (DINOv3 ViT-L conditioning + flow-matching DiTs + FlexiDualGrid VAE) |
| Task        | Single-image 3D mesh generation with per-vertex PBR materials         |
| Input       | One image (PNG/JPEG/WebP/...), any aspect ratio                       |
| Output      | Triangle mesh in a centered unit cube ([-0.5, 0.5]^3); optional PBR: base color, metallic, roughness, alpha |
| License     | TRELLIS.2 weights: `MIT`; DINOv3: `DINOv3 License`; RMBG-2.0: `CC BY-NC 4.0` (commercial via BRIA) |
| Source      | [trellis-ggml](https://github.com/Asher-1/trellis-ggml) → `trellis2-ggml` release |

## Files

All files are published in the `trellis2-ggml` release of
[cloudViewer_downloads](https://github.com/Asher-1/cloudViewer_downloads)
(sizes measured from the release assets):

| Filename                | Size (approx.) | Role              |
|-------------------------|----------------|-------------------|
| `dino_f16.gguf`         | ~580 MB        | DINOv3 conditioning encoder (f16) |
| `dino_q8.gguf`          | ~310 MB        | DINOv3 conditioning encoder (q8, recommended) |
| `ss_flow_q8.gguf`       | ~1.35 GB       | Sparse-structure flow DiT |
| `ss_dec_f16.gguf`       | ~140 MB        | Occupancy decoder (f16) |
| `ss_dec_q8.gguf`        | ~140 MB        | Occupancy decoder (q8) |
| `slat_flow_q8.gguf`     | ~1.35 GB       | Shape-SLAT flow, 512 |
| `slat_flow_1024_q8.gguf`| ~1.35 GB       | Shape-SLAT flow, 1024 cascade |
| `shape_dec_f16.gguf`    | ~905 MB        | Shape decoder (dual-grid VAE) |
| `shape_enc_f16.gguf`    | ~676 MB        | PBR: shape encoder |
| `tex_dec_f16.gguf`      | ~905 MB        | PBR: texture decoder |
| `tex_slat_flow_512_q8.gguf` | ~1.35 GB   | PBR: texture flow, 512 |
| `tex_slat_flow_1024_q8.gguf`| ~1.35 GB   | PBR: texture flow, 1024 |
| `rmbg_f16.gguf`         | ~420 MB        | Background removal (shared with qRMBG) |

## Presets

| Preset | Files | VRAM (GPU) |
|--------|-------|------------|
| Coarse 64³ preview | dino + ss_flow + ss_dec | ~4.5 GB |
| Standard 512 + PBR | coarse set + slat_flow + shape_dec + PBR set | ~8 GB |
| Full 1024 + PBR | standard set + slat_flow_1024 + tex flow 1024 | ~12 GB |

## Download

Mirror hosted by ACloudViewer:

`https://github.com/Asher-1/cloudViewer_downloads/releases/download/trellis2-ggml/<file>`

The model cache directory is `~/cloudViewer_data/extract/trellis_models/` (see
`aicore_trellis_model_cache_dir`). The dialog downloads missing preset files
automatically before a run.

## Backends

- CUDA / Vulkan / Metal / CPU — shared ggml v0.18.1 runtime via AICore
  (CUDA gains the Q8_0→Q8_0 copy kernel from the `trellis_merged` ggml patch,
  avoiding a dequant→F32→requant roundtrip for quantized-weight copies).
- Device selection follows AICore's `auto` order (CUDA → Vulkan → CPU on
  Linux/Windows, Metal → CPU on macOS).
- The 64³ occupancy decoder (dense conv3d) always runs on the CPU; the shape
  decoder is auto-placed on the GPU when VRAM permits
  (`aicore_trellis_options_set_shape_dec_placement`).
