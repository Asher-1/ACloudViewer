# qTrellis Model Card — TRELLIS.2 image-to-3D

## Model

| Field       | Value                                                                 |
|-------------|-----------------------------------------------------------------------|
| Architecture | TRELLIS.2 image-to-3D (DINOv3 ViT-L conditioning + flow-matching DiTs + FlexiDualGrid VAE) |
| Task        | Single-image 3D mesh generation with per-vertex PBR materials         |
| Input       | One image (PNG/JPEG/WebP/...), any aspect ratio                       |
| Output      | Triangle mesh in a centered unit cube ([-0.5, 0.5]^3); optional PBR: base color, metallic, roughness, alpha |
| License     | TRELLIS.2 weights: `MIT`; DINOv3: `DINOv3 License`; RMBG-2.0: `CC BY-NC 4.0` (commercial via BRIA) |
| Source      | [trellis-ggml](https://github.com/Asher-1/trellis-ggml) -> HF mirror [Asher-1/Trellis2-models](https://huggingface.co/Asher-1/Trellis2-models) |

## Files

All files are published on the unified Hugging Face mirror
[Asher-1/Trellis2-models](https://huggingface.co/Asher-1/Trellis2-models)
(sizes and SHA-256 are the exact LFS metadata; the SHA-256 oid is the
canonical content fingerprint used by the plugin to verify downloads). The
plugin fetches every model (f16, q8, and published f32) from this single source -
the GitHub `trellis2-ggml` release is kept for legacy reference only
and is not queried at runtime.

| Filename                | Size (exact)      | Role              |
|-------------------------|-------------------|-------------------|
| `dino_f16.gguf`         | 606,992,192 B     | DINOv3 conditioning encoder (f16) |
| `dino_q8.gguf`          | 323,876,672 B     | DINOv3 conditioning encoder (q8) |
| `dino_f32.gguf`         | 1,212,544,832 B   | DINOv3 conditioning encoder (f32) |
| `ss_flow_f16.gguf`      | 2,615,168,864 B   | Sparse-structure flow DiT (f16) |
| `ss_flow_q8.gguf`       | 1,418,183,264 B   | Sparse-structure flow DiT (q8) |
| `ss_flow_f32.gguf`      | 5,168,762,720 B   | Sparse-structure flow DiT (f32) |
| `ss_dec_f16.gguf`       | 147,379,616 B     | Occupancy decoder (f16) |
| `ss_dec_q8.gguf`        | 147,379,616 B     | Occupancy decoder (q8) |
| `ss_dec_f32.gguf`       | 294,689,888 B     | Occupancy decoder (f32) |
| `slat_flow_f16.gguf`    | 2,615,319,424 B   | Shape-SLAT flow, 512 (f16) |
| `slat_flow_q8.gguf`     | 1,418,253,184 B   | Shape-SLAT flow, 512 (q8) |
| `slat_flow_f32.gguf`    | 5,169,060,736 B   | Shape-SLAT flow, 512 (f32) |
| `slat_flow_1024_f16.gguf`| 2,630,208,384 B  | Shape-SLAT flow, 1024 cascade (f16) |
| `slat_flow_1024_q8.gguf`| 1,418,253,184 B   | Shape-SLAT flow, 1024 cascade (q8) |
| `slat_flow_1024_f32.gguf` | 5,169,060,736 B | Shape-SLAT flow, 1024 cascade (f32) |
| `shape_dec_f16.gguf`    | 948,745,408 B     | Shape decoder (dual-grid VAE) |
| `shape_dec_f32.gguf`    | 1,896,943,680 B   | Shape decoder (dual-grid VAE, f32) |
| `shape_enc_f16.gguf`    | 709,034,048 B     | PBR: shape encoder |
| `tex_dec_f16.gguf`      | 948,713,856 B     | PBR: texture decoder |
| `tex_slat_flow_512_f16.gguf` | 2,615,421,184 B | PBR: texture flow, 512 (f16) |
| `tex_slat_flow_512_q8.gguf`  | 1,418,308,864 B | PBR: texture flow, 512 (q8) |
| `tex_slat_flow_1024_f16.gguf`| 2,615,421,184 B | PBR: texture flow, 1024 (f16) |
| `tex_slat_flow_1024_q8.gguf` | 1,418,308,864 B | PBR: texture flow, 1024 (q8) |
| `rmbg_f16.gguf`         | 441,451,648 B     | Background removal (shared with qRMBG) |
| `rmbg_f32.gguf`         | 882,846,304 B     | Background removal, f32 reference |
| `rmbg_q8.gguf`          | 258,974,848 B     | Background removal (q8) |

## Presets

Default presets use the **f16** variants throughout (upstream's recommended
precision); q8 alternatives stay selectable in the dialog.

| Preset | Files | Disk (f16 defaults) | VRAM (GPU) |
|--------|-------|---------------------|------------|
| Coarse 64³ preview | dino + ss_flow + ss_dec | ~3.4 GB | ~3.5 GB |
| Standard 512 + PBR | coarse set + slat_flow + shape_dec + PBR set | ~11.2 GB | ~12 GB |
| Full 1024 + PBR | standard set + slat_flow_1024 + tex flow 1024 | ~16.5 GB | ~17 GB |

## Download

Runtime source of truth (single source for all models):

`https://huggingface.co/Asher-1/Trellis2-models/resolve/main/<file>`

TRELLIS pipeline weights use `~/cloudViewer_data/extract/trellis_models/` (see
`aicore_trellis_model_cache_dir`). The optional RMBG dependency uses the shared
`~/cloudViewer_data/extract/rmbg_models/` cache (see
`aicore_rmbg_model_cache_dir`). The dialog downloads missing files
automatically before a run. Integrity checking is layered:

- **Download path** — the streamed bytes are hashed while writing and
  compared against the published LFS **SHA-256** (plus exact size and GGUF
  magic); a mismatch deletes the file and reports the error.
- **Per-dialog presence check** — lightweight (GGUF magic + exact size, no
  full read), so multi-GB caches are not re-hashed on every dialog open.
- **Manual deployments** — files placed in the cache directory by hand are
  accepted when size and magic match; `verifyModelFileSha256()` (plugin
  catalog helper) performs the full content check on demand.

## Backends

- CUDA / Vulkan / Metal / CPU — shared repository-pinned ggml runtime via AICore
  (CUDA gains the Q8_0→Q8_0 copy kernel from the `trellis_merged` ggml patch,
  avoiding a dequant→F32→requant roundtrip for quantized-weight copies).
- Device selection follows AICore's `auto` order (CUDA → Vulkan → CPU on
  Linux/Windows, Metal → CPU on macOS).
- The 64³ occupancy decoder (dense conv3d) always runs on the CPU; the shape
  decoder is auto-placed on the GPU when VRAM permits
  (`aicore_trellis_options_set_shape_dec_placement`).
