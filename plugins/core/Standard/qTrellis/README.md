# qTrellis — TRELLIS.2 image-to-3D (GGML)

Turns a single image into a 3D mesh with per-vertex PBR materials, entirely in
C++/ggml (no PyTorch at runtime). In-tree port of
[trellis-ggml](https://github.com/Asher-1/trellis-ggml), integrated into the
AICore shared runtime (`aicore_trellis_*` C API, `core/AICore/src/tasks/trellis/`).

## Pipeline

```
image ──► (RMBG-2.0 background removal, optional) ──► preprocess
      ──► DINOv3 encode ──► sparse-structure flow ──► occupancy decode (64³)
      ──► shape-SLAT flow ──► shape decode (512³ / 1024³ dual grid)
      ──► mesh extraction ──► (PBR texture stage) ──► ccMesh / GLB
```

## Features

- **Three quality presets** — Coarse 64³ preview, Standard 512³ + PBR
  (recommended), Full 1024³ cascade + PBR.
- **PBR textures** — per-vertex base color (imported as RGB colors),
  metallic / roughness / alpha (imported as scalar fields).
- **AI background removal** — reuses the in-tree RMBG-2.0 engine
  (`rmbg_f16.gguf`, shared with the qRMBG plugin; falls back to the
  solid-color heuristic when absent).
- **GLB export** — UV-atlas-textured glTF 2.0 binary via `aicore_trellis_bake_glb`
  (xatlas + meshoptimizer, all CPU).
- **Model auto-download** — models are fetched from the
  [`trellis2-ggml` release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/trellis2-ggml)
  into `~/cloudViewer_data/extract/trellis_models/` (see
  `aicore_trellis_model_cache_dir`); missing files are downloaded on demand.
- **Try sample data** — shared `ecvTestDataRepository` (Image2Mesh set:
  `image_to_mesh_data.zip` with 33 curated single-image samples; auto-downloaded
  into `~/cloudViewer_data/extract/image_to_mesh_data/`, then pickable from a
  combo).

## Build

```bash
cmake -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QTRELLIS=ON ..
cmake --build build_app --target QTRELLIS_PLUGIN -j4
```

Requires AICore (`libAICore.so`). The TRELLIS.2 models need a GPU with at
least ~8 GB VRAM for the 512 preset (or ~12 GB for the 1024 cascade); CPU
inference works but is slow.

## Models

| Role | Files (trellis2-ggml release) |
|------|-------------------------------|
| Conditioning | `dino_f16.gguf` / `dino_q8.gguf` |
| Sparse structure | `ss_flow_q8.gguf`, `ss_dec_f16.gguf` / `ss_dec_q8.gguf` |
| Shape 512 | `slat_flow_q8.gguf`, `shape_dec_f16.gguf` |
| Shape 1024 | `slat_flow_1024_q8.gguf` |
| PBR texturing | `shape_enc_f16.gguf`, `tex_dec_f16.gguf`, `tex_slat_flow_512_q8.gguf`, `tex_slat_flow_1024_q8.gguf` |
| Background removal | `rmbg_f16.gguf` (shared with qRMBG) |

See [models/MODEL_CARD.md](models/MODEL_CARD.md) for details and licenses.
