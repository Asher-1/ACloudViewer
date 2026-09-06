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
- **Model auto-download** — TRELLIS pipeline models are fetched from the unified
  [Asher-1/Trellis2-models](https://huggingface.co/Asher-1/Trellis2-models)
  Hugging Face mirror into `~/cloudViewer_data/extract/trellis_models/` (see
  `aicore_trellis_model_cache_dir`). The optional RMBG dependency is shared
  with qRMBG and is stored in `~/cloudViewer_data/extract/rmbg_models/` (see
  `aicore_rmbg_model_cache_dir`); missing files are downloaded on demand.
  Every download is verified against the mirror's official LFS **SHA-256**
  fingerprint (streamed while downloading, no extra I/O) plus exact size and
  GGUF magic; manually deployed files are validated the same way on demand.
  The GitHub `trellis2-ggml` release is kept for legacy/fallback reference
  but is no longer queried by the plugin.
- **Try sample data** — shared `ecvTestDataRepository` (Image2Mesh set:
  `image_to_mesh_data.zip` with 33 curated single-image samples; auto-downloaded
  into `~/cloudViewer_data/extract/image_to_mesh_data/`, then pickable from a
  combo).
- **Live per-step previews** — the dialog's *Pipeline steps* strip shows a
  thumbnail per core stage (source → preprocess/RMBG → SS voxel set → mesh
  keyframe → texture → GLB). Voxel sets (`T2VOX01`) stream live during the
  sparse-structure flow, marching-cubes mesh keyframes (`T2MESH01`) replay
  after the final decode (AICore preview callbacks, ABI 2).
- **One-click export** — *Generate + GLB* runs the full chain and writes the
  textured GLB into the Save-GLB directory (defaults to `~/Downloads/TRELLIS`).
  The **Export / Print** page re-bakes the last result with a chosen atlas
  size / component filter, and **Print wrap (CGAL)** re-meshes it into a
  watertight Alpha-Wrap print mesh on a worker thread (vertex-PBR preview
  entity in the DB tree, plus a full projected-PBR GLB for textured
  generations) when the build ships CGAL >= 5.5.
- **Backend numerical parity** — the engine matches the upstream reference
  numerics on every backend: exact materialized F32 attention (chunked above
  the 12 GiB score budget; flash opt-in), F32-accumulate matmuls in the
  diffusion graphs, cuBLAS TF32 and Vulkan fp16-accumulate pipelines disabled
  (`aicore::apply_trellis_math_profile`), matching the upstream CUDA/Vulkan
  parity fixes.
- **CuMesh GPU chart clustering (optional)** — when ACloudViewer is built
  with `AICore_USE_CUMESH=ON` (needs CUDA + libtorch, `CUMESH_TORCH_DIR`),
  the GLB bake's Auto unwrap uses CuMesh normal-cone chart clustering as
  hard xatlas chart boundaries, matching the upstream mesh2glb default;
  otherwise the chartless `simple_unwrap` fallback is used (same as
  upstream's `T2GLB_NOCUMESH`).
- **f32 exact mode** — the Quantization combo uses the published f32 chaotic
  chain (DINO, sparse/SLAT flows, occupancy and shape decoders). Texture-only
  weights remain on their published f16 files; all required files download
  from the same HF catalog.
- **Backend A/B harness** — `core/AICore/src/tasks/trellis/tools/trellis_backend_ab.py` runs one image
  through several backends/qualities and prints a per-stage timing table
  with geometry hashes (same methodology as `ggml_upgrade_verify.py`).

## Build

```bash
cmake -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QTRELLIS=ON ..
cmake --build build_app --target QTRELLIS_PLUGIN -j4
```

Requires AICore (`libAICore.so`). The TRELLIS.2 models need a GPU with at
least ~8 GB VRAM for the 512 preset (or ~12 GB for the 1024 cascade); CPU
inference works but is slow.

## Models

Default presets use the **q8** chain (every model that publishes a q8 variant;
halves the memory footprint so the 512 preset fits small GPUs). The
precision-sensitive decoders `shape_dec` / `shape_enc` / `tex_dec` have no q8
variant and always stay f16 — sparse subdivision and UV decoding are not
robust to Q8 weight rounding. The full **f16** reference chain stays
selectable in the dialog's Quantization combo.

| Role | Files (HF mirror) |
|------|-------------------|
| Conditioning | `dino_f16.gguf` / `dino_q8.gguf` / `dino_f32.gguf` |
| Sparse structure | `ss_flow_f16.gguf` / `ss_flow_q8.gguf` / `ss_flow_f32.gguf`, `ss_dec_f16.gguf` / `ss_dec_q8.gguf` / `ss_dec_f32.gguf` |
| Shape 512 | `slat_flow_f16.gguf` / `slat_flow_q8.gguf` / `slat_flow_f32.gguf`, `shape_dec_f16.gguf` / `shape_dec_f32.gguf` |
| Shape 1024 | `slat_flow_1024_f16.gguf` / `slat_flow_1024_q8.gguf` / `slat_flow_1024_f32.gguf` |
| PBR texturing | `shape_enc_f16.gguf`, `tex_dec_f16.gguf`, `tex_slat_flow_512_f16.gguf` / `tex_slat_flow_512_q8.gguf`, `tex_slat_flow_1024_f16.gguf` / `tex_slat_flow_1024_q8.gguf` |
| Background removal | `rmbg_f16.gguf` (shared with qRMBG) |

See [models/MODEL_CARD.md](models/MODEL_CARD.md) for details and licenses.
