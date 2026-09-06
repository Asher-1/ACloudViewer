# qTrellis — TRELLIS.2 Image-to-3D (GGML)

<p align="center">
  <img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qTrellis/images/qTrellis_f16_1024_pbr.png" width="49%" alt="TRELLIS.2 Full 1024 PBR result">
  <img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qTrellis/images/qTrellis_q8_512_pbr.png" width="49%" alt="TRELLIS.2 Standard 512 PBR result">
</p>

Turns a single image into a 3D triangle mesh with per-vertex PBR materials,
running **natively in-process on ggml** (CPU / CUDA / Vulkan / Metal) — no
Python, PyTorch or external services. In-tree port of
[trellis-ggml](https://github.com/Asher-1/trellis-ggml), integrated into the
AICore shared runtime (`aicore_trellis_*` C API,
`core/AICore/src/tasks/trellis/`).

## Pipeline

```
image ──► (RMBG-2.0 background removal, optional) ──► preprocess
      ──► DINOv3 encode ──► sparse-structure flow ──► occupancy decode (64³)
      ──► shape-SLAT flow ──► shape decode (512³ / 1024³ dual grid)
      ──► mesh extraction ──► (PBR texture stage) ──► ccMesh / GLB
```

## Requirements

- Built with `-DAICore_ENABLED=ON -DPLUGIN_STANDARD_QTRELLIS=ON`
- GGUF models from the
  [cloudViewer_downloads "trellis2-ggml" release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/trellis2-ggml)
  — the dialog auto-downloads missing preset files into
  `~/cloudViewer_data/extract/trellis_models/` on first run
- A GPU with at least ~8 GB VRAM for the Standard 512 preset (~12 GB for the
  Full 1024 cascade); CPU inference works but is slow

## Workflow

1. **Plugins -> Trellis Image-to-3D**
2. Pick an image (open dialog or **Try sample data**).
3. Choose a **preset** and optional **PBR textures** / **AI background
   removal**, then set steps / guidance / seed / device.
4. Click **Generate** — stage progress (DINO → flows → decoder → mesh) is
   shown; the resulting mesh lands in the DB tree.

## Outputs

- DB tree: `ccMesh` (TRELLIS_<preset>_<device>) with normals; PBR base color
  imported as RGB colors, metallic / roughness / alpha as scalar fields.
- GLB export: UV-atlas-textured glTF 2.0 binary via
  `aicore_trellis_bake_glb` (xatlas + meshoptimizer, all CPU).

## Presets

| Preset | Files | VRAM (GPU) |
|--------|-------|------------|
| Coarse 64³ preview | dino + ss_flow + ss_dec | ~4.5 GB |
| Standard 512 + PBR | coarse set + slat_flow + shape_dec + PBR set | ~8 GB |
| Full 1024 + PBR | standard set + slat_flow_1024 + tex flow 1024 | ~12 GB |

## Notes

- The 64³ occupancy decoder (dense conv3d) always runs on the CPU; the shape
  decoder is auto-placed on the GPU when VRAM permits.
- Background removal reuses the in-tree RMBG-2.0 engine (`rmbg_f16.gguf`,
  shared with the qRMBG plugin); it falls back to the solid-color heuristic
  when the model is absent.
- Device selection follows AICore's `auto` order (CUDA → Vulkan → CPU on
  Linux/Windows, Metal → CPU on macOS).
- Model catalog and C-API are contract-tested in
  `core/AICore/tests/trellis/`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Model download stalls | Check the network; the dialog retries missing files on the next run |
| Generate fails with "degenerate alpha bounding box" | The image has no foreground alpha after background removal; disable RMBG or use an image with a distinct subject |
| Out of memory | Use the Coarse 64³ preset or the `q8` decoder variants to cut VRAM |
| Slow on CPU | Expected — the flow DiTs are large; use a GPU backend (`cuda` / `vulkan` / `metal`) |
