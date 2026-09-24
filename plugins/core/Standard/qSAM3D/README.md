# qSAM3D — SAM 3D Objects Image to 3D

Single-image 3D generation with **SAM 3D Objects** running on the in-tree
ggml runtime (`libAICore`, task `sam3d`, ported from the sam-3d-objects-ggml
C++ runtime). The plugin produces:

- a **Gaussian splat PLY** (the neural output of the full pipeline), and
- an optional **FlexiCubes mesh** (101-channel sparse mesh decoder + FlexiCubes
  surface extraction), added to the DB tree as a mesh entity.

The textured-PBR GLB stage of the upstream runtime (CUDA-only renderer plus
non-commercial / GPL license opt-ins) is intentionally not enabled by default.

## Requirements

- `AICore_ENABLED=ON` and `-DPLUGIN_STANDARD_QSAM3D=ON`.
- The published GGUF weights of
  [Asher-1/SAM_3D_OBJECTS_GGUF](https://huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF)
  inside the shared model cache (`~/cloudViewer_data/extract/sam3d_models` by
  default, or the platform data root). The one-click AICore validation gate
  downloads and verifies them; the minimum set for the light tier is
  `moge_vitl-f16.gguf` plus the five `q4_k` stage files:
  - `ss_generator-q4_k.gguf`, `ss_decoder-q4_k.gguf`,
    `slat_generator-q4_k.gguf`, `slat_decoder_gs-q4_k.gguf`,
    `slat_decoder_mesh-q4_k.gguf` (mesh output only).
- Optional: the shared **RMBG** model (`rmbg_q8.gguf` / `rmbg_f16.gguf` under
  the RMBG cache, downloadable via qRMBG) for automatic object-mask matting.
  Without it the source image is used as-is (or provide an image whose alpha
  channel is the binary mask).

## Usage

1. `Plugins > SAM 3D Objects Image to 3D`.
2. Pick the source image and the output directory for the Gaussian PLY.
3. Choose device / quantization / steps / seed. Defaults follow the official
   trajectory gate (`q4_k`, 25 steps, seed 42).
4. Run. Stage progress and the AICore log appear in the dialog; the mesh is
   added to the DB tree and the PLY path is printed when the pipeline ends.

First runs load ~2-6 GB of weights from the model cache; an RTX 3060-class GPU
completes a full 25-step run in roughly 2.5-5 minutes depending on
quantization.

## Notes

- The mask semantics follow the official pipeline: binary alpha (`> 0`
  foreground). RMBG matting, image alpha channels, and mask images all fold
  into the same contract.
- All model metadata (URLs, sizes, SHA-256) is owned by the AICore runtime
  catalog and verified by the validation gate — the plugin carries no second
  model table.
- Weights inherit the upstream SAM 3D Objects license.
