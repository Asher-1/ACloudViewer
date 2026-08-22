# qSAM3 — SAM 2 / 2.1 / 3 Segmentation

Interactive image segmentation with Segment Anything 2 / 2.1 / 3 models,
running **natively in-process on ggml** (CPU / CUDA / Vulkan / Metal) — no
Python, PyTorch or external services. The plugin is a Qt re-implementation of
the upstream sam3-ggml ImGui demo (`examples/main_image.cpp`), keeping the
same layout and interactions.

## Requirements

- Built with `-DAICore_ENABLED=ON -DPLUGIN_STANDARD_QSAM3=ON`
- A GGUF model from the
  [cloudViewer_downloads "sam" release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/sam)
  (the plugin's model combo lists all 39 published models; a local file can be
  loaded via **Browse...**)

## Workflow

1. **Plugins -> SAM3 Image & Video Segmentation**
2. Choose a model family in the combo and a device, then **Load** — the
   backend (CPU / CUDA / Vulkan / Metal) is resolved and reported in the
   status bar.
3. Load an image (open dialog or drag & drop on the canvas).
4. Prompt and **Segment**:

   | Mode | Interaction | Works with |
   |------|-------------|------------|
   | **Points** | left-click = foreground (green), right-click = background (red) | all models |
   | **Box (PVS)** | drag a rectangle on the canvas (cyan) | all models |
   | **Exemplar (PCS)** | type a text prompt (Enter to run); optional exemplar boxes | SAM3 family only (text detector) |

5. **Clear** removes the prompts; **Export masks** pushes the current
   detections into the DB tree as annotated images.

## Outputs

- Detections list: per-instance score / IoU / box, mask pixel coverage.
- DB tree: annotated `ccImage` (`SAM3_<source>_<device>`) with a mask tint
  overlay and metadata (`score`, `iou`, `box`, mask size, runtime, device,
  model) under the `SAM3/` namespace.

## Models & performance

- 39 published GGUF models: SAM3 (text + detector), SAM3-visual,
  SAM2.1 / SAM2 Hiera backbones; quantizations f16 / f32 / q4_0 / q4_1 / q8_0
  (sam3-f32 is not published — too large).
- The image encoder dominates runtime; GPU backends are several times faster
  than CPU. `sam2.1_hiera_tiny_f16.gguf` is a good default.

## Notes

- Text prompts require a **SAM3** model (`sam3-*`); SAM2 / SAM2.1 / SAM3-visual
  models are visual-only and hide the text row.
- The C-API (`aicore/sam3_capi.h`) also exposes the video tracker
  (track / propagate / add-instance / refine); the tracker is contract-tested
  in `core/AICore/tests/sam3/`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Model combo empty | Check that `AICore_ENABLED=ON` was used when configuring |
| Load fails | Verify the GGUF is one of the published sam models and that the device backend is available (CUDA on NVIDIA, Vulkan on most Linux/Windows GPUs) |
| Text prompt row hidden | The loaded model is visual-only (SAM2 / SAM2.1 / SAM3-visual); load a `sam3-*` model |
