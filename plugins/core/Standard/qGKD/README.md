# qGKD — General Keypoint Detection (GKDT)

General (open-world) keypoint detection on still and DB images with the
**GKDT-L** transformer (ECCV 2026), running through the in-tree ggml runtime
in `core/AICore/src/tasks/gkd/` — no Python at runtime.

![qGKD](images/qGKD.svg)

## Prompt modes

| Mode | Inputs | Result |
|---|---|---|
| **Text** | keypoint texts ("nose", "left eye", …) | one keypoint per text |
| **Visual (1-shot)** | support image + support keypoints (pixel coords) | one keypoint per support keypoint |
| **Multimodal** | keypoint texts **and** support image + keypoints (counts must match) | text row *i* fuses with visual row *i* (official semantics) |
| **Multi-object** | object classes + keypoint texts (+ optional support) | YOLO-World boxes (existing `aicore_yolo_*` WORLD models) → GKD per box |

Optional ROI bbox (`x1 y1 x2 y2`, empty = whole image) restricts the query
region; keypoints are rendered with a per-keypoint score threshold.

## Model management

The model combo lists the published catalog from AICore
(`aicore_gkd_model_*`; default = **GKDT-L (Q4_K), 483 MiB**; the deprecated
upstream q4_0 build is not cataloged). Missing models are downloaded
automatically from
[Hugging Face Asher-1/GKD_GGUF](https://huggingface.co/Asher-1/GKD_GGUF) and
SHA-256 verified at ingestion (digests pinned in
`core/AICore/include/aicore/asset_digests.h`). Cache:
`~/cloudViewer_data/extract/gkd_models/`. Custom GGUF paths are also accepted.

## Sample data

The **Try sample data** button (shared `ecvTestDataRepository` flow) downloads
the official GKDT demo images
([general_keypoint_detection_data](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/general_keypoint_detection_data),
SHA-256 pinned, cached under `~/cloudViewer_data/download/`), extracts and
fills the sample combo. Selecting a sample auto-fills the query image and,
for the official demos, the exact prompt configuration
(`2007_007524.jpg` → the 5 face-prompt text demo; `2007_003778.jpg` → the
1-shot visual demo prompts). A custom image can always be picked with Browse.

Multi-object mode reuses the YOLO-World detector models from the existing
yolo task catalog — no second model table.

## Outputs

- **DB**: rendered image with keypoints (prompt labels + scores) and boxes;
  metadata records mode, keypoint/object counts, per-stage timings, device,
  and model (`GKD/*` keys, name prefix `GKD_`).
- **PNG export** to a user directory.

## Build

```bash
cmake -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QGKD=ON ..
cmake --build . --target QGKD_PLUGIN -j4
```

Requires `AICore_ENABLED=ON` (the plugin is skipped with a warning
otherwise).

## Validation

```bash
# One-click AICore gate (light tier downloads + runs gkd_fullset-q4_K.gguf)
cmake --build build_app --target aicore-validate-all -j1
# Complete quantization matrix (all five GGUF files)
python3 core/AICore/scripts/validate_all.py --build build_app --backend cuda --full
```

## License

GKDT weights and source are for **academic research and educational use
only** (ECCV 2026; commercial use prohibited) — see
[models/MODEL_CARD.md](models/MODEL_CARD.md).
