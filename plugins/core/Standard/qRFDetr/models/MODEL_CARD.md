# qRFDetr Model Card

## Model

| Field        | Value                                                                  |
|--------------|------------------------------------------------------------------------|
| Architecture | RF-DETR (real-time DETR with deformable attention, Roboflow)           |
| Task         | Open-vocabulary object detection (COCO 80 classes) + optional instance segmentation |
| Input        | RGB image letterboxed to 640×640 (configurable at export)              |
| Output       | Detections JSON (class_id / class_name / score / box) + per-detection PNG masks (seg variants) |
| License      | Models: [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) (see [rf-detr.cpp](https://github.com/mudler/rf-detr.cpp) for conversion details) |
| Source       | [rf-detr.cpp](https://github.com/mudler/rf-detr.cpp) → `RF-DETR-GGUF` release |

## Files

| Filename                    | Size (approx.) | Notes                                  |
|-----------------------------|----------------|----------------------------------------|
| `rfdetr-nano-f16.gguf`      | ~65 MB         | Detection, fastest                     |
| `rfdetr-small-f16.gguf`     | ~115 MB        | Detection                              |
| `rfdetr-base-f16.gguf`      | ~190 MB        | Detection (default)                    |
| `rfdetr-medium-f16.gguf`    | ~300 MB        | Detection                              |
| `rfdetr-large-f16.gguf`     | ~475 MB        | Detection, highest accuracy            |
| `rfdetr-seg-nano-f16.gguf`  | ~85 MB         | Segmentation                           |
| `rfdetr-seg-small-f16.gguf` | ~135 MB        | Segmentation                           |
| `rfdetr-seg-medium-f16.gguf`| ~330 MB        | Segmentation                           |

## Download

Mirror hosted by ACloudViewer:

`https://github.com/Asher-1/cloudViewer_downloads/releases/download/RF-DETR-GGUF/rfdetr-<variant>-f16.gguf`

The model cache directory is `rfdetr_models/` (see `aicore_rfdetr_model_cache_dir`).

## Backends

- CUDA / Vulkan / Metal / CPU — all ggml backends are supported through
  AICore's unified device resolution (`auto` picks the best available GPU).
- Thread count is configurable (0 = auto).
