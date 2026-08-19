# qYOLO model card

## Model

| Field        | Value                                                                 |
|--------------|-----------------------------------------------------------------------|
| Architecture | Ultralytics YOLO — YOLOv8 and YOLO26 families (GGUF export)           |
| Task         | Object detection (COCO 80 classes); metric depth (yolo26n-depth)      |
| Input        | RGB image letterboxed to the model's image size                       |
| Output       | Detections JSON (class_id / class_name / score / box); depth: per-pixel depth map in meters at model resolution |
| License      | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (Ultralytics) |
| Source       | ultralytics-ggml conversion → `yolo_gguf_models` release (hosted on cloudViewer_downloads) |

## Files

**11 variants × 3 quantizations = 33 GGUF files** in the
[yolo_gguf_models release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/yolo_gguf_models).
Filename pattern: `<variant>-<quant>.gguf` (e.g. `yolov8n-f16.gguf`,
`yolo26n-depth-q8_0.gguf`).

| Variant family | Variants          | Task                | Head                                    | end2end |
|----------------|-------------------|---------------------|-----------------------------------------|---------|
| YOLOv8         | n / s / m / l / x | Detection (COCO 80) | classic + NMS                           | no      |
| YOLO26         | n / s / m / l / x | Detection (COCO 80) | end-to-end (NMS baked into the head)    | yes     |
| YOLO26 depth   | yolo26n-depth     | Metric depth        | end-to-end                              | yes     |

The task is a property of the model: detection models answer with boxes
(confidence / IoU / top-K post-processing applies), while the depth model
returns a per-pixel depth map in meters at the model resolution —
confidence / IoU do not apply and their UI controls are disabled.

Quantizations (relative to the f16 size, per-parameter byte width):

| Quant | Size vs f16 | Notes                            |
|-------|-------------|----------------------------------|
| f32   | ~2×         | float32 reference                |
| f16   | 1×          | half precision (**recommended**) |
| q8_0  | ~0.5×       | 8-bit quant, smallest download   |

## Download

Mirror hosted by ACloudViewer:

`https://github.com/Asher-1/cloudViewer_downloads/releases/download/yolo_gguf_models/<variant>-<quant>.gguf`

The model cache directory is `yolo_models/` (see `aicore_yolo_model_cache_dir`).

## Inference benchmarks

Not yet measured for this GGML port. The matrix (median ms/image end-to-end,
per variant and quant, CPU and GPU backends — same harness as the
[RF-DETR card](../../qRFDetr/models/MODEL_CARD.md)) will be appended here
once a benchmark run lands.

## Backends

- CUDA / Vulkan / Metal / CPU — all ggml backends are supported through
  AICore's unified device resolution (`auto` picks the best available GPU).
- Thread count is configurable (0 = auto).
