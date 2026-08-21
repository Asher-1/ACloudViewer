# qYOLO model card

## Model

| Field        | Value                                                                 |
|--------------|-----------------------------------------------------------------------|
| Architecture | Ultralytics YOLO — YOLOv8 and YOLO26 families (GGUF export)           |
| Task         | Object detection (COCO 80 classes), instance segmentation, metric depth |
| Input        | RGB image letterboxed to the model's image size                       |
| Output       | Detection boxes (class_id / score / box); instance masks (binary per-object); depth: per-pixel depth map in meters |
| License      | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (Ultralytics) |
| Source       | ultralytics-ggml conversion -> `yolo_gguf_models` release (hosted on cloudViewer_downloads) |

## Files

**21 variants x 3 quantizations = 63 GGUF files** in the
[yolo_gguf_models release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/yolo_gguf_models).
Filename pattern: `<variant>-<quant>.gguf` (e.g. `yolov8n-f16.gguf`,
`yolov8n-seg-f16.gguf`, `yolo26n-depth-q8_0.gguf`).

| Variant family | Variants          | Task                | Head                                    | end2end |
|----------------|-------------------|---------------------|-----------------------------------------|---------|
| YOLOv8         | n / s / m / l / x | Detection (COCO 80) | classic + NMS                           | no      |
| YOLO26         | n / s / m / l / x | Detection (COCO 80) | end-to-end (NMS baked into the head)    | yes     |
| YOLOv8-seg     | n / s / m / l / x | Segmentation        | classic + NMS + Proto                   | no      |
| YOLO26-seg     | n / s / m / l / x | Segmentation        | end-to-end + Proto26                    | yes     |
| YOLO26 depth   | yolo26n-depth     | Metric depth        | end-to-end                              | yes     |

## Download

Mirror hosted by ACloudViewer:

`https://github.com/Asher-1/cloudViewer_downloads/releases/download/yolo_gguf_models/<variant>-<quant>.gguf`

The model cache directory is `yolo_models/` (see `aicore_yolo_model_cache_dir`).

## Inference benchmarks

Benchmarks from the upstream [ultralytics-ggml](https://github.com/Asher-1/ultralytics-ggml)
repository (commit `8c356b7a`). Source benchmark artifacts are in
`cpp_ggml/benchmarks/`.

### Detection latency by backend (CPU / CUDA / Vulkan)

![Latency by backend](https://raw.githubusercontent.com/Asher-1/ultralytics-ggml/main/cpp_ggml/benchmarks/latency_by_backend.png)

Median ms/image end-to-end for each model family x dtype x backend.

### Latency matrix (all models, all dtypes)

![Latency matrix](https://raw.githubusercontent.com/Asher-1/ultralytics-ggml/main/cpp_ggml/benchmarks/latency_matrix.png)

### Speed by dtype (F32 vs F16 vs Q8_0)

![Speed by dtype](https://raw.githubusercontent.com/Asher-1/ultralytics-ggml/main/cpp_ggml/benchmarks/speed_by_dtype.png)

### Speed by model (nano through xlarge)

![Speed by model](https://raw.githubusercontent.com/Asher-1/ultralytics-ggml/main/cpp_ggml/benchmarks/speed_by_model.png)

### Segment latency by backend

![Segment latency](https://raw.githubusercontent.com/Asher-1/ultralytics-ggml/main/cpp_ggml/benchmarks/seg_latency.png)

### Depth parity (bus scene, F16 vs reference)

![Depth parity bus](https://raw.githubusercontent.com/Asher-1/ultralytics-ggml/main/cpp_ggml/benchmarks/depth_parity_bus.png)

### Depth latency by backend

![Depth latency](https://raw.githubusercontent.com/Asher-1/ultralytics-ggml/main/cpp_ggml/benchmarks/depth_latency.png)

### Speedup summary

| Model | F16 CPU (ms) | F16 CUDA (ms) | F16 Vulkan (ms) | Speedup (GPU vs CPU) |
|-------|-------------|--------------|----------------|---------------------|
| yolov8n | ~25 | ~2 | ~3 | 8-12x |
| yolov8s | ~65 | ~4 | ~6 | 10-16x |
| yolov8m | ~140 | ~8 | ~12 | 12-18x |
| yolov8l | ~240 | ~14 | ~20 | 12-17x |
| yolov8x | ~390 | ~22 | ~32 | 12-18x |

Full benchmark matrix available in:
`https://github.com/Asher-1/ultralytics-ggml/tree/main/cpp_ggml/benchmarks/`

## Quantization

Relative to the f16 size, per-parameter byte width:

| Quant | Size vs f16 | Notes                            |
|-------|-------------|----------------------------------|
| f32   | ~2x         | float32 reference                |
| f16   | 1x          | half precision (**recommended**) |
| q8_0  | ~0.5x       | 8-bit quant, smallest download   |

## Backends

- CUDA / Vulkan / Metal / CPU -- all ggml backends are supported through
  AICore's unified device resolution (`auto` picks the best available GPU).
- Thread count is configurable (0 = auto).