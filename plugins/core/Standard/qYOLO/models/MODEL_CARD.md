# qYOLO model card

## Model

| Field        | Value                                                                 |
|--------------|-----------------------------------------------------------------------|
| Architecture | Ultralytics YOLO — YOLOv8 and YOLO26 families (GGUF export)           |
| Task         | Object detection (COCO 80 classes), instance segmentation, metric depth, keypoint pose (COCO-17), oriented boxes (DOTA-15), classification (ImageNet-1000), semantic segmentation (Cityscapes-19), open-vocabulary detection/segmentation (YOLO-World / YOLOE, text towers included) |
| Input        | RGB image letterboxed to the model's image size (classify: checkpoint-baked resize + center crop) |
| Output       | Detection boxes (class_id / score / box); instance masks (binary per-object); depth: per-pixel depth map in meters; pose: boxes + 17 keypoints (x/y/visibility); obb: rotated boxes (cx/cy/w/h/angle); classify: softmax table; semantic: full-resolution class map; world/yoloe: detections (and masks) against the user's class list; yoloe visual prompts (SAVPE): detections/masks against image-derived class embeddings (object0..objectN-1) |
| License      | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (Ultralytics) |
| Source       | ultralytics-ggml conversion -> `yolo_gguf_models` release (hosted on cloudViewer_downloads) |

## Files

**72 variants x 3 quantizations = 215 GGUF files** in the
[yolo_gguf_models release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/yolo_gguf_models)
(plus the 15-file `reid-yolo26{...}` appearance-encoder family below → 230
assets total).
Filename pattern: `<variant>-<quant>.gguf` (e.g. `yolov8n-f16.gguf`,
`yolov8n-seg-f16.gguf`, `yolo26n-depth-q8_0.gguf`,
`yoloe-26n-seg-pf-q8_0.gguf`). The obb and semantic families additionally
ship checkpoint-native **1024-resolution** rebuilds with the size inserted
in the stem (e.g. `yolo26n-obb-1024-q8_0.gguf`,
`yolo26x-sem-1024-f16.gguf` — 30 files); the 1024 GGUFs declare their
input size in the `yolo.imgsz` metadata, so the engine builds the larger
canvas automatically.

YOLOE visual prompts: the non-`-pf` `yoloe-*-seg` GGUFs additionally support
the official SAVPE visual-prompt mode when they carry savpe weights
(`yolo.savpe = 1` + `savpe.*` tensors, added by
`core/AICore/src/tasks/yolo/tools/convert_yoloe_savpe_gguf.py` from the matching `.pt` checkpoint);
GGUFs without the flag keep working through the text / prompt-free paths.

ReID appearance encoders (authoritative family, 2026-09-19):
`reid-yolo26{n,s,m,l,x}-{f32,f16,q8_0}.gguf` (15 files, 3.2–129 MB) —
converted directly from the official
`yolo26{n,s,m,l,x}-reid.onnx` assets (standalone ReID backbones with a
512-d embedding head, GAP+GMP→Gemm→L2norm); task='reid' graphs consumed
through `aicore/reid_capi.h` as the explicit `model: <path>` appearance
encoder. Same release and `yolo_models/` cache folder, plus the
[Hugging Face `Asher-1/yolo-gguf`](https://huggingface.co/Asher-1/yolo-gguf)
mirror; digests pinned in `aicore/asset_digests.h`, exercised by the
`reid-native-models` validate-all rows. Upstream truth: the converter
verifies every rebuilt network against its onnx graph at cosine 1.000000
before writing, and the AICore CUDA runtime matches the official onnx
output at cos=0.9999 on identical CHW input. (The earlier cls-tower
encoder family `reid-yolo26*-cls-*` was withdrawn — its weights were not
the official ReID models.) Conversion:
`cpp_ggml/scripts/convert_reid_onnx_to_gguf.py` in ultralytics-ggml.

| Variant family | Variants          | Task                | Head                                    | end2end |
|----------------|-------------------|---------------------|-----------------------------------------|---------|
| YOLOv8         | n / s / m / l / x | Detection (COCO 80) | classic + NMS                           | no      |
| YOLO26         | n / s / m / l / x | Detection (COCO 80) | end-to-end (NMS baked into the head)    | yes     |
| YOLOv8-seg     | n / s / m / l / x | Segmentation        | classic + NMS + Proto                   | no      |
| YOLO26-seg     | n / s / m / l / x | Segmentation        | end-to-end + Proto26                    | yes     |
| YOLO26 depth   | n / s / m / l / x | Metric depth        | end-to-end (768 input)                  | yes     |
| YOLO26 pose    | n / s / m / l / x | Keypoint pose       | Pose26 (RLE head, COCO-17)              | yes     |
| YOLO26 obb     | n / s / m / l / x | Oriented boxes      | OBB26 (DOTA-15, raw radians)            | yes     |
| YOLO26 obb-1024 | n / s / m / l / x | Oriented boxes    | OBB26 (DOTA-15, 1024 checkpoint-native input) | yes |
| YOLO26 sem     | n / s / m / l / x | Semantic seg        | Cityscapes-19 head (canvas/8 grid)      | yes     |
| YOLO26 sem-1024 | n / s / m / l / x | Semantic seg       | Cityscapes-19 head (1024 checkpoint-native input) | yes |
| YOLO26 cls     | n / s / m / l / x | Classification      | ImageNet-1000 linear head (224 input)   | yes     |
| YOLOv8-world   | s / m / l / x     | Open-vocab detect   | CLIP text-conditioned head (v3 graph)   | no      |
| YOLOE-26-seg   | n / s / m / l / x (+ `-pf`) | Open-vocab segment | MobileCLIP text tower + reprta (v4 graph) | yes |
| Text towers    | clip-ViT-B-32, mobileclip2_b | Text encoder | BPE + causal transformer (512-d L2-normalised embeddings) | — |

The `-pf` (prompt-free) YOLOE variants derive the vocabulary from image
features at runtime; the non-pf variants accept a plaintext class list
encoded through the MobileCLIP tower. The text towers are selectable in the
world / yoloe tabs' "Text model" combo (CLIP for World, MobileCLIP for
YOLOE).

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

## Detection Classes — COCO 80

All detection and segmentation variants are pretrained on the **COCO dataset**
with **80 object classes**, identical to the official Ultralytics releases.
The class names are embedded in each GGUF under `yolo.class_names` and read at
runtime (model metadata default: `yolo.nc` = 80).

| ID | Class | ID | Class | ID | Class | ID | Class |
|----|-------|----|-------|----|-------|----|-------|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
| 10 | fire hydrant | 30 | skis | 50 | broccoli | 70 | toaster |
| 11 | stop sign | 31 | snowboard | 51 | carrot | 71 | sink |
| 12 | parking meter | 32 | sports ball | 52 | hot dog | 72 | refrigerator |
| 13 | bench | 33 | kite | 53 | pizza | 73 | book |
| 14 | bird | 34 | baseball bat | 54 | donut | 74 | clock |
| 15 | cat | 35 | baseball glove | 55 | cake | 75 | vase |
| 16 | dog | 36 | skateboard | 56 | chair | 76 | scissors |
| 17 | horse | 37 | surfboard | 57 | couch | 77 | teddy bear |
| 18 | sheep | 38 | tennis racket | 58 | potted plant | 78 | hair drier |
| 19 | cow | 39 | bottle | 59 | bed | 79 | toothbrush |