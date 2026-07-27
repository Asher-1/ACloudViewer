# ELoFTR GGML models

Official EfficientLoFTR pretrained weights: **outdoor only** (`eloftr_outdoor.ckpt`, MegaDepth).

| File | Quantization | Notes |
|------|--------------|-------|
| [`eloftr_outdoor-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/ELoFTR/eloftr_outdoor-f32.gguf) | F32 | reference |
| [`eloftr_outdoor-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/ELoFTR/eloftr_outdoor-f16.gguf) | F16 | **plugin default** |
| [`eloftr_outdoor-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/ELoFTR/eloftr_outdoor-q8_0.gguf) | Q8_0 | experimental |

Cache: `~/cloudViewer_data/extract/eloftr_models/`

## Indoor model

There is **no** public `eloftr_indoor*.gguf`. Upstream EfficientLoFTR does not release an indoor checkpoint ([GitHub issue #35](https://github.com/zju3dv/EfficientLoFTR/issues/35)); `indoor_ds*.ckpt` in LoFTR dataset folders is the **original LoFTR** architecture and cannot be converted with `convert_eloftr_to_gguf.py`. qLightGlue lists **Outdoor** variants only.

After training your own indoor checkpoint, export with the same script using `--output models/eloftr_indoor-f32.gguf` and add a plugin download entry.

## Export (maintainers)

1. PyTorch → F32: `EfficientLoFTR/scripts/convert_eloftr_to_gguf.py`
2. F32 → F16/Q8: `aicore_eloftr_quantize` or `core/AICore/scripts/export_gguf_variants.py`
3. Parity: `EfficientLoFTR/cpp/BENCHMARK.md` and `assets/ggml_validation_20260727/`

## Backend parity

RepVGG backbone + coarse matcher run on CPU/CUDA/Vulkan with shared ggml graph; coarse correlation uses CPU post-process on all backends today. Match scores should agree within float tolerance across backends when using the same quantization.

Validation figures: [EfficientLoFTR README](https://github.com/zju3dv/EfficientLoFTR#ggml-validation-figures).
