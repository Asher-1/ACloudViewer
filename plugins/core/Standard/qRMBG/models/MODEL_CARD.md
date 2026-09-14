# qRMBG Model Card

## Model

| Field       | Value                                                                 |
|-------------|-----------------------------------------------------------------------|
| Architecture | BiRefNet-Swin-L (bilateral reference network, Swin-L encoder)        |
| Task        | Background removal (binary alpha matte)                               |
| Input       | RGB image resized to 1024×1024 (aspect-ratio padded)                  |
| Output      | RGBA composite at the **original** image resolution (transparent bg)  |
| License     | BiRefNet: `Apache-2.0`; RMBG-2.0 model weights: `briaai/RMBG-2.0` (see [HF repo](https://huggingface.co/briaai/RMBG-2.0)) |
| Source      | [RMBG-2.0-GGML](https://github.com/Asher-1/RMBG-2.0-GGML) → `trellis2-ggml` release |

## Files

All three quantization variants are published in the `trellis2-ggml` release
(sizes measured from the release assets):

| Filename            | Size (approx.) | Format | Notes                          |
|---------------------|----------------|--------|--------------------------------|
| `rmbg_f32.gguf`     | ~840 MB        | GGUF   | Unified single file: encoder + decoder, F32 reference |
| `rmbg_f16.gguf`     | ~420 MB        | GGUF   | Unified single file: encoder + decoder, F16 (recommended) |
| `rmbg_q8.gguf`      | ~245 MB        | GGUF   | Unified single file: encoder + decoder, 8-bit quant |

## Download

Mirror hosted by ACloudViewer (`trellis2-ggml` release):

`https://github.com/Asher-1/cloudViewer_downloads/releases/download/trellis2-ggml/rmbg_<f32|f16|q8>.gguf`

> Note: `rmbg_q8_0.gguf` / `rmbg_q4_K.gguf` are **not** published upstream; the
> model combo only lists entries that exist on the release.

The model cache directory is `rmbg_models/` (see `aicore_rmbg_model_cache_dir`).

## Backends

- CUDA (custom `rmbg-deform-im2col` / `rmbg-conv2d` / `rmbg-swin` kernels)
- Vulkan (7 custom shaders)
- CPU / Metal (vanilla ggml operator fallback)

Device selection follows AICore's `auto` order (CUDA → Vulkan → CPU on
Linux/Windows, Metal → CPU on macOS) and can be overridden in the dialog.
