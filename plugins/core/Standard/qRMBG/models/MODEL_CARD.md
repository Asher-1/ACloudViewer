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

| Filename            | Size (approx.) | Format | Notes                          |
|---------------------|----------------|--------|--------------------------------|
| `rmbg_f16.gguf`     | ~650 MB        | GGUF   | Unified single file: encoder + decoder, F16 |

## Download

Mirror hosted by ACloudViewer:

`https://github.com/Asher-1/cloudViewer_downloads/releases/download/trellis2-ggml/rmbg_f16.gguf`

The model cache directory is `rmbg_models/` (see `aicore_rmbg_model_cache_dir`).

## Backends

- CUDA (custom `rmbg-deform-im2col` / `rmbg-conv2d` / `rmbg-swin` kernels)
- Vulkan (7 custom shaders)
- CPU / Metal (vanilla ggml operator fallback)

Device selection follows AICore's `auto` order (CUDA → Vulkan → CPU on
Linux/Windows, Metal → CPU on macOS) and can be overridden in the dialog.
