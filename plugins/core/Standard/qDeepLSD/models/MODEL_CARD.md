# DeepLSD GGUF models

Two official checkpoints from [DeepLSD](https://github.com/cvg/DeepLSD) — same VGGUNet architecture, different training data:

| Variant | Checkpoint | Use case |
|---------|------------|----------|
| **Wireframe** | `deeplsd_wireframe.tar` | Indoor / wireframe scenes |
| **MegaDepth (md)** | `deeplsd_md.tar` | Outdoor / general phototourism |

Each variant ships as F32 / F16 / Q8_0 GGUF (quantized via `aicore_gguf_quantize deeplsd`).

## Wireframe

| File | Quant | Notes |
|------|-------|-------|
| [`deeplsd_wireframe-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DeepLSD/deeplsd_wireframe-f32.gguf) | F32 | reference |
| [`deeplsd_wireframe-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DeepLSD/deeplsd_wireframe-f16.gguf) | F16 | **plugin default** |
| [`deeplsd_wireframe-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DeepLSD/deeplsd_wireframe-q8_0.gguf) | Q8_0 | smaller; experimental |

## MegaDepth (md)

| File | Quant | Notes |
|------|-------|-------|
| [`deeplsd_md-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DeepLSD/deeplsd_md-f32.gguf) | F32 | reference |
| [`deeplsd_md-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DeepLSD/deeplsd_md-f16.gguf) | F16 | recommended for outdoor |
| [`deeplsd_md-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DeepLSD/deeplsd_md-q8_0.gguf) | Q8_0 | experimental |

Convert locally: `python scripts/convert_deeplsd_to_gguf.py --checkpoint weights/deeplsd_md.tar --output models/deeplsd_md-f32.gguf`
