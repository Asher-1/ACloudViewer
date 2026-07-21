# FreeSplatter GGUF Models

## Available Models

### Scene Models (Scene Reconstruction)
- `freesplatter-scene-f16.gguf` — Half precision, recommended balance of speed/quality
- `freesplatter-scene-f32.gguf` — Full precision, best quality
- `freesplatter-scene-q8_0.gguf` — 8-bit quantized, smaller footprint

### Object Models (Object Reconstruction)
- `freesplatter-object-f16.gguf` — Half precision
- `freesplatter-object-f32.gguf` — Full precision
- `freesplatter-object-q8_0.gguf` — 8-bit quantized

## Download

Models are auto-downloaded from:
```
https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/
```

## Model Architecture

All models share the same transformer backbone:
- Patch tokenizer: 8x8 conv, 1024 output channels
- 24-layer self-attention transformer (1024 embd, 16 heads, 64 head_dim)
- Gaussian head: unpatchify to 23 channels per pixel

### Gaussian Channel Layout (23ch, scene mode)
| Channel | Field |
|---------|-------|
| 0-2     | xyz position |
| 3-14    | SH coefficients (degree 1, 4x3) |
| 15      | opacity (sigmoid-activated) |
| 16-18   | scale (activated) |
| 19-22   | rotation quaternion (w,x,y,z, normalized) |

## Hyperparameters
- Image size: 512x512
- Patch size: 8x8
- SH degree: 1
- Scale activation: sigmoid in [scale_min_act, scale_max_act]
