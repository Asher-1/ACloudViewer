---
license: apache-2.0
base_model: TencentARC/FreeSplatter
pipeline_tag: image-to-3d
library_name: gguf
tags:
  - gguf
  - ggml
  - gaussian-splatting
  - 3d-reconstruction
  - free-splatter.cpp
  - pose-free
---

# free-splatter.cpp — GGUF weights

GGUF conversions of [TencentARC/FreeSplatter](https://github.com/TencentARC/FreeSplatter)
for **[free-splatter.cpp](https://github.com/localai-org/free-splatter.cpp)**, a small
C/C++ ([ggml](https://github.com/ggml-org/ggml)) inference engine that runs the
FreeSplatter neural-network front-half on CPU or a Vulkan GPU — **no camera poses,
no Python**.

Given N uncalibrated images it returns, **for every input pixel**, a 3D Gaussian
(position · spherical-harmonic colour · opacity · scale · rotation) that a
Gaussian-splatting viewer can render. The rasterizer and PnP pose solver are out
of scope — these weights cover the patch tokenizer → multi-view transformer →
per-pixel Gaussian head.

## Files

| file | variant | precision | size | notes |
|---|---|---|---|---|
| `freesplatter-scene-f16.gguf` | scene | f16 | ~596 MB | **recommended**; 2 overlapping views of a scene |
| `freesplatter-scene-f32.gguf` | scene | f32 | ~1.2 GB | full precision (reference / CPU) |
| `freesplatter-object-f16.gguf` | object | f16 | ~596 MB | 3–4 views around one object |

All variants share the same transformer backbone; they differ only in the head
config (`gaussian_channels`, `sh_residual`, `use_2dgs`) and what they were trained
on. The GGUF carries those as `free-splatter.*` KV metadata, so the engine
configures itself from the file.

## Usage

```sh
# build the engine (see the repo README), then:
free_splatter-cli --device vulkan --splat scene.splat \
  freesplatter-scene-f16.gguf view1.jpg view2.jpg
```

Each view is center-cropped and resized to 512×512. The output `.splat` (antimatter15
format) opens in the bundled WebGL viewer or any Gaussian-splatting viewer; pass
`--out result.f32` instead for the raw `[N, 512, 512, gaussian_channels]` float32
Gaussian tensor. There is also a drop-photos-in-the-browser web app and a
multi-scene demo-video renderer — see the project repo.

## How these were made

Converted with [`scripts/convert.py`](https://github.com/localai-org/free-splatter.cpp/blob/master/scripts/convert.py)
directly from the upstream FreeSplatter `.safetensors` checkpoints (needs only
`torch` + `safetensors` + `gguf`). The converter maps the `transformer.*` weights
of pieces 1–3 and fails loudly on any unmapped or missing tensor — no weight is
silently dropped.

## License & attribution

Apache-2.0. These are derivative weights of
[TencentARC/FreeSplatter](https://github.com/TencentARC/FreeSplatter) (Apache-2.0);
all credit for the model goes to the FreeSplatter authors. The GGUF packaging and
the inference engine are part of free-splatter.cpp.
