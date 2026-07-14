---
license: apache-2.0
library_name: ggml
pipeline_tag: depth-estimation
tags:
  - depth-anything
  - depth-anything-3
  - depth-anything-2
  - depth-estimation
  - monocular-depth
  - camera-pose
  - gguf
  - ggml
  - cpp
  - localai
base_model:
  - depth-anything/DA3-SMALL
  - depth-anything/DA3-BASE
  - depth-anything/DA3-LARGE
  - depth-anything/DA3-GIANT
  - depth-anything/DA3MONO-LARGE
  - depth-anything/DA3METRIC-LARGE
  - depth-anything/DA3NESTED-GIANT-LARGE
  - depth-anything/Depth-Anything-V2-Small
  - depth-anything/Depth-Anything-V2-Base
  - depth-anything/Depth-Anything-V2-Large
  - depth-anything/Depth-Anything-V2-Metric-Hypersim-Small
  - depth-anything/Depth-Anything-V2-Metric-Hypersim-Base
  - depth-anything/Depth-Anything-V2-Metric-Hypersim-Large
  - depth-anything/Depth-Anything-V2-Metric-VKITTI-Small
  - depth-anything/Depth-Anything-V2-Metric-VKITTI-Base
  - depth-anything/Depth-Anything-V2-Metric-VKITTI-Large
---

# Depth Anything 3 — GGUF weights for [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp)

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team.**

GGUF conversions of [ByteDance Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3),
for use with **[depth-anything.cpp](https://github.com/mudler/depth-anything.cpp)** — a from-scratch
C++17 / [ggml](https://github.com/ggml-org/ggml) port. No Python, no PyTorch, no CUDA toolkit at
inference: one self-contained GGUF file plus a small native library and CLI, **faster than PyTorch
on CPU** and **bit-exact** against the original (correlation 1.0, verified component by component).

Given an image, the engine recovers a dense **depth** map, per-pixel **confidence**, camera
**extrinsics (3×4)** and **intrinsics (3×3)**, an optional **sky** mask, a back-projected **3D point
cloud**, and exports to **glb / COLMAP / PLY**.

## Files in this repo

Each GGUF is fully self-contained — every dimension, hyperparameter and preprocessing constant is
baked into the file; the loader reads them, nothing is hardcoded.

| File | Source checkpoint | Backbone | Depth type | Output |
|------|-------------------|----------|-----------|--------|
| `depth-anything-small-f32.gguf` | `DA3-SMALL` | ViT-S | relative | depth + conf + pose |
| `depth-anything-base-f32.gguf` | `DA3-BASE` | ViT-B | relative | depth + conf + pose |
| `depth-anything-base-f16.gguf` | `DA3-BASE` | ViT-B | relative | depth + conf + pose |
| `depth-anything-base-q8_0.gguf` | `DA3-BASE` | ViT-B | relative | depth + conf + pose (near-lossless) |
| `depth-anything-base-q4_k.gguf` | `DA3-BASE` | ViT-B | relative | depth + conf + pose (**99 MB**) |
| `depth-anything-large-f32.gguf` | `DA3-LARGE` | ViT-L | relative | depth + conf + pose |
| `depth-anything-giant-f32.gguf` | `DA3-GIANT` | ViT-g | relative | depth + conf + pose + 3D Gaussians |
| `depth-anything-mono-large-f32.gguf` | `DA3MONO-LARGE` | ViT-L | relative (monocular) | depth + sky |
| `depth-anything-metric-large-f32.gguf` | `DA3METRIC-LARGE` | ViT-L | **metric** | metric depth + sky |
| `depth-anything-nested-anyview.gguf` | `DA3NESTED-GIANT-LARGE` (anyview branch) | ViT-g | relative | depth + conf + pose |
| `depth-anything-nested-metric.gguf` | `DA3NESTED-GIANT-LARGE` (metric branch) | ViT-L | **metric** | depth + sky |

> The nested model is a **two-file pair**: the engine loads the anyview (ViT-g) branch and the
> metric (ViT-L) branch together and aligns them to produce metric-scale depth + pose. Download
> both `depth-anything-nested-anyview.gguf` and `depth-anything-nested-metric.gguf`.

### Depth Anything V2

The same engine also runs [Depth Anything **V2**](https://github.com/DepthAnything/Depth-Anything-V2)
checkpoints. DA2 is **depth only** — no confidence, pose or sky. **Relative** models output an inverse
depth map through a `ReLU` head; **metric** models output depth in **metres** through a
`Sigmoid × max_depth` head (`max_depth=20` for the indoor Hypersim variants, `max_depth=80` for the
outdoor VKITTI variants). The ViT-g (Giant) DA2 checkpoint is not shipped (its `Depth-Anything-V2-Giant`
HF repo is gated/unreleased).

Each model below ships in f32 plus f16 / q8_0 / q6_k / q5_k / q4_k quants (only the f32 + a representative
quant are listed for brevity; the full set is in `SHA256SUMS`).

| File | Source checkpoint | Backbone | Depth type | Output |
|------|-------------------|----------|-----------|--------|
| `depth-anything2-small-f32.gguf` | `Depth-Anything-V2-Small` | ViT-S | relative | inverse depth |
| `depth-anything2-small-q8_0.gguf` | `Depth-Anything-V2-Small` | ViT-S | relative | inverse depth (near-lossless) |
| `depth-anything2-base-f32.gguf` | `Depth-Anything-V2-Base` | ViT-B | relative | inverse depth |
| `depth-anything2-large-f32.gguf` | `Depth-Anything-V2-Large` | ViT-L | relative | inverse depth |
| `depth-anything2-large-q4_k.gguf` | `Depth-Anything-V2-Large` | ViT-L | relative | inverse depth (smallest) |
| `depth-anything2-metric-hypersim-small-f32.gguf` | `Depth-Anything-V2-Metric-Hypersim-Small` | ViT-S | **metric** (≤20 m, indoor) | depth in metres |
| `depth-anything2-metric-hypersim-base-f32.gguf` | `Depth-Anything-V2-Metric-Hypersim-Base` | ViT-B | **metric** (≤20 m, indoor) | depth in metres |
| `depth-anything2-metric-hypersim-large-f32.gguf` | `Depth-Anything-V2-Metric-Hypersim-Large` | ViT-L | **metric** (≤20 m, indoor) | depth in metres |
| `depth-anything2-metric-vkitti-small-f32.gguf` | `Depth-Anything-V2-Metric-VKITTI-Small` | ViT-S | **metric** (≤80 m, outdoor) | depth in metres |
| `depth-anything2-metric-vkitti-base-f32.gguf` | `Depth-Anything-V2-Metric-VKITTI-Base` | ViT-B | **metric** (≤80 m, outdoor) | depth in metres |
| `depth-anything2-metric-vkitti-large-f32.gguf` | `Depth-Anything-V2-Metric-VKITTI-Large` | ViT-L | **metric** (≤80 m, outdoor) | depth in metres |

**Parity.** Every DA2 GGUF is verified against the upstream `DepthAnythingV2` forward (correlation > 0.999
end-to-end at f32, q8_0 near-lossless at corr 0.99962, q4_k at 0.99944). The one exception is
`depth-anything2-metric-vkitti-small` at corr **0.9983** — this is **not a porting defect** (the C++ route
matches the reference `Sigmoid × 80` math exactly); it is the inherent ≤20× amplification of backbone
fp-rounding noise by the widest metric scale on the smallest backbone. Absolute error stays sub-1%
(mean 0.57% of 80 m), and the same ViT-S backbone scores 0.9996 in relative mode. Accepted as near-lossless.

### Which one should I use?

- **Just trying it out / CPU:** `depth-anything-base-q4_k.gguf` (99 MB, near-lossless).
- **Best quality/speed default:** `depth-anything-base-q8_0.gguf`.
- **Smallest / fastest:** `depth-anything-small-f32.gguf`.
- **Highest quality + 3D reconstruction (point cloud / Gaussians):** `depth-anything-giant-f32.gguf`.
- **Single-image depth with sky mask:** `depth-anything-mono-large-f32.gguf`.
- **Metric-scale depth (meters), single model:** `depth-anything-metric-large-f32.gguf`.
- **Best metric-scale depth + pose:** the nested pair (`depth-anything-nested-anyview.gguf` +
  `depth-anything-nested-metric.gguf`).

## Usage

### depth-anything.cpp (CLI)

```bash
git clone https://github.com/mudler/depth-anything.cpp && cd depth-anything.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# download a weight from this repo
hf download mudler/depth-anything.cpp-gguf depth-anything-base-q4_k.gguf --local-dir models

./build/da3 depth models/depth-anything-base-q4_k.gguf image.jpg --out depth.png
./build/da3 depth models/depth-anything-base-q4_k.gguf image.jpg --pose poses.json
./build/da3 reconstruct models/depth-anything-giant-f32.gguf image.jpg --ply cloud.ply

# metric-scale depth from the single metric model
./build/da3 depth models/depth-anything-metric-large-f32.gguf image.jpg --out depth.png

# metric-scale depth + pose from the nested pair (anyview + metric branches)
./build/da3 depth models/depth-anything-nested-anyview.gguf image.jpg \
    --metric-model models/depth-anything-nested-metric.gguf --pfm depth.pfm
```

See the [README](https://github.com/mudler/depth-anything.cpp) for multi-view, glb/COLMAP export,
quantization and the flat C API.

### LocalAI

```bash
local-ai run depth-anything-3-base
```

## Performance

Faster than PyTorch on CPU at half the memory, bit-exact. AMD Ryzen 9 9950X3D, `threads=16`,
504×336, sustained:

| engine | quant | model MB | load ms | infer ms | peak RAM MB | vs PyTorch |
|--------|-------|---------:|--------:|---------:|------------:|-----------:|
| PyTorch | f32 | 516 | 749 | 416.9 | 1328 | 1.00× |
| **C++/ggml** | f32 | 393 | **112** | **346.4** | **614** | **1.20×** |
| **C++/ggml** | q8_0 | 142 | **40** | **319.4** | **363** | **1.31×** |
| **C++/ggml** | q4_k | **99** | **25** | 395.2 | **320** | 1.05× |

Full methodology in [`benchmarks/BENCHMARK.md`](https://github.com/mudler/depth-anything.cpp/blob/master/benchmarks/BENCHMARK.md).

## License

The GGUF weights are derived from the official Depth Anything 3 checkpoints and inherit their
**Apache-2.0** license. The depth-anything.cpp code is MIT.

## Citation

```bibtex
@article{depthanything3,
  title   = {Depth Anything 3: Recovering the Visual Space from Any Views},
  author  = {ByteDance Seed},
  year    = {2025}
}
```
