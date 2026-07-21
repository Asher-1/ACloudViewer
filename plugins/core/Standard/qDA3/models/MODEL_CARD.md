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

## Download (CloudViewer)

CloudViewer hosts pre-built GGUF weights on [cloudViewer_downloads](https://github.com/Asher-1/cloudViewer_downloads) release [**DA3**](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/DA3).

Base URL:

```
https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/
```

The **qDA3** plugin dialog and automatic reconstruction pipeline use the same URLs. Each row below lists the **exact release asset size** (2026-07-11). CPU benchmark numbers for **Base** variants are from [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp) on AMD Ryzen 9 9950X3D, 16 threads, **504×336** input; Large/Giant/Nested rows use relative ratings where no published numbers exist.

### Base（ViT-B）— 日常深度 / 位姿

| 下载 | 量化 | 体积 | 输出 | CPU 推理* | CPU 加载* | 峰值内存* | 质量 | 推荐场景 |
|------|------|------|------|-----------|-----------|-----------|------|----------|
| [`depth-anything-base-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q4_k.gguf) | Q4_K | **99.1 MB** | depth + conf + pose | ~395 ms | ~25 ms | ~320 MB | 近无损量化 | 最小体积、快速试用、带宽受限 |
| [`depth-anything-base-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q8_0.gguf) | Q8_0 | **141.8 MB** | depth + conf + pose | **~319 ms** | ~40 ms | ~363 MB | 近无损 | **默认推荐**（质量/速度/体积均衡） |
| [`depth-anything-base-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-f16.gguf) | F16 | **222.0 MB** | depth + conf + pose | ~350 ms（估） | ~70 ms（估） | ~450 MB（估） | 高 | 需比 Q8 更稳的半精度 |
| [`depth-anything-base-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-f32.gguf) | F32 | **393.0 MB** | depth + conf + pose | ~346 ms | ~112 ms | ~614 MB | 参考精度 | 对标 / 量化源模型 |

### Large（ViT-L）— 更高深度质量

| 下载 | 量化 | 体积 | 输出 | 相对速度 | 相对内存 | 质量 | 推荐场景 |
|------|------|------|------|----------|----------|------|----------|
| [`depth-anything-large-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-large-q4_k.gguf) | Q4_K | **300.2 MB** | depth + conf + pose | 中等 | 中等 | 优于 Base | 大模型 + 小体积 |
| [`depth-anything-large-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-large-q8_0.gguf) | Q8_0 | **448.2 MB** | depth + conf + pose | 较慢 | 较高 | 很好 | 高质量单图/多视图深度 |

### Giant（ViT-g）— 最高质量 + 3D Gaussians

| 下载 | 量化 | 体积 | 输出 | 相对速度 | 相对内存 | 质量 | 推荐场景 |
|------|------|------|------|----------|----------|------|----------|
| [`depth-anything-giant-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-giant-q4_k.gguf) | Q4_K | **904.6 MB** | depth + conf + pose + **3D Gaussians** | 慢 | 高 | 最佳（量化） | 3D 重建；建议 GPU |
| [`depth-anything-giant-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-giant-q8_0.gguf) | Q8_0 | **1.42 GB** | depth + conf + pose + **3D Gaussians** | 很慢 | 很高 | 最佳 | **3D Reconstruct** 模式首选 |

### Nested（metric depth + pose）— 自动重建 metric 流水线

需 **同时**下载 anyview 分支 + metric 分支；anyview 提供相对深度与位姿，metric 分支对齐到米制尺度。

| 下载 | 分支 | 量化 | 体积 | 输出 | 相对速度 | 相对内存 | 说明 |
|------|------|------|------|------|----------|----------|------|
| [`depth-anything-nested-anyview-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-anyview-q4_k.gguf) | AnyView (ViT-g) | Q4_K | **904.6 MB** | depth + conf + pose | 慢 | 高 | 与 metric 配对；体积较小 |
| [`depth-anything-nested-anyview-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-anyview-q8_0.gguf) | AnyView (ViT-g) | Q8_0 | **1.42 GB** | depth + conf + pose | 很慢 | 很高 | **Automatic Reconstruction** 默认 anyview |
| [`depth-anything-nested-metric.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-metric.gguf) | Metric (ViT-L) | F32 | **1.24 GB** | metric depth + sky | 慢 | 高 | 必须与 anyview 变体一起使用 |

> \* **Base CPU benchmark**（504×336，16 线程）：PyTorch f32 基线为 infer ~417 ms / RAM ~1328 MB；C++ q8_0 为 **1.31×** 更快、RAM 约 **363 MB**。更高分辨率或 Giant/Nested 模型请优先选 **CUDA / Vulkan / Metal**。
>
> **对比 PyTorch（Base q8_0）：** 模型 142 MB vs 516 MB；加载 40 ms vs 749 ms；推理 319 ms vs 417 ms；峰值 RAM 363 MB vs 1328 MB。

### 选型速查

| 用途 | 推荐模型（点击下载） |
|------|---------------------|
| 快速试用 / CPU | [`depth-anything-base-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q4_k.gguf) |
| 日常默认 | [`depth-anything-base-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q8_0.gguf) |
| 更高深度质量 | [`depth-anything-large-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-large-q8_0.gguf) |
| 3D 点云 / Gaussians | [`depth-anything-giant-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-giant-q8_0.gguf) |
| 自动重建 metric depth + pose | [`depth-anything-nested-anyview-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-anyview-q8_0.gguf) + [`depth-anything-nested-metric.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-metric.gguf) |

手动下载示例：

```bash
curl -L -O https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q8_0.gguf
```

Additional variants (Small, Mono, standalone Metric, Depth Anything V2, etc.) are listed below but are **not** currently published on `cloudViewer_downloads`; obtain them from [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf) or convert your own.

## Files in this repo

Each GGUF is fully self-contained — every dimension, hyperparameter and preprocessing constant is
baked into the file; the loader reads them, nothing is hardcoded.

**CloudViewer 已发布模型**（含体积、性能与下载链接）见上文 [Download (CloudViewer)](#download-cloudviewer) 分表。

| File | Source checkpoint | Backbone | Depth type | Output | CloudViewer 下载 |
|------|-------------------|----------|-----------|--------|------------------|
| `depth-anything-small-f32.gguf` | `DA3-SMALL` | ViT-S | relative | depth + conf + pose | — |
| [`depth-anything-base-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-f32.gguf) | `DA3-BASE` | ViT-B | relative | depth + conf + pose | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-f32.gguf) |
| [`depth-anything-base-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-f16.gguf) | `DA3-BASE` | ViT-B | relative | depth + conf + pose | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-f16.gguf) |
| [`depth-anything-base-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q8_0.gguf) | `DA3-BASE` | ViT-B | relative | depth + conf + pose (near-lossless) | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q8_0.gguf) |
| [`depth-anything-base-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q4_k.gguf) | `DA3-BASE` | ViT-B | relative | depth + conf + pose | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-base-q4_k.gguf) |
| [`depth-anything-large-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-large-q8_0.gguf) | `DA3-LARGE` | ViT-L | relative | depth + conf + pose | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-large-q8_0.gguf) |
| [`depth-anything-large-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-large-q4_k.gguf) | `DA3-LARGE` | ViT-L | relative | depth + conf + pose | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-large-q4_k.gguf) |
| [`depth-anything-giant-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-giant-q8_0.gguf) | `DA3-GIANT` | ViT-g | relative | depth + conf + pose + 3D Gaussians | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-giant-q8_0.gguf) |
| [`depth-anything-giant-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-giant-q4_k.gguf) | `DA3-GIANT` | ViT-g | relative | depth + conf + pose + 3D Gaussians | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-giant-q4_k.gguf) |
| `depth-anything-mono-large-f32.gguf` | `DA3MONO-LARGE` | ViT-L | relative (monocular) | depth + sky | — |
| `depth-anything-metric-large-f32.gguf` | `DA3METRIC-LARGE` | ViT-L | **metric** | metric depth + sky | — |
| [`depth-anything-nested-anyview-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-anyview-q8_0.gguf) | `DA3NESTED-GIANT-LARGE` (anyview) | ViT-g | relative | depth + conf + pose | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-anyview-q8_0.gguf) |
| [`depth-anything-nested-anyview-q4_k.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-anyview-q4_k.gguf) | `DA3NESTED-GIANT-LARGE` (anyview) | ViT-g | relative | depth + conf + pose | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-anyview-q4_k.gguf) |
| [`depth-anything-nested-metric.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-metric.gguf) | `DA3NESTED-GIANT-LARGE` (metric) | ViT-L | **metric** | depth + sky | [下载](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DA3/depth-anything-nested-metric.gguf) |

> The nested model is a **two-file pair**: the engine loads the anyview (ViT-g) branch and the
> metric (ViT-L) branch together and aligns them to produce metric-scale depth + pose. Download
> both a `depth-anything-nested-anyview-*.gguf` variant and `depth-anything-nested-metric.gguf`.

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

See the [选型速查](#选型速查) table in **Download (CloudViewer)** above for hosted models with sizes and benchmarks.

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
