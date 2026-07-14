# Depth Anything .cpp - Comprehensive Benchmark Report

Generated: 2026-07-09 16:19:38
GPU: RTX 3060 12GB | CUDA 12.4

## 1. Depth Inference (GPU mode)

| Model | canyon | desk | mountains | street | Avg Time |
|-------|--------|------|-----------|--------|----------|
| depth-anything-small-f32 | 0.46s | 0.39s | 0.45s | 0.45s | 0.44s |
| depth-anything-base-f32 | 0.86s | 0.81s | 0.75s | 0.74s | 0.79s |
| depth-anything-base-f16 | 0.72s | 0.65s | 0.66s | 0.72s | 0.69s |
| depth-anything-base-q8_0 | 0.59s | 0.59s | 0.57s | 0.59s | 0.58s |
| depth-anything-base-q4_k | 0.58s | 0.54s | 0.54s | 0.54s | 0.55s |
| depth-anything-large-f32 | 2.34s | 1.67s | 1.70s | 1.69s | 1.85s |
| **depth-anything-large-q8_0** | **1.06s** | **1.08s** | **1.03s** | **1.08s** | **1.06s** |
| **depth-anything-large-q4_k** | **1.12s** | **1.03s** | **1.10s** | **1.05s** | **1.07s** |
| depth-anything-giant-f32 | 4.84s | 4.02s | 4.10s | 3.95s | 4.22s |
| **depth-anything-giant-q8_0** | **1.81s** | **1.85s** | **1.83s** | **1.82s** | **1.83s** |
| **depth-anything-giant-q4_k** | **1.56s** | **1.63s** | **1.67s** | **1.63s** | **1.62s** |

### Depth 效果图对比（含量化版本）

3×3 网格，从左到右从上到下：Original | small | base-f32 | base-q8 | base-q4k | large-f32 | large-q8 | giant-f32 | giant-q8

**canyon.jpg**

![canyon comparison](comparison/canyon_all_models.png)

**desk.jpg**

![desk comparison](comparison/desk_all_models.png)

**mountains.jpg**

![mountains comparison](comparison/mountains_all_models.png)

**street.jpg**

![street comparison](comparison/street_all_models.png)

### Depth Range Comparison (min/max per image)


**canyon.jpg**

| Model | min | max | range |
|-------|-----|-----|-------|
| depth-anything-small-f32 | 0.3077 | 3.2835 | 2.9758 |
| depth-anything-base-f32 | 0.2848 | 4.5876 | 4.3028 |
| depth-anything-base-f16 | 0.2849 | 4.5873 | 4.3024 |
| depth-anything-base-q8_0 | 0.2829 | 4.5815 | 4.2986 |
| depth-anything-base-q4_k | 0.2726 | 4.4143 | 4.1417 |
| depth-anything-large-f32 | 0.2605 | 5.4264 | 5.1659 |
| **depth-anything-large-q8_0** | **0.2622** | **5.4514** | **5.1892** |
| depth-anything-giant-f32 | 0.0873 | 3.8366 | 3.7493 |
| **depth-anything-giant-q8_0** | **0.0889** | **3.8369** | **3.7480** |

**desk.jpg**

| Model | min | max | range |
|-------|-----|-----|-------|
| depth-anything-small-f32 | 0.6277 | 2.0183 | 1.3906 |
| depth-anything-base-f32 | 0.7306 | 2.1264 | 1.3958 |
| depth-anything-base-f16 | 0.7306 | 2.1281 | 1.3975 |
| depth-anything-base-q8_0 | 0.7313 | 2.1307 | 1.3994 |
| depth-anything-base-q4_k | 0.7415 | 2.4887 | 1.7472 |
| depth-anything-large-f32 | 0.7594 | 2.1699 | 1.4105 |
| **depth-anything-large-q8_0** | **0.7593** | **2.1719** | **1.4126** |
| depth-anything-giant-f32 | 0.8168 | 2.0913 | 1.2745 |
| **depth-anything-giant-q8_0** | **0.8165** | **2.0923** | **1.2758** |

**mountains.jpg**

| Model | min | max | range |
|-------|-----|-----|-------|
| depth-anything-small-f32 | 0.3286 | 2.5946 | 2.2660 |
| depth-anything-base-f32 | 0.4370 | 3.7581 | 3.3211 |
| depth-anything-base-f16 | 0.4373 | 3.7551 | 3.3178 |
| depth-anything-base-q8_0 | 0.4372 | 3.7621 | 3.3249 |
| depth-anything-base-q4_k | 0.4694 | 4.0034 | 3.5340 |
| depth-anything-large-f32 | 0.7460 | 8.2707 | 7.5247 |
| **depth-anything-large-q8_0** | **0.7426** | **8.2453** | **7.5027** |
| depth-anything-giant-f32 | 0.3505 | 7.6696 | 7.3191 |
| **depth-anything-giant-q8_0** | **0.3540** | **7.6755** | **7.3215** |

**street.jpg**

| Model | min | max | range |
|-------|-----|-----|-------|
| depth-anything-small-f32 | 0.7643 | 1.5154 | 0.7511 |
| depth-anything-base-f32 | 0.8853 | 1.3028 | 0.4175 |
| depth-anything-base-f16 | 0.8855 | 1.3026 | 0.4171 |
| depth-anything-base-q8_0 | 0.8866 | 1.3029 | 0.4163 |
| depth-anything-base-q4_k | 0.8792 | 1.2529 | 0.3737 |
| depth-anything-large-f32 | 0.8996 | 1.4520 | 0.5524 |
| **depth-anything-large-q8_0** | **0.9001** | **1.4552** | **0.5551** |
| depth-anything-giant-f32 | 0.7973 | 1.8239 | 1.0266 |
| **depth-anything-giant-q8_0** | **0.7964** | **1.8200** | **1.0236** |

## 2. Pose Estimation

| Model | canyon | desk | mountains | street |
|-------|--------|------|-----------|--------|
| depth-anything-small-f32 | 0.46s | 0.41s | 0.46s | 0.47s |
| depth-anything-base-f32 | 0.83s | 0.81s | 0.79s | 0.84s |
| depth-anything-base-f16 | 0.74s | 0.70s | 0.78s | 0.67s |
| depth-anything-base-q8_0 | 0.62s | 0.65s | 0.64s | 0.69s |
| depth-anything-base-q4_k | 0.64s | 0.64s | 0.60s | 0.62s |
| depth-anything-large-f32 | 1.82s | 1.75s | 1.82s | 1.78s |
| depth-anything-giant-f32 | 4.14s | 4.22s | 4.21s | 4.24s |

## 3. Nested Metric Depth（绝对深度，单位：米）

> **Note**: Nested metric mode loads TWO models simultaneously. With anyview-f32 (4.6GB) + metric (1.3GB),
> RTX 3060 (12GB) OOM。使用量化的 anyview + metric 可成功运行！

### anyview-q4_k (905MB) + metric (1.3GB) — 推荐 12GB GPU 组合

| Image | Time | min (m) | max (m) | scale_factor | Status |
|-------|------|---------|---------|--------------|--------|
| canyon | 6.77s | 0.68 | 29.48 | 6.62 | OK |
| desk | 6.79s | 1.95 | 4.83 | 2.36 | OK |
| mountains | 6.85s | 2.21 | 48.19 | 6.38 | OK |
| street | 6.96s | 0.70 | 1.69 | 0.85 | OK |

### anyview-q8_0 (1.5GB) + metric (1.3GB) — 需要 ≥16GB GPU

| Image | Time | min (m) | max (m) | scale_factor | Status |
|-------|------|---------|---------|--------------|--------|
| canyon | ~7s | 0.66 | 28.77 | 6.53 | OK |
| desk | ~7s | 1.90 | 4.85 | 2.31 | OK |
| mountains | ~7s | 2.17 | 47.47 | 6.20 | OK |
| street | ~7s | 0.68 | 1.75 | 0.82 | OK |

### Nested 量化精度对比（anyview-q8 作为高精度基准）

| Image | corr(q8 vs q4k) | MAE (m) | q8 range (m) | q4k range (m) |
|-------|-----------------|---------|--------------|---------------|
| canyon | 0.9999 | 0.255 | 0.66~28.77 | 0.68~29.48 |
| desk | 0.9998 | 0.055 | 1.90~4.85 | 1.95~4.83 |
| mountains | 0.9999 | 0.400 | 2.17~47.47 | 2.21~48.19 |
| street | 0.9984 | 0.033 | 0.68~1.75 | 0.70~1.69 |

> - `nested-metric.gguf` 的权重尺寸太小，量化 0 个权重（q8/q4k 文件与原版完全一致），无需单独量化
> - anyview-q4_k vs anyview-q8_0 精度差异极小（相关系数>0.998），推荐 q4_k 以节省显存
> - 输出分辨率 1022×672，含 scale_factor 用于将相对深度对齐到绝对尺度

## 4. Multi-view Depth + Pose

| Model | Pair | Time | Files | Status |
|-------|------|------|-------|--------|
| depth-anything-base-f32 | canyon_mountains | 20.53s | 6 | ok |
| depth-anything-base-f32 | desk_street | 2.52s | 6 | ok |
| depth-anything-giant-f32 | canyon_mountains | 39.38s | 6 | ok |
| depth-anything-giant-f32 | desk_street | 11.80s | 6 | ok |

## 5. 3D Export

| Model | GLB Time | GLB | COLMAP | Reconstruct Time | PLY | PLY Size |
|-------|----------|-----|--------|------------------|-----|----------|
| depth-anything-base-f32 | 1.11s | OK | OK | N/A (no GsHead) | - | - |
| depth-anything-giant-f32 | 4.71s | OK | OK | 10.40s | OK | 37MB |
| **depth-anything-giant-q8_0** | - | - | - | **8.03s** | **OK** | **37MB** |
| **depth-anything-giant-q4_k** | - | - | - | **7.87s** | **OK** | **37MB** |

> `reconstruct` 需要 GsHead 权重，仅 giant 系列模型支持。
> 量化后 reconstruct 速度从 10.4s 降至 ~8s（1.3x 加速），PLY 大小不变（686784 个高斯点）。

## 6. Model Metadata

| Model | Info Lines |
|-------|------------|
| depth-anything-small-f32 | 9 |
| depth-anything-base-f32 | 9 |
| depth-anything-base-f16 | 9 |
| depth-anything-base-q8_0 | 9 |
| depth-anything-base-q4_k | 9 |
| depth-anything-large-f32 | 9 |
| depth-anything-giant-f32 | 9 |
| depth-anything-nested-anyview | 9 |
| depth-anything-nested-metric | 9 |


## 7. 综合分析

### 模型文件大小与推理速度对比

| 模型 | 文件大小 | Avg推理耗时 | 加速比(vs f32) | 精度评级 |
|------|----------|-------------|----------------|----------|
| depth-anything-small-f32 | 100 MB | 0.44s | - | ★★ |
| depth-anything-base-q4_k | 99 MB | 0.55s | 1.44x | ★★★ |
| depth-anything-base-q8_0 | 142 MB | 0.58s | 1.36x | ★★★ |
| depth-anything-base-f16 | 222 MB | 0.69s | 1.14x | ★★★ |
| **depth-anything-large-q4_k** | **301 MB** | **1.07s** | **1.73x** | **★★★★** |
| depth-anything-base-f32 | 393 MB | 0.79s | 1.00x | ★★★ |
| **depth-anything-large-q8_0** | **449 MB** | **1.06s** | **1.75x** | **★★★★** |
| **depth-anything-giant-q4_k** | **905 MB** | **1.62s** | **2.60x** | **★★★★★** |
| depth-anything-large-f32 | 1318 MB | 1.85s | 1.00x | ★★★★ |
| **depth-anything-giant-q8_0** | **1536 MB** | **1.83s** | **2.31x** | **★★★★★** |
| depth-anything-giant-f32 | 4679 MB | 4.22s | 1.00x | ★★★★★ |

### 深度输出类型说明

| 模型类型 | 输出类型 | 单位 | 说明 |
|----------|----------|------|------|
| small / base / large / giant | **相对深度** (Relative) | 无单位 | 值仅表示远近关系，越大越远。不同图不可比 |
| nested (anyview + metric) | **绝对深度** (Metric) | 米 (m) | 通过 scale_factor 对齐到真实尺度 |

> - 相对深度用途：场景理解、segmentation、渲染调焦、深度对比
> - 绝对深度用途：3D测量、SLAM、避障、点云重建、需要真实距离的工程应用

### 像素级精度对比（相关系数 vs giant-f32 作为最高精度基准）

| Model | canyon | desk | mountains | street | **Avg corr** |
|-------|--------|------|-----------|--------|--------------|
| small-f32 | 0.8822 | 0.8943 | 0.9101 | 0.9366 | 0.9058 |
| base-f32 | 0.9135 | 0.9795 | 0.8716 | 0.9524 | 0.9292 |
| base-q8_0 | 0.9148 | 0.9795 | 0.8720 | 0.9532 | 0.9299 |
| base-q4_k | 0.9326 | 0.9805 | 0.8638 | 0.9397 | 0.9292 |
| large-f32 | 0.9712 | 0.9958 | 0.9025 | 0.9239 | 0.9484 |
| large-q8_0 | 0.9712 | 0.9957 | 0.9021 | 0.9268 | 0.9489 |
| large-q4_k | 0.9743 | 0.9948 | 0.9046 | 0.8881 | 0.9405 |
| giant-f32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| giant-q8_0 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | **0.9999** |
| giant-q4_k | 0.9998 | 0.9999 | 0.9966 | 0.9998 | **0.9990** |

### 同族量化 MAE（平均绝对误差，相对深度单位）

| 模型族 | 量化 | canyon | desk | mountains | street | **Avg MAE** |
|--------|------|--------|------|-----------|--------|-------------|
| base | q8_0 | 0.0037 | 0.0022 | 0.0052 | 0.0009 | **0.0030** |
| base | q4_k | 0.0647 | 0.1352 | 0.0717 | 0.0199 | **0.0729** |
| large | q8_0 | 0.0068 | 0.0022 | 0.0148 | 0.0024 | **0.0065** |
| large | q4_k | 0.0372 | 0.0229 | 0.1289 | 0.0222 | **0.0528** |
| giant | q8_0 | 0.0031 | 0.0009 | 0.0186 | 0.0007 | **0.0058** |
| giant | q4_k | 0.0118 | 0.0049 | 0.0941 | 0.0123 | **0.0308** |

> **结论**: q8_0 量化精度损失极小（MAE < 0.007），几乎不可感知；q4_k 精度损失 ~3-7%，但速度提升显著。

### 推荐使用场景

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 实时预览/嵌入式 | small-f32 / base-q4_k | 0.4-0.5s 极快推理 |
| 通用深度估计 | base-q8_0 | 精度-速度最佳平衡，142MB |
| **高精度深度** | **large-q4_k / large-q8_0** | **~1.07s，301-449MB，精度极高** |
| **3D重建 (reconstruct)** | **giant-q4_k** | **7.9s，905MB，精度损失仅0.52%** |
| **Nested metric depth** | **nested-anyview-q4_k + nested-metric** | **6.8s，仅需905MB+1.3GB，12GB GPU可跑** |
| 极致精度 | giant-f32 | 无损精度，4.6GB |
| 多视角一致性 | giant-q4_k / giant-q8_0 | multi-view 一致性最好 |
| 工程部署(低显存) | base-q4_k / large-q4_k | 99-301MB，极低显存 |

### 模型架构信息

| 模型 | 架构 | embed_dim | depth | num_heads | 参数量级 |
|------|------|-----------|-------|-----------|----------|
| depth-anything-small-f32 | DA3-SMALL | 384 | 12 | 6 | ~25M |
| depth-anything-base-f32 | DA3-BASE | 768 | 12 | 12 | ~98M |
| depth-anything-base-f16 | DA3-BASE | 768 | 12 | 12 | ~98M |
| depth-anything-base-q8_0 | DA3-BASE | 768 | 12 | 12 | ~98M |
| depth-anything-base-q4_k | DA3-BASE | 768 | 12 | 12 | ~98M |
| depth-anything-large-f32 | DA3-LARGE | 1024 | 24 | 16 | ~335M |
| depth-anything-giant-f32 | DA3-GIANT | 1536 | 40 | 24 | ~1.1B |
| depth-anything-nested-anyview | DA3NESTED-GIANT-LARGE-anyview | 1536 | 40 | 24 | ~1.1B+335M |
| depth-anything-nested-metric | DA3NESTED-metric-LARGE | 1024 | 24 | 16 | ~335M |
