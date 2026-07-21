# FreeSplatter GGUF Models

GGUF weights for the **qFreeSplatter** plugin (AICore / FreeSplatter inference).

## Download (CloudViewer)

所有模型托管于 [cloudViewer_downloads](https://github.com/Asher-1/cloudViewer_downloads) release [**3dgs**](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/3dgs)。

Base URL：

```
https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/
```

插件对话框与自动下载使用相同 URL。每张输入图中心裁剪并缩放到 **512×512**；各精度变体共享同一 Transformer 骨干，仅量化格式与数值精度不同。

### Scene 模型（2 视图场景重建）

适用于有重叠的 **2 张**场景照片（室内/室外）。插件会将多余输入均匀下采样至 2 张。

| 下载 | 量化 | 体积 | 相对速度 | 相对质量 | 峰值内存（估） | 说明 |
|------|------|------|----------|----------|----------------|------|
| [`freesplatter-scene-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-q8_0.gguf) | Q8_0 | **324.5 MB** | 最快 | 良好（近无损量化） | ~0.8–1.2 GB | **默认推荐**；体积最小、加载最快，适合 CPU / 集成 GPU |
| [`freesplatter-scene-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f16.gguf) | F16 | **595.8 MB** | 中等 | 很好 | ~1.2–1.8 GB | 上游 FreeSplatter 推荐的 scene 精度；质量/速度均衡 |
| [`freesplatter-scene-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f32.gguf) | F32 | **1.15 GB** | 最慢 | 最佳（参考精度） | ~2.0–3.0 GB | 全精度参考模型；CPU 推理较慢，适合对标/离线批处理 |

### Object 模型（多视图物体重建）

适用于围绕单一物体拍摄的 **3 张及以上**照片（插件最多使用 16 张，超出会均匀下采样）。

| 下载 | 量化 | 体积 | 相对速度 | 相对质量 | 峰值内存（估） | 说明 |
|------|------|------|----------|----------|----------------|------|
| [`freesplatter-object-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-q8_0.gguf) | Q8_0 | **324.5 MB** | 最快 | 良好 | ~0.8–1.2 GB × 视图数 | **默认推荐**；多视图时总耗时随视图数线性增长 |
| [`freesplatter-object-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f16.gguf) | F16 | **595.8 MB** | 中等 | 很好 | ~1.2–1.8 GB × 视图数 | 物体细节更稳，适合 4–8 视图 |
| [`freesplatter-object-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f32.gguf) | F32 | **1.15 GB** | 最慢 | 最佳 | ~2.0–3.0 GB × 视图数 | 全精度；多视图 + F32 建议配合 CUDA / Vulkan |

> **体积说明：** 上表体积为 GitHub Release 资产精确字节数（2026-07-16 发布）。Scene / Object 同精度文件大小相同，差异在 GGUF 内嵌的 head 配置（`gaussian_channels`、`sh_residual`、`use_2dgs`）。
>
> **性能说明：** FreeSplatter 尚无公开的标准化 benchmark；相对速度/内存为同设备、同视图数下的经验排序。启用 **CUDA / Vulkan / Metal** 后端可显著缩短推理时间。GPU 与 GUI 同卡时，建议关闭 SIBR 查看器再跑推理以避免显存争用。

### 选型建议

| 场景 | 推荐模型（点击下载） |
|------|---------------------|
| 快速试用 / 笔记本 CPU | [`freesplatter-scene-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-q8_0.gguf) 或 [`freesplatter-object-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-q8_0.gguf) |
| 日常质量（Scene 2 视图） | [`freesplatter-scene-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f16.gguf) |
| 日常质量（Object 多视图） | [`freesplatter-object-q8_0.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-q8_0.gguf)（≤8 视图）或 [`freesplatter-object-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f16.gguf) |
| 最高质量 / 对标 | [`freesplatter-scene-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-f32.gguf) / [`freesplatter-object-f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-object-f32.gguf) + GPU 后端 |

手动下载示例：

```bash
curl -L -O https://github.com/Asher-1/cloudViewer_downloads/releases/download/3dgs/freesplatter-scene-q8_0.gguf
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

- Image size: 512×512
- Patch size: 8×8
- SH degree: 1
- Scale activation: sigmoid in [scale_min_act, scale_max_act]
