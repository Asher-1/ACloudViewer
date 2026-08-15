# AI inference plugins (AICore)

ACloudViewer provides five AI plugins through the unified inference library **`libAICore.so`** (ggml). **No Python/PyTorch runtime is required.**

| Plugin | Guide | CMake option | Summary |
|--------|-------|--------------|---------|
| **qDA3** | [qDA3 user guide](qDA3.md) | `PLUGIN_STANDARD_QDA3` | Depth Anything V3: monocular/multi-view depth, camera pose, COLMAP/GLB export, automatic reconstruction integration |
| **qDeepLSD** | [qDeepLSD user guide](qDeepLSD.md) | `PLUGIN_STANDARD_QDEEPLSD` | DeepLSD wireframe: line-segment extraction from images, GGUF df/angle + LSD post-process |
| **qFaceDetect** | [qFaceDetect user guide](qFaceDetect.md) | `PLUGIN_STANDARD_QFACEDETECT` | face-detect.cpp: SCRFD/YuNet detection, ArcFace/SFace verify, age/gender, anti-spoof |
| **qLightGlue** | [qLightGlue user guide](qLightGlue.md) | `PLUGIN_STANDARD_QLIGHTGLUE` | SIFT/ALIKED LightGlue GGUF — sparse matching |
| **qFreeSplatter** | [qFreeSplatter user guide](qFreeSplatter.md) | `PLUGIN_STANDARD_QFREESPLATTER` | FreeSplatter: uncalibrated photos → 3D Gaussian splats; **Face Capture** tab (OpenCV Haar or AICore GGUF detector); SIBR PLY; optional qSIBR preview |

## Prerequisites

- `-DAICore_ENABLED=ON` (build `core/AICore` → `libAICore.so`)
- GUI: `-DBUILD_GUI=ON`
- **qFaceDetect:** system libjpeg (e.g. `libjpeg-dev` on Ubuntu)
- **qDA3 + automatic reconstruction:** `-DBUILD_RECONSTRUCTION=ON`
- **FreeSplatter one-click Visualize:** `-DPLUGIN_STANDARD_QSIBR=ON` (Linux/Windows; macOS CI disables qSIBR by default)
- **FreeSplatter Face Capture tab:** `-DBUILD_OPENCV=ON` (webcam + Haar or GGML face detector via AICore; GGUF packs from [qFaceDetect release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/qFaceDetect))
- **Inference acceleration:** Vulkan on Linux/Windows, Metal + CPU on macOS (Vulkan unsupported — MoltenVK SPIR-V translation limitations); SYCL/CUDA are optional developer backends

## Typical build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDA3=ON \
  -DPLUGIN_STANDARD_QDEEPLSD=ON \
  -DPLUGIN_STANDARD_QFACEDETECT=ON \
  -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
  -DPLUGIN_STANDARD_QMANUAL_CALIB=ON \
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  -DBUILD_RECONSTRUCTION=ON \
  .

cmake --build build_app --target ACloudViewer QDA3_PLUGIN QFACEDETECT_PLUGIN QFREESPLATTER_PLUGIN -j$(nproc)
```

## 标定插件（qManualCalib）

非 AICore 插件，示例数据随源码 `tests/data/` 集成，无需额外下载。

| 插件 | 文档 | CMake 选项 |
|------|------|------------|
| **qManualCalib** | [qManualCalib 使用指南](qManualCalib.md) | `PLUGIN_STANDARD_QMANUAL_CALIB` |

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DBUILD_OPENCV=ON \
  -DPLUGIN_STANDARD_QMANUAL_CALIB=ON \
  .

cmake --build build_app --target QMANUAL_CALIB_PLUGIN ACloudViewer -j$(nproc)
```

## 更多资料

- 插件目录完整 README（开发者细节、测试、C API）：[`plugins/core/Standard/qDA3/README.md`](../../../plugins/core/Standard/qDA3/README.md)、[`plugins/core/Standard/qDeepLSD/README.md`](../../../plugins/core/Standard/qDeepLSD/README.md)、[`plugins/core/Standard/qFaceDetect/README.md`](../../../plugins/core/Standard/qFaceDetect/README.md)、[`plugins/core/Standard/qLightGlue/README.md`](../../../plugins/core/Standard/qLightGlue/README.md)、[`plugins/core/Standard/qFreeSplatter/README.md`](../../../plugins/core/Standard/qFreeSplatter/README.md)
- qManualCalib 开发者 README：[`plugins/core/Standard/qManualCalib/README.md`](../../../plugins/core/Standard/qManualCalib/README.md)
- 插件总索引：[`plugins/README.md`](../../../plugins/README.md)
- Sphinx 构建时会将上述 README 同步到 `docs/source/plugins/`（见 `docs/source/conf.py`）
