# AI inference plugins (AICore)

ACloudViewer provides five AI plugins through the unified inference library **`libAICore.so`** (ggml). **No Python/PyTorch runtime is required.**

| Plugin | Guide | CMake option | Summary |
|--------|-------|--------------|---------|
| **qDA3** | [qDA3 user guide](qDA3.md) | `PLUGIN_STANDARD_QDA3` | Depth Anything V3: monocular/multi-view depth, camera pose, COLMAP/GLB export, automatic reconstruction integration |
| **qDeepLSD** | [qDeepLSD user guide](qDeepLSD.md) | `PLUGIN_STANDARD_QDEEPLSD` | DeepLSD wireframe: line-segment extraction from images, GGUF df/angle + LSD post-process |
| **qFaceDetect** | [qFaceDetect user guide](qFaceDetect.md) | `PLUGIN_STANDARD_QFACEDETECT` | face-detect.cpp: SCRFD/YuNet detection, ArcFace/SFace verify, age/gender, anti-spoof |
| **qLightGlue** | [qLightGlue user guide](qLightGlue.md) | `PLUGIN_STANDARD_QLIGHTGLUE` | SIFT/ALIKED LightGlue GGUF — sparse matching |
| **qFreeSplatter** | [qFreeSplatter user guide](qFreeSplatter.md) | `PLUGIN_STANDARD_QFREESPLATTER` | FreeSplatter: uncalibrated photos → 3D Gaussian splats; **Face Capture** tab (OpenCV Haar or AICore GGUF detector); SIBR PLY; optional qSIBR preview |
| **qSAM3** | [qSAM3 user guide](qSAM3.md) | `PLUGIN_STANDARD_QSAM3` | SAM2/SAM2.1/SAM3: promptable segmentation + video tracking (GGUF, DINOv2 + Hiera backbones) |
| **qTrellis** | [qTrellis README](../../../plugins/core/Standard/qTrellis/README.md) | `PLUGIN_STANDARD_QTRELLIS` | TRELLIS.2: single-image → 3D triangle mesh with PBR materials (GGUF, DINOv3 + flow-matching DiTs + FlexiDualGrid VAE) |

## Prerequisites

- `-DAICore_ENABLED=ON` (build `core/AICore` → `libAICore.so`)
- GUI: `-DBUILD_GUI=ON`
- **qFaceDetect:** image decode via Qt built-in codecs (no system libjpeg needed)
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
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DPLUGIN_STANDARD_QRFDETR=ON \
  -DPLUGIN_STANDARD_QRMBG=ON \
  -DPLUGIN_STANDARD_QYOLO=ON \
  -DPLUGIN_STANDARD_QSAM3=ON \
  -DPLUGIN_STANDARD_QTRELLIS=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  -DBUILD_RECONSTRUCTION=ON \
  .

cmake --build build_app --target ACloudViewer QDA3_PLUGIN QFACEDETECT_PLUGIN QFREESPLATTER_PLUGIN -j$(nproc)
```

## More resources

- Full plugin READMEs (developer details, tests, C API): [`plugins/core/Standard/qDA3/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qDA3/README.md), [`plugins/core/Standard/qDeepLSD/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qDeepLSD/README.md), [`plugins/core/Standard/qFaceDetect/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFaceDetect/README.md), [`plugins/core/Standard/qLightGlue/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qLightGlue/README.md), [`plugins/core/Standard/qFreeSplatter/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFreeSplatter/README.md), [`plugins/core/Standard/qSAM3/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSAM3/README.md), [`plugins/core/Standard/qTrellis/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qTrellis/README.md)
- Plugin catalog: [`plugins/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/README.md)
- Sphinx doc build syncs the above READMEs into `docs/source/plugins/` (see `docs/source/conf.py`)

## Image asset policy

Plugin user guides in this directory must reference plugin-owned images through
the GitHub `blob/main/...?...raw=1` URL, for example
`https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFaceDetect/images/qFaceDetect.png?raw=1`.
This keeps direct GitHub Markdown viewing and Sphinx-rendered pages independent
of their different relative directories. `docs/source/_static/plugin-assets/`
remains for Sphinx-owned pages such as the documentation landing page; it is
not the image source for guides in this directory.
