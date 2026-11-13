# AI inference plugins (AICore)

ACloudViewer provides twenty-two AI plugins through the unified inference library **`libAICore.so`** (ggml). **No Python/PyTorch runtime is required.**

| Plugin | Guide | CMake option | Summary |
|--------|-------|--------------|---------|
| **qDA3** | [qDA3 user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qDA3.md) | `PLUGIN_STANDARD_QDA3` | Depth Anything V3: monocular/multi-view depth, camera pose, COLMAP/GLB export, automatic reconstruction integration |
| **qDeepLSD** | [qDeepLSD user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qDeepLSD.md) | `PLUGIN_STANDARD_QDEEPLSD` | DeepLSD wireframe: line-segment extraction from images, GGUF df/angle + LSD post-process |
| **qFaceDetect** | [qFaceDetect user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qFaceDetect.md) | `PLUGIN_STANDARD_QFACEDETECT` | face-detect.cpp: SCRFD/YuNet detection, ArcFace/SFace verify, age/gender, anti-spoof |
| **qLightGlue** | [qLightGlue user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qLightGlue.md) | `PLUGIN_STANDARD_QLIGHTGLUE` | SIFT/ALIKED LightGlue GGUF — sparse matching |
| **qFreeSplatter** | [qFreeSplatter user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qFreeSplatter.md) | `PLUGIN_STANDARD_QFREESPLATTER` | FreeSplatter: uncalibrated photos → 3D Gaussian splats; **Face Capture** tab (OpenCV Haar or AICore GGUF detector); SIBR PLY; optional qSIBR preview |
| **qSAM3** | [qSAM3 user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSAM3.md) | `PLUGIN_STANDARD_QSAM3` | SAM2/SAM2.1/SAM3: promptable segmentation + video tracking (GGUF, DINOv2 + Hiera backbones) |
| **qTrellis** | [qTrellis user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qTrellis.md) | `PLUGIN_STANDARD_QTRELLIS` | TRELLIS.2: single-image → 3D triangle mesh with PBR materials (GGUF, DINOv3 + flow-matching DiTs + FlexiDualGrid VAE) |
| **qYOLO** | [qYOLO user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qYOLO.md) | `PLUGIN_STANDARD_QYOLO` | Ultralytics YOLOv8/YOLO26: COCO-80 detection, instance segmentation, metric depth, pose, OBB, classification + open-vocab World/YOLOE |
| **qRFDetr** | [qRFDetr user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRFDetr.md) | `PLUGIN_STANDARD_QRFDETR` | RF-DETR: real-time COCO detection & instance segmentation with per-class allowlist (image + live video) |
| **qRMBG** | [qRMBG user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRMBG.md) | `PLUGIN_STANDARD_QRMBG` | RMBG-2.0 (BiRefNet-Swin-L): transparent background removal, raw alpha matte, image + live video |
| **qGKD** | [qGKD user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qGKD.md) | `PLUGIN_STANDARD_QGKD` | GKDT-L (ECCV 2026) open-world keypoint detection: text / visual 1-shot / multimodal / multi-object (YOLO-World) prompts |
| **qInstantMesh** | [qInstantMesh user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qInstantMesh.md) | `PLUGIN_STANDARD_QINSTANTMESH` | InstantMesh: single image → textured 3D mesh (Zero123++ multi-view diffusion → FlexiCubes → xatlas GLB/OBJ export) |
| **qLingbotMap** | [qLingbotMap user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qLingbotMap.md) | `PLUGIN_STANDARD_QLINGBOTMAP` | LingBot-Map GCT: streaming RGB-D 3D reconstruction from ordered image folders / video with live per-frame map, frustums and sky masking |
| **qMapAnything** | [qMapAnything user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qMapAnything.md) | `PLUGIN_STANDARD_QMAPANYTHING` | MapAnything: feed-forward metric multi-view reconstruction across VGGT-Omega / MapAnything / Pi3 / Pi3X / VGGT-1B / DUSt3R variants, ETH3D benchmark anchoring |
| **qOCR** | [qOCR user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qOCR.md) | `PLUGIN_STANDARD_QOCR` | PP-OCRv4 detection + recognition (English / Japanese / Chinese+English recipes) with opt-in SLANet_plus layout and table stages, PP-Structure style DB overlay |
| **qOpenPCDet** | [qOpenPCDet user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qOpenPCDet.md) | `PLUGIN_STANDARD_QOPENPCDET` | OpenPCDet LiDAR 3D detection: 18 KITTI/nuScenes checkpoints × 3 quantizations; per-class wireframe boxes + CSV export |
| **qQuerySplat** | [qQuerySplat user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qQuerySplat.md) | `PLUGIN_STANDARD_QQUERYSPLAT` | QuerySplat: 2–16 unposed photos → 3D Gaussian Splatting scene (524 288 gaussians) with optional test-time optimization |
| **qRAM** | [qRAM user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRAM.md) | `PLUGIN_STANDARD_QRAM` | Recognize Anything / RAM++ / Tag2Text: image tagging + captioning over batches, DB images and camera/video streams; JSON/CSV/text export |
| **qRelateAnything** | [qRelateAnything user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRelateAnything.md) | `PLUGIN_STANDARD_QRELATEANYTHING` | Scene relation graphs: interchangeable detector (YOLOE / YOLO-World / RF-DETR) → subject–predicate–object edges with Kalman-tracked video smoothing |
| **qSAM3D** | [qSAM3D user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSAM3D.md) | `PLUGIN_STANDARD_QSAM3D` | SAM 3D Objects: single image → Gaussian splat PLY + optional FlexiCubes mesh |
| **qVLXSeek** | [qVLXSeek user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qVLXSeek.md) | `PLUGIN_STANDARD_QVLXSEEK` | VLX-Seek 1.5-10B open-vocabulary grounding: detection / counting / VQA / region OCR via region retrieval |
| **qAnyLabeling** | [qAnyLabeling user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qAnyLabeling.md) | `PLUGIN_STANDARD_QANYLABELING` | X-AnyLabeling-style annotation workbench — shapes, brush, magic wand, video/3D annotation; every AI assist runs locally through the AICore catalogs |

**3D Gaussian Splatting toolkit:** real-time rendering and novel-view
synthesis via the [SIBR viewer](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSIBR/README.md)
(`PLUGIN_STANDARD_QSIBR`), and 3DGS training from photos / video via the
[qSpirula](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSpirula/README.md)
Spirula Studio integration — [user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSpirula.md)
(`-DBUILD_SPIRULA=ON -DPLUGIN_STANDARD_QSPIRULA=ON`).

## Prerequisites

- `-DAICore_ENABLED=ON` (build `core/AICore` → `libAICore.so`)
- GUI: `-DBUILD_GUI=ON`
- **qFaceDetect:** image decode via Qt built-in codecs (no system libjpeg needed)
- **qDA3 + automatic reconstruction:** `-DBUILD_RECONSTRUCTION=ON`
- **qLingbotMap video input:** `-DBUILD_OPENCV=ON` (image-folder input works without it)
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
  -DPLUGIN_STANDARD_QSAM3D=ON \
  -DPLUGIN_STANDARD_QTRELLIS=ON \
  -DPLUGIN_STANDARD_QLINGBOTMAP=ON \
  -DPLUGIN_STANDARD_QGKD=ON \
  -DPLUGIN_STANDARD_QOCR=ON \
  -DPLUGIN_STANDARD_QRAM=ON \
  -DPLUGIN_STANDARD_QRELATEANYTHING=ON \
  -DPLUGIN_STANDARD_QANYLABELING=ON \
  -DPLUGIN_STANDARD_QMAPANYTHING=ON \
  -DPLUGIN_STANDARD_QOPENPCDET=ON \
  -DPLUGIN_STANDARD_QINSTANTMESH=ON \
  -DPLUGIN_STANDARD_QVLXSEEK=ON \
  -DPLUGIN_STANDARD_QQUERYSPLAT=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  -DBUILD_RECONSTRUCTION=ON \
  -DBUILD_SPIRULA=ON \
  -DPLUGIN_STANDARD_QSPIRULA=ON \
  .

cmake --build build_app --target ACloudViewer QDA3_PLUGIN QFACEDETECT_PLUGIN QFREESPLATTER_PLUGIN -j$(nproc)
```

## More resources

- Full plugin READMEs (developer details, tests, C API), one per plugin:
  [`qDA3`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qDA3/README.md) ·
  [`qDeepLSD`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qDeepLSD/README.md) ·
  [`qFaceDetect`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFaceDetect/README.md) ·
  [`qLightGlue`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qLightGlue/README.md) ·
  [`qFreeSplatter`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFreeSplatter/README.md) ·
  [`qSAM3`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSAM3/README.md) ·
  [`qTrellis`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qTrellis/README.md) ·
  [`qYOLO`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qYOLO/README.md) ·
  [`qRFDetr`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRFDetr/README.md) ·
  [`qRMBG`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRMBG/README.md) ·
  [`qGKD`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qGKD/README.md) ·
  [`qInstantMesh`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qInstantMesh/README.md) ·
  [`qLingbotMap`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qLingbotMap/README.md) ·
  [`qMapAnything`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qMapAnything/README.md) ·
  [`qOCR`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qOCR/README.md) ·
  [`qOpenPCDet`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qOpenPCDet/README.md) ·
  [`qQuerySplat`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qQuerySplat/README.md) ·
  [`qRAM`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRAM/README.md) ·
  [`qRelateAnything`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRelateAnything/README.md) ·
  [`qSAM3D`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSAM3D/README.md) ·
  [`qVLXSeek`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qVLXSeek/README.md) ·
  [`qAnyLabeling`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qAnyLabeling/README.md)
- Plugin catalog: [`plugins/README.md`](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/README.md)
- Sphinx doc build syncs the above READMEs into `docs/source/plugins/` (see `docs/source/conf.py`)

## Image asset policy

Plugin user guides in this directory must reference plugin-owned images with
the canonical GitHub raw URL, for example
`https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qFaceDetect/images/qFaceDetect.png`.
This keeps direct GitHub Markdown viewing and Sphinx-rendered pages independent
of their different relative directories (guides are copied into the Sphinx
srcdir at build time, so relative paths would break there). `docs/source/_static/plugin-assets/`
remains for Sphinx-owned pages such as the documentation landing page; it is
not the image source for guides in this directory.
