# ACloudViewer Plugins

Each plugin ships its own **README** next to its source code. This file is the catalog and build index; detailed usage lives in the per-plugin documents below.

```
plugins/
├── core/
│   ├── IO/          # File format import/export
│   └── Standard/    # Processing, analysis, visualization, AI
└── README.md        # ← you are here
```

## Documentation convention

| Location | Purpose |
|----------|---------|
| `plugins/core/<Category>/<PluginName>/README.md` | **Authoritative** plugin doc: features, build flags, usage, models |
| `docs/guides/plugins/` | **User guides** for AI plugins (qDA3, qDeepLSD, qFreeSplatter, qLightGlue); synced into Sphinx |
| `plugins/core/<Category>/<PluginName>/info.json` | Plugin metadata shown in the app |
| [`BUILD.md`](https://github.com/Asher-1/ACloudViewer/blob/main/BUILD.md) | CMake option table and common build recipes |
| [`docs/source/cpp_api/plugins.rst`](https://github.com/Asher-1/ACloudViewer/blob/main/docs/source/cpp_api/plugins.rst) | Sphinx plugin overview (READMEs are synced at doc-build time) |

When adding a new plugin, add a `README.md` in the plugin directory and register the CMake option in [`BUILD.md`](https://github.com/Asher-1/ACloudViewer/blob/main/BUILD.md).

## Plugin UI grouping

Plugins self-declare a UI group via the optional `"group"` field in their
`info.json` (read by `ccDefaultPluginInterface::pluginGroup()`). Grouped
plugins cluster behind their own `Plugins` menu section and their own
draggable toolbar instead of the generic plugin menu/toolbar. The only group
today is `"aicore"` (the AI inference family: qDA3, qDeepLSD, qFaceDetect,
qFreeSplatter, qGKD, qInstantMesh, qLightGlue, qLingbotMap, qMapAnything,
qOCR, qOpenPCDet, qQuerySplat, qRAM, qRelateAnything, qRFDetr, qRMBG,
qSAM3, qSAM3D, qTrellis, qVLXSeek, qYOLO, qAnyLabeling); ungrouped plugins
keep the historical order and toolbar.

---

## AI inference plugins (AICore)

AICore-backed GUI plugins link **`libAICore.so`**, which bundles the enabled
inference tasks with one shared ggml runtime (including RelateAnything).

| Plugin | CMake option | User guide | README |
|--------|--------------|------------|--------|
| **qDA3** | `PLUGIN_STANDARD_QDA3` | [docs/guides/plugins/qDA3.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qDA3.md) | [qDA3/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qDA3/README.md) |
| **qDeepLSD** | `PLUGIN_STANDARD_QDEEPLSD` | [docs/guides/plugins/qDeepLSD.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qDeepLSD.md) | [qDeepLSD/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qDeepLSD/README.md) |
| **qFaceDetect** | `PLUGIN_STANDARD_QFACEDETECT` | [docs/guides/plugins/qFaceDetect.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qFaceDetect.md) | [qFaceDetect/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFaceDetect/README.md) |
| **qLightGlue** | `PLUGIN_STANDARD_QLIGHTGLUE` | [docs/guides/plugins/qLightGlue.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qLightGlue.md) | [qLightGlue/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qLightGlue/README.md) |
| **qFreeSplatter** | `PLUGIN_STANDARD_QFREESPLATTER` | [docs/guides/plugins/qFreeSplatter.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qFreeSplatter.md) | [qFreeSplatter/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFreeSplatter/README.md) |
| **qRFDetr** | `PLUGIN_STANDARD_QRFDETR` | [docs/guides/plugins/qRFDetr.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRFDetr.md) | [qRFDetr/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRFDetr/README.md) |
| **qRMBG** | `PLUGIN_STANDARD_QRMBG` | [docs/guides/plugins/qRMBG.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRMBG.md) | [qRMBG/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRMBG/README.md) |
| **qYOLO** | `PLUGIN_STANDARD_QYOLO` | [docs/guides/plugins/qYOLO.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qYOLO.md) | [qYOLO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qYOLO/README.md) |
| **qAnyLabeling** | `PLUGIN_STANDARD_QANYLABELING` | [docs/guides/plugins/qAnyLabeling.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qAnyLabeling.md) | [qAnyLabeling/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qAnyLabeling/README.md) |
| **qSAM3** | `PLUGIN_STANDARD_QSAM3` | [docs/guides/plugins/qSAM3.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSAM3.md) | [qSAM3/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSAM3/README.md) |
| **qSAM3D** | `PLUGIN_STANDARD_QSAM3D` | [docs/guides/plugins/qSAM3D.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSAM3D.md) | [qSAM3D/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSAM3D/README.md) |
| **qTrellis** | `PLUGIN_STANDARD_QTRELLIS` | [docs/guides/plugins/qTrellis.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qTrellis.md) | [qTrellis/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qTrellis/README.md) |
| **qSpirula** | `PLUGIN_STANDARD_QSPIRULA` | [docs/guides/plugins/qSpirula.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSpirula.md) | [qSpirula/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSpirula/README.md) |
| **qGKD** | `PLUGIN_STANDARD_QGKD` | [docs/guides/plugins/qGKD.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qGKD.md) | [qGKD/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qGKD/README.md) |
| **qLingbotMap** | `PLUGIN_STANDARD_QLINGBOTMAP` | [docs/guides/plugins/qLingbotMap.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qLingbotMap.md) | [qLingbotMap/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qLingbotMap/README.md) |
| **qRAM** | `PLUGIN_STANDARD_QRAM` | [docs/guides/plugins/qRAM.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRAM.md) | [qRAM/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRAM/README.md) |
| **qRelateAnything** | `PLUGIN_STANDARD_QRELATEANYTHING` | [docs/guides/plugins/qRelateAnything.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRelateAnything.md) | [qRelateAnything/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRelateAnything/README.md) |
| **qMapAnything** | `PLUGIN_STANDARD_QMAPANYTHING` | [docs/guides/plugins/qMapAnything.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qMapAnything.md) | [qMapAnything/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qMapAnything/README.md) |
| **qOpenPCDet** | `PLUGIN_STANDARD_QOPENPCDET` | [docs/guides/plugins/qOpenPCDet.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qOpenPCDet.md) | [qOpenPCDet/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qOpenPCDet/README.md) |
| **qQuerySplat** | `PLUGIN_STANDARD_QQUERYSPLAT` | [docs/guides/plugins/qQuerySplat.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qQuerySplat.md) | [qQuerySplat/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qQuerySplat/README.md) |
| **qVLXSeek** | `PLUGIN_STANDARD_QVLXSEEK` | [docs/guides/plugins/qVLXSeek.md](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qVLXSeek.md) | [qVLXSeek/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qVLXSeek/README.md) |

**Core library:** enable with `-DAICore_ENABLED=ON` (auto-enables `GGML_ENABLED`).

**Typical build:**

```bash
cmake -DBUILD_GUI=ON \
      -DAICore_ENABLED=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DPLUGIN_STANDARD_QDEEPLSD=ON \
      -DPLUGIN_STANDARD_QFACEDETECT=ON \
      -DPLUGIN_STANDARD_QFREESPLATTER=ON \
      -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
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
      ..
```

- **qDA3 + Automatic Reconstruction:** also set `-DBUILD_RECONSTRUCTION=ON`.
- **qFreeSplatter in-app visualization:** `-DPLUGIN_STANDARD_QSIBR=ON` (Linux/Windows; macOS OpenGL limits may apply).
- **GPU inference:** `-DBUILD_CUDA_MODULE=ON` for ggml CUDA backend.

---

## Standard plugins (processing & analysis)

| Plugin | README |
|--------|--------|
| q3DMASC | [core/Standard/q3DMASC/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/q3DMASC/README.md) |
| qAnimation | [core/Standard/qAnimation/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qAnimation/README.md) |
| qBroom | [core/Standard/qBroom/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qBroom/README.md) |
| qCanupo | [core/Standard/qCanupo/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qCanupo/README.md) |
| qCloudLayers | [core/Standard/qCloudLayers/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qCloudLayers/README.md) |
| qColorimetricSegmenter | [core/Standard/qColorimetricSegmenter/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qColorimetricSegmenter/README.md) |
| qCompass | [core/Standard/qCompass/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qCompass/README.md) |
| qCork | [core/Standard/qCork/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qCork/README.md) |
| qCSF | [core/Standard/qCSF/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qCSF/README.md) |
| qFacets | [core/Standard/qFacets/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qFacets/README.md) |
| qHoughNormals | [core/Standard/qHoughNormals/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qHoughNormals/README.md) |
| qJSonRPCPlugin | [core/Standard/qJSonRPCPlugin/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qJSonRPCPlugin/README.md) |
| qM3C2 | [core/Standard/qM3C2/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qM3C2/README.md) |
| qMPlane | [core/Standard/qMPlane/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qMPlane/README.md) |
| qPCL | [core/Standard/qPCL/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qPCL/README.md) |
| qPCV | [core/Standard/qPCV/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qPCV/README.md) |
| qPoissonRecon | [core/Standard/qPoissonRecon/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qPoissonRecon/README.md) |
| qPythonRuntime | [core/Standard/qPythonRuntime/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qPythonRuntime/README.md) |
| qRANSAC_SD | [core/Standard/qRANSAC_SD/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRANSAC_SD/README.md) |
| qSIBR | [core/Standard/qSIBR/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSIBR/README.md) |
| qSRA | [core/Standard/qSRA/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSRA/README.md) |
| qTreeIso | [core/Standard/qTreeIso/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qTreeIso/README.md) |
| qVoxFall | [core/Standard/qVoxFall/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qVoxFall/README.md) |

Submodule plugins (when present): qColorimetricSegmenter, qMasonry, qMPlane, qJSonRPCPlugin, qG3Point.

---

## I/O plugins

| Plugin | README |
|--------|--------|
| qCoreIO | [core/IO/qCoreIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qCoreIO/README.md) |
| qAdditionalIO | [core/IO/qAdditionalIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qAdditionalIO/README.md) |
| qCSVMatrixIO | [core/IO/qCSVMatrixIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qCSVMatrixIO/README.md) |
| qDracoIO | [core/IO/qDracoIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qDracoIO/README.md) |
| qE57IO | [core/IO/qE57IO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qE57IO/README.md) |
| qFBXIO | [core/IO/qFBXIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qFBXIO/README.md) |
| qLASIO | [core/IO/qLASIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qLASIO/README.md) |
| qLASFWFIO | [core/IO/qLASFWFIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qLASFWFIO/README.md) |
| qMeshIO | [core/IO/qMeshIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qMeshIO/README.md) |
| qPDALIO | [core/IO/qPDALIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qPDALIO/README.md) |
| qPhotoscanIO | [core/IO/qPhotoscanIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qPhotoscanIO/README.md) |
| qRDBIO | [core/IO/qRDBIO/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qRDBIO/README.md) |
| qStepCADImport | [core/IO/qStepCADImport/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/IO/qStepCADImport/README.md) |

---

## See also

- [BUILD.md](https://github.com/Asher-1/ACloudViewer/blob/main/BUILD.md) — full CMake plugin option table
- [Creating a plugin](https://github.com/Asher-1/ACloudViewer/blob/main/docs/source/cpp_api/plugins.rst) — plugin API and scaffolding guide
