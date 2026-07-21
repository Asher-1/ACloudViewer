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
| `docs/guides/plugins/` | **User guides** for AI plugins (qDA3, qFreeSplatter); synced into Sphinx |
| `plugins/core/<Category>/<PluginName>/info.json` | Plugin metadata shown in the app |
| [`BUILD.md`](../BUILD.md) | CMake option table and common build recipes |
| [`docs/source/cpp_api/plugins.rst`](../docs/source/cpp_api/plugins.rst) | Sphinx plugin overview (READMEs are synced at doc-build time) |

When adding a new plugin, add a `README.md` in the plugin directory and register the CMake option in [`BUILD.md`](../BUILD.md).

---

## AI inference plugins (AICore)

Both plugins link **`libAICore.so`**, which bundles DA3 and FreeSplatter with a single ggml copy.

| Plugin | CMake option | User guide | README |
|--------|--------------|------------|--------|
| **qDA3** | `PLUGIN_STANDARD_QDA3` | [docs/guides/plugins/qDA3.md](../docs/guides/plugins/qDA3.md) | [qDA3/README.md](core/Standard/qDA3/README.md) |
| **qFreeSplatter** | `PLUGIN_STANDARD_QFREESPLATTER` | [docs/guides/plugins/qFreeSplatter.md](../docs/guides/plugins/qFreeSplatter.md) | [qFreeSplatter/README.md](core/Standard/qFreeSplatter/README.md) |

**Core library:** enable with `-DAICore_ENABLED=ON` (auto-enables `GGML_ENABLED`).

**Typical build:**

```bash
cmake -DBUILD_GUI=ON \
      -DAICore_ENABLED=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DPLUGIN_STANDARD_QFREESPLATTER=ON \
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
| q3DMASC | [core/Standard/q3DMASC/README.md](core/Standard/q3DMASC/README.md) |
| qAnimation | [core/Standard/qAnimation/README.md](core/Standard/qAnimation/README.md) |
| qBroom | [core/Standard/qBroom/README.md](core/Standard/qBroom/README.md) |
| qCanupo | [core/Standard/qCanupo/README.md](core/Standard/qCanupo/README.md) |
| qCloudLayers | [core/Standard/qCloudLayers/README.md](core/Standard/qCloudLayers/README.md) |
| qColorimetricSegmenter | [core/Standard/qColorimetricSegmenter/README.md](core/Standard/qColorimetricSegmenter/README.md) |
| qCompass | [core/Standard/qCompass/README.md](core/Standard/qCompass/README.md) |
| qCork | [core/Standard/qCork/README.md](core/Standard/qCork/README.md) |
| qCSF | [core/Standard/qCSF/README.md](core/Standard/qCSF/README.md) |
| qFacets | [core/Standard/qFacets/README.md](core/Standard/qFacets/README.md) |
| qHoughNormals | [core/Standard/qHoughNormals/README.md](core/Standard/qHoughNormals/README.md) |
| qJSonRPCPlugin | [core/Standard/qJSonRPCPlugin/README.md](core/Standard/qJSonRPCPlugin/README.md) |
| qM3C2 | [core/Standard/qM3C2/README.md](core/Standard/qM3C2/README.md) |
| qMPlane | [core/Standard/qMPlane/README.md](core/Standard/qMPlane/README.md) |
| qPCL | [core/Standard/qPCL/README.md](core/Standard/qPCL/README.md) |
| qPCV | [core/Standard/qPCV/README.md](core/Standard/qPCV/README.md) |
| qPoissonRecon | [core/Standard/qPoissonRecon/README.md](core/Standard/qPoissonRecon/README.md) |
| qPythonRuntime | [core/Standard/qPythonRuntime/README.md](core/Standard/qPythonRuntime/README.md) |
| qRANSAC_SD | [core/Standard/qRANSAC_SD/README.md](core/Standard/qRANSAC_SD/README.md) |
| qSIBR | [core/Standard/qSIBR/README.md](core/Standard/qSIBR/README.md) |
| qSRA | [core/Standard/qSRA/README.md](core/Standard/qSRA/README.md) |
| qTreeIso | [core/Standard/qTreeIso/README.md](core/Standard/qTreeIso/README.md) |
| qVoxFall | [core/Standard/qVoxFall/README.md](core/Standard/qVoxFall/README.md) |

Submodule plugins (when present): qColorimetricSegmenter, qMasonry, qMPlane, qJSonRPCPlugin, qG3Point.

---

## I/O plugins

| Plugin | README |
|--------|--------|
| qCoreIO | [core/IO/qCoreIO/README.md](core/IO/qCoreIO/README.md) |
| qAdditionalIO | [core/IO/qAdditionalIO/README.md](core/IO/qAdditionalIO/README.md) |
| qCSVMatrixIO | [core/IO/qCSVMatrixIO/README.md](core/IO/qCSVMatrixIO/README.md) |
| qDracoIO | [core/IO/qDracoIO/README.md](core/IO/qDracoIO/README.md) |
| qE57IO | [core/IO/qE57IO/README.md](core/IO/qE57IO/README.md) |
| qFBXIO | [core/IO/qFBXIO/README.md](core/IO/qFBXIO/README.md) |
| qLASIO | [core/IO/qLASIO/README.md](core/IO/qLASIO/README.md) |
| qLASFWFIO | [core/IO/qLASFWFIO/README.md](core/IO/qLASFWFIO/README.md) |
| qMeshIO | [core/IO/qMeshIO/README.md](core/IO/qMeshIO/README.md) |
| qPDALIO | [core/IO/qPDALIO/README.md](core/IO/qPDALIO/README.md) |
| qPhotoscanIO | [core/IO/qPhotoscanIO/README.md](core/IO/qPhotoscanIO/README.md) |
| qRDBIO | [core/IO/qRDBIO/README.md](core/IO/qRDBIO/README.md) |
| qStepCADImport | [core/IO/qStepCADImport/README.md](core/IO/qStepCADImport/README.md) |

---

## See also

- [BUILD.md](../BUILD.md) — full CMake plugin option table
- [Creating a plugin](../docs/source/cpp_api/plugins.rst) — plugin API and scaffolding guide
