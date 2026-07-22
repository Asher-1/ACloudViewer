# AI inference plugins (AICore)

ACloudViewer provides two AI plugins through the unified inference library **`libAICore.so`** (ggml). **No Python/PyTorch runtime is required.**

| Plugin | Guide | CMake option | Summary |
|--------|-------|--------------|---------|
| **qDA3** | [qDA3 user guide](qDA3.md) | `PLUGIN_STANDARD_QDA3` | Depth Anything V3: monocular/multi-view depth, camera pose, COLMAP/GLB export, automatic reconstruction integration |
| **qFreeSplatter** | [qFreeSplatter user guide](qFreeSplatter.md) | `PLUGIN_STANDARD_QFREESPLATTER` | FreeSplatter: uncalibrated photos → 3D Gaussian splats, SIBR-compatible PLY, optional qSIBR preview |

## Prerequisites

- `-DAICore_ENABLED=ON` (build `core/AICore` → `libAICore.so`)
- GUI: `-DBUILD_GUI=ON`
- **qDA3 + automatic reconstruction:** `-DBUILD_RECONSTRUCTION=ON`
- **FreeSplatter one-click Visualize:** `-DPLUGIN_STANDARD_QSIBR=ON` (Linux/Windows; macOS CI disables qSIBR by default)
- **GPU acceleration (recommended):** `-DBUILD_CUDA_MODULE=ON`; optional ggml OpenCL/Metal (see each plugin guide)

## Typical build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDA3=ON \
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  -DBUILD_RECONSTRUCTION=ON \
  .

cmake --build build_app --target ACloudViewer QDA3_PLUGIN QFREESPLATTER_PLUGIN -j$(nproc)
```

## More resources

- Full plugin READMEs (developer details, tests, C API): [`plugins/core/Standard/qDA3/README.md`](../../../plugins/core/Standard/qDA3/README.md), [`plugins/core/Standard/qFreeSplatter/README.md`](../../../plugins/core/Standard/qFreeSplatter/README.md)
- Plugin catalog: [`plugins/README.md`](../../../plugins/README.md)
- Sphinx doc build syncs the above READMEs into `docs/source/plugins/` (see `docs/source/conf.py`)
