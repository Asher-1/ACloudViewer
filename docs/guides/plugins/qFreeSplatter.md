# qFreeSplatter — FreeSplatter 3D Gaussian Splatting

Turn ordinary photos into **3D Gaussian splatting** point clouds — **no camera poses and no Python required**. Shares `libAICore.so` with qDA3 (ggml inference for [FreeSplatter](https://github.com/TencentARC/FreeSplatter)).

## Workflow

```
Input images (2+) → FreeSplatterDialog → FreeSplatterWorker → libAICore (gaussian_capi)
    → SIBR-compatible PLY → DB tree → [optional] qSIBR Gaussian Viewer
```

## Enable and build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  .

cmake --build build_app --target QFREESPLATTER_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `AICore_ENABLED` | Build `libAICore.so` with FreeSplatter support |
| `PLUGIN_STANDARD_QFREESPLATTER` | This plugin |
| `PLUGIN_STANDARD_QSIBR` | Optional; enables **Visualize (SIBR)** button (runtime invocation, no static link) |
| `PLUGIN_STANDARD_QFREESPLATTER_TOOLS` | Optional CLI `free_splatter-cli` |
| `AICore_BUILD_TESTS` | Unit tests under `core/AICore/tests/gaussian/` |

## GUI usage

**Menu:** Plugins → **FreeSplatter 3D Reconstruction**

| Step | Action |
|------|--------|
| 1 | Choose **Model** type: Scene (2 views) or Object (3+ views) |
| 2 | Select **GGUF model** (F16/F32/Q8_0; auto-download on first run) |
| 3 | **Add Images:** files, folder, or multi-select from DB tree |
| 4 | **Device:** `Auto` / Metal (macOS) / CUDA / OpenCL / CPU |
| 5 | **Run** → export PLY, optionally **Add to DB** |
| 6 | **Visualize** (requires `PLUGIN_STANDARD_QSIBR=ON`) → launch qSIBR Gaussian Viewer |

### Input constraints

| Model | Minimum images | Use case |
|-------|----------------|----------|
| Scene | **2** | Indoor / outdoor scenes |
| Object | **3+** | Single object |

Optional: **Estimate poses** (PnP), **Opacity threshold**, Basic/Full PLY fields.

### Inference device (Auto)

Same as qDA3: macOS **Metal → CUDA → CPU**; Linux/Windows **CUDA → OpenCL → CPU**. Vulkan is disabled in this build.

### Models and cache

Auto-download source: [cloudViewer_downloads/3dgs](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/3dgs)

| Type | Recommended file | Approx. size |
|------|------------------|--------------|
| Scene F16 | `freesplatter-scene-f16.gguf` | ~400 MB |
| Object F16 | `freesplatter-object-f16.gguf` | ~400 MB |

Cache directory: `~/cloudViewer_data/extract/freesplatter_models` (override with `CLOUDVIEWER_DATA_ROOT`).

## Output

- **PLY:** SIBR / 3D Gaussian splatting viewer compatible (OpenGL coordinates; SH, opacity, scale, rotation)
- **DB tree:** point cloud entity with `FS_` prefix and model-type tag (see `ecvPluginDbNaming`)

## qSIBR integration

1. Run FreeSplatter to produce a PLY  
2. Click **Visualize**, or manually: Plugins → SIBR → **3D Gaussian Splatting Viewer**  
3. On macOS, qSIBR may be disabled due to OpenGL limits; you can still export PLY for external viewing

## Tests (optional)

```bash
cmake -B build -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON ...
cmake --build build --target test_loader test_parity
ctest -LE model   # fast tests without GGUF assets
```

## Further reading

- Full plugin README: [`plugins/core/Standard/qFreeSplatter/README.md`](../../../plugins/core/Standard/qFreeSplatter/README.md)
- [FreeSplatter](https://github.com/TencentARC/FreeSplatter) · [free-splatter.cpp](https://github.com/LocalAI-io/free-splatter.cpp)
