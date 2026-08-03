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
  -DBUILD_OPENCV=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  .

cmake --build build_app --target QFREESPLATTER_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `BUILD_OPENCV` | Required for **Face Capture** tab (`HAS_OPENCV_FACE_CAPTURE`) |
| `AICore_ENABLED` | Build `libAICore.so` with FreeSplatter **and** face-detect (Face Capture GGML detector) |
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
| 4 | **Device:** `Auto` / Metal / SYCL / Vulkan (Linux/Windows) / CPU |
| 5 | **Run** → export PLY, optionally **Add to DB** |
| 6 | **Visualize** (requires `PLUGIN_STANDARD_QSIBR=ON`) → launch qSIBR Gaussian Viewer |

### Input constraints

| Model | Minimum images | Use case |
|-------|----------------|----------|
| Scene | **2** | Indoor / outdoor scenes |
| Object | **3+** | Single object |

Optional: **Estimate poses** (PnP), **Opacity threshold**, Basic/Full PLY fields.

### Face Capture tab

Shown when OpenCV is built with **videoio + objdetect** (`BUILD_OPENCV=ON` → `HAS_OPENCV_FACE_CAPTURE` in the plugin). Uses the webcam to capture **five guided face angles** (front, ±45°, ±15° pitch), crops to 512×512, adds them to the input list, and can auto-start reconstruction when the **Object** model is ready.

| Control | Description |
|---------|-------------|
| **Face detector** | **OpenCV Haar Cascade** (bundled cascade, no download) or any **GGML pack** from the AICore face-detect catalog (same list as qFaceDetect, excluding the landmarks-only pack) |
| **Default GGML** | **Buffalo L** when listed; falls back to Haar on first open |
| **Start Camera** | Opens the default webcam and begins guided auto-capture |
| **Reset** | Clears captured frames and restarts the angle sequence |

**Face detector backends**

| Backend | Source | Live preview | Notes |
|---------|--------|--------------|-------|
| **OpenCV Haar** | Plugin resource `haarcascade_frontalface_alt2.xml` | Every frame | Fast, no model download; weaker on profile / low light |
| **GGML (AICore)** | [cloudViewer_downloads qFaceDetect](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/qFaceDetect) | Every 2nd frame (bbox held between runs) | SCRFD / YuNet via `libAICore.so`; auto-download on first use |

GGML detector packs (Buffalo L/M/S/SC, AntelopeV2, YuNet+SFace) are listed by `aicore_facedetect_detector_model_at()` in `core/AICore/include/aicore/facedetect_capi.h`. Cache: `~/cloudViewer_data/extract/facedetect_models` (shared with qFaceDetect). **`PLUGIN_STANDARD_QFACEDETECT` is not required** — Face Capture links AICore directly.

Recommended: **Object** model + **Buffalo L** or **Buffalo SC** detector for tighter face boxes before splatting. For licensing, prefer **YuNet + SFace** (Apache-2.0). Pack details: [qFaceDetect MODEL_CARD](../../../plugins/core/Standard/qFaceDetect/models/MODEL_CARD.md).

### Inference device (Auto)

Same as qDA3: Auto uses macOS **Metal → CPU** or Linux/Windows **Vulkan → CPU**. Vulkan is **unsupported on macOS** (MoltenVK SPIR-V translation limitations cause inference crashes). SYCL and CUDA remain explicit developer devices when built.

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
- Face detector GGUF packs: [qFaceDetect user guide](qFaceDetect.md) · [MODEL_CARD](../../../plugins/core/Standard/qFaceDetect/models/MODEL_CARD.md)
- [FreeSplatter](https://github.com/TencentARC/FreeSplatter) · [free-splatter.cpp](https://github.com/LocalAI-io/free-splatter.cpp)
