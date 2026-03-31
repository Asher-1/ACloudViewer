# qSIBR — SIBR Framework Integration

## Introduction

**qSIBR** integrates the **[SIBR](https://sibr.gitlabpages.inria.fr/)** framework for **image-based and neural rendering** inside ACloudViewer, including:

- **Real-time 3D Gaussian Splatting** (CUDA where available)
- **ULR** and **ULR v2** novel-view synthesis
- **Textured mesh** and **point-based** viewing
- **Remote Gaussian** training viewer
- **Dataset preprocessing tools** (COLMAP preparation, unwrap, tone mapping, mesh tools, etc.)

Viewers and preprocessing tools can work **bidirectionally** with ACloudViewer (export scenes, inspect results, round-trip with the main DB).

## Usage/Algorithm

- Use **`-SIBR_VIEWER`** to launch an interactive viewer: pass the **viewer name**, then **viewer-specific options** (`--path`, `--model-path`, `--width`, `--height`, `--device`, etc.).
- Use **`-SIBR_TOOL`** to run an embedded **dataset tool** by name, followed by that tool’s arguments (often after `--`).

## Parameters

### `-SIBR_VIEWER`

| Viewer name | Notes |
|-------------|--------|
| `ulr` | Unstructured Lumigraph Rendering |
| `ulrv2` | ULR v2/v3 with texture arrays |
| `texturedmesh` | Textured mesh viewer |
| `pointbased` | Point-based rendering |
| `gaussian` | 3D Gaussian Splatting (**CUDA**; requires `--model-path`) |
| `remoteGaussian` | Remote training viewer |

Common options (after the viewer name): `--path`, `--model-path`, `--width`, `--height`, `--iteration`, `--device`, `--no-interop`, `--ip`, `--port`.

### `-SIBR_TOOL`

Embedded tools include: `prepareColmap4Sibr`, `tonemapper`, `unwrapMesh`, `textureMesh`, `clippingPlanes`, `cropFromCenter`, `nvmToSIBR`, `distordCrop`, `cameraConverter`, `alignMeshes`.

## Screenshots

![SIBR plugin](images/sibr_plugin.png)

## ACloudViewer CLI

**Viewers**

```bash
ACloudViewer -SILENT -SIBR_VIEWER ulr --path /data/colmap_scene --width 1280 --height 720
ACloudViewer -SILENT -SIBR_VIEWER gaussian --path /data/scene --model-path /data/gaussian_out --device 0
```

**Dataset tools**

```bash
ACloudViewer -SILENT -SIBR_TOOL prepareColmap4Sibr -- /path/to/colmap
```

Pass tool-specific flags after the tool name; stop before the next ACloudViewer token that starts with `-` and an uppercase letter (see `qSIBRCommands.h`).

## Build

```bash
-DPLUGIN_STANDARD_QSIBR=ON
```

**Note:** The full SIBR stack (especially CUDA Gaussian viewers) is **not supported on macOS** in typical configurations; prefer Linux or Windows for complete functionality.

## Dependencies

**Boost**, **OpenCV**, **GLEW**, **GLFW**, **Assimp**; **CUDA** for Gaussian splatting viewers where applicable.

## References

- [SIBR documentation](https://sibr.gitlabpages.inria.fr/)
- Plugin sources: `plugins/core/Standard/qSIBR/`
