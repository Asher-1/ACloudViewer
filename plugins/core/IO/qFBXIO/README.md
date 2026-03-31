# qFBXIO — Autodesk FBX

## Introduction

**qFBXIO** imports and exports **Autodesk FBX** scenes: meshes, cameras, lights, and animation to the extent supported by the **FBX SDK** and this integration. Binary FBX is common for size; ASCII can help debugging.

## Supported formats

| Format | Notes |
|--------|--------|
| **FBX** | Autodesk interchange (binary or ASCII per SDK and export settings). |

## Usage

Use **File → Open / Save** for FBX. Feature coverage (skins, materials, constraints) depends on SDK version and how entities map into ACloudViewer—validate with a small test scene before large batch jobs.

## ACloudViewer CLI

```bash
ACloudViewer -SILENT -FBX [-EXPORT_FMT <fmt>] ...
```

| Flag | Description |
|------|-------------|
| `-FBX` | Use the FBX I/O plugin for export/import routing. |
| `-EXPORT_FMT <fmt>` | Sets the default FBX output format string consumed by `FBXFilter::SetDefaultOutputFormat` (accepted values depend on the SDK integration—check logs or source for your build). |

## Build

```bash
-DPLUGIN_IO_QFBX=ON
```

Configure **FBX SDK** paths (`FBX_SDK_INCLUDE_DIR`, library paths, optional `FBX_SDK_SHARED_MODE`). The CMake script can download or locate the SDK per `cmake/FBX_download.cmake`.

## Dependencies

- **Autodesk FBX SDK** — License and redistribution terms apply per Autodesk requirements.

## References

- FBX SDK documentation (Autodesk developer network).
