# qE57IO — ASTM E57 point cloud format

## Introduction

**qE57IO** reads and writes **ASTM E57** (`.e57`) files used for terrestrial and structured 3D imaging: organized scans, unstructured point clouds, and rich metadata. The implementation is based on **libE57Format**.

## Supported formats

| Format | Notes |
|--------|--------|
| **E57** | ASTM E57-3D imaging format; may include intensity, color, and scan structure. |

## Usage

Import or export `.e57` via **File → Open / Save** or batch workflows. Large coordinates may require a **global shift**; intensity and color can be skipped on import if you need a slimmer cloud.

## ACloudViewer CLI

```bash
ACloudViewer -SILENT -E57 [-GLOBAL_SHIFT <value>] [-IGNORE_INTENSITY] [-IGNORE_COLOR] ...
```

| Flag | Description |
|------|-------------|
| `-E57` | Use the E57 I/O plugin for the following operations. |
| `-GLOBAL_SHIFT <value>` | Global coordinate shift string (required token after the flag—see main app docs for format). |
| `-IGNORE_INTENSITY` | Ignore intensity on import when applicable. |
| `-IGNORE_COLOR` | Ignore color channels on import when applicable. |

## Build

```bash
-DPLUGIN_IO_QE57=ON
```

The plugin builds **libE57Format** from `extern/libE57Format` and links **Xerces-C++** as required by that stack.

## Dependencies

- **libE57Format** — Reference E57 reader/writer (vendored submodule in this repo).
- **Xerces-C++** — XML parser used by the E57 library (`USING_STATIC_XERCES` may apply).

## References

- libE57Format: [https://github.com/asmaloney/libE57Format](https://github.com/asmaloney/libE57Format)
- ASTM E57 standard (3D imaging systems).
