# qLASIO — LAS / LAZ point clouds

## Introduction

**qLASIO** adds read and write support for **ASPRS LAS** and compressed **LAZ** point cloud files using **LASzip**. It is the preferred LAS/LAZ path in ACloudViewer (replacing older PDAL- or LASlib-only workflows for typical discrete-point use).

## Supported formats

| Format | Notes |
|--------|--------|
| **LAS** | ASPRS Lidar Data Exchange Format (multiple point record formats). |
| **LAZ** | LASzip-compressed LAS. |

## Usage

- **Open / save** LAS and LAZ from the **File** menu or drag-and-drop when the plugin is installed.
- **Extra dimensions** and **LAZ** output options are available in the save dialog where the cloud and build support them.
- For **full waveform** lidar data, prefer modern LAS support in **qLASIO** where available; the legacy **qLASFWFIO** plugin is deprecated and platform-limited in this tree.

## ACloudViewer CLI

Use with `-SILENT` and the **`LAS`** command group:

```bash
ACloudViewer -SILENT -LAS [-EXTRA_FIELDS] [-TILE_SIZE <n>] [-SAVE_LAZ] [-LAS_VERSION <ver>] ...
```

| Flag | Description |
|------|-------------|
| `-LAS` | Use the LAS/LAZ I/O pipeline for subsequent file operations. |
| `-EXTRA_FIELDS` | Enable handling of extra bytes / extra dimensions where supported. |
| `-TILE_SIZE <n>` | Tile size for applicable operations (see console output for validation). |
| `-SAVE_LAZ` | Prefer LAZ when saving (when the writer path supports it). |
| `-LAS_VERSION <ver>` | Target LAS version for export when applicable. |

Exact behavior follows the implementation in `LasCommands.cpp` and your build’s LASzip version.

## Build

```bash
-DPLUGIN_IO_QLAS=ON
```

Ensure **LASzip** is discoverable by CMake (`find_package(LASzip)` or `pkg_check_modules(laszip)`).

## Dependencies

- **LASzip** — Compression and LAS 1.4+ LAZ support.

## References

- LASzip: [https://laszip.org/](https://laszip.org/)
- ASPRS LAS specification (industry reference for LAS/LAZ fields).
