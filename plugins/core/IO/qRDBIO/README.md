# qRDBIO — RIEGL RDB 2

## Introduction

**qRDBIO** loads **RIEGL DataBase 2 (RDB 2)** point cloud data produced by RIEGL sensors and workflows. Access to the **RDB SDK** is subject to RIEGL’s member download terms.

## Supported formats

| Format | Notes |
|--------|--------|
| **RDB 2** | RIEGL’s database format for point cloud storage and attributes. |

## Usage

Open RDB files through **File → Open** when the plugin and SDK are installed. Attribute and schema details follow RIEGL’s RDB specification for your SDK version.

## ACloudViewer CLI

**None** — use the GUI or higher-level automation that loads files by filter; there are no dedicated `-SILENT` tokens for RDB in the standard command set.

## Build

```bash
-DPLUGIN_IO_QRDB=ON -Drdb_DIR=/path/to/rdblib-.../interface/cpp
```

`rdb_DIR` must point to the directory containing **`rdb-config.cmake`**. Optional: `PLUGIN_IO_QRDB_FETCH_DEPENDENCY` to download the SDK during configure (see plugin `CMakeLists.txt`).

## Dependencies

- **RIEGL RDB SDK** — Obtain from the [RIEGL Members Area](https://www.riegl.com/) after registration.

## References

- RIEGL RDB documentation and SDK (vendor portal).
