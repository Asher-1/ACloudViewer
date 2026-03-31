# qPDALIO — PDAL-based LAS (deprecated)

## Introduction

**qPDALIO** provided **LAS** read/write through **PDAL**. It is **deprecated**; use **qLASIO** for LAS/LAZ in current ACloudViewer builds. This plugin remains only for backward compatibility and is not recommended for new work.

## Supported formats

| Format | Notes |
|--------|--------|
| **LAS** (legacy path) | Via **PDAL** pipelines when the plugin was active; PDAL **≥ 1.6** in CMake. |

## Usage

Prefer migrating to **qLASIO**. If you must keep this plugin, open LAS as before—behavior matches the older PDAL integration, not the modern LASzip stack.

## ACloudViewer CLI

**None.** qPDALIO does not register dedicated silent-mode commands; use **qLASIO** and its `-LAS` flags for batch LAS/LAZ.

## Build

```bash
-DPLUGIN_IO_QPDAL=ON
```

CMake emits a **deprecation** warning when this option is enabled.

## Dependencies

- **PDAL ≥ 1.6** — `find_package(PDAL REQUIRED CONFIG)`.
- On some PDAL versions, **jsoncpp** path via `JSON_ROOT_DIR`.

## References

- PDAL: [https://pdal.io/](https://pdal.io/)
- Migration: disable `PLUGIN_IO_QPDAL`, enable `PLUGIN_IO_QLAS`, re-test pipelines.
