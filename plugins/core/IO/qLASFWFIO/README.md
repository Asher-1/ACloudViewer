# qLASFWFIO — LAS full waveform (legacy)

## Introduction

**qLASFWFIO** handles **LAS 1.3+** point clouds with **full waveform (FWF)** payload: waveform packets attached to returns for analysis beyond discrete points.

> **Deprecated** — Prefer **qLASIO** for current LAS/LAZ workflows. This plugin exists for legacy compatibility; CMake in this repository currently enables it **only on Windows** and shows a deprecation warning.

## Supported formats

| Format | Notes |
|--------|--------|
| **LAS / LAZ with FWF** | Discrete points plus waveform data per LAS 1.3+ conventions. |

## Usage

Open waveform-capable LAS through the file dialog or `-FWF_O`. Saving clouds with waveform data uses `-FWF_SAVE_CLOUDS` with optional `ALL_AT_ONCE` and `COMPRESSED` (LAZ) modes—see CLI below.

## ACloudViewer CLI

**Load** (optional global shift may precede the filename, consistent with other loaders):

```bash
ACloudViewer -SILENT -FWF_O [-GLOBAL_SHIFT ...] <file.las>
```

**Save** clouds using the LAS FWF writer:

```bash
ACloudViewer -SILENT -FWF_SAVE_CLOUDS [ALL_AT_ONCE] [COMPRESSED] ...
```

| Token | Description |
|-------|-------------|
| `-FWF_O` | Load using the LAS FWF filter. |
| `-FWF_SAVE_CLOUDS` | Save using LAS FWF export; may follow with `ALL_AT_ONCE` and/or `COMPRESSED` (LAZ). |

## Build

```bash
-DPLUGIN_IO_QLAS_FWF=ON
```

**Platform:** the supplied `CMakeLists.txt` wraps the plugin in `if (WIN32)`; non-Windows builds skip it unless you extend the project.

## Dependencies

- **LASlib** / **LASzip** — Fetched or supplied per `cmake/LasLibZip_download.cmake` and linked as **LASlib** on Windows.
- Includes headers from **qPDALIO** in this tree (`target_include_directories`).

## References

- ASPRS LAS specification (full waveform sections).
- Prefer **qLASIO** README for maintained LAS/LAZ guidance.
