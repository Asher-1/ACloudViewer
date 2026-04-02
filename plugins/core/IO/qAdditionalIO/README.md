# qAdditionalIO — Bundler and specialty formats

## Introduction

**qAdditionalIO** adds infrequently used **photogrammetry, polyline, and point-with-meta** formats on top of core I/O. The headline CLI path is **Bundler** import with optional orthorectification-related options.

## Supported formats

| Area | Formats |
|------|---------|
| SfM / photogrammetry | **Bundler** (`*.out`), **ICM** (clouds + calibrated images meta/ascii) |
| Points / polylines | **PN** (point + normal), **POV** (clouds + sensor info), **PV** (point + value) |
| CAD / hydro | **Salome Hydro polylines** (`*.poly`) |
| Other | **SinusX** (`*.sx`), **Mensi Soisic** (`*.soi`) |

## Usage

Use **File → Import** for most filters. For **Bundler**, you can supply an `.out` file and optional alternate keypoints, scaling, undistortion, and colored DTM generation from the CLI.

## ACloudViewer CLI

Bundler import (filename is **required** immediately after `-BUNDLER_IMPORT`):

```bash
ACloudViewer -SILENT -BUNDLER_IMPORT bundler.out \
  [-ALT_KEYPOINTS <file>] [-SCALE_FACTOR <float>] [-UNDISTORT] [-COLOR_DTM <vertices>] ...
```

| Flag | Description |
|------|-------------|
| `-BUNDLER_IMPORT` | Start Bundler import; next argument is the Bundler **filename**. |
| `-ALT_KEYPOINTS <file>` | Alternate keypoints file. |
| `-SCALE_FACTOR <float>` | Scale factor for imported geometry. |
| `-UNDISTORT` | Enable image undistortion when supported. |
| `-COLOR_DTM <vertices>` | Generate colored DTM with the given vertex count. |

Other formats in this plugin are normally opened from the GUI.

## Build

```bash
-DPLUGIN_IO_QADDITIONAL=ON
```

## Dependencies

Uses ACloudViewer core I/O stack only—no extra user-installed runtime beyond a standard build.

## References

- Bundler / Snavely SfM output format (research tools ecosystem).
