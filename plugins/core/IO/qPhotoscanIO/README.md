# qPhotoscanIO — Agisoft PhotoScan / Metashape PSZ

## Introduction

**qPhotoscanIO** reads **Agisoft PhotoScan / Metashape** project archives in **PSZ** form (ZIP-based). Typical content includes cameras, sparse points, and related metadata for alignment with ACloudViewer workflows.

## Supported formats

| Format | Notes |
|--------|--------|
| **PSZ** | Compressed PhotoScan/Metashape project; contents depend on export options in Agisoft. |

## Usage

Open a `.psz` from **File → Open**. Confirm cameras, keypoints, and sparse cloud after import—PSZ contents vary by export settings.

## ACloudViewer CLI

Load the PSZ with `-O`, then pass Photoscan options (order can match your main CLI parser; flags are parsed by the `PHOTOSCAN` command):

```bash
ACloudViewer -SILENT -O project.psz -PHOTOSCAN [-LOAD_KEYPOINTS] [-LOAD_CAMERAS]
```

| Flag | Description |
|------|-------------|
| `-PHOTOSCAN` | Activates Photoscan import options for the current load context. |
| `-LOAD_KEYPOINTS` | Import tie / keypoints when present. |
| `-LOAD_CAMERAS` | Import camera poses and intrinsics when supported. |

## Build

```bash
-DPLUGIN_IO_QPHOTOSCAN=ON
```

## Dependencies

- **QuaZIP** — Built from `extern/quazip` and linked as `quazip` for ZIP access inside PSZ.

## References

- Agisoft Metashape / PhotoScan documentation (project export formats).
