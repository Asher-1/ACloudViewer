# VoxFall (plugin) — Voxel-Based Rockfall Analysis

## Introduction

**VoxFall** performs **voxel-based rockfall detection** and **volumetric analysis** by comparing surfaces (typically meshes) from two epochs in a grid aligned with a chosen **azimuth**, highlighting **loss** and **gain** regions. It is aimed at monitoring rock faces and similar scenes.

## Usage/Algorithm

Two meshes (or compatible entities) representing successive surveys are compared in a voxelized domain. Parameters control voxel size, viewing/azimuth direction, and whether to export result meshes and loss/gain products.

## Parameters

| Option | Role |
|--------|------|
| `-VOXEL_SIZE` | Voxel edge length (scene units) |
| `-AZIMUTH` | Azimuth angle (degrees) |
| `-EXPORT_MESHES` | Export cluster / result meshes |
| `-LOSS_GAIN` | Enable loss and gain analysis |

## Screenshots

![VoxFall dialog](images/voxfall_dialog.jpg)

![VoxFall product](images/voxfall_product.jpg)

## ACloudViewer CLI

Provide **at least two meshes** on the stack (`-O` each file) before `-VOXFALL`:

```bash
ACloudViewer -SILENT -O epoch1.ply -O epoch2.ply -VOXFALL -VOXEL_SIZE 0.1 -AZIMUTH 0 -EXPORT_MESHES -LOSS_GAIN -AUTO_SAVE ON -SAVE_CLOUDS
```

## Build

```bash
-DPLUGIN_STANDARD_QVOXFALL=ON
```

## Dependencies

None beyond the libraries required by the core application and this plugin’s implementation.

## References

- CloudCompare wiki: [VoxFall (plugin)](https://www.cloudcompare.org/doc/wiki/index.php/VoxFall_(plugin))
