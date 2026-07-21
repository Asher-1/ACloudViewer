# qManualCalib Tools

CLI utilities ported from `calibration/apps/tools`, built on `MCALIB_CALIB_IO` and `MCALIB_BEV_STITCH`.

## Build

```bash
cmake -DPLUGIN_STANDARD_QMANUAL_CALIB=ON -DMCALIB_BUILD_TOOLS=ON ..
cmake --build . --target mcalib_rosbag2image mcalib_export_bev
```

`MCALIB_BUILD_EXPORT_BEV_CLI=ON` is kept as an alias for `MCALIB_BUILD_TOOLS`.

With `PLUGIN_STANDARD_QMANUAL_CALIB=ON`, OpenCV is built with `calib3d` and `objdetect` (ArUco) so static detector CLIs are available.

## C++ CLIs

| Binary | Original tool | Description |
|--------|---------------|-------------|
| `mcalib_export_bev` | — | Batch export BEV images from bag |
| `mcalib_rosbag2image` | `rosbag2image_node` | Export compressed camera images to JPG |
| `mcalib_rosbag2pcd` | `rosbag2pcd_node` | Export combined lidar topic to binary PCD |
| `mcalib_rosbag_merge` | `rosbag_merge_node` | Merge multiple bags into `merge.bag` |
| `mcalib_extrinsic_compare` | `extrinsic_compare_node` | Compare camera/lidar extrinsics |
| `mcalib_static_aruco_detect` | `static_aruco_detect_node` | ArUco marker detection (headless) |
| `mcalib_static_chessboard_detect` | `static_chessboard_detect_node` | Chessboard corner detection |

## Python scripts (`scripts/vehicle_config_tools/`)

- `generate_vehicle_config.py`
- `generate_vehicle_config_gui.py`

See `scripts/vehicle_config_tools/README.md` for dependencies.
