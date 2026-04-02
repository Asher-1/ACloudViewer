# Point Cloud Library Wrapper (plugin)

## Introduction

qPCL is a simple interface to some methods of the [PCL](http://pointclouds.org/) library.

Therefore, if you use this tool for a scientific publication, please cite the PCL project before citing CloudCompare (which is also very good but less important in this particular case ;).

CloudCompare simply adds dialogs to set some parameters (see below) and a seamless integration in its own workflow.

This plugin has been developed by Luca Penasa (University of Padova).

![PCL](images/pcl.png)

## Available methods

The following methods are currently accessible through this interface:

- **Estimate Normals and Curvature** — to compute the normals (and optionally the curvature) of a point cloud
- **[Fast Global Registration](https://www.cloudcompare.org/doc/wiki/index.php/Fast_Global_Registration)** — to register point clouds (with normals) with no initial/rough alignment
- **[Statistical Outliers Removal](https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html)** — cleaning filter (the [SOR filter](https://www.cloudcompare.org/doc/wiki/index.php/SOR_filter) is also integrated into the standalone version)
- **[MLS (Moving Least Squares) smoothing / upsampling](https://pcl.readthedocs.io/projects/tutorials/en/latest/resampling.html)** — to smooth (and optionally to upsample) a point cloud

## ACloudViewer CLI

The qPCL plugin registers many CLI commands, each corresponding to a PCL algorithm:

| Command | Description | Key Options |
|---------|-------------|-------------|
| `-PCL_SOR` | Statistical outlier removal | `-K` (neighbors), `-STD` (std dev multiplier) |
| `-PCL_NORMAL_ESTIMATION` | Normal estimation | `-KNN` (neighbors), `-RADIUS` (search radius) |
| `-PCL_MLS` | Moving Least Squares smoothing | `-SEARCH_RADIUS`, `-ORDER`, `-COMPUTE_NORMALS` |
| `-PCL_EUCLIDEAN_CLUSTER` | Euclidean clustering | `-TOLERANCE`, `-MIN_SIZE`, `-MAX_SIZE` |
| `-PCL_SAC_SEGMENTATION` | SAC model segmentation | `-MODEL`, `-METHOD`, `-DIST_THRESH`, `-MAX_ITER` |
| `-PCL_REGION_GROWING` | Region growing segmentation | `-SMOOTHNESS`, `-NEIGHBORS` |
| `-PCL_GREEDY_TRIANGULATION` | Greedy triangulation meshing | (various triangle params) |
| `-PCL_POISSON_RECON` | Poisson surface reconstruction | `-DEPTH`, `-SCALE`, `-SAMPLES_PER_NODE` |
| `-PCL_MARCHING_CUBES` | Marching cubes meshing | `-GRID_RES`, `-ISO_LEVEL` |
| `-PCL_CONVEX_HULL` | Convex hull computation | `-ALPHA`, `-DIMENSION` |
| `-PCL_DON_SEGMENTATION` | Difference of Normals | `-SMALL_SCALE`, `-LARGE_SCALE`, `-MIN_DON`, `-MAX_DON` |
| `-PCL_MINCUT_SEGMENTATION` | Min-cut segmentation | `-FX`, `-FY`, `-FZ`, `-SIGMA`, `-BACK_RADIUS` |
| `-PCL_FAST_GLOBAL_REGISTRATION` | Fast global registration | `-VOXEL_LEAF`, `-NORMAL_RADIUS`, `-MAX_CORR_DIST` |
| `-PCL_EXTRACT_SIFT` | SIFT keypoint extraction | `-MODE`, `-OCTAVES`, `-MIN_SCALE`, `-MIN_CONTRAST` |
| `-PCL_PROJECTION_FILTER` | Projection filter | `-A`, `-B`, `-C`, `-D` (plane coefficients) |
| `-PCL_GENERAL_FILTERS` | Pass-through / voxel grid / SOR | `-MIN`, `-MAX`, `-LEAF` (voxel size) |
| `-PCL_TEMPLATE_ALIGNMENT` | Template alignment | `-FEATURE_RADIUS`, `-REF_INDEX` |
| `-PCL_CORRESPONDENCE_MATCHING` | Correspondence matching | `-MODEL_RADIUS`, `-SCENE_RADIUS`, `-GC` |

### Example

```bash
# Statistical outlier removal
ACloudViewer -SILENT -O noisy_cloud.ply -PCL_SOR -K 50 -STD 1.0 -SAVE_CLOUDS

# Normal estimation
ACloudViewer -SILENT -O cloud.ply -PCL_NORMAL_ESTIMATION -KNN 30 -SAVE_CLOUDS

# MLS smoothing
ACloudViewer -SILENT -O cloud.ply -PCL_MLS -SEARCH_RADIUS 0.05 -COMPUTE_NORMALS -SAVE_CLOUDS
```

## Build

```cmake
-DPLUGIN_STANDARD_QPCL=ON
```

## Dependencies

- [PCL](http://pointclouds.org/) (Point Cloud Library) — follow the instructions from their website.

## References

- PCL: [pointclouds.org](http://pointclouds.org/)
- CloudCompare wiki: [Point Cloud Library Wrapper (plugin)](https://www.cloudcompare.org/doc/wiki/index.php/Point_Cloud_Library_Wrapper_(plugin))
