// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "util/ply.h"

namespace colmap {

// Optional fused-point cleanup before Poisson meshing.
// Voxel + SOR logic matches ccPointCloud::VoxelDownSample /
// ccPointCloud::RemoveStatisticalOutliers (libs/CV_db).
struct FusedPointFilterOptions {
    bool enabled = false;

    // Voxel edge length in the same units as the reconstruction (typically
    // meters).
    // <= 0 skips voxel downsampling.
    double voxel_size = 0.02;

    // When true, skip voxel downsampling (e.g. DA3 fusion already voxelized).
    bool skip_voxel_downsample = false;

    bool sor_enabled = true;
    int sor_nb_neighbors = 20;
    double sor_std_ratio = 2.0;

    // Thread count for SOR KNN (-1 = hardware concurrency).
    int num_threads = -1;
};

struct FusedPointCloudWithVisibility {
    std::vector<PlyPoint> points;
    std::vector<std::vector<int>> visibility;
};

std::vector<PlyPoint> VoxelDownsamplePlyPoints(
        const std::vector<PlyPoint>& points, double voxel_size);

std::vector<PlyPoint> StatisticalOutlierRemovalPlyPoints(
        const std::vector<PlyPoint>& points,
        int nb_neighbors,
        double std_ratio);

std::vector<PlyPoint> FilterFusedPlyPoints(
        const std::vector<PlyPoint>& points,
        const FusedPointFilterOptions& options);

FusedPointCloudWithVisibility FilterFusedPlyPointsWithVisibility(
        const FusedPointCloudWithVisibility& cloud,
        const FusedPointFilterOptions& options);

}  // namespace colmap
