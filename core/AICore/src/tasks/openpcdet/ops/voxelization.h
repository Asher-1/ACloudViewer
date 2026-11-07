// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CPU voxelization for PointPillar-style detectors. Replicates the behaviour
// of OpenPCDet's VoxelGenerator (spconv.utils.VoxelGenerator) at inference
// time:
//   - points outside point_cloud_range are dropped
//   - points are binned into voxels by floor((p - range_min) / voxel_size)
//   - each voxel holds up to max_points_per_voxel (random early points kept)
//   - up to max_voxels voxels are returned (voxels with < min_points are
//   dropped)
//   - voxel_coords are packed as [batch_idx, z, y, x] to match OpenPCDet
//     (here batch_idx is always 0 — the engine processes one scene at a time)
//
// Output is dense [max_voxels, max_points_per_voxel, point_dim] float32 (padded
// with zeros) plus [max_voxels] int32 num_points and [max_voxels, 4] int32
// coords. The caller trims to actual_voxels.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace openpcdet {

struct voxelization_params {
    std::vector<float> point_cloud_range;  // [x0,y0,z0,x1,y1,z1]
    std::vector<float> voxel_size;         // [vx,vy,vz]
    std::vector<int32_t> grid_size;        // [nx,ny,nz]
    int32_t max_points_per_voxel = 32;
    int32_t max_voxels = 16000;
    int32_t min_points_per_voxel = 1;
};

struct voxel_output {
    // [actual_voxels * max_points_per_voxel * point_dim] row-major
    std::vector<float> voxels;        // dense, zero-padded
    std::vector<int32_t> num_points;  // [actual_voxels]
    std::vector<int32_t> coords;      // [actual_voxels * 4] : [batch,z,y,x]
    int32_t actual_voxels = 0;
};

struct mean_voxel_output {
    std::vector<float> features;  // [actual_voxels, point_dim]
    std::vector<int32_t> coords;  // [actual_voxels, 4] : [batch,z,y,x]
    int32_t actual_voxels = 0;
};

// Returns false on bad params. points is [n_points * point_dim] row-major.
bool voxelize(const voxelization_params& p,
              const float* points,
              int32_t n_points,
              int32_t point_dim,
              voxel_output& out,
              std::string& err);

// Voxelize and compute MeanVFE online. This preserves voxel/point insertion
// order while avoiding the dense [max_voxels,max_points,point_dim] staging
// tensor when the caller only needs per-voxel means.
bool voxelize_mean(const voxelization_params& p,
                   const float* points,
                   int32_t n_points,
                   int32_t point_dim,
                   mean_voxel_output& out,
                   std::string& err);

}  // namespace openpcdet
