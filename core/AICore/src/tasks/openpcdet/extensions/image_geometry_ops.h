// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace openpcdet::ext {

enum class depth_bin_mode {
    uniform,
    linear_increasing,
    spacing_increasing,
};

struct frustum_grid_params {
    std::vector<int32_t> grid_size;        // X, Y, Z
    std::vector<float> point_cloud_range;  // xmin...zmax
    int image_height = 0;
    int image_width = 0;
    int depth_bins = 0;
    float depth_min = 0.0f;
    float depth_max = 0.0f;
    depth_bin_mode bin_mode = depth_bin_mode::uniform;
};

// CaDDN FrustumGridGenerator. Matrices are row-major. Output is
// [X,Y,Z,3] with normalized image-x/image-y/depth coordinates for
// align_corners=true grid sampling. Invalid projections are filled with -2.
bool generate_frustum_grid(const float lidar_to_camera[16],
                           const float camera_to_image[12],
                           const frustum_grid_params& params,
                           std::vector<float>& grid,
                           std::string& error);

struct bev_splat_params {
    std::vector<float> lower_bound;  // x,y,z
    std::vector<float> step;         // x,y,z
    std::vector<int32_t> grid_size;  // x,y,z
    int batch_count = 1;
};

// BEVFusion sum pooling. Geometry is [N,3], features [N,C], and optional
// batch indices are [N] (all zero when null). Output is row-major
// [batch,y,x,z*channels], matching torch.cat(bev.unbind(z), channel).
bool bev_splat_sum(const float* geometry,
                   const float* features,
                   const int32_t* batch_indices,
                   int64_t point_count,
                   int channels,
                   const bev_splat_params& params,
                   std::vector<float>& output,
                   std::string& error);

}  // namespace openpcdet::ext
