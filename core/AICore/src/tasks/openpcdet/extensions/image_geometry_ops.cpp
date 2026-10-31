// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "image_geometry_ops.h"

#include <cmath>

namespace openpcdet::ext {

namespace {

float depth_bin(float depth, const frustum_grid_params& params) {
    if (params.bin_mode == depth_bin_mode::uniform) {
        return (depth - params.depth_min) * params.depth_bins /
               (params.depth_max - params.depth_min);
    }
    if (params.bin_mode == depth_bin_mode::linear_increasing) {
        const float size = 2.0f * (params.depth_max - params.depth_min) /
                           (params.depth_bins * (params.depth_bins + 1.0f));
        return -0.5f +
               0.5f * std::sqrt(1.0f +
                                8.0f * (depth - params.depth_min) / size);
    }
    return params.depth_bins *
           (std::log1p(depth) - std::log1p(params.depth_min)) /
           (std::log1p(params.depth_max) - std::log1p(params.depth_min));
}

}  // namespace

bool generate_frustum_grid(const float lidar_to_camera[16],
                           const float camera_to_image[12],
                           const frustum_grid_params& params,
                           std::vector<float>& grid,
                           std::string& error) {
    grid.clear();
    if (!lidar_to_camera || !camera_to_image || params.grid_size.size() != 3 ||
        params.point_cloud_range.size() != 6 || params.image_height <= 1 ||
        params.image_width <= 1 || params.depth_bins <= 1 ||
        !(params.depth_max > params.depth_min) || params.grid_size[0] <= 0 ||
        params.grid_size[1] <= 0 || params.grid_size[2] <= 0) {
        error = "bad frustum-grid input";
        return false;
    }
    const int nx = params.grid_size[0];
    const int ny = params.grid_size[1];
    const int nz = params.grid_size[2];
    const float voxel[3] = {
            (params.point_cloud_range[3] - params.point_cloud_range[0]) / nx,
            (params.point_cloud_range[4] - params.point_cloud_range[1]) / ny,
            (params.point_cloud_range[5] - params.point_cloud_range[2]) / nz,
    };
    grid.resize(static_cast<size_t>(nx) * ny * nz * 3);
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                const float lidar[4] = {
                        params.point_cloud_range[0] + (x + 0.5f) * voxel[0],
                        params.point_cloud_range[1] + (y + 0.5f) * voxel[1],
                        params.point_cloud_range[2] + (z + 0.5f) * voxel[2],
                        1.0f};
                float camera[4] = {};
                for (int row = 0; row < 4; ++row)
                    for (int column = 0; column < 4; ++column)
                        camera[row] += lidar_to_camera[row * 4 + column] *
                                       lidar[column];
                float image[3] = {};
                for (int row = 0; row < 3; ++row)
                    for (int column = 0; column < 4; ++column)
                        image[row] += camera_to_image[row * 4 + column] *
                                      camera[column];
                float u = image[0] / image[2];
                float v = image[1] / image[2];
                const float depth = image[2] - camera_to_image[11];
                float d = depth_bin(depth, params);
                u = u / (params.image_width - 1.0f) * 2.0f - 1.0f;
                v = v / (params.image_height - 1.0f) * 2.0f - 1.0f;
                d = d / (params.depth_bins - 1.0f) * 2.0f - 1.0f;
                float* destination =
                        grid.data() +
                        ((static_cast<size_t>(x) * ny + y) * nz + z) * 3;
                destination[0] = std::isfinite(u) ? u : -2.0f;
                destination[1] = std::isfinite(v) ? v : -2.0f;
                destination[2] = std::isfinite(d) ? d : -2.0f;
            }
        }
    }
    return true;
}

bool bev_splat_sum(const float* geometry,
                   const float* features,
                   const int32_t* batch_indices,
                   int64_t point_count,
                   int channels,
                   const bev_splat_params& params,
                   std::vector<float>& output,
                   std::string& error) {
    output.clear();
    if ((!geometry && point_count != 0) || (!features && point_count != 0) ||
        point_count < 0 || channels <= 0 || params.lower_bound.size() != 3 ||
        params.step.size() != 3 || params.grid_size.size() != 3 ||
        params.batch_count <= 0) {
        error = "bad BEV-splat input";
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(params.lower_bound[axis]) ||
            !std::isfinite(params.step[axis]) || params.step[axis] <= 0.0f ||
            params.grid_size[axis] <= 0) {
            error = "bad BEV-splat geometry";
            return false;
        }
    }
    const int nx = params.grid_size[0];
    const int ny = params.grid_size[1];
    const int nz = params.grid_size[2];
    output.assign(
            static_cast<size_t>(params.batch_count) * ny * nx * nz * channels,
            0.0f);
    for (int64_t point = 0; point < point_count; ++point) {
        const int batch = batch_indices ? batch_indices[point] : 0;
        int coordinate[3];
        for (int axis = 0; axis < 3; ++axis) {
            coordinate[axis] = static_cast<int>(std::trunc(
                    (geometry[point * 3 + axis] - params.lower_bound[axis]) /
                    params.step[axis]));
        }
        if (batch < 0 || batch >= params.batch_count || coordinate[0] < 0 ||
            coordinate[0] >= nx || coordinate[1] < 0 || coordinate[1] >= ny ||
            coordinate[2] < 0 || coordinate[2] >= nz)
            continue;
        float* destination =
                output.data() +
                (((static_cast<size_t>(batch) * ny + coordinate[1]) * nx +
                  coordinate[0]) *
                         nz +
                 coordinate[2]) *
                        channels;
        for (int channel = 0; channel < channels; ++channel)
            destination[channel] += features[point * channels + channel];
    }
    return true;
}

}  // namespace openpcdet::ext
