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

enum class roi_pool_method {
    max,
    average,
};

struct proposal_result {
    std::vector<int32_t> indices;
    std::vector<float> scores;
    std::vector<int32_t> labels;
};

// OpenPCDet RoIHeadTemplate proposal_layer for one sample. Class scores are
// row-major [box_count, class_count]. Labels are returned one-based.
bool select_proposals(const float* boxes,
                      int64_t box_count,
                      const float* class_scores,
                      int class_count,
                      bool scores_are_normalized,
                      int pre_max,
                      int post_max,
                      float nms_threshold,
                      proposal_result& output,
                      std::string& error);

// Boxes use OpenPCDet's [x, y, z, dx, dy, dz, heading] convention. The
// returned row-major mask is [box_count, point_count].
bool points_in_boxes(const float* points,
                     int64_t point_count,
                     const float* boxes,
                     int64_t box_count,
                     std::vector<uint8_t>& mask,
                     std::string& error);

// GPU-accelerated points_in_boxes. Falls back to error if CUDA not compiled.
bool points_in_boxes_cuda(const float* points,
                          int64_t point_count,
                          const float* boxes,
                          int64_t box_count,
                          std::vector<uint8_t>& mask,
                          std::string& error);

// Inference-only counterpart of RoIAwarePool3d. Features are [point_count, C]
// and output is [box_count, out_x, out_y, out_z, C]. Points are scanned in
// input order and each output voxel accepts at most max_points_per_voxel.
bool roi_aware_pool3d(const float* boxes,
                      int64_t box_count,
                      const float* points,
                      int64_t point_count,
                      const float* features,
                      int channels,
                      int out_x,
                      int out_y,
                      int out_z,
                      int max_points_per_voxel,
                      roi_pool_method method,
                      std::vector<float>& output,
                      std::string& error);

// Generate the dense grid used by PVRCNN RoI-grid pooling. Output is
// [box_count, grid_size^3, 3], in x/y/z lexicographic order.
bool roi_grid_points(const float* boxes,
                     int64_t box_count,
                     int grid_size,
                     std::vector<float>& local_points,
                     std::vector<float>& global_points,
                     std::string& error);

// SECONDHead rotated BEV pooling. Input features are row-major [H,W,C]. The
// output uses PyTorch NCHW flattening order [box,C,grid_y,grid_x] and exactly
// matches affine_grid/grid_sample with align_corners=true and zero padding.
bool rotated_bev_roi_align(const float* features,
                           int height,
                           int width,
                           int channels,
                           const float* boxes,
                           int64_t box_count,
                           int grid_size,
                           float min_x,
                           float min_y,
                           float voxel_size_x,
                           float voxel_size_y,
                           int downsample_ratio,
                           std::vector<float>& output,
                           std::string& error);

// PointRCNN RoIPointPool3d inference contract. Each RoI receives exactly
// sample_count points in input order; short sets repeat cyclically and empty
// RoIs are zero-filled. Output is [box_count, sample_count, 3 + channels].
bool roi_point_pool3d(const float* boxes,
                      int64_t box_count,
                      const float* points,
                      int64_t point_count,
                      const float* features,
                      int channels,
                      const float extra_width[3],
                      int sample_count,
                      std::vector<float>& output,
                      std::vector<uint8_t>& empty_flags,
                      std::string& error);

// Convert the xyz prefix of pooled RoI points from global coordinates into
// each proposal's canonical frame, matching PointRCNNHead.roipool3d_gpu.
bool canonicalize_roi_points(const float* boxes,
                             int64_t box_count,
                             int sample_count,
                             int feature_width,
                             std::vector<float>& pooled,
                             std::string& error);

// Decode class-agnostic seven-parameter RoI residuals and transform their xyz
// coordinates back to the global frame.
bool decode_roi_residuals(const float* rois,
                          const float* residuals,
                          int64_t roi_count,
                          std::vector<float>& boxes,
                          std::string& error);

}  // namespace openpcdet::ext
