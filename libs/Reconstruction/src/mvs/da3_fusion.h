// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "util/ply.h"

namespace colmap {
namespace mvs {

struct DA3FusionOptions {
    // Minimum fused pixels in a COLMAP-style consistency group.
    int min_num_pixels = 3;

    // Minimum distinct views visible in a fused point.
    int min_num_views = 2;

    // Sample every N pixels (1 = full resolution, matches COLMAP).
    int pixel_stride = 1;

    // Voxel size for DA3 consensus fusion (m). This is NOT the UI "Fused point
    // filter" voxel; it controls how multi-view samples merge inside
    // FuseDA3DepthMaps.
    double fusion_voxel_size = 0.006;

    // Match COLMAP stereo fusion max_image_size (-1 = full resolution).
    int max_image_size = -1;

    // Maximum reprojection error in pixels.
    double max_reproj_error = 4.0;

    // Maximum relative depth error |proj_z - depth| / depth.
    double max_depth_error = 0.04;

    // Maximum 3D disagreement between unprojections from different views (m).
    double max_point_dist = 0.025;

    // Maximum angular difference between normals in degrees.
    double max_normal_error = 15.0;

    // Maximum graph traversal depth (same semantics as StereoFusion).
    int max_traversal_depth = 100;

    // Maximum number of other views to traverse per fused group.
    int check_num_images = 50;

    // Maximum fused pixels per point (same role as
    // StereoFusion.max_num_pixels).
    int max_num_pixels = 10000;

    // Fusion worker threads (-1 = auto).
    int num_threads = -1;

    // When true, consistency checks only traverse views that overlap the
    // reference (reduces false rejects from non-overlapping cameras).
    bool restrict_to_overlapping_views = true;

    // Accumulate consistent samples into voxels (one point per voxel). Avoids
    // per-view duplicate shells that cause ghost layers in dense output mode.
    bool use_voxel_consensus = true;

    // Legacy: emit one point per pixel (causes multi-view ghosting). Ignored
    // when use_voxel_consensus=true.
    bool use_dense_fusion = false;
};

struct DA3FusionResult {
    std::vector<PlyPoint> points;
    // Per-point visible mvs::Model image indices (for fused.ply.vis).
    std::vector<std::vector<int>> visibility;
    // Stats for adaptive direct-fusion quality checks (skip-geometric path).
    size_t num_valid_depth_pixels = 0;
    size_t num_accepted_samples = 0;
    size_t num_skipped_samples = 0;
};

// Fuse DA3 depth maps using COLMAP-style graph traversal with DA3-tuned
// consistency thresholds (bilinear depth sampling, voxel consensus).
// input_type: "geometric" (.geometric.bin) or "photometric" (.photometric.bin).
DA3FusionResult FuseDA3DepthMaps(const std::string& dense_workspace_path,
                                 const DA3FusionOptions& options,
                                 const std::string& input_type = "geometric");

}  // namespace mvs
}  // namespace colmap
