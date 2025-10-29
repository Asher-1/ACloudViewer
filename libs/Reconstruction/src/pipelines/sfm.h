// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "controllers/incremental_mapper.h"
#include "optim/bundle_adjustment.h"

namespace cloudViewer {

int AutomaticReconstruct(const std::string& workspace_path,
                         const std::string& image_path,
                         const std::string& mask_path = "",
                         const std::string& vocab_tree_path = "",
                         const std::string& data_type = "individual",
                         const std::string& quality = "high",
                         const std::string& mesher = "poisson",
                         const std::string& camera_model = "SIMPLE_RADIAL",
                         bool single_camera = false,
                         bool sparse = true,
                         bool dense = true,
                         int num_threads = -1,
                         bool use_gpu = true,
                         const std::string& gpu_index = "-1");

int BundleAdjustment(
        const std::string& input_path,
        const std::string& output_path,
        const colmap::BundleAdjustmentOptions& bundle_adjustment_options =
                colmap::BundleAdjustmentOptions());

int ExtractColor(const std::string& image_path,
                 const std::string& input_path,
                 const std::string& output_path);

int NormalMapper(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& input_path,
        const std::string& output_path,
        const std::string& image_list_path = "",
        const colmap::IncrementalMapperOptions& incremental_mapper_options =
                colmap::IncrementalMapperOptions());

int HierarchicalMapper(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& output_path,
        int num_workers = -1,
        int image_overlap = 50,
        int leaf_max_num_images = 500,
        const colmap::IncrementalMapperOptions& incremental_mapper_options =
                colmap::IncrementalMapperOptions());

int FilterPoints(const std::string& input_path,
                 const std::string& output_path,
                 std::size_t min_track_len = 2,
                 double max_reproj_error = 4.0,
                 double min_tri_angle = 1.5);

int TriangulatePoints(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& input_path,
        const std::string& output_path,
        bool clear_points = false,
        const colmap::IncrementalMapperOptions& incremental_mapper_options =
                colmap::IncrementalMapperOptions());

int RigBundleAdjust(
        const std::string& input_path,
        const std::string& output_path,
        const std::string& rig_config_path,
        bool estimate_rig_relative_poses = true,
        bool refine_relative_poses = true,
        const colmap::BundleAdjustmentOptions& bundle_adjustment_options =
                colmap::BundleAdjustmentOptions());

}  // namespace cloudViewer
