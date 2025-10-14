// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "exe/sfm.h"

#include "pipelines/option_utils.h"
#include "pipelines/sfm.h"

namespace cloudViewer {

int AutomaticReconstruct(const std::string& workspace_path,
                         const std::string& image_path,
                         const std::string& mask_path /*= ""*/,
                         const std::string& vocab_tree_path /*= ""*/,
                         const std::string& data_type /*= "individual"*/,
                         const std::string& quality /*= "high"*/,
                         const std::string& mesher /*= "poisson"*/,
                         const std::string& camera_model /*= "SIMPLE_RADIAL"*/,
                         bool single_camera /*= false*/,
                         bool sparse /*= true*/,
                         bool dense /*= true*/,
                         int num_threads /*= -1*/,
                         bool use_gpu /*= true*/,
                         const std::string& gpu_index /*= "-1"*/) {
#ifdef CUDA_ENABLED
    dense = true;
#else
    dense = false;
#endif

    OptionsParser parser;
    parser.registerOption("workspace_path", &workspace_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("mask_path", &mask_path);
    parser.registerOption("vocab_tree_path", &vocab_tree_path);
    // supported {individual, video, internet}
    parser.registerOption("data_type", &data_type);
    // supported {low, medium, high, extreme}
    parser.registerOption("quality", &quality);
    parser.registerOption("mesher", &mesher);
    parser.registerOption("camera_model", &camera_model);
    parser.registerOption("single_camera", &single_camera);
    parser.registerOption("sparse", &sparse);
    parser.registerOption("dense", &dense);
    parser.registerOption("num_threads", &num_threads);
    parser.registerOption("use_gpu", &use_gpu);
    parser.registerOption("gpu_index", &gpu_index);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunAutomaticReconstructor(parser.getArgc(),
                                             parser.getArgv());
}

int BundleAdjustment(
        const std::string& input_path,
        const std::string& output_path,
        const colmap::BundleAdjustmentOptions& bundle_adjustment_options) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.addBundleAdjustmentOptions(bundle_adjustment_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunBundleAdjuster(parser.getArgc(), parser.getArgv());
}

int ExtractColor(const std::string& image_path,
                 const std::string& input_path,
                 const std::string& output_path) {
    OptionsParser parser;
    parser.registerOption("image_path", &image_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunColorExtractor(parser.getArgc(), parser.getArgv());
}

int NormalMapper(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& input_path,
        const std::string& output_path,
        const std::string& image_list_path /*= ""*/,
        const colmap::IncrementalMapperOptions& incremental_mapper_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("image_list_path", &image_list_path);
    parser.addMapperOptions(incremental_mapper_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunMapper(parser.getArgc(), parser.getArgv());
}

int HierarchicalMapper(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& output_path,
        int num_workers /*= -1*/,
        int image_overlap /*= 50*/,
        int leaf_max_num_images /*= 500*/,
        const colmap::IncrementalMapperOptions& incremental_mapper_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("num_workers", &num_workers);
    parser.registerOption("image_overlap", &image_overlap);
    parser.registerOption("leaf_max_num_images", &leaf_max_num_images);
    parser.addMapperOptions(incremental_mapper_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunHierarchicalMapper(parser.getArgc(), parser.getArgv());
}

int FilterPoints(const std::string& input_path,
                 const std::string& output_path,
                 std::size_t min_track_len /*= 2*/,
                 double max_reproj_error /*= 4.0*/,
                 double min_tri_angle /*= 1.5*/) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("min_track_len", &min_track_len);
    parser.registerOption("max_reproj_error", &max_reproj_error);
    parser.registerOption("min_tri_angle", &min_tri_angle);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunPointFiltering(parser.getArgc(), parser.getArgv());
}

int TriangulatePoints(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& input_path,
        const std::string& output_path,
        bool clear_points /*= false*/,
        const colmap::IncrementalMapperOptions& incremental_mapper_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    // Whether to clear all existing points and observations
    parser.registerOption("clear_points", &clear_points);
    parser.addMapperOptions(incremental_mapper_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunPointTriangulator(parser.getArgc(), parser.getArgv());
}

int RigBundleAdjust(
        const std::string& input_path,
        const std::string& output_path,
        const std::string& rig_config_path,
        bool estimate_rig_relative_poses /*= true*/,
        bool refine_relative_poses /*= true*/,
        const colmap::BundleAdjustmentOptions& bundle_adjustment_options) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.registerOption("rig_config_path", &rig_config_path);
    // Whether to optimize the relative poses of the camera rigs.
    parser.registerOption("estimate_rig_relative_poses",
                          &estimate_rig_relative_poses);
    parser.registerOption("RigBundleAdjustment.refine_relative_poses",
                          &refine_relative_poses);
    parser.addBundleAdjustmentOptions(bundle_adjustment_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunRigBundleAdjuster(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer
