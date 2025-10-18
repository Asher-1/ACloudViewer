// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pipelines/option_utils.h"

namespace cloudViewer {

OptionsParser::OptionsParser() {
    argc_ = 0;
    argv_ = nullptr;
    reset();
}

bool OptionsParser::parseOptions(int& argc, char**& argv) {
    // First, put all options without a section and then those with a
    // section. This is necessary as otherwise older Boost versions will
    // write the options without a section in between other sections and
    // therefore the errors will be assigned to the wrong section if read
    // later.

    ReleaseOptions(argc, argv);
    unsigned long capacity = 2 * getParametersCount() + 1;
    if (capacity == 0) {
        return false;
    }

    // add application name
    argv = new char*[capacity];
    SetValue("options", 0, argv);
    argc = 1;

    // add other optinons
    for (const auto& option : options_bool_) {
        SetValue("--" + option.first, argc, argv);
        argc += 1;
        bool bool_flag = *option.second;
        if (bool_flag)
            SetValue("true", argc, argv);
        else
            SetValue("false", argc, argv);
        argc += 1;
    }

    for (const auto& option : options_int_) {
        SetValue("--" + option.first, argc, argv);
        argc += 1;
        SetValue(std::to_string(*option.second), argc, argv);
        argc += 1;
    }

    for (const auto& option : options_double_) {
        SetValue("--" + option.first, argc, argv);
        argc += 1;
        SetValue(std::to_string(*option.second), argc, argv);
        argc += 1;
    }

    for (const auto& option : options_string_) {
        SetValue("--" + option.first, argc, argv);
        argc += 1;
        SetValue(*option.second, argc, argv);
        argc += 1;
    }

    if (argc == 0 || !argv) return false;

    return true;
}

void OptionsParser::reset() {
    releaseOptions();
    options_bool_.clear();
    options_int_.clear();
    options_double_.clear();
    options_string_.clear();
}

void OptionsParser::addExtractionOptions(
        const colmap::ImageReaderOptions& image_reader_options,
        const colmap::SiftExtractionOptions& sift_extraction_options) {
    registerOption("ImageReader.mask_path", &image_reader_options.mask_path);
    registerOption("ImageReader.camera_model",
                   &image_reader_options.camera_model);
    registerOption("ImageReader.single_camera",
                   &image_reader_options.single_camera);
    registerOption("ImageReader.single_camera_per_folder",
                   &image_reader_options.single_camera_per_folder);
    registerOption("ImageReader.single_camera_per_image",
                   &image_reader_options.single_camera_per_image);
    registerOption("ImageReader.existing_camera_id",
                   &image_reader_options.existing_camera_id);
    registerOption("ImageReader.camera_params",
                   &image_reader_options.camera_params);
    registerOption("ImageReader.default_focal_length_factor",
                   &image_reader_options.default_focal_length_factor);
    registerOption("ImageReader.camera_mask_path",
                   &image_reader_options.camera_mask_path);

    registerOption("SiftExtraction.num_threads",
                   &sift_extraction_options.num_threads);
    registerOption("SiftExtraction.use_gpu", &sift_extraction_options.use_gpu);
    registerOption("SiftExtraction.gpu_index",
                   &sift_extraction_options.gpu_index);
    registerOption("SiftExtraction.max_image_size",
                   &sift_extraction_options.max_image_size);
    registerOption("SiftExtraction.max_num_features",
                   &sift_extraction_options.max_num_features);
    registerOption("SiftExtraction.first_octave",
                   &sift_extraction_options.first_octave);
    registerOption("SiftExtraction.num_octaves",
                   &sift_extraction_options.num_octaves);
    registerOption("SiftExtraction.octave_resolution",
                   &sift_extraction_options.octave_resolution);
    registerOption("SiftExtraction.peak_threshold",
                   &sift_extraction_options.peak_threshold);
    registerOption("SiftExtraction.edge_threshold",
                   &sift_extraction_options.edge_threshold);
    registerOption("SiftExtraction.estimate_affine_shape",
                   &sift_extraction_options.estimate_affine_shape);
    registerOption("SiftExtraction.max_num_orientations",
                   &sift_extraction_options.max_num_orientations);
    registerOption("SiftExtraction.upright", &sift_extraction_options.upright);
    registerOption("SiftExtraction.domain_size_pooling",
                   &sift_extraction_options.domain_size_pooling);
    registerOption("SiftExtraction.dsp_min_scale",
                   &sift_extraction_options.dsp_min_scale);
    registerOption("SiftExtraction.dsp_max_scale",
                   &sift_extraction_options.dsp_max_scale);
    registerOption("SiftExtraction.dsp_num_scales",
                   &sift_extraction_options.dsp_num_scales);
}

void OptionsParser::addMatchingOptions(
        const colmap::SiftMatchingOptions& sift_matching_options) {
    registerOption("SiftMatching.num_threads",
                   &sift_matching_options.num_threads);
    registerOption("SiftMatching.use_gpu", &sift_matching_options.use_gpu);
    registerOption("SiftMatching.gpu_index", &sift_matching_options.gpu_index);
    registerOption("SiftMatching.max_ratio", &sift_matching_options.max_ratio);
    registerOption("SiftMatching.max_distance",
                   &sift_matching_options.max_distance);
    registerOption("SiftMatching.cross_check",
                   &sift_matching_options.cross_check);
    registerOption("SiftMatching.max_error", &sift_matching_options.max_error);
    registerOption("SiftMatching.max_num_matches",
                   &sift_matching_options.max_num_matches);
    registerOption("SiftMatching.confidence",
                   &sift_matching_options.confidence);
    registerOption("SiftMatching.max_num_trials",
                   &sift_matching_options.max_num_trials);
    registerOption("SiftMatching.min_inlier_ratio",
                   &sift_matching_options.min_inlier_ratio);
    registerOption("SiftMatching.min_num_inliers",
                   &sift_matching_options.min_num_inliers);
    registerOption("SiftMatching.multiple_models",
                   &sift_matching_options.multiple_models);
    registerOption("SiftMatching.guided_matching",
                   &sift_matching_options.guided_matching);
}

void OptionsParser::addExhaustiveMatchingOptions(
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::ExhaustiveMatchingOptions& exhaustive_matching_options) {
    addMatchingOptions(sift_matching_options);
    registerOption("ExhaustiveMatching.block_size",
                   &exhaustive_matching_options.block_size);
}

void OptionsParser::addSequentialMatchingOptions(
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::SequentialMatchingOptions& sequential_matching_options) {
    addMatchingOptions(sift_matching_options);

    registerOption("SequentialMatching.overlap",
                   &sequential_matching_options.overlap);
    registerOption("SequentialMatching.quadratic_overlap",
                   &sequential_matching_options.quadratic_overlap);
    registerOption("SequentialMatching.loop_detection",
                   &sequential_matching_options.loop_detection);
    registerOption("SequentialMatching.loop_detection_period",
                   &sequential_matching_options.loop_detection_period);
    registerOption("SequentialMatching.loop_detection_num_images",
                   &sequential_matching_options.loop_detection_num_images);
    registerOption(
            "SequentialMatching.loop_detection_num_nearest_neighbors",
            &sequential_matching_options.loop_detection_num_nearest_neighbors);
    registerOption("SequentialMatching.loop_detection_num_checks",
                   &sequential_matching_options.loop_detection_num_checks);
    registerOption(
            "SequentialMatching.loop_detection_num_images_after_verification",
            &sequential_matching_options
                     .loop_detection_num_images_after_verification);
    registerOption(
            "SequentialMatching.loop_detection_max_num_features",
            &sequential_matching_options.loop_detection_max_num_features);
    registerOption("SequentialMatching.vocab_tree_path",
                   &sequential_matching_options.vocab_tree_path);
}

void OptionsParser::addVocabTreeMatchingOptions(
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::VocabTreeMatchingOptions& vocab_tree_matching_options) {
    addMatchingOptions(sift_matching_options);

    registerOption("VocabTreeMatching.num_images",
                   &vocab_tree_matching_options.num_images);
    registerOption("VocabTreeMatching.num_nearest_neighbors",
                   &vocab_tree_matching_options.num_nearest_neighbors);
    registerOption("VocabTreeMatching.num_checks",
                   &vocab_tree_matching_options.num_checks);
    registerOption("VocabTreeMatching.num_images_after_verification",
                   &vocab_tree_matching_options.num_images_after_verification);
    registerOption("VocabTreeMatching.max_num_features",
                   &vocab_tree_matching_options.max_num_features);
    registerOption("VocabTreeMatching.vocab_tree_path",
                   &vocab_tree_matching_options.vocab_tree_path);
    registerOption("VocabTreeMatching.match_list_path",
                   &vocab_tree_matching_options.match_list_path);
}

void OptionsParser::addSpatialMatchingOptions(
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::SpatialMatchingOptions& spatial_matching_options) {
    addMatchingOptions(sift_matching_options);

    registerOption("SpatialMatching.is_gps", &spatial_matching_options.is_gps);
    registerOption("SpatialMatching.ignore_z",
                   &spatial_matching_options.ignore_z);
    registerOption("SpatialMatching.max_num_neighbors",
                   &spatial_matching_options.max_num_neighbors);
    registerOption("SpatialMatching.max_distance",
                   &spatial_matching_options.max_distance);
}

void OptionsParser::addTransitiveMatchingOptions(
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::TransitiveMatchingOptions& transitive_matching_options) {
    addMatchingOptions(sift_matching_options);

    registerOption("TransitiveMatching.batch_size",
                   &transitive_matching_options.batch_size);
    registerOption("TransitiveMatching.num_iterations",
                   &transitive_matching_options.num_iterations);
}

void OptionsParser::addMapperOptions(
        const colmap::IncrementalMapperOptions& incremental_mapper_options) {
    registerOption("Mapper.min_num_matches",
                   &incremental_mapper_options.min_num_matches);
    registerOption("Mapper.ignore_watermarks",
                   &incremental_mapper_options.ignore_watermarks);
    registerOption("Mapper.multiple_models",
                   &incremental_mapper_options.multiple_models);
    registerOption("Mapper.max_num_models",
                   &incremental_mapper_options.max_num_models);
    registerOption("Mapper.max_model_overlap",
                   &incremental_mapper_options.max_model_overlap);
    registerOption("Mapper.min_model_size",
                   &incremental_mapper_options.min_model_size);
    registerOption("Mapper.init_image_id1",
                   &incremental_mapper_options.init_image_id1);
    registerOption("Mapper.init_image_id2",
                   &incremental_mapper_options.init_image_id2);
    registerOption("Mapper.init_num_trials",
                   &incremental_mapper_options.init_num_trials);
    registerOption("Mapper.extract_colors",
                   &incremental_mapper_options.extract_colors);
    registerOption("Mapper.num_threads",
                   &incremental_mapper_options.num_threads);
    registerOption("Mapper.min_focal_length_ratio",
                   &incremental_mapper_options.min_focal_length_ratio);
    registerOption("Mapper.max_focal_length_ratio",
                   &incremental_mapper_options.max_focal_length_ratio);
    registerOption("Mapper.max_extra_param",
                   &incremental_mapper_options.max_extra_param);
    registerOption("Mapper.ba_refine_focal_length",
                   &incremental_mapper_options.ba_refine_focal_length);
    registerOption("Mapper.ba_refine_principal_point",
                   &incremental_mapper_options.ba_refine_principal_point);
    registerOption("Mapper.ba_refine_extra_params",
                   &incremental_mapper_options.ba_refine_extra_params);
    registerOption("Mapper.ba_min_num_residuals_for_cpu_multi_threading",
                   &incremental_mapper_options
                            .ba_min_num_residuals_for_cpu_multi_threading);
    registerOption("Mapper.ba_local_num_images",
                   &incremental_mapper_options.ba_local_num_images);
    registerOption("Mapper.ba_local_function_tolerance",
                   &incremental_mapper_options.ba_local_function_tolerance);
    registerOption("Mapper.ba_local_max_num_iterations",
                   &incremental_mapper_options.ba_local_max_num_iterations);
#ifdef PBA_ENABLED
    registerOption("Mapper.ba_global_use_pba",
                   &incremental_mapper_options.ba_global_use_pba);
    registerOption("Mapper.ba_global_pba_gpu_index",
                   &incremental_mapper_options.ba_global_pba_gpu_index);
#endif
    registerOption("Mapper.ba_global_images_ratio",
                   &incremental_mapper_options.ba_global_images_ratio);
    registerOption("Mapper.ba_global_points_ratio",
                   &incremental_mapper_options.ba_global_points_ratio);
    registerOption("Mapper.ba_global_images_freq",
                   &incremental_mapper_options.ba_global_images_freq);
    registerOption("Mapper.ba_global_points_freq",
                   &incremental_mapper_options.ba_global_points_freq);
    registerOption("Mapper.ba_global_function_tolerance",
                   &incremental_mapper_options.ba_global_function_tolerance);
    registerOption("Mapper.ba_global_max_num_iterations",
                   &incremental_mapper_options.ba_global_max_num_iterations);
    registerOption("Mapper.ba_global_max_refinements",
                   &incremental_mapper_options.ba_global_max_refinements);
    registerOption("Mapper.ba_global_max_refinement_change",
                   &incremental_mapper_options.ba_global_max_refinement_change);
    registerOption("Mapper.ba_local_max_refinements",
                   &incremental_mapper_options.ba_local_max_refinements);
    registerOption("Mapper.ba_local_max_refinement_change",
                   &incremental_mapper_options.ba_local_max_refinement_change);
    registerOption("Mapper.snapshot_path",
                   &incremental_mapper_options.snapshot_path);
    registerOption("Mapper.snapshot_images_freq",
                   &incremental_mapper_options.snapshot_images_freq);
    registerOption("Mapper.fix_existing_images",
                   &incremental_mapper_options.fix_existing_images);

    // IncrementalMapper.
    registerOption("Mapper.init_min_num_inliers",
                   &incremental_mapper_options.mapper.init_min_num_inliers);
    registerOption("Mapper.init_max_error",
                   &incremental_mapper_options.mapper.init_max_error);
    registerOption("Mapper.init_max_forward_motion",
                   &incremental_mapper_options.mapper.init_max_forward_motion);
    registerOption("Mapper.init_min_tri_angle",
                   &incremental_mapper_options.mapper.init_min_tri_angle);
    registerOption("Mapper.init_max_reg_trials",
                   &incremental_mapper_options.mapper.init_max_reg_trials);
    registerOption("Mapper.abs_pose_max_error",
                   &incremental_mapper_options.mapper.abs_pose_max_error);
    registerOption("Mapper.abs_pose_min_num_inliers",
                   &incremental_mapper_options.mapper.abs_pose_min_num_inliers);
    registerOption(
            "Mapper.abs_pose_min_inlier_ratio",
            &incremental_mapper_options.mapper.abs_pose_min_inlier_ratio);
    registerOption("Mapper.filter_max_reproj_error",
                   &incremental_mapper_options.mapper.filter_max_reproj_error);
    registerOption("Mapper.filter_min_tri_angle",
                   &incremental_mapper_options.mapper.filter_min_tri_angle);
    registerOption("Mapper.max_reg_trials",
                   &incremental_mapper_options.mapper.max_reg_trials);
    registerOption("Mapper.local_ba_min_tri_angle",
                   &incremental_mapper_options.mapper.local_ba_min_tri_angle);

    // IncrementalTriangulator.
    registerOption("Mapper.tri_max_transitivity",
                   &incremental_mapper_options.triangulation.max_transitivity);
    registerOption(
            "Mapper.tri_create_max_angle_error",
            &incremental_mapper_options.triangulation.create_max_angle_error);
    registerOption(
            "Mapper.tri_continue_max_angle_error",
            &incremental_mapper_options.triangulation.continue_max_angle_error);
    registerOption(
            "Mapper.tri_merge_max_reproj_error",
            &incremental_mapper_options.triangulation.merge_max_reproj_error);
    registerOption("Mapper.tri_complete_max_reproj_error",
                   &incremental_mapper_options.triangulation
                            .complete_max_reproj_error);
    registerOption("Mapper.tri_complete_max_transitivity",
                   &incremental_mapper_options.triangulation
                            .complete_max_transitivity);
    registerOption(
            "Mapper.tri_re_max_angle_error",
            &incremental_mapper_options.triangulation.re_max_angle_error);
    registerOption("Mapper.tri_re_min_ratio",
                   &incremental_mapper_options.triangulation.re_min_ratio);
    registerOption("Mapper.tri_re_max_trials",
                   &incremental_mapper_options.triangulation.re_max_trials);
    registerOption("Mapper.tri_min_angle",
                   &incremental_mapper_options.triangulation.min_angle);
    registerOption(
            "Mapper.tri_ignore_two_view_tracks",
            &incremental_mapper_options.triangulation.ignore_two_view_tracks);
}

void OptionsParser::addImagePairsMatchingOptions(
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::ImagePairsMatchingOptions& image_pairs_matching_options) {
    addMatchingOptions(sift_matching_options);

    registerOption("ImagePairsMatching.block_size",
                   &image_pairs_matching_options.block_size);
}

void OptionsParser::addBundleAdjustmentOptions(
        const colmap::BundleAdjustmentOptions& bundle_adjustment_options) {
    registerOption(
            "BundleAdjustment.max_num_iterations",
            &bundle_adjustment_options.solver_options.max_num_iterations);
    registerOption("BundleAdjustment.max_linear_solver_iterations",
                   &bundle_adjustment_options.solver_options
                            .max_linear_solver_iterations);
    registerOption(
            "BundleAdjustment.function_tolerance",
            &bundle_adjustment_options.solver_options.function_tolerance);

    registerOption(
            "BundleAdjustment.gradient_tolerance",
            &bundle_adjustment_options.solver_options.gradient_tolerance);
    registerOption(
            "BundleAdjustment.parameter_tolerance",
            &bundle_adjustment_options.solver_options.parameter_tolerance);

    registerOption("BundleAdjustment.refine_focal_length",
                   &bundle_adjustment_options.refine_focal_length);
    registerOption("BundleAdjustment.refine_principal_point",
                   &bundle_adjustment_options.refine_principal_point);
    registerOption("BundleAdjustment.refine_extra_params",
                   &bundle_adjustment_options.refine_extra_params);
    registerOption("BundleAdjustment.refine_extrinsics",
                   &bundle_adjustment_options.refine_extrinsics);
}

void OptionsParser::addPatchMatchStereoOptions(
        const colmap::mvs::PatchMatchOptions& patch_match_options) {
    registerOption("PatchMatchStereo.max_image_size",
                   &patch_match_options.max_image_size);
    registerOption("PatchMatchStereo.gpu_index",
                   &patch_match_options.gpu_index);
    registerOption("PatchMatchStereo.depth_min",
                   &patch_match_options.depth_min);
    registerOption("PatchMatchStereo.depth_max",
                   &patch_match_options.depth_max);
    registerOption("PatchMatchStereo.window_radius",
                   &patch_match_options.window_radius);
    registerOption("PatchMatchStereo.window_step",
                   &patch_match_options.window_step);
    registerOption("PatchMatchStereo.sigma_spatial",
                   &patch_match_options.sigma_spatial);

    registerOption("PatchMatchStereo.sigma_color",
                   &patch_match_options.sigma_color);
    registerOption("PatchMatchStereo.num_samples",
                   &patch_match_options.num_samples);
    registerOption("PatchMatchStereo.ncc_sigma",
                   &patch_match_options.ncc_sigma);
    registerOption("PatchMatchStereo.min_triangulation_angle",
                   &patch_match_options.min_triangulation_angle);
    registerOption("PatchMatchStereo.incident_angle_sigma",
                   &patch_match_options.incident_angle_sigma);
    registerOption("PatchMatchStereo.num_iterations",
                   &patch_match_options.num_iterations);
    registerOption("PatchMatchStereo.geom_consistency",
                   &patch_match_options.geom_consistency);

    registerOption("PatchMatchStereo.geom_consistency_regularizer",
                   &patch_match_options.geom_consistency_regularizer);
    registerOption("PatchMatchStereo.geom_consistency_max_cost",
                   &patch_match_options.geom_consistency_max_cost);
    registerOption("PatchMatchStereo.filter", &patch_match_options.filter);
    registerOption("PatchMatchStereo.filter_min_ncc",
                   &patch_match_options.filter_min_ncc);
    registerOption("PatchMatchStereo.filter_min_triangulation_angle",
                   &patch_match_options.filter_min_triangulation_angle);
    registerOption("PatchMatchStereo.filter_min_num_consistent",
                   &patch_match_options.filter_min_num_consistent);
    registerOption("PatchMatchStereo.filter_geom_consistency_max_cost",
                   &patch_match_options.filter_geom_consistency_max_cost);
    registerOption("PatchMatchStereo.cache_size",
                   &patch_match_options.cache_size);
    registerOption("PatchMatchStereo.allow_missing_files",
                   &patch_match_options.allow_missing_files);
    registerOption("PatchMatchStereo.write_consistency_graph",
                   &patch_match_options.write_consistency_graph);
}

void OptionsParser::addStereoFusionOptions(
        const colmap::mvs::StereoFusionOptions& stereo_fusion_options) {
    registerOption("StereoFusion.mask_path", &stereo_fusion_options.mask_path);
    registerOption("StereoFusion.num_threads",
                   &stereo_fusion_options.num_threads);
    registerOption("StereoFusion.max_image_size",
                   &stereo_fusion_options.max_image_size);
    registerOption("StereoFusion.min_num_pixels",
                   &stereo_fusion_options.min_num_pixels);
    registerOption("StereoFusion.max_num_pixels",
                   &stereo_fusion_options.max_num_pixels);
    registerOption("StereoFusion.max_traversal_depth",
                   &stereo_fusion_options.max_traversal_depth);
    registerOption("StereoFusion.max_reproj_error",
                   &stereo_fusion_options.max_reproj_error);
    registerOption("StereoFusion.max_depth_error",
                   &stereo_fusion_options.max_depth_error);
    registerOption("StereoFusion.max_normal_error",
                   &stereo_fusion_options.max_normal_error);
    registerOption("StereoFusion.check_num_images",
                   &stereo_fusion_options.check_num_images);
    registerOption("StereoFusion.cache_size",
                   &stereo_fusion_options.cache_size);
    registerOption("StereoFusion.use_cache", &stereo_fusion_options.use_cache);
}

void OptionsParser::addPoissonMeshingOptions(
        const colmap::mvs::PoissonMeshingOptions& poisson_meshing_options) {
    registerOption("PoissonMeshing.point_weight",
                   &poisson_meshing_options.point_weight);
    registerOption("PoissonMeshing.depth", &poisson_meshing_options.depth);
    registerOption("PoissonMeshing.color", &poisson_meshing_options.color);
    registerOption("PoissonMeshing.trim", &poisson_meshing_options.trim);
    registerOption("PoissonMeshing.num_threads",
                   &poisson_meshing_options.num_threads);
}

void OptionsParser::addDelaunayMeshingOptions(
        const colmap::mvs::DelaunayMeshingOptions& delaunay_meshing_options) {
    registerOption("DelaunayMeshing.max_proj_dist",
                   &delaunay_meshing_options.max_proj_dist);
    registerOption("DelaunayMeshing.max_depth_dist",
                   &delaunay_meshing_options.max_depth_dist);
    registerOption("DelaunayMeshing.visibility_sigma",
                   &delaunay_meshing_options.visibility_sigma);
    registerOption("DelaunayMeshing.distance_sigma_factor",
                   &delaunay_meshing_options.distance_sigma_factor);
    registerOption("DelaunayMeshing.quality_regularization",
                   &delaunay_meshing_options.quality_regularization);
    registerOption("DelaunayMeshing.max_side_length_factor",
                   &delaunay_meshing_options.max_side_length_factor);
    registerOption("DelaunayMeshing.max_side_length_percentile",
                   &delaunay_meshing_options.max_side_length_percentile);
    registerOption("DelaunayMeshing.num_threads",
                   &delaunay_meshing_options.num_threads);
}

}  // namespace cloudViewer
