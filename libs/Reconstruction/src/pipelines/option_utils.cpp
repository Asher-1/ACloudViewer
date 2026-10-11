// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pipelines/option_utils.h"

#include "feature/aliked.h"
#include "feature/loma.h"
#include "feature/sift.h"

namespace cloudViewer {

OptionsParser::OptionsParser() {
    argc_ = 0;
    argv_ = nullptr;
    reset();
}

bool OptionsParser::parseOptions(int& argc, char**& argv) {
    ReleaseOptions(argc, argv);
    const std::size_t capacity = 2 * options_.size() + 1;

    // add application name
    argv = new char*[capacity];
    SetValue("options", 0, argv);
    argc = 1;

    for (const auto& option : options_) {
        SetValue("--" + option.first, argc, argv);
        argc += 1;
        SetValue(option.second, argc, argv);
        argc += 1;
    }

    return argc > 0 && argv != nullptr;
}

void OptionsParser::reset() {
    releaseOptions();
    options_.clear();
}

void OptionsParser::addFeatureExtractionOptions(
        const colmap::ImageReaderOptions& image_reader_options,
        const colmap::FeatureExtractionOptions& extraction_options) {
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

    registerOption("FeatureExtraction.type", &extraction_options.type);
    registerOption("FeatureExtraction.num_threads",
                   &extraction_options.num_threads);
    registerOption("FeatureExtraction.use_gpu", &extraction_options.use_gpu);
    registerOption("FeatureExtraction.gpu_index",
                   &extraction_options.gpu_index);
    registerOption("FeatureExtraction.max_image_size",
                   &extraction_options.max_image_size);

    const colmap::SiftExtractionOptions& sift = *extraction_options.sift;
    registerOption("SiftExtraction.max_num_features", &sift.max_num_features);
    registerOption("SiftExtraction.first_octave", &sift.first_octave);
    registerOption("SiftExtraction.num_octaves", &sift.num_octaves);
    registerOption("SiftExtraction.octave_resolution", &sift.octave_resolution);
    registerOption("SiftExtraction.peak_threshold", &sift.peak_threshold);
    registerOption("SiftExtraction.edge_threshold", &sift.edge_threshold);
    registerOption("SiftExtraction.estimate_affine_shape",
                   &sift.estimate_affine_shape);
    registerOption("SiftExtraction.max_num_orientations",
                   &sift.max_num_orientations);
    registerOption("SiftExtraction.upright", &sift.upright);
    registerOption("SiftExtraction.domain_size_pooling",
                   &sift.domain_size_pooling);
    registerOption("SiftExtraction.dsp_min_scale", &sift.dsp_min_scale);
    registerOption("SiftExtraction.dsp_max_scale", &sift.dsp_max_scale);
    registerOption("SiftExtraction.dsp_num_scales", &sift.dsp_num_scales);

    const colmap::AlikedExtractionOptions& aliked = *extraction_options.aliked;
    registerOption("AlikedExtraction.max_num_features",
                   &aliked.max_num_features);
    registerOption("AlikedExtraction.min_score", &aliked.min_score);
    registerOption("AlikedExtraction.n16rot_model_path",
                   &aliked.n16rot_model_path);
    registerOption("AlikedExtraction.n32_model_path", &aliked.n32_model_path);
    registerOption("AlikedExtraction.device", &aliked.device);

    const colmap::LomaExtractionOptions& loma = *extraction_options.loma;
    registerOption("LomaExtraction.max_num_features", &loma.max_num_features);
    registerOption("LomaExtraction.min_score", &loma.min_score);
    registerOption("LomaExtraction.detector_model_path",
                   &loma.detector_model_path);
    registerOption("LomaExtraction.descriptor_model_path",
                   &loma.descriptor_model_path);
    registerOption("LomaExtraction.descriptor_b128_model_path",
                   &loma.descriptor_b128_model_path);
    registerOption("LomaExtraction.device", &loma.device);
}

void OptionsParser::addFeatureMatchingOptions(
        const colmap::FeatureMatchingOptions& matching_options) {
    registerOption("FeatureMatching.type", &matching_options.type);
    registerOption("FeatureMatching.num_threads",
                   &matching_options.num_threads);
    registerOption("FeatureMatching.use_gpu", &matching_options.use_gpu);
    registerOption("FeatureMatching.gpu_index", &matching_options.gpu_index);
    registerOption("FeatureMatching.guided_matching",
                   &matching_options.guided_matching);
    registerOption("FeatureMatching.skip_geometric_verification",
                   &matching_options.skip_geometric_verification);
    registerOption("FeatureMatching.rig_verification",
                   &matching_options.rig_verification);
    registerOption("FeatureMatching.skip_image_pairs_in_same_frame",
                   &matching_options.skip_image_pairs_in_same_frame);
    registerOption("FeatureMatching.max_num_matches",
                   &matching_options.max_num_matches);

    const colmap::SiftMatchingOptions& sift = *matching_options.sift;
    registerOption("SiftMatching.max_ratio", &sift.max_ratio);
    registerOption("SiftMatching.max_distance", &sift.max_distance);
    registerOption("SiftMatching.cross_check", &sift.cross_check);
    registerOption("SiftMatching.cpu_brute_force_matcher",
                   &sift.cpu_brute_force_matcher);
    registerOption("SiftMatching.lightglue_min_score",
                   &sift.lightglue.min_score);
    registerOption("SiftMatching.lightglue_model_path",
                   &sift.lightglue.model_path);

    const colmap::AlikedMatchingOptions& aliked = *matching_options.aliked;
    registerOption("AlikedMatching.brute_force_min_cossim",
                   &aliked.brute_force.min_cossim);
    registerOption("AlikedMatching.brute_force_max_ratio",
                   &aliked.brute_force.max_ratio);
    registerOption("AlikedMatching.brute_force_cross_check",
                   &aliked.brute_force.cross_check);
    registerOption("AlikedMatching.bruteforce_model_path",
                   &aliked.brute_force.model_path);
    registerOption("AlikedMatching.lightglue_min_score",
                   &aliked.lightglue.min_score);
    registerOption("AlikedMatching.lightglue_model_path",
                   &aliked.lightglue.model_path);

    const colmap::LomaMatchingOptions& loma = *matching_options.loma;
    registerOption("LomaMatching.min_score", &loma.min_score);
    registerOption("LomaMatching.b_model_path", &loma.b.model_path);
    registerOption("LomaMatching.b128_model_path", &loma.b128.model_path);
    registerOption("LomaMatching.r_model_path", &loma.r.model_path);
    registerOption("LomaMatching.l_model_path", &loma.l.model_path);
    registerOption("LomaMatching.g_model_path", &loma.g.model_path);
    registerOption("LomaMatching.brute_force_min_cossim",
                   &loma.brute_force.min_cossim);
    registerOption("LomaMatching.brute_force_max_ratio",
                   &loma.brute_force.max_ratio);
    registerOption("LomaMatching.brute_force_cross_check",
                   &loma.brute_force.cross_check);
    registerOption("LomaMatching.brute_force_model_path",
                   &loma.brute_force.model_path);
    registerOption("LomaMatching.device", &loma.device);
}

void OptionsParser::addTwoViewGeometryOptions(
        const colmap::TwoViewGeometryOptions& geometry_options) {
    registerOption("TwoViewGeometry.min_num_inliers",
                   &geometry_options.min_num_inliers);
    registerOption("TwoViewGeometry.multiple_models",
                   &geometry_options.multiple_models);
    registerOption("TwoViewGeometry.compute_relative_pose",
                   &geometry_options.compute_relative_pose);
    registerOption("TwoViewGeometry.detect_watermark",
                   &geometry_options.detect_watermark);
    registerOption("TwoViewGeometry.multiple_ignore_watermark",
                   &geometry_options.multiple_ignore_watermark);
    registerOption("TwoViewGeometry.watermark_detection_max_error",
                   &geometry_options.watermark_detection_max_error);
    registerOption("TwoViewGeometry.filter_stationary_matches",
                   &geometry_options.filter_stationary_matches);
    registerOption("TwoViewGeometry.stationary_matches_max_error",
                   &geometry_options.stationary_matches_max_error);
    registerOption("TwoViewGeometry.use_degensac",
                   &geometry_options.use_degensac);
    registerOption("TwoViewGeometry.use_sampson_refinement",
                   &geometry_options.use_sampson_refinement);
    registerOption("TwoViewGeometry.max_error",
                   &geometry_options.ransac_options.max_error);
    registerOption("TwoViewGeometry.confidence",
                   &geometry_options.ransac_options.confidence);
    registerOption("TwoViewGeometry.max_num_trials",
                   &geometry_options.ransac_options.max_num_trials);
    registerOption("TwoViewGeometry.min_inlier_ratio",
                   &geometry_options.min_inlier_ratio);
    registerOption("TwoViewGeometry.random_seed",
                   &geometry_options.ransac_options.random_seed);
}

void OptionsParser::addExhaustivePairingOptions(
        const colmap::FeatureMatchingOptions& matching_options,
        const colmap::TwoViewGeometryOptions& geometry_options,
        const colmap::ExhaustivePairingOptions& pairing_options) {
    addFeatureMatchingOptions(matching_options);
    addTwoViewGeometryOptions(geometry_options);
    registerOption("ExhaustiveMatching.block_size",
                   &pairing_options.block_size);
}

void OptionsParser::addSequentialPairingOptions(
        const colmap::FeatureMatchingOptions& matching_options,
        const colmap::TwoViewGeometryOptions& geometry_options,
        const colmap::SequentialPairingOptions& pairing_options) {
    addFeatureMatchingOptions(matching_options);
    addTwoViewGeometryOptions(geometry_options);

    registerOption("SequentialMatching.overlap", &pairing_options.overlap);
    registerOption("SequentialMatching.quadratic_overlap",
                   &pairing_options.quadratic_overlap);
    registerOption("SequentialMatching.expand_rig_images",
                   &pairing_options.expand_rig_images);
    registerOption("SequentialMatching.loop_detection",
                   &pairing_options.loop_detection);
    registerOption("SequentialMatching.loop_detection_period",
                   &pairing_options.loop_detection_period);
    registerOption("SequentialMatching.loop_detection_num_images",
                   &pairing_options.loop_detection_num_images);
    registerOption("SequentialMatching.loop_detection_min_index_distance",
                   &pairing_options.loop_detection_min_index_distance);
    registerOption("SequentialMatching.loop_detection_num_nearest_neighbors",
                   &pairing_options.loop_detection_num_nearest_neighbors);
    registerOption("SequentialMatching.loop_detection_num_checks",
                   &pairing_options.loop_detection_num_checks);
    registerOption(
            "SequentialMatching.loop_detection_num_images_after_verification",
            &pairing_options.loop_detection_num_images_after_verification);
    registerOption("SequentialMatching.loop_detection_max_num_features",
                   &pairing_options.loop_detection_max_num_features);
    registerOption("SequentialMatching.vocab_tree_path",
                   &pairing_options.vocab_tree_path);
    registerOption("SequentialMatching.num_threads",
                   &pairing_options.num_threads);
}

void OptionsParser::addVocabTreePairingOptions(
        const colmap::FeatureMatchingOptions& matching_options,
        const colmap::TwoViewGeometryOptions& geometry_options,
        const colmap::VocabTreePairingOptions& pairing_options) {
    addFeatureMatchingOptions(matching_options);
    addTwoViewGeometryOptions(geometry_options);

    registerOption("VocabTreeMatching.num_images", &pairing_options.num_images);
    registerOption("VocabTreeMatching.num_nearest_neighbors",
                   &pairing_options.num_nearest_neighbors);
    registerOption("VocabTreeMatching.num_checks", &pairing_options.num_checks);
    registerOption("VocabTreeMatching.num_images_after_verification",
                   &pairing_options.num_images_after_verification);
    registerOption("VocabTreeMatching.max_num_features",
                   &pairing_options.max_num_features);
    registerOption("VocabTreeMatching.vocab_tree_path",
                   &pairing_options.vocab_tree_path);
    registerOption("VocabTreeMatching.match_list_path",
                   &pairing_options.match_list_path);
    registerOption("VocabTreeMatching.num_threads",
                   &pairing_options.num_threads);
}

void OptionsParser::addSpatialPairingOptions(
        const colmap::FeatureMatchingOptions& matching_options,
        const colmap::TwoViewGeometryOptions& geometry_options,
        const colmap::SpatialPairingOptions& pairing_options) {
    addFeatureMatchingOptions(matching_options);
    addTwoViewGeometryOptions(geometry_options);

    registerOption("SpatialMatching.ignore_z", &pairing_options.ignore_z);
    registerOption("SpatialMatching.max_num_neighbors",
                   &pairing_options.max_num_neighbors);
    registerOption("SpatialMatching.min_num_neighbors",
                   &pairing_options.min_num_neighbors);
    registerOption("SpatialMatching.max_distance",
                   &pairing_options.max_distance);
}

void OptionsParser::addTransitivePairingOptions(
        const colmap::FeatureMatchingOptions& matching_options,
        const colmap::TwoViewGeometryOptions& geometry_options,
        const colmap::TransitivePairingOptions& pairing_options) {
    addFeatureMatchingOptions(matching_options);
    addTwoViewGeometryOptions(geometry_options);

    registerOption("TransitiveMatching.batch_size",
                   &pairing_options.batch_size);
    registerOption("TransitiveMatching.num_iterations",
                   &pairing_options.num_iterations);
}

void OptionsParser::addImportedPairingOptions(
        const colmap::FeatureMatchingOptions& matching_options,
        const colmap::TwoViewGeometryOptions& geometry_options,
        const colmap::ImportedPairingOptions& pairing_options) {
    addFeatureMatchingOptions(matching_options);
    addTwoViewGeometryOptions(geometry_options);

    registerOption("ImagePairsMatching.block_size",
                   &pairing_options.block_size);
}

void OptionsParser::addMapperOptions(
        const colmap::IncrementalPipelineOptions& incremental_mapper_options) {
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
    registerOption(
            "Mapper.structure_less_registration_fallback",
            &incremental_mapper_options.structure_less_registration_fallback);
    registerOption(
            "Mapper.structure_less_registration_only",
            &incremental_mapper_options.structure_less_registration_only);
    registerOption("Mapper.extract_colors",
                   &incremental_mapper_options.extract_colors);
    registerOption("Mapper.num_threads",
                   &incremental_mapper_options.num_threads);
    registerOption("Mapper.random_seed",
                   &incremental_mapper_options.random_seed);
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
    registerOption("Mapper.ba_refine_sensor_from_rig",
                   &incremental_mapper_options.ba_refine_sensor_from_rig);
    registerOption("Mapper.ba_local_function_tolerance",
                   &incremental_mapper_options.ba_local_function_tolerance);
    registerOption("Mapper.ba_local_max_num_iterations",
                   &incremental_mapper_options.ba_local_max_num_iterations);
    registerOption("Mapper.ba_global_frames_ratio",
                   &incremental_mapper_options.ba_global_frames_ratio);
    registerOption("Mapper.ba_global_points_ratio",
                   &incremental_mapper_options.ba_global_points_ratio);
    registerOption("Mapper.ba_global_frames_freq",
                   &incremental_mapper_options.ba_global_frames_freq);
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
    registerOption("Mapper.ba_use_gpu", &incremental_mapper_options.ba_use_gpu);
    registerOption("Mapper.ba_gpu_index",
                   &incremental_mapper_options.ba_gpu_index);
    registerOption("Mapper.ba_local_backend",
                   &incremental_mapper_options.ba_local_backend);
    registerOption("Mapper.ba_global_backend",
                   &incremental_mapper_options.ba_global_backend);
    registerOption("Mapper.ba_min_num_residuals_for_cpu_multi_threading",
                   &incremental_mapper_options
                            .ba_min_num_residuals_for_cpu_multi_threading);
    registerOption("Mapper.snapshot_path",
                   &incremental_mapper_options.snapshot_path);
    registerOption("Mapper.snapshot_frames_freq",
                   &incremental_mapper_options.snapshot_frames_freq);
    registerOption("Mapper.fix_existing_frames",
                   &incremental_mapper_options.fix_existing_frames);
    registerOption("Mapper.max_runtime_seconds",
                   &incremental_mapper_options.max_runtime_seconds);

    // IncrementalMapper.
    const colmap::IncrementalMapper::Options& mapper =
            incremental_mapper_options.mapper;
    registerOption("Mapper.init_min_num_inliers", &mapper.init_min_num_inliers);
    registerOption("Mapper.init_max_error", &mapper.init_max_error);
    registerOption("Mapper.init_max_forward_motion",
                   &mapper.init_max_forward_motion);
    registerOption("Mapper.init_min_tri_angle", &mapper.init_min_tri_angle);
    registerOption("Mapper.init_max_reg_trials", &mapper.init_max_reg_trials);
    registerOption("Mapper.abs_pose_max_error", &mapper.abs_pose_max_error);
    registerOption("Mapper.abs_pose_min_num_inliers",
                   &mapper.abs_pose_min_num_inliers);
    registerOption("Mapper.abs_pose_min_inlier_ratio",
                   &mapper.abs_pose_min_inlier_ratio);
    registerOption("Mapper.filter_max_reproj_error",
                   &mapper.filter_max_reproj_error);
    registerOption("Mapper.filter_min_tri_angle", &mapper.filter_min_tri_angle);
    registerOption("Mapper.max_reg_trials", &mapper.max_reg_trials);
    registerOption("Mapper.ba_local_num_images", &mapper.ba_local_num_images);
    registerOption("Mapper.ba_local_min_tri_angle",
                   &mapper.ba_local_min_tri_angle);
    registerOption("Mapper.ba_global_ignore_redundant_points3D",
                   &mapper.ba_global_ignore_redundant_points3D);
    registerOption(
            "Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain",
            &mapper.ba_global_ignore_redundant_points3D_min_coverage_gain);

    // IncrementalTriangulator.
    const colmap::IncrementalTriangulator::Options& tri =
            incremental_mapper_options.triangulation;
    registerOption("Mapper.tri_max_transitivity", &tri.max_transitivity);
    registerOption("Mapper.tri_create_max_angle_error",
                   &tri.create_max_angle_error);
    registerOption("Mapper.tri_continue_max_angle_error",
                   &tri.continue_max_angle_error);
    registerOption("Mapper.tri_merge_max_reproj_error",
                   &tri.merge_max_reproj_error);
    registerOption("Mapper.tri_complete_max_reproj_error",
                   &tri.complete_max_reproj_error);
    registerOption("Mapper.tri_complete_max_transitivity",
                   &tri.complete_max_transitivity);
    registerOption("Mapper.tri_re_max_angle_error", &tri.re_max_angle_error);
    registerOption("Mapper.tri_re_min_ratio", &tri.re_min_ratio);
    registerOption("Mapper.tri_re_max_trials", &tri.re_max_trials);
    registerOption("Mapper.tri_min_angle", &tri.min_angle);
    registerOption("Mapper.tri_ignore_two_view_tracks",
                   &tri.ignore_two_view_tracks);
}

void OptionsParser::addBundleAdjustmentOptions(
        const colmap::BundleAdjustmentOptions& bundle_adjustment_options) {
    registerOption("BundleAdjustment.refine_focal_length",
                   &bundle_adjustment_options.refine_focal_length);
    registerOption("BundleAdjustment.refine_principal_point",
                   &bundle_adjustment_options.refine_principal_point);
    registerOption("BundleAdjustment.refine_extra_params",
                   &bundle_adjustment_options.refine_extra_params);
    registerOption("BundleAdjustment.refine_rig_from_world",
                   &bundle_adjustment_options.refine_rig_from_world);
    registerOption("BundleAdjustment.refine_sensor_from_rig",
                   &bundle_adjustment_options.refine_sensor_from_rig);
    registerOption("BundleAdjustment.refine_points3D",
                   &bundle_adjustment_options.refine_points3D);
    registerOption("BundleAdjustment.constant_rig_from_world_rotation",
                   &bundle_adjustment_options.constant_rig_from_world_rotation);
    registerOption("BundleAdjustment.min_track_length",
                   &bundle_adjustment_options.min_track_length);
    registerOption("BundleAdjustment.backend",
                   &bundle_adjustment_options.backend);

    const colmap::CeresBundleAdjustmentOptions& ceres =
            *bundle_adjustment_options.ceres;
    registerOption("BundleAdjustmentCeres.max_num_iterations",
                   &ceres.solver_options.max_num_iterations);
    registerOption("BundleAdjustmentCeres.max_linear_solver_iterations",
                   &ceres.solver_options.max_linear_solver_iterations);
    registerOption("BundleAdjustmentCeres.function_tolerance",
                   &ceres.solver_options.function_tolerance);
    registerOption("BundleAdjustmentCeres.gradient_tolerance",
                   &ceres.solver_options.gradient_tolerance);
    registerOption("BundleAdjustmentCeres.parameter_tolerance",
                   &ceres.solver_options.parameter_tolerance);
    registerOption("BundleAdjustmentCeres.use_gpu", &ceres.use_gpu);
    registerOption("BundleAdjustmentCeres.gpu_index", &ceres.gpu_index);
    registerOption("BundleAdjustmentCeres.min_num_images_gpu_solver",
                   &ceres.min_num_images_gpu_solver);
    registerOption(
            "BundleAdjustmentCeres.min_num_residuals_for_cpu_multi_threading",
            &ceres.min_num_residuals_for_cpu_multi_threading);
    registerOption(
            "BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver",
            &ceres.max_num_images_direct_dense_cpu_solver);
    registerOption(
            "BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver",
            &ceres.max_num_images_direct_sparse_cpu_solver);
    registerOption(
            "BundleAdjustmentCeres.max_num_images_direct_dense_gpu_solver",
            &ceres.max_num_images_direct_dense_gpu_solver);
    registerOption(
            "BundleAdjustmentCeres.max_num_images_direct_sparse_gpu_solver",
            &ceres.max_num_images_direct_sparse_gpu_solver);

    const colmap::CasparBundleAdjustmentOptions& caspar =
            *bundle_adjustment_options.caspar;
    registerOption("BundleAdjustmentCaspar.solver_iter_max",
                   &caspar.solver_iter_max);
    registerOption("BundleAdjustmentCaspar.pcg_iter_max", &caspar.pcg_iter_max);
    registerOption("BundleAdjustmentCaspar.diag_init", &caspar.diag_init);
    registerOption("BundleAdjustmentCaspar.diag_min", &caspar.diag_min);
    registerOption("BundleAdjustmentCaspar.diag_scaling_up",
                   &caspar.diag_scaling_up);
    registerOption("BundleAdjustmentCaspar.diag_scaling_down",
                   &caspar.diag_scaling_down);
    registerOption("BundleAdjustmentCaspar.diag_exit_value",
                   &caspar.diag_exit_value);
    registerOption("BundleAdjustmentCaspar.score_exit_value",
                   &caspar.score_exit_value);
    registerOption("BundleAdjustmentCaspar.pcg_rel_error_exit",
                   &caspar.pcg_rel_error_exit);
    registerOption("BundleAdjustmentCaspar.pcg_rel_score_exit",
                   &caspar.pcg_rel_score_exit);
    registerOption("BundleAdjustmentCaspar.pcg_rel_decrease_min",
                   &caspar.pcg_rel_decrease_min);
    registerOption("BundleAdjustmentCaspar.solver_rel_decrease_min",
                   &caspar.solver_rel_decrease_min);
    registerOption("BundleAdjustmentCaspar.gpu_index", &caspar.gpu_index);
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

void OptionsParser::addMeshPostProcessingOptions(
        const colmap::mvs::MeshPostProcessingOptions& options) {
    registerOption("MeshPostProcessing.enabled", &options.enabled);
    registerOption("MeshPostProcessing.remove_small_components",
                   &options.remove_small_components);
    registerOption("MeshPostProcessing.remove_degenerate_faces",
                   &options.remove_degenerate_faces);
    registerOption("MeshPostProcessing.simplify", &options.simplify);
    registerOption("MeshPostProcessing.smooth", &options.smooth);
    registerOption("MeshPostProcessing.preserve_boundary",
                   &options.preserve_boundary);
    registerOption("MeshPostProcessing.prune_error", &options.prune_error);
    registerOption("MeshPostProcessing.target_face_ratio",
                   &options.target_face_ratio);
    registerOption("MeshPostProcessing.simplify_error",
                   &options.simplify_error);
    registerOption("MeshPostProcessing.max_aspect_ratio",
                   &options.max_aspect_ratio);
    registerOption("MeshPostProcessing.smoothing_iterations",
                   &options.smoothing_iterations);
    registerOption("MeshPostProcessing.smoothing_lambda",
                   &options.smoothing_lambda);
    registerOption("MeshPostProcessing.smoothing_mu", &options.smoothing_mu);
}

}  // namespace cloudViewer
