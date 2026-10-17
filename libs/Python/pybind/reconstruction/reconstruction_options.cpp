// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/reconstruction_options.h"

#include <Logging.h>

#include "controllers/pairing.h"
#include "estimators/bundle_adjustment_ceres.h"
#include "feature/sift.h"
#include "mvs/delaunay_meshing.h"
#include "mvs/mesh_postprocessing.h"
#include "mvs/mesh_simplification.h"
#include "pipelines/mvs.h"
#include "pipelines/option_utils.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/hash_container_casters.h"

namespace cloudViewer {
namespace reconstruction {
namespace options {

void pybind_image_reader_options(py::module& m) {
    // cloudViewer.reconstruction.options.ImageReaderOptions
    py::class_<colmap::ImageReaderOptions,
               std::shared_ptr<colmap::ImageReaderOptions>>
            image_reader_options(m, "ImageReaderOptions",
                                 "Image Reader option class.");
    image_reader_options.def(py::init<>())
            .def("check", &colmap::ImageReaderOptions::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "database_path", &colmap::ImageReaderOptions::database_path,
                    "str: (Default ``''``) Path to database in which to store "
                    "the extracted data.")
            .def_readwrite(
                    "image_path", &colmap::ImageReaderOptions::image_path,
                    "str: (Default ``''``) Root path to folder which contains "
                    "the images.")
            .def_readwrite(
                    "mask_path", &colmap::ImageReaderOptions::mask_path,
                    "str: (Default ``''``) Optional root path to folder which "
                    "contains image masks. For a given image, the "
                    "corresponding mask must have the same sub-path below "
                    "this root as the image has below image_path. The filename "
                    "must be equal, "
                    "aside from the added extension .png. For example, for an "
                    "image "
                    "image_path/abc/012.jpg, the mask would be "
                    "mask_path/abc/012.jpg.png. No features "
                    "will be extracted in regions where the mask image is "
                    "black (pixel intensity "
                    "value 0 in grayscale).")
            .def_readwrite("image_names",
                           &colmap::ImageReaderOptions::image_names,
                           "List(str, ...): Optional list of images to read. "
                           "The list must contain the relative path "
                           "of the images with respect to the image_path.")
            .def_readwrite("image_list",
                           &colmap::ImageReaderOptions::image_names,
                           "Deprecated alias of ``image_names``. "
                           "Optional list of images to read. "
                           "The list must contain the relative path "
                           "of the images with respect to the image_path.")
            .def_readwrite(
                    "camera_model", &colmap::ImageReaderOptions::camera_model,
                    "str: (Default ``SIMPLE_RADIAL``) Name of the camera "
                    "model. Supported camera model:"
                    "(``SIMPLE_PINHOLE``, ``PINHOLE``, ``SIMPLE_RADIAL``, "
                    "``RADIAL``, ``OPENCV``, ``OPENCV_FISHEYE``, "
                    "``FULL_OPENCV``, ``FOV``, ``SIMPLE_RADIAL_FISHEYE``, "
                    "``RADIAL_FISHEYE``, ``THIN_PRISM_FISHEYE``, "
                    "``SIMPLE_RADIAL_FISHEYE``)")
            .def_readwrite(
                    "single_camera", &colmap::ImageReaderOptions::single_camera,
                    "bool: (Default ``False``) Set to ``True`` to enable the "
                    "same camera for all images.")
            .def_readwrite(
                    "single_camera_per_folder",
                    &colmap::ImageReaderOptions::single_camera_per_folder,
                    "bool: (Default ``False``) Set to ``True`` to enable the "
                    "same camera for all images in the same sub-folder.")
            .def_readwrite(
                    "single_camera_per_image",
                    &colmap::ImageReaderOptions::single_camera_per_image,
                    "bool: (Default ``False``) Set to ``True`` to enable a "
                    "different camera for each image.")
            .def_readwrite(
                    "existing_camera_id",
                    &colmap::ImageReaderOptions::existing_camera_id,
                    "int: (Default "
                    "``kInvalidCameraId=std::numeric_limits<camera_t>::max()``)"
                    " Whether to "
                    "explicitly use an existing camera for all images. "
                    "Note that in this case the specified camera model and "
                    "parameters are ignored.")
            .def_readwrite(
                    "camera_params", &colmap::ImageReaderOptions::camera_params,
                    "str: (Default ``''``) Manual specification of "
                    "camera parameters. If empty, camera parameters will "
                    "be extracted from EXIF, i.e. principal point and focal "
                    "length.")
            .def_readwrite(
                    "default_focal_length_factor",
                    &colmap::ImageReaderOptions::default_focal_length_factor,
                    "float: (Default ``1.2``) If camera parameters are not "
                    "specified manually and the image does not have focal "
                    "length EXIF information, the focal length is set to "
                    "the value ``default_focal_length_factor * max(width, "
                    "height)``.")
            .def_readwrite("camera_mask_path",
                           &colmap::ImageReaderOptions::camera_mask_path,
                           "str: (Default ``''``) Optional path to an image "
                           "file specifying a mask for all images. No features "
                           "will be extracted in regions where the mask is "
                           "black (pixel intensity value 0 in grayscale).");
}

void pybind_sift_extraction_options(py::module& m) {
    // upstream pycolmap feature/extraction.cc: SiftExtractionOptions only
    // carries the SIFT-specific detector/descriptor parameters; the shared
    // num_threads / use_gpu / gpu_index / max_image_size live on
    // FeatureExtractionOptions.
    py::native_enum<colmap::SiftExtractionOptions::Normalization>(
            m, "Normalization", "enum.Enum", "Descriptor normalization types.")
            .value("L1_ROOT",
                   colmap::SiftExtractionOptions::Normalization::L1_ROOT)
            .value("L2", colmap::SiftExtractionOptions::Normalization::L2)
            .export_values()
            .finalize();

    py::class_<colmap::SiftExtractionOptions,
               std::shared_ptr<colmap::SiftExtractionOptions>>
            sift_extraction_options(m, "SiftExtractionOptions",
                                    "Sift Extraction option class.");
    sift_extraction_options.def(py::init<>())
            .def("check", &colmap::SiftExtractionOptions::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "max_num_features",
                    &colmap::SiftExtractionOptions::max_num_features,
                    "int: (Default ``8192``) Maximum number of features to "
                    "detect, keeping larger-scale features.")
            .def_readwrite(
                    "first_octave",
                    &colmap::SiftExtractionOptions::first_octave,
                    "int: (Default ``-1``) First octave in the pyramid, i.e. "
                    "-1 upsamples the image by one level.")
            .def_readwrite(
                    "num_octaves", &colmap::SiftExtractionOptions::num_octaves,
                    "int: (Default ``4``) Number of octaves in the pyramid.")
            .def_readwrite("octave_resolution",
                           &colmap::SiftExtractionOptions::octave_resolution,
                           "int: (Default ``3``) Number of levels per octave.")
            .def_readwrite("peak_threshold",
                           &colmap::SiftExtractionOptions::peak_threshold,
                           "float: (Default ``0.02 / octave_resolution``) Peak "
                           "threshold for detection.")
            .def_readwrite(
                    "edge_threshold",
                    &colmap::SiftExtractionOptions::edge_threshold,
                    "float: (Default ``10.0``) Edge threshold for detection.")
            .def_readwrite(
                    "estimate_affine_shape",
                    &colmap::SiftExtractionOptions::estimate_affine_shape,
                    "bool: (Default ``False``) Estimate affine shape of SIFT "
                    "features in the form of oriented ellipses as opposed to "
                    "original SIFT which estimates oriented disks.")
            .def_readwrite(
                    "max_num_orientations",
                    &colmap::SiftExtractionOptions::max_num_orientations,
                    "int: (Default ``2``) Maximum number of orientations per "
                    "keypoint if not estimate_affine_shape.")
            .def_readwrite(
                    "upright", &colmap::SiftExtractionOptions::upright,
                    "bool: (Default ``False``) Fix the orientation to 0 for "
                    "upright features.")
            .def_readwrite(
                    "darkness_adaptivity",
                    &colmap::SiftExtractionOptions::darkness_adaptivity,
                    "bool: (Default ``False``) Whether to adapt the feature "
                    "detection depending on the image darkness. Note that this "
                    "feature is only available in the OpenGL SiftGPU version.")
            .def_readwrite(
                    "domain_size_pooling",
                    &colmap::SiftExtractionOptions::domain_size_pooling,
                    "bool: (Default ``False``) Domain-size pooling computes an "
                    "average SIFT descriptor across multiple scales around the "
                    "detected scale. This was proposed in Domain-Size Pooling "
                    "in Local Descriptors and Network Architectures, J. Dong "
                    "and S. Soatto, CVPR 2015.")
            .def_readwrite("dsp_min_scale",
                           &colmap::SiftExtractionOptions::dsp_min_scale,
                           "float: (Default ``1.0 / 6.0``) min scale.")
            .def_readwrite("dsp_max_scale",
                           &colmap::SiftExtractionOptions::dsp_max_scale,
                           "float: (Default ``3.0``) max scale.")
            .def_readwrite("dsp_num_scales",
                           &colmap::SiftExtractionOptions::dsp_num_scales,
                           "int: (Default ``10``) The number of scales.")
            .def_readwrite(
                    "force_covariant_extractor",
                    &colmap::SiftExtractionOptions::force_covariant_extractor,
                    "bool: (Default ``False``) Whether to force usage of the "
                    "covariant VLFeat implementation.")
            .def_readwrite(
                    "normalization",
                    &colmap::SiftExtractionOptions::normalization,
                    "Normalization: (Default ``L1_ROOT``) L1-normalizes each "
                    "descriptor followed by element-wise square rooting. This "
                    "normalization is usually better than standard "
                    "L2-normalization. See ``Three things everyone should know "
                    "to improve object retrieval``, Relja Arandjelovic and "
                    "Andrew Zisserman, CVPR 2012.");
}

void pybind_sift_matching_options(py::module& m) {
    // upstream pycolmap feature/matching.cc: SiftMatchingOptions only carries
    // the SIFT descriptor-matching thresholds; the shared num_threads /
    // use_gpu / gpu_index / max_num_matches / guided_matching live on
    // FeatureMatchingOptions and the RANSAC thresholds on
    // TwoViewGeometryOptions.
    py::class_<colmap::SiftMatchingOptions,
               std::shared_ptr<colmap::SiftMatchingOptions>>
            sift_matching_options(m, "SiftMatchingOptions",
                                  "Sift Matching option class.");
    sift_matching_options.def(py::init<>())
            .def("check", &colmap::SiftMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("max_ratio", &colmap::SiftMatchingOptions::max_ratio,
                           "float: (Default ``0.8``) Maximum distance ratio "
                           "between first and second best match.")
            .def_readwrite(
                    "max_distance", &colmap::SiftMatchingOptions::max_distance,
                    "float: (Default ``0.7``) Maximum distance to best match.")
            .def_readwrite("cross_check",
                           &colmap::SiftMatchingOptions::cross_check,
                           "bool: (Default ``True``) Whether to enable cross "
                           "checking in matching.")
            .def_readwrite(
                    "cpu_brute_force_matcher",
                    &colmap::SiftMatchingOptions::cpu_brute_force_matcher,
                    "bool: (Default ``False``) Whether to use brute-force "
                    "instead of the nearest-neighbor index for CPU matching.")
            .def_readwrite("lightglue", &colmap::SiftMatchingOptions::lightglue,
                           "LightGlueONNXMatchingOptions: LightGlue matching "
                           "options used by the SIFT_LIGHTGLUE matcher.");
}

void pybind_featurepairs_matching_options(py::module& m) {
    // cloudViewer.reconstruction.pipeline.FeaturePairsMatchingOptions
    py::class_<colmap::FeaturePairsMatchingOptions>
            featurepairs_matching_options(
                    m, "FeaturePairsMatchingOptions",
                    "FeaturePairs Matching option class.");
    featurepairs_matching_options.def(py::init<>())
            .def("check", &colmap::FeaturePairsMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("verify_matches",
                           &colmap::FeaturePairsMatchingOptions::verify_matches,
                           "bool: (Default ``True``) Whether to geometrically "
                           "verify the given matches.")
            .def_readwrite(
                    "match_list_path",
                    &colmap::FeaturePairsMatchingOptions::match_list_path,
                    "str: (Default ``''``) Path to the file with the matches.");
}

void pybind_bundle_adjustment_options(py::module& m) {
    // upstream pycolmap estimators/bundle_adjustment.cc: the ceres-specific
    // solver configuration is its own options struct, shared with the
    // BundleAdjustmentOptions::ceres member (shared_ptr holder).
    py::class_<colmap::CeresBundleAdjustmentOptions,
               std::shared_ptr<colmap::CeresBundleAdjustmentOptions>>
            ceres_options(m, "CeresBundleAdjustmentOptions",
                          "Ceres bundle adjustment backend option class.");
    ceres_options.def(py::init<>())
            .def("check", &colmap::CeresBundleAdjustmentOptions::Check,
                 "Check parameters validation.")
            .def("create_solver_options",
                 &colmap::CeresBundleAdjustmentOptions::CreateSolverOptions,
                 "config"_a, "problem"_a,
                 "Create the ceres solver options for the given problem.")
            .def_readwrite(
                    "loss_function_type",
                    &colmap::CeresBundleAdjustmentOptions::loss_function_type,
                    "CeresLossFunctionType: (Default ``TRIVIAL``) Loss "
                    "function types: Trivial (non-robust) and Cauchy / "
                    "Soft-L1 / Huber (robust) loss.")
            .def_readwrite(
                    "loss_function_scale",
                    &colmap::CeresBundleAdjustmentOptions::loss_function_scale,
                    "float: (Default ``1.0``) Scaling factor determines "
                    "residual at which robustification takes place.")
            .def_readwrite(
                    "loss_function_weight",
                    &colmap::CeresBundleAdjustmentOptions::loss_function_weight,
                    "float: (Default ``1.0``) Weight applied to the loss "
                    "function.")
            .def_readwrite("use_gpu",
                           &colmap::CeresBundleAdjustmentOptions::use_gpu,
                           "bool: (Default ``False``) Whether to use Ceres' "
                           "CUDA linear algebra library, if available.")
            .def_readwrite("gpu_index",
                           &colmap::CeresBundleAdjustmentOptions::gpu_index,
                           "str: (Default ``-1``) Which GPU to use for solving "
                           "the problem.")
            .def_readwrite(
                    "solver_options",
                    &colmap::CeresBundleAdjustmentOptions::solver_options,
                    "ceres.SolverOptions: Raw ceres solver options.")
            .def_readwrite(
                    "min_num_images_gpu_solver",
                    &colmap::CeresBundleAdjustmentOptions::
                            min_num_images_gpu_solver,
                    "int: (Default ``50``) Minimum number of images to use the "
                    "GPU solver.")
            .def_readwrite(
                    "min_num_residuals_for_cpu_multi_threading",
                    &colmap::CeresBundleAdjustmentOptions::
                            min_num_residuals_for_cpu_multi_threading,
                    "int: (Default ``50000``) Minimum number of residuals to "
                    "enable multi-threading. Note that single-threaded is "
                    "typically better for small bundle adjustment problems due "
                    "to the overhead of threading.")
            .def_readwrite("max_num_images_direct_dense_cpu_solver",
                           &colmap::CeresBundleAdjustmentOptions::
                                   max_num_images_direct_dense_cpu_solver,
                           "int: (Default ``50``) Threshold to switch between "
                           "direct, sparse, and iterative solvers.")
            .def_readwrite(
                    "max_num_images_direct_sparse_cpu_solver",
                    &colmap::CeresBundleAdjustmentOptions::
                            max_num_images_direct_sparse_cpu_solver,
                    "int: (Default ``1000``) Threshold to switch between "
                    "direct, sparse, and iterative solvers.")
            .def_readwrite("max_num_images_direct_dense_gpu_solver",
                           &colmap::CeresBundleAdjustmentOptions::
                                   max_num_images_direct_dense_gpu_solver,
                           "int: (Default ``200``) Threshold to switch between "
                           "direct, sparse, and iterative solvers.")
            .def_readwrite(
                    "max_num_images_direct_sparse_gpu_solver",
                    &colmap::CeresBundleAdjustmentOptions::
                            max_num_images_direct_sparse_gpu_solver,
                    "int: (Default ``4000``) Threshold to switch between "
                    "direct, sparse, and iterative solvers.")
            .def_readwrite(
                    "auto_select_solver_type",
                    &colmap::CeresBundleAdjustmentOptions::
                            auto_select_solver_type,
                    "bool: (Default ``True``) Whether to automatically select "
                    "the linear solver type from the problem size.");

    // cloudViewer.reconstruction.estimators.BundleAdjustmentOptions
    py::class_<colmap::BundleAdjustmentOptions,
               std::shared_ptr<colmap::BundleAdjustmentOptions>>
            bundle_adjustment_options(m, "BundleAdjustmentOptions",
                                      "Bundle Adjustment option class.");
    bundle_adjustment_options.def(py::init<>())
            .def("check", &colmap::BundleAdjustmentOptions::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "refine_focal_length",
                    &colmap::BundleAdjustmentOptions::refine_focal_length,
                    "bool: (Default ``True``) Whether to refine the focal "
                    "length parameter group.")
            .def_readwrite(
                    "refine_principal_point",
                    &colmap::BundleAdjustmentOptions::refine_principal_point,
                    "bool: (Default ``False``) Whether to refine the principal "
                    "point parameter group.")
            .def_readwrite(
                    "refine_extra_params",
                    &colmap::BundleAdjustmentOptions::refine_extra_params,
                    "bool: (Default ``True``) Whether to refine the extra "
                    "parameter group.")
            .def_readwrite(
                    "refine_rig_from_world",
                    &colmap::BundleAdjustmentOptions::refine_rig_from_world,
                    "bool: (Default ``True``) Whether to refine the "
                    "frame (rig) from world extrinsic parameter group.")
            .def_readwrite(
                    "refine_sensor_from_rig",
                    &colmap::BundleAdjustmentOptions::refine_sensor_from_rig,
                    "bool: (Default ``True``) Whether to refine the sensor "
                    "from rig extrinsic parameter group.")
            .def_readwrite(
                    "constant_rig_from_world_rotation",
                    &colmap::BundleAdjustmentOptions::
                            constant_rig_from_world_rotation,
                    "bool: (Default ``False``) Whether to keep the rotation "
                    "component of rig_from_world constant. Only takes effect "
                    "when refine_rig_from_world is true.")
            .def_readwrite("refine_points3D",
                           &colmap::BundleAdjustmentOptions::refine_points3D,
                           "bool: (Default ``True``) Whether to refine the 3D "
                           "points.")
            .def_readwrite(
                    "min_track_length",
                    &colmap::BundleAdjustmentOptions::min_track_length,
                    "int: (Default ``0``) Minimum track length for a 3D point "
                    "to be included in the optimization.")
            .def_readwrite("print_summary",
                           &colmap::BundleAdjustmentOptions::print_summary,
                           "bool: (Default ``True``) Whether to print a final "
                           "summary.")
            .def_readwrite(
                    "backend", &colmap::BundleAdjustmentOptions::backend,
                    "BundleAdjustmentBackend: (Default ``CERES``) Solver "
                    "backend to use for bundle adjustment.")
            .def_readwrite("ceres", &colmap::BundleAdjustmentOptions::ceres,
                           "CeresBundleAdjustmentOptions: Ceres backend "
                           "options.")
            .def_readwrite("caspar", &colmap::BundleAdjustmentOptions::caspar,
                           "CasparBundleAdjustmentOptions: CASPAR backend "
                           "options.");
}

void pybind_incremental_triangulator_options(py::module& m) {
    // cloudViewer.reconstruction.options.IncrementalTriangulatorOptions
    py::class_<colmap::IncrementalTriangulator::Options>
            incremental_triangulator_options(
                    m, "IncrementalTriangulatorOptions",
                    "Incremental Triangulator option class.");
    incremental_triangulator_options.def(py::init<>())
            .def("check", &colmap::IncrementalTriangulator::Options::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "max_transitivity",
                    &colmap::IncrementalTriangulator::Options::max_transitivity,
                    "int: (Default ``1``) Maximum transitivity to search for "
                    "correspondences.")
            .def_readwrite("create_max_angle_error",
                           &colmap::IncrementalTriangulator::Options::
                                   create_max_angle_error,
                           "float: (Default ``2.0``) Maximum angular error to "
                           "create new triangulations.")
            .def_readwrite("continue_max_angle_error",
                           &colmap::IncrementalTriangulator::Options::
                                   continue_max_angle_error,
                           "float: (Default ``2.0``) Maximum angular error to "
                           "continue existing triangulations.")
            .def_readwrite("merge_max_reproj_error",
                           &colmap::IncrementalTriangulator::Options::
                                   merge_max_reproj_error,
                           "float: (Default ``4.0``) Maximum reprojection "
                           "error in pixels to merge triangulations.")
            .def_readwrite("complete_max_reproj_error",
                           &colmap::IncrementalTriangulator::Options::
                                   complete_max_reproj_error,
                           "float: (Default ``4.0``) Maximum reprojection "
                           "error to complete an existing triangulation.")
            .def_readwrite("complete_max_transitivity",
                           &colmap::IncrementalTriangulator::Options::
                                   complete_max_transitivity,
                           "int: (Default ``5``) Maximum transitivity for "
                           "track completion.")
            .def_readwrite("re_max_angle_error",
                           &colmap::IncrementalTriangulator::Options::
                                   re_max_angle_error,
                           "float: (Default ``5.0``) Maximum angular error to "
                           "re-triangulate under-reconstructed image pairs.")
            .def_readwrite(
                    "re_min_ratio",
                    &colmap::IncrementalTriangulator::Options::re_min_ratio,
                    "float: (Default ``0.2``) Minimum ratio of common "
                    "triangulations between an image pair over the number of "
                    "correspondences between that image pair to be considered "
                    "as under-reconstructed.")
            .def_readwrite(
                    "re_max_trials",
                    &colmap::IncrementalTriangulator::Options::re_max_trials,
                    "int: (Default ``1``) Maximum number of trials to "
                    "re-triangulate an image pair.")
            .def_readwrite("min_angle",
                           &colmap::IncrementalTriangulator::Options::min_angle,
                           "float: (Default ``1.5``) Minimum pairwise "
                           "triangulation angle for a stable triangulation.")
            .def_readwrite("ignore_two_view_tracks",
                           &colmap::IncrementalTriangulator::Options::
                                   ignore_two_view_tracks,
                           "bool: (Default ``True``) Whether to ignore "
                           "two-view tracks.")
            .def_readwrite("min_focal_length_ratio",
                           &colmap::IncrementalTriangulator::Options::
                                   min_focal_length_ratio,
                           "float: (Default ``0.1``) Thresholds for bogus "
                           "camera parameters. Images with bogus camera "
                           "parameters are ignored in triangulation.")
            .def_readwrite("max_focal_length_ratio",
                           &colmap::IncrementalTriangulator::Options::
                                   max_focal_length_ratio,
                           "float: (Default ``10.0``) Thresholds for bogus "
                           "camera parameters. Images with bogus camera "
                           "parameters are ignored in triangulation.")
            .def_readwrite(
                    "max_extra_param",
                    &colmap::IncrementalTriangulator::Options::max_extra_param,
                    "float: (Default ``1.0``) Thresholds for bogus camera "
                    "parameters. Images with bogus camera parameters are "
                    "ignored in triangulation.");
}

void pybind_incremental_mapper_options(py::module& m) {
    // upstream pycolmap sfm/incremental_mapper.cc: IncrementalMapperOptions
    // is the per-stage IncrementalMapper::Options sub-structure.
    py::class_<colmap::IncrementalMapper::Options>
            incremental_mapper_sub_options(
                    m, "IncrementalMapperOptions",
                    "Options for the incremental mapper (one stage of the "
                    "incremental pipeline).");
    incremental_mapper_sub_options.def(py::init<>())
            .def("check", &colmap::IncrementalMapper::Options::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "init_min_num_inliers",
                    &colmap::IncrementalMapper::Options::init_min_num_inliers,
                    "int: (Default ``100``) Minimum number of inliers for "
                    "initial image pair.")
            .def_readwrite(
                    "init_max_error",
                    &colmap::IncrementalMapper::Options::init_max_error,
                    "float: (Default ``4.0``) Maximum error in pixels for "
                    "two-view geometry estimation for initial image pair.")
            .def_readwrite("init_max_forward_motion",
                           &colmap::IncrementalMapper::Options::
                                   init_max_forward_motion,
                           "float: (Default ``0.95``) Maximum forward motion "
                           "for initial image pair.")
            .def_readwrite(
                    "init_min_tri_angle",
                    &colmap::IncrementalMapper::Options::init_min_tri_angle,
                    "float: (Default ``16.0``) Minimum triangulation angle for "
                    "initial image pair.")
            .def_readwrite(
                    "init_max_reg_trials",
                    &colmap::IncrementalMapper::Options::init_max_reg_trials,
                    "int: (Default ``2``) Maximum number of trials to use an "
                    "image for initialization.")
            .def_readwrite(
                    "abs_pose_max_error",
                    &colmap::IncrementalMapper::Options::abs_pose_max_error,
                    "float: (Default ``12.0``) Maximum reprojection error in "
                    "absolute pose estimation.")
            .def_readwrite("abs_pose_min_num_inliers",
                           &colmap::IncrementalMapper::Options::
                                   abs_pose_min_num_inliers,
                           "int: (Default ``30``) Minimum number of inliers in "
                           "absolute pose estimation.")
            .def_readwrite("abs_pose_min_inlier_ratio",
                           &colmap::IncrementalMapper::Options::
                                   abs_pose_min_inlier_ratio,
                           "float: (Default ``0.25``) Minimum inlier ratio in "
                           "absolute pose estimation.")
            .def_readwrite("abs_pose_refine_focal_length",
                           &colmap::IncrementalMapper::Options::
                                   abs_pose_refine_focal_length,
                           "bool: (Default ``True``) Whether to estimate the "
                           "focal length in absolute pose estimation.")
            .def_readwrite("abs_pose_refine_extra_params",
                           &colmap::IncrementalMapper::Options::
                                   abs_pose_refine_extra_params,
                           "bool: (Default ``True``) Whether to estimate the "
                           "extra parameters in absolute pose estimation.")
            .def_readwrite(
                    "ba_local_num_images",
                    &colmap::IncrementalMapper::Options::ba_local_num_images,
                    "int: (Default ``6``) Number of images to optimize in "
                    "local bundle adjustment.")
            .def_readwrite(
                    "ba_local_min_tri_angle",
                    &colmap::IncrementalMapper::Options::ba_local_min_tri_angle,
                    "float: (Default ``6.0``) Minimum triangulation for images "
                    "to be chosen in local bundle adjustment.")
            .def_readwrite(
                    "min_focal_length_ratio",
                    &colmap::IncrementalMapper::Options::min_focal_length_ratio,
                    "float: (Default ``0.1``) Opening angle of ~130deg. "
                    "Thresholds for bogus camera parameters. "
                    "Images with bogus camera parameters are filtered and "
                    "ignored in triangulation.")
            .def_readwrite(
                    "max_focal_length_ratio",
                    &colmap::IncrementalMapper::Options::max_focal_length_ratio,
                    "float: (Default ``10.0``) Opening angle of ~5deg. "
                    "Thresholds for bogus camera parameters. "
                    "Images with bogus camera parameters are filtered and "
                    "ignored in triangulation.")
            .def_readwrite("max_extra_param",
                           &colmap::IncrementalMapper::Options::max_extra_param,
                           "float: (Default ``1.0``) Thresholds for bogus "
                           "camera parameters. "
                           "Images with bogus camera parameters are filtered "
                           "and ignored in triangulation.")
            .def_readwrite("filter_max_reproj_error",
                           &colmap::IncrementalMapper::Options::
                                   filter_max_reproj_error,
                           "float: (Default ``4.0``) Maximum reprojection "
                           "error in pixels for observations.")
            .def_readwrite(
                    "filter_min_tri_angle",
                    &colmap::IncrementalMapper::Options::filter_min_tri_angle,
                    "float: (Default ``1.5``) Minimum triangulation angle in "
                    "degrees for stable 3D points.")
            .def_readwrite("max_reg_trials",
                           &colmap::IncrementalMapper::Options::max_reg_trials,
                           "int: (Default ``3``) Maximum number of trials to "
                           "register an image.")
            .def_readwrite(
                    "fix_existing_frames",
                    &colmap::IncrementalMapper::Options::fix_existing_frames,
                    "bool: (Default ``False``) If reconstruction is provided "
                    "as input, fix the existing frame poses.")
            .def_property(
                    "constant_rigs",
                    [](const colmap::IncrementalMapper::Options& self) {
                        return std::vector<colmap::rig_t>(
                                self.constant_rigs.begin(),
                                self.constant_rigs.end());
                    },
                    [](colmap::IncrementalMapper::Options& self,
                       const std::vector<colmap::rig_t>& rigs) {
                        self.constant_rigs.clear();
                        self.constant_rigs.insert(rigs.begin(), rigs.end());
                    },
                    "list(int, ...): Rigs for which to fix the "
                    "sensor_from_rig transformation, independent of "
                    "ba_refine_sensor_from_rig.")
            .def_property(
                    "constant_cameras",
                    [](const colmap::IncrementalMapper::Options& self) {
                        return std::vector<colmap::camera_t>(
                                self.constant_cameras.begin(),
                                self.constant_cameras.end());
                    },
                    [](colmap::IncrementalMapper::Options& self,
                       const std::vector<colmap::camera_t>& cameras) {
                        self.constant_cameras.clear();
                        self.constant_cameras.insert(cameras.begin(),
                                                     cameras.end());
                    },
                    "list(int, ...): Cameras whose intrinsic parameters stay "
                    "fixed independent of refine_focal_length, "
                    "refine_principal_point, and refine_extra_params.")
            .def_readwrite("ba_global_ignore_redundant_points3D",
                           &colmap::IncrementalMapper::Options::
                                   ba_global_ignore_redundant_points3D,
                           "bool: (Default ``False``) Whether to (temporarily) "
                           "exclude redundant 3D points from the global bundle "
                           "adjustment for sufficiently large reconstructions.")
            .def_readwrite(
                    "ba_global_prune_points_min_coverage_gain",
                    &colmap::IncrementalMapper::Options::
                            ba_global_ignore_redundant_points3D_min_coverage_gain,
                    "float: (Default ``0.05``) The coverage gain threshold "
                    "for a 3D point to be considered redundant in the global "
                    "bundle adjustment.")
            .def_readwrite(
                    "use_prior_position",
                    &colmap::IncrementalMapper::Options::use_prior_position,
                    "bool: (Default ``False``) Whether to use the position "
                    "prior to fix the gauge and constrain the bundle "
                    "adjustment. Requires at least 3 registered images with "
                    "a position prior.")
            .def_readwrite(
                    "use_robust_loss_on_prior_position",
                    &colmap::IncrementalMapper::Options::
                            use_robust_loss_on_prior_position,
                    "bool: (Default ``False``) Whether to use a robust loss "
                    "on the position prior residuals.")
            .def_readwrite(
                    "prior_position_loss_scale",
                    &colmap::IncrementalMapper::Options::
                            prior_position_loss_scale,
                    "float: (Default ``7.815``) The scaling factor for the "
                    "position prior loss function.")
            .def_readwrite("random_seed",
                           &colmap::IncrementalMapper::Options::random_seed,
                           "int: (Default ``-1``) PRNG seed for all stochastic "
                           "methods during reconstruction.")
            .def_readwrite("num_threads",
                           &colmap::IncrementalMapper::Options::num_threads,
                           "int: (Default ``-1``) Number of threads.")
            .def_readwrite(
                    "image_selection_method",
                    &colmap::IncrementalMapper::Options::image_selection_method,
                    "ImageSelectionMethod: (Default "
                    "``ImageSelectionMethod::MIN_UNCERTAINTY``) Method to find "
                    "and select next best image to register.");

    // cloudViewer.reconstruction.options.ImageSelectionMethod
    py::native_enum<colmap::IncrementalMapper::Options::ImageSelectionMethod>(
            incremental_mapper_sub_options, "ImageSelectionMethod", "enum.Enum",
            "Enum class for Reconstruction ImageSelection Method.")
            .value("MAX_VISIBLE_POINTS_NUM",
                   colmap::IncrementalMapper::Options::ImageSelectionMethod::
                           MAX_VISIBLE_POINTS_NUM)
            .value("MAX_VISIBLE_POINTS_RATIO",
                   colmap::IncrementalMapper::Options::ImageSelectionMethod::
                           MAX_VISIBLE_POINTS_RATIO)
            .value("MIN_UNCERTAINTY",
                   colmap::IncrementalMapper::Options::ImageSelectionMethod::
                           MIN_UNCERTAINTY)
            .export_values()
            .finalize();

    // upstream pycolmap sfm/incremental_mapper.cc: IncrementalPipelineOptions
    // is the top-level pipeline options class.
    py::class_<colmap::IncrementalPipelineOptions> incremental_mapper_options(
            m, "IncrementalPipelineOptions",
            "Options for the incremental mapping pipeline. The nested "
            "``mapper`` and ``triangulation`` members configure the "
            "per-stage IncrementalMapper and IncrementalTriangulator.");
    incremental_mapper_options.def(py::init<>())
            .def("check", &colmap::IncrementalPipelineOptions::Check,
                 "Check parameters validation.")
            .def("get_mapper", &colmap::IncrementalPipelineOptions::Mapper,
                 "Get Incremental Mapper sub options.")
            .def(
                    "is_initial_pair_provided",
                    [](const colmap::IncrementalPipelineOptions& self) {
                        return self.init_image_id1 != -1 ||
                               self.init_image_id2 != -1;
                    },
                    "Whether an initial image pair was specified.")
            .def("get_triangulation",
                 &colmap::IncrementalPipelineOptions::Triangulation,
                 "Get Incremental triangulator options.")
            .def("get_local_bundle_adjustment",
                 &colmap::IncrementalPipelineOptions::LocalBundleAdjustment,
                 "Get local BundleAdjustment options.")
            .def("get_global_bundle_adjustment",
                 &colmap::IncrementalPipelineOptions::GlobalBundleAdjustment,
                 "Get global BundleAdjustment options.")
            .def_readwrite("min_num_matches",
                           &colmap::IncrementalPipelineOptions::min_num_matches,
                           "int: (Default ``15``) The minimum number of "
                           "matches for inlier matches to be considered.")
            .def_readwrite(
                    "ignore_watermarks",
                    &colmap::IncrementalPipelineOptions::ignore_watermarks,
                    "bool: (Default ``False``) Whether to ignore the "
                    "inlier matches of watermark image pairs.")
            .def_readwrite("multiple_models",
                           &colmap::IncrementalPipelineOptions::multiple_models,
                           "bool: (Default ``True``) Whether to reconstruct "
                           "multiple sub-models.")
            .def_readwrite("max_num_models",
                           &colmap::IncrementalPipelineOptions::max_num_models,
                           "int: (Default ``50``) The number of sub-models to "
                           "reconstruct.")
            .def_readwrite(
                    "max_model_overlap",
                    &colmap::IncrementalPipelineOptions::max_model_overlap,
                    "int: (Default ``20``) The maximum number of overlapping "
                    "images between sub-models. If the current sub-models "
                    "shares more than this number of images with another "
                    "model, then the reconstruction is stopped.")
            .def_readwrite("min_model_size",
                           &colmap::IncrementalPipelineOptions::min_model_size,
                           "int: (Default ``10``) The minimum number of "
                           "registered images of a sub-model, otherwise the "
                           "sub-model is discarded.")
            .def_readwrite("init_image_id1",
                           &colmap::IncrementalPipelineOptions::init_image_id1,
                           "int: (Default ``-1``) The image identifiers used "
                           "to initialize the reconstruction. "
                           "Note that only one or both image identifiers can "
                           "be specified. In the former case, the second image "
                           "is automatically determined.")
            .def_readwrite("init_image_id2",
                           &colmap::IncrementalPipelineOptions::init_image_id2,
                           "int: (Default ``-1``) The image identifiers used "
                           "to initialize the reconstruction. "
                           "Note that only one or both image identifiers can "
                           "be specified. In the former case, the second image "
                           "is automatically determined.")
            .def_readwrite("init_num_trials",
                           &colmap::IncrementalPipelineOptions::init_num_trials,
                           "int: (Default ``200``) The number of trials to "
                           "initialize the reconstruction.")
            .def_readwrite("structure_less_registration_fallback",
                           &colmap::IncrementalPipelineOptions::
                                   structure_less_registration_fallback,
                           "bool: (Default ``True``) Whether to fall back to "
                           "structure-less registration when the structure-"
                           "based registration of an image fails.")
            .def_readwrite("structure_less_registration_only",
                           &colmap::IncrementalPipelineOptions::
                                   structure_less_registration_only,
                           "bool: (Default ``False``) Whether to only use "
                           "structure-less registration.")
            .def_readwrite("extract_colors",
                           &colmap::IncrementalPipelineOptions::extract_colors,
                           "bool: (Default ``True``) Whether to extract colors "
                           "for reconstructed points.")
            .def_readwrite("num_threads",
                           &colmap::IncrementalPipelineOptions::num_threads,
                           "int: (Default ``-1``) The number of threads to use "
                           "during reconstruction.")
            .def_readwrite("random_seed",
                           &colmap::IncrementalPipelineOptions::random_seed,
                           "int: (Default ``-1``) PRNG seed for all stochastic "
                           "methods during reconstruction.")
            .def_readwrite(
                    "min_focal_length_ratio",
                    &colmap::IncrementalPipelineOptions::min_focal_length_ratio,
                    "float: (Default ``0.1``) Thresholds for filtering images "
                    "with degenerate intrinsics.")
            .def_readwrite(
                    "max_focal_length_ratio",
                    &colmap::IncrementalPipelineOptions::max_focal_length_ratio,
                    "float: (Default ``10.0``) Thresholds for filtering images "
                    "with degenerate intrinsics.")
            .def_readwrite("max_extra_param",
                           &colmap::IncrementalPipelineOptions::max_extra_param,
                           "float: (Default ``1.0``) Thresholds for filtering "
                           "images with degenerate intrinsics.")
            .def_readwrite(
                    "ba_refine_focal_length",
                    &colmap::IncrementalPipelineOptions::ba_refine_focal_length,
                    "bool: (Default ``True``) Which intrinsic parameters to "
                    "optimize during the reconstruction.")
            .def_readwrite("ba_refine_principal_point",
                           &colmap::IncrementalPipelineOptions::
                                   ba_refine_principal_point,
                           "bool: (Default ``False``) Which intrinsic "
                           "parameters to optimize during the reconstruction.")
            .def_readwrite(
                    "ba_refine_extra_params",
                    &colmap::IncrementalPipelineOptions::ba_refine_extra_params,
                    "bool: (Default ``True``) Which intrinsic parameters to "
                    "optimize during the reconstruction.")
            .def_readwrite(
                    "ba_refine_sensor_from_rig",
                    &colmap::IncrementalPipelineOptions::
                            ba_refine_sensor_from_rig,
                    "bool: (Default ``True``) Whether to optimize the "
                    "sensor_from_rig transformations during reconstruction.")
            .def_readwrite("ba_min_num_residuals_for_cpu_multi_threading",
                           &colmap::IncrementalPipelineOptions::
                                   ba_min_num_residuals_for_cpu_multi_threading,
                           "int: (Default ``50000``) The minimum number of "
                           "residuals per bundle adjustment problem to enable "
                           "multi-threading solving of the problems.")
            .def_readwrite("ba_local_function_tolerance",
                           &colmap::IncrementalPipelineOptions::
                                   ba_local_function_tolerance,
                           "float: (Default ``0.0``) Ceres solver function "
                           "tolerance for local bundle adjustment.")
            .def_readwrite("ba_local_max_num_iterations",
                           &colmap::IncrementalPipelineOptions::
                                   ba_local_max_num_iterations,
                           "int: (Default ``-1``) The maximum number of local "
                           "bundle adjustment iterations (-1 derives the "
                           "backend default).")
            .def_readwrite(
                    "ba_global_frames_ratio",
                    &colmap::IncrementalPipelineOptions::ba_global_frames_ratio,
                    "float: (Default ``1.1``) The growth rates after which to "
                    "perform global bundle adjustment.")
            .def_readwrite(
                    "ba_global_points_ratio",
                    &colmap::IncrementalPipelineOptions::ba_global_points_ratio,
                    "float: (Default ``1.1``) The growth rates after which to "
                    "perform global bundle adjustment.")
            .def_readwrite(
                    "ba_global_frames_freq",
                    &colmap::IncrementalPipelineOptions::ba_global_frames_freq,
                    "int: (Default ``500``) The growth rates after which to "
                    "perform global bundle adjustment.")
            .def_readwrite(
                    "ba_global_points_freq",
                    &colmap::IncrementalPipelineOptions::ba_global_points_freq,
                    "int: (Default ``250000``) The growth rates after which to "
                    "perform global bundle adjustment.")
            .def_readwrite("ba_global_function_tolerance",
                           &colmap::IncrementalPipelineOptions::
                                   ba_global_function_tolerance,
                           "float: (Default ``0.0``) Ceres solver function "
                           "tolerance for global bundle adjustment")
            .def_readwrite("ba_global_max_num_iterations",
                           &colmap::IncrementalPipelineOptions::
                                   ba_global_max_num_iterations,
                           "int: (Default ``-1``) The maximum number of global "
                           "bundle adjustment iterations (-1 derives the "
                           "backend default).")
            .def_readwrite(
                    "ba_local_max_refinements",
                    &colmap::IncrementalPipelineOptions::
                            ba_local_max_refinements,
                    "int: (Default ``2``) The thresholds for iterative bundle "
                    "adjustment refinements.")
            .def_readwrite("ba_local_max_refinement_change",
                           &colmap::IncrementalPipelineOptions::
                                   ba_local_max_refinement_change,
                           "float: (Default ``0.001``) The thresholds for "
                           "iterative bundle adjustment refinements.")
            .def_readwrite("ba_global_max_refinements",
                           &colmap::IncrementalPipelineOptions::
                                   ba_global_max_refinements,
                           "int: (Default ``5``) The thresholds for iterative "
                           "bundle adjustment refinements.")
            .def_readwrite("ba_global_max_refinement_change",
                           &colmap::IncrementalPipelineOptions::
                                   ba_global_max_refinement_change,
                           "float: (Default ``0.0005``) The thresholds for "
                           "iterative bundle adjustment refinements.")
            .def_readwrite("ba_use_gpu",
                           &colmap::IncrementalPipelineOptions::ba_use_gpu,
                           "Whether to use Ceres' CUDA sparse linear algebra "
                           "library, if available.")
            .def_readwrite("ba_gpu_index",
                           &colmap::IncrementalPipelineOptions::ba_gpu_index,
                           "Index of CUDA GPU to use for BA, if available.")
            .def_readwrite("snapshot_path",
                           &colmap::IncrementalPipelineOptions::snapshot_path,
                           "str: (Default ``''``) Path to a folder with "
                           "reconstruction snapshots during incremental "
                           "reconstruction. Snapshots will be saved according "
                           "to the specified frequency of registered images.")
            .def_readwrite(
                    "snapshot_frames_freq",
                    &colmap::IncrementalPipelineOptions::snapshot_frames_freq,
                    "int: (Default ``0``) The frequency of registered images.")
            .def_readwrite("image_names",
                           &colmap::IncrementalPipelineOptions::image_names,
                           "list(str, ...): Which images to reconstruct. If no "
                           "images are specified, all images will be "
                           "reconstructed by default.")
            .def_readwrite("load_all_images",
                           &colmap::IncrementalPipelineOptions::load_all_images,
                           "bool: (Default ``False``) Whether to load all "
                           "images into the database cache even when only a "
                           "subset is reconstructed.")
            .def_readwrite(
                    "fix_existing_frames",
                    &colmap::IncrementalPipelineOptions::fix_existing_frames,
                    "bool: (Default ``False``) If reconstruction is provided "
                    "as input, fix the existing frame poses.")
            .def_readwrite(
                    "max_runtime_seconds",
                    &colmap::IncrementalPipelineOptions::max_runtime_seconds,
                    "int: (Default ``-1``) Maximum runtime in seconds after "
                    "which the reconstruction stops (-1 disables the limit).")
            .def_readwrite("mapper",
                           &colmap::IncrementalPipelineOptions::mapper,
                           "IncrementalPipelineOptions: Incremental mapper "
                           "sub options.")
            .def_readwrite("triangulation",
                           &colmap::IncrementalPipelineOptions::triangulation,
                           "IncrementalTriangulatorOptions: Incremental "
                           "triangulator options.")
            // --- fork parity members absent from the historical binding ---
            .def_readwrite(
                    "ba_local_backend",
                    &colmap::IncrementalPipelineOptions::ba_local_backend,
                    "BundleAdjustmentBackend: (Default ``CERES``) The bundle "
                    "adjustment backend for local BA. CASPAR is only "
                    "available in builds with RECONSTRUCTION_CASPAR_ENABLED; "
                    "unsupported requests fall back to Ceres with a "
                    "warning.")
            .def_readwrite(
                    "ba_global_backend",
                    &colmap::IncrementalPipelineOptions::ba_global_backend,
                    "BundleAdjustmentBackend: (Default ``CERES``) The bundle "
                    "adjustment backend for global BA.")
            .def_readwrite(
                    "use_prior_position",
                    &colmap::IncrementalPipelineOptions::use_prior_position,
                    "bool: (Default ``False``) Whether to use position priors "
                    "for BA and convert them to a local ENU frame when "
                    "loading the database cache.")
            .def_readwrite(
                    "use_robust_loss_on_prior_position",
                    &colmap::IncrementalPipelineOptions::
                            use_robust_loss_on_prior_position,
                    "bool: (Default ``False``) Whether to use a robust loss "
                    "on the position prior residuals.")
            .def_readwrite(
                    "prior_position_loss_scale",
                    &colmap::IncrementalPipelineOptions::
                            prior_position_loss_scale,
                    "float: (Default ``7.815``) The scaling factor for the "
                    "position prior loss function.")
            .def_property(
                    "constant_rigs",
                    [](const colmap::IncrementalPipelineOptions& self) {
                        return std::vector<colmap::rig_t>(
                                self.constant_rigs.begin(),
                                self.constant_rigs.end());
                    },
                    [](colmap::IncrementalPipelineOptions& self,
                       const std::vector<colmap::rig_t>& rigs) {
                        self.constant_rigs.clear();
                        self.constant_rigs.insert(rigs.begin(), rigs.end());
                    },
                    "list(int, ...): Rigs for which to fix the "
                    "sensor_from_rig transformation, independent of "
                    "ba_refine_sensor_from_rig.")
            .def_property(
                    "constant_cameras",
                    [](const colmap::IncrementalPipelineOptions& self) {
                        return std::vector<colmap::camera_t>(
                                self.constant_cameras.begin(),
                                self.constant_cameras.end());
                    },
                    [](colmap::IncrementalPipelineOptions& self,
                       const std::vector<colmap::camera_t>& cameras) {
                        self.constant_cameras.clear();
                        self.constant_cameras.insert(cameras.begin(),
                                                     cameras.end());
                    },
                    "list(int, ...): Cameras whose intrinsic parameters stay "
                    "fixed independent of ba_refine_focal_length, "
                    "ba_refine_principal_point, and ba_refine_extra_params.");
}

void pybind_patch_match_options(py::module& m) {
    // cloudViewer.reconstruction.options.PatchMatchOptions
    py::class_<colmap::mvs::PatchMatchOptions> patch_match_options(
            m, "PatchMatchOptions", "PatchMatch option class.");
    patch_match_options.def(py::init<>())
            .def("check", &colmap::mvs::PatchMatchOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("max_image_size",
                           &colmap::mvs::PatchMatchOptions::max_image_size,
                           "int: (Default ``-1``) Maximum image size in either "
                           "dimension.")
            .def_readwrite(
                    "gpu_index", &colmap::mvs::PatchMatchOptions::gpu_index,
                    "str: (Default ``'-1'``) Index of the GPU used for patch "
                    "match. For multi-GPU usage, you should separate multiple "
                    "GPU indices by comma, e.g., ``0,1,2,3``.")
            .def_readwrite("num_threads",
                           &colmap::mvs::PatchMatchOptions::num_threads,
                           "int: (Default ``-1``) Number of threads for "
                           "processing. -1 uses all available threads.")
            // Upstream pycolmap parity (src/pycolmap/pipeline/mvs.cc): the
            // dense-pipeline entry point attached to its options class,
            // forwarding to the fork's StereoPatchMatch wrapper.
            .def(
                    "patch_match_stereo",
                    [](const colmap::mvs::PatchMatchOptions& options,
                       const std::string& workspace_path,
                       const std::string& config_path,
                       const std::string& workspace_format,
                       const std::string& pmvs_option_name) {
                        return cloudViewer::StereoPatchMatch(
                                workspace_path, config_path, workspace_format,
                                pmvs_option_name, options);
                    },
                    "workspace_path"_a, "config_path"_a = "",
                    "workspace_format"_a = "COLMAP",
                    "pmvs_option_name"_a = "option-all",
                    "Run the CUDA patch-match stereo pipeline on a dense "
                    "workspace; returns 0 on success.");
}

void pybind_stereo_fusion_options(py::module& m) {
    // cloudViewer.reconstruction.options.StereoFusionOptions
    py::class_<colmap::mvs::StereoFusionOptions> stereo_fusion_options(
            m, "StereoFusionOptions", "Stereo fusion option class.");
    stereo_fusion_options.def(py::init<>())
            .def("check", &colmap::mvs::StereoFusionOptions::Check,
                 "Check parameters validation.")
            .def("print", &colmap::mvs::StereoFusionOptions::Print,
                 "Print the options to stdout..")
            .def_readwrite("mask_path",
                           &colmap::mvs::StereoFusionOptions::mask_path,
                           "str: (Default ``''``) Path for PNG masks. Same "
                           "format expected as ImageReaderOptions.")
            .def_readwrite("num_threads",
                           &colmap::mvs::StereoFusionOptions::num_threads,
                           "int: (Default ``-1``) The number of threads to use "
                           "during fusion.")
            .def_readwrite("max_image_size",
                           &colmap::mvs::StereoFusionOptions::max_image_size,
                           "int: (Default ``-1``) Maximum image size in either "
                           "dimension.")
            .def_readwrite("min_num_pixels",
                           &colmap::mvs::StereoFusionOptions::min_num_pixels,
                           "int: (Default ``5``) Minimum number of fused "
                           "pixels to produce a point.")
            .def_readwrite("max_num_pixels",
                           &colmap::mvs::StereoFusionOptions::max_num_pixels,
                           "int: (Default ``10000``) Maximum number of pixels "
                           "to fuse into a single point.")
            .def_readwrite(
                    "max_traversal_depth",
                    &colmap::mvs::StereoFusionOptions::max_traversal_depth,
                    "int: (Default ``100``) Maximum depth in consistency graph "
                    "traversal.")
            .def_readwrite("max_reproj_error",
                           &colmap::mvs::StereoFusionOptions::max_reproj_error,
                           "float: (Default ``2.0``) Maximum relative "
                           "difference between measured and projected pixel.")
            .def_readwrite("max_depth_error",
                           &colmap::mvs::StereoFusionOptions::max_depth_error,
                           "float: (Default ``0.01``) Maximum relative "
                           "difference between measured and projected depth.")
            .def_readwrite(
                    "max_normal_error",
                    &colmap::mvs::StereoFusionOptions::max_normal_error,
                    "float: (Default ``10.0``) Maximum angular difference in "
                    "degrees of normals of pixels to be fused.")
            .def_readwrite("check_num_images",
                           &colmap::mvs::StereoFusionOptions::check_num_images,
                           "int: (Default ``50``) Number of overlapping "
                           "images to transitively check for fusing points.")
            .def_readwrite("use_cache",
                           &colmap::mvs::StereoFusionOptions::use_cache,
                           "bool: (Default ``False``) Flag indicating whether "
                           "to use LRU cache or pre-load all data.")
            .def_readwrite(
                    "cache_size", &colmap::mvs::StereoFusionOptions::cache_size,
                    "float: (Default ``32.0``) Cache size in gigabytes for "
                    "fusion. The fusion keeps the bitmaps, depth maps, normal "
                    "maps, and consistency graphs of this number of images in "
                    "memory. A higher value leads to less disk access and "
                    "faster fusion, while a lower value leads to reduced "
                    "memory usage. Note that a single image can consume a lot "
                    "of memory, if the consistency graph is dense.")
            .def_readwrite("bounding_box",
                           &colmap::mvs::StereoFusionOptions::bounding_box,
                           "List((), ()): The bounding box size.")
            // Upstream pycolmap parity: the fusion entry point attached to
            // its options class, forwarding to the fork's StereoFuse
            // wrapper.
            .def(
                    "stereo_fusion",
                    [](const colmap::mvs::StereoFusionOptions& options,
                       const std::string& workspace_path,
                       const std::string& output_path,
                       const std::string& bbox_path,
                       const std::string& stereo_input_type,
                       const std::string& output_type,
                       const std::string& workspace_format,
                       const std::string& pmvs_option_name) {
                        return cloudViewer::StereoFuse(
                                workspace_path, output_path, bbox_path,
                                stereo_input_type, output_type,
                                workspace_format, pmvs_option_name, options);
                    },
                    "workspace_path"_a, "output_path"_a, "bbox_path"_a = "",
                    "stereo_input_type"_a = "geometric",
                    "output_type"_a = "PLY", "workspace_format"_a = "COLMAP",
                    "pmvs_option_name"_a = "option-all",
                    "Fuse the patch-match depth/normal maps into a dense "
                    "point cloud; returns 0 on success.");
}

void pybind_poisson_meshing_options(py::module& m) {
    // cloudViewer.reconstruction.options.PoissonMeshingOptions
    py::class_<colmap::mvs::PoissonMeshingOptions> poisson_meshing_options(
            m, "PoissonMeshingOptions", "Poisson meshing option class.");
    poisson_meshing_options.def(py::init<>())
            .def("check", &colmap::mvs::PoissonMeshingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("point_weight",
                           &colmap::mvs::PoissonMeshingOptions::point_weight,
                           "float: (Default ``1.0``) This floating point value "
                           "specifies the importance that interpolation of the "
                           "point samples is given in the formulation of the "
                           "screened Poisson equation. The results of the "
                           "original (unscreened) Poisson Reconstruction can "
                           "be obtained by setting this value to 0.")
            .def_readwrite(
                    "depth", &colmap::mvs::PoissonMeshingOptions::depth,
                    "int: (Default ``'13'``) This integer is the maximum "
                    "depth of the tree that will be used for surface "
                    "reconstruction. Running at depth d corresponds to solving "
                    "on a voxel grid whose resolution is no larger than 2^d x "
                    "2^d x 2^d. Note that since the reconstructor adapts the "
                    "octree to the sampling density, the specified "
                    "reconstruction depth is only an upper bound.")
            .def_readwrite(
                    "color", &colmap::mvs::PoissonMeshingOptions::color,
                    "float: (Default ``'32.0'``) If specified, the "
                    "reconstruction code assumes that the input is equipped "
                    "with colors and will extrapolate the color values to the "
                    "vertices of the reconstructed mesh. The floating point "
                    "value specifies the relative importance of finer color "
                    "estimates over lower ones.")
            .def_readwrite("trim", &colmap::mvs::PoissonMeshingOptions::trim,
                           "float: (Default ``'10.0'``) This floating point "
                           "values specifies the value for mesh trimming. The "
                           "subset of the mesh with signal value less than the "
                           "trim value is discarded.")
            .def_readwrite("num_threads",
                           &colmap::mvs::PoissonMeshingOptions::num_threads,
                           "int: (Default ``'-1'``) The number of threads used "
                           "for the Poisson reconstruction.");
}

void pybind_delaunay_meshing_options(py::module& m) {
    // cloudViewer.reconstruction.options.DelaunayMeshingOptions
    py::class_<colmap::mvs::DelaunayMeshingOptions> delaunay_meshing_options(
            m, "DelaunayMeshingOptions", "Delaunay meshing option class.");
    delaunay_meshing_options.def(py::init<>())
            .def("check", &colmap::mvs::DelaunayMeshingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("max_proj_dist",
                           &colmap::mvs::DelaunayMeshingOptions::max_proj_dist,
                           "float: (Default ``20.0``) Unify input points into "
                           "one cell in the Delaunay triangulation that fall "
                           "within a reprojected radius of the given pixels.")
            .def_readwrite(
                    "max_depth_dist",
                    &colmap::mvs::DelaunayMeshingOptions::max_depth_dist,
                    "float: (Default ``'0.05'``) Maximum relative depth "
                    "difference between input point and a vertex of an "
                    "existing cell in the Delaunay triangulation, otherwise a "
                    "new vertex is created in the triangulation.")
            .def_readwrite(
                    "visibility_sigma",
                    &colmap::mvs::DelaunayMeshingOptions::visibility_sigma,
                    "float: (Default ``'3.0'``) The standard deviation of wrt. "
                    "the number of images seen by each point. Increasing this "
                    "value decreases the influence of points seen in few "
                    "images.")
            .def_readwrite(
                    "distance_sigma_factor",
                    &colmap::mvs::DelaunayMeshingOptions::distance_sigma_factor,
                    "float: (Default ``'1.0'``) The factor that is applied to "
                    "the computed distance sigma, which is automatically "
                    "computed as the 25th percentile of edge lengths. A higher "
                    "value will increase the smoothness of the surface.")
            .def_readwrite("quality_regularization",
                           &colmap::mvs::DelaunayMeshingOptions::
                                   quality_regularization,
                           "float: (Default ``'1.0'``) A higher quality "
                           "regularization leads to a smoother surface.")
            .def_readwrite(
                    "max_side_length_factor",
                    &colmap::mvs::DelaunayMeshingOptions::
                            max_side_length_factor,
                    "float: (Default ``'25.0'``) Filtering thresholds for "
                    "outlier surface mesh faces. If the longest side of a mesh "
                    "face (longest out of 3) exceeds the side lengths of all "
                    "faces at a certain percentile by the given factor, then "
                    "it is considered an outlier mesh face and discarded.")
            .def_readwrite(
                    "max_side_length_percentile",
                    &colmap::mvs::DelaunayMeshingOptions::
                            max_side_length_percentile,
                    "float: (Default ``'95.0'``) Filtering thresholds for "
                    "outlier surface mesh faces. If the longest side of a mesh "
                    "face (longest out of 3) exceeds the side lengths of all "
                    "faces at a certain percentile by the given factor, then "
                    "it is considered an outlier mesh face and discarded.")
            .def_readwrite("num_threads",
                           &colmap::mvs::DelaunayMeshingOptions::num_threads,
                           "int: (Default ``'-1'``) The number of threads to "
                           "use for reconstruction. Default is all threads.")
            // Upstream pycolmap parity: the meshing entry points attached
            // to their options class. The fork engine exposes the same
            // operations as free functions (Sparse/DenseDelaunayMeshing,
            // PoissonMeshing).
            .def(
                    "sparse_delaunay_meshing",
                    [](const colmap::mvs::DelaunayMeshingOptions& options,
                       const std::filesystem::path& input_path,
                       const std::filesystem::path& output_path) {
                        colmap::mvs::SparseDelaunayMeshing(options, input_path,
                                                           output_path);
                    },
                    "input_path"_a, "output_path"_a,
                    "Delaunay mesh the sparse reconstruction; input is a "
                    "sparse COLMAP model folder.")
            .def(
                    "dense_delaunay_meshing",
                    [](const colmap::mvs::DelaunayMeshingOptions& options,
                       const std::filesystem::path& input_path,
                       const std::filesystem::path& output_path) {
                        colmap::mvs::DenseDelaunayMeshing(options, input_path,
                                                          output_path);
                    },
                    "input_path"_a, "output_path"_a,
                    "Delaunay mesh the dense workspace (fused point cloud "
                    "with visibility); input is a processed dense COLMAP "
                    "workspace folder.")
            .def(
                    "poisson_meshing",
                    [](const colmap::mvs::DelaunayMeshingOptions&,
                       const colmap::mvs::PoissonMeshingOptions& poisson,
                       const std::filesystem::path& input_path,
                       const std::filesystem::path& output_path) {
                        return colmap::mvs::PoissonMeshing(poisson, input_path,
                                                           output_path);
                    },
                    "poisson_meshing_options"_a, "input_path"_a,
                    "output_path"_a,
                    "Run Poisson surface reconstruction on a fused point "
                    "cloud; returns 0 on success.");

    // Upstream pycolmap parity (src/pycolmap/pipeline/mvs.cc):
    // MeshSimplificationOptions (the simplify_mesh entry point has no fork
    // engine counterpart and is recorded in the alignment manifest).
    py::class_<colmap::mvs::MeshSimplificationOptions>
            mesh_simplification_options(m, "MeshSimplificationOptions",
                                        "Options for mesh simplification.");
    mesh_simplification_options.def(py::init<>())
            .def("check", &colmap::mvs::MeshSimplificationOptions::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "target_face_ratio",
                    &colmap::mvs::MeshSimplificationOptions::target_face_ratio,
                    "float: (Default ``0.1``) Fraction of faces to retain, in "
                    "(0, 1].")
            .def_readwrite("max_error",
                           &colmap::mvs::MeshSimplificationOptions::max_error,
                           "float: (Default ``0.0``) Maximum quadric error per "
                           "collapse; 0 = disabled.")
            .def_readwrite(
                    "boundary_weight",
                    &colmap::mvs::MeshSimplificationOptions::boundary_weight,
                    "float: (Default ``1000.0``) Penalty weight for boundary "
                    "edges; 0 = disabled.")
            .def_readwrite(
                    "num_threads",
                    &colmap::mvs::MeshSimplificationOptions::num_threads,
                    "int: (Default ``-1``) Number of threads; default is all "
                    "threads.")
            .def_readwrite(
                    "interpolate_colors",
                    &colmap::mvs::MeshSimplificationOptions::interpolate_colors,
                    "bool: (Default ``True``) Blend colors on collapse vs. "
                    "pick lower-error vertex.");
}

void pybind_mesh_post_processing_options(py::module& m) {
    py::class_<colmap::mvs::MeshPostProcessingOptions> options(
            m, "MeshPostProcessingOptions",
            "meshoptimizer cleanup and smoothing options.");
    options.def(py::init<>())
            .def("check", &colmap::mvs::MeshPostProcessingOptions::Check)
            .def_readwrite("enabled",
                           &colmap::mvs::MeshPostProcessingOptions::enabled)
            .def_readwrite("remove_small_components",
                           &colmap::mvs::MeshPostProcessingOptions::
                                   remove_small_components)
            .def_readwrite("remove_degenerate_faces",
                           &colmap::mvs::MeshPostProcessingOptions::
                                   remove_degenerate_faces)
            .def_readwrite("simplify",
                           &colmap::mvs::MeshPostProcessingOptions::simplify)
            .def_readwrite("smooth",
                           &colmap::mvs::MeshPostProcessingOptions::smooth)
            .def_readwrite(
                    "preserve_boundary",
                    &colmap::mvs::MeshPostProcessingOptions::preserve_boundary)
            .def_readwrite("prune_error",
                           &colmap::mvs::MeshPostProcessingOptions::prune_error)
            .def_readwrite(
                    "target_face_ratio",
                    &colmap::mvs::MeshPostProcessingOptions::target_face_ratio)
            .def_readwrite(
                    "simplify_error",
                    &colmap::mvs::MeshPostProcessingOptions::simplify_error)
            .def_readwrite(
                    "max_aspect_ratio",
                    &colmap::mvs::MeshPostProcessingOptions::max_aspect_ratio)
            .def_readwrite("smoothing_iterations",
                           &colmap::mvs::MeshPostProcessingOptions::
                                   smoothing_iterations)
            .def_readwrite(
                    "smoothing_lambda",
                    &colmap::mvs::MeshPostProcessingOptions::smoothing_lambda)
            .def_readwrite(
                    "smoothing_mu",
                    &colmap::mvs::MeshPostProcessingOptions::smoothing_mu);
}

void pybind_reconstruction_options_early(py::module& m) {
    // Registered before the feature binder: the feature extraction and
    // import entry points take these as default-argument values, which
    // pybind converts at registration time. def_submodule is idempotent
    // for an already-created submodule (PyImport_AddModule returns the
    // same module object), so creating "pipeline"/"feature" here is safe.
    py::module m_pipeline = m.attr("pipeline")
                                    ? m.attr("pipeline").cast<py::module>()
                                    : m.def_submodule("pipeline");
    py::module m_feature = m.attr("feature")
                                   ? m.attr("feature").cast<py::module>()
                                   : m.def_submodule("feature");
    // upstream pipeline/images.cc
    pybind_image_reader_options(m_pipeline);
    // upstream feature/extraction.cc and feature/matching.cc (all of
    // these are default-argument values of the feature entry points)
    pybind_sift_extraction_options(m_feature);
    pybind_sift_matching_options(m_feature);
    // upstream pipeline/match_features.cc binds the *PairingOptions family
    // in the pipeline binder; the fork-only FeaturePairsMatchingOptions
    // (controllers/pairing.h) lands here.
    pybind_featurepairs_matching_options(m_pipeline);
    // upstream estimators/bundle_adjustment.cc and sfm/incremental_*.cc
    // (default-argument values of the image/sfm binder entry points)
    py::module m_estimators = m.attr("estimators")
                                      ? m.attr("estimators").cast<py::module>()
                                      : m.def_submodule("estimators");
    py::module m_sfm = m.attr("sfm") ? m.attr("sfm").cast<py::module>()
                                     : m.def_submodule("sfm");
    pybind_bundle_adjustment_options(m_estimators);
    pybind_incremental_triangulator_options(m_sfm);
    pybind_incremental_mapper_options(m_sfm);
    // upstream pipeline/mvs.cc and pipeline/meshing.cc (the fork engine
    // owns these under mvs/, so they land on the mvs submodule)
    py::module m_mvs = m.attr("mvs") ? m.attr("mvs").cast<py::module>()
                                     : m.def_submodule("mvs");
    pybind_patch_match_options(m_mvs);
    pybind_stereo_fusion_options(m_mvs);
    pybind_poisson_meshing_options(m_mvs);
    pybind_delaunay_meshing_options(m_mvs);
    pybind_mesh_post_processing_options(m_mvs);
}

// The historical catch-all "options" submodule is dissolved: every options
// class registers on the functional submodule that owns it (see
// pybind_reconstruction_options_early, which must run FIRST so that the
// default-argument values of the other binders' entry points already
// resolve).

}  // namespace options
}  // namespace reconstruction
}  // namespace cloudViewer
