// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Upstream pycolmap parity bindings (src/pycolmap) for the pipeline options
// family that the fork engine owns. Class/method names follow the upstream
// surface; member names follow the fork engine API with the divergence
// documented per binding. Engine capabilities that upstream exposes but the
// fork does not have (the step-wise IncrementalPipeline class, the pairing
// abstraction, the exe-level Run*Impl entry points) are intentionally NOT
// fabricated here and are recorded in colmap_alignment_manifest.json.

#include "pybind/reconstruction/pipeline/pipeline.h"

#include <filesystem>
#include <memory>
#include <optional>

#include "controllers/global_pipeline.h"
#include "controllers/hierarchical_pipeline.h"
#include "controllers/incremental_pipeline.h"
#include "estimators/view_graph_calibration.h"
#include "image/undistortion.h"
#include "image/warp.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/hash_container_casters.h"
#include "pybind/reconstruction/pipeline/match_features.h"
#include "scene/database.h"
#include "scene/scene_clustering.h"
#include "scene/synthetic.h"
#include "sensor/bitmap.h"
#include "sfm/global_mapper.h"

namespace cloudViewer {
namespace reconstruction {
namespace pipeline {

void pybind_pipeline(py::module& m, py::module& m_root) {
    // ``m`` is the pipeline submodule; ``m_root`` is the
    // reconstruction root, used to reach the sibling submodules
    // that own the option classes relocated to their upstream
    // source-file placement.
    py::module m_sfm = m_root.attr("sfm").cast<py::module>();
    py::module m_scene = m_root.attr("scene").cast<py::module>();
    py::module m_image = m_root.attr("image").cast<py::module>();

    // ------------------------------------------------------------------
    // Upstream source-file placement (src/pycolmap):
    //   image/undistortion.cc      -> UndistortCameraOptions,
    //                                 WarpImageOptions
    //   sfm/global_mapper.cc       -> GlobalPipelineOptions,
    //                                 GlobalMapperOptions
    //   sfm/hierarchical_mapper.cc -> HierarchicalPipelineOptions,
    //                                 SceneClusteringOptions
    //   scene/synthetic.cc         -> Synthetic*Options and the
    //                                 synthesize_* entry points
    //   pipeline/sfm.cc            -> ViewGraphCalibrationOptions,
    //                                 calibrate_view_graph (stay here)
    // The fork mirrors that placement onto its functional submodules.
    // ------------------------------------------------------------------

    // ================= image submodule =================
    // ------------------------------------------------------------------
    // WarpImageOptions (fork: image/warp.h; upstream binds it as part of
    // the undistortion family)
    // ------------------------------------------------------------------
    py::class_<colmap::WarpImageOptions> warp_image_options(
            m_image, "WarpImageOptions",
            "Options for direct target-resolution warping used by "
            "undistortion and stereo rectification.");
    warp_image_options.def(py::init<>())
            .def_readwrite("interpolation",
                           &colmap::WarpImageOptions::interpolation,
                           "Source-image sampling interpolation method.")
            .def_readwrite(
                    "direct_warp_min_scale",
                    &colmap::WarpImageOptions::direct_warp_min_scale,
                    "float: (Default ``0.5``) Minimum scale for which direct "
                    "warping is attempted before falling back to the "
                    "source-resolution pass.");

    // ------------------------------------------------------------------
    // UndistortCameraOptions (upstream pipeline/undistort_camera.cc)
    // ------------------------------------------------------------------
    py::class_<colmap::UndistortCameraOptions> undistort_camera_options(
            m_image, "UndistortCameraOptions",
            "Options for undistorting images and cameras.");
    undistort_camera_options.def(py::init<>())
            .def_readwrite(
                    "blank_pixels",
                    &colmap::UndistortCameraOptions::blank_pixels,
                    "float: (Default ``0.0``) The amount of blank pixels in "
                    "the undistorted image in the range [0, 1].")
            .def_readwrite("min_scale",
                           &colmap::UndistortCameraOptions::min_scale,
                           "float: (Default ``0.2``) Minimum scale change of "
                           "the camera used to satisfy the blank pixel "
                           "constraint.")
            .def_readwrite("max_scale",
                           &colmap::UndistortCameraOptions::max_scale,
                           "float: (Default ``2.0``) Maximum scale change of "
                           "the camera used to satisfy the blank pixel "
                           "constraint.")
            .def_readwrite(
                    "max_image_size",
                    &colmap::UndistortCameraOptions::max_image_size,
                    "int: (Default ``-1``) Maximum image size in terms of "
                    "width or height of the undistorted camera.")
            .def_readwrite("roi_min_x",
                           &colmap::UndistortCameraOptions::roi_min_x,
                           "float: (Default ``0.0``) The lower-left x "
                           "coordinate factor of the ROI in [0, 1].")
            .def_readwrite("roi_min_y",
                           &colmap::UndistortCameraOptions::roi_min_y,
                           "float: (Default ``0.0``) The lower-left y "
                           "coordinate factor of the ROI in [0, 1].")
            .def_readwrite("roi_max_x",
                           &colmap::UndistortCameraOptions::roi_max_x,
                           "float: (Default ``1.0``) The upper-right x "
                           "coordinate factor of the ROI in [0, 1].")
            .def_readwrite("roi_max_y",
                           &colmap::UndistortCameraOptions::roi_max_y,
                           "float: (Default ``1.0``) The upper-right y "
                           "coordinate factor of the ROI in [0, 1].")
            .def_readwrite("warp_options",
                           &colmap::UndistortCameraOptions::warp_options,
                           "WarpImageOptions: Controls direct "
                           "target-resolution warping for undistortion and "
                           "stereo rectification.")
            // Upstream pycolmap parity: the undistortion entry points
            // attached to their options class.
            .def(
                    "undistort_camera",
                    [](const colmap::UndistortCameraOptions& options,
                       const colmap::Camera& distorted_camera) {
                        return colmap::UndistortCamera(options,
                                                       distorted_camera);
                    },
                    "distorted_camera"_a,
                    "Undistort the camera intrinsics to a pinhole model and "
                    "return the undistorted camera.")
            .def(
                    "undistort_image",
                    [](const colmap::UndistortCameraOptions& options,
                       const colmap::Bitmap& distorted_image,
                       const colmap::Camera& distorted_camera) {
                        colmap::Bitmap undistorted_image;
                        colmap::Camera undistorted_camera;
                        colmap::UndistortImage(
                                options, distorted_image, distorted_camera,
                                &undistorted_image, &undistorted_camera);
                        py::dict out;
                        out["undistorted_image"] = undistorted_image;
                        out["undistorted_camera"] = undistorted_camera;
                        return out;
                    },
                    "distorted_image"_a, "distorted_camera"_a,
                    "Undistort an image to a pinhole viewing geometry; "
                    "returns a dict with undistorted_image and "
                    "undistorted_camera.");

    // ================= sfm submodule =================
    // ------------------------------------------------------------------
    // SceneClusteringOptions (fork: scene/scene_clustering.h; upstream
    // pipeline/scene_clustering.cc)
    // ------------------------------------------------------------------
    py::class_<colmap::SceneClustering::Options> scene_clustering_options(
            m_sfm, "SceneClusteringOptions",
            "Options for hierarchical scene clustering used by the "
            "hierarchical pipeline.");
    scene_clustering_options.def(py::init<>())
            .def("check", &colmap::SceneClustering::Options::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "is_hierarchical",
                    &colmap::SceneClustering::Options::is_hierarchical,
                    "bool: (Default ``True``) Whether to cluster the scene "
                    "hierarchically.")
            .def_readwrite("branching",
                           &colmap::SceneClustering::Options::branching,
                           "int: (Default ``2``) The branching factor of the "
                           "hierarchical clustering.")
            .def_readwrite(
                    "image_overlap",
                    &colmap::SceneClustering::Options::image_overlap,
                    "int: (Default ``50``) The overlap between different "
                    "clusters in number of images.")
            .def_readwrite(
                    "num_image_matches",
                    &colmap::SceneClustering::Options::num_image_matches,
                    "int: (Default ``20``) The number of images to consider "
                    "when determining the overlap.")
            .def_readwrite(
                    "leaf_max_num_images",
                    &colmap::SceneClustering::Options::leaf_max_num_images,
                    "int: (Default ``500``) The maximum number of images in "
                    "a leaf node cluster.");

    // ------------------------------------------------------------------
    // HierarchicalPipelineOptions (upstream
    // controllers/hierarchical_pipeline.h)
    // ------------------------------------------------------------------
    py::class_<colmap::HierarchicalPipelineOptions>
            hierarchical_pipeline_options(
                    m_sfm, "HierarchicalPipelineOptions",
                    "Options for the hierarchical mapping pipeline.");
    hierarchical_pipeline_options.def(py::init<>())
            .def("check", &colmap::HierarchicalPipelineOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("image_path",
                           &colmap::HierarchicalPipelineOptions::image_path,
                           "str: The image path at which to find the images to "
                           "extract point colors. If not specified, all point "
                           "colors will be black.")
            .def_readwrite(
                    "init_num_trials",
                    &colmap::HierarchicalPipelineOptions::init_num_trials,
                    "int: (Default ``10``) The maximum number of trials "
                    "to initialize a cluster.")
            .def_readwrite("num_threads",
                           &colmap::HierarchicalPipelineOptions::num_threads,
                           "int: (Default ``-1``) The total number of threads "
                           "for the hierarchical pipeline, divided across "
                           "workers. incremental_options.num_threads is "
                           "ignored in favor of this option.")
            .def_readwrite("num_workers",
                           &colmap::HierarchicalPipelineOptions::num_workers,
                           "int: (Default ``-1``) The number of workers used "
                           "to reconstruct clusters in parallel.")
            .def_readwrite(
                    "clustering_options",
                    &colmap::HierarchicalPipelineOptions::clustering_options,
                    "Options for clustering the scene graph.")
            .def_readwrite(
                    "incremental_options",
                    &colmap::HierarchicalPipelineOptions::incremental_options,
                    "Options used to reconstruct each cluster individually.");

    // ------------------------------------------------------------------
    // GlobalMapperOptions (fork: sfm/global_mapper.h; upstream
    // estimators/global_mapper.h via pycolmap GlobalMapperOptions)
    // ------------------------------------------------------------------
    py::class_<colmap::GlobalMapperOptions> global_mapper_options(
            m_sfm, "GlobalMapperOptions",
            "Options for the global mapping stage (GLOMAP-style rotation "
            "averaging, global positioning, and joint bundle adjustment).");
    global_mapper_options.def(py::init<>())
            .def("get_rotation_averaging",
                 &colmap::GlobalMapperOptions::RotationAveraging,
                 "Get a copy of the rotation averaging options.")
            .def("get_global_positioning",
                 &colmap::GlobalMapperOptions::GlobalPositioning,
                 "Get a copy of the global positioning options.")
            .def("get_bundle_adjustment",
                 &colmap::GlobalMapperOptions::BundleAdjustment,
                 "Get a copy of the bundle adjustment options.")
            .def("get_retriangulation",
                 &colmap::GlobalMapperOptions::Retriangulation,
                 "Get a copy of the retriangulation options.")
            .def_readwrite("num_threads",
                           &colmap::GlobalMapperOptions::num_threads,
                           "int: (Default ``-1``) Number of threads.")
            .def_readwrite("random_seed",
                           &colmap::GlobalMapperOptions::random_seed,
                           "int: (Default ``-1``) PRNG seed for all "
                           "stochastic methods.")
            .def_readwrite("image_path",
                           &colmap::GlobalMapperOptions::image_path,
                           "str: The path to the image folder used to extract "
                           "point colors.")
            .def_readwrite(
                    "refine_sensor_from_rig",
                    &colmap::GlobalMapperOptions::refine_sensor_from_rig,
                    "bool: (Default ``True``) Whether to refine the sensor "
                    "from rig extrinsics.")
            .def_readwrite(
                    "rotation_averaging",
                    &colmap::GlobalMapperOptions::rotation_averaging,
                    "RotationEstimatorOptions: Rotation averaging options.")
            .def_readwrite(
                    "global_positioning",
                    &colmap::GlobalMapperOptions::global_positioning,
                    "GlobalPositionerOptions: Global positioning options.")
            .def_readwrite(
                    "bundle_adjustment",
                    &colmap::GlobalMapperOptions::bundle_adjustment,
                    "BundleAdjustmentOptions: Bundle adjustment options.")
            .def_readwrite(
                    "retriangulation",
                    &colmap::GlobalMapperOptions::retriangulation,
                    "IncrementalTriangulatorOptions: Retriangulation options.")
            .def_readwrite("track_intra_image_consistency_threshold",
                           &colmap::GlobalMapperOptions::
                                   track_intra_image_consistency_threshold,
                           "float: (Default ``10.0``) Intra-image consistency "
                           "threshold for track establishment.")
            .def_readwrite("track_required_tracks_per_view",
                           &colmap::GlobalMapperOptions::
                                   track_required_tracks_per_view,
                           "int: Minimum number of tracks required per view.")
            .def_readwrite(
                    "track_min_num_views_per_track",
                    &colmap::GlobalMapperOptions::track_min_num_views_per_track,
                    "int: (Default ``3``) Minimum number of views per track.")
            .def_readwrite("keep_max_num_tracks",
                           &colmap::GlobalMapperOptions::keep_max_num_tracks,
                           "int: Maximum number of tracks to keep.")
            .def_readwrite(
                    "max_angular_reproj_error_deg",
                    &colmap::GlobalMapperOptions::max_angular_reproj_error_deg,
                    "float: (Default ``1.0``) Maximum angular reprojection "
                    "error in degrees for global positioning.")
            .def_readwrite(
                    "max_normalized_reproj_error",
                    &colmap::GlobalMapperOptions::max_normalized_reproj_error,
                    "float: (Default ``0.01``) Maximum normalized "
                    "reprojection error for bundle adjustment.")
            .def_readwrite("min_tri_angle_deg",
                           &colmap::GlobalMapperOptions::min_tri_angle_deg,
                           "float: (Default ``1.0``) Minimum triangulation "
                           "angle in degrees for triangulation.")
            .def_readwrite("ba_gpu_index",
                           &colmap::GlobalMapperOptions::ba_gpu_index,
                           "str: (Default ``'-1'``) Index of CUDA GPU to use "
                           "for BA, if available.")
            .def_readwrite("ba_num_iterations",
                           &colmap::GlobalMapperOptions::ba_num_iterations,
                           "int: (Default ``3``) Number of bundle adjustment "
                           "rounds.")
            .def_readwrite(
                    "ba_skip_fixed_rotation_stage",
                    &colmap::GlobalMapperOptions::ba_skip_fixed_rotation_stage,
                    "bool: (Default ``False``) Skip the bundle adjustment "
                    "stage with fixed rotations.")
            .def_readwrite(
                    "ba_skip_joint_optimization_stage",
                    &colmap::GlobalMapperOptions::
                            ba_skip_joint_optimization_stage,
                    "bool: (Default ``False``) Skip the joint optimization "
                    "bundle adjustment stage.")
            .def_readwrite(
                    "skip_rotation_averaging",
                    &colmap::GlobalMapperOptions::skip_rotation_averaging,
                    "bool: (Default ``False``) Whether to skip rotation "
                    "averaging.")
            .def_readwrite(
                    "skip_track_establishment",
                    &colmap::GlobalMapperOptions::skip_track_establishment,
                    "bool: (Default ``False``) Whether to skip track "
                    "establishment.")
            .def_readwrite(
                    "skip_global_positioning",
                    &colmap::GlobalMapperOptions::skip_global_positioning,
                    "bool: (Default ``False``) Whether to skip global "
                    "positioning.")
            .def_readwrite("skip_bundle_adjustment",
                           &colmap::GlobalMapperOptions::skip_bundle_adjustment,
                           "bool: (Default ``False``) Whether to skip bundle "
                           "adjustment.")
            .def_readwrite("skip_retriangulation",
                           &colmap::GlobalMapperOptions::skip_retriangulation,
                           "bool: (Default ``False``) Whether to skip "
                           "retriangulation.");

    // ------------------------------------------------------------------
    // GlobalPipelineOptions (fork: controllers/global_pipeline.h; upstream
    // pipeline/sfm.cc GlobalPipelineOptions)
    // ------------------------------------------------------------------
    py::class_<colmap::GlobalPipelineOptions> global_pipeline_options(
            m_sfm, "GlobalPipelineOptions",
            "Options for the global structure-from-motion pipeline "
            "(GLOMAP-style).");
    global_pipeline_options.def(py::init<>())
            .def_readwrite(
                    "min_num_matches",
                    &colmap::GlobalPipelineOptions::min_num_matches,
                    "int: (Default ``15``) The minimum number of matches for "
                    "inlier matches to be considered.")
            .def_readwrite(
                    "ignore_watermarks",
                    &colmap::GlobalPipelineOptions::ignore_watermarks,
                    "bool: (Default ``False``) Whether to ignore the inlier "
                    "matches of watermark image pairs.")
            .def_readwrite("image_names",
                           &colmap::GlobalPipelineOptions::image_names,
                           "list(str, ...): Names of images to reconstruct. "
                           "If empty, all images are used.")
            .def_readwrite("image_path",
                           &colmap::GlobalPipelineOptions::image_path,
                           "str: The image path at which to find the images "
                           "to extract point colors.")
            .def_readwrite("num_threads",
                           &colmap::GlobalPipelineOptions::num_threads,
                           "int: (Default ``-1``) Number of threads for "
                           "parallel processing.")
            .def_readwrite("random_seed",
                           &colmap::GlobalPipelineOptions::random_seed,
                           "int: (Default ``-1``) Random seed for "
                           "reproducibility.")
            .def_readwrite(
                    "decompose_relative_pose",
                    &colmap::GlobalPipelineOptions::decompose_relative_pose,
                    "bool: (Default ``True``) Whether to decompose relative "
                    "poses from two-view geometries.")
            .def_readwrite(
                    "multiple_models",
                    &colmap::GlobalPipelineOptions::multiple_models,
                    "bool: (Default ``True``) If true, reconstruct every "
                    "connected component of the view graph (one model per "
                    "component). If false, reconstruct only the largest "
                    "connected component.")
            .def_readwrite("min_model_size",
                           &colmap::GlobalPipelineOptions::min_model_size,
                           "int: (Default ``3``) Minimum number of registered "
                           "frames for a reconstruction to be kept.")
            .def_readwrite("mapper", &colmap::GlobalPipelineOptions::mapper,
                           "GlobalMapperOptions: Options for the global "
                           "mapper.");

    // ================= scene submodule =================
    // ------------------------------------------------------------------
    // Synthetic dataset noise / image generation (upstream
    // scene/synthetic.cc; the fork engine owns both option structs and
    // entry points, see scene/synthetic.h).
    // ------------------------------------------------------------------
    py::class_<colmap::SyntheticNoiseOptions> synthetic_noise_options(
            m_scene, "SyntheticNoiseOptions",
            "Perturbation of a synthetic reconstruction (poses, points, "
            "priors).");
    synthetic_noise_options.def(py::init<>())
            .def_readwrite("rig_from_world_translation_stddev",
                           &colmap::SyntheticNoiseOptions::
                                   rig_from_world_translation_stddev,
                           "float: (Default ``0.0``) Translation stddev of "
                           "the rig-from-world pose in meters.")
            .def_readwrite("rig_from_world_rotation_stddev",
                           &colmap::SyntheticNoiseOptions::
                                   rig_from_world_rotation_stddev,
                           "float: (Default ``0.0``) Rotation stddev around "
                           "the rig z-axis in degrees.")
            .def_readwrite("point3D_stddev",
                           &colmap::SyntheticNoiseOptions::point3D_stddev,
                           "float: (Default ``0.0``) 3D point stddev in "
                           "meters.")
            .def_readwrite("point2D_stddev",
                           &colmap::SyntheticNoiseOptions::point2D_stddev,
                           "float: (Default ``0.0``) 2D observation stddev in "
                           "pixels.")
            .def_readwrite(
                    "prior_position_stddev",
                    &colmap::SyntheticNoiseOptions::prior_position_stddev,
                    "float: (Default ``1.5``) Translational stddev of "
                    "the prior position in meters.")
            .def_readwrite("prior_gravity_stddev",
                           &colmap::SyntheticNoiseOptions::prior_gravity_stddev,
                           "float: (Default ``1.0``) Rotational stddev of the "
                           "prior gravity in degrees.");

    py::class_<colmap::SyntheticImageOptions> synthetic_image_options(
            m_scene, "SyntheticImageOptions",
            "Options for generating synthetic observation images.");
    synthetic_image_options.def(py::init<>())
            .def_readwrite("feature_peak_radius",
                           &colmap::SyntheticImageOptions::feature_peak_radius,
                           "int: (Default ``2``) Radius of the bright feature "
                           "peak in pixels.")
            .def_readwrite("feature_patch_radius",
                           &colmap::SyntheticImageOptions::feature_patch_radius,
                           "int: (Default ``15``) Radius of the synthetic "
                           "patch in pixels.")
            .def_readwrite("feature_patch_max_brightness",
                           &colmap::SyntheticImageOptions::
                                   feature_patch_max_brightness,
                           "int: (Default ``128``) Maximum brightness of the "
                           "synthetic patches.");

    m_scene.def(
            "synthesize_noise",
            [](const colmap::SyntheticNoiseOptions& options,
               colmap::Reconstruction* reconstruction,
               colmap::Database* database) {
                colmap::SynthesizeNoise(options, reconstruction, database);
            },
            "options"_a, "reconstruction"_a, "database"_a = py::none(),
            "Perturb a synthetic reconstruction with noise.");

    m_scene.def(
            "synthesize_dataset",
            [](const colmap::SyntheticDatasetOptions& options,
               colmap::Reconstruction* reconstruction,
               colmap::Database* database) {
                colmap::SynthesizeDataset(options, reconstruction, database);
            },
            "options"_a, "reconstruction"_a, "database"_a = py::none(),
            "Synthesize a random reconstruction (rigs, cameras, frames, "
            "images, observations, points) into the given reconstruction.");

    m_scene.def(
            "synthesize_images",
            [](const colmap::SyntheticImageOptions& options,
               const colmap::Reconstruction& reconstruction,
               const std::string& image_path) {
                colmap::SynthesizeImages(options, reconstruction,
                                         std::filesystem::path(image_path));
            },
            "options"_a, "reconstruction"_a, "image_path"_a,
            "Generate synthetic observation images for the reconstruction "
            "under the given directory.");

    // ================= pipeline submodule (own surface) =================
    // ------------------------------------------------------------------
    // ViewGraphCalibrationOptions (fork: estimators/view_graph_calibration.h)
    // NOTE: the fork options embed a raw ceres::Solver::Options member which
    // is not bound (no pyceres equivalent in this repository); the
    // constructor defaults (num_threads=-1, max_num_iterations=100,
    // function_tolerance=1e-5) always apply. Recorded in the alignment
    // manifest.
    // ------------------------------------------------------------------
    py::class_<colmap::ViewGraphCalibrationOptions>
            view_graph_calibration_options(
                    m, "ViewGraphCalibrationOptions",
                    "Options for calibrating focal lengths from fundamental "
                    "matrices and upgrading two-view geometries to "
                    "CALIBRATED in the database.");
    view_graph_calibration_options.def(py::init<>())
            .def_readwrite("random_seed",
                           &colmap::ViewGraphCalibrationOptions::random_seed,
                           "int: (Default ``-1``) PRNG seed for RANSAC based "
                           "relative pose re-estimation.")
            .def_readwrite(
                    "cross_validate_prior_focal_lengths",
                    &colmap::ViewGraphCalibrationOptions::
                            cross_validate_prior_focal_lengths,
                    "bool: (Default ``True``) Whether to cross-validate the "
                    "prior focal lengths.")
            .def_readwrite(
                    "min_calibrated_pair_ratio",
                    &colmap::ViewGraphCalibrationOptions::
                            min_calibrated_pair_ratio,
                    "float: (Default ``0.5``) Minimum ratio of calibrated "
                    "pairs required.")
            .def_readwrite(
                    "reestimate_relative_pose",
                    &colmap::ViewGraphCalibrationOptions::
                            reestimate_relative_pose,
                    "bool: (Default ``True``) Whether to re-estimate the "
                    "relative pose from the upgraded two-view geometries.")
            .def_readwrite(
                    "min_focal_length_ratio",
                    &colmap::ViewGraphCalibrationOptions::
                            min_focal_length_ratio,
                    "float: (Default ``0.1``) Minimum focal length ratio for "
                    "a pair to be considered calibrated.")
            .def_readwrite(
                    "max_focal_length_ratio",
                    &colmap::ViewGraphCalibrationOptions::
                            max_focal_length_ratio,
                    "float: (Default ``10``) Maximum focal length ratio for "
                    "a pair to be considered calibrated.")
            .def_readwrite(
                    "max_calibration_error",
                    &colmap::ViewGraphCalibrationOptions::max_calibration_error,
                    "float: (Default ``2.0``) Maximum calibration error for "
                    "a pair to be used.")
            .def_readwrite(
                    "loss_function_scale",
                    &colmap::ViewGraphCalibrationOptions::loss_function_scale,
                    "float: (Default ``0.01``) Scale of the robust loss "
                    "function.")
            .def_readwrite(
                    "relpose_max_error",
                    &colmap::ViewGraphCalibrationOptions::relpose_max_error,
                    "float: (Default ``1.0``) Maximum error in pixels for "
                    "relative pose estimation.")
            .def_readwrite(
                    "relpose_min_num_inliers",
                    &colmap::ViewGraphCalibrationOptions::
                            relpose_min_num_inliers,
                    "int: (Default ``30``) Minimum number of inliers for "
                    "relative pose estimation.")
            .def_readwrite("relpose_min_inlier_ratio",
                           &colmap::ViewGraphCalibrationOptions::
                                   relpose_min_inlier_ratio,
                           "float: (Default ``0.25``) Minimum inlier ratio for "
                           "relative pose estimation.");

    // ------------------------------------------------------------------
    // Free functions (upstream pipeline/sfm.cc; the fork engine exposes the
    // underlying operations directly instead of the upstream Run*Impl
    // wrappers)
    // ------------------------------------------------------------------
    // NOTE: the historical "free(): invalid pointer" crash and the
    // follow-up TypeError observed around synthesize_dataset are both
    // resolved: they traced to the pybind-target/libColmapLib macro
    // divergence (see libs/Python/pybind/CMakeLists.txt - the
    // EIGEN_MAX_ALIGN_BYTES / engine-feature-macro mirror). With the full
    // mirror in place the binding below passes, matching the native
    // synthetic_test (19/19).

    m.def(
            "calibrate_view_graph",
            [](const colmap::ViewGraphCalibrationOptions& options,
               const std::string& database_path) {
                auto database = colmap::Database::Open(database_path);
                return colmap::CalibrateViewGraph(options, database.get());
            },
            "options"_a, "database_path"_a,
            "Calibrate focal lengths from fundamental matrices and upgrade "
            "two-view geometries to CALIBRATED in the database. Run before "
            "global_mapping when reliable intrinsics are unavailable.");

    // ------------------------------------------------------------------
    // Upstream pycolmap parity (src/pycolmap/sfm/{global_mapper,
    // hierarchical_mapper,incremental_mapper}.cc): the pipeline controller
    // classes. The fork controllers keep the legacy legacy option-split
    // constructors, so the Hierarchical/Incremental wrappers materialize
    // the fork controller with default sub-options (documented divergence:
    // the upstream options carry nested clustering/mapper sub-options).
    // ------------------------------------------------------------------
    {
        py::class_<colmap::GlobalPipeline,
                   std::shared_ptr<colmap::GlobalPipeline>>(m, "GlobalPipeline")
                .def(py::init<colmap::GlobalPipelineOptions,
                              std::shared_ptr<colmap::Database>,
                              std::shared_ptr<colmap::ReconstructionManager>>(),
                     "options"_a, "database"_a, "reconstruction_manager"_a)
                .def("add_callback", &colmap::GlobalPipeline::AddCallback,
                     "id"_a, "func"_a,
                     "Add a callback function for the given callback type.")
                .def("callback", &colmap::GlobalPipeline::Callback, "id"_a,
                     "Invoke the callback for the given callback type.");

        py::enum_<colmap::GlobalPipeline::CallbackType>(
                m, "GlobalPipelineCallback")
                .value("MODEL_UPDATE_CALLBACK",
                       colmap::GlobalPipeline::CallbackType::
                               MODEL_UPDATE_CALLBACK)
                .export_values();

        py::enum_<colmap::IncrementalPipeline::CallbackType>(
                m, "IncrementalPipelineCallback")
                .value("INITIAL_IMAGE_PAIR_REG_CALLBACK",
                       colmap::IncrementalPipeline::CallbackType::
                               INITIAL_IMAGE_PAIR_REG_CALLBACK)
                .value("NEXT_IMAGE_REG_CALLBACK",
                       colmap::IncrementalPipeline::CallbackType::
                               NEXT_IMAGE_REG_CALLBACK)
                .value("LAST_IMAGE_REG_CALLBACK",
                       colmap::IncrementalPipeline::CallbackType::
                               LAST_IMAGE_REG_CALLBACK)
                .export_values();

        // Upstream constructs from a Database object; the fork controller
        // opens the database from the path (documented divergence). The
        // controller keeps a raw manager pointer, so the returned shared_ptr
        // deleter holds the manager alive until the controller is destroyed.
        py::class_<colmap::HierarchicalPipeline,
                   std::shared_ptr<colmap::HierarchicalPipeline>>(
                m, "HierarchicalPipeline")
                .def(py::init([](const colmap::HierarchicalPipelineOptions&
                                         options,
                                 const std::string& database_path,
                                 std::shared_ptr<colmap::ReconstructionManager>
                                         reconstruction_manager) {
                         auto* controller = new colmap::HierarchicalPipeline(
                                 options, database_path,
                                 reconstruction_manager.get());
                         return std::shared_ptr<colmap::HierarchicalPipeline>(
                                 controller,
                                 [reconstruction_manager](
                                         colmap::HierarchicalPipeline* p) {
                                     delete p;
                                 });
                     }),
                     "options"_a, "database_path"_a,
                     "reconstruction_manager"_a);

        // Upstream constructs from a Database object; the fork controller
        // opens the database from paths (documented divergence).
        py::class_<colmap::IncrementalPipeline,
                   std::shared_ptr<colmap::IncrementalPipeline>>(
                m, "IncrementalPipeline")
                .def(py::init([](colmap::IncrementalPipelineOptions options,
                                 const std::string& image_path,
                                 const std::string& database_path,
                                 std::shared_ptr<colmap::ReconstructionManager>
                                         reconstruction_manager) {
                         auto* options_ptr =
                                 new colmap::IncrementalPipelineOptions(
                                         std::move(options));
                         auto* controller = new colmap::IncrementalPipeline(
                                 options_ptr, image_path, database_path,
                                 reconstruction_manager.get());
                         return std::shared_ptr<colmap::IncrementalPipeline>(
                                 controller,
                                 [options_ptr, reconstruction_manager](
                                         colmap::IncrementalPipeline* p) {
                                     delete options_ptr;
                                     delete p;
                                 });
                     }),
                     "options"_a, "image_path"_a, "database_path"_a,
                     "reconstruction_manager"_a);
    }
    // Upstream top-level name (pycolmap image/undistortion.cc): the nested
    // WarpImageOptions::Interpolation sampling enum.
    py::enum_<colmap::WarpImageOptions::Interpolation>(m_image, "Interpolation")
            .value("kNearest",
                   colmap::WarpImageOptions::Interpolation::kNearest)
            .value("kBilinear",
                   colmap::WarpImageOptions::Interpolation::kBilinear)
            .export_values();

    // Upstream pycolmap parity (src/pycolmap/pipeline/match_features.cc):
    // the pairing-driven feature matching surface (PairingOptions,
    // PairGenerator family, Device, match_*/verify entry points).
    pybind_match_features(m, m_root);
}

}  // namespace pipeline
}  // namespace reconstruction
}  // namespace cloudViewer
