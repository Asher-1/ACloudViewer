// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "util/hash_containers.h"
#include "estimators/bundle_adjustment.h"
#include "estimators/generalized_pose.h"
#include "geometry/pose.h"
#include "scene/database.h"
#include "scene/database_cache.h"
#include "scene/reconstruction.h"
#include "sfm/incremental_triangulator.h"
#include "util/alignment.h"

namespace colmap {

// Class that provides all functionality for the incremental reconstruction
// procedure. Example usage:
//
//  IncrementalMapper mapper(&database_cache);
//  mapper.BeginReconstruction(&reconstruction);
//  Rigid3d cam2_from_cam1;
//  CHECK(mapper.FindInitialImagePair(options, image_id1, image_id2,
//                                    cam2_from_cam1));
//  mapper.RegisterInitialImagePair(options, image_id1, image_id2,
//                                  cam2_from_cam1);
//  while (...) {
//    const auto next_image_ids = mapper.FindNextImages(options);
//    for (const auto image_id : next_image_ids) {
//      CHECK(mapper.RegisterNextImage(options, image_id));
//      if (...) {
//        mapper.AdjustLocalBundle(...);
//      } else {
//        mapper.AdjustGlobalBundle(...);
//      }
//    }
//  }
//  mapper.EndReconstruction(false);
//
class IncrementalMapper {
public:
    struct Options {
        // Minimum number of inliers for initial image pair.
        int init_min_num_inliers = 100;

        // Maximum error in pixels for two-view geometry estimation for initial
        // image pair.
        double init_max_error = 4.0;

        // Maximum forward motion for initial image pair.
        double init_max_forward_motion = 0.95;

        // Minimum triangulation angle for initial image pair.
        double init_min_tri_angle = 16.0;

        // Maximum number of trials to use an image for initialization.
        int init_max_reg_trials = 2;

        // Maximum reprojection error in absolute pose estimation.
        double abs_pose_max_error = 12.0;

        // Minimum number of inliers in absolute pose estimation.
        int abs_pose_min_num_inliers = 30;

        // Minimum inlier ratio in absolute pose estimation.
        double abs_pose_min_inlier_ratio = 0.25;

        // Whether to estimate the focal length in absolute pose estimation.
        bool abs_pose_refine_focal_length = true;

        // Whether to estimate the extra parameters in absolute pose estimation.
        bool abs_pose_refine_extra_params = true;

        // Number of images to optimize in local bundle adjustment.
        int local_ba_num_images = 6;

        // Minimum triangulation for images to be chosen in local bundle
        // adjustment.
        double local_ba_min_tri_angle = 6.0;

        // Thresholds for bogus camera parameters. Images with bogus camera
        // parameters are filtered and ignored in triangulation.
        double min_focal_length_ratio = 0.1;   // Opening angle of ~130deg
        double max_focal_length_ratio = 10.0;  // Opening angle of ~5deg
        double max_extra_param = 1.0;

        // Maximum reprojection error in pixels for observations.
        double filter_max_reproj_error = 4.0;

        // Minimum triangulation angle in degrees for stable 3D points.
        double filter_min_tri_angle = 1.5;

        // Maximum number of trials to register an image.
        int max_reg_trials = 3;

        // If reconstruction is provided as input, fix the existing image poses.
        bool fix_existing_images = false;

        // List of rigs for which to fix the sensor_from_rig transformation,
        // independent of ba_refine_sensor_from_rig (upstream d3ccaf35,
        // named fix_existing_frames there).
        NodeHashSet<rig_t> constant_rigs;

        // List of cameras for which to fix the camera parameters independent
        // of refine_focal_length, refine_principal_point, and
        // refine_extra_params (upstream d3ccaf35 parity).
        NodeHashSet<camera_t> constant_cameras;

        // If enabled, redundant 3D points are (temporally) excluded from the
        // global bundle adjustment and only used to compute their reprojection
        // error in the end. This is only done for reconstructions that have
        // reached sufficient size with at least 10 registered frames.
        bool ba_global_ignore_redundant_points3D = false;

        // PRNG seed for all stochastic methods during reconstruction
        // (upstream parity, d3ccaf35 incremental_mapper.h).
        int random_seed = -1;

        // The coverage gain threshold for a 3D point to be considered
        // redundant in the global bundle adjustment. A larger value means more
        // 3D points are pruned.
        double ba_global_ignore_redundant_points3D_min_coverage_gain = 0.05;

        // Whether to use the position prior to fix the gauge and constrain
        // the bundle adjustment (upstream parity). Requires at least 3
        // registered images with a position prior.
        bool use_prior_position = false;

        // Whether to use a robust loss on the position prior residuals.
        bool use_robust_loss_on_prior_position = false;

        // The scaling factor for the position prior loss function.
        double prior_position_loss_scale = 7.815;

        // Number of threads.
        int num_threads = -1;

        // Method to find and select next best image to register.
        enum class ImageSelectionMethod {
            MAX_VISIBLE_POINTS_NUM,
            MAX_VISIBLE_POINTS_RATIO,
            MIN_UNCERTAINTY,
        };
        ImageSelectionMethod image_selection_method =
                ImageSelectionMethod::MIN_UNCERTAINTY;

        bool Check() const;
    };

    struct LocalBundleAdjustmentReport {
        size_t num_merged_observations = 0;
        size_t num_completed_observations = 0;
        size_t num_filtered_observations = 0;
        size_t num_adjusted_observations = 0;
    };

    // Create incremental mapper. The database cache must live for the entire
    // life-time of the incremental mapper.
    explicit IncrementalMapper(const DatabaseCache* database_cache);

    // Prepare the mapper for a new reconstruction, which might have existing
    // registered images (in which case `RegisterNextImage` must be called) or
    // which is empty (in which case `RegisterInitialImagePair` must be called).
    void BeginReconstruction(Reconstruction* reconstruction);

    // Cleanup the mapper after the current reconstruction is done. If the
    // model is discarded, the number of total and shared registered images will
    // be updated accordingly.
    void EndReconstruction(const bool discard);

    // Find initial image pair to seed the incremental reconstruction. The
    // image pairs should be passed to `RegisterInitialImagePair`. This
    // function automatically ignores image pairs that failed to register
    // previously. On success, the intrinsics estimated for the chosen pair by
    // two-view solvers (e.g. shared focal) have already been seeded into the
    // reconstruction (upstream SeedEstimatedInitialCameras placement).
    bool FindInitialImagePair(const Options& options,
                              image_t& image_id1,
                              image_t& image_id2,
                              Rigid3d& cam2_from_cam1);

    // Find best next image to register in the incremental reconstruction. The
    // images should be passed to `RegisterNextImage`. This function
    // automatically ignores images that failed to registered for
    // `max_reg_trials`.
    std::vector<image_t> FindNextImages(const Options& options,
                                        bool structure_less = false);

    // Attempt to seed the reconstruction from an image pair. The relative
    // pose comes from `FindInitialImagePair`; this function no longer
    // re-estimates the two-view geometry (upstream d3ccaf35 form).
    bool RegisterInitialImagePair(const Options& options,
                                  const image_t image_id1,
                                  const image_t image_id2,
                                  const Rigid3d& cam2_from_cam1);

    // Attempt to register image to the existing model. This requires that
    // a previous call to `RegisterInitialImagePair` was successful.
    bool RegisterNextImage(const Options& options, const image_t image_id);

    // Upstream parity (d3ccaf35): register a full multi-sensor frame through
    // the generalized absolute pose estimator, using 2D-3D correspondences
    // from every image of the frame. Only valid for rigs with more than one
    // sensor.
    bool RegisterNextGeneralFrame(const Options& options, Frame& frame);

    // Upstream parity (d3ccaf35): structure-less resectioning of a single
    // image from 2D-2D correspondences to already registered images, used
    // as a fallback when structure-based registration fails.
    bool RegisterNextStructureLessImage(const Options& options,
                                        const image_t image_id);

    // Triangulate observations of image.
    size_t TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                            const image_t image_id);

    // Retriangulate image pairs that should have common observations according
    // to the scene graph but don't due to drift, etc. To handle drift, the
    // employed reprojection error thresholds should be relatively large. If the
    // thresholds are too large, non-robust bundle adjustment will break down;
    // if the thresholds are too small, we cannot fix drift effectively.
    size_t Retriangulate(const IncrementalTriangulator::Options& tri_options);

    // Complete tracks by transitively following the scene graph
    // correspondences. This is especially effective after bundle adjustment,
    // since many cameras and point locations might have improved. Completion of
    // tracks enables better subsequent registration of new images.
    size_t CompleteTracks(const IncrementalTriangulator::Options& tri_options);

    // Merge tracks by using scene graph correspondences. Similar to
    // `CompleteTracks`, this is effective after bundle adjustment and improves
    // the redundancy in subsequent bundle adjustments.
    size_t MergeTracks(const IncrementalTriangulator::Options& tri_options);

    // Adjust locally connected images and points of a reference image. In
    // addition, refine the provided 3D points. Only images connected to the
    // reference image are optimized. If the provided 3D points are not locally
    // connected to the reference image, their observing images are set as
    // constant in the adjustment.
    LocalBundleAdjustmentReport AdjustLocalBundle(
            const Options& options,
            const BundleAdjustmentOptions& ba_options,
            const IncrementalTriangulator::Options& tri_options,
            const image_t image_id,
            const NodeHashSet<point3D_t>& point3D_ids);

    // Global bundle adjustment with the configured backend.
    bool AdjustGlobalBundle(const Options& options,
                            const BundleAdjustmentOptions& ba_options);
    // Filter images and point observations.
    // Filter frames (and their images) whose pose or camera parameters are
    // unreliable. Filtering and bookkeeping happen at frame granularity
    // (upstream d3ccaf35 FilterFrames; the fork keeps the
    // fix_existing_images option name for CLI compatibility).
    size_t FilterFrames(const Options& options);
    size_t FilterPoints(const Options& options);

    // Complete and merge all tracks (upstream parity, d3ccaf35
    // incremental_mapper.h).
    size_t CompleteAndMergeTracks(
            const IncrementalTriangulator::Options& tri_options);

    // Iterative local refinement: multiple rounds of local bundle adjustment
    // until the changed-observation ratio converges (upstream d3ccaf35
    // IncrementalMapper::IterativeLocalRefinement).
    void IterativeLocalRefinement(
            int max_num_refinements,
            double max_refinement_change,
            const Options& options,
            const BundleAdjustmentOptions& ba_options,
            const IncrementalTriangulator::Options& tri_options,
            image_t image_id);

    // Iterative global refinement: complete/merge/retriangulate the tracks
    // and run global bundle adjustment until convergence (upstream parity,
    // d3ccaf35 IncrementalMapper::IterativeGlobalRefinement).
    void IterativeGlobalRefinement(
            int max_num_refinements,
            double max_refinement_change,
            const Options& options,
            const BundleAdjustmentOptions& ba_options,
            const IncrementalTriangulator::Options& tri_options,
            bool normalize_reconstruction = true);

    const Reconstruction& GetReconstruction() const;

    // Getter functions (upstream d3ccaf35 parity; the reconstruction getter
    // keeps the fork's GetReconstruction name for API compatibility).
    class ObservationManager& ObservationManager() const;
    IncrementalTriangulator& Triangulator() const;
    const NodeHashSet<frame_t>& FilteredFrames() const;
    const NodeHashSet<frame_t>& ExistingFrameIds() const;
    const NodeHashMap<camera_t, size_t>& NumRegImagesPerCamera() const;

    // Number of images that are registered in at least on reconstruction.
    size_t NumTotalRegImages() const;

    // Number of shared images between current reconstruction and all other
    // previous reconstructions.
    size_t NumSharedRegImages() const;

    // Number of registered frames per rig (upstream parity, d3ccaf35).
    const NodeHashMap<rig_t, size_t>& NumRegFramesPerRig() const;

    // Get changed 3D points, since the last call to `ClearModifiedPoints3D`.
    const NodeHashSet<point3D_t>& GetModifiedPoints3D();

    // Clear the collection of changed 3D points.
    void ClearModifiedPoints3D();

    // Estimate two view geometry and checks if it is suitable for
    // initialization (upstream d3ccaf35 form; used e.g. for a user-provided
    // initial pair). On success, estimated intrinsics are seeded into the
    // reconstruction cameras.
    bool EstimateInitialTwoViewGeometry(const Options& options,
                                        const image_t image_id1,
                                        const image_t image_id2,
                                        Rigid3d& cam2_from_cam1);

private:
    // Register / De-register frame in current reconstruction and update
    // the per-rig frame count plus the per-image counters of every image in
    // the frame (upstream d3ccaf35 form: no standalone image-level events).
    void RegisterFrameEvent(frame_t frame_id);
    void DeRegisterFrameEvent(frame_t frame_id);

    // Registration statistics shared across a reconstruction run (upstream
    // d3ccaf35 form: bundled in a single struct instead of loose members).
    struct RegistrationStatistics {
        // Number of images that are registered in at least one
        // reconstruction.
        size_t num_total_reg_images = 0;

        // Number of shared images between current reconstruction and all
        // other previous reconstructions.
        size_t num_shared_reg_images = 0;

        // Images and image pairs that have been used for initialization.
        // Each image and image pair is only tried once for initialization.
        NodeHashMap<image_t, size_t> init_num_reg_trials;
        NodeHashSet<image_pair_t> init_image_pairs;

        // The number of registered frames/images per rig/camera. Used to
        // avoid duplicate refinement of rig/camera parameters and degradation
        // of already refined parameters in local bundle adjustment when
        // multiple frames share rigs or images share intrinsics.
        NodeHashMap<rig_t, size_t> num_reg_frames_per_rig;
        NodeHashMap<camera_t, size_t> num_reg_images_per_camera;

        // The number of reconstructions in which images are registered.
        NodeHashMap<image_t, size_t> num_registrations;

        // Number of trials to register image in current reconstruction. Used
        // to set an upper bound to the number of trials to register an image.
        NodeHashMap<image_t, size_t> num_reg_trials;
        NodeHashMap<image_t, size_t> num_structure_less_reg_trials;
    };

    // Class that holds all necessary data from database in memory.
    const DatabaseCache* database_cache_;

    // Class that holds data of the reconstruction.
    Reconstruction* reconstruction_;

    // Class that is responsible for incremental triangulation.
    std::unique_ptr<IncrementalTriangulator> triangulator_;

    // Upstream parity (d3ccaf35): per-frame observation statistics and
    // bookkeeping shared with the triangulator.
    std::shared_ptr<class ObservationManager> obs_manager_;

    // Registration statistics (see struct above).
    RegistrationStatistics reg_stats_;

    // Frames that have been filtered in current reconstruction.
    NodeHashSet<frame_t> filtered_frames_;

    // Frames that were registered before beginning the reconstruction.
    // This frame list will be non-empty, if the reconstruction is continued
    // from an existing reconstruction.
    NodeHashSet<frame_t> existing_frame_ids_;
};

}  // namespace colmap
