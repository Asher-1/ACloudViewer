// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>
#include <functional>
#include <memory>

#include "scene/reconstruction_manager.h"
#include "sfm/incremental_mapper.h"
#include "util/hash_containers.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {

struct IncrementalPipelineOptions {
public:
    // The minimum number of matches for inlier matches to be considered.
    int min_num_matches = 15;

    // Whether to ignore the inlier matches of watermark image pairs.
    bool ignore_watermarks = false;

    // Whether to reconstruct multiple sub-models.
    bool multiple_models = true;

    // The number of sub-models to reconstruct.
    int max_num_models = 50;

    // The maximum number of overlapping images between sub-models. If the
    // current sub-models shares more than this number of images with another
    // model, then the reconstruction is stopped.
    int max_model_overlap = 20;

    // The minimum number of registered images of a sub-model, otherwise the
    // sub-model is discarded.
    int min_model_size = 10;

    // The image identifiers used to initialize the reconstruction. Note that
    // only one or both image identifiers can be specified. In the former case,
    // the second image is automatically determined.
    int init_image_id1 = -1;
    int init_image_id2 = -1;

    // The number of trials to initialize the reconstruction.
    int init_num_trials = 200;

    // Whether to extract colors for reconstructed points.
    bool extract_colors = true;

    // The number of threads to use during reconstruction.
    int num_threads = -1;

    // Thresholds for filtering images with degenerate intrinsics.
    double min_focal_length_ratio = 0.1;
    double max_focal_length_ratio = 10.0;
    double max_extra_param = 1.0;

    // Which intrinsic parameters to optimize during the reconstruction.
    bool ba_refine_focal_length = true;
    bool ba_refine_principal_point = false;
    bool ba_refine_extra_params = true;

    // Whether to refine the sensor-from-rig poses in bundle adjustment
    // (upstream d3ccaf35 incremental_pipeline.h).
    bool ba_refine_sensor_from_rig = true;

    // The minimum number of residuals per bundle adjustment problem to
    // enable multi-threading solving of the problems.
    int ba_min_num_residuals_for_cpu_multi_threading = 50000;

    // Ceres solver function tolerance for local bundle adjustment
    double ba_local_function_tolerance = 0.0;

    // The maximum number of local bundle adjustment iterations. -1 means
    // "use the configured backend's default" (upstream d3ccaf35 sentinel).
    int ba_local_max_num_iterations = -1;

    // The bundle adjustment backend for local/global BA. CASPAR is only
    // available in builds with RECONSTRUCTION_CASPAR_ENABLED; unsupported
    // requests fall back to Ceres with a warning.
    BundleAdjustmentBackend ba_local_backend = BundleAdjustmentBackend::CERES;
    BundleAdjustmentBackend ba_global_backend = BundleAdjustmentBackend::CERES;

    // The growth rates after which to perform global bundle adjustment.
    double ba_global_frames_ratio = 1.1;
    double ba_global_points_ratio = 1.1;
    int ba_global_frames_freq = 500;
    int ba_global_points_freq = 250000;

    // Ceres solver function tolerance for global bundle adjustment
    double ba_global_function_tolerance = 0.0;

    // The maximum number of global bundle adjustment iterations.
    int ba_global_max_num_iterations = -1;

    // The thresholds for iterative bundle adjustment refinements.
    int ba_local_max_refinements = 2;
    double ba_local_max_refinement_change = 0.001;
    int ba_global_max_refinements = 5;
    double ba_global_max_refinement_change = 0.0005;

    // Whether to use Ceres' CUDA sparse linear algebra library, if available.
    bool ba_use_gpu = false;
    std::string ba_gpu_index = "-1";

    // Path to a folder with reconstruction snapshots during incremental
    // reconstruction. Snapshots will be saved according to the specified
    // frequency of registered images.
    std::filesystem::path snapshot_path;
    int snapshot_frames_freq = 0;

    // Optional list of image names to reconstruct. If no images are
    // specified, all images will be reconstructed by default.
    std::vector<std::string> image_names;

    // Whether to load all images from the database, including those without
    // correspondences. Only useful for triangulation where all images are
    // already registered and should retain their keypoints. Should not be
    // enabled for incremental SfM.
    bool load_all_images = false;

    // If reconstruction is provided as input, fix the existing image poses.
    bool fix_existing_frames = false;

    // Upstream parity (a2bd15c2 IncrementalPipelineOptions): enable fallback
    // to structure-less image registration using 2D-2D correspondences, if
    // structure-based registration fails using 2D-3D correspondences.
    bool structure_less_registration_fallback = true;

    // Only use structure-less and skip structure-based image registration.
    bool structure_less_registration_only = false;

    // Maximum runtime in seconds for the reconstruction process.
    // If set to a non-positive value, the process will run until completion.
    int max_runtime_seconds = -1;

    // List of rigs for which to fix the sensor_from_rig transformation,
    // independent of ba_refine_sensor_from_rig (upstream parity;
    // fork consumes the same FlatHashSet type as the mapper options).
    FlatHashSet<rig_t> constant_rigs;

    // List of cameras for which to fix the camera parameters independent
    // of refine_focal_length, refine_principal_point, and refine_extra_params.
    FlatHashSet<camera_t> constant_cameras;

    // Random seed for randomized sub-steps (RANSAC in absolute pose
    // estimation and prior alignment); -1 leaves the default non-determinism.
    int random_seed = -1;

    // Whether to use position priors for BA and to convert them to a local
    // ENU frame when loading the database cache (upstream parity,
    // d3ccaf35 incremental_pipeline.h).
    bool use_prior_position = false;
    bool use_robust_loss_on_prior_position = false;
    double prior_position_loss_scale = 7.815;

    IncrementalMapper::Options mapper;
    IncrementalTriangulator::Options triangulation;

    IncrementalMapper::Options Mapper() const;
    IncrementalTriangulator::Options Triangulation() const;
    BundleAdjustmentOptions LocalBundleAdjustment() const;
    BundleAdjustmentOptions GlobalBundleAdjustment() const;

    // Upstream parity (d3ccaf35): resolve the -1 iteration sentinel to the
    // configured backend's default (Ceres 25/50, Caspar solver_iter_max).
    int EffBaLocalMaxNumIterations() const;
    int EffBaGlobalMaxNumIterations() const;

    bool Check() const;
};

// Class that controls the incremental mapping procedure by iteratively
// initializing reconstructions from the same scene graph.
class IncrementalPipeline : public Thread {
public:
    // Upstream parity (d3ccaf35 IncrementalPipeline::CallbackType): named so
    // the binding layer can export it under the upstream top-level name
    // IncrementalPipelineCallback. Unscoped, so the bare enumerator names
    // used by the fork code paths keep resolving.
    enum CallbackType {
        INITIAL_IMAGE_PAIR_REG_CALLBACK,
        NEXT_IMAGE_REG_CALLBACK,
        LAST_IMAGE_REG_CALLBACK,
    };

    // Upstream parity (a2bd15c2 IncrementalPipeline::Status): outcome of one
    // reconstruction attempt, driving the discard/keep/retry state machine.
    enum class Status {
        SUCCESS,
        INTERRUPTED,
        CONTINUE,
        STOP,
        NO_INITIAL_PAIR,
        BAD_INITIAL_PAIR,
        UNKNOWN_SENSOR_FROM_RIG,
    };

    IncrementalPipeline(const IncrementalPipelineOptions* options,
                        const std::filesystem::path& image_path,
                        const std::filesystem::path& database_path,
                        ReconstructionManager* reconstruction_manager);

    // Upstream parity (1882ef4b #3926): reconstruct from an already loaded
    // database cache. The pipeline works on a filtered copy created with
    // DatabaseCache::CreateFromCache, so callers such as the hierarchical
    // mapper can share one cache across workers without re-reading the
    // database.
    IncrementalPipeline(const IncrementalPipelineOptions* options,
                        const std::filesystem::path& image_path,
                        std::shared_ptr<const DatabaseCache> database_cache,
                        ReconstructionManager* reconstruction_manager);

    // Upstream parity (d3ccaf35 IncrementalPipeline::SetCheckIfStoppedFunc):
    // an external cancellation probe polled alongside the thread stop flag;
    // when it returns true the controller stops at the next checkpoint.
    void SetCheckIfStoppedFunc(std::function<bool()> func);

    Status Reconstruct(IncrementalMapper& mapper,
                       const IncrementalMapper::Options& mapper_options,
                       bool continue_reconstruction);

    Status ReconstructSubModel(IncrementalMapper& mapper,
                               const IncrementalMapper::Options& mapper_options,
                               Reconstruction& reconstruction);

    Status InitializeReconstruction(
            IncrementalMapper& mapper,
            const IncrementalMapper::Options& mapper_options,
            Reconstruction& reconstruction);

    bool CheckRunGlobalRefinement(const Reconstruction& reconstruction,
                                  size_t ba_prev_num_reg_frames,
                                  size_t ba_prev_num_points) const;

    // Upstream parity (a2bd15c2 IncrementalPipeline::CheckReachedMaxRuntime):
    // polled alongside ShouldStop() at every checkpoint.
    bool CheckReachedMaxRuntime() const;

private:
    void Run();
    bool LoadDatabase();

    bool ShouldStop();

    const IncrementalPipelineOptions* options_;
    const std::string image_path_;
    const std::string database_path_;
    ReconstructionManager* reconstruction_manager_;
    // Optional source cache; when set, LoadDatabase filters it instead of
    // reading database_path_.
    std::shared_ptr<const DatabaseCache> source_database_cache_;
    std::shared_ptr<DatabaseCache> database_cache_;
    std::function<bool()> check_if_stopped_func_;
    Timer total_run_timer_;
    // Upstream parity (a2bd15c2 CustomizeIncrementalPipelineOptions): the
    // minimum model size capped at half of the loaded images. Upstream
    // rewrites options->min_model_size in place; the fork's options are
    // borrowed const, so the effective value lives here.
    int min_model_size_ = 0;
};

// Globally filter points and images in mapper.
size_t FilterPoints(const IncrementalPipelineOptions& options,
                    IncrementalMapper* mapper);
size_t FilterFrames(const IncrementalPipelineOptions& options,
                    IncrementalMapper* mapper);

// Globally complete and merge tracks in mapper.
size_t CompleteAndMergeTracks(const IncrementalPipelineOptions& options,
                              IncrementalMapper* mapper);

}  // namespace colmap
