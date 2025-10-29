// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "base/reconstruction_manager.h"
#include "sfm/incremental_mapper.h"
#include "util/threading.h"

namespace colmap {

struct IncrementalMapperOptions {
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

    // The minimum number of residuals per bundle adjustment problem to
    // enable multi-threading solving of the problems.
    int ba_min_num_residuals_for_cpu_multi_threading = 50000;

    // The number of images to optimize in local bundle adjustment.
    int ba_local_num_images = 6;

    // Ceres solver function tolerance for local bundle adjustment
    double ba_local_function_tolerance = 0.0;

    // The maximum number of local bundle adjustment iterations.
    int ba_local_max_num_iterations = 25;

#ifdef PBA_ENABLED
    // Whether to use PBA in global bundle adjustment.
    bool ba_global_use_pba = false;

    // The GPU index for PBA bundle adjustment.
    int ba_global_pba_gpu_index = -1;
#endif
    // The growth rates after which to perform global bundle adjustment.
    double ba_global_images_ratio = 1.1;
    double ba_global_points_ratio = 1.1;
    int ba_global_images_freq = 500;
    int ba_global_points_freq = 250000;

    // Ceres solver function tolerance for global bundle adjustment
    double ba_global_function_tolerance = 0.0;

    // The maximum number of global bundle adjustment iterations.
    int ba_global_max_num_iterations = 50;

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
    std::string snapshot_path = "";
    int snapshot_images_freq = 0;

    // Which images to reconstruct. If no images are specified, all images will
    // be reconstructed by default.
    std::unordered_set<std::string> image_names;

    // If reconstruction is provided as input, fix the existing image poses.
    bool fix_existing_images = false;

    IncrementalMapper::Options mapper;
    IncrementalTriangulator::Options triangulation;

    IncrementalMapper::Options Mapper() const;
    IncrementalTriangulator::Options Triangulation() const;
    BundleAdjustmentOptions LocalBundleAdjustment() const;
    BundleAdjustmentOptions GlobalBundleAdjustment() const;
#ifdef PBA_ENABLED
    ParallelBundleAdjuster::Options ParallelGlobalBundleAdjustment() const;
#endif

    bool Check() const;
};

// Class that controls the incremental mapping procedure by iteratively
// initializing reconstructions from the same scene graph.
class IncrementalMapperController : public Thread {
public:
    enum {
        INITIAL_IMAGE_PAIR_REG_CALLBACK,
        NEXT_IMAGE_REG_CALLBACK,
        LAST_IMAGE_REG_CALLBACK,
    };

    IncrementalMapperController(const IncrementalMapperOptions* options,
                                const std::string& image_path,
                                const std::string& database_path,
                                ReconstructionManager* reconstruction_manager);

private:
    void Run();
    bool LoadDatabase();
    void Reconstruct(const IncrementalMapper::Options& init_mapper_options);

    const IncrementalMapperOptions* options_;
    const std::string image_path_;
    const std::string database_path_;
    ReconstructionManager* reconstruction_manager_;
    DatabaseCache database_cache_;
};

// Globally filter points and images in mapper.
size_t FilterPoints(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper);
size_t FilterImages(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper);

// Globally complete and merge tracks in mapper.
size_t CompleteAndMergeTracks(const IncrementalMapperOptions& options,
                              IncrementalMapper* mapper);

}  // namespace colmap
