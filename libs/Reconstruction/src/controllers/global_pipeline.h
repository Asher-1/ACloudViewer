// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Upstream COLMAP dbb41680 (controllers/global_pipeline.h) port. Fork
// adaptations: include paths, base/reconstruction_manager.h location.

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "base/reconstruction_manager.h"
#include "sfm/global_mapper.h"
#include "util/base_controller.h"

namespace colmap {

struct GlobalPipelineOptions {
    // The minimum number of matches for inlier matches to be considered.
    int min_num_matches = 15;

    // Whether to ignore the inlier matches of watermark image pairs.
    bool ignore_watermarks = false;

    // Names of images to reconstruct. If empty, all images are used.
    std::vector<std::string> image_names;

    // The image path at which to find the images to extract point colors.
    std::filesystem::path image_path;

    // Number of threads for parallel processing.
    int num_threads = -1;

    // Random seed for reproducibility.
    int random_seed = -1;

    // Whether to decompose relative poses from two-view geometries.
    bool decompose_relative_pose = true;

    // If true (default), reconstruct every connected component of the view
    // graph (one model per component). If false, reconstruct only the largest
    // connected component.
    bool multiple_models = true;

    // Minimum number of registered frames for a reconstruction to be kept.
    // Reconstructions with fewer registered frames are discarded.
    int min_model_size = 3;

    // Options for the global mapper.
    GlobalMapperOptions mapper;
};

class GlobalPipeline : public BaseController {
public:
    enum CallbackType {
        // Triggered after global positioning, after each global refinement
        // iteration, and after retriangulation, so the in-progress
        // reconstruction
        // can be rendered.
        MODEL_UPDATE_CALLBACK,
    };

    GlobalPipeline(
            GlobalPipelineOptions options,
            std::shared_ptr<Database> database,
            std::shared_ptr<ReconstructionManager> reconstruction_manager);

    void Run() override;

private:
    struct ReconstructionStats {
        // Number of components that failed during rotation averaging or
        // mapping.
        size_t num_failed = 0;

        // Number of components discarded for having too few registered frames.
        size_t num_too_small = 0;
    };

    // Run the full global SfM pipeline on the given database cache and return
    // the resulting reconstruction, or nullopt if mapping fails. The
    // in-progress reconstruction lives in the manager so callbacks can render
    // it; on success it is moved out of the manager slot and the caller decides
    // whether to re-insert it (the fork's manager stores Reconstruction by
    // value, so the reconstruction is returned by value instead of a shared_ptr
    // alias).
    std::optional<Reconstruction> ReconstructSingleComponent(
            const std::shared_ptr<const DatabaseCache>& database_cache,
            const GlobalMapperOptions& mapper_options);

    // Partition the input view graph once using rotation averaging and
    // reconstruct each resulting component at most once.
    ReconstructionStats ReconstructMultiComponents(
            const GlobalMapperOptions& mapper_options);

    const GlobalPipelineOptions options_;
    std::shared_ptr<DatabaseCache> database_cache_;
    std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
