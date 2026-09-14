// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>

#include "controllers/incremental_pipeline.h"
#include "scene/reconstruction_manager.h"
#include "util/threading.h"

namespace colmap {

// Reconstruct disconnected verified-match components independently. Component
// discovery includes images with no verified edge; min_component_size controls
// which components are large enough to attempt reconstruction.
class GlobalMapperController : public Thread {
public:
    struct Options {
        std::filesystem::path image_path;
        std::filesystem::path database_path;
        int min_component_size = 2;
        int num_workers = -1;
        int init_num_trials = 10;

        bool Check() const;
    };

    GlobalMapperController(const Options& options,
                           const IncrementalMapperOptions& mapper_options,
                           ReconstructionManager* reconstruction_manager);

private:
    void Run() override;

    const Options options_;
    const IncrementalMapperOptions mapper_options_;
    ReconstructionManager* reconstruction_manager_;
};

}  // namespace colmap
