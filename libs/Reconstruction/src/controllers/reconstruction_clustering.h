// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "scene/reconstruction.h"
#include "scene/reconstruction_clustering.h"
#include "scene/reconstruction_manager.h"
#include "util/base_controller.h"

namespace colmap {

// Controller that clusters frames from a reconstruction
// and splits it into multiple reconstructions based on clustering.
// Note: this module is experimental and should be verified carefully
// before use in production pipelines.
class ReconstructionClustererController : public BaseController {
public:
    ReconstructionClustererController(
            const ReconstructionClusteringOptions& options,
            std::shared_ptr<Reconstruction> reconstruction,
            std::shared_ptr<ReconstructionManager> reconstruction_manager);

    // Runs the pruning and clustering algorithm.
    // Results are stored in the reconstruction manager passed to the
    // constructor.
    void Run() override;

private:
    const ReconstructionClusteringOptions options_;
    std::shared_ptr<Reconstruction> reconstruction_;
    std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
