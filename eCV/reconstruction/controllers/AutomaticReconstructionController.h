// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Thin wrapper around colmap::AutomaticReconstructionController
// This allows eCV to reuse the implementation from libs/Reconstruction
// while adding visualization-specific data collection

#include "controllers/automatic_reconstruction.h"
#include "util/ply.h"

namespace colmap {
class Reconstruction;
}

namespace colmap {
class ReconstructionManager;
}

namespace cloudViewer {

// eCV-specific wrapper that extends colmap's AutomaticReconstructionController
// Main additions:
// 1. Data collection for UI visualization via hook methods
// 2. autoVisualization option for automatic scene integration
class AutomaticReconstructionController
    : public colmap::AutomaticReconstructionController {
public:
    // Extend Options with eCV-specific settings
    struct Options : public colmap::AutomaticReconstructionController::Options {
        // Whether to add the reconstruction results to DBRoot automatically
        bool autoVisualization = true;
    };

    AutomaticReconstructionController(
            const Options& options,
            colmap::ReconstructionManager* reconstruction_manager);

    // eCV-specific: Public data for UI visualization
    // These collect results during reconstruction for display
    std::vector<std::vector<colmap::PlyPoint>> fused_points_;
    std::vector<std::string> meshing_paths_;
    std::vector<std::string> textured_paths_;
    bool texturing_success_ = false;

protected:
    // Override hook methods to collect data for visualization
    void OnFusedPointsGenerated(
            size_t reconstruction_idx,
            const std::vector<colmap::PlyPoint>& points) override;
    void OnMeshGenerated(size_t reconstruction_idx,
                         const std::string& mesh_path) override;
    void OnTexturedMeshGenerated(size_t reconstruction_idx,
                                 const std::string& textured_path) override;

    // Override to initialize data containers
    void RunDenseMapper() override;

private:
    const Options ecv_options_;
};

}  // namespace cloudViewer
