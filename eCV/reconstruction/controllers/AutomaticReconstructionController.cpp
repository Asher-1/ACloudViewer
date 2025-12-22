// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AutomaticReconstructionController.h"

#include "base/reconstruction_manager.h"

namespace cloudViewer {

using namespace colmap;

AutomaticReconstructionController::AutomaticReconstructionController(
        const Options& options,
        colmap::ReconstructionManager* reconstruction_manager)
    : colmap::AutomaticReconstructionController(
              static_cast<const colmap::AutomaticReconstructionController::
                                  Options&>(options),
              reconstruction_manager),
      ecv_options_(options) {
    // eCV-specific initialization if needed
}

void AutomaticReconstructionController::RunDenseMapper() {
    // Clear eCV-specific data containers before reconstruction
    fused_points_.clear();
    meshing_paths_.clear();
    textured_paths_.clear();
    texturing_success_ = false;

    // Call base class implementation
    // It will call our hook methods to collect data
    colmap::AutomaticReconstructionController::RunDenseMapper();
}

void AutomaticReconstructionController::OnFusedPointsGenerated(
        size_t reconstruction_idx,
        const std::vector<colmap::PlyPoint>& points) {
    // eCV-specific: Collect fused points for visualization
    fused_points_.push_back(points);
}

void AutomaticReconstructionController::OnMeshGenerated(
        size_t reconstruction_idx, const std::string& mesh_path) {
    // eCV-specific: Collect mesh path for visualization
    meshing_paths_.push_back(mesh_path);
}

void AutomaticReconstructionController::OnTexturedMeshGenerated(
        size_t reconstruction_idx, const std::string& textured_path) {
    // eCV-specific: Collect textured mesh path and mark success
    textured_paths_.push_back(textured_path);
    texturing_success_ = true;
}

}  // namespace cloudViewer
