// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT

#pragma once

#include "optim/bundle_adjustment.h"

namespace colmap {

// The generated Caspar graph supports the upstream Pinhole/SimpleRadial
// factors. The legacy reconstruction adapter maps an optional camera-only
// Frame/Rig to Caspar's fixed sensor_from_rig input and synchronizes image
// poses from a solved Frame. Other camera models remain on Ceres, matching the
// upstream Caspar adapter boundary.
bool SolveCasparBundleAdjustment(const BundleAdjustmentOptions& options,
                                 const BundleAdjustmentConfig& config,
                                 Reconstruction* reconstruction,
                                 ceres::Solver::Summary* ceres_summary);

}  // namespace colmap
