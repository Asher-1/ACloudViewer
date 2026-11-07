// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "estimators/bundle_adjustment.h"

namespace colmap {

// Upstream parity (estimators/bundle_adjustment_caspar.h): solver parameters
// mirroring caspar::SolverParams, stored as double to round-trip through
// OptionManager regardless of the float/double build. Bound as
// BundleAdjustmentOptions::caspar and consumed by the Caspar backend.
struct CasparBundleAdjustmentOptions {
    int solver_iter_max = 200;
    int pcg_iter_max = 20;
    double diag_init = 1.0;
    double diag_min = 1e-12;
    double diag_scaling_up = 2.0;
    double diag_scaling_down = 0.333333;
    double diag_exit_value = 1e3;
    double score_exit_value = 0.0;
    double pcg_rel_error_exit = 1e-4;
    // Negative value disables the corresponding early-exit criterion.
    double pcg_rel_score_exit = -1.0;
    double pcg_rel_decrease_min = -1.0;
    double solver_rel_decrease_min = 1.0;
    std::string gpu_index = "-1";
    bool collect_iteration_data = false;

    bool Check() const { return solver_iter_max > 0 && pcg_iter_max > 0; }
};

// Upstream parity: Ceres-independent summary surface for the Caspar solver
// (the per-iteration caspar::IterationData vector stays engine-internal).
struct CasparBundleAdjustmentSummary : public BundleAdjustmentSummary {
    int iteration_count = 0;
    double initial_score = 0.0;
};

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
