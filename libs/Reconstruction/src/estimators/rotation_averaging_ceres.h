// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ceres/ceres.h>

#include <memory>

#include "estimators/bundle_adjustment_ceres.h"
#include "estimators/rotation_averaging.h"
#include "math/math.h"
#include "util/hash_containers.h"
#include "util/types.h"

namespace colmap {

class PoseGraph;
class Reconstruction;

struct CeresRotationAveragerOptions {
    CeresLossFunctionType loss_function_type = CeresLossFunctionType::HUBER;
    double loss_function_scale = DegToRad(5.0);
    RotationAveragingReweighting reweighting =
            RotationAveragingReweighting::UNIFORM;
    ceres::Solver::Options solver_options;
    bool skip_initialization = false;
    bool refine_sensor_from_rig = true;

    CeresRotationAveragerOptions();
};

// Optimizes frame and sensor rotations in place. This low-level solver does
// not apply pose priors, outlier filtering, or frame deregistration.
class CeresRotationAverager {
public:
    CeresRotationAverager(const CeresRotationAveragerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction);
    CeresRotationAverager(const CeresRotationAverager&) = delete;
    CeresRotationAverager& operator=(const CeresRotationAverager&) = delete;

    ceres::Solver::Options solver_options;

    ceres::Solver::Summary Solve();
    ceres::Problem& Problem();
    const ceres::Problem& Problem() const;
    void AddRelativeRotationResidual(
            image_t image_id1,
            image_t image_id2,
            const Eigen::Quaterniond& cam2_from_cam1,
            const std::shared_ptr<ceres::LossFunction>& loss_function);

private:
    void InitializeRotations(const CeresRotationAveragerOptions& options,
                             const PoseGraph& pose_graph,
                             const FlatHashSet<image_t>& image_ids);
    void SetupParameterBlocks(const CeresRotationAveragerOptions& options,
                              const FlatHashSet<image_t>& image_ids);

    Reconstruction& reconstruction_;
    FlatHashMap<ceres::LossFunction*, std::shared_ptr<ceres::LossFunction>>
            losses_;
    std::unique_ptr<ceres::Problem> problem_;
};

std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
        const CeresRotationAveragerOptions& options,
        const PoseGraph& pose_graph,
        Reconstruction& reconstruction);

}  // namespace colmap
