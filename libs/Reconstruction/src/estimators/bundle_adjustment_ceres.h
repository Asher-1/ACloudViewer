// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ceres/ceres.h>

#include <map>
#include <memory>
#include <optional>
#include <unordered_set>

#include "estimators/bundle_adjustment.h"
#include "geometry/pose_prior.h"
#include "optim/manifold.h"

namespace colmap {

// Ceres-specific bundle adjustment summary with access to full solver
// details (upstream parity, d3ccaf35 bundle_adjustment_ceres.h). The fork
// keeps its bool Solve + ceres Summary() interface; the summary is built
// lazily from the last solve via CeresBundleAdjuster::summary().
struct CeresBundleAdjustmentSummary : public BundleAdjustmentSummary {
    ceres::Solver::Summary ceres_summary;

    std::string BriefReport() const override;

    static std::shared_ptr<CeresBundleAdjustmentSummary> Create(
            ceres::Solver::Summary ceres_summary);
};

// Bundle adjustment based on Ceres-Solver (upstream
// bundle_adjustment_ceres.{h,cc} parity). Enables most flexible
// configurations and provides best solution quality. The fork's CASPAR GPU
// backend is dispatched first inside Solve() when requested, falling back to
// Ceres for problems Caspar cannot represent.
class CeresBundleAdjuster : public BundleAdjuster {
public:
    CeresBundleAdjuster(const BundleAdjustmentOptions& options,
                        const BundleAdjustmentConfig& config);

    bool Solve(Reconstruction* reconstruction) override;

    void SetPosePriors(PosePriorBundleAdjustmentOptions options,
                       std::vector<PosePrior> pose_priors) override;

    const ceres::Solver::Summary& Summary() const override;

    // Upstream parity (d3ccaf35): the backend-typed summary of the last
    // Solve, with the mapped termination type and residual count. Built
    // from the ceres solver summary; nullptr before the first solve.
    std::shared_ptr<CeresBundleAdjustmentSummary> summary() const;

    // Upstream parity (d3ccaf35): direct access to the underlying problem,
    // e.g. for covariance estimation over the assembled problem.
    ceres::Problem& Problem();
    const ceres::Problem& Problem() const;

private:
    void SetUp(Reconstruction* reconstruction,
               ceres::LossFunction* loss_function);
    void TearDown(Reconstruction* reconstruction);

    void AddImageToProblem(const image_t image_id,
                           Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function);

    void AddPointToProblem(const point3D_t point3D_id,
                           Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function);

    void ParameterizeCameras(Reconstruction* reconstruction);
    void ParameterizeRigsAndFrames(Reconstruction* reconstruction);
    void ParameterizePoints(Reconstruction* reconstruction);

    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
    std::set<image_t> parameterized_image_ids_;
    std::unordered_set<camera_t> camera_ids_;
    std::optional<PosePriorBundleAdjustmentOptions> pose_prior_options_;
    std::vector<PosePrior> pose_priors_;
    std::unordered_map<point3D_t, size_t> point3D_num_observations_;

    // W3-2b step 5 (upstream parity, d3ccaf35): the pose parameter blocks
    // are single Rigid3d blocks (Eigen::Vector7d params, [qx, qy, qz, qw,
    // tx, ty, tz]) held as node-based-map shadows; the refined poses are
    // written back to the frames/rigs in TearDown(). Shadow blocks keep the
    // parameter addresses stable while SetUp adds images and leave the
    // fork's qvec/tvec scene storage untouched for every other consumer.
    struct SensorPoseBlock {
        rig_t rig_id;
        sensor_t sensor_id;
        Rigid3d sensor_from_rig;
    };
    struct FramePoseBlock {
        frame_t frame_id;
        Rigid3d rig_from_world;
    };
    std::map<std::pair<rig_t, sensor_t>, SensorPoseBlock> sensor_blocks_;
    std::map<frame_t, FramePoseBlock> frame_blocks_;
    // Frame blocks are shared by every image of a frame; guard the one-time
    // manifold installation (repeated installation aborts).
    std::unordered_set<const double*> manifold_marked_blocks_;

    SensorPoseBlock& GetOrCreateSensorBlock(rig_t rig_id,
                                            sensor_t sensor_id,
                                            const Rigid3d& sensor_from_rig);
    FramePoseBlock& GetOrCreateFrameBlock(frame_t frame_id,
                                          const Rigid3d& rig_from_world);

    // Upstream parity (d3ccaf35): global gauge fixing after parameterization.
    // TWO_CAMS_FROM_WORLD fixes one full frame pose plus one translation
    // dimension of a second frame; THREE_POINTS is the degenerate-case
    // fallback that fixes three non-collinear 3D points. Both operate on
    // the single Rigid3d shadow blocks.
    void FixGaugeWithTwoCamsFromWorld(Reconstruction* reconstruction);
    void FixGaugeWithThreePoints(Reconstruction* reconstruction);
};

}  // namespace colmap
