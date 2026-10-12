// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ceres/ceres.h>

#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <unordered_set>

#include "estimators/bundle_adjustment.h"
#include "geometry/pose_prior.h"
#include "geometry/sim3.h"
#include "math/math.h"
#include "optim/manifold.h"
#include "util/enum_utils.h"

namespace colmap {

// Loss function types: Trivial (non-robust) and Soft-L1/Cauchy/Huber (robust)
// losses. HUBER is the upstream default for the global mapper.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(
        CeresLossFunctionType, 0, TRIVIAL, SOFT_L1, CAUCHY, HUBER);

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
struct CeresPosePriorBundleAdjustmentOptions {
    // Loss function for the prior position residuals.
    CeresLossFunctionType prior_position_loss_function_type =
            CeresLossFunctionType::TRIVIAL;

    // Threshold on the residual for the robust loss.
    double prior_position_loss_scale = std::sqrt(kChiSquare95ThreeDof);

    bool Check() const {
        CHECK_OPTION_GE(prior_position_loss_scale, 0.0);
        return true;
    }
};

// Upstream parity (bundle_adjustment_ceres.h): the Ceres-specific option
// group, bound as BundleAdjustmentOptions::ceres.
struct CeresBundleAdjustmentOptions {
    CeresLossFunctionType loss_function_type = CeresLossFunctionType::TRIVIAL;
    double loss_function_scale = 1.0;
    double loss_function_weight = 1.0;
    bool use_gpu = false;
    std::string gpu_index = "-1";
    ceres::Solver::Options solver_options;
    int min_num_images_gpu_solver = 50;
    int min_num_residuals_for_cpu_multi_threading = 50000;
    int max_num_images_direct_dense_cpu_solver = 50;
    int max_num_images_direct_sparse_cpu_solver = 1000;
    int max_num_images_direct_dense_gpu_solver = 200;
    int max_num_images_direct_sparse_gpu_solver = 4000;
    bool auto_select_solver_type = true;

    CeresBundleAdjustmentOptions();

    // Create a new loss function based on the specified options. The caller
    // takes ownership of the loss function.
    ceres::LossFunction* CreateLossFunction() const;

    // Create options tailored for given bundle adjustment config and problem.
    ceres::Solver::Options CreateSolverOptions(
            const BundleAdjustmentConfig& config,
            const ceres::Problem& problem) const;

    bool Check() const;
};

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

    // Upstream parity (d3ccaf35): the factory binds the reconstruction and
    // Solve() takes no arguments, returning the shared summary. The fork's
    // Solve(Reconstruction*) entry point is unchanged for existing callers.
    void BindReconstruction(Reconstruction* reconstruction);
    std::shared_ptr<BundleAdjustmentSummary> Solve();

    struct FramePoseBlock {
        frame_t frame_id;
        Rigid3d rig_from_world;
    };

    // Pose shadow blocks registered in the problem, keyed by frame. The
    // covariance estimator resolves pose parameter blocks through this map:
    // the fork's frames store split qvec/tvec (database-compatible), so the
    // rig-from-world block the BA registers is an internal 7-dim shadow and
    // cannot be resolved statically from the reconstruction alone (upstream
    // reads frame.RigFromWorld().params.data() directly).
    const std::map<frame_t, FramePoseBlock>& frame_blocks() const;

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

    // Upstream parity (a2bd15c2 PosePriorBundleAdjuster): filters the pose
    // priors to the configured images, robustly aligns the reconstruction to
    // them and normalizes it (fixed scale) for numerical stability. Falls
    // back to two-camera gauge fixing when fewer than three priors remain or
    // the alignment fails. Returns whether the priors constrain the problem.
    bool PreparePosePriors(Reconstruction* reconstruction);
    // Adds the covariance-weighted position prior of every parameterized
    // image, after the reprojection residuals are parameterized.
    void AddPosePriorsToProblem(Reconstruction* reconstruction);

    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
    std::set<image_t> parameterized_image_ids_;
    std::unordered_set<camera_t> camera_ids_;
    std::optional<PosePriorBundleAdjustmentOptions> pose_prior_options_;
    std::vector<PosePrior> pose_priors_;
    bool use_prior_position_ = false;
    Sim3d normalized_from_metric_;
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
    std::map<std::pair<rig_t, sensor_t>, SensorPoseBlock> sensor_blocks_;
    std::map<frame_t, FramePoseBlock> frame_blocks_;
    // Upstream-parity factory binding for the argument-less Solve().
    Reconstruction* bound_reconstruction_ = nullptr;
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

// Upstream parity (d3ccaf35): the Ceres factory binds the reconstruction.
std::unique_ptr<CeresBundleAdjuster> CreateDefaultCeresBundleAdjuster(
        const BundleAdjustmentOptions& options,
        const BundleAdjustmentConfig& config,
        Reconstruction& reconstruction);

// Upstream parity overload with a report header.
void PrintSolverSummary(const ceres::Solver::Summary& summary,
                        const std::string& header);

}  // namespace colmap
