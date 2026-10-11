// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

#include "geometry/pose_prior.h"
#include "math/math.h"
#include "optim/manifold.h"
#include "optim/ransac.h"
#include "scene/reconstruction.h"
#include "util/alignment.h"
#include "util/enum_utils.h"
#include "util/types.h"

namespace colmap {

// Keep the solver selection independent of the caller. Incremental mapping,
// global mapping, the GUI controller, and the CLI all enter through
// BundleAdjuster, so a backend must be selected here rather than in one caller.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(BundleAdjustmentBackend, 0, CERES, CASPAR);

// Termination type for bundle adjustment, independent of solver backend
// (upstream parity, d3ccaf35 estimators/bundle_adjustment.h).
MAKE_ENUM_CLASS_OVERLOAD_STREAM(BundleAdjustmentTerminationType,
                                0,
                                CONVERGENCE,
                                NO_CONVERGENCE,
                                FAILURE,
                                USER_SUCCESS,
                                USER_FAILURE);

// The gauge fixing strategy for bundle adjustment (upstream parity,
// d3ccaf35 estimators/bundle_adjustment.h). Without gauge fixing, the
// reconstruction problem has a global 7-DoF null space that makes the
// normal equations singular for noise-free problems.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(BundleAdjustmentGauge,
                                -1,
                                UNSPECIFIED,
                                TWO_CAMS_FROM_WORLD,
                                THREE_POINTS);

// Summary of bundle adjustment results, independent of solver backend
// (upstream parity). The fork keeps its bool Solve + ceres Summary()
// interface; the backend-specific summary subclass carries the full solver
// details (see CeresBundleAdjustmentSummary in bundle_adjustment_ceres.h).
struct BundleAdjustmentSummary {
    BundleAdjustmentTerminationType termination_type =
            BundleAdjustmentTerminationType::FAILURE;
    // Number of residuals connected to at least one variable parameter
    // block. Excludes residuals where all connected parameters are constant.
    int num_residuals = 0;

    bool IsSolutionUsable() const;
    virtual std::string BriefReport() const;

    virtual ~BundleAdjustmentSummary() = default;
};

// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentConfig {
public:
    BundleAdjustmentConfig();

    size_t NumImages() const;
    size_t NumPoints() const;
    size_t NumConstantCameras() const;

    // Upstream parity (d3ccaf35 bundle_adjustment.h).
    size_t NumConstantSensorFromRigPoses() const {
        return constant_sensor_from_rig_poses_.size();
    }
    size_t NumConstantPoses() const;
    size_t NumConstantTvecs() const;
    size_t NumVariablePoints() const;
    size_t NumConstantPoints() const;

    // Determine the number of residuals for the given reconstruction. The
    // number of residuals equals the number of observations times two.
    size_t NumResiduals(const Reconstruction& reconstruction) const;

    // Add / remove images from the configuration.
    void AddImage(const image_t image_id);
    // Upstream parity: points in this set are excluded from the residual
    // construction of the bundle adjustment problem.
    void IgnorePoint(const point3D_t point3D_id);
    bool IsIgnoredPoint(const point3D_t point3D_id) const;
    bool HasImage(const image_t image_id) const;
    void RemoveImage(const image_t image_id);

    // Set cameras of added images as constant or variable. By default all
    // cameras of added images are variable. Note that the corresponding images
    // have to be added prior to calling these methods.
    void SetConstantCamera(const camera_t camera_id);
    void SetVariableCamera(const camera_t camera_id);
    bool IsConstantCamera(const camera_t camera_id) const;

    // Set the pose of added images as constant. The pose is defined as the
    // rotational and translational part of the projection matrix.
    void SetConstantPose(const image_t image_id);
    void SetVariablePose(const image_t image_id);
    bool HasConstantPose(const image_t image_id) const;

    // Upstream parity (d3ccaf35 estimators/bundle_adjustment.h): frame-level
    // pose and rig-extrinsic constants. The rig-from-world pose of the
    // owning frame is fixed as a whole (rotation and translation); the
    // sensor-from-rig extrinsics of a rig sensor are fixed independently.
    void SetConstantRigFromWorldPose(const frame_t frame_id);
    void SetVariableRigFromWorldPose(const frame_t frame_id);
    bool HasConstantRigFromWorldPose(const frame_t frame_id) const;
    void SetConstantSensorFromRigPose(const sensor_t sensor_id);
    void SetVariableSensorFromRigPose(const sensor_t sensor_id);
    bool HasConstantSensorFromRigPose(const sensor_t sensor_id) const;

    // Upstream parity (d3ccaf35): fix the global gauge of the problem.
    // UNSPECIFIED leaves the gauge unhandled (matching the fork's previous
    // behavior); TWO_CAMS_FROM_WORLD fixes one full frame pose and one
    // translation dimension of a second frame; THREE_POINTS fixes three
    // non-collinear 3D points.
    void FixGauge(BundleAdjustmentGauge gauge);
    BundleAdjustmentGauge FixedGauge() const;

    // Set the translational part of the pose, hence the constant pose
    // indices may be in [0, 1, 2] and must be unique. Note that the
    // corresponding images have to be added prior to calling these methods.
    void SetConstantTvec(const image_t image_id, const std::vector<int>& idxs);
    void RemoveConstantTvec(const image_t image_id);
    bool HasConstantTvec(const image_t image_id) const;

    // Add / remove points from the configuration. Note that points can either
    // be variable or constant but not both at the same time.
    void AddVariablePoint(const point3D_t point3D_id);
    void AddConstantPoint(const point3D_t point3D_id);
    bool HasPoint(const point3D_t point3D_id) const;
    bool HasVariablePoint(const point3D_t point3D_id) const;
    bool HasConstantPoint(const point3D_t point3D_id) const;
    void RemoveVariablePoint(const point3D_t point3D_id);
    void RemoveConstantPoint(const point3D_t point3D_id);

    // Access configuration data.
    const std::unordered_set<image_t>& Images() const;
    const std::unordered_set<point3D_t>& VariablePoints() const;
    const std::unordered_set<point3D_t>& ConstantPoints() const;
    const std::vector<int>& ConstantTvec(const image_t image_id) const;

private:
    std::unordered_set<camera_t> constant_camera_ids_;
    std::unordered_set<image_t> image_ids_;
    std::unordered_set<point3D_t> ignored_point3D_ids_;
    std::unordered_set<point3D_t> variable_point3D_ids_;
    std::unordered_set<point3D_t> constant_point3D_ids_;
    std::unordered_set<image_t> constant_poses_;
    std::unordered_map<image_t, std::vector<int>> constant_tvecs_;
    std::unordered_set<frame_t> constant_rig_from_world_poses_;
    std::unordered_set<sensor_t> constant_sensor_from_rig_poses_;
    BundleAdjustmentGauge fixed_gauge_ = BundleAdjustmentGauge::UNSPECIFIED;
};

// Backend-specific option groups. Upstream parity
// (estimators/bundle_adjustment.h): the Ceres and Caspar options are owned
// through shared pointers so the option manager can bind to them, and copies
// of BundleAdjustmentOptions deep-copy both groups.
struct CeresBundleAdjustmentOptions;
struct CasparBundleAdjustmentOptions;

struct BundleAdjustmentBackendOptions {
    std::shared_ptr<CeresBundleAdjustmentOptions> ceres;
    std::shared_ptr<CasparBundleAdjustmentOptions> caspar;

    BundleAdjustmentBackendOptions();
    BundleAdjustmentBackendOptions(const BundleAdjustmentBackendOptions& other);
    BundleAdjustmentBackendOptions& operator=(
            const BundleAdjustmentBackendOptions& other);
    BundleAdjustmentBackendOptions(BundleAdjustmentBackendOptions&& other) =
            default;
    BundleAdjustmentBackendOptions& operator=(
            BundleAdjustmentBackendOptions&& other) = default;
};

struct BundleAdjustmentOptions : public BundleAdjustmentBackendOptions {
    // Whether to refine the focal length parameter group.
    bool refine_focal_length = true;

    // Whether to refine the principal point parameter group.
    bool refine_principal_point = false;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = true;

    // Whether to refine the sensor from rig poses.
    bool refine_sensor_from_rig = true;

    // Whether to refine the rig from world poses.
    bool refine_rig_from_world = true;

    // Whether to refine the 3D points.
    bool refine_points3D = true;

    // Minimum track length of a 3D point to be included in the problem.
    int min_track_length = 0;

    // Whether to keep the rig-from-world rotations constant (only refine the
    // translations).
    bool constant_rig_from_world_rotation = false;

    // Whether to print a final summary.
    bool print_summary = true;

    // Which backend solves the problem; the matching option group above
    // configures it.
    BundleAdjustmentBackend backend = BundleAdjustmentBackend::CERES;

    // Cooperative cancellation hook, evaluated once per solver iteration.
    std::function<bool()> check_if_stopped;

    bool Check() const;
};

// Solver-agnostic pose prior bundle adjustment options; the Ceres-specific
// loss configuration is bound as `ceres` (upstream parity).
struct CeresPosePriorBundleAdjustmentOptions;

struct PosePriorBundleAdjustmentBackendOptions {
    std::shared_ptr<CeresPosePriorBundleAdjustmentOptions> ceres;

    PosePriorBundleAdjustmentBackendOptions();
    PosePriorBundleAdjustmentBackendOptions(
            const PosePriorBundleAdjustmentBackendOptions& other);
    PosePriorBundleAdjustmentBackendOptions& operator=(
            const PosePriorBundleAdjustmentBackendOptions& other);
    PosePriorBundleAdjustmentBackendOptions(
            PosePriorBundleAdjustmentBackendOptions&& other) = default;
    PosePriorBundleAdjustmentBackendOptions& operator=(
            PosePriorBundleAdjustmentBackendOptions&& other) = default;
};

struct PosePriorBundleAdjustmentOptions
    : public PosePriorBundleAdjustmentBackendOptions {
    // Fallback if no prior position covariance is provided.
    double prior_position_fallback_stddev = 1.0;

    // Sim3 alignment options.
    RANSACOptions alignment_ransac_options;

    bool Check() const;
};

#ifdef CASPAR_ENABLED
// Runs the subset of Caspar's generated graph that is representable by the
// legacy image-pose reconstruction model. Returns false without modifying the
// reconstruction when a requested problem needs a Caspar factor variant that
// has not yet been ported; BundleAdjuster then runs the Ceres path.
bool SolveCasparBundleAdjustment(const BundleAdjustmentOptions& options,
                                 const BundleAdjustmentConfig& config,
                                 Reconstruction* reconstruction,
                                 ceres::Solver::Summary* ceres_summary);
#endif

// Abstract base class for bundle adjustment, independent of the solver
// backend (upstream dbb41680 parity; the fork keeps the bool
// Solve(Reconstruction*) signature and the ceres summary accessor so
// existing callers stay source-compatible). The fork's backend dispatch
// (CASPAR first, Ceres fallback) is performed by the backend implementation
// returned from CreateDefaultBundleAdjuster.
class BundleAdjuster {
public:
    BundleAdjuster(const BundleAdjustmentOptions& options,
                   const BundleAdjustmentConfig& config)
        : options_(options), config_(config) {}
    virtual ~BundleAdjuster() = default;

    NON_COPYABLE(BundleAdjuster)

    virtual bool Solve(Reconstruction* reconstruction) = 0;

    // W7 pose-prior bundle adjustment: when set, a Sim3 alignment against the
    // priors precedes the solve and an absolute position prior residual is
    // added for every image with a prior (upstream
    // CreatePosePriorBundleAdjuster parity).
    virtual void SetPosePriors(PosePriorBundleAdjustmentOptions options,
                               std::vector<PosePrior> pose_priors) = 0;

    // Get the solver summary for the last call to `Solve`.
    virtual const ceres::Solver::Summary& Summary() const = 0;

    const BundleAdjustmentOptions& Options() const { return options_; }
    const BundleAdjustmentConfig& Config() const { return config_; }

protected:
    BundleAdjustmentOptions options_;
    BundleAdjustmentConfig config_;
};

// Factory function to create bundle adjusters (upstream parity). Returns the
// backend implementation selected by the options; the Ceres implementation
// carries the CASPAR-first dispatch with Ceres fallback.
std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
        const BundleAdjustmentOptions& options,
        const BundleAdjustmentConfig& config);

void PrintSolverSummary(const ceres::Solver::Summary& summary);

}  // namespace colmap
