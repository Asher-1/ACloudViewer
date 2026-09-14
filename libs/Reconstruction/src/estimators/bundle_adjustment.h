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
enum class BundleAdjustmentBackend { CERES, CASPAR };

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

struct BundleAdjustmentOptions {
    // Loss function types: Trivial (non-robust) and Cauchy/Huber (robust)
    // losses. HUBER is the upstream COLMAP default for the global mapper.
    enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY, HUBER };
    LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

    // Scaling factor determines residual at which robustification takes place.
    double loss_function_scale = 1.0;

    // Whether to refine the focal length parameter group.
    bool refine_focal_length = true;

    // Whether to refine the principal point parameter group.
    bool refine_principal_point = false;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = true;

    // Whether to refine the extrinsic parameter group.
    bool refine_extrinsics = true;

    // Whether to refine the rig-from-world poses of frames (upstream parity,
    // dbb41680 estimators/bundle_adjustment.h). Until the frame-shared
    // parameter blocks land, this gates the legacy per-image extrinsic
    // blocks.
    bool refine_rig_from_world = true;

    // Minimum track length for a 3D point to be included in bundle
    // adjustment (upstream parity, dbb41680). Zero disables the filter.
    int min_track_length = 0;

    // Whether to refine the sensor-from-rig poses of non-reference rig
    // sensors (upstream parity, dbb41680).
    bool refine_sensor_from_rig = true;

    // Whether to keep the rotation of all rig-from-world poses constant,
    // refining positions only (upstream parity, dbb41680; the global
    // mapper's fixed-rotation stage enables this per pass).
    bool constant_rig_from_world_rotation = false;

    // Whether to print a final summary.
    bool print_summary = true;

    // Ceres remains the portable default. CASPAR is available only in CUDA
    // builds that explicitly enable RECONSTRUCTION_CASPAR_ENABLED.
    BundleAdjustmentBackend backend = BundleAdjustmentBackend::CERES;

    // CLI/UI-facing switch until the legacy option parser can expose the enum
    // directly. It is equivalent to backend == CASPAR.
    bool use_caspar = false;
    int caspar_gpu_index = -1;
    int caspar_max_num_iterations = 200;

    // Optional cooperative cancellation hook, checked once per Ceres
    // iteration. Return true to abort the solve while keeping intermediate
    // results; used by the graceful-shutdown path.
    std::function<bool()> check_if_stopped;

    // Whether to use Ceres' CUDA linear algebra library, if available.
    bool use_gpu = false;
    std::string gpu_index = "-1";

    // Heuristic threshold to switch from CPU to GPU based solvers.
    // Typically, the GPU is faster for large problems but the overhead of
    // transferring memory from the CPU to the GPU leads to better CPU
    // performance for small problems. This depends on the specific problem and
    // hardware.
    int min_num_images_gpu_solver = 50;

    // Heuristic threshold on the minimum number of residuals to enable
    // multi-threading. Note that single-threaded is typically better for small
    // bundle adjustment problems due to the overhead of threading.
    int min_num_residuals_for_cpu_multi_threading = 50000;

    // Heuristic thresholds to switch between direct, sparse, and iterative
    // solvers. These thresholds may not be optimal for all types of problems.
    int max_num_images_direct_dense_cpu_solver = 50;
    int max_num_images_direct_sparse_cpu_solver = 1000;
    int max_num_images_direct_dense_gpu_solver = 200;
    int max_num_images_direct_sparse_gpu_solver = 4000;

    // Ceres-Solver options.
    ceres::Solver::Options solver_options;

    BundleAdjustmentOptions() {
        solver_options.function_tolerance = 0.0;
        solver_options.gradient_tolerance = 1e-4;
        solver_options.parameter_tolerance = 0.0;
        solver_options.logging_type = ceres::LoggingType::SILENT;
        solver_options.minimizer_progress_to_stdout = false;
        solver_options.max_num_iterations = 100;
        solver_options.max_linear_solver_iterations = 200;
        solver_options.max_num_consecutive_invalid_steps = 10;
        solver_options.max_consecutive_nonmonotonic_steps = 10;
        solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
    }

    // Create a new loss function based on the specified options. The caller
    // takes ownership of the loss function.
    ceres::LossFunction* CreateLossFunction() const;

    // Create options tailored for given bundle adjustment config and problem.
    ceres::Solver::Options CreateSolverOptions(
            const BundleAdjustmentConfig& config,
            const ceres::Problem& problem) const;

    bool Check() const;
};

// Solver-agnostic pose prior bundle adjustment options (upstream parity; the
// fork keeps a flat struct until W18.5 introduces the backend pimpl split).
struct PosePriorBundleAdjustmentOptions {
    // Fallback if no prior position covariance is provided.
    double prior_position_fallback_stddev = 1.0;

    // Sim3 alignment options.
    RANSACOptions alignment_ransac_options;

    // Loss function for the prior position residuals.
    BundleAdjustmentOptions::LossFunctionType
            prior_position_loss_function_type =
                    BundleAdjustmentOptions::LossFunctionType::TRIVIAL;

    // Threshold on the residual for the robust loss.
    double prior_position_loss_scale = std::sqrt(kChiSquare95ThreeDof);

    bool Check() const {
        CHECK_OPTION_GE(prior_position_fallback_stddev, 0.0);
        CHECK_OPTION_GE(prior_position_loss_scale, 0.0);
        return true;
    }
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
