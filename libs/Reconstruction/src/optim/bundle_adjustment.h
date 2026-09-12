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
#include <unordered_set>

#include "base/reconstruction.h"
#include "optim/manifold.h"
#include "util/alignment.h"

namespace colmap {

// Keep the solver selection independent of the caller. Incremental mapping,
// global mapping, the GUI controller, and the CLI all enter through
// BundleAdjuster, so a backend must be selected here rather than in one caller.
enum class BundleAdjustmentBackend { CERES, CASPAR };

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
    std::unordered_set<point3D_t> variable_point3D_ids_;
    std::unordered_set<point3D_t> constant_point3D_ids_;
    std::unordered_set<image_t> constant_poses_;
    std::unordered_map<image_t, std::vector<int>> constant_tvecs_;
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

// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
class BundleAdjuster {
public:
    BundleAdjuster(const BundleAdjustmentOptions& options,
                   const BundleAdjustmentConfig& config);

    bool Solve(Reconstruction* reconstruction);

    // Get the Ceres solver summary for the last call to `Solve`.
    const ceres::Solver::Summary& Summary() const;

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

protected:
    void ParameterizeCameras(Reconstruction* reconstruction);
    void ParameterizePoints(Reconstruction* reconstruction);

    const BundleAdjustmentOptions options_;
    BundleAdjustmentConfig config_;
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
    std::unordered_set<camera_t> camera_ids_;
    std::unordered_map<point3D_t, size_t> point3D_num_observations_;

    // W3-2b step 4: shadow parameter blocks for refined sensor_from_rig
    // poses, in the fork's [w, x, y, z] qvec convention (the fork stores
    // sensor_from_rig as a Rigid3d whose Eigen quaternion coefficients are
    // [x, y, z, w]). Node-based map so block addresses stay stable while
    // SetUp adds images.
    struct SensorPoseBlock {
        rig_t rig_id;
        sensor_t sensor_id;
        Eigen::Vector4d qvec;
        Eigen::Vector3d tvec;
    };
    std::map<std::pair<rig_t, sensor_t>, SensorPoseBlock> sensor_blocks_;
    // Frame blocks are shared by every image of a frame; guard the one-time
    // quaternion manifold installation (repeated installation aborts).
    std::unordered_set<const double*> manifold_marked_blocks_;

    SensorPoseBlock& GetOrCreateSensorBlock(rig_t rig_id,
                                            sensor_t sensor_id,
                                            const Rigid3d& sensor_from_rig);
};

void PrintSolverSummary(const ceres::Solver::Summary& summary);

}  // namespace colmap
