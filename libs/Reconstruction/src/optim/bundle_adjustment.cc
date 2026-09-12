// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "optim/bundle_adjustment.h"

#include <map>

#if defined(CASPAR_ENABLED)
#include "optim/bundle_adjustment_caspar.h"
#endif

#include <iomanip>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/projection.h"
#include "optim/manifold.h"
#include "util/cuda.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentConfig::BundleAdjustmentConfig() {}

size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
    return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCameras() const {
    return constant_camera_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoses() const {
    return constant_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantTvecs() const {
    return constant_tvecs_.size();
}

size_t BundleAdjustmentConfig::NumVariablePoints() const {
    return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoints() const {
    return constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumResiduals(
        const Reconstruction& reconstruction) const {
    // Count the number of observations for all added images.
    size_t num_observations = 0;
    for (const image_t image_id : image_ids_) {
        num_observations += reconstruction.Image(image_id).NumPoints3D();
    }

    // Count the number of observations for all added 3D points that are not
    // already added as part of the images above.

    auto NumObservationsForPoint =
            [this, &reconstruction](const point3D_t point3D_id) {
                size_t num_observations_for_point = 0;
                const auto& point3D = reconstruction.Point3D(point3D_id);
                for (const auto& track_el : point3D.Track().Elements()) {
                    if (image_ids_.count(track_el.image_id) == 0) {
                        num_observations_for_point += 1;
                    }
                }
                return num_observations_for_point;
            };

    for (const auto point3D_id : variable_point3D_ids_) {
        num_observations += NumObservationsForPoint(point3D_id);
    }
    for (const auto point3D_id : constant_point3D_ids_) {
        num_observations += NumObservationsForPoint(point3D_id);
    }

    return 2 * num_observations;
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) {
    image_ids_.insert(image_id);
}

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
    return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
    image_ids_.erase(image_id);
}

void BundleAdjustmentConfig::SetConstantCamera(const camera_t camera_id) {
    constant_camera_ids_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamera(const camera_t camera_id) {
    constant_camera_ids_.erase(camera_id);
}

bool BundleAdjustmentConfig::IsConstantCamera(const camera_t camera_id) const {
    return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
}

void BundleAdjustmentConfig::SetConstantPose(const image_t image_id) {
    CHECK(HasImage(image_id));
    CHECK(!HasConstantTvec(image_id));
    constant_poses_.insert(image_id);
}

void BundleAdjustmentConfig::SetVariablePose(const image_t image_id) {
    constant_poses_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantPose(const image_t image_id) const {
    return constant_poses_.find(image_id) != constant_poses_.end();
}

void BundleAdjustmentConfig::SetConstantTvec(const image_t image_id,
                                             const std::vector<int>& idxs) {
    CHECK_GT(idxs.size(), 0);
    CHECK_LE(idxs.size(), 3);
    CHECK(HasImage(image_id));
    CHECK(!HasConstantPose(image_id));
    CHECK(!VectorContainsDuplicateValues(idxs))
            << "Tvec indices must not contain duplicates";
    constant_tvecs_.emplace(image_id, idxs);
}

void BundleAdjustmentConfig::RemoveConstantTvec(const image_t image_id) {
    constant_tvecs_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantTvec(const image_t image_id) const {
    return constant_tvecs_.find(image_id) != constant_tvecs_.end();
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::Images() const {
    return image_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::VariablePoints()
        const {
    return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::ConstantPoints()
        const {
    return constant_point3D_ids_;
}

const std::vector<int>& BundleAdjustmentConfig::ConstantTvec(
        const image_t image_id) const {
    return constant_tvecs_.at(image_id);
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
    CHECK(!HasConstantPoint(point3D_id));
    variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
    CHECK(!HasVariablePoint(point3D_id));
    constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
    return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(
        const point3D_t point3D_id) const {
    return variable_point3D_ids_.find(point3D_id) !=
           variable_point3D_ids_.end();
}

bool BundleAdjustmentConfig::HasConstantPoint(
        const point3D_t point3D_id) const {
    return constant_point3D_ids_.find(point3D_id) !=
           constant_point3D_ids_.end();
}

void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
    variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
    constant_point3D_ids_.erase(point3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction* BundleAdjustmentOptions::CreateLossFunction() const {
    ceres::LossFunction* loss_function = nullptr;
    switch (loss_function_type) {
        case LossFunctionType::TRIVIAL:
            loss_function = new ceres::TrivialLoss();
            break;
        case LossFunctionType::SOFT_L1:
            loss_function = new ceres::SoftLOneLoss(loss_function_scale);
            break;
        case LossFunctionType::CAUCHY:
            loss_function = new ceres::CauchyLoss(loss_function_scale);
            break;
        case LossFunctionType::HUBER:
            loss_function = new ceres::HuberLoss(loss_function_scale);
            break;
    }
    CHECK_NOTNULL(loss_function);
    return loss_function;
}

bool BundleAdjustmentOptions::Check() const {
    CHECK_OPTION_GE(loss_function_scale, 0);
    CHECK_OPTION_LT(max_num_images_direct_dense_cpu_solver,
                    max_num_images_direct_sparse_cpu_solver);
    CHECK_OPTION_LT(max_num_images_direct_dense_gpu_solver,
                    max_num_images_direct_sparse_gpu_solver);
    return true;
}

ceres::Solver::Options BundleAdjustmentOptions::CreateSolverOptions(
        const BundleAdjustmentConfig& config,
        const ceres::Problem& problem) const {
    ceres::Solver::Options custom_solver_options = solver_options;
    if (VLOG_IS_ON(2)) {
        custom_solver_options.minimizer_progress_to_stdout = true;
        custom_solver_options.logging_type =
                ceres::LoggingType::PER_MINIMIZER_ITERATION;
    }

    const int num_images = config.NumImages();
    const bool has_sparse =
            custom_solver_options.sparse_linear_algebra_library_type !=
            ceres::NO_SPARSE;

    int max_num_images_direct_dense_solver =
            max_num_images_direct_dense_cpu_solver;
    int max_num_images_direct_sparse_solver =
            max_num_images_direct_sparse_cpu_solver;

    if (use_gpu) {
        LOG_FIRST_N(WARNING, 1)
                << "Ceres bundle adjustment is CPU-only in this build; "
                   "use the independent Caspar backend for GPU BA.";
    }

    if (num_images <= max_num_images_direct_dense_solver) {
        custom_solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    } else if (has_sparse &&
               num_images <= max_num_images_direct_sparse_solver) {
        custom_solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {  // Indirect sparse (preconditioned CG) solver.
        custom_solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        custom_solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
    }

    if (problem.NumResiduals() < min_num_residuals_for_cpu_multi_threading) {
        custom_solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
        custom_solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
    } else {
        custom_solver_options.num_threads =
                GetEffectiveNumThreads(custom_solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
        custom_solver_options.num_linear_solver_threads =
                GetEffectiveNumThreads(
                        custom_solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
    }

    std::string solver_error;
    CHECK(custom_solver_options.IsValid(&solver_error)) << solver_error;
    return custom_solver_options;
}

namespace {

// Wraps the options-level check_if_stopped hook into a Ceres iteration
// callback: returning USER_ABORT stops the minimizer after the current
// iteration while keeping all accepted parameter updates.
class CancellationCallback : public ceres::IterationCallback {
public:
    explicit CancellationCallback(std::function<bool()> check_if_stopped)
        : check_if_stopped_(std::move(check_if_stopped)) {}

    ceres::CallbackReturnType operator()(
            const ceres::IterationSummary&) override {
        // Upstream Ceres 2.x renamed SOLVER_ABORT to USER_ABORT; the
        // self-built Ceres in this repo keeps the SOLVER_ABORT spelling.
        return check_if_stopped_ && check_if_stopped_()
                       ? ceres::SOLVER_ABORT
                       : ceres::SOLVER_CONTINUE;
    }

private:
    std::function<bool()> check_if_stopped_;
};

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// BundleAdjuster
////////////////////////////////////////////////////////////////////////////////

BundleAdjuster::BundleAdjuster(const BundleAdjustmentOptions& options,
                               const BundleAdjustmentConfig& config)
    : options_(options), config_(config) {
    CHECK(options_.Check());
}

bool BundleAdjuster::Solve(Reconstruction* reconstruction) {
    CHECK_NOTNULL(reconstruction);
    CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

    const bool use_caspar =
            options_.use_caspar ||
            options_.backend == BundleAdjustmentBackend::CASPAR;
    if (use_caspar) {
#if defined(CASPAR_ENABLED)
        if (SolveCasparBundleAdjustment(options_, config_, reconstruction,
                                        &summary_)) {
            return true;
        }
        LOG(WARNING) << "Caspar BA cannot represent this legacy problem; "
                        "using the Ceres backend for this solve";
#else
        LOG(WARNING) << "Caspar BA was requested, but this build does not "
                        "enable RECONSTRUCTION_CASPAR_ENABLED; using Ceres";
#endif
    }

    problem_.reset(new ceres::Problem());

    ceres::LossFunction* loss_function = options_.CreateLossFunction();
    SetUp(reconstruction, loss_function);

    if (problem_->NumResiduals() == 0) {
        return false;
    }

    ceres::Solver::Options solver_options =
            options_.CreateSolverOptions(config_, *problem_);

    // Cooperative cancellation: check_if_stopped is evaluated once per
    // iteration; a true return aborts the minimizer while all accepted
    // parameter updates stay applied via TearDown().
    CancellationCallback cancellation_callback(options_.check_if_stopped);
    if (options_.check_if_stopped) {
        solver_options.callbacks.push_back(&cancellation_callback);
    }

    ceres::Solve(solver_options, problem_.get(), &summary_);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options_.print_summary) {
        PrintHeading2("Bundle adjustment report");
        PrintSolverSummary(summary_);
    }

    TearDown(reconstruction);

    return true;
}

const ceres::Solver::Summary& BundleAdjuster::Summary() const {
    return summary_;
}

void BundleAdjuster::SetUp(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
    // Warning: AddPointsToProblem assumes that AddImageToProblem is called
    // first. Do not change order of instructions!

    manifold_marked_blocks_.clear();

    // W3-2b step 4: forward dual-track sync. The pose parameter blocks are
    // the frame storage; make sure it holds the freshest camera poses before
    // the problem is built (the global mapper pre-syncs frames itself, but
    // image-level writers like the legacy incremental mapper may not).
    for (const image_t image_id : config_.Images()) {
        if (!reconstruction->ExistsImage(image_id)) {
            continue;
        }
        Image& image = reconstruction->Image(image_id);
        if (!image.HasFrameId() || !image.HasPose() ||
            !reconstruction->ExistsFrame(image.FrameId())) {
            continue;
        }
        const Rigid3d cam_from_world(
                Eigen::Quaterniond(image.Qvec()(0), image.Qvec()(1),
                                   image.Qvec()(2), image.Qvec()(3)),
                image.Tvec());
        reconstruction->Frame(image.FrameId())
                .SetCamFromWorld(image.CameraId(), cam_from_world);
    }
    for (const image_t image_id : config_.Images()) {
        AddImageToProblem(image_id, reconstruction, loss_function);
    }
    for (const auto point3D_id : config_.VariablePoints()) {
        AddPointToProblem(point3D_id, reconstruction, loss_function);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
        AddPointToProblem(point3D_id, reconstruction, loss_function);
    }

    ParameterizeCameras(reconstruction);
    ParameterizePoints(reconstruction);
}

void BundleAdjuster::TearDown(Reconstruction* reconstruction) {
    if (reconstruction == nullptr) {
        return;
    }
    // W3-2b step 4: the pose parameter blocks were the frame storage itself,
    // so the refined rig_from_world poses are already in place. Write the
    // refined sensor_from_rig poses back to the rigs and mirror the composed
    // camera poses into the legacy per-image buffers so both pose tracks stay
    // consistent for every consumer.
    for (const auto& [key, block] : sensor_blocks_) {
        if (!reconstruction->ExistsRig(key.first)) {
            continue;
        }
        Rig& rig = reconstruction->Rig(key.first);
        if (!rig.HasSensor(block.sensor_id)) {
            continue;
        }
        const Eigen::Quaterniond q(block.qvec(0), block.qvec(1), block.qvec(2),
                                   block.qvec(3));
        rig.SetSensorFromRig(block.sensor_id,
                             Rigid3d(q.normalized(), block.tvec));
    }

    for (const image_t image_id : config_.Images()) {
        if (!reconstruction->ExistsImage(image_id)) {
            continue;
        }
        Image& image = reconstruction->Image(image_id);
        if (!image.HasFrameId() ||
            !reconstruction->ExistsFrame(image.FrameId())) {
            continue;
        }
        const Frame& frame = reconstruction->Frame(image.FrameId());
        if (!frame.HasPose()) {
            continue;
        }
        Rigid3d cam_from_world = frame.RigFromWorld();
        const Rig& rig = reconstruction->Rig(frame.RigId());
        const sensor_t sensor_id =
                reconstruction->Camera(image.CameraId()).SensorId();
        if (rig.NumSensors() > 1 && rig.HasSensor(sensor_id) &&
            !rig.IsRefSensor(sensor_id) &&
            rig.HasSensorFromRig(sensor_id)) {
            cam_from_world = cam_from_world * rig.SensorFromRig(sensor_id);
        }
        const Eigen::Quaterniond& q = cam_from_world.rotation();
        image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
        image.SetTvec(cam_from_world.translation());
    }
    sensor_blocks_.clear();
}

BundleAdjuster::SensorPoseBlock& BundleAdjuster::GetOrCreateSensorBlock(
        const rig_t rig_id,
        const sensor_t sensor_id,
        const Rigid3d& sensor_from_rig) {
    const auto key = std::make_pair(rig_id, sensor_id);
    const auto it = sensor_blocks_.find(key);
    if (it != sensor_blocks_.end()) {
        return it->second;
    }
    SensorPoseBlock block;
    block.rig_id = rig_id;
    block.sensor_id = sensor_id;
    const Eigen::Quaterniond& q = sensor_from_rig.rotation();
    block.qvec = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
    block.tvec = sensor_from_rig.translation();
    return sensor_blocks_.emplace(key, block).first->second;
}

void BundleAdjuster::AddImageToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
    Image& image = reconstruction->Image(image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());

    // W3-2b step 4: the pose parameter blocks are the frame storage shared
    // by every image of the frame; images without a frame (legacy fixtures)
    // keep their own buffers.
    bool frame_pose = false;
    bool compose_rig = false;
    bool frame_rotation_constant = false;
    sensor_t rig_sensor_id{};
    Eigen::Vector4d* sensor_qvec_data = nullptr;
    Eigen::Vector3d* sensor_tvec_data = nullptr;
    double* qvec_data = nullptr;
    double* tvec_data = nullptr;
    double* camera_params_data = camera.ParamsData();

    if (image.HasFrameId() && reconstruction->ExistsFrame(image.FrameId())) {
        Frame& frame = reconstruction->Frame(image.FrameId());
        frame_pose = true;
        qvec_data = frame.RigFromWorldQvec().data();
        tvec_data = frame.RigFromWorldTvec().data();
        const Rig& rig = reconstruction->Rig(frame.RigId());
        frame_rotation_constant =
                options_.constant_rig_from_world_rotation;
        if (options_.refine_sensor_from_rig && rig.NumSensors() > 1 &&
            rig.HasSensor(camera.SensorId()) &&
            !rig.IsRefSensor(camera.SensorId()) &&
            rig.HasSensorFromRig(camera.SensorId())) {
            compose_rig = true;
            rig_sensor_id = camera.SensorId();
            SensorPoseBlock& block =
                    GetOrCreateSensorBlock(frame.RigId(), rig_sensor_id,
                                           rig.SensorFromRig(rig_sensor_id));
            sensor_qvec_data = &block.qvec;
            sensor_tvec_data = &block.tvec;
        }
    } else {
        // CostFunction assumes unit quaternions.
        image.NormalizeQvec();
        qvec_data = image.Qvec().data();
        tvec_data = image.Tvec().data();
    }

    const bool constant_pose =
            !options_.refine_extrinsics || !options_.refine_rig_from_world ||
            config_.HasConstantPose(image_id);

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D()) {
            continue;
        }

        num_observations += 1;
        point3D_num_observations_[point2D.Point3DId()] += 1;

        Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
        assert(point3D.Track().Length() > 1);

        ceres::CostFunction* cost_function = nullptr;

        if (constant_pose) {
            // Read the pose through the frame-aware accessor so stale legacy
            // buffers (non-config track images) cannot leak in.
            const Rigid3d pose_cam_from_world =
                    frame_pose ? image.CamFromWorld()
                               : Rigid3d(Eigen::Quaterniond(
                                             image.Qvec()(0), image.Qvec()(1),
                                             image.Qvec()(2), image.Qvec()(3)),
                                         image.Tvec());
            const Eigen::Quaterniond pose_q = pose_cam_from_world.rotation();
            const Eigen::Vector4d pose_qvec(
                    pose_q.w(), pose_q.x(), pose_q.y(), pose_q.z());
            if (camera.ModelId() == EquirectangularCameraModel::kModelId) {
                cost_function =
                        EquirectangularBundleAdjustmentConstantPoseCostFunction::Create(
                                pose_qvec, pose_cam_from_world.translation(),
                                point2D.XY());
            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::kModelId:                                                \
        cost_function =                                                        \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
                        pose_qvec, pose_cam_from_world.translation(),          \
                        point2D.XY());                                         \
        break;

                CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
            }
            }

            problem_->AddResidualBlock(cost_function, loss_function,
                                       point3D.XYZ().data(),
                                       camera_params_data);
        } else if (compose_rig &&
                   camera.ModelId() == EquirectangularCameraModel::kModelId) {
            // Fork note: no composed equirectangular rig functor yet; freeze
            // the composed camera pose at its setup value (2-block residual).
            const Rigid3d pose_cam_from_world = image.CamFromWorld();
            const Eigen::Quaterniond pose_q = pose_cam_from_world.rotation();
            const Eigen::Vector4d pose_qvec(
                    pose_q.w(), pose_q.x(), pose_q.y(), pose_q.z());
            cost_function =
                    EquirectangularBundleAdjustmentConstantPoseCostFunction::Create(
                            pose_qvec, pose_cam_from_world.translation(),
                            point2D.XY());
            problem_->AddResidualBlock(cost_function, loss_function,
                                       point3D.XYZ().data(),
                                       camera_params_data);
        } else {
            if (compose_rig) {
                // Multi-sensor frame: composed residual over the shared
                // rig pose and this sensor's pose (equirectangular cameras
                // in rigs take the frozen branch above).
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::kModelId:                                                \
        cost_function =                                                        \
                FrameRigBundleAdjustmentCostFunction<CameraModel>::Create(     \
                        point2D.XY());                                         \
        break;

                    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                }
                problem_->AddResidualBlock(
                        cost_function, loss_function, qvec_data, tvec_data,
                        sensor_qvec_data->data(), sensor_tvec_data->data(),
                        point3D.XYZ().data(), camera_params_data);
            } else {
                if (camera.ModelId() ==
                    EquirectangularCameraModel::kModelId) {
                    cost_function =
                            EquirectangularBundleAdjustmentCostFunction::Create(
                                    point2D.XY());
                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
    case CameraModel::kModelId:                                            \
        cost_function = BundleAdjustmentCostFunction<CameraModel>::Create( \
                point2D.XY());                                             \
        break;

                    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
                    }
                }
                problem_->AddResidualBlock(cost_function, loss_function,
                                           qvec_data, tvec_data,
                                           point3D.XYZ().data(),
                                           camera_params_data);
            }
        }
    }

    if (num_observations > 0) {
        camera_ids_.insert(image.CameraId());

        // Set pose parameterization. Constant-pose images use frozen
        // composed-pose functors that never register the frame block, so
        // only variable residuals may mark it here.
        if (frame_pose && !constant_pose) {
            if (frame_rotation_constant) {
                problem_->SetParameterBlockConstant(qvec_data);
            } else if (manifold_marked_blocks_.insert(qvec_data).second) {
                // ceres::LocalParameterization* quaternion_parameterization =
                //     new ceres::QuaternionParameterization;
                // problem_->SetParameterization(qvec_data,
                // quaternion_parameterization);
                SetQuaternionManifold(problem_.get(), qvec_data);
            }
            if (compose_rig) {
                if (manifold_marked_blocks_.insert(sensor_qvec_data->data())
                            .second) {
                    SetQuaternionManifold(problem_.get(),
                                          sensor_qvec_data->data());
                }
                if (!options_.refine_sensor_from_rig) {
                    problem_->SetParameterBlockConstant(sensor_qvec_data->data());
                    problem_->SetParameterBlockConstant(sensor_tvec_data->data());
                }
            }
            if (config_.HasConstantTvec(image_id)) {
                const std::vector<int>& constant_tvec_idxs =
                        config_.ConstantTvec(image_id);
                // ceres::SubsetParameterization* tvec_parameterization =
                //     new ceres::SubsetParameterization(3, constant_tvec_idxs);
                // problem_->SetParameterization(tvec_data,
                // tvec_parameterization);
                SetSubsetManifold(3, constant_tvec_idxs, problem_.get(),
                                  tvec_data);
            }
        }
    }
}

void BundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
    Point3D& point3D = reconstruction->Point3D(point3D_id);

    // Upstream parity (dbb41680, Caspar wiring): points on tracks shorter
    // than min_track_length are excluded from the problem entirely.
    if (options_.min_track_length > 0 &&
        static_cast<int>(point3D.Track().Length()) <
                options_.min_track_length) {
        return;
    }

    // Is 3D point already fully contained in the problem? I.e. its entire track
    // is contained in `variable_image_ids`, `constant_image_ids`,
    // `constant_x_image_ids`.
    if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
        return;
    }

    for (const auto& track_el : point3D.Track().Elements()) {
        // Skip observations that were already added in `FillImages`.
        if (config_.HasImage(track_el.image_id)) {
            continue;
        }

        point3D_num_observations_[point3D_id] += 1;

        Image& image = reconstruction->Image(track_el.image_id);
        Camera& camera = reconstruction->Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);

        // We do not want to refine the camera of images that are not
        // part of `constant_image_ids_`, `constant_image_ids_`,
        // `constant_x_image_ids_`.
        if (camera_ids_.count(image.CameraId()) == 0) {
            camera_ids_.insert(image.CameraId());
            config_.SetConstantCamera(image.CameraId());
        }

        ceres::CostFunction* cost_function = nullptr;

        if (camera.ModelId() == EquirectangularCameraModel::kModelId) {
            cost_function =
                    EquirectangularBundleAdjustmentConstantPoseCostFunction::Create(
                            image.Qvec(), image.Tvec(), point2D.XY());
        } else {
            switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
    case CameraModel::kModelId:                                                \
        cost_function =                                                        \
                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
                        image.Qvec(), image.Tvec(), point2D.XY());             \
        break;

            CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }
        }
        problem_->AddResidualBlock(cost_function, loss_function,
                                   point3D.XYZ().data(), camera.ParamsData());
    }
}

void BundleAdjuster::ParameterizeCameras(Reconstruction* reconstruction) {
    const bool constant_camera = !options_.refine_focal_length &&
                                 !options_.refine_principal_point &&
                                 !options_.refine_extra_params;
    for (const camera_t camera_id : camera_ids_) {
        Camera& camera = reconstruction->Camera(camera_id);

        if (constant_camera || config_.IsConstantCamera(camera_id)) {
            problem_->SetParameterBlockConstant(camera.ParamsData());
            continue;
        } else {
            std::vector<int> const_camera_params;

            if (!options_.refine_focal_length) {
                const std::vector<size_t>& params_idxs =
                        camera.FocalLengthIdxs();
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(),
                                           params_idxs.end());
            }
            if (!options_.refine_principal_point) {
                const std::vector<size_t>& params_idxs =
                        camera.PrincipalPointIdxs();
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(),
                                           params_idxs.end());
            }
            if (!options_.refine_extra_params) {
                const std::vector<size_t>& params_idxs =
                        camera.ExtraParamsIdxs();
                const_camera_params.insert(const_camera_params.end(),
                                           params_idxs.begin(),
                                           params_idxs.end());
            }

            if (const_camera_params.size() > 0) {
                // ceres::SubsetParameterization* camera_params_parameterization
                // =
                //     new ceres::SubsetParameterization(
                //         static_cast<int>(camera.NumParams()),
                //         const_camera_params);
                // problem_->SetParameterization(camera.ParamsData(),
                //                               camera_params_parameterization);
                SetSubsetManifold(static_cast<int>(camera.NumParams()),
                                  const_camera_params, problem_.get(),
                                  camera.ParamsData());
            }
        }
    }
}

void BundleAdjuster::ParameterizePoints(Reconstruction* reconstruction) {
    for (const auto elem : point3D_num_observations_) {
        Point3D& point3D = reconstruction->Point3D(elem.first);
        if (point3D.Track().Length() > elem.second) {
            problem_->SetParameterBlockConstant(point3D.XYZ().data());
        }
    }

    for (const point3D_t point3D_id : config_.ConstantPoints()) {
        Point3D& point3D = reconstruction->Point3D(point3D_id);
        problem_->SetParameterBlockConstant(point3D.XYZ().data());
    }
}

////////////////////////////////////////////////////////////////////////////////
// RigBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

void PrintSolverSummary(const ceres::Solver::Summary& summary) {
    std::cout << std::right << std::setw(16) << "Residuals : ";
    std::cout << std::left << summary.num_residuals_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Parameters : ";
    std::cout << std::left << summary.num_effective_parameters_reduced
              << std::endl;

    std::cout << std::right << std::setw(16) << "Iterations : ";
    std::cout << std::left
              << summary.num_successful_steps + summary.num_unsuccessful_steps
              << std::endl;

    std::cout << std::right << std::setw(16) << "Time : ";
    std::cout << std::left << summary.total_time_in_seconds << " [s]"
              << std::endl;

    std::cout << std::right << std::setw(16) << "Initial cost : ";
    std::cout << std::right << std::setprecision(6)
              << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
              << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "Final cost : ";
    std::cout << std::right << std::setprecision(6)
              << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
              << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "Termination : ";

    std::string termination = "";

    switch (summary.termination_type) {
        case ceres::CONVERGENCE:
            termination = "Convergence";
            break;
        case ceres::NO_CONVERGENCE:
            termination = "No convergence";
            break;
        case ceres::FAILURE:
            termination = "Failure";
            break;
        case ceres::USER_SUCCESS:
            termination = "User success";
            break;
        case ceres::USER_FAILURE:
            termination = "User failure";
            break;
        default:
            termination = "Unknown";
            break;
    }

    std::cout << std::right << termination << std::endl;
    std::cout << std::endl;
}

}  // namespace colmap
