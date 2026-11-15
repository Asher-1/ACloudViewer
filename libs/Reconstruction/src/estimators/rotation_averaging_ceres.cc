// SPDX-License-Identifier: BSD-3-Clause

#include "estimators/rotation_averaging_ceres.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "optim/manifold.h"
#include "scene/pose_graph.h"
#include "scene/reconstruction.h"
#include "util/logging.h"
#include "util/threading.h"

namespace colmap {
namespace {

// Upstream CreateCeresLossFunction (estimators/ceres_loss_function.cc)
// semantics: `robust_scale` sets the residual at which robustification takes
// place and `weight` multiplies the loss output through ceres::ScaledLoss.
std::shared_ptr<ceres::LossFunction> CreateLoss(
        const CeresLossFunctionType type,
        const double robust_scale,
        const double weight = 1.0) {
    if (std::isnan(robust_scale) || robust_scale < 0.0 ||
        !std::isfinite(weight) || weight <= 0.0) {
        throw std::invalid_argument("invalid Ceres loss configuration");
    }
    std::unique_ptr<ceres::LossFunction> loss;
    switch (type) {
        case CeresLossFunctionType::TRIVIAL:
            loss = std::make_unique<ceres::TrivialLoss>();
            break;
        case CeresLossFunctionType::SOFT_L1:
            loss = std::make_unique<ceres::SoftLOneLoss>(robust_scale);
            break;
        case CeresLossFunctionType::CAUCHY:
            loss = std::make_unique<ceres::CauchyLoss>(robust_scale);
            break;
        case CeresLossFunctionType::HUBER:
            loss = std::make_unique<ceres::HuberLoss>(robust_scale);
            break;
    }
    if (!loss) throw std::invalid_argument("unknown Ceres loss function");
    if (weight == 1.0) return loss;
    return std::make_shared<ceres::ScaledLoss>(
            loss.release(), weight, ceres::TAKE_OWNERSHIP);
}

double* MaybeGetSensorRotationBlock(const Image& image) {
    if (image.IsRefInFrame()) return nullptr;
    return image.FramePtr()->RigPtr()->CamFromRigQvec(image.CameraId()).data();
}

template <typename T>
Eigen::Quaternion<T> WxyzQuaternion(const T* q) {
    return Eigen::Quaternion<T>(q[0], q[1], q[2], q[3]);
}

// Storage adapter of RelativeRotationCostFunctor
// (cost_functions/motion_averaging.h): this fork stores frame and sensor
// rotations as [w, x, y, z] qvecs, so the blocks are read in WXYZ order. The
// residual is the same sensor-1-frame expression
//   Log(prior^-1 * sensor2_from_world * sensor1_from_world^-1).
struct RelativeRotationWxyzCostFunctor {
    template <typename T>
    bool operator()(const T* const rig1_from_world,
                    const T* const rig2_from_world,
                    T* residuals) const {
        const T* parameters[] = {rig1_from_world, rig2_from_world};
        return (*this)(parameters, residuals);
    }

    // For rigs, parameters[0] and [1] are rig1_from_world and rig2_from_world.
    // sensor1_index and sensor2_index locate the sensor_from_rig blocks;
    // -1 denotes identity, and shared sensors may use the same block.
    // With same_frame=true, omit the rig_from_world blocks and index only
    // the sensor_from_rig blocks.
    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        Eigen::Quaternion<T> error =
                same_frame ? Eigen::Quaternion<T>::Identity()
                           : WxyzQuaternion(parameters[1]) *
                                     WxyzQuaternion(parameters[0]).conjugate();
        if (sensor2_index >= 0) {
            error = WxyzQuaternion(parameters[sensor2_index]) * error;
        }
        if (sensor1_index >= 0) {
            error = error * WxyzQuaternion(parameters[sensor1_index]).conjugate();
        }
        error = sensor2_from_sensor1_prior.conjugate().cast<T>() * error;
        const T error_wxyz[4] = {error.w(), error.x(), error.y(), error.z()};
        ceres::QuaternionToAngleAxis(error_wxyz, residuals);
        return true;
    }

    static ceres::CostFunction* Create(
            const Eigen::Quaterniond& cam2_from_cam1) {
        return new ceres::AutoDiffCostFunction<RelativeRotationWxyzCostFunctor,
                                               3, 4, 4>(
                new RelativeRotationWxyzCostFunctor{cam2_from_cam1});
    }

    Eigen::Quaterniond sensor2_from_sensor1_prior;
    int sensor1_index = -1;
    int sensor2_index = -1;
    bool same_frame = false;
};

const Rigid3d kUnknownTranslationIdentity(
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()));

}  // namespace

CeresRotationAveragerOptions::CeresRotationAveragerOptions() {
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.max_num_iterations = 100;
    solver_options.num_threads = -1;
}

CeresRotationAverager::CeresRotationAverager(
        const RotationEstimatorOptions& options,
        const PoseGraph& pose_graph,
        Reconstruction& reconstruction)
    : solver_options(THROW_CHECK_NOTNULL(options.ceres)->solver_options),
      reconstruction_(reconstruction) {
    const CeresRotationAveragerOptions& ceres_options = *options.ceres;
    solver_options.num_threads =
            GetEffectiveNumThreads(solver_options.num_threads);
    if (VLOG_IS_ON(2)) {
        solver_options.minimizer_progress_to_stdout = true;
    }
    std::shared_ptr<ceres::LossFunction> loss =
            CreateLoss(ceres_options.loss_function_type,
                       DegToRad(ceres_options.loss_function_scale));
    FlatHashSet<image_t> image_ids;
    int max_num_matches = 0;
    for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
        const auto [image_id1, image_id2] = Database::PairIdToImagePair(pair_id);
        if (!reconstruction_.ExistsImage(image_id1) ||
            !reconstruction_.ExistsImage(image_id2)) {
            throw std::invalid_argument("rotation edge references unknown image");
        }
        image_ids.insert(image_id1);
        image_ids.insert(image_id2);
        max_num_matches = std::max(max_num_matches, edge.num_matches);
    }
    if (image_ids.empty()) {
        throw std::invalid_argument("Ceres rotation averaging requires edges");
    }
    if (pose_graph.ConnectedFrameComponents(reconstruction_, false).size() != 1) {
        throw std::invalid_argument(
                "Ceres rotation averaging requires a connected pose graph");
    }

    InitializeRotations(options, pose_graph, image_ids);
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_ = std::make_unique<ceres::Problem>(problem_options);
    SetupParameterBlocks(options, image_ids);

    for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
        if (options.reweighting == RotationAveragingReweighting::INLIER_MATCH_COUNT &&
            max_num_matches > 0) {
            if (edge.num_matches == 0) continue;
            loss = CreateLoss(ceres_options.loss_function_type,
                              DegToRad(ceres_options.loss_function_scale),
                              static_cast<double>(edge.num_matches) /
                                      max_num_matches);
        }
        const auto [image_id1, image_id2] = Database::PairIdToImagePair(pair_id);
        AddRelativeRotationResidual(
                image_id1, image_id2, edge.cam2_from_cam1.rotation(), loss);
    }
}

void CeresRotationAverager::InitializeRotations(
        const RotationEstimatorOptions& options,
        const PoseGraph& pose_graph,
        const FlatHashSet<image_t>& image_ids) {
    // Validate fixed sensors before replacing invalid initial rotations, so a
    // rejected configuration leaves the reconstruction untouched.
    for (const image_t image_id : image_ids) {
        const Image& image = reconstruction_.Image(image_id);
        if (image.IsRefInFrame()) continue;
        const std::optional<Rigid3d> pose =
                image.FramePtr()->RigPtr()->MaybeSensorFromRig(
                        image.CameraPtr()->SensorId());
        const bool refine_sensor =
                options.refine_sensor_from_rig &&
                (!pose.has_value() || pose->translation().hasNaN());
        if (!refine_sensor &&
            (!pose.has_value() || !pose->rotation().coeffs().allFinite())) {
            throw std::invalid_argument(
                    "rotation averaging requires finite sensor rotations");
        }
    }
    for (const image_t image_id : image_ids) {
        const Image& image = reconstruction_.Image(image_id);
        if (image.IsRefInFrame()) continue;
        Rig& rig = *image.FramePtr()->RigPtr();
        const sensor_t sensor_id = image.CameraPtr()->SensorId();
        if (rig.HasSensorFromRig(sensor_id) &&
            !rig.SensorFromRigQvec(sensor_id).allFinite()) {
            rig.ResetSensorFromRig(sensor_id);
        }
    }
    if (!options.skip_initialization) {
        InitializeFromMaximumSpanningTree(pose_graph,
                                          image_ids,
                                          reconstruction_,
                                          options.refine_sensor_from_rig);
    }
}

void CeresRotationAverager::SetupParameterBlocks(
        const RotationEstimatorOptions& options,
        const FlatHashSet<image_t>& image_ids) {
    const auto add_rotation = [&](double* block) {
        if (!problem_->HasParameterBlock(block)) {
            problem_->AddParameterBlock(block, 4);
            SetQuaternionManifoldWxyz(problem_.get(), block);
        }
        return block;
    };
    for (const image_t image_id : image_ids) {
        const Image& image = reconstruction_.Image(image_id);
        Frame& frame = *image.FramePtr();
        if (!frame.HasPose()) {
            frame.SetRigFromWorld(kUnknownTranslationIdentity);
        }
        add_rotation(frame.RigFromWorldQvec().data());
        if (!image.IsRefInFrame()) {
            Rig& rig = *frame.RigPtr();
            const sensor_t sensor_id = image.CameraPtr()->SensorId();
            if (!rig.HasSensorFromRig(sensor_id)) {
                rig.SetSensorFromRig(sensor_id, kUnknownTranslationIdentity);
            }
            double* sensor = add_rotation(MaybeGetSensorRotationBlock(image));
            // Fully calibrated sensors (finite translation) stay constant
            // unless the caller flips them via Problem().
            if (!options.refine_sensor_from_rig ||
                !rig.SensorFromRigTvec(sensor_id).hasNaN()) {
                problem_->SetParameterBlockConstant(sensor);
            }
        }
    }
    const image_t root_image_id = *std::min_element(image_ids.begin(), image_ids.end());
    problem_->SetParameterBlockConstant(
            reconstruction_.Frame(reconstruction_.Image(root_image_id).FrameId())
                    .RigFromWorldQvec()
                    .data());
}

ceres::Solver::Summary CeresRotationAverager::Solve() {
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, problem_.get(), &summary);
    if (summary.IsSolutionUsable()) {
        for (const auto& [frame_id, frame] : reconstruction_.Frames()) {
            if (frame.HasPose() &&
                problem_->HasParameterBlock(
                        reconstruction_.Frame(frame_id).RigFromWorldQvec().data())) {
                reconstruction_.RegisterFrame(frame_id);
            }
        }
    }
    return summary;
}

ceres::Problem& CeresRotationAverager::Problem() { return *problem_; }

const ceres::Problem& CeresRotationAverager::Problem() const { return *problem_; }

void CeresRotationAverager::AddRelativeRotationResidual(
        const image_t image_id1,
        const image_t image_id2,
        const Eigen::Quaterniond& cam2_from_cam1,
        const std::shared_ptr<ceres::LossFunction>& loss_function) {
    if (!reconstruction_.ExistsImage(image_id1) ||
        !reconstruction_.ExistsImage(image_id2)) {
        throw std::invalid_argument("rotation edge references unknown image");
    }
    Image& image1 = reconstruction_.Image(image_id1);
    Image& image2 = reconstruction_.Image(image_id2);
    Frame& frame1 = reconstruction_.Frame(image1.FrameId());
    Frame& frame2 = reconstruction_.Frame(image2.FrameId());
    double* rotation1 = frame1.RigFromWorldQvec().data();
    double* rotation2 = frame2.RigFromWorldQvec().data();
    if (!problem_->HasParameterBlock(rotation1) ||
        !problem_->HasParameterBlock(rotation2)) {
        throw std::invalid_argument("rotation blocks have not been configured");
    }
    double* sensor1 = MaybeGetSensorRotationBlock(image1);
    double* sensor2 = MaybeGetSensorRotationBlock(image2);
    if ((sensor1 != nullptr && !problem_->HasParameterBlock(sensor1)) ||
        (sensor2 != nullptr && !problem_->HasParameterBlock(sensor2))) {
        throw std::invalid_argument("sensor rotation blocks have not been configured");
    }
    const bool same_frame = image1.FrameId() == image2.FrameId();
    if (same_frame && sensor1 == sensor2) {
        LOG(WARNING) << "Skipping self-loop for image pair " << image_id1
                     << ", " << image_id2;
        return;
    }
    ceres::LossFunction* loss = loss_function.get();
    if (loss != nullptr) losses_.try_emplace(loss, loss_function);
    if (sensor1 == nullptr && sensor2 == nullptr && !same_frame) {
        problem_->AddResidualBlock(RelativeRotationWxyzCostFunctor::Create(cam2_from_cam1),
                                   loss,
                                   rotation1,
                                   rotation2);
        return;
    }
    std::vector<double*> blocks;
    if (!same_frame) blocks = {rotation1, rotation2};
    const auto sensor_index = [&blocks](double* sensor) {
        if (sensor == nullptr) return -1;
        const auto it = std::find(blocks.begin(), blocks.end(), sensor);
        if (it != blocks.end()) return static_cast<int>(it - blocks.begin());
        blocks.push_back(sensor);
        return static_cast<int>(blocks.size() - 1);
    };
    auto* cost = new ceres::DynamicAutoDiffCostFunction<RelativeRotationWxyzCostFunctor, 8>(
            new RelativeRotationWxyzCostFunctor{
                    cam2_from_cam1, sensor_index(sensor1), sensor_index(sensor2), same_frame});
    for (size_t i = 0; i < blocks.size(); ++i) cost->AddParameterBlock(4);
    cost->SetNumResiduals(3);
    problem_->AddResidualBlock(cost, loss, blocks);
}

std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
        const RotationEstimatorOptions& options,
        const PoseGraph& pose_graph,
        Reconstruction& reconstruction) {
    return std::make_unique<CeresRotationAverager>(options, pose_graph, reconstruction);
}

}  // namespace colmap
