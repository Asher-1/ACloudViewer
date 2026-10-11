// SPDX-License-Identifier: BSD-3-Clause

#include "estimators/rotation_averaging_ceres.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

#include "estimators/cost_functions/motion_averaging.h"
#include "optim/manifold.h"
#include "scene/pose_graph.h"
#include "scene/reconstruction.h"
#include "util/logging.h"

namespace colmap {
namespace {

std::shared_ptr<ceres::LossFunction> CreateLoss(
        const CeresLossFunctionType type,
        const double scale,
        const double weight = 1.0) {
    const double weighted_scale = scale * weight;
    switch (type) {
        case CeresLossFunctionType::TRIVIAL:
            return std::make_shared<ceres::TrivialLoss>();
        case CeresLossFunctionType::SOFT_L1:
            return std::make_shared<ceres::SoftLOneLoss>(weighted_scale);
        case CeresLossFunctionType::CAUCHY:
            return std::make_shared<ceres::CauchyLoss>(weighted_scale);
        case CeresLossFunctionType::HUBER:
            return std::make_shared<ceres::HuberLoss>(weighted_scale);
    }
    throw std::invalid_argument("unknown Ceres loss function");
}

double* MaybeGetSensorRotationBlock(Image& image) {
    if (image.IsRefInFrame()) return nullptr;
    Frame& frame = *image.FramePtr();
    Rig& rig = *frame.RigPtr();
    return rig.CamFromRigQvec(image.CameraId()).data();
}

struct RelativeRotationWxyzCostFunctor {
    explicit RelativeRotationWxyzCostFunctor(
            const Eigen::Quaterniond& cam2_from_cam1)
        : cam2_from_cam1(cam2_from_cam1) {}

    template <typename T>
    bool operator()(const T* const rig1_from_world,
                    const T* const rig2_from_world,
                    T* residuals) const {
        const Eigen::Quaternion<T> q1(rig1_from_world[0],
                                      rig1_from_world[1],
                                      rig1_from_world[2],
                                      rig1_from_world[3]);
        const Eigen::Quaternion<T> q2(rig2_from_world[0],
                                      rig2_from_world[1],
                                      rig2_from_world[2],
                                      rig2_from_world[3]);
        const Eigen::Quaternion<T> error =
                q2.conjugate() * cam2_from_cam1.cast<T>() * q1;
        const T error_wxyz[4] = {
                error.w(), error.x(), error.y(), error.z()};
        ceres::QuaternionToAngleAxis(error_wxyz, residuals);
        return true;
    }

    static ceres::CostFunction* Create(
            const Eigen::Quaterniond& cam2_from_cam1) {
        return new ceres::AutoDiffCostFunction<RelativeRotationWxyzCostFunctor,
                                               3,
                                               4,
                                               4>(
                new RelativeRotationWxyzCostFunctor(cam2_from_cam1));
    }

    Eigen::Quaterniond cam2_from_cam1;
};

struct RelativeRotationRigWxyzCostFunctor {
    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        Eigen::Quaternion<T> error = cam2_from_cam1.cast<T>();
        if (sensor2_index >= 0) {
            const T* sensor2 = parameters[sensor2_index];
            error = Eigen::Quaternion<T>(sensor2[0], sensor2[1], sensor2[2], sensor2[3])
                            .conjugate() *
                    error;
        }
        if (sensor1_index >= 0) {
            const T* sensor1 = parameters[sensor1_index];
            error = error * Eigen::Quaternion<T>(sensor1[0], sensor1[1], sensor1[2], sensor1[3]);
        }
        if (!same_frame) {
            const T* rig1 = parameters[0];
            const T* rig2 = parameters[1];
            error = Eigen::Quaternion<T>(rig2[0], rig2[1], rig2[2], rig2[3]).conjugate() *
                    error * Eigen::Quaternion<T>(rig1[0], rig1[1], rig1[2], rig1[3]);
        }
        const T error_wxyz[4] = {error.w(), error.x(), error.y(), error.z()};
        ceres::QuaternionToAngleAxis(error_wxyz, residuals);
        return true;
    }

    Eigen::Quaterniond cam2_from_cam1;
    int sensor1_index = -1;
    int sensor2_index = -1;
    bool same_frame = false;
};

}  // namespace

CeresRotationAveragerOptions::CeresRotationAveragerOptions() {
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.logging_type = ceres::LoggingType::SILENT;
}

CeresRotationAverager::CeresRotationAverager(
        const CeresRotationAveragerOptions& options,
        const PoseGraph& pose_graph,
        Reconstruction& reconstruction)
    : solver_options(options.solver_options), reconstruction_(reconstruction) {
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
            max_num_matches > 0 && edge.num_matches == 0) {
            continue;
        }
        const double weight =
                options.reweighting == RotationAveragingReweighting::INLIER_MATCH_COUNT &&
                        max_num_matches > 0
                    ? static_cast<double>(edge.num_matches) / max_num_matches
                    : 1.0;
        const auto [image_id1, image_id2] = Database::PairIdToImagePair(pair_id);
        AddRelativeRotationResidual(
                image_id1,
                image_id2,
                edge.cam2_from_cam1.rotation(),
                CreateLoss(options.loss_function_type, options.loss_function_scale, weight));
    }
}

void CeresRotationAverager::InitializeRotations(
        const CeresRotationAveragerOptions& options,
        const PoseGraph& pose_graph,
        const FlatHashSet<image_t>& image_ids) {
    if (options.skip_initialization) return;

    const image_t root_image_id = *std::min_element(image_ids.begin(), image_ids.end());
    FlatHashSet<image_t> visited = {root_image_id};
    reconstruction_.Frame(reconstruction_.Image(root_image_id).FrameId())
            .SetRigFromWorld(Rigid3d());
    std::queue<image_t> pending;
    pending.push(root_image_id);
    while (!pending.empty()) {
        const image_t parent = pending.front();
        pending.pop();
        const Eigen::Quaterniond parent_rotation =
                reconstruction_.Image(parent).CamFromWorld().rotation();
        for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
            const auto [first, second] = Database::PairIdToImagePair(pair_id);
            const bool parent_is_first = first == parent;
            const image_t child = parent_is_first ? second : first;
            if ((!parent_is_first && second != parent) || !image_ids.count(child) ||
                visited.count(child)) {
                continue;
            }
            const Eigen::Quaterniond child_rotation = parent_is_first
                    ? edge.cam2_from_cam1.rotation() * parent_rotation
                    : edge.cam2_from_cam1.rotation().inverse() * parent_rotation;
            reconstruction_.Frame(reconstruction_.Image(child).FrameId())
                    .SetRigFromWorld(Rigid3d(
                            child_rotation,
                            Eigen::Vector3d::Constant(
                                    std::numeric_limits<double>::quiet_NaN())));
            visited.insert(child);
            pending.push(child);
        }
    }
}

void CeresRotationAverager::SetupParameterBlocks(
        const CeresRotationAveragerOptions& options,
        const FlatHashSet<image_t>& image_ids) {
    for (const image_t image_id : image_ids) {
        Image& image = reconstruction_.Image(image_id);
        Frame& frame = reconstruction_.Frame(image.FrameId());
        if (!frame.HasPose()) {
            frame.SetRigFromWorld(Rigid3d(
                    Eigen::Quaterniond::Identity(),
                    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())));
        }
        double* rotation = frame.RigFromWorldQvec().data();
        if (!problem_->HasParameterBlock(rotation)) {
            problem_->AddParameterBlock(rotation, 4);
            SetQuaternionManifoldWxyz(problem_.get(), rotation);
        }
        if (!image.IsRefInFrame()) {
            Rig& rig = *frame.RigPtr();
            const sensor_t sensor_id(SensorType::CAMERA, image.CameraId());
            const bool known_sensor = rig.HasSensorFromRig(sensor_id);
            if (!known_sensor) {
                rig.SetSensorFromRig(
                        sensor_id,
                        Rigid3d(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d::Constant(
                                        std::numeric_limits<double>::quiet_NaN())));
            }
            double* sensor_rotation = MaybeGetSensorRotationBlock(image);
            if (!problem_->HasParameterBlock(sensor_rotation)) {
                problem_->AddParameterBlock(sensor_rotation, 4);
                SetQuaternionManifoldWxyz(problem_.get(), sensor_rotation);
            }
            if (!options.refine_sensor_from_rig || known_sensor) {
                problem_->SetParameterBlockConstant(sensor_rotation);
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
    if (same_frame && sensor1 == sensor2) return;
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
    auto* cost = new ceres::DynamicAutoDiffCostFunction<RelativeRotationRigWxyzCostFunctor, 8>(
            new RelativeRotationRigWxyzCostFunctor{
                    cam2_from_cam1, sensor_index(sensor1), sensor_index(sensor2), same_frame});
    for (size_t i = 0; i < blocks.size(); ++i) cost->AddParameterBlock(4);
    cost->SetNumResiduals(3);
    problem_->AddResidualBlock(cost, loss, blocks);
}

std::unique_ptr<CeresRotationAverager> CreateDefaultCeresRotationAverager(
        const CeresRotationAveragerOptions& options,
        const PoseGraph& pose_graph,
        Reconstruction& reconstruction) {
    return std::make_unique<CeresRotationAverager>(options, pose_graph, reconstruction);
}

}  // namespace colmap
