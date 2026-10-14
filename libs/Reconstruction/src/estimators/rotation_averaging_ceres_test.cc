// SPDX-License-Identifier: BSD-3-Clause

#include "estimators/rotation_averaging_ceres.h"

#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "estimators/rotation_averaging_l1_irls.h"
#include "scene/database.h"
#include "scene/frame.h"
#include "scene/image.h"
#include "scene/pose_graph.h"
#include "scene/reconstruction.h"
#include "scene/synthetic.h"

// Port of upstream rotation_averaging_ceres_test.cc (b1d64b31 + 9e653da3).
// Fork adaptation: rig and frame poses are value-returning accessors over
// qvec/tvec storage, so in-place upstream mutations go through
// SetSensorFromRig/SetRigFromWorld and parameter blocks are the qvec storage.
namespace colmap {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Eigen::Quaterniond ZRotation(const double angle) {
    return Eigen::Quaterniond(
            Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
}

PoseGraph::Edge Edge(const Eigen::Quaterniond& rotation,
                     const int num_matches = 10) {
    PoseGraph::Edge edge;
    edge.cam2_from_cam1 = Rigid3d(rotation, Eigen::Vector3d::Zero());
    edge.num_matches = num_matches;
    return edge;
}

Reconstruction MakeTrivialReconstruction(const std::vector<image_t>& ids,
                                         const bool posed = true) {
    Reconstruction reconstruction;
    Camera camera = Camera::CreateFromModelId(
            1, CameraModelId::kSimplePinhole, 10, 10, 5);
    reconstruction.AddCameraWithTrivialRig(camera);
    for (const image_t image_id : ids) {
        Image image;
        image.SetImageId(image_id);
        image.SetCameraId(camera.CameraId());
        reconstruction.AddImageWithTrivialFrame(image);
        if (posed) {
            reconstruction.Frame(reconstruction.Image(image_id).FrameId())
                    .SetRigFromWorld(Rigid3d());
        }
    }
    return reconstruction;
}

void SetSensorRotation(Rig& rig,
                       const sensor_t& sensor_id,
                       const Eigen::Quaterniond& rotation) {
    rig.SetSensorFromRig(
            sensor_id,
            Rigid3d(rotation, rig.SensorFromRig(sensor_id).translation()));
}

void SetSensorTranslationX(Rig& rig, const sensor_t& sensor_id, double x) {
    Rigid3d pose = rig.SensorFromRig(sensor_id);
    pose.translation()[0] = x;
    rig.SetSensorFromRig(sensor_id, pose);
}

Reconstruction MakeRigReconstruction(const int num_rigs = 1,
                                     const int num_cameras = 2) {
    Reconstruction reconstruction;
    SyntheticDatasetOptions options;
    options.num_rigs = num_rigs;
    options.num_cameras_per_rig = num_cameras;
    options.num_frames_per_rig = 3;
    options.num_points3D = 5;
    SynthesizeDataset(options, &reconstruction);
    for (const auto& [id, frame] : reconstruction.Frames()) {
        reconstruction.Frame(id).SetRigFromWorld(Rigid3d(
                Eigen::Quaterniond(
                        Eigen::AngleAxisd(0.12 * id, Eigen::Vector3d::UnitX())) *
                        ZRotation(0.17 * id),
                frame.RigFromWorld().translation()));
    }
    for (const auto& [id, rig] : reconstruction.Rigs()) {
        for (const auto& [sensor_id, pose] : rig.NonRefSensors()) {
            SetSensorRotation(
                    reconstruction.Rig(id),
                    sensor_id,
                    Eigen::Quaterniond(Eigen::AngleAxisd(
                            0.1 * sensor_id.id, Eigen::Vector3d::UnitY())) *
                            ZRotation(0.07 * sensor_id.id));
        }
    }
    return reconstruction;
}

std::vector<std::vector<image_t>> RigImages(
        const Reconstruction& reconstruction) {
    std::map<frame_t, std::map<camera_t, image_t>> images;
    for (const auto& [id, image] : reconstruction.Images()) {
        images[image.FrameId()][image.CameraId()] = id;
    }
    std::vector<std::vector<image_t>> result;
    for (const auto& [_, frame] : images) {
        result.emplace_back();
        for (const auto& [camera, image] : frame) result.back().push_back(image);
    }
    return result;
}

Eigen::Quaterniond RelativeRotation(const Reconstruction& reconstruction,
                                    const image_t image1,
                                    const image_t image2) {
    return reconstruction.Image(image2).CamFromWorld().rotation() *
           reconstruction.Image(image1).CamFromWorld().rotation().inverse();
}

PoseGraph MakePoseGraph(const Reconstruction& reconstruction) {
    PoseGraph graph;
    for (const auto& [id1, image1] : reconstruction.Images()) {
        for (const auto& [id2, image2] : reconstruction.Images()) {
            if (id1 < id2) {
                graph.AddEdge(
                        id1, id2, Edge(RelativeRotation(reconstruction, id1, id2)));
            }
        }
    }
    return graph;
}

void ExpectRelativeRotations(const Reconstruction& reconstruction,
                             const PoseGraph& graph,
                             const double tolerance) {
    for (const auto& [pair, edge] : graph.ValidEdges()) {
        const auto [id1, id2] = Database::PairIdToImagePair(pair);
        EXPECT_NEAR(RelativeRotation(reconstruction, id1, id2)
                            .angularDistance(edge.cam2_from_cam1.rotation()),
                    0,
                    tolerance);
    }
}

double* SensorRotationBlock(Reconstruction& reconstruction,
                            const image_t image_id) {
    const Image& image = reconstruction.Image(image_id);
    return image.FramePtr()->RigPtr()->CamFromRigQvec(image.CameraId()).data();
}

TEST(CeresRotationAverager, RecoversNominalRotations) {
    Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3});
    PoseGraph pose_graph;
    pose_graph.AddEdge(1, 2, Edge(ZRotation(0.2)));
    pose_graph.AddEdge(2, 3, Edge(ZRotation(-0.1)));
    RotationEstimatorOptions options;

    auto averager = CreateDefaultCeresRotationAverager(
            options, pose_graph, reconstruction);
    EXPECT_EQ(averager->Problem().NumParameterBlocks(), 3);
    EXPECT_EQ(averager->Problem().NumResidualBlocks(), 2);
    EXPECT_EQ(averager->solver_options.linear_solver_type,
              ceres::SPARSE_NORMAL_CHOLESKY);
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    ExpectRelativeRotations(reconstruction, pose_graph, 1e-10);
}

TEST(CeresRotationAverager, MatchCountReweighting) {
    Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3});
    RotationEstimatorOptions options;
    options.skip_initialization = true;
    options.ceres->loss_function_type = CeresLossFunctionType::HUBER;
    options.ceres->loss_function_scale = RadToDeg(0.1);
    for (const int num_matches : {10, 0}) {
        PoseGraph graph;
        graph.AddEdge(1, 2, Edge(ZRotation(0.2), num_matches));
        graph.AddEdge(2, 3, Edge(ZRotation(0.2), num_matches / 2));
        graph.AddEdge(1, 3, Edge(ZRotation(0.2), 0));
        for (const auto reweighting :
             {RotationAveragingReweighting::UNIFORM,
              RotationAveragingReweighting::INLIER_MATCH_COUNT}) {
            options.reweighting = reweighting;
            auto averager = CreateDefaultCeresRotationAverager(
                    options, graph, reconstruction);
            double cost;
            ASSERT_TRUE(averager->Problem().Evaluate(
                    ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr,
                    nullptr));
            // Each Huber cost is 0.015; weights are either (1, 1, 1) or
            // (1, 0.5, 0). The weight scales the loss output (ScaledLoss), not
            // the robust scale.
            const bool uniform =
                    reweighting == RotationAveragingReweighting::UNIFORM ||
                    num_matches == 0;
            EXPECT_NEAR(cost, uniform ? 0.045 : 0.0225, 1e-12);
        }
    }
}

TEST(CeresRotationAverager, AddsIndividualRelativeRotationResidual) {
    Reconstruction reconstruction = MakeTrivialReconstruction({1, 2});
    PoseGraph pose_graph;
    pose_graph.AddEdge(1, 2, Edge(ZRotation(0.2)));
    auto averager = CreateDefaultCeresRotationAverager(
            RotationEstimatorOptions(), pose_graph, reconstruction);

    auto loss = std::make_shared<ceres::CauchyLoss>(0.05);
    averager->AddRelativeRotationResidual(1, 1, ZRotation(0.2), loss);
    averager->AddRelativeRotationResidual(1, 2, ZRotation(0.2), loss);
    EXPECT_EQ(averager->Problem().NumResidualBlocks(), 2);
    std::weak_ptr<ceres::LossFunction> retained_loss = loss;
    loss.reset();
    EXPECT_FALSE(retained_loss.expired());
    EXPECT_TRUE(averager->Solve().IsSolutionUsable());
}

TEST(CeresRotationAverager, RejectsUnconfiguredRotationBlocks) {
    Reconstruction reconstruction = MakeRigReconstruction();
    const auto images = RigImages(reconstruction);
    PoseGraph graph;
    graph.AddEdge(images[0][0], images[1][0], Edge(ZRotation(0.2)));
    RotationEstimatorOptions options;
    options.skip_initialization = true;
    auto averager =
            CreateDefaultCeresRotationAverager(options, graph, reconstruction);

    for (const auto& [image1, image2] :
         {std::pair{images[2][0], images[0][0]},
          std::pair{images[0][0], images[2][0]},
          std::pair{images[0][1], images[1][0]},
          std::pair{images[0][0], images[1][1]}}) {
        EXPECT_THROW(averager->AddRelativeRotationResidual(
                             image1, image2, ZRotation(0.2), nullptr),
                     std::invalid_argument);
        EXPECT_EQ(averager->Problem().NumParameterBlocks(), 2);
        EXPECT_EQ(averager->Problem().NumResidualBlocks(), 1);
    }
}

TEST(CeresRotationAverager, SelectsMstOrSuppliedInitialization) {
    Reconstruction supplied = MakeTrivialReconstruction({1, 2});
    supplied.Frame(1).SetRigFromWorld(Rigid3d(ZRotation(0.7), Eigen::Vector3d::Zero()));
    supplied.Frame(2).SetRigFromWorld(Rigid3d(ZRotation(0.9), Eigen::Vector3d::Zero()));
    Reconstruction initialized = MakeTrivialReconstruction({1, 2}, false);
    PoseGraph pose_graph;
    pose_graph.AddEdge(1, 2, Edge(ZRotation(0.2)));

    auto mst = CreateDefaultCeresRotationAverager(
            RotationEstimatorOptions(), pose_graph, initialized);
    EXPECT_NEAR((initialized.Frame(2).RigFromWorld().rotation() *
                 initialized.Frame(1).RigFromWorld().rotation().inverse())
                        .angularDistance(ZRotation(0.2)),
                0.0,
                1e-12);
    EXPECT_TRUE(initialized.Frame(1)
                        .RigFromWorld()
                        .translation()
                        .array()
                        .isNaN()
                        .all());

    RotationEstimatorOptions options;
    options.skip_initialization = true;
    auto preserved =
            CreateDefaultCeresRotationAverager(options, pose_graph, supplied);
    EXPECT_NEAR(supplied.Frame(1).RigFromWorld().rotation().angularDistance(
                        ZRotation(0.7)),
                0.0,
                1e-15);
    EXPECT_NEAR(supplied.Frame(2).RigFromWorld().rotation().angularDistance(
                        ZRotation(0.9)),
                0.0,
                1e-15);
}

TEST(CeresRotationAverager, RejectsDisconnectedOrUnknownGraph) {
    Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3, 4});
    PoseGraph disconnected;
    disconnected.AddEdge(1, 2, Edge(ZRotation(0.1)));
    disconnected.AddEdge(3, 4, Edge(ZRotation(0.2)));
    EXPECT_THROW(CreateDefaultCeresRotationAverager(RotationEstimatorOptions(),
                                                    disconnected,
                                                    reconstruction),
                 std::invalid_argument);

    PoseGraph unknown;
    unknown.AddEdge(1, 5, Edge(ZRotation(0.1)));
    EXPECT_THROW(CreateDefaultCeresRotationAverager(
                         RotationEstimatorOptions(), unknown, reconstruction),
                 std::invalid_argument);
}

TEST(CeresRotationAverager, CalibratedRigs) {
    const Reconstruction truth = MakeRigReconstruction(2);
    const PoseGraph graph = MakePoseGraph(truth);
    for (const bool skip_initialization : {false, true}) {
        Reconstruction reconstruction = truth;
        const auto images = RigImages(reconstruction);
        for (const auto& [id, rig] : truth.Rigs()) {
            for (const auto& [sensor, pose] : rig.NonRefSensors()) {
                SetSensorTranslationX(reconstruction.Rig(id), sensor, kNaN);
            }
        }
        RotationEstimatorOptions options;
        options.skip_initialization = skip_initialization;
        options.refine_sensor_from_rig = false;
        auto averager = CreateDefaultCeresRotationAverager(
                options, graph, reconstruction);
        EXPECT_EQ(averager->Problem().NumParameterBlocks(),
                  truth.NumFrames() + truth.NumRigs());
        EXPECT_EQ(averager->Problem().NumResidualBlocks(), graph.NumEdges());
        averager->AddRelativeRotationResidual(
                images[1][1],
                images[0][0],
                RelativeRotation(truth, images[1][1], images[0][0]),
                nullptr);
        EXPECT_EQ(averager->Problem().NumResidualBlocks(),
                  graph.NumEdges() + 1);
        ASSERT_TRUE(averager->Solve().IsSolutionUsable());
        ExpectRelativeRotations(reconstruction, graph, 1e-8);
        for (const auto& [id, rig] : truth.Rigs()) {
            for (const auto& [sensor, pose] : rig.NonRefSensors()) {
                Rig& estimated_rig = reconstruction.Rig(id);
                EXPECT_TRUE(averager->Problem().IsParameterBlockConstant(
                        estimated_rig.CamFromRigQvec(sensor.id).data()));
                EXPECT_EQ(estimated_rig.SensorFromRig(sensor).rotation().coeffs(),
                          pose->rotation().coeffs());
            }
        }
    }
}

TEST(CeresRotationAverager, CalibratedRigWithDisconnectedImageGraph) {
    Reconstruction reconstruction = MakeRigReconstruction();
    const auto images = RigImages(reconstruction);
    PoseGraph graph;
    graph.AddEdge(images[0][0],
                  images[1][0],
                  Edge(RelativeRotation(reconstruction, images[0][0], images[1][0])));
    graph.AddEdge(images[1][1],
                  images[2][1],
                  Edge(RelativeRotation(reconstruction, images[1][1], images[2][1])));
    for (const frame_t id : std::vector<frame_t>(reconstruction.RegFrameIds())) {
        reconstruction.DeRegisterFrame(id);
    }
    auto averager = CreateDefaultCeresRotationAverager(
            RotationEstimatorOptions(), graph, reconstruction);
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    ExpectRelativeRotations(reconstruction, graph, 1e-7);
}

TEST(CeresRotationAverager, EstimatesUncalibratedRigs) {
    const Reconstruction truth = MakeRigReconstruction(2);
    const PoseGraph graph = MakePoseGraph(truth);
    // Missing calibration, rotation-only calibration, supplied warm start,
    // NaN rotation, and NaN warm start.
    for (const int mode : {0, 1, 2, 3, 4}) {
        Reconstruction reconstruction = truth;
        for (const auto& [id, rig] : truth.Rigs()) {
            for (const auto& [sensor, pose] : rig.NonRefSensors()) {
                Rig& initial_rig = reconstruction.Rig(id);
                if (mode == 0) {
                    initial_rig.ResetSensorFromRig(sensor);
                    continue;
                }
                Rigid3d initial = initial_rig.SensorFromRig(sensor);
                if (mode >= 3) {
                    initial.rotation().coeffs().setConstant(kNaN);
                } else {
                    initial.rotation() = ZRotation(0.15) * initial.rotation();
                }
                initial.translation()[0] = kNaN;
                initial_rig.SetSensorFromRig(sensor, initial);
            }
        }
        RotationEstimatorOptions options;
        options.skip_initialization = mode == 2 || mode == 4;
        options.ceres->solver_options.function_tolerance = 1e-12;
        options.ceres->solver_options.gradient_tolerance = 1e-12;
        options.ceres->solver_options.parameter_tolerance = 1e-12;
        auto averager = CreateDefaultCeresRotationAverager(
                options, graph, reconstruction);
        EXPECT_EQ(averager->Problem().NumParameterBlocks(),
                  truth.NumFrames() + truth.NumRigs());
        ASSERT_TRUE(averager->Solve().IsSolutionUsable());
        for (const auto& [id, rig] : truth.Rigs()) {
            for (const auto& [sensor, pose] : rig.NonRefSensors()) {
                const Rigid3d estimated =
                        reconstruction.Rig(id).SensorFromRig(sensor);
                EXPECT_NEAR(estimated.rotation().angularDistance(
                                    pose->rotation()),
                            0,
                            1e-7)
                        << "mode=" << mode;
                if (mode == 0) {
                    EXPECT_TRUE(estimated.translation().array().isNaN().all());
                }
            }
        }
        ExpectRelativeRotations(reconstruction, graph, 1e-7);
    }
}

TEST(CeresRotationAverager, RefinesCalibrationThroughPreparedProblem) {
    const Reconstruction truth = MakeRigReconstruction();
    Reconstruction reconstruction = truth;
    const auto images = RigImages(truth);
    PoseGraph graph;
    graph.AddEdge(images[0][0],
                  images[0][1],
                  Edge(RelativeRotation(truth, images[0][0], images[0][1])));
    const Image& image = reconstruction.Image(images[0][1]);
    Rig& rig = *image.FramePtr()->RigPtr();
    const sensor_t sensor_id = image.CameraPtr()->SensorId();
    SetSensorRotation(rig, sensor_id,
                      ZRotation(0.2) * rig.SensorFromRig(sensor_id).rotation());
    auto averager = CreateDefaultCeresRotationAverager(
            RotationEstimatorOptions(), graph, reconstruction);
    double* rotation = SensorRotationBlock(reconstruction, images[0][1]);
    EXPECT_TRUE(averager->Problem().IsParameterBlockConstant(rotation));
    averager->Problem().SetParameterBlockVariable(rotation);
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    EXPECT_NEAR(RelativeRotation(reconstruction, images[0][0], images[0][1])
                        .angularDistance(RelativeRotation(
                                truth, images[0][0], images[0][1])),
                0,
                1e-8);
}

TEST(CeresRotationAverager, MissingSensorRotation) {
    Reconstruction reconstruction = MakeRigReconstruction();
    const auto images = RigImages(reconstruction);
    PoseGraph graph;
    // The sensor has no reference image from the same frame in the graph.
    for (const auto& [id1, id2] : {std::pair{images[0][0], images[1][0]},
                                   std::pair{images[1][0], images[2][1]}}) {
        graph.AddEdge(id1, id2, Edge(RelativeRotation(reconstruction, id1, id2)));
    }
    const Image& image = reconstruction.Image(images[0][1]);
    Rig& rig = *image.FramePtr()->RigPtr();
    const sensor_t sensor_id = image.CameraPtr()->SensorId();
    rig.ResetSensorFromRig(sensor_id);
    reconstruction.DeRegisterFrame(reconstruction.Image(images[2][1]).FrameId());
    const Reconstruction before = reconstruction;
    RotationEstimatorOptions options;
    options.skip_initialization = true;
    options.refine_sensor_from_rig = false;
    EXPECT_THROW(
            CreateDefaultCeresRotationAverager(options, graph, reconstruction),
            std::invalid_argument);
    EXPECT_EQ(reconstruction.Frames(), before.Frames());
    EXPECT_EQ(reconstruction.Rigs(), before.Rigs());
    EXPECT_EQ(reconstruction.RegFrameIds(), before.RegFrameIds());

    rig.SetSensorFromRig(
            sensor_id,
            Rigid3d(Eigen::Quaterniond(Eigen::Vector4d::Constant(kNaN)),
                    Eigen::Vector3d::Zero()));
    for (const bool refine : {false, true}) {
        options.refine_sensor_from_rig = refine;
        EXPECT_THROW(
                CreateDefaultCeresRotationAverager(options, graph, reconstruction),
                std::invalid_argument);
        EXPECT_EQ(reconstruction.Frames(), before.Frames());
        EXPECT_EQ(reconstruction.RegFrameIds(), before.RegFrameIds());
    }
    rig.ResetSensorFromRig(sensor_id);
    options.refine_sensor_from_rig = true;
    options.skip_initialization = false;
    auto averager =
            CreateDefaultCeresRotationAverager(options, graph, reconstruction);
    EXPECT_EQ(rig.SensorFromRig(sensor_id).rotation().coeffs(),
              Eigen::Quaterniond::Identity().coeffs());
    EXPECT_FALSE(averager->Problem().IsParameterBlockConstant(
            SensorRotationBlock(reconstruction, images[0][1])));
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    ExpectRelativeRotations(reconstruction, graph, 1e-7);

    averager.reset();
    SetSensorRotation(rig, sensor_id,
                      Eigen::Quaterniond(Eigen::Vector4d::Constant(kNaN)));
    averager = CreateDefaultCeresRotationAverager(options, graph, reconstruction);
    EXPECT_EQ(rig.SensorFromRig(sensor_id).rotation().coeffs(),
              Eigen::Quaterniond::Identity().coeffs());
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    ExpectRelativeRotations(reconstruction, graph, 1e-7);
}

TEST(RotationEstimatorOptions, DeepCopiesCeresOptions) {
    RotationEstimatorOptions options;
    ASSERT_NE(options.ceres, nullptr);
    EXPECT_EQ(options.ceres->loss_function_type, CeresLossFunctionType::CAUCHY);
    options.ceres->loss_function_scale = 0.1;

    RotationEstimatorOptions copied = options;
    ASSERT_NE(copied.ceres, nullptr);
    EXPECT_NE(copied.ceres.get(), options.ceres.get());
    EXPECT_EQ(copied.ceres->loss_function_scale, 0.1);
    copied.ceres->loss_function_scale = 0.2;
    EXPECT_EQ(options.ceres->loss_function_scale, 0.1);

    RotationEstimatorOptions assigned;
    assigned = options;
    ASSERT_NE(assigned.ceres, nullptr);
    EXPECT_NE(assigned.ceres.get(), options.ceres.get());
    EXPECT_EQ(assigned.ceres->loss_function_scale, 0.1);
}

// Fork regression: OptionManager registers raw pointers into the nested
// backend options, so copy-assignment (ResetOptions) must keep the pointees.
TEST(RotationEstimatorOptions, AssignmentPreservesNestedOptionAddresses) {
    RotationEstimatorOptions options;
    const CeresRotationAveragerOptions* ceres = options.ceres.get();
    const L1IrlsRotationAveragerOptions* l1_irls = options.l1_irls.get();
    options.ceres->max_num_warm_start_iterations = 0;
    options.l1_irls->max_num_l1_iterations = 1;

    options = RotationEstimatorOptions();
    EXPECT_EQ(options.ceres.get(), ceres);
    EXPECT_EQ(options.l1_irls.get(), l1_irls);
    EXPECT_EQ(options.ceres->max_num_warm_start_iterations, 5);
    EXPECT_EQ(options.l1_irls->max_num_l1_iterations, 5);
}

}  // namespace
}  // namespace colmap
