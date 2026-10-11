// SPDX-License-Identifier: BSD-3-Clause

#include "estimators/rotation_averaging_ceres.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "scene/image.h"
#include "scene/pose_graph.h"
#include "scene/reconstruction.h"
#include "scene/synthetic.h"

namespace colmap {
namespace {

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

Reconstruction MakeTrivialReconstruction(const std::vector<image_t>& ids) {
    Reconstruction reconstruction;
    Camera camera = Camera::CreateFromModelId(
            1, CameraModelId::kSimplePinhole, 10, 10, 5);
    reconstruction.AddCameraWithTrivialRig(camera);
    for (const image_t image_id : ids) {
        Image image;
        image.SetImageId(image_id);
        image.SetCameraId(camera.CameraId());
        reconstruction.AddImageWithTrivialFrame(image);
    }
    return reconstruction;
}

Eigen::Quaterniond RelativeRotation(const Reconstruction& reconstruction,
                                    const image_t image_id1,
                                    const image_t image_id2) {
    return reconstruction.Image(image_id2).CamFromWorld().rotation() *
           reconstruction.Image(image_id1).CamFromWorld().rotation().inverse();
}

TEST(CeresRotationAverager, RecoversNominalRotations) {
    Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3});
    PoseGraph pose_graph;
    pose_graph.AddEdge(1, 2, Edge(ZRotation(0.2)));
    pose_graph.AddEdge(2, 3, Edge(ZRotation(-0.1)));

    CeresRotationAveragerOptions options;
    auto averager = CreateDefaultCeresRotationAverager(
            options, pose_graph, reconstruction);
    EXPECT_EQ(averager->Problem().NumParameterBlocks(), 3);
    EXPECT_EQ(averager->Problem().NumResidualBlocks(), 2);
    EXPECT_EQ(averager->solver_options.linear_solver_type,
              ceres::SPARSE_NORMAL_CHOLESKY);
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    EXPECT_NEAR(RelativeRotation(reconstruction, 1, 2)
                        .angularDistance(ZRotation(0.2)),
                0.0,
                1e-10);
    EXPECT_NEAR(RelativeRotation(reconstruction, 2, 3)
                        .angularDistance(ZRotation(-0.1)),
                0.0,
                1e-10);
}

TEST(CeresRotationAverager, MatchCountReweightingSkipsZeroWeightEdges) {
    Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3});
    PoseGraph graph;
    graph.AddEdge(1, 2, Edge(ZRotation(0.2), 10));
    graph.AddEdge(2, 3, Edge(ZRotation(0.2), 5));
    graph.AddEdge(1, 3, Edge(ZRotation(0.2), 0));

    CeresRotationAveragerOptions options;
    options.skip_initialization = true;
    options.reweighting = RotationAveragingReweighting::INLIER_MATCH_COUNT;
    auto averager = CreateDefaultCeresRotationAverager(options, graph, reconstruction);
    EXPECT_EQ(averager->Problem().NumResidualBlocks(), 2);
}

TEST(CeresRotationAverager, RejectsDisconnectedOrUnknownGraph) {
    Reconstruction reconstruction = MakeTrivialReconstruction({1, 2, 3, 4});
    PoseGraph disconnected;
    disconnected.AddEdge(1, 2, Edge(ZRotation(0.1)));
    disconnected.AddEdge(3, 4, Edge(ZRotation(0.2)));
    EXPECT_THROW(CreateDefaultCeresRotationAverager(
                         CeresRotationAveragerOptions(),
                         disconnected,
                         reconstruction),
                 std::invalid_argument);

    PoseGraph unknown;
    unknown.AddEdge(1, 5, Edge(ZRotation(0.1)));
    EXPECT_THROW(CreateDefaultCeresRotationAverager(
                         CeresRotationAveragerOptions(), unknown, reconstruction),
                 std::invalid_argument);
}

TEST(CeresRotationAverager, PreservesCalibratedRigRotation) {
    Reconstruction reconstruction;
    SyntheticDatasetOptions dataset_options;
    dataset_options.num_rigs = 1;
    dataset_options.num_cameras_per_rig = 2;
    dataset_options.num_frames_per_rig = 2;
    dataset_options.num_points3D = 2;
    SynthesizeDataset(dataset_options, &reconstruction);

    std::vector<image_t> images;
    for (const auto& [image_id, image] : reconstruction.Images()) {
        images.push_back(image_id);
    }
    ASSERT_EQ(images.size(), 4);
    PoseGraph graph;
    for (size_t i = 1; i < images.size(); ++i) {
        graph.AddEdge(images[0],
                      images[i],
                      Edge(RelativeRotation(reconstruction, images[0], images[i])));
    }
    const Image* non_ref = nullptr;
    for (const image_t image_id : images) {
        const Image& image = reconstruction.Image(image_id);
        if (!image.IsRefInFrame()) {
            non_ref = &image;
            break;
        }
    }
    ASSERT_NE(non_ref, nullptr);
    const Eigen::Vector4d calibration_before =
            reconstruction.Rig(non_ref->FramePtr()->RigId())
                    .CamFromRigQvec(non_ref->CameraId());

    CeresRotationAveragerOptions options;
    options.refine_sensor_from_rig = false;
    auto averager = CreateDefaultCeresRotationAverager(options, graph, reconstruction);
    ASSERT_TRUE(averager->Solve().IsSolutionUsable());
    const Eigen::Vector4d calibration_after =
            reconstruction.Rig(non_ref->FramePtr()->RigId())
                    .CamFromRigQvec(non_ref->CameraId());
    EXPECT_EQ(calibration_after, calibration_before);
}

TEST(CeresRotationAverager, RetainsExternallyAddedLoss) {
    Reconstruction reconstruction = MakeTrivialReconstruction({1, 2});
    PoseGraph graph;
    graph.AddEdge(1, 2, Edge(ZRotation(0.2)));
    auto averager = CreateDefaultCeresRotationAverager(
            CeresRotationAveragerOptions(), graph, reconstruction);
    auto loss = std::make_shared<ceres::CauchyLoss>(0.05);
    std::weak_ptr<ceres::LossFunction> weak_loss = loss;
    averager->AddRelativeRotationResidual(1, 2, ZRotation(0.2), loss);
    loss.reset();
    EXPECT_FALSE(weak_loss.expired());
    EXPECT_EQ(averager->Problem().NumResidualBlocks(), 2);
    EXPECT_TRUE(averager->Solve().IsSolutionUsable());
}

}  // namespace
}  // namespace colmap
