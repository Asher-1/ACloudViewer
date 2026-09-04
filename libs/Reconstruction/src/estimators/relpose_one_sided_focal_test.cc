// Deterministic numeric gate for the six-point one-sided focal solver.
#define TEST_NAME "estimators/relpose_one_sided_focal"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <Eigen/Geometry>

#include "base/camera.h"
#include "base/camera_models.h"
#include "estimators/relpose_one_sided_focal.h"
#include "util/testing.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestExactMinimalSampleRecoversFocalAndPose) {
    constexpr double kFocal = 800.0;
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.18, Eigen::Vector3d(0.2, -0.4, 0.7).normalized())
            .toRotationMatrix();
    const Eigen::Vector3d translation(0.7, -0.2, 0.5);

    const std::vector<Eigen::Vector3d> points = {
        {0.2, -0.1, 2.2}, {-0.4, 0.3, 2.8}, {0.5, 0.2, 3.1},
        {-0.3, -0.4, 2.5}, {0.1, 0.5, 3.4}, {-0.6, -0.2, 3.0}};
    std::vector<Eigen::Vector2d> points1;
    Camera camera2;
    camera2.InitializeWithId(SimpleRadialCameraModel::kModelId, 800.0, 1600,
                             1200);
    camera2.Params(3) = 0.06;
    std::vector<CamRayWithJac> points2;
    points1.reserve(points.size());
    points2.reserve(points.size());
    for (const Eigen::Vector3d& point : points) {
        const Eigen::Vector3d point2 = rotation * point + translation;
        points1.emplace_back(kFocal * point.x() / point.z(),
                             kFocal * point.y() / point.z());
        const Eigen::Vector2d pixel = camera2.WorldToImage(
            Eigen::Vector2d(point2.x() / point2.z(), point2.y() / point2.z()));
        const auto ray_with_jac = camera2.CamRayFromImgWithJac(pixel);
        BOOST_REQUIRE(ray_with_jac.has_value());
        points2.push_back(*ray_with_jac);
    }

    const auto models =
        RelativePoseOneSidedFocalEstimator::Estimate(points1, points2);
    BOOST_REQUIRE(!models.empty());

    Eigen::Matrix3d expected_tx;
    const Eigen::Vector3d t = translation.normalized();
    expected_tx << 0.0, -t.z(), t.y(), t.z(), 0.0, -t.x(), -t.y(), t.x(), 0.0;
    const Eigen::Matrix3d expected_e = expected_tx * rotation;

    bool found = false;
    for (const auto& model : models) {
        if (std::abs(model.focal - kFocal) / kFocal > 1e-3) {
            continue;
        }
        const Eigen::Matrix3d e = model.E.normalized();
        const double e_error = std::min((e - expected_e.normalized()).norm(),
                                        (e + expected_e.normalized()).norm());
        if (e_error > 1e-2) {
            continue;
        }
        std::vector<double> residuals;
        RelativePoseOneSidedFocalEstimator::Residuals(
            points1, points2, model, &residuals);
        if (std::all_of(residuals.begin(), residuals.end(),
                        [](const double residual) { return residual < 1e-8; })) {
            found = true;
            break;
        }
    }
    BOOST_CHECK(found);
}

BOOST_AUTO_TEST_CASE(TestRejectsNonMinimalInput) {
    const std::vector<Eigen::Vector2d> points1(5, Eigen::Vector2d::Zero());
    const std::vector<CamRayWithJac> points2(5, CamRayWithJac::Zero());
    BOOST_CHECK(RelativePoseOneSidedFocalEstimator::Estimate(points1, points2)
                    .empty());
}

BOOST_AUTO_TEST_CASE(TestTinySolverRefinementReducesTangentSampsonCost) {
    constexpr double kFocal = 800.0;
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.23, Eigen::Vector3d(0.3, -0.2, 0.7).normalized())
            .toRotationMatrix();
    const Eigen::Vector3d translation(0.5, -0.3, 0.8);
    Camera camera2;
    camera2.InitializeWithId(SimpleRadialCameraModel::kModelId, 800.0, 1600,
                             1200);
    camera2.Params(3) = 0.04;

    std::vector<Eigen::Vector2d> points1;
    std::vector<CamRayWithJac> points2;
    for (int i = 0; i < 20; ++i) {
        const Eigen::Vector3d point(
            0.08 * (i % 5) - 0.15,
            0.07 * (i / 5) - 0.1,
            2.5 + 0.1 * i + 0.05 * static_cast<double>((i * i) % 7));
        const Eigen::Vector3d point2 = rotation * point + translation;
        points1.emplace_back(kFocal * point.x() / point.z(),
                             kFocal * point.y() / point.z());
        const Eigen::Vector2d pixel = camera2.WorldToImage(
            Eigen::Vector2d(point2.x() / point2.z(), point2.y() / point2.z()));
        const auto ray_with_jac = camera2.CamRayFromImgWithJac(pixel);
        BOOST_REQUIRE(ray_with_jac.has_value());
        points2.push_back(*ray_with_jac);
    }

    RelativePoseOneSidedFocalEstimator::M_t model;
    const Eigen::Matrix3d initial_rotation =
        Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitY()).toRotationMatrix() *
        rotation;
    const Eigen::Vector3d initial_translation =
        translation + Eigen::Vector3d(0.03, -0.02, 0.01);
    model.E = CrossProductMatrix(initial_translation.normalized()) *
              initial_rotation;
    model.focal = 0.93 * kFocal;
    std::vector<double> before;
    RelativePoseOneSidedFocalEstimator::Residuals(points1, points2, model,
                                                   &before);
    const double before_cost =
        std::accumulate(before.begin(), before.end(), 0.0);
    BOOST_REQUIRE(RelativePoseOneSidedFocalEstimator::Refine(points1, points2,
                                                              &model));
    std::vector<double> after;
    RelativePoseOneSidedFocalEstimator::Residuals(points1, points2, model,
                                                   &after);
    const double after_cost = std::accumulate(after.begin(), after.end(), 0.0);
    BOOST_CHECK(after_cost < before_cost);
    BOOST_TEST_MESSAGE("TinySolver focal=" << model.focal
                                             << ", tangent cost "
                                             << before_cost << " -> "
                                             << after_cost);
    BOOST_CHECK(std::abs(model.focal - kFocal) / kFocal < 0.02);
}
