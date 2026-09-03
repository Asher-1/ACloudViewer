// Deterministic numeric gate for the six-point one-sided focal solver.
#define TEST_NAME "estimators/relpose_one_sided_focal"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Geometry>

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
    std::vector<Eigen::Vector2d> points2;
    points1.reserve(points.size());
    points2.reserve(points.size());
    for (const Eigen::Vector3d& point : points) {
        const Eigen::Vector3d point2 = rotation * point + translation;
        points1.emplace_back(kFocal * point.x() / point.z(),
                             kFocal * point.y() / point.z());
        // The public adapter stores calibrated rays as normalized image-plane
        // coordinates (x/z, y/z); the solver restores the homogeneous z=1.
        points2.emplace_back(point2.x() / point2.z(), point2.y() / point2.z());
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
    const std::vector<Eigen::Vector2d> points(5, Eigen::Vector2d::Zero());
    BOOST_CHECK(RelativePoseOneSidedFocalEstimator::Estimate(points, points)
                    .empty());
}
