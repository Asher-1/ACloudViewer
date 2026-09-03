// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
#include "estimators/relpose_one_sided_focal.h"

#include <cmath>
#include <limits>

#include <PoseLib/solvers/relpose_6pt_onesided_focal.h>

namespace colmap {

namespace {

Eigen::Matrix3d EssentialFromPose(const Eigen::Matrix3d& rotation,
                                  const Eigen::Vector3d& translation) {
    const Eigen::Vector3d t = translation.normalized();
    Eigen::Matrix3d tx;
    tx << 0.0, -t.z(), t.y(), t.z(), 0.0, -t.x(), -t.y(), t.x(), 0.0;
    return tx * rotation;
}

}  // namespace

std::vector<RelativePoseOneSidedFocalEstimator::M_t>
RelativePoseOneSidedFocalEstimator::Estimate(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2) {
    if (points1.size() != points2.size() ||
        points1.size() < kMinNumSamples) {
        return {};
    }
    std::vector<Eigen::Vector3d> x1(points1.size());
    std::vector<Eigen::Vector3d> x2(points2.size());
    for (size_t i = 0; i < points1.size(); ++i) {
        x1[i] = points1[i].homogeneous();
        x2[i] = points2[i].homogeneous();
    }

    poselib::ImagePairVector image_pairs;
    // The full template solves 1/f^2 directly and is more accurate than the
    // compact fundamental-matrix recovery on pixel-scale data.
    poselib::relpose_6pt_onesided_focal(x1, x2, &image_pairs, false);

    std::vector<M_t> models;
    models.reserve(image_pairs.size());
    for (const auto& image_pair : image_pairs) {
        const double focal = image_pair.camera1.focal();
        if (!(focal > 0.0) || !std::isfinite(focal)) {
            continue;
        }
        M_t model;
        model.E = EssentialFromPose(image_pair.pose.R(), image_pair.pose.t);
        model.focal = focal;
        models.push_back(model);
    }
    return models;
}

void RelativePoseOneSidedFocalEstimator::Residuals(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2,
    const M_t& model, std::vector<double>* residuals) {
    residuals->assign(points1.size(), std::numeric_limits<double>::max());
    if (points1.size() != points2.size() || !(model.focal > 0.0)) {
        return;
    }
    const Eigen::Matrix3d K_inv =
        Eigen::Vector3d(1.0 / model.focal, 1.0 / model.focal, 1.0)
            .asDiagonal();
    const Eigen::Matrix3d mixed = model.E * K_inv;
    for (size_t i = 0; i < points1.size(); ++i) {
        const Eigen::Vector3d x1 = points1[i].homogeneous();
        const Eigen::Vector3d x2 = points2[i].homogeneous();
        const Eigen::Vector3d line1 = mixed.transpose() * x2;
        const double numerator = x2.dot(mixed * x1);
        const double denom = line1.head<2>().squaredNorm();
        if (denom > 1e-15 && std::isfinite(denom)) {
            (*residuals)[i] = numerator * numerator / denom;
        }
    }
}

}  // namespace colmap
