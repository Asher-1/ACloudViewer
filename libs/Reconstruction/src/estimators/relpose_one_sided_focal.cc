// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
#include "estimators/relpose_one_sided_focal.h"

#include "base/essential_matrix.h"

#include <ceres/tiny_solver.h>

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

class TinyOneSidedFocalTangentSampsonCost {
public:
    using Scalar = double;
    static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
    static constexpr int NUM_PARAMETERS = 8;

    TinyOneSidedFocalTangentSampsonCost(
        const std::vector<Eigen::Vector2d>& points1,
        const std::vector<CamRayWithJac>& points2)
        : points1_(points1), points2_(points2) {}

    int NumResiduals() const { return static_cast<int>(points1_.size()); }

    bool operator()(const double* parameters,
                    double* residuals,
                    double* jacobian) const {
        Evaluate(parameters, residuals);
        if (jacobian == nullptr) {
            return true;
        }

        // Ceres 1.14's TinySolver autodiff wrapper only accepts a compile-time
        // residual count. LO-RANSAC refines a variable number of inliers, so
        // use a bounded central difference Jacobian while retaining TinySolver's
        // fixed-size 8-parameter Levenberg-Marquardt loop.
        constexpr double kStep = 1e-6;
        const int num_residuals = NumResiduals();
        Eigen::Matrix<double, NUM_PARAMETERS, 1> plus;
        Eigen::Matrix<double, NUM_PARAMETERS, 1> minus;
        std::vector<double> plus_residuals(num_residuals);
        std::vector<double> minus_residuals(num_residuals);
        for (int parameter = 0; parameter < NUM_PARAMETERS; ++parameter) {
            for (int i = 0; i < NUM_PARAMETERS; ++i) {
                plus[i] = parameters[i];
                minus[i] = parameters[i];
            }
            plus[parameter] += kStep;
            minus[parameter] -= kStep;
            Evaluate(plus.data(), plus_residuals.data());
            Evaluate(minus.data(), minus_residuals.data());
            for (int residual = 0; residual < num_residuals; ++residual) {
                jacobian[residual + parameter * num_residuals] =
                    (plus_residuals[residual] - minus_residuals[residual]) /
                    (2.0 * kStep);
            }
        }
        return true;
    }

private:
    void Evaluate(const double* parameters, double* residuals) const {
        Eigen::Quaterniond rotation(parameters[3], parameters[0], parameters[1],
                                    parameters[2]);
        rotation.normalize();
        Eigen::Vector3d translation(parameters[4], parameters[5], parameters[6]);
        translation /= std::sqrt(translation.squaredNorm() + 1e-24);

        Eigen::Matrix3d tx;
        tx << 0.0, -translation.z(), translation.y(), translation.z(), 0.0,
            -translation.x(), -translation.y(), translation.x(), 0.0;
        Eigen::Matrix3d mixed = tx * rotation.toRotationMatrix();
        mixed.leftCols<2>() *= std::exp(-parameters[7]);

        for (size_t i = 0; i < points1_.size(); ++i) {
            const Eigen::Vector3d point1 = points1_[i].homogeneous();
            const Eigen::Vector3d& ray2 = points2_[i].ray;
            const Eigen::Matrix<double, 3, 2>& jacobian2 =
                points2_[i].jacobian;
            const Eigen::Vector3d mixed_point1 = mixed * point1;
            const double numerator = ray2.dot(mixed_point1);
            const Eigen::Vector2d gradient1 =
                (mixed.transpose() * ray2).head<2>();
            const Eigen::Vector2d gradient2 =
                jacobian2.transpose() * mixed_point1;
            residuals[i] = numerator /
                           std::sqrt(gradient1.squaredNorm() +
                                     gradient2.squaredNorm() + 1e-24);
        }
    }

    const std::vector<Eigen::Vector2d>& points1_;
    const std::vector<CamRayWithJac>& points2_;
};

bool SelectPoseFromEssential(const Eigen::Matrix3d& essential,
                             const double focal,
                             const std::vector<Eigen::Vector2d>& points1,
                             const std::vector<CamRayWithJac>& points2,
                             Eigen::Matrix3d* rotation,
                             Eigen::Vector3d* translation) {
    Eigen::Matrix3d rotation1;
    Eigen::Matrix3d rotation2;
    Eigen::Vector3d translation_seed;
    DecomposeEssentialMatrix(essential, &rotation1, &rotation2,
                             &translation_seed);

    const Eigen::Matrix3d rotations[] = {rotation1, rotation2};
    const Eigen::Vector3d translations[] = {translation_seed, -translation_seed};
    double best_cost = std::numeric_limits<double>::infinity();
    for (const Eigen::Matrix3d& candidate_rotation : rotations) {
        for (const Eigen::Vector3d& candidate_translation : translations) {
            RelativePoseOneSidedFocalEstimator::M_t candidate;
            candidate.E = EssentialFromPose(candidate_rotation,
                                            candidate_translation);
            candidate.focal = focal;
            std::vector<double> residuals;
            RelativePoseOneSidedFocalEstimator::Residuals(points1, points2,
                                                           candidate, &residuals);
            double cost = 0.0;
            for (const double residual : residuals) {
                if (!std::isfinite(residual)) {
                    cost = std::numeric_limits<double>::infinity();
                    break;
                }
                cost += residual;
            }
            if (cost < best_cost) {
                best_cost = cost;
                *rotation = candidate_rotation;
                *translation = candidate_translation;
            }
        }
    }
    return std::isfinite(best_cost);
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
        x2[i] = points2[i].ray;
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

bool RelativePoseOneSidedFocalEstimator::Refine(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    M_t* model) {
    if (model == nullptr || points1.size() != points2.size() ||
        points1.size() < kMinNumSamples || !(model->focal > 0.0) ||
        !std::isfinite(model->focal)) {
        return false;
    }

    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    if (!SelectPoseFromEssential(model->E, model->focal, points1, points2,
                                 &rotation, &translation)) {
        return false;
    }

    TinyOneSidedFocalTangentSampsonCost cost(points1, points2);
    ceres::TinySolver<TinyOneSidedFocalTangentSampsonCost> solver;

    Eigen::Matrix<double, 8, 1> parameters;
    const Eigen::Quaterniond quaternion(rotation);
    parameters.head<4>() = quaternion.coeffs();
    parameters.segment<3>(4) = translation.normalized();
    parameters[7] = std::log(model->focal);
    solver.Solve(cost, &parameters);

    const Eigen::Quaterniond refined_rotation(
        parameters[3], parameters[0], parameters[1], parameters[2]);
    const Eigen::Vector3d refined_translation = parameters.segment<3>(4);
    const double refined_focal = std::exp(parameters[7]);
    if (!parameters.allFinite() || refined_translation.squaredNorm() <= 1e-24 ||
        !(refined_focal > 0.0) || !std::isfinite(refined_focal)) {
        return false;
    }

    model->E = EssentialFromPose(refined_rotation.normalized().toRotationMatrix(),
                                 refined_translation);
    model->focal = refined_focal;
    return true;
}

void RelativePoseOneSidedFocalEstimator::Residuals(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2,
    const M_t& model, std::vector<double>* residuals) {
    residuals->assign(points1.size(), std::numeric_limits<double>::max());
    if (points1.size() != points2.size() || !(model.focal > 0.0)) {
        return;
    }
    const Eigen::Matrix3d mixed =
        model.E * Eigen::Vector3d(1.0 / model.focal, 1.0 / model.focal, 1.0)
                      .asDiagonal();
    for (size_t i = 0; i < points1.size(); ++i) {
        const Eigen::Vector3d x1 = points1[i].homogeneous();
        const Eigen::Vector3d& ray2 = points2[i].ray;
        const Eigen::Vector3d mixed_x1 = mixed * x1;
        const Eigen::Vector3d line1 = mixed.transpose() * ray2;
        const double numerator = ray2.dot(mixed_x1);
        // The unknown-focal image point stays in pixel coordinates, whose
        // measurement Jacobian is [I; 0]. The calibrated side contributes
        // d(ray2)/d(pixel2), preserving a pixel-unit tangent Sampson score.
        const Eigen::Vector2d gradient2 =
            points2[i].jacobian.transpose() * mixed_x1;
        const double denom =
            line1.head<2>().squaredNorm() + gradient2.squaredNorm();
        if (denom > 1e-15 && std::isfinite(denom)) {
            (*residuals)[i] = numerator * numerator / denom;
        }
    }
}

}  // namespace colmap
