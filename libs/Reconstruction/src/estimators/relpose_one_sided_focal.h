// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
#pragma once

#include <Eigen/Core>
#include <vector>

namespace colmap {

// Minimal six-point relative pose estimator with an unknown focal length in
// the first (pixel) view and calibrated normalized points in the second view.
// PoseLib is optional; this header is only compiled into the reconstruction
// target when RECONSTRUCTION_FETCH_POSELIB is enabled.
class RelativePoseOneSidedFocalEstimator {
public:
    using X_t = Eigen::Vector2d;
    using Y_t = Eigen::Vector2d;
    struct M_t {
        Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
        double focal = 0.0;
    };
    static const int kMinNumSamples = 6;

    static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                     const std::vector<Y_t>& points2);
    static void Residuals(const std::vector<X_t>& points1,
                          const std::vector<Y_t>& points2,
                          const M_t& model,
                          std::vector<double>* residuals);
};

}  // namespace colmap
