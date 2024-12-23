// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "pipelines/registration/TransformationEstimation.h"

#include <Eigen.h>
#include <Eigen/Geometry>

#include <ecvPointCloud.h>

namespace cloudViewer {
namespace pipelines {
namespace registration {

double TransformationEstimationPointToPoint::ComputeRMSE(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty()) return 0.0;
    double err = 0.0;
    for (const auto &c : corres) {
        err += (source.getEigenPoint(c[0]) - target.getEigenPoint(c[1])).squaredNorm();
    }
    return std::sqrt(err / (double)corres.size());
}

Eigen::Matrix4d TransformationEstimationPointToPoint::ComputeTransformation(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty()) return Eigen::Matrix4d::Identity();
    Eigen::MatrixXd source_mat(3, corres.size());
    Eigen::MatrixXd target_mat(3, corres.size());
    for (size_t i = 0; i < corres.size(); i++) {
        source_mat.block<3, 1>(0, i) = source.getEigenPoint(corres[i][0]);
        target_mat.block<3, 1>(0, i) = target.getEigenPoint(corres[i][1]);
    }
    return Eigen::umeyama(source_mat, target_mat, with_scaling_);
}

double TransformationEstimationPointToPlane::ComputeRMSE(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || target.hasNormals() == false) return 0.0;
    double err = 0.0, r;
    for (const auto &c : corres) {
        r = (source.getEigenPoint(c[0]) - target.getEigenPoint(c[1]))
                    .dot(target.getEigenNormal(c[1]));
        err += r * r;
    }
    return std::sqrt(err / (double)corres.size());
}

Eigen::Matrix4d TransformationEstimationPointToPlane::ComputeTransformation(
    const ccPointCloud& source,
    const ccPointCloud& target,
    const CorrespondenceSet& corres) const {
    if (corres.empty() || target.hasNormals() == false)
        return Eigen::Matrix4d::Identity();

    auto compute_jacobian_and_residual = [&](int i, Eigen::Vector6d& J_r,
                                             double& r, double& w) {
            const Eigen::Vector3d& vs = source.getEigenPoint(corres[i][0]);
            const Eigen::Vector3d& vt = target.getEigenPoint(corres[i][1]);
            const Eigen::Vector3d& nt = target.getEigenNormal(corres[i][1]);
            r = (vs - vt).dot(nt);
            w = kernel_->Weight(r);
            J_r.block<3, 1>(0, 0) = vs.cross(nt);
            J_r.block<3, 1>(3, 0) = nt;
    };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    double r2;
    std::tie(JTJ, JTr, r2) =
        cloudViewer::utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
            compute_jacobian_and_residual, (int)corres.size());

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
        cloudViewer::utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4d::Identity();
};

}  // namespace registration
}  // namespace pipelines
}  // namespace cloudViewer
