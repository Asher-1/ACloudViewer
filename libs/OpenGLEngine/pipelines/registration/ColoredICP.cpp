// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#include "pipelines/registration/ColoredICP.h"

#include <Eigen/Dense>
#include <iostream>
#include <Console.h>
#include <Eigen.h>

#include <ecvKDTreeFlann.h>
#include <ecvKDTreeSearchParam.h>
#include <ecvPointCloud.h>

namespace cloudViewer {
namespace pipelines {
namespace registration {

namespace {

class PointCloudForColoredICP : public ccPointCloud {
public:
    std::vector<Eigen::Vector3d> color_gradient_;
};

std::shared_ptr<PointCloudForColoredICP> InitializePointCloudForColoredICP(
    const ccPointCloud& target,
    const geometry::KDTreeSearchParamHybrid& search_param) {
    CVLib::utility::LogDebug("InitializePointCloudForColoredICP");

    geometry::KDTreeFlann tree;
    tree.SetGeometry(target);

    auto output = std::make_shared<PointCloudForColoredICP>();

    output->reserveThePointsTable(target.size());
    if (target.hasColors())
    {
        output->reserveTheRGBTable();
    }
    if (target.hasNormals())
    {
        output->reserveTheNormsTable();
    }

    for (unsigned int i = 0; i < target.size(); ++i)
    {
        output->addPoint(*target.getPoint(i));

        if (target.hasColors())
        {
            output->addRGBColor(target.getPointColor(i));
        }

        if (target.hasNormals())
        {
            output->addNorm(target.getPointNormal(i));
        }
    }

    size_t n_points = output->size();
    output->color_gradient_.resize(n_points, Eigen::Vector3d::Zero());

    Eigen::Vector3d colors;
    for (size_t k = 0; k < n_points; k++) {
        const Eigen::Vector3d& vt = output->getEigenPoint(k);
        const Eigen::Vector3d& nt = output->getEigenNormal(k);

        const ecvColor::Rgb& col =
            output->getPointColor(static_cast<unsigned int>(k));
        colors = ecvColor::Rgb::ToEigen(col);
        double it = (colors(0) + colors(1) + colors(2)) / 3.0;

        std::vector<int> point_idx;
        std::vector<double> point_squared_distance;

        if (tree.SearchHybrid(vt, search_param.radius_, search_param.max_nn_,
            point_idx, point_squared_distance) >= 4) {
            // approximate image gradient of vt's tangential plane
            size_t nn = point_idx.size();
            Eigen::MatrixXd A(nn, 3);
            Eigen::MatrixXd b(nn, 1);
            A.setZero();
            b.setZero();
            for (size_t i = 1; i < nn; i++) {
                int P_adj_idx = point_idx[i];
                Eigen::Vector3d vt_adj = output->getEigenPoint(P_adj_idx);
                Eigen::Vector3d vt_proj = vt_adj - (vt_adj - vt).dot(nt) * nt;

                const ecvColor::Rgb& col =
                    output->getPointColor(static_cast<unsigned int>(P_adj_idx));
                colors = ecvColor::Rgb::ToEigen(col);
                double it_adj = (colors(0) + colors(1) + colors(2)) / 3.0;
                A(i - 1, 0) = (vt_proj(0) - vt(0));
                A(i - 1, 1) = (vt_proj(1) - vt(1));
                A(i - 1, 2) = (vt_proj(2) - vt(2));
                b(i - 1, 0) = (it_adj - it);
            }
            // adds orthogonal constraint
            A(nn - 1, 0) = (nn - 1) * nt(0);
            A(nn - 1, 1) = (nn - 1) * nt(1);
            A(nn - 1, 2) = (nn - 1) * nt(2);
            b(nn - 1, 0) = 0;
            // solving linear equation
            bool is_success;
            Eigen::MatrixXd x;
            std::tie(is_success, x) = CVLib::utility::SolveLinearSystemPSD(
                A.transpose() * A, A.transpose() * b);
            if (is_success) {
                output->color_gradient_[k] = x;
            }
        }
    }
    return output;
}


}  // namespace

Eigen::Matrix4d TransformationEstimationForColoredICP::ComputeTransformation(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || target.hasNormals() == false ||
        target.hasColors() == false || source.hasColors() == false)
        return Eigen::Matrix4d::Identity();

    double sqrt_lambda_geometric = sqrt(lambda_geometric_);
    double lambda_photometric = 1.0 - lambda_geometric_;
    double sqrt_lambda_photometric = sqrt(lambda_photometric);

    const auto &target_c = (const PointCloudForColoredICP &)target;

    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, CVLib::utility::Vector6d_allocator> &J_r,
                std::vector<double> &r, std::vector<double>& w) {
                size_t cs = corres[i][0];
                size_t ct = corres[i][1];
				const Eigen::Vector3d &vs = source.getEigenPoint(cs);
				const Eigen::Vector3d &vt = target.getEigenPoint(ct);
				const Eigen::Vector3d &nt = target.getEigenNormal(ct);

                J_r.resize(2);
                r.resize(2);
                w.resize(2);

                J_r[0].block<3, 1>(0, 0) = sqrt_lambda_geometric * vs.cross(nt);
                J_r[0].block<3, 1>(3, 0) = sqrt_lambda_geometric * nt;
                r[0] = sqrt_lambda_geometric * (vs - vt).dot(nt);
                w[0] = kernel_->Weight(r[0]);

                // project vs into vt's tangential plane
                Eigen::Vector3d vs_proj = vs - (vs - vt).dot(nt) * nt;
				const ecvColor::Rgb& col_source =
					source.getPointColor(static_cast<unsigned int>(cs));
				Eigen::Vector3d colors = ecvColor::Rgb::ToEigen(col_source);
                double is = (colors(0) + colors(1) + colors(2)) / 3.0;

				const ecvColor::Rgb& col_target = 
					target.getPointColor(static_cast<unsigned int>(ct));
				colors = ecvColor::Rgb::ToEigen(col_target);
                double it = (colors(0) + colors(1) + colors(2)) / 3.0;

                const Eigen::Vector3d &dit = target_c.color_gradient_[ct];
                double is0_proj = (dit.dot(vs_proj - vt)) + it;

                const Eigen::Matrix3d M =
                        (Eigen::Matrix3d() << 1.0 - nt(0) * nt(0),
                         -nt(0) * nt(1), -nt(0) * nt(2), -nt(0) * nt(1),
                         1.0 - nt(1) * nt(1), -nt(1) * nt(2), -nt(0) * nt(2),
                         -nt(1) * nt(2), 1.0 - nt(2) * nt(2))
                                .finished();

                const Eigen::Vector3d &ditM = -dit.transpose() * M;
                J_r[1].block<3, 1>(0, 0) =
                        sqrt_lambda_photometric * vs.cross(ditM);
                J_r[1].block<3, 1>(3, 0) = sqrt_lambda_photometric * ditM;
                r[1] = sqrt_lambda_photometric * (is - is0_proj);
                w[1] = kernel_->Weight(r[1]);
            };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    double r2;
    std::tie(JTJ, JTr, r2) =
            CVLib::utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                    compute_jacobian_and_residual, (int)corres.size());

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            CVLib::utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4d::Identity();
}

double TransformationEstimationForColoredICP::ComputeRMSE(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const CorrespondenceSet &corres) const {
    double sqrt_lambda_geometric = sqrt(lambda_geometric_);
    double lambda_photometric = 1.0 - lambda_geometric_;
    double sqrt_lambda_photometric = sqrt(lambda_photometric);
    const auto &target_c = (const PointCloudForColoredICP &)target;

    double residual = 0.0;
    for (size_t i = 0; i < corres.size(); i++) {
        size_t cs = corres[i][0];
        size_t ct = corres[i][1];
		const Eigen::Vector3d &vs = source.getEigenPoint(cs);
		const Eigen::Vector3d &vt = target.getEigenPoint(ct);
		const Eigen::Vector3d &nt = target.getEigenNormal(ct);
        Eigen::Vector3d vs_proj = vs - (vs - vt).dot(nt) * nt;
		const ecvColor::Rgb& col_source =
			source.getPointColor(static_cast<unsigned int>(cs));
		Eigen::Vector3d colors = ecvColor::Rgb::ToEigen(col_source);
		double is = (colors(0) + colors(1) + colors(2)) / 3.0;

		const ecvColor::Rgb& col_target =
			target.getPointColor(static_cast<unsigned int>(ct));
		colors = ecvColor::Rgb::ToEigen(col_target);
		double it = (colors(0) + colors(1) + colors(2)) / 3.0;

        const Eigen::Vector3d &dit = target_c.color_gradient_[ct];
        double is0_proj = (dit.dot(vs_proj - vt)) + it;
        double residual_geometric = sqrt_lambda_geometric * (vs - vt).dot(nt);
        double residual_photometric = sqrt_lambda_photometric * (is - is0_proj);
        residual += residual_geometric * residual_geometric +
                    residual_photometric * residual_photometric;
    }
    return residual;
};

RegistrationResult RegistrationColoredICP(
        const ccPointCloud &source,
        const ccPointCloud &target,
        double max_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimationForColoredICP& estimation
        /*TransformationEstimationForColoredICP()*/,
        const ICPConvergenceCriteria
        & criteria /* = ICPConvergenceCriteria()*/) {
    auto target_c = InitializePointCloudForColoredICP(
            target, geometry::KDTreeSearchParamHybrid(max_distance * 2.0, 30));
    return RegistrationICP(
            source, *target_c, max_distance, init, estimation, criteria);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace cloudViewer
