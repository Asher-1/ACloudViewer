// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>

#include "pipelines/registration/Registration.h"
#include "pipelines/registration/RobustKernel.h"
#include "pipelines/registration/TransformationEstimation.h"

class ccPointCloud;
namespace cloudViewer {

namespace pipelines {
namespace registration {

class RegistrationResult;

class TransformationEstimationForColoredICP : public TransformationEstimation {
public:
    ~TransformationEstimationForColoredICP() override{};

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    explicit TransformationEstimationForColoredICP(
            double lambda_geometric = 0.968,
            std::shared_ptr<RobustKernel> kernel =
                    cloudViewer::make_shared<L2Loss>())
        : lambda_geometric_(lambda_geometric), kernel_(std::move(kernel)) {
        if (lambda_geometric_ < 0 || lambda_geometric_ > 1.0) {
            lambda_geometric_ = 0.968;
        }
    }

public:
    double ComputeRMSE(const ccPointCloud& source,
                       const ccPointCloud& target,
                       const CorrespondenceSet& corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const ccPointCloud& source,
            const ccPointCloud& target,
            const CorrespondenceSet& corres) const override;

public:
    double lambda_geometric_ = 0.968;
    /// shared_ptr to an Abstract RobustKernel that could mutate at runtime.
    std::shared_ptr<RobustKernel> kernel_ = cloudViewer::make_shared<L2Loss>();

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::ColoredICP;
};

/// \brief Function for Colored ICP registration.
///
/// This is implementation of following paper
/// J. Park, Q.-Y. Zhou, V. Koltun,
/// Colored Point Cloud Registration Revisited, ICCV 2017.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_distance Maximum correspondence points-pair distance.
/// \param init Initial transformation estimation.
/// Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
/// [0., 0., 0., 1.]]).
/// \param estimation TransformationEstimationForColoredICP method. Can only
/// change the lambda_geometric value and the robust kernel used in the
/// optimization
/// \param criteria  Convergence criteria.
RegistrationResult RegistrationColoredICP(
        const ccPointCloud& source,
        const ccPointCloud& target,
        double max_distance,
        const Eigen::Matrix4d& init = Eigen::Matrix4d::Identity(),
        const TransformationEstimationForColoredICP& estimation =
                TransformationEstimationForColoredICP(),
        const ICPConvergenceCriteria& criteria = ICPConvergenceCriteria());

}  // namespace registration
}  // namespace pipelines
}  // namespace cloudViewer
