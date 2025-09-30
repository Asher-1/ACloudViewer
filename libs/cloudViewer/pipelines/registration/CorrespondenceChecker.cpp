// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pipelines/registration/CorrespondenceChecker.h"

#include <Eigen/Dense>
#include <Logging.h>

#include <ecvPointCloud.h>

namespace cloudViewer {
namespace pipelines {
namespace registration {

bool CorrespondenceCheckerBasedOnEdgeLength::Check(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d & /*transformation*/) const {
    for (size_t i = 0; i < corres.size(); i++) {
        for (size_t j = i + 1; j < corres.size(); j++) {
            // check edge ij
            double dis_source = (source.getEigenPoint(corres[i](0)) -
                                 source.getEigenPoint(corres[j](0)))
                                        .norm();
            double dis_target = (target.getEigenPoint(corres[i](1)) -
                                 target.getEigenPoint(corres[j](1)))
                                        .norm();
            if (dis_source < dis_target * similarity_threshold_ ||
                dis_target < dis_source * similarity_threshold_) {
                return false;
            }
        }
    }
    return true;
}

bool CorrespondenceCheckerBasedOnDistance::Check(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d &transformation) const {
    for (const auto &c : corres) {
        const auto &pt = source.getEigenPoint(c(0));
        Eigen::Vector3d pt_trans =
                (transformation * Eigen::Vector4d(pt(0), pt(1), pt(2), 1.0))
                        .block<3, 1>(0, 0);
        if ((target.getEigenPoint(c(1)) - pt_trans).norm() > distance_threshold_) {
            return false;
        }
    }
    return true;
}

bool CorrespondenceCheckerBasedOnNormal::Check(
        const ccPointCloud &source,
        const ccPointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d &transformation) const {
    if (!source.hasNormals() || !target.hasNormals()) {
        utility::LogWarning(
                "[CorrespondenceCheckerBasedOnNormal::Check] Pointcloud has no "
                "normals.");
        return true;
    }
    double cos_normal_angle_threshold = std::cos(normal_angle_threshold_);
    for (const auto &c : corres) {
        const auto &normal = source.getEigenNormal(c(0));
        Eigen::Vector3d normal_trans =
                (transformation *
                 Eigen::Vector4d(normal(0), normal(1), normal(2), 0.0))
                        .block<3, 1>(0, 0);
        if (target.getEigenNormal(c(1)).dot(normal_trans) <
            cos_normal_angle_threshold) {
            return false;
        }
    }
    return true;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace cloudViewer
