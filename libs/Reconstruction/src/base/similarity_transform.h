// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "estimators/similarity_transform.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

struct RANSACOptions;
class Reconstruction;

// 3D similarity transformation with 7 degrees of freedom.
class SimilarityTransform3 {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    SimilarityTransform3();

    explicit SimilarityTransform3(const Eigen::Matrix3x4d& matrix);

    explicit SimilarityTransform3(
            const Eigen::Transform<double, 3, Eigen::Affine>& transform);

    SimilarityTransform3(const double scale,
                         const Eigen::Vector4d& qvec,
                         const Eigen::Vector3d& tvec);

    void Write(const std::string& path);

    template <bool kEstimateScale = true>
    bool Estimate(const std::vector<Eigen::Vector3d>& src,
                  const std::vector<Eigen::Vector3d>& dst);

    SimilarityTransform3 Inverse() const;

    void TransformPoint(Eigen::Vector3d* xyz) const;
    void TransformPose(Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) const;

    Eigen::Matrix4d Matrix() const;
    double Scale() const;
    Eigen::Vector4d Rotation() const;
    Eigen::Vector3d Translation() const;

    static SimilarityTransform3 FromFile(const std::string& path);

private:
    Eigen::Transform<double, 3, Eigen::Affine> transform_;
};

// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers. An alignment
// is verified by reprojecting common 3D point observations.
// The min_inlier_observations threshold determines how many observations
// in a common image must reproject within the given threshold.
bool ComputeAlignmentBetweenReconstructions(
        const Reconstruction& src_reconstruction,
        const Reconstruction& ref_reconstruction,
        const double min_inlier_observations,
        const double max_reproj_error,
        Eigen::Matrix3x4d* alignment);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <bool kEstimateScale>
bool SimilarityTransform3::Estimate(const std::vector<Eigen::Vector3d>& src,
                                    const std::vector<Eigen::Vector3d>& dst) {
    const auto results =
            SimilarityTransformEstimator<3, kEstimateScale>().Estimate(src,
                                                                       dst);
    if (results.empty()) {
        return false;
    }

    CHECK_EQ(results.size(), 1);
    transform_.matrix().topLeftCorner<3, 4>() = results[0];

    return true;
}

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::SimilarityTransform3)
