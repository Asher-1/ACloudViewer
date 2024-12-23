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

#pragma once

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <vector>

#include "pipelines/odometry/OdometryOption.h"
#include <Eigen.h>

namespace cloudViewer {

namespace geometry {
class Image;
}

namespace geometry {
class RGBDImage;
}

namespace pipelines {
namespace odometry {

typedef std::vector<Eigen::Vector4i, utility::Vector4i_allocator>
        CorrespondenceSetPixelWise;

/// \class RGBDOdometryJacobian
///
/// \brief Base class that computes Jacobian from two RGB-D images.
class RGBDOdometryJacobian {
public:
    /// \brief Default Constructor.
    RGBDOdometryJacobian() {}
    virtual ~RGBDOdometryJacobian() {}

public:
    /// Function to compute i-th row of J and r
    /// the vector form of J_r is basically 6x1 matrix, but it can be
    /// easily extendable to 6xn matrix.
    /// See RGBDOdometryJacobianFromHybridTerm for this case.
    virtual void ComputeJacobianAndResidual(
            int row,
            std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
            std::vector<double> &r,
            std::vector<double>& w,
            const geometry::RGBDImage &source,
            const geometry::RGBDImage &target,
            const geometry::Image &source_xyz,
            const geometry::RGBDImage &target_dx,
            const geometry::RGBDImage &target_dy,
            const Eigen::Matrix3d &intrinsic,
            const Eigen::Matrix4d &extrinsic,
            const CorrespondenceSetPixelWise &corresps) const = 0;
};

/// \class RGBDOdometryJacobianFromColorTerm
///
/// \brief Class to compute Jacobian using color term.
///
/// Energy: (I_p-I_q)^2
/// reference:
/// F. Steinbrucker, J. Sturm, and D. Cremers.
/// Real-time visual odometry from dense RGB-D images.
/// In ICCV Workshops, 2011.
class RGBDOdometryJacobianFromColorTerm : public RGBDOdometryJacobian {
public:
    /// \brief Default Constructor.
    RGBDOdometryJacobianFromColorTerm() {}
    ~RGBDOdometryJacobianFromColorTerm() override {}

public:
    /// \brief Parameterized Constructor
    void ComputeJacobianAndResidual(
            int row,
            std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
            std::vector<double> &r,
            std::vector<double> &w,
            const geometry::RGBDImage &source,
            const geometry::RGBDImage &target,
            const geometry::Image &source_xyz,
            const geometry::RGBDImage &target_dx,
            const geometry::RGBDImage &target_dy,
            const Eigen::Matrix3d &intrinsic,
            const Eigen::Matrix4d &extrinsic,
            const CorrespondenceSetPixelWise &corresps) const override;
};

/// \class RGBDOdometryJacobianFromHybridTerm
///
/// \brief Class to compute Jacobian using hybrid term.
///
/// Energy: \f$(I_p - I_q)^2 + \lambda(D_p - (D_q)')^2\f$, where
/// \f$ I_p \f$ denotes the intensity at pixel p in the source,
/// \f$ I_q \f$ denotes the intensity at pixel q in the target.
/// \f$ D_p \f$ denotes the depth pixel p in the source,
/// \f$ D_q \f$ denotes the depth pixel q in the target.
/// q is obtained by transforming p with extrinsic then
/// projecting with intrinsics.
/// Reference: J. Park, Q.Y. Zhou, and V. Koltun,
/// Colored Point Cloud Registration Revisited, ICCV, 2017.s
class RGBDOdometryJacobianFromHybridTerm : public RGBDOdometryJacobian {
public:
    /// \brief Default Constructor.
    RGBDOdometryJacobianFromHybridTerm() {}
    ~RGBDOdometryJacobianFromHybridTerm() override {}

public:
    /// \brief Parameterized Constructor.
    void ComputeJacobianAndResidual(
            int row,
            std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
            std::vector<double> &r,
            std::vector<double> &w,
            const geometry::RGBDImage &source,
            const geometry::RGBDImage &target,
            const geometry::Image &source_xyz,
            const geometry::RGBDImage &target_dx,
            const geometry::RGBDImage &target_dy,
            const Eigen::Matrix3d &intrinsic,
            const Eigen::Matrix4d &extrinsic,
            const CorrespondenceSetPixelWise &corresps) const override;
};

}  // namespace odometry
}  // namespace pipelines
}  // namespace cloudViewer
