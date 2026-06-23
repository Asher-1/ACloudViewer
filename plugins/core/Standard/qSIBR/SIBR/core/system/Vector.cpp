// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/system/Vector.hpp"

#include <cmath>

#include "core/system/Quaternion.hpp"

namespace sibr {

Vector3f toColorFloat(Vector3ub& colorUB) {
    return colorUB.cast<float>().unaryExpr(
            [](unsigned char c) { return (float)c / 255.0f; });
}

Vector3ub toColorUB(Vector3f& colorFloat) {
    return colorFloat.unaryExpr([](float f) { return std::floor(f * 255.0f); })
            .cast<unsigned char>();
}

Eigen::Matrix<float, 4, 4, Eigen::DontAlign> alignRotationMatrix(
        const sibr::Vector3f& from, const sibr::Vector3f& to) {
    sibr::Quaternionf q = sibr::Quaternionf::FromTwoVectors(from, to);
    q.normalize();
    Eigen::Matrix3f R = q.toRotationMatrix();
    sibr::Matrix4f R4;
    R4.setIdentity();
    R4.block<3, 3>(0, 0) = R;
    return R4;
}

}  // namespace sibr
