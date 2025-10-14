// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// Transform ellipsoidal GPS coordinates to Cartesian GPS coordinate
// representation and vice versa.
class GPSTransform {
public:
    enum ELLIPSOID { GRS80, WGS84 };

    explicit GPSTransform(const int ellipsoid = GRS80);

    std::vector<Eigen::Vector3d> EllToXYZ(
            const std::vector<Eigen::Vector3d>& ell) const;

    std::vector<Eigen::Vector3d> XYZToEll(
            const std::vector<Eigen::Vector3d>& xyz) const;

private:
    // Semimajor axis.
    double a_;
    // Semiminor axis.
    double b_;
    // Flattening.
    double f_;
    // Numerical eccentricity.
    double e2_;
};

}  // namespace colmap
