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

    // Upstream COLMAP ENU API (dbb41680). EllToXYZ/XYZToEll above are the
    // fork-legacy spellings of EllipsoidToECEF/ECEFToEllipsoid and remain the
    // single implementation source for the ECEF conversion below.
    // Convert ellipsoidal (lat/lon/alt) to ENU coordinates.
    // The reference point (ref_lat, ref_lon, ref_alt) defines the ENU origin.
    std::vector<Eigen::Vector3d> EllipsoidToENU(
            const std::vector<Eigen::Vector3d>& lat_lon_alt,
            double ref_lat,
            double ref_lon,
            double ref_alt) const;

    // Convert ECEF to ENU coordinates.
    // The reference point (ref_ecef) defines the ENU origin.
    std::vector<Eigen::Vector3d> ECEFToENU(
            const std::vector<Eigen::Vector3d>& xyz_in_ecef,
            const Eigen::Vector3d& ref_ecef) const;

    // Convert ENU to ellipsoidal (lat/lon/alt) coordinates.
    // The reference point (ref_lat, ref_lon, ref_alt) defines the ENU origin.
    std::vector<Eigen::Vector3d> ENUToEllipsoid(
            const std::vector<Eigen::Vector3d>& xyz_in_enu,
            double ref_lat,
            double ref_lon,
            double ref_alt) const;

    // Convert ENU to ECEF coordinates.
    // The reference point (ref_lat, ref_lon, ref_alt) defines the ENU origin.
    std::vector<Eigen::Vector3d> ENUToECEF(
            const std::vector<Eigen::Vector3d>& xyz_in_enu,
            double ref_lat,
            double ref_lon,
            double ref_alt) const;

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
