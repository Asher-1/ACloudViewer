// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OrientedBoundingBox.h"

#include "CVLog.h"

// EIGEN
#include <Eigen/Eigenvalues>

// STL
#include <algorithm>
#include <cstdint>
#include <numeric>

using namespace cloudViewer;

OrientedBoundingBox& OrientedBoundingBox::Clear() {
    center_.setZero();
    extent_.setZero();
    R_ = Eigen::Matrix3d::Identity();
    color_.setZero();
    return *this;
}

double OrientedBoundingBox::volume() const {
    return extent_(0) * extent_(1) * extent_(2);
}

std::vector<Eigen::Vector3d> OrientedBoundingBox::GetBoxPoints() const {
    Eigen::Vector3d x_axis = R_ * Eigen::Vector3d(extent_(0) / 2, 0, 0);
    Eigen::Vector3d y_axis = R_ * Eigen::Vector3d(0, extent_(1) / 2, 0);
    Eigen::Vector3d z_axis = R_ * Eigen::Vector3d(0, 0, extent_(2) / 2);
    std::vector<Eigen::Vector3d> points(8);
    points[0] = center_ - x_axis - y_axis - z_axis;
    points[1] = center_ + x_axis - y_axis - z_axis;
    points[2] = center_ - x_axis + y_axis - z_axis;
    points[3] = center_ - x_axis - y_axis + z_axis;
    points[4] = center_ + x_axis + y_axis + z_axis;
    points[5] = center_ - x_axis + y_axis + z_axis;
    points[6] = center_ + x_axis - y_axis + z_axis;
    points[7] = center_ + x_axis + y_axis - z_axis;
    return points;
}

std::vector<size_t> OrientedBoundingBox::GetPointIndicesWithinBoundingBox(
        const std::vector<Eigen::Vector3d>& points) const {
    return GetPointIndicesWithinBoundingBox(
            CCVector3::fromArrayContainer(points));
}

std::vector<size_t> OrientedBoundingBox::GetPointIndicesWithinBoundingBox(
        const std::vector<CCVector3>& points) const {
    std::vector<size_t> indices;
    Eigen::Vector3d dx = R_ * Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d dy = R_ * Eigen::Vector3d(0, 1, 0);
    Eigen::Vector3d dz = R_ * Eigen::Vector3d(0, 0, 1);
    Eigen::Vector3d halfExtent = GetHalfExtent();
    for (size_t idx = 0; idx < points.size(); idx++) {
        Eigen::Vector3d d = CCVector3d::fromArray(points[idx]) - center_;
        if (std::abs(d.dot(dx)) <= halfExtent(0) &&
            std::abs(d.dot(dy)) <= halfExtent(1) &&
            std::abs(d.dot(dz)) <= halfExtent(2)) {
            indices.push_back(idx);
        }
    }
    return indices;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
        const BoundingBox& aabox) {
    OrientedBoundingBox obox;
    obox.center_ = CCVector3d::fromArray(aabox.getCenter());
    obox.extent_ = CCVector3d::fromArray(aabox.getDiagVec());
    obox.R_ = Eigen::Matrix3d::Identity();
    return obox;
}
