// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "visualization/utility/PointCloudPicker.h"

#include <Logging.h>
#include <ecvBBox.h>
#include <ecvPointCloud.h>
#include <ecvOrientedBBox.h>

namespace cloudViewer {
namespace visualization {

PointCloudPicker& PointCloudPicker::Clear() {
    picked_indices_.clear();
    return *this;
}

bool PointCloudPicker::isEmpty() const {
    return (!pointcloud_ptr_ || picked_indices_.empty());
}

Eigen::Vector3d PointCloudPicker::GetMinBound() const {
    if (pointcloud_ptr_) {
        return ((const ccPointCloud&)(*pointcloud_ptr_)).GetMinBound();
    } else {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
}

Eigen::Vector3d PointCloudPicker::GetMaxBound() const {
    if (pointcloud_ptr_) {
        return ((const ccPointCloud&)(*pointcloud_ptr_)).GetMaxBound();
    } else {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
}

Eigen::Vector3d PointCloudPicker::getGeometryCenter() const {
    if (pointcloud_ptr_) {
        return ((const ccPointCloud&)(*pointcloud_ptr_)).getGeometryCenter();
    } else {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
}

ccBBox PointCloudPicker::getAxisAlignedBoundingBox()
        const {
    if (pointcloud_ptr_) {
        return ccBBox::CreateFromPoints(
                ((const ccPointCloud&)(*pointcloud_ptr_)).getPoints());
    } else {
        return ccBBox();
    }
}

ecvOrientedBBox PointCloudPicker::getOrientedBoundingBox() const {
    if (pointcloud_ptr_) {
        return ecvOrientedBBox::CreateFromPoints(
                ((const ccPointCloud&)(*pointcloud_ptr_)).getPoints());
    } else {
        return ecvOrientedBBox();
    }
}

PointCloudPicker& PointCloudPicker::transform(
        const Eigen::Matrix4d& /*transformation*/) {
    // Do nothing
    return *this;
}

PointCloudPicker& PointCloudPicker::translate(
        const Eigen::Vector3d& translation, bool relative) {
    // Do nothing
    return *this;
}

PointCloudPicker& PointCloudPicker::scale(const double s, 
										  const Eigen::Vector3d &center) {
    // Do nothing
    return *this;
}

PointCloudPicker& PointCloudPicker::rotate(const Eigen::Matrix3d& R,
										   const Eigen::Vector3d &center) {
    // Do nothing
    return *this;
}

bool PointCloudPicker::SetPointCloud(
        std::shared_ptr<const ccHObject> ptr) {
    if (!ptr || !ptr->isKindOf(CV_TYPES::POINT_CLOUD)) {
        return false;
    }
    pointcloud_ptr_ = ptr;
    return true;
}

}  // namespace visualization
}  // namespace cloudViewer
