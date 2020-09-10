// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#include "Visualization/Utility/PointCloudPicker.h"

#include <ecvBBox.h>
#include <ecvOrientedBBox.h>
#include <ecvPointCloud.h>
#include <Console.h>

namespace cloudViewer {
namespace visualization {

PointCloudPicker& PointCloudPicker::Clear() {
    picked_indices_.clear();
    return *this;
}

bool PointCloudPicker::isEmpty() const {
    return (!pointcloud_ptr_ || picked_indices_.empty());
}

Eigen::Vector3d PointCloudPicker::getMinBound() const {
    if (pointcloud_ptr_) {
        return ((const ccPointCloud&)(*pointcloud_ptr_)).getMinBound();
    } else {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
}

Eigen::Vector3d PointCloudPicker::getMaxBound() const {
    if (pointcloud_ptr_) {
        return ((const ccPointCloud&)(*pointcloud_ptr_)).getMaxBound();
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
