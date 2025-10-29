// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvHObject.h>

#include <Eigen/Core>
#include <memory>
#include <vector>

class ccBBox;
class ecvOrientedBBox;

namespace cloudViewer {
namespace visualization {

/// A utility class to store picked points of a pointcloud
class PointCloudPicker : public ccHObject {
public:
    PointCloudPicker(const char* name = "PointCloudPicker") : ccHObject(name) {}
    ~PointCloudPicker() override {}

    // inherited methods (ccHObject)
    virtual bool isSerializable() const override { return true; }

    //! Returns unique class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::CUSTOM_H_OBJECT;
    }

public:
    PointCloudPicker& Clear();
    bool isEmpty() const;
    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;
    virtual ccBBox GetAxisAlignedBoundingBox() const override;
    virtual ecvOrientedBBox GetOrientedBoundingBox() const override;
    virtual PointCloudPicker& Transform(
            const Eigen::Matrix4d& transformation) override;
    virtual PointCloudPicker& Translate(const Eigen::Vector3d& translation,
                                        bool relative = true) override;
    virtual PointCloudPicker& Scale(const double s,
                                    const Eigen::Vector3d& center) override;
    virtual PointCloudPicker& Rotate(const Eigen::Matrix3d& R,
                                     const Eigen::Vector3d& center) override;
    bool SetPointCloud(std::shared_ptr<const ccHObject> ptr);

public:
    std::shared_ptr<const ccHObject> pointcloud_ptr_;
    std::vector<size_t> picked_indices_;
};

}  // namespace visualization
}  // namespace cloudViewer
