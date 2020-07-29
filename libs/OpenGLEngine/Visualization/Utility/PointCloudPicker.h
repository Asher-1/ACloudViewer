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

#pragma once
#include "qGL.h"

#include <Eigen/Core>
#include <memory>
#include <vector>

#include <ecvHObject.h>

class ccBBox;
class ecvOrientedBBox;

namespace cloudViewer {

namespace geometry {
class PointCloud;
}
namespace visualization {

/// A utility class to store picked points of a pointcloud
class OPENGL_ENGINE_LIB_API PointCloudPicker : public ccHObject {
public:
    PointCloudPicker(const char* name = "PointCloudPicker")
        : ccHObject(name) {}
    ~PointCloudPicker() override {}

	//inherited methods (ccHObject)
	virtual bool isSerializable() const override { return true; }

	//! Returns unique class ID
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::CUSTOM_H_OBJECT; }

public:
    PointCloudPicker& Clear();
    bool isEmpty() const;
	virtual Eigen::Vector3d getMinBound() const override;
	virtual Eigen::Vector3d getMaxBound() const override;
	virtual Eigen::Vector3d getGeometryCenter() const override;
	virtual ccBBox getAxisAlignedBoundingBox() const override;
	virtual ecvOrientedBBox getOrientedBoundingBox() const override;
    virtual PointCloudPicker& transform(const Eigen::Matrix4d& transformation) override;
    virtual PointCloudPicker& translate(const Eigen::Vector3d& translation, 
		bool relative = true) override;
    virtual PointCloudPicker& scale(const double s, const Eigen::Vector3d &center) override;
    virtual PointCloudPicker& rotate(const Eigen::Matrix3d& R, const Eigen::Vector3d &center) override;
    bool SetPointCloud(std::shared_ptr<const ccHObject> ptr);

public:
    std::shared_ptr<const ccHObject> pointcloud_ptr_;
    std::vector<size_t> picked_indices_;
};

}  // namespace visualization
}  // namespace cloudViewer
