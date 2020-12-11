// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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
#include <memory>
#include <vector>

#include <Image.h>
#include <ecvHObject.h>

class ccMesh;
class ccPointCloud;
namespace cloudViewer {

namespace visualization {
class ViewControl;
class SelectionPolygonVolume;

/// A 2D polygon used for selection on screen
/// It is a utility class for Visualization
/// The coordinates in SelectionPolygon are lower-left corner based (the OpenGL
/// convention).
class SelectionPolygon : public ccHObject {
public:
    enum class SectionPolygonType {
        Unfilled = 0,
        Rectangle = 1,
        Polygon = 2,
    };

public:
	SelectionPolygon(const char* name = "SelectionPolygon")
		: ccHObject(name) {}
	~SelectionPolygon() override {}

	//inherited methods (ccHObject)
	virtual bool isSerializable() const override { return true; }

	//! Returns unique class ID
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::CUSTOM_H_OBJECT; }

public:
    SelectionPolygon &Clear();
    bool isEmpty() const;
    virtual Eigen::Vector2d getMin2DBound() const override;
    virtual Eigen::Vector2d getMax2DBound() const override;
    void FillPolygon(int width, int height);
    std::shared_ptr<ccPointCloud> CropPointCloud(
            const ccPointCloud &input, const ViewControl &view);
    std::shared_ptr<ccMesh> CropTriangleMesh(
            const ccMesh &input, const ViewControl &view);
    std::shared_ptr<SelectionPolygonVolume> CreateSelectionPolygonVolume(
            const ViewControl &view);

private:
    std::shared_ptr<ccPointCloud> CropPointCloudInRectangle(
            const ccPointCloud &input, const ViewControl &view);
    std::shared_ptr<ccPointCloud> CropPointCloudInPolygon(
            const ccPointCloud &input, const ViewControl &view);
    std::shared_ptr<ccMesh> CropTriangleMeshInRectangle(
            const ccMesh &input, const ViewControl &view);
    std::shared_ptr<ccMesh> CropTriangleMeshInPolygon(
            const ccMesh &input, const ViewControl &view);
    std::vector<size_t> CropInRectangle(
            const std::vector<CCVector3> &input, const ViewControl &view);
    std::vector<size_t> CropInPolygon(const std::vector<CCVector3> &input,
                                      const ViewControl &view);

public:
    std::vector<Eigen::Vector2d> polygon_;
    bool is_closed_ = false;
    geometry::Image polygon_interior_mask_;
    SectionPolygonType polygon_type_ = SectionPolygonType::Unfilled;
};

}  // namespace visualization
}  // namespace cloudViewer
