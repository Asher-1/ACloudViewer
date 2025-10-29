// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Image.h>
#include <ecvHObject.h>

#include <Eigen/Core>
#include <memory>
#include <vector>

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
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    enum class SectionPolygonType {
        Unfilled = 0,
        Rectangle = 1,
        Polygon = 2,
    };

public:
    SelectionPolygon(const char *name = "SelectionPolygon") : ccHObject(name) {}
    ~SelectionPolygon() override {}

    // inherited methods (ccHObject)
    virtual bool isSerializable() const override { return true; }

    //! Returns unique class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::CUSTOM_H_OBJECT;
    }

public:
    SelectionPolygon &Clear();
    bool isEmpty() const;
    virtual Eigen::Vector2d GetMin2DBound() const override;
    virtual Eigen::Vector2d GetMax2DBound() const override;
    void FillPolygon(int width, int height);
    std::shared_ptr<ccPointCloud> CropPointCloud(const ccPointCloud &input,
                                                 const ViewControl &view);
    std::shared_ptr<ccMesh> CropTriangleMesh(const ccMesh &input,
                                             const ViewControl &view);
    std::shared_ptr<SelectionPolygonVolume> CreateSelectionPolygonVolume(
            const ViewControl &view);

private:
    std::shared_ptr<ccPointCloud> CropPointCloudInRectangle(
            const ccPointCloud &input, const ViewControl &view);
    std::shared_ptr<ccPointCloud> CropPointCloudInPolygon(
            const ccPointCloud &input, const ViewControl &view);
    std::shared_ptr<ccMesh> CropTriangleMeshInRectangle(
            const ccMesh &input, const ViewControl &view);
    std::shared_ptr<ccMesh> CropTriangleMeshInPolygon(const ccMesh &input,
                                                      const ViewControl &view);
    std::vector<size_t> CropInRectangle(const std::vector<CCVector3> &input,
                                        const ViewControl &view);
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
