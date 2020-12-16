// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                            -
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

#include "t/geometry/PointCloud.h"

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "core/EigenConverter.h"
#include "core/ShapeUtil.h"
#include "core/Tensor.h"
#include "core/TensorList.h"

namespace cloudViewer {
namespace t {
namespace geometry {

PointCloud::PointCloud(core::Dtype dtype, const core::Device &device)
    : Geometry(Geometry::GeometryType::PointCloud, 3),
      device_(device),
      point_attr_(TensorListMap("points")) {
    SetPoints(core::TensorList({3}, dtype, device_));
}

PointCloud::PointCloud(const core::TensorList &points)
    : PointCloud(points.GetDtype(), points.GetDevice()) {
    points.AssertElementShape({3});
    SetPoints(points);
}

PointCloud::PointCloud(const std::unordered_map<std::string, core::TensorList>
                               &map_keys_to_tensorlists)
    : PointCloud(map_keys_to_tensorlists.at("points").GetDtype(),
                 map_keys_to_tensorlists.at("points").GetDevice()) {
    map_keys_to_tensorlists.at("points").AssertElementShape({3});
    point_attr_.Assign(map_keys_to_tensorlists);
}

core::Tensor PointCloud::GetMinBound() const {
    return GetPoints().AsTensor().Min({0});
}

core::Tensor PointCloud::GetMaxBound() const {
    return GetPoints().AsTensor().Max({0});
}

core::Tensor PointCloud::GetCenter() const {
    return GetPoints().AsTensor().Mean({0});
}

PointCloud &PointCloud::Transform(const core::Tensor &transformation) {
    utility::LogError("Unimplemented");
    return *this;
}

PointCloud &PointCloud::Translate(const core::Tensor &translation,
                                  bool relative) {
    translation.AssertShape({3});
    core::Tensor transform = translation.Copy();
    if (!relative) {
        transform -= GetCenter();
    }
    GetPoints().AsTensor() += transform;
    return *this;
}

PointCloud &PointCloud::Scale(double scale, const core::Tensor &center) {
    center.AssertShape({3});
    core::Tensor points = GetPoints().AsTensor();
    points.Sub_(center).Mul_(scale).Add_(center);
    return *this;
}

PointCloud &PointCloud::Rotate(const core::Tensor &R,
                               const core::Tensor &center) {
    utility::LogError("Unimplemented");
    return *this;
}

geometry::PointCloud PointCloud::FromLegacyPointCloud(
        const ccPointCloud &pcd_legacy,
        core::Dtype dtype,
        const core::Device &device) {
    geometry::PointCloud pcd(dtype, device);
    if (pcd_legacy.hasPoints()) {
        pcd.SetPoints(core::eigen_converter::EigenVector3dVectorToTensorList(
                pcd_legacy.getEigenPoints(), dtype, device));
    } else {
        utility::LogWarning(
                "Creating from an empty legacy pointcloud, an empty pointcloud "
                "with default dtype and device will be created.");
    }
    if (pcd_legacy.hasColors()) {
        pcd.SetPointColors(
                core::eigen_converter::EigenVector3dVectorToTensorList(
                        pcd_legacy.getEigenColors(), dtype, device));
    }
    if (pcd_legacy.hasNormals()) {
        pcd.SetPointNormals(
                core::eigen_converter::EigenVector3dVectorToTensorList(
                        pcd_legacy.getEigenNormals(), dtype, device));
    }
    return pcd;
}

ccPointCloud PointCloud::ToLegacyPointCloud() const {
   ccPointCloud pcd_legacy;
    if (HasPoints()) {
        const core::TensorList &points = GetPoints();
        for (int64_t i = 0; i < points.GetSize(); i++) {
            pcd_legacy.addEigenPoint(
                    core::eigen_converter::TensorToEigenVector3d(points[i]));
        }
    }
    if (HasPointColors()) {
        const core::TensorList &colors = GetPointColors();
        for (int64_t i = 0; i < colors.GetSize(); i++) {
            pcd_legacy.addEigenColor(
                    core::eigen_converter::TensorToEigenVector3d(colors[i]));
        }
    }
    if (HasPointNormals()) {
        const core::TensorList &normals = GetPointNormals();
        for (int64_t i = 0; i < normals.GetSize(); i++) {
            pcd_legacy.addEigenNorm(
                    core::eigen_converter::TensorToEigenVector3d(normals[i]));
        }
    }
    return pcd_legacy;
}

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
