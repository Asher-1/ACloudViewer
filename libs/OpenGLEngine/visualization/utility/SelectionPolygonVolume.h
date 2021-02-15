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
#include <string>
#include <vector>

#include <CVGeom.h>
#include <IJsonConvertible.h>

class ccMesh;
class ccPointCloud;

namespace cloudViewer {

namespace visualization {

/// \class SelectionPolygonVolume
///
/// \brief Select a polygon volume for cropping.
class SelectionPolygonVolume : public cloudViewer::utility::IJsonConvertible {
public:
    ~SelectionPolygonVolume() override {}

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;
    /// Function to crop point cloud.
    ///
    /// \param input The input point cloud.
    std::shared_ptr<ccPointCloud> CropPointCloud(
            const ccPointCloud &input) const;
    /// Function to crop crop triangle mesh.
    ///
    /// \param input The input triangle mesh.
    std::shared_ptr<ccMesh> CropTriangleMesh(
            const ccMesh &input) const;

private:
    std::shared_ptr<ccPointCloud> CropPointCloudInPolygon(
            const ccPointCloud &input) const;
    std::shared_ptr<ccMesh> CropTriangleMeshInPolygon(
            const ccMesh &input) const;
    std::vector<size_t> CropInPolygon(
            const std::vector<Eigen::Vector3d> &input) const;
    std::vector<size_t> CropInPolygon(
            const std::vector<CCVector3> &input) const;

public:
    /// One of `{x, y, z}`.
    std::string orthogonal_axis_ = "";
    /// Bounding polygon boundary.
    std::vector<Eigen::Vector3d> bounding_polygon_;
    /// Minimum axis value.
    double axis_min_ = 0.0;
    /// Maximum axis value.
    double axis_max_ = 0.0;
};

}  // namespace visualization
}  // namespace cloudViewer
