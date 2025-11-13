// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVGeom.h>
#include <IJsonConvertible.h>

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

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
    std::shared_ptr<ccMesh> CropTriangleMesh(const ccMesh &input) const;
    /// Function to crop point cloud with polygon boundaries
    ///
    /// \param input The input point Cloud.
    std::vector<size_t> CropInPolygon(const ccPointCloud &input) const;

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
