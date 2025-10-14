// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

class ccMesh;

namespace cloudViewer {
namespace geometry {

class TetraMesh;

class Qhull {
public:
    static std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
    ComputeConvexHull(const std::vector<Eigen::Vector3d>& points);

    static std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
    ComputeConvexHull(const std::vector<CCVector3>& points);

    static std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
    ComputeDelaunayTetrahedralization(
            const std::vector<Eigen::Vector3d>& points);

    static std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
    ComputeDelaunayTetrahedralization(const std::vector<CCVector3>& points);
};

}  // namespace geometry
}  // namespace cloudViewer
