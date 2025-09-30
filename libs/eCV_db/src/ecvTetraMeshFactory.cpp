// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>

#include "ecvPointCloud.h"
#include "ecvQhull.h"
#include "ecvTetraMesh.h"

namespace cloudViewer {
namespace geometry {

std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
TetraMesh::CreateFromPointCloud(const ccPointCloud& point_cloud) {
    if (point_cloud.size() < 4) {
        CVLog::Error(
                "[CreateFromPointCloud] not enough points to create a "
                "tetrahedral mesh.");
    }
    return Qhull::ComputeDelaunayTetrahedralization(point_cloud.getPoints());
}
}  // namespace geometry
}  // namespace cloudViewer
