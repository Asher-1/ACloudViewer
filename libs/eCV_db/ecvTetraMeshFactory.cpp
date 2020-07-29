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
    return utility::Qhull::ComputeDelaunayTetrahedralization(
		point_cloud.getPoints());
}

std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
TetraMesh::computeConvexHull() const {
	return utility::Qhull::ComputeConvexHull(vertices_);
}

}  // namespace geometry
}  // namespace cloudViewer
