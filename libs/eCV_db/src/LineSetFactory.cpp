// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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

#include "LineSet.h"

#include <Helper.h>
#include <Eigen/Dense>
#include <unordered_set>

#include "ecvBBox.h"
#include "ecvOrientedBBox.h"
#include "ecvMesh.h"
#include "ecvTetraMesh.h"
#include "ecvPointCloud.h"

namespace cloudViewer {
namespace geometry {

std::shared_ptr<LineSet> 
LineSet::CreateFromPointCloudCorrespondences(
        const ccPointCloud &cloud0,
        const ccPointCloud &cloud1,
        const std::vector<std::pair<int, int>> &correspondences) {
    auto lineset_ptr = cloudViewer::make_shared<LineSet>();
    unsigned int point0_size = cloud0.size();
    unsigned int point1_size = cloud1.size();
    lineset_ptr->points_.resize(point0_size + point1_size);
    for (unsigned int i = 0; i < point0_size; i++)
        lineset_ptr->points_[i] = cloud0.getEigenPoint(i);
    for (unsigned int i = 0; i < point1_size; i++)
        lineset_ptr->points_[point0_size + i] = cloud1.getEigenPoint(i);

    size_t corr_size = correspondences.size();
    lineset_ptr->lines_.resize(corr_size);
    for (size_t i = 0; i < corr_size; i++)
        lineset_ptr->lines_[i] =
                Eigen::Vector2i(correspondences[i].first,
                                point0_size + correspondences[i].second);
    return lineset_ptr;
}

std::shared_ptr<LineSet> 
LineSet::CreateFromTriangleMesh(const ccMesh &mesh)
{
    auto line_set = cloudViewer::make_shared<LineSet>();
    line_set->points_ = mesh.getEigenVertices();

    std::unordered_set<Eigen::Vector2i,
                       cloudViewer::utility::hash_eigen<Eigen::Vector2i>>
            inserted_edges;
    auto InsertEdge = [&](int vidx0, int vidx1) {
        Eigen::Vector2i edge(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
        if (inserted_edges.count(edge) == 0) {
            inserted_edges.insert(edge);
            line_set->lines_.push_back(Eigen::Vector2i(vidx0, vidx1));
        }
    };

	for (unsigned int i = 0; i < mesh.size(); ++i)
	{
		const cloudViewer::VerticesIndexes* tri = mesh.getTriangleVertIndexes(i);
		InsertEdge(tri->i1, tri->i2);
		InsertEdge(tri->i2, tri->i3);
		InsertEdge(tri->i3, tri->i1);
	}

    return line_set;
}

std::shared_ptr<LineSet> 
LineSet::CreateFromOrientedBoundingBox(const ecvOrientedBBox &box) 
{
    auto line_set = cloudViewer::make_shared<LineSet>();
    line_set->points_ = box.getBoxPoints();
    line_set->lines_.push_back(Eigen::Vector2i(0, 1));
    line_set->lines_.push_back(Eigen::Vector2i(1, 7));
    line_set->lines_.push_back(Eigen::Vector2i(7, 2));
    line_set->lines_.push_back(Eigen::Vector2i(2, 0));
    line_set->lines_.push_back(Eigen::Vector2i(3, 6));
    line_set->lines_.push_back(Eigen::Vector2i(6, 4));
    line_set->lines_.push_back(Eigen::Vector2i(4, 5));
    line_set->lines_.push_back(Eigen::Vector2i(5, 3));
    line_set->lines_.push_back(Eigen::Vector2i(0, 3));
    line_set->lines_.push_back(Eigen::Vector2i(1, 6));
    line_set->lines_.push_back(Eigen::Vector2i(7, 4));
    line_set->lines_.push_back(Eigen::Vector2i(2, 5));
    line_set->paintUniformColor(box.getColor());
    return line_set;
}

std::shared_ptr<LineSet>
LineSet::CreateFromAxisAlignedBoundingBox(const ccBBox &box) 
{
    auto line_set = cloudViewer::make_shared<LineSet>();
    line_set->points_ = box.getBoxPoints();
    line_set->lines_.push_back(Eigen::Vector2i(0, 1));
    line_set->lines_.push_back(Eigen::Vector2i(1, 7));
    line_set->lines_.push_back(Eigen::Vector2i(7, 2));
    line_set->lines_.push_back(Eigen::Vector2i(2, 0));
    line_set->lines_.push_back(Eigen::Vector2i(3, 6));
    line_set->lines_.push_back(Eigen::Vector2i(6, 4));
    line_set->lines_.push_back(Eigen::Vector2i(4, 5));
    line_set->lines_.push_back(Eigen::Vector2i(5, 3));
    line_set->lines_.push_back(Eigen::Vector2i(0, 3));
    line_set->lines_.push_back(Eigen::Vector2i(1, 6));
    line_set->lines_.push_back(Eigen::Vector2i(7, 4));
    line_set->lines_.push_back(Eigen::Vector2i(2, 5));
    line_set->paintUniformColor(box.getColor());
    return line_set;
}

std::shared_ptr<LineSet> 
LineSet::CreateFromTetraMesh(const TetraMesh &mesh) {
    auto line_set = cloudViewer::make_shared<LineSet>();
    line_set->points_ = mesh.vertices_;

    std::unordered_set<Eigen::Vector2i,
                       cloudViewer::utility::hash_eigen<Eigen::Vector2i>>
            inserted_edges;
    auto InsertEdge = [&](int vidx0, int vidx1) {
        Eigen::Vector2i edge(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
        if (inserted_edges.count(edge) == 0) {
            inserted_edges.insert(edge);
            line_set->lines_.push_back(Eigen::Vector2i(vidx0, vidx1));
        }
    };
    for (const auto &tetra : mesh.tetras_) {
        InsertEdge(tetra(0), tetra(1));
        InsertEdge(tetra(1), tetra(2));
        InsertEdge(tetra(2), tetra(0));
        InsertEdge(tetra(3), tetra(0));
        InsertEdge(tetra(3), tetra(1));
        InsertEdge(tetra(3), tetra(2));
    }

    return line_set;
}

std::shared_ptr<LineSet> LineSet::CreateCameraVisualization(
        int view_width_px,
        int view_height_px,
        const Eigen::Matrix3d &intrinsic,
        const Eigen::Matrix4d &extrinsic,
        double scale) {
    Eigen::Matrix4d intrinsic4;
    intrinsic4 << intrinsic(0, 0), intrinsic(0, 1), intrinsic(0, 2), 0.0,
            intrinsic(1, 0), intrinsic(1, 1), intrinsic(1, 2), 0.0,
            intrinsic(2, 0), intrinsic(2, 1), intrinsic(2, 2), 0.0, 0.0, 0.0,
            0.0, 1.0;
    Eigen::Matrix4d m = (intrinsic4 * extrinsic).inverse();
    auto lines = cloudViewer::make_shared<geometry::LineSet>();

    auto mult = [](const Eigen::Matrix4d &m,
                   const Eigen::Vector3d &v) -> Eigen::Vector3d {
        Eigen::Vector4d v4(v.x(), v.y(), v.z(), 1.0);
        auto result = m * v4;
        return Eigen::Vector3d{result.x() / result.w(), result.y() / result.w(),
                               result.z() / result.w()};
    };
    double w = double(view_width_px);
    double h = double(view_height_px);
    // Matrix m transforms from homogenous pixel coordinates to world
    // coordinates so x and y need to be multiplied by z. In the case of the
    // first point, the eye point, z=0, so x and y will be zero, too regardless
    // of their initial values as the center.
    lines->points_.push_back(mult(m, Eigen::Vector3d{0.0, 0.0, 0.0}));
    lines->points_.push_back(mult(m, Eigen::Vector3d{0.0, 0.0, scale}));
    lines->points_.push_back(mult(m, Eigen::Vector3d{w * scale, 0.0, scale}));
    lines->points_.push_back(
            mult(m, Eigen::Vector3d{w * scale, h * scale, scale}));
    lines->points_.push_back(mult(m, Eigen::Vector3d{0.0, h * scale, scale}));

    lines->lines_.push_back({0, 1});
    lines->lines_.push_back({0, 2});
    lines->lines_.push_back({0, 3});
    lines->lines_.push_back({0, 4});
    lines->lines_.push_back({1, 2});
    lines->lines_.push_back({2, 3});
    lines->lines_.push_back({3, 4});
    lines->lines_.push_back({4, 1});
    lines->paintUniformColor({0.0f, 0.0f, 1.0f});

    return lines;
}

}  // namespace geometry
}  // namespace cloudViewer
