// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.erow.cn
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

#include "visualization/rendering/filament/FilamentGeometryBuffersBuilder.h"

#include <ecvBBox.h>
#include <ecvOrientedBBox.h>
#include <LineSet.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <VoxelGrid.h>
#include <Octree.h>

#include <t/geometry/PointCloud.h>

namespace cloudViewer {
namespace visualization {
namespace rendering {

namespace {
static const Eigen::Vector3d kDefaultVoxelColor(0.5, 0.5, 0.5);

// Coordinates of 8 vertices in a cuboid (assume origin (0,0,0), size 1)
const static std::vector<Eigen::Vector3i> kCuboidVertexOffsets{
        Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(1, 1, 0),
        Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(1, 0, 1),
        Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(1, 1, 1),
};

// Vertex indices of 12 triangles in a cuboid, for right-handed manifold mesh
const static std::vector<Eigen::Vector3i> kCuboidTrianglesVertexIndices{
        Eigen::Vector3i(0, 2, 1), Eigen::Vector3i(0, 1, 4),
        Eigen::Vector3i(0, 4, 2), Eigen::Vector3i(5, 1, 7),
        Eigen::Vector3i(5, 7, 4), Eigen::Vector3i(5, 4, 1),
        Eigen::Vector3i(3, 7, 1), Eigen::Vector3i(3, 1, 2),
        Eigen::Vector3i(3, 2, 7), Eigen::Vector3i(6, 4, 7),
        Eigen::Vector3i(6, 7, 2), Eigen::Vector3i(6, 2, 4),
};

// Vertex indices of 12 lines in a cuboid
const static std::vector<Eigen::Vector2i> kCuboidLinesVertexIndices{
        Eigen::Vector2i(0, 1), Eigen::Vector2i(0, 2), Eigen::Vector2i(0, 4),
        Eigen::Vector2i(3, 1), Eigen::Vector2i(3, 2), Eigen::Vector2i(3, 7),
        Eigen::Vector2i(5, 1), Eigen::Vector2i(5, 4), Eigen::Vector2i(5, 7),
        Eigen::Vector2i(6, 2), Eigen::Vector2i(6, 4), Eigen::Vector2i(6, 7),
};

static void AddVoxelFaces(ccMesh& mesh,
                          const std::vector<Eigen::Vector3d>& vertices,
                          const Eigen::Vector3d& color) {
    for (const Eigen::Vector3i& triangle_vertex_indices : kCuboidTrianglesVertexIndices) {
        int n = int(mesh.getVerticeSize());
        mesh.addTriangle(Eigen::Vector3i(n, n + 1, n + 2));
        mesh.addVertice(vertices[triangle_vertex_indices(0)]);
        mesh.addVertice(vertices[triangle_vertex_indices(1)]);
        mesh.addVertice(vertices[triangle_vertex_indices(2)]);
        mesh.addVertexColor(color);
        mesh.addVertexColor(color);
        mesh.addVertexColor(color);
    }
}

static void AddLineFace(ccMesh& mesh,
                        const Eigen::Vector3d& start,
                        const Eigen::Vector3d& end,
                        const Eigen::Vector3d& half_width,
                        const Eigen::Vector3d& color) {
    int n = int(mesh.getVerticeSize());
    mesh.addTriangle(Eigen::Vector3i(n, n + 1, n + 3));
    mesh.addTriangle(Eigen::Vector3i(n, n + 3, n + 2));
    mesh.addVertice(start - half_width);
    mesh.addVertice(start + half_width);
    mesh.addVertice(end - half_width);
    mesh.addVertice(end + half_width);
    mesh.addVertexColor(color);
    mesh.addVertexColor(color);
    mesh.addVertexColor(color);
    mesh.addVertexColor(color);
}

static std::shared_ptr<ccMesh> CreateTriangleMeshFromVoxelGrid(
        const geometry::VoxelGrid& voxel_grid) {
    ccPointCloud* baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh = std::make_shared<ccMesh>(baseVertices);
    auto num_voxels = voxel_grid.voxels_.size();
    if (!baseVertices->reserve(static_cast<unsigned>(36 * num_voxels)))
    {
        utility::LogError("not enough memory!");
    }
    if (!baseVertices->reserveTheRGBTable())
    {
        utility::LogError("not enough memory!");
    }

    std::vector<Eigen::Vector3d>
            vertices;  // putting outside loop enables reuse
    for (auto& it : voxel_grid.voxels_) {
        vertices.clear();
        const geometry::Voxel& voxel = it.second;
        // 8 vertices in a voxel
        Eigen::Vector3d base_vertex =
                voxel_grid.origin_ +
                voxel.grid_index_.cast<double>() * voxel_grid.voxel_size_;
        for (const Eigen::Vector3i& vertex_offset : kCuboidVertexOffsets) {
            vertices.push_back(base_vertex + vertex_offset.cast<double>() *
                                                     voxel_grid.voxel_size_);
        }

        // Voxel color (applied to all points)
        Eigen::Vector3d voxel_color;
        if (voxel_grid.HasColors()) {
            voxel_color = voxel.color_;
        } else {
            voxel_color = kDefaultVoxelColor;
        }

        AddVoxelFaces(*mesh, vertices, voxel_color);
    }

    //do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType* normals = mesh->getTriNormsTable();
        if (normals)
        {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    mesh->addChild(baseVertices);

    return mesh;
}

static std::shared_ptr<ccMesh> CreateTriangleMeshFromOctree(
        const geometry::Octree& octree) {
    auto mesh = std::make_shared<ccMesh>();

    // We cannot have a real line with a width in pixels, we can only fake a
    // line as rectangles. This value works nicely on the assumption that the
    // octree fills about 80% of the viewing area.
    double line_half_width = 0.0015 * octree.size_;

    auto f = [&mesh = *mesh, line_half_width](
                     const std::shared_ptr<geometry::OctreeNode>& node,
                     const std::shared_ptr<geometry::OctreeNodeInfo>& node_info)
            -> bool {
        Eigen::Vector3d base_vertex = node_info->origin_.cast<double>();
        std::vector<Eigen::Vector3d> vertices;
        for (const Eigen::Vector3i& vertex_offset : kCuboidVertexOffsets) {
            vertices.push_back(base_vertex + vertex_offset.cast<double>() *
                                                     double(node_info->size_));
        }

        auto leaf_node =
                std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(node);
        if (leaf_node) {
            AddVoxelFaces(mesh, vertices, leaf_node->color_);
        } else {
            // We cannot have lines in a TriangleMesh, obviously, so fake them
            // with two crossing planes.
            for (const Eigen::Vector2i& line_vertex_indices :
                 kCuboidLinesVertexIndices) {
                auto& start = vertices[line_vertex_indices(0)];
                auto& end = vertices[line_vertex_indices(1)];
                Eigen::Vector3d w(line_half_width, 0.0, 0.0);
                // if (end - start).dot({1, 0, 0}) ~= 0, then use z, not x
                if (std::abs(end.y() - start.y()) < 0.1 &&
                    std::abs(end.z() - start.z()) < 0.1) {
                    w = {0.0, 0.0, line_half_width};
                }
                AddLineFace(mesh, start, end, w, {0.0, 0.0, 0.0});

                w = {0.0, line_half_width, 0.0};
                // if (end - start).dot({0, 1, 0}) ~= 0, then use z, not y
                if (std::abs(end.x() - start.x()) < 0.1 &&
                    std::abs(end.z() - start.z()) < 0.1) {
                    w = {0.0, 0.0, line_half_width};
                }
                AddLineFace(mesh, start, end, w, {0.0, 0.0, 0.0});
            }
        }

        return false;
    };

    octree.Traverse(f);

    return mesh;
}

}  // namespace

class TemporaryLineSetBuilder : public LineSetBuffersBuilder {
public:
    explicit TemporaryLineSetBuilder(std::shared_ptr<geometry::LineSet> lines)
        : LineSetBuffersBuilder(*lines), lines_(lines) {}

private:
    std::shared_ptr<geometry::LineSet> lines_;
};

class TemporaryMeshBuilder : public TriangleMeshBuffersBuilder {
public:
    explicit TemporaryMeshBuilder(std::shared_ptr<ccMesh> mesh)
        : TriangleMeshBuffersBuilder(*mesh), mesh_(mesh) {}

private:
    std::shared_ptr<ccMesh> mesh_;
};

std::unique_ptr<GeometryBuffersBuilder> GeometryBuffersBuilder::GetBuilder(
        const ccHObject& geometry) {
    using GT = CV_TYPES::GeometryType;

    switch (geometry.getClassID()) {
        case GT::MESH:
            return std::make_unique<TriangleMeshBuffersBuilder>(
                    static_cast<const ccMesh&>(geometry));

        case GT::POINT_CLOUD:
            return std::make_unique<PointCloudBuffersBuilder>(
                    static_cast<const ccPointCloud&>(geometry));

        case GT::LINESET:
            return std::make_unique<LineSetBuffersBuilder>(
                    static_cast<const geometry::LineSet&>(geometry));
        case GT::ORIENTED_BBOX: {
            auto obb =
                    static_cast<const ecvOrientedBBox&>(geometry);
            auto lines = geometry::LineSet::CreateFromOrientedBoundingBox(obb);
            lines->paintUniformColor(obb.color_);
            return std::make_unique<TemporaryLineSetBuilder>(lines);
        }
        case GT::BBOX: {
            auto aabb = static_cast<const ccBBox&>(
                    geometry);
            auto lines =
                    geometry::LineSet::CreateFromAxisAlignedBoundingBox(aabb);
            lines->paintUniformColor(aabb.getColor());
            return std::make_unique<TemporaryLineSetBuilder>(lines);
        }
        case GT::VOXEL_GRID: {
            auto voxel_grid = static_cast<const geometry::VoxelGrid&>(geometry);
            auto mesh = CreateTriangleMeshFromVoxelGrid(voxel_grid);
            return std::make_unique<TemporaryMeshBuilder>(mesh);
        }
        case GT::POINT_OCTREE2: {
            auto octree = static_cast<const geometry::Octree&>(geometry);
            auto mesh = CreateTriangleMeshFromOctree(octree);
            return std::make_unique<TemporaryMeshBuilder>(mesh);
        }
        default:
            break;
    }

    return nullptr;
}

std::unique_ptr<GeometryBuffersBuilder> GeometryBuffersBuilder::GetBuilder(
        const t::geometry::PointCloud& geometry) {
    return std::make_unique<TPointCloudBuffersBuilder>(geometry);
}

void GeometryBuffersBuilder::DeallocateBuffer(void* buffer,
                                              size_t size,
                                              void* user_ptr) {
    free(buffer);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
