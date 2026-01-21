// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <unordered_map>

// LOCAL
#include "ecvMeshBase.h"

namespace cloudViewer {
namespace geometry {

/// \class HalfEdgeTriangleMesh
///
/// \brief HalfEdgeTriangleMesh inherits TriangleMesh class with the addition of
/// HalfEdge data structure for each half edge in the mesh as well as related
/// functions.
class CV_DB_LIB_API HalfEdgeTriangleMesh : public ecvMeshBase {
public:
    /// \class HalfEdge
    ///
    /// \brief HalfEdge class contains vertex, triangle info about a half edge,
    /// as well as relations of next and twin half edge.
    class HalfEdge {
    public:
        /// \brief Default Constructor.
        ///
        /// Initializes all members of the instance with invalid values.
        HalfEdge()
            : next_(-1),
              twin_(-1),
              vertex_indices_(-1, -1),
              triangle_index_(-1) {}
        HalfEdge(const Eigen::Vector2i &vertex_indices,
                 int triangle_index,
                 int next,
                 int twin);
        /// Returns `true` iff the half edge is the boundary (has not twin, i.e.
        /// twin index == -1).
        bool IsBoundary() const { return twin_ == -1; }

    public:
        /// Index of the next HalfEdge in the same triangle.
        int next_;
        /// Index of the twin HalfEdge.
        int twin_;
        /// Index of the ordered vertices forming this half edge.
        Eigen::Vector2i vertex_indices_;
        /// Index of the triangle containing this half edge.
        int triangle_index_;
    };

public:
    /// \brief Default Constructor.
    ///
    /// Creates an empty instance with GeometryType of HalfEdgeTriangleMesh.
    HalfEdgeTriangleMesh(const char *name = "HalfEdgeTriangleMesh")
        : ecvMeshBase(name) {}

    ~HalfEdgeTriangleMesh() override {}

    // inherited methods (ccHObject)
    virtual bool isSerializable() const override { return true; }
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::HALF_EDGE_MESH;
    }

public:
    virtual HalfEdgeTriangleMesh &clear() override;

    inline std::size_t edgeSize() const { return half_edges_.size(); }

    /// Returns `true` if the mesh contains triangles.
    virtual bool hasTriangles() const override {
        return vertices_.size() > 0 && triangles_.size() > 0;
    }

    /// Returns `true` if the mesh contains triangle normals.
    bool HasTriangleNormals() const {
        return hasTriangles() && triangles_.size() == triangle_normals_.size();
    }

    /// Returns `true` if half-edges have already been computed.
    bool hasHalfEdges() const;

    /// Query manifold boundary half edges from a starting vertex
    /// If query vertex is not on boundary, empty vector will be returned.
    std::vector<int> boundaryHalfEdgesFromVertex(int vertex_index) const;

    /// Query manifold boundary vertices from a starting vertex
    /// If query vertex is not on boundary, empty vector will be returned.
    std::vector<int> boundaryVerticesFromVertex(int vertex_index) const;

    /// Returns a vector of boundaries. A boundary is a vector of vertices.
    std::vector<std::vector<int>> getBoundaries() const;

    HalfEdgeTriangleMesh &operator+=(const HalfEdgeTriangleMesh &mesh);

    HalfEdgeTriangleMesh operator+(const HalfEdgeTriangleMesh &mesh) const;

    /// Convert HalfEdgeTriangleMesh from TriangleMesh. Throws exception if the
    /// input mesh is not manifold.
    static std::shared_ptr<HalfEdgeTriangleMesh> CreateFromTriangleMesh(
            const ccMesh &mesh);

protected:
    /// Returns the next half edge from starting vertex of the input half edge,
    /// in a counterclock wise manner. Returns -1 if when hitting a boundary.
    /// This is done by traversing to the next, next and twin half edge.
    int nextHalfEdgeFromVertex(int init_half_edge_index) const;
    int nextHalfEdgeOnBoundary(int curr_half_edge_index) const;

public:
    /// List of triangles in the mesh.
    std::vector<Eigen::Vector3i> triangles_;
    /// List of triangle normals in the mesh.
    std::vector<Eigen::Vector3d> triangle_normals_;
    /// List of HalfEdge in the mesh.
    std::vector<HalfEdge> half_edges_;

    /// Counter-clockwise ordered half-edges started from each vertex.
    /// If the vertex is on boundary, the starting edge must be on boundary too.
    std::vector<std::vector<int>> ordered_half_edge_from_vertex_;
};

}  // namespace geometry
}  // namespace cloudViewer
