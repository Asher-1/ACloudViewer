// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
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

#include "ecvHalfEdgeMesh.h"

// LOCAL
#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "ecvHObjectCaster.h"

#include <Logging.h>

// EIGEN
#include <Eigen/Dense>

// SYSTEM
#include <unordered_map>
#include <array>
#include <numeric>
#include <tuple>

using namespace cloudViewer;
namespace cloudViewer {
namespace geometry {

ecvHalfEdgeMesh::HalfEdge::HalfEdge(const Eigen::Vector2i &vertex_indices,
                                         int triangle_index,
                                         int next,
                                         int twin)
    : next_(next),
      twin_(twin),
      vertex_indices_(vertex_indices),
      triangle_index_(triangle_index) {}

ecvHalfEdgeMesh &ecvHalfEdgeMesh::clear() {
    ecvMeshBase::clear();
    half_edges_.clear();
    ordered_half_edge_from_vertex_.clear();
    return *this;
}

bool ecvHalfEdgeMesh::hasHalfEdges() const {
    return half_edges_.size() > 0 &&
           vertices_.size() == ordered_half_edge_from_vertex_.size();
}

int ecvHalfEdgeMesh::nextHalfEdgeFromVertex(int half_edge_index) const {
    const HalfEdge &curr_he = half_edges_[half_edge_index];
    int next_he_index = curr_he.next_;
    const HalfEdge &next_he = half_edges_[next_he_index];
    int next_next_he_index = next_he.next_;
    const HalfEdge &next_next_he = half_edges_[next_next_he_index];
    int next_next_twin_he_index = next_next_he.twin_;
    return next_next_twin_he_index;
}

std::vector<int> ecvHalfEdgeMesh::boundaryHalfEdgesFromVertex(
        int vertex_index) const {
    int init_he_index = ordered_half_edge_from_vertex_[vertex_index][0];
    const HalfEdge &init_he = half_edges_[init_he_index];

    if (!init_he.IsBoundary()) {
        utility::LogError("The vertex {:d} is not on boundary.", vertex_index);
    }

    std::vector<int> boundary_half_edge_indices;
    int curr_he_index = init_he_index;
    boundary_half_edge_indices.push_back(curr_he_index);
    curr_he_index = nextHalfEdgeOnBoundary(curr_he_index);
    while (curr_he_index != init_he_index && curr_he_index != -1) {
        boundary_half_edge_indices.push_back(curr_he_index);
        curr_he_index = nextHalfEdgeOnBoundary(curr_he_index);
    }
    return boundary_half_edge_indices;
}

std::vector<int> ecvHalfEdgeMesh::boundaryVerticesFromVertex(
        int vertex_index) const {
    std::vector<int> boundary_half_edges =
            boundaryHalfEdgesFromVertex(vertex_index);
    std::vector<int> boundary_vertices;
    for (const int &half_edge_idx : boundary_half_edges) {
        boundary_vertices.push_back(
                half_edges_[half_edge_idx].vertex_indices_(0));
    }
    return boundary_vertices;
}

std::vector<std::vector<int>> ecvHalfEdgeMesh::getBoundaries() const {
    std::vector<std::vector<int>> boundaries;
    std::unordered_set<int> visited;

    for (int vertex_ind = 0; vertex_ind < int(vertices_.size()); ++vertex_ind) {
        if (visited.find(vertex_ind) != visited.end()) {
            continue;
        }
        // It is guaranteed that if a vertex in on boundary, the starting
        // edge must be on boundary. After purging, it's also guaranteed that
        // a vertex always have out-going half-edges after purging.
        int first_half_edge_ind = ordered_half_edge_from_vertex_[vertex_ind][0];
        if (half_edges_[first_half_edge_ind].IsBoundary()) {
            std::vector<int> boundary = boundaryVerticesFromVertex(vertex_ind);
            boundaries.push_back(boundary);
            for (int boundary_vertex : boundary) {
                visited.insert(boundary_vertex);
            }
        }
        visited.insert(vertex_ind);
    }
    return boundaries;
}

int ecvHalfEdgeMesh::nextHalfEdgeOnBoundary(
        int curr_half_edge_index) const {
    if (!hasHalfEdges() || curr_half_edge_index >= int(half_edges_.size()) ||
        curr_half_edge_index == -1) {
        utility::LogWarning(
                "edge index {:d} out of range or half-edges not available.",
                curr_half_edge_index);
        return -1;
    }
    if (!half_edges_[curr_half_edge_index].IsBoundary()) {
        utility::LogWarning(
                "The currented half-edge index {:d} is on boundary.",
                curr_half_edge_index);
        return -1;
    }

    // curr_half_edge's end point and next_half_edge's start point is the same
    // vertex. It is guaranteed that next_half_edge is the first edge
    // in ordered_half_edge_from_vertex_ and next_half_edge is a boundary edge.
    int vertex_index = half_edges_[curr_half_edge_index].vertex_indices_(1);
    int next_half_edge_index = ordered_half_edge_from_vertex_[vertex_index][0];
    if (!half_edges_[next_half_edge_index].IsBoundary()) {
        utility::LogWarning(
                "[nextHalfEdgeOnBoundary] The next half-edge along the "
                "boundary is not a boundary edge.");
        return -1;
    }
    return next_half_edge_index;
}

std::shared_ptr<ecvHalfEdgeMesh>
ecvHalfEdgeMesh::CreateFromTriangleMesh(const ccMesh &mesh) {
    ccPointCloud *baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    auto mesh_cpy = cloudViewer::make_shared<ccMesh>(baseVertices);
    auto het_mesh = cloudViewer::make_shared<ecvHalfEdgeMesh>();

    // Copy
    *mesh_cpy = mesh;

    // Purge to remove duplications
    mesh_cpy->removeDuplicatedVertices();
    mesh_cpy->removeDuplicatedTriangles();
    mesh_cpy->removeUnreferencedVertices();
    mesh_cpy->removeDegenerateTriangles();

    // Collect half edges
    // Check: for valid manifolds, there mustn't be duplicated half-edges
    std::unordered_map<Eigen::Vector2i, size_t, utility::hash_eigen<Eigen::Vector2i>>
            vertex_indices_to_half_edge_index;

    for (size_t triangle_index = 0;
         triangle_index < mesh_cpy->size(); triangle_index++) {
        const Eigen::Vector3i &triangle = mesh_cpy->getTriangle(triangle_index);
        size_t num_half_edges = het_mesh->half_edges_.size();

        size_t he_0_index = num_half_edges;
        size_t he_1_index = num_half_edges + 1;
        size_t he_2_index = num_half_edges + 2;
        HalfEdge he_0(Eigen::Vector2i(triangle(0), triangle(1)),
                      int(triangle_index), int(he_1_index), -1);
        HalfEdge he_1(Eigen::Vector2i(triangle(1), triangle(2)),
                      int(triangle_index), int(he_2_index), -1);
        HalfEdge he_2(Eigen::Vector2i(triangle(2), triangle(0)),
                      int(triangle_index), int(he_0_index), -1);

        if (vertex_indices_to_half_edge_index.find(he_0.vertex_indices_) !=
                    vertex_indices_to_half_edge_index.end() ||
            vertex_indices_to_half_edge_index.find(he_1.vertex_indices_) !=
                    vertex_indices_to_half_edge_index.end() ||
            vertex_indices_to_half_edge_index.find(he_2.vertex_indices_) !=
                    vertex_indices_to_half_edge_index.end()) {
            utility::LogError(
                    "ComputeHalfEdges failed. Duplicated half-edges.");
        }

        het_mesh->half_edges_.push_back(he_0);
        het_mesh->half_edges_.push_back(he_1);
        het_mesh->half_edges_.push_back(he_2);
        vertex_indices_to_half_edge_index[he_0.vertex_indices_] = he_0_index;
        vertex_indices_to_half_edge_index[he_1.vertex_indices_] = he_1_index;
        vertex_indices_to_half_edge_index[he_2.vertex_indices_] = he_2_index;
    }

    // Fill twin half-edge. In the previous step, it is already guaranteed that
    // each half-edge can have at most one twin half-edge.
    for (size_t this_he_index = 0; this_he_index < het_mesh->half_edges_.size();
         this_he_index++) {
        HalfEdge &this_he = het_mesh->half_edges_[this_he_index];
        Eigen::Vector2i twin_end_points(this_he.vertex_indices_(1),
                                        this_he.vertex_indices_(0));
        if (this_he.twin_ == -1 &&
            vertex_indices_to_half_edge_index.find(twin_end_points) !=
                    vertex_indices_to_half_edge_index.end()) {
            size_t twin_he_index =
                    vertex_indices_to_half_edge_index[twin_end_points];
            HalfEdge &twin_he = het_mesh->half_edges_[twin_he_index];
            this_he.twin_ = int(twin_he_index);
            twin_he.twin_ = int(this_he_index);
        }
    }

    // Get out-going half-edges from each vertex. This can be done during
    // half-edge construction. Done here for readability.
    std::vector<std::vector<int>> half_edges_from_vertex(
            mesh_cpy->getVerticeSize());
    for (size_t half_edge_index = 0;
         half_edge_index < het_mesh->half_edges_.size(); half_edge_index++) {
        int src_vertex_index =
                het_mesh->half_edges_[half_edge_index].vertex_indices_(0);
        half_edges_from_vertex[src_vertex_index].push_back(
                int(half_edge_index));
    }

    // Find ordered half-edges from each vertex by traversal. To be a valid
    // manifold, there can be at most 1 boundary half-edge from each vertex.
    het_mesh->ordered_half_edge_from_vertex_.resize(mesh_cpy->getVerticeSize());
    for (size_t vertex_index = 0; vertex_index < mesh_cpy->getVerticeSize();
         vertex_index++) {
        size_t num_boundaries = 0;
        int init_half_edge_index = 0;
        for (const int &half_edge_index :
             half_edges_from_vertex[vertex_index]) {
            if (het_mesh->half_edges_[half_edge_index].IsBoundary()) {
                num_boundaries++;
                init_half_edge_index = half_edge_index;
            }
        }
        if (num_boundaries > 1) {
            utility::LogError("ComputeHalfEdges failed. Invalid vertex.");
        }
        // If there is a boundary edge, start from that; otherwise start
        // with any half-edge (default 0) started from this vertex.
        if (num_boundaries == 0) {
            init_half_edge_index = half_edges_from_vertex[vertex_index][0];
        }

        // Push edges to ordered_half_edge_from_vertex_.
        int curr_he_index = init_half_edge_index;
        het_mesh->ordered_half_edge_from_vertex_[vertex_index].push_back(
                curr_he_index);
        int next_next_twin_he_index =
                het_mesh->nextHalfEdgeFromVertex(curr_he_index);
        curr_he_index = next_next_twin_he_index;
        while (curr_he_index != -1 && curr_he_index != init_half_edge_index) {
            het_mesh->ordered_half_edge_from_vertex_[vertex_index].push_back(
                    curr_he_index);
            next_next_twin_he_index =
                    het_mesh->nextHalfEdgeFromVertex(curr_he_index);
            curr_he_index = next_next_twin_he_index;
        }
    }

    mesh_cpy->computeVertexNormals();
    het_mesh->vertices_ = mesh_cpy->getEigenVertices();
    het_mesh->vertex_normals_ = mesh_cpy->getVertexNormals();
    het_mesh->vertex_colors_ = mesh_cpy->getVertexColors();
    het_mesh->triangles_ = mesh_cpy->getTriangles();
    het_mesh->triangle_normals_ = mesh_cpy->getTriangleNormals();

    return het_mesh;
}

ecvHalfEdgeMesh &ecvHalfEdgeMesh::operator+=(
        const ecvHalfEdgeMesh &mesh) {
    ecvMeshBase::operator+=(mesh);
    // TODO
    return *this;
}

ecvHalfEdgeMesh ecvHalfEdgeMesh::operator+(
        const ecvHalfEdgeMesh &mesh) const {
    return (ecvHalfEdgeMesh(*this) += mesh);
}

}  // namespace geometry
}  // namespace cloudViewer
