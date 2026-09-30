// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "mesh_postprocess.hpp"

#if defined(SAM3D_USE_MESHFIX_GPL)
#include "tmesh.h"
#endif

namespace sam3d {

bool repair_mesh_boundaries_official_meshfix(NativeMesh& mesh,
                                             int max_boundary_edges,
                                             bool refine,
                                             std::string& error) {
#if !defined(SAM3D_USE_MESHFIX_GPL)
    (void)mesh;
    (void)max_boundary_edges;
    (void)refine;
    error = "official MeshFix repair is disabled; configure with "
            "-DSAM3D_GGML_MESHFIX_GPL=ON after accepting its GPL-3.0 or "
            "commercial license";
    return false;
#else
    if (max_boundary_edges < 0 || mesh.positions.empty() ||
        mesh.positions.size() % 3 != 0 || mesh.indices.empty() ||
        mesh.indices.size() % 3 != 0) {
        error = "MeshFix repair requires non-empty triangle positions and "
                "indices";
        return false;
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    if (vertex_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        mesh.indices.size() / 3 >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
        error = "MeshFix repair input exceeds its indexed-mesh limits";
        return false;
    }
    for (uint32_t index : mesh.indices) {
        if (index >= vertex_count ||
            index > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            error = "MeshFix repair received an out-of-range triangle index";
            return false;
        }
    }

    using namespace T_MESH;
    Basic_TMesh repaired;
    const bool previous_quiet = TMesh::quiet;
    TMesh::quiet = true;
    try {
        for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
            const float* position = mesh.positions.data() + vertex * 3;
            repaired.V.appendTail(
                    repaired.newVertex(static_cast<double>(position[0]),
                                       static_cast<double>(position[1]),
                                       static_cast<double>(position[2])));
        }

        std::vector<ExtVertex*> indexed_vertices(vertex_count);
        size_t vertex_index = 0;
        for (Node* node = repaired.V.head(); node != nullptr;
             node = node->next()) {
            auto* vertex = static_cast<Vertex*>(node->data);
            indexed_vertices[vertex_index++] = new ExtVertex(vertex);
        }
        for (size_t face = 0; face < mesh.indices.size() / 3; ++face) {
            const uint32_t* triangle = mesh.indices.data() + face * 3;
            if (!repaired.CreateIndexedTriangle(
                        indexed_vertices.data(), static_cast<int>(triangle[0]),
                        static_cast<int>(triangle[1]),
                        static_cast<int>(triangle[2]))) {
                error = "MeshFix rejected a triangle while rebuilding "
                        "connectivity";
                TMesh::quiet = previous_quiet;
                return false;
            }
        }
        repaired.fixConnectivity();
        repaired.eulerUpdate();
        repaired.fillSmallBoundaries(max_boundary_edges, refine);

        std::unordered_map<const Vertex*, uint32_t> remap;
        remap.reserve(static_cast<size_t>(repaired.V.numels()));
        std::vector<float> positions;
        positions.reserve(static_cast<size_t>(repaired.V.numels()) * 3);
        uint32_t output_vertex = 0;
        for (Node* node = repaired.V.head(); node != nullptr;
             node = node->next()) {
            auto* vertex = static_cast<Vertex*>(node->data);
            remap.emplace(vertex, output_vertex++);
            positions.push_back(static_cast<float>(vertex->x));
            positions.push_back(static_cast<float>(vertex->y));
            positions.push_back(static_cast<float>(vertex->z));
        }
        std::vector<uint32_t> indices;
        indices.reserve(static_cast<size_t>(repaired.T.numels()) * 3);
        for (Node* node = repaired.T.head(); node != nullptr;
             node = node->next()) {
            auto* triangle = static_cast<Triangle*>(node->data);
            const auto first = remap.find(triangle->v1());
            const auto second = remap.find(triangle->v2());
            const auto third = remap.find(triangle->v3());
            if (first == remap.end() || second == remap.end() ||
                third == remap.end()) {
                error = "MeshFix emitted a triangle with an unknown vertex";
                TMesh::quiet = previous_quiet;
                return false;
            }
            indices.insert(indices.end(),
                           {first->second, second->second, third->second});
        }
        TMesh::quiet = previous_quiet;
        mesh.positions = std::move(positions);
        mesh.indices = std::move(indices);
        mesh.normals.clear();
        mesh.texcoords.clear();
        mesh.colors.clear();
        mesh.vertex_attributes.clear();
        return true;
    } catch (...) {
        TMesh::quiet = previous_quiet;
        error = "MeshFix repair failed while rebuilding or triangulating the "
                "mesh";
        return false;
    }
#endif
}

}  // namespace sam3d
