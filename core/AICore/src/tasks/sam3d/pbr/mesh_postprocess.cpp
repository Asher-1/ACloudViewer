// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mesh_postprocess.hpp"

#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkQuadricDecimation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sam3d {
namespace {

bool valid_mesh(const NativeMesh& mesh) {
    if (mesh.positions.empty() || mesh.positions.size() % 3 != 0 ||
        mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        return false;
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    for (uint32_t index : mesh.indices) {
        if (index >= vertex_count) return false;
    }
    return true;
}

bool valid_vertex_attributes(const NativeMesh& mesh) {
    const size_t vertex_count = mesh.positions.size() / 3;
    return (mesh.normals.empty() || mesh.normals.size() == vertex_count * 3) &&
           (mesh.texcoords.empty() ||
            mesh.texcoords.size() == vertex_count * 2) &&
           (mesh.colors.empty() || mesh.colors.size() == vertex_count * 3) &&
           (mesh.vertex_attributes.empty() ||
            mesh.vertex_attributes.size() == vertex_count * 6);
}

uint64_t edge_key(uint32_t first, uint32_t second) {
    const uint32_t low = std::min(first, second);
    const uint32_t high = std::max(first, second);
    return (static_cast<uint64_t>(low) << 32) | high;
}

struct DisjointSet {
    explicit DisjointSet(size_t count) : parent(count), rank(count, 0) {
        for (size_t index = 0; index < count; ++index)
            parent[index] = static_cast<uint32_t>(index);
    }

    uint32_t find(uint32_t value) {
        while (parent[value] != value) {
            parent[value] = parent[parent[value]];
            value = parent[value];
        }
        return value;
    }

    void unite(uint32_t first, uint32_t second) {
        first = find(first);
        second = find(second);
        if (first == second) return;
        if (rank[first] < rank[second]) std::swap(first, second);
        parent[second] = first;
        if (rank[first] == rank[second]) ++rank[first];
    }

    std::vector<uint32_t> parent;
    std::vector<uint8_t> rank;
};

struct MeshEdge {
    uint32_t first = 0;
    uint32_t second = 0;
    std::vector<uint32_t> faces;
};

struct DinicEdge {
    int to = 0;
    int reverse = 0;
    double capacity = 0.0;
};

// The official graph is undirected. We represent every undirected capacity by
// two directed residual edges, then use a conventional Dinic residual search.
class Dinic {
public:
    explicit Dinic(size_t node_count)
        : graph_(node_count), level_(node_count), next_(node_count) {}

    void add_directed(int from, int to, double capacity) {
        DinicEdge forward{to, static_cast<int>(graph_[to].size()), capacity};
        DinicEdge reverse{from, static_cast<int>(graph_[from].size()), 0.0};
        graph_[from].push_back(forward);
        graph_[to].push_back(reverse);
    }

    void add_undirected(int first, int second, double capacity) {
        add_directed(first, second, capacity);
        add_directed(second, first, capacity);
    }

    void max_flow(int source, int sink) {
        while (build_levels(source, sink)) {
            std::fill(next_.begin(), next_.end(), 0);
            while (send(source, sink, std::numeric_limits<double>::infinity()) >
                   kEpsilon) {
            }
        }
    }

    // Return the maximal source-side minimum cut. Minimum cuts form a lattice;
    // traversing residual edges backwards from the sink selects the maximal
    // source partition, matching igraph's deterministic partition on ties.
    std::vector<uint8_t> source_partition(int source, int sink) const {
        std::vector<uint8_t> can_reach_sink(graph_.size(), 0);
        std::queue<int> pending;
        pending.push(sink);
        can_reach_sink[sink] = 1;
        while (!pending.empty()) {
            const int node = pending.front();
            pending.pop();
            for (const DinicEdge& edge : graph_[node]) {
                const DinicEdge& reverse = graph_[edge.to][edge.reverse];
                if (reverse.capacity > kEpsilon && !can_reach_sink[edge.to]) {
                    can_reach_sink[edge.to] = 1;
                    pending.push(edge.to);
                }
            }
        }
        std::vector<uint8_t> source_side(graph_.size(), 1);
        for (size_t node = 0; node < graph_.size(); ++node) {
            source_side[node] = can_reach_sink[node] ? 0 : 1;
        }
        source_side[sink] = 0;
        source_side[source] = 1;
        return source_side;
    }

private:
    static constexpr double kEpsilon = 1.0e-9;

    bool build_levels(int source, int sink) {
        std::fill(level_.begin(), level_.end(), -1);
        std::queue<int> pending;
        pending.push(source);
        level_[source] = 0;
        while (!pending.empty()) {
            const int node = pending.front();
            pending.pop();
            for (const DinicEdge& edge : graph_[node]) {
                if (edge.capacity > kEpsilon && level_[edge.to] < 0) {
                    level_[edge.to] = level_[node] + 1;
                    pending.push(edge.to);
                }
            }
        }
        return level_[sink] >= 0;
    }

    double send(int node, int sink, double flow) {
        if (node == sink) return flow;
        for (int& edge_index = next_[node];
             edge_index < static_cast<int>(graph_[node].size()); ++edge_index) {
            DinicEdge& edge = graph_[node][edge_index];
            if (edge.capacity <= kEpsilon ||
                level_[edge.to] != level_[node] + 1)
                continue;
            const double delivered =
                    send(edge.to, sink, std::min(flow, edge.capacity));
            if (delivered <= kEpsilon) continue;
            edge.capacity -= delivered;
            graph_[edge.to][edge.reverse].capacity += delivered;
            return delivered;
        }
        return 0.0;
    }

    std::vector<std::vector<DinicEdge>> graph_;
    std::vector<int> level_;
    std::vector<int> next_;
};

float component_quantile_75(const std::vector<uint32_t>& faces,
                            const std::vector<float>& visibility) {
    std::vector<float> values;
    values.reserve(faces.size());
    for (uint32_t face : faces) values.push_back(visibility[face]);
    std::sort(values.begin(), values.end());
    const float position = 0.75f * static_cast<float>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    return values[lower] + (values[upper] - values[lower]) *
                                   (position - static_cast<float>(lower));
}

float edge_length(const NativeMesh& mesh, const MeshEdge& edge) {
    const float* first =
            mesh.positions.data() + static_cast<size_t>(edge.first) * 3;
    const float* second =
            mesh.positions.data() + static_cast<size_t>(edge.second) * 3;
    const float dx = first[0] - second[0];
    const float dy = first[1] - second[1];
    const float dz = first[2] - second[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float new_boundary_area(const NativeMesh& mesh,
                        const std::vector<uint32_t>& edge_indices,
                        const std::vector<MeshEdge>& edges) {
    float center[3] = {0.0f, 0.0f, 0.0f};
    for (uint32_t edge_index : edge_indices) {
        const MeshEdge& edge = edges[edge_index];
        const float* first =
                mesh.positions.data() + static_cast<size_t>(edge.first) * 3;
        const float* second =
                mesh.positions.data() + static_cast<size_t>(edge.second) * 3;
        for (int axis = 0; axis < 3; ++axis)
            center[axis] += first[axis] + second[axis];
    }
    const float scale = 1.0f / static_cast<float>(edge_indices.size() * 2);
    for (float& value : center) value *= scale;

    float area = 0.0f;
    for (uint32_t edge_index : edge_indices) {
        const MeshEdge& edge = edges[edge_index];
        const float* first =
                mesh.positions.data() + static_cast<size_t>(edge.first) * 3;
        const float* second =
                mesh.positions.data() + static_cast<size_t>(edge.second) * 3;
        const float first_relative[3] = {first[0] - center[0],
                                         first[1] - center[1],
                                         first[2] - center[2]};
        const float second_relative[3] = {second[0] - center[0],
                                          second[1] - center[1],
                                          second[2] - center[2]};
        const float cross[3] = {
                first_relative[1] * second_relative[2] -
                        first_relative[2] * second_relative[1],
                first_relative[2] * second_relative[0] -
                        first_relative[0] * second_relative[2],
                first_relative[0] * second_relative[1] -
                        first_relative[1] * second_relative[0],
        };
        area += 0.5f * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                                 cross[2] * cross[2]);
    }
    return area;
}

template <size_t kWidth>
void compact_attribute(std::vector<float>& values,
                       const std::vector<int32_t>& remap,
                       size_t compact_vertex_count) {
    if (values.empty()) return;
    std::vector<float> compact(compact_vertex_count * kWidth);
    for (size_t old_vertex = 0; old_vertex < remap.size(); ++old_vertex) {
        const int32_t new_vertex = remap[old_vertex];
        if (new_vertex < 0) continue;
        std::copy_n(values.data() + old_vertex * kWidth, kWidth,
                    compact.data() + static_cast<size_t>(new_vertex) * kWidth);
    }
    values = std::move(compact);
}

}  // namespace

bool simplify_mesh_official_vtk(NativeMesh& mesh,
                                const MeshSimplifyOptions& options,
                                std::string& error) {
    if (!valid_mesh(mesh)) {
        error = "VTK mesh simplification requires a valid non-empty triangle "
                "mesh";
        return false;
    }
    if (!std::isfinite(options.target_reduction) ||
        options.target_reduction < 0.0f || options.target_reduction >= 1.0f) {
        error = "VTK target reduction must be finite and in [0, 1)";
        return false;
    }

    vtkNew<vtkPoints> points;
    points->SetDataTypeToFloat();
    const size_t vertex_count = mesh.positions.size() / 3;
    points->SetNumberOfPoints(static_cast<vtkIdType>(vertex_count));
    for (size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const float* position = mesh.positions.data() + vertex * 3;
        points->SetPoint(static_cast<vtkIdType>(vertex), position[0],
                         position[1], position[2]);
    }
    vtkNew<vtkCellArray> triangles;
    const size_t face_count = mesh.indices.size() / 3;
    triangles->AllocateEstimate(static_cast<vtkIdType>(face_count), 3);
    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t* source = mesh.indices.data() + face * 3;
        const vtkIdType ids[] = {static_cast<vtkIdType>(source[0]),
                                 static_cast<vtkIdType>(source[1]),
                                 static_cast<vtkIdType>(source[2])};
        triangles->InsertNextCell(3, ids);
    }
    vtkNew<vtkPolyData> input;
    input->SetPoints(points);
    input->SetPolys(triangles);

    vtkNew<vtkQuadricDecimation> decimator;
    // These are PyVista 0.48.4 PolyDataFilters.decimate() defaults. Keep each
    // parameter explicit so a VTK default change cannot silently alter assets.
    decimator->SetVolumePreservation(false);
    decimator->SetAttributeErrorMetric(false);
    decimator->SetScalarsAttribute(false);
    decimator->SetVectorsAttribute(false);
    decimator->SetNormalsAttribute(false);
    decimator->SetTCoordsAttribute(false);
    decimator->SetTensorsAttribute(false);
    decimator->SetScalarsWeight(0.1);
    decimator->SetVectorsWeight(0.1);
    decimator->SetNormalsWeight(0.1);
    decimator->SetTCoordsWeight(0.1);
    decimator->SetTensorsWeight(0.1);
    decimator->SetTargetReduction(options.target_reduction);
    decimator->SetWeighBoundaryConstraintsByLength(false);
    decimator->SetBoundaryWeightFactor(1.0);
    decimator->SetInputData(input);
    decimator->Update();

    vtkPolyData* output = decimator->GetOutput();
    if (output == nullptr || output->GetNumberOfPoints() == 0 ||
        output->GetNumberOfPolys() == 0) {
        error = "vtkQuadricDecimation produced an empty mesh";
        return false;
    }
    std::vector<float> positions(
            static_cast<size_t>(output->GetNumberOfPoints()) * 3);
    for (vtkIdType vertex = 0; vertex < output->GetNumberOfPoints(); ++vertex) {
        double point[3];
        output->GetPoint(vertex, point);
        positions[static_cast<size_t>(vertex) * 3 + 0] =
                static_cast<float>(point[0]);
        positions[static_cast<size_t>(vertex) * 3 + 1] =
                static_cast<float>(point[1]);
        positions[static_cast<size_t>(vertex) * 3 + 2] =
                static_cast<float>(point[2]);
    }
    std::vector<uint32_t> indices;
    vtkCellArray* output_triangles = output->GetPolys();
    output_triangles->InitTraversal();
    vtkIdType count = 0;
    const vtkIdType* ids = nullptr;
    while (output_triangles->GetNextCell(count, ids)) {
        if (count != 3 || ids == nullptr || ids[0] < 0 || ids[1] < 0 ||
            ids[2] < 0 || ids[0] > std::numeric_limits<uint32_t>::max() ||
            ids[1] > std::numeric_limits<uint32_t>::max() ||
            ids[2] > std::numeric_limits<uint32_t>::max()) {
            error = "vtkQuadricDecimation emitted a non-triangle or invalid "
                    "index";
            return false;
        }
        indices.push_back(static_cast<uint32_t>(ids[0]));
        indices.push_back(static_cast<uint32_t>(ids[1]));
        indices.push_back(static_cast<uint32_t>(ids[2]));
    }
    if (indices.empty()) {
        error = "vtkQuadricDecimation emitted no triangle cells";
        return false;
    }
    mesh.positions = std::move(positions);
    mesh.indices = std::move(indices);
    // The Python path drops decoder attributes before xatlas and baking.
    mesh.normals.clear();
    mesh.texcoords.clear();
    mesh.colors.clear();
    mesh.vertex_attributes.clear();
    return true;
}

bool filter_invisible_faces_official(NativeMesh& mesh,
                                     const std::vector<float>& visibility,
                                     const MeshVisibilityFilterOptions& options,
                                     MeshVisibilityFilterStats* stats,
                                     std::string& error,
                                     std::vector<uint32_t>* mincut_candidates) {
    if (!valid_mesh(mesh) || !valid_vertex_attributes(mesh)) {
        error = "visibility/mincut filtering requires a valid triangle mesh "
                "and vertex attributes";
        return false;
    }
    const size_t face_count = mesh.indices.size() / 3;
    if (visibility.size() != face_count) {
        error = "visibility/mincut filtering requires one frequency per mesh "
                "face";
        return false;
    }
    if (!std::isfinite(options.max_hole_size) || options.max_hole_size < 0.0f) {
        error = "maximum hole size must be finite and non-negative";
        return false;
    }
    for (float value : visibility) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            error = "visibility frequencies must be finite and in [0, 1]";
            return false;
        }
    }

    MeshVisibilityFilterStats local_stats;
    std::vector<MeshEdge> edges;
    edges.reserve(face_count * 3 / 2);
    std::unordered_map<uint64_t, uint32_t> edge_map;
    edge_map.reserve(face_count * 3);
    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t* triangle = mesh.indices.data() + face * 3;
        for (int local_edge = 0; local_edge < 3; ++local_edge) {
            const uint32_t first = triangle[local_edge];
            const uint32_t second = triangle[(local_edge + 1) % 3];
            const uint64_t key = edge_key(first, second);
            auto [iterator, inserted] =
                    edge_map.emplace(key, static_cast<uint32_t>(edges.size()));
            if (inserted)
                edges.push_back(
                        {std::min(first, second), std::max(first, second), {}});
        }
    }

    // `utils3d.torch.compute_edges` uses torch.unique on sorted endpoint pairs,
    // which emits edges in lexicographic order. Preserve that order before
    // constructing the dual graph; it also makes min-cut tie handling stable.
    std::sort(edges.begin(), edges.end(),
              [](const MeshEdge& first, const MeshEdge& second) {
                  if (first.first != second.first)
                      return first.first < second.first;
                  return first.second < second.second;
              });
    edge_map.clear();
    edge_map.reserve(edges.size());
    for (uint32_t edge = 0; edge < edges.size(); ++edge) {
        edge_map.emplace(edge_key(edges[edge].first, edges[edge].second), edge);
    }
    std::vector<std::array<uint32_t, 3>> face_edges(face_count);
    for (size_t face = 0; face < face_count; ++face) {
        const uint32_t* triangle = mesh.indices.data() + face * 3;
        for (int local_edge = 0; local_edge < 3; ++local_edge) {
            const uint64_t key = edge_key(triangle[local_edge],
                                          triangle[(local_edge + 1) % 3]);
            const uint32_t edge_index = edge_map.at(key);
            edges[edge_index].faces.push_back(static_cast<uint32_t>(face));
            face_edges[face][local_edge] = edge_index;
        }
    }

    DisjointSet all_faces(face_count);
    for (const MeshEdge& edge : edges) {
        for (size_t index = 1; index < edge.faces.size(); ++index) {
            all_faces.unite(edge.faces[0], edge.faces[index]);
        }
    }
    std::unordered_map<uint32_t, std::vector<uint32_t>> component_map;
    component_map.reserve(face_count);
    for (uint32_t face = 0; face < face_count; ++face)
        component_map[all_faces.find(face)].push_back(face);

    std::vector<uint8_t> outer(face_count, 0);
    for (const auto& entry : component_map) {
        const float quantile = component_quantile_75(entry.second, visibility);
        const float threshold = std::min(0.5f, std::max(0.25f, quantile));
        for (uint32_t face : entry.second) {
            if (visibility[face] > threshold) {
                outer[face] = 1;
                ++local_stats.outer_face_count;
            }
        }
    }

    std::vector<uint32_t> invisible;
    invisible.reserve(face_count);
    for (uint32_t face = 0; face < face_count; ++face) {
        if (visibility[face] == 0.0f) invisible.push_back(face);
    }
    local_stats.invisible_face_count = invisible.size();
    if (invisible.empty()) {
        if (stats != nullptr) *stats = local_stats;
        return true;
    }

    const int source = static_cast<int>(face_count);
    const int target = source + 1;
    Dinic graph(face_count + 2);
    constexpr double kOfficialWeightScale = 1000.0;
    for (const MeshEdge& edge : edges) {
        if (edge.faces.size() == 2) {
            // `_fill_holes` produces a float32 torch.norm result and multiplies
            // its NumPy float32 array by 1000 before handing capacities to
            // igraph. Keep that rounding point explicit; using a double
            // product changes otherwise tied cuts on the canonical mesh.
            const float capacity = edge_length(mesh, edge) * 1000.0f;
            graph.add_undirected(static_cast<int>(edge.faces[0]),
                                 static_cast<int>(edge.faces[1]),
                                 static_cast<double>(capacity));
        }
    }
    for (uint32_t face : invisible)
        graph.add_undirected(static_cast<int>(face), source,
                             kOfficialWeightScale);
    for (uint32_t face = 0; face < face_count; ++face) {
        if (outer[face])
            graph.add_undirected(static_cast<int>(face), target,
                                 kOfficialWeightScale);
    }
    graph.max_flow(source, target);
    const std::vector<uint8_t> source_side =
            graph.source_partition(source, target);
    std::vector<uint8_t> proposed(face_count, 0);
    for (uint32_t face = 0; face < face_count; ++face) {
        if (source_side[face]) {
            proposed[face] = 1;
            ++local_stats.mincut_face_count;
        }
    }
    if (mincut_candidates != nullptr) {
        mincut_candidates->clear();
        mincut_candidates->reserve(local_stats.mincut_face_count);
        for (uint32_t face = 0; face < face_count; ++face) {
            if (proposed[face]) mincut_candidates->push_back(face);
        }
    }
    if (local_stats.mincut_face_count == 0) {
        if (stats != nullptr) *stats = local_stats;
        return true;
    }

    DisjointSet cut_faces(face_count);
    for (const MeshEdge& edge : edges) {
        for (size_t index = 1; index < edge.faces.size(); ++index) {
            if (proposed[edge.faces[0]] && proposed[edge.faces[index]]) {
                cut_faces.unite(edge.faces[0], edge.faces[index]);
            }
        }
    }
    std::unordered_map<uint32_t, std::vector<uint32_t>> cut_component_map;
    for (uint32_t face = 0; face < face_count; ++face) {
        if (proposed[face])
            cut_component_map[cut_faces.find(face)].push_back(face);
    }

    std::vector<uint8_t> remove(face_count, 0);
    for (const auto& entry : cut_component_map) {
        std::vector<float> component_visibility;
        component_visibility.reserve(entry.second.size());
        for (uint32_t face : entry.second)
            component_visibility.push_back(visibility[face]);
        std::sort(component_visibility.begin(), component_visibility.end());
        // torch.median() uses the lower middle value for even cardinalities.
        if (component_visibility[(component_visibility.size() - 1) / 2] > 0.25f)
            continue;

        std::unordered_map<uint32_t, uint32_t> cut_edge_counts;
        cut_edge_counts.reserve(entry.second.size() * 3);
        for (uint32_t face : entry.second) {
            for (uint32_t edge_index : face_edges[face])
                ++cut_edge_counts[edge_index];
        }
        std::vector<uint32_t> new_boundary_edges;
        for (const auto& cut_edge : cut_edge_counts) {
            if (cut_edge.second == 1 &&
                edges[cut_edge.first].faces.size() != 1) {
                new_boundary_edges.push_back(cut_edge.first);
            }
        }

        bool valid = true;
        if (!new_boundary_edges.empty()) {
            DisjointSet boundary_components(new_boundary_edges.size());
            std::unordered_map<uint32_t, uint32_t> vertex_to_first_edge;
            vertex_to_first_edge.reserve(new_boundary_edges.size() * 2);
            for (uint32_t local_edge = 0;
                 local_edge < new_boundary_edges.size(); ++local_edge) {
                const MeshEdge& edge = edges[new_boundary_edges[local_edge]];
                for (uint32_t vertex : {edge.first, edge.second}) {
                    auto [iterator, inserted] =
                            vertex_to_first_edge.emplace(vertex, local_edge);
                    if (!inserted)
                        boundary_components.unite(local_edge, iterator->second);
                }
            }
            std::unordered_map<uint32_t, std::vector<uint32_t>> boundary_groups;
            for (uint32_t local_edge = 0;
                 local_edge < new_boundary_edges.size(); ++local_edge) {
                boundary_groups[boundary_components.find(local_edge)].push_back(
                        new_boundary_edges[local_edge]);
            }
            for (const auto& boundary_group : boundary_groups) {
                if (new_boundary_area(mesh, boundary_group.second, edges) >
                    options.max_hole_size) {
                    valid = false;
                    break;
                }
            }
        }
        if (valid) {
            for (uint32_t face : entry.second) remove[face] = 1;
        }
    }

    for (uint8_t value : remove) local_stats.removed_face_count += value;
    if (local_stats.removed_face_count == 0) {
        if (stats != nullptr) *stats = local_stats;
        return true;
    }

    const size_t old_vertex_count = mesh.positions.size() / 3;
    std::vector<int32_t> remap(old_vertex_count, -1);
    // utils3d.torch.remove_unreferenced_vertices uses torch.unique(faces),
    // which returns the retained original indices in ascending order. Preserve
    // that ordering so native mincut tensors can compare byte-for-byte before
    // the subsequent MeshFix boundary repair.
    for (size_t face = 0; face < face_count; ++face) {
        if (remove[face]) continue;
        remap[mesh.indices[face * 3]] = 0;
        remap[mesh.indices[face * 3 + 1]] = 0;
        remap[mesh.indices[face * 3 + 2]] = 0;
    }
    size_t compact_vertex_count = 0;
    for (size_t vertex = 0; vertex < old_vertex_count; ++vertex) {
        if (remap[vertex] == 0)
            remap[vertex] = static_cast<int32_t>(compact_vertex_count++);
    }
    std::vector<uint32_t> compact_indices;
    compact_indices.reserve((face_count - local_stats.removed_face_count) * 3);
    for (size_t face = 0; face < face_count; ++face) {
        if (remove[face]) continue;
        for (uint32_t vertex :
             {mesh.indices[face * 3], mesh.indices[face * 3 + 1],
              mesh.indices[face * 3 + 2]}) {
            compact_indices.push_back(static_cast<uint32_t>(remap[vertex]));
        }
    }
    compact_attribute<3>(mesh.positions, remap, compact_vertex_count);
    compact_attribute<3>(mesh.normals, remap, compact_vertex_count);
    compact_attribute<2>(mesh.texcoords, remap, compact_vertex_count);
    compact_attribute<3>(mesh.colors, remap, compact_vertex_count);
    compact_attribute<6>(mesh.vertex_attributes, remap, compact_vertex_count);
    mesh.indices = std::move(compact_indices);
    if (stats != nullptr) *stats = local_stats;
    return true;
}

}  // namespace sam3d
