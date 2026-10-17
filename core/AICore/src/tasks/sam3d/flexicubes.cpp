// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "flexicubes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "flexicubes_tables.hpp"

namespace sam3d {
namespace {

constexpr int kFeatureChannels = 101;
constexpr int kVertexAttributes =
        10;  // sdf + deform xyz + RGB + normal map xyz
constexpr size_t kNoFeature = std::numeric_limits<size_t>::max();

constexpr int kCorners[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
};
constexpr int kEdges[12][2] = {
        {0, 1}, {1, 5}, {4, 5}, {0, 4}, {2, 3}, {3, 7},
        {6, 7}, {2, 6}, {2, 0}, {3, 1}, {7, 5}, {6, 4},
};

struct GridKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator<(const GridKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct VertexData {
    std::array<float, kVertexAttributes> sum{};
    uint32_t count = 0;
    uint32_t index = 0;
};

struct EdgeKey {
    uint32_t first = 0;
    uint32_t second = 0;

    bool operator<(const EdgeKey& other) const {
        return first != other.first ? first < other.first
                                    : second < other.second;
    }
};

struct UniqueEdge {
    EdgeKey key;
    int count = 0;
    int surface_index = -1;
};

struct Cell {
    GridKey key;
    size_t feature_index = kNoFeature;
    std::array<uint32_t, 8> vertices{};
};

struct SurfaceCell {
    const Cell* cell = nullptr;
    int case_id = 0;
    std::array<UniqueEdge*, 12> edges{};
    std::array<uint32_t, 12> dual_vertices{};
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

float sigmoid(float value) {
    // Preserve the direct form used by torch.sigmoid in the official gamma
    // normalization. The mathematically equivalent negative branch changes
    // enough final-bit rounding to select the other quad diagonal.
    return 1.0f / (1.0f + std::exp(-value));
}

Vec3 operator+(const Vec3& left, const Vec3& right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 operator*(const Vec3& value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

const float* feature_row(const float* features, size_t feature_index) {
    return features + feature_index * kFeatureChannels;
}

float vertex_component(const VertexData& vertex, int component) {
    if (vertex.count == 0) return component == 0 ? 1.0f : 0.0f;
    return vertex.sum[component] / static_cast<float>(vertex.count);
}

float cell_weight(const float* features, const Cell& cell, int offset) {
    // SparseFeatures2Mesh._calc_layout inserts weights before the optional
    // six-channel color payload: sdf[0:8], deform[8:32], weights[32:53],
    // color[53:101]. Keep this tied to that declaration rather than visual
    // grouping of vertex/cell attributes in cube2mesh.py.
    return cell.feature_index == kNoFeature
                   ? 0.0f
                   : feature_row(features, cell.feature_index)[32 + offset];
}

float normalized_beta(const float* features, const Cell& cell, int edge) {
    return std::tanh(cell_weight(features, cell, edge)) * 0.99f + 1.0f;
}

float normalized_gamma(const float* features, const Cell& cell) {
    return sigmoid(cell_weight(features, cell, 20)) * 0.99f + 0.005f;
}

float normalized_alpha(const float* features, const Cell& cell, int corner) {
    return std::tanh(cell_weight(features, cell, 12 + corner)) * 0.99f + 1.0f;
}

Vec3 deformed_position(const GridKey& key,
                       const VertexData& vertex,
                       int resolution) {
    constexpr float kDeformScaleAdjust = 1.0f - 1e-8f;
    const float scale =
            kDeformScaleAdjust / (static_cast<float>(resolution) * 2.0f);
    return {
            static_cast<float>(key.x) / resolution - 0.5f +
                    scale * std::tanh(vertex_component(vertex, 1)),
            static_cast<float>(key.y) / resolution - 0.5f +
                    scale * std::tanh(vertex_component(vertex, 2)),
            static_cast<float>(key.z) / resolution - 0.5f +
                    scale * std::tanh(vertex_component(vertex, 3)),
    };
}

Vec3 interpolate_position(const Vec3& first,
                          const Vec3& second,
                          float first_weight,
                          float second_weight) {
    const float denominator = second_weight - first_weight;
    return (first * second_weight + second * -first_weight) *
           (1.0f / denominator);
}

void interpolate_attributes(const VertexData& first,
                            const VertexData& second,
                            float first_weight,
                            float second_weight,
                            std::array<float, 6>& output) {
    const float denominator = second_weight - first_weight;
    for (int channel = 0; channel < 6; ++channel) {
        const float first_value = sigmoid(vertex_component(first, 4 + channel));
        const float second_value =
                sigmoid(vertex_component(second, 4 + channel));
        output[channel] =
                (first_value * second_weight - second_value * first_weight) /
                denominator;
    }
}

void append_triangle(FlexiCubesResult& output,
                     uint32_t first,
                     uint32_t second,
                     uint32_t third) {
    output.indices.push_back(first);
    output.indices.push_back(second);
    output.indices.push_back(third);
}

}  // namespace

bool decode_flexicubes(const float* cube_features,
                       const int32_t* coordinates,
                       int64_t sparse_cell_count,
                       int resolution,
                       FlexiCubesResult& output,
                       std::string& error) {
    output = {};
    if (cube_features == nullptr || coordinates == nullptr ||
        sparse_cell_count <= 0 || resolution <= 0) {
        error = "FlexiCubes requires non-empty features, coordinates, and a "
                "positive resolution";
        return false;
    }

    std::map<GridKey, size_t> sparse_cells;
    std::map<GridKey, VertexData> vertices;
    for (int64_t index = 0; index < sparse_cell_count; ++index) {
        const int32_t* coord = coordinates + index * 4;
        if (coord[0] != 0) {
            error = "native FlexiCubes currently requires a single batch with "
                    "batch index zero";
            return false;
        }
        const GridKey cell_key{coord[1], coord[2], coord[3]};
        if (cell_key.x < 0 || cell_key.y < 0 || cell_key.z < 0 ||
            cell_key.x >= resolution || cell_key.y >= resolution ||
            cell_key.z >= resolution) {
            error = "sparse cell coordinate is outside the FlexiCubes "
                    "resolution";
            return false;
        }
        const auto [cell_it, inserted] =
                sparse_cells.emplace(cell_key, static_cast<size_t>(index));
        if (!inserted) {
            error = "duplicate sparse cell coordinate";
            return false;
        }
        const float* features =
                feature_row(cube_features, static_cast<size_t>(index));
        for (int corner = 0; corner < 8; ++corner) {
            const GridKey vertex_key{
                    cell_key.x + kCorners[corner][0],
                    cell_key.y + kCorners[corner][1],
                    cell_key.z + kCorners[corner][2],
            };
            VertexData& vertex = vertices[vertex_key];
            vertex.sum[0] +=
                    features[corner] - 1.0f / static_cast<float>(resolution);
            for (int channel = 0; channel < 3; ++channel) {
                vertex.sum[1 + channel] += features[8 + corner * 3 + channel];
            }
            for (int channel = 0; channel < 6; ++channel) {
                vertex.sum[4 + channel] += features[53 + corner * 6 + channel];
            }
            ++vertex.count;
        }
    }

    // Official get_dense_attrs processes an entire dense grid. Only cells
    // adjacent to an explicitly populated vertex can become a surface; build
    // exactly that finite subset, preserving dense-grid lexicographic order.
    std::map<GridKey, bool> candidate_cells;
    for (const auto& [vertex_key, vertex] : vertices) {
        (void)vertex;
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    const GridKey cell_key{vertex_key.x - dx, vertex_key.y - dy,
                                           vertex_key.z - dz};
                    if (cell_key.x >= 0 && cell_key.y >= 0 && cell_key.z >= 0 &&
                        cell_key.x < resolution && cell_key.y < resolution &&
                        cell_key.z < resolution) {
                        candidate_cells.emplace(cell_key, true);
                    }
                }
            }
        }
    }

    for (const auto& [cell_key, present] : candidate_cells) {
        (void)present;
        for (const auto& corner : kCorners) {
            vertices.try_emplace({cell_key.x + corner[0],
                                  cell_key.y + corner[1],
                                  cell_key.z + corner[2]});
        }
    }
    std::vector<const VertexData*> vertex_by_index;
    std::vector<GridKey> vertex_keys;
    vertex_by_index.reserve(vertices.size());
    vertex_keys.reserve(vertices.size());
    uint32_t vertex_index = 0;
    for (auto& [key, vertex] : vertices) {
        vertex.index = vertex_index++;
        vertex_by_index.push_back(&vertex);
        vertex_keys.push_back(key);
    }

    std::vector<Cell> cells;
    cells.reserve(candidate_cells.size());
    for (const auto& [cell_key, present] : candidate_cells) {
        (void)present;
        Cell cell;
        cell.key = cell_key;
        const auto sparse = sparse_cells.find(cell_key);
        if (sparse != sparse_cells.end()) cell.feature_index = sparse->second;
        for (int corner = 0; corner < 8; ++corner) {
            const GridKey vertex_key{cell_key.x + kCorners[corner][0],
                                     cell_key.y + kCorners[corner][1],
                                     cell_key.z + kCorners[corner][2]};
            cell.vertices[corner] = vertices.find(vertex_key)->second.index;
        }
        cells.push_back(cell);
    }

    std::vector<SurfaceCell> surface_cells;
    surface_cells.reserve(cells.size() / 4);
    for (const Cell& cell : cells) {
        int case_id = 0;
        int negative_count = 0;
        for (int corner = 0; corner < 8; ++corner) {
            const float sdf = vertex_component(
                    *vertex_by_index[cell.vertices[corner]], 0);
            if (sdf < 0.0f) {
                case_id |= 1 << corner;
                ++negative_count;
            }
        }
        if (negative_count > 0 && negative_count < 8) {
            SurfaceCell surface;
            surface.cell = &cell;
            surface.case_id = case_id;
            surface_cells.push_back(surface);
        }
    }
    if (surface_cells.empty()) return true;

    std::map<GridKey, size_t> surface_by_cell;
    for (size_t index = 0; index < surface_cells.size(); ++index) {
        surface_by_cell.emplace(surface_cells[index].cell->key, index);
    }
    // The DMC ambiguity resolver checks only a neighbouring problematic case.
    // PyTorch builds problem_config_full before writing any inverted case IDs,
    // so all neighbour tests must read this immutable snapshot.
    std::vector<int> original_cases;
    original_cases.reserve(surface_cells.size());
    for (const SurfaceCell& surface : surface_cells)
        original_cases.push_back(surface.case_id);
    for (size_t index = 0; index < surface_cells.size(); ++index) {
        SurfaceCell& surface = surface_cells[index];
        const auto& config =
                flexicubes_tables::kCheckTable[original_cases[index]];
        if (config[0] != 1) continue;
        const GridKey neighbor{surface.cell->key.x + config[1],
                               surface.cell->key.y + config[2],
                               surface.cell->key.z + config[3]};
        const auto match = surface_by_cell.find(neighbor);
        if (match == surface_by_cell.end()) continue;
        const int neighbor_case = original_cases[match->second];
        if (flexicubes_tables::kCheckTable[neighbor_case][0] == 1)
            surface.case_id = config[4];
    }

    std::map<EdgeKey, UniqueEdge> edges;
    for (SurfaceCell& surface : surface_cells) {
        for (int edge = 0; edge < 12; ++edge) {
            // FlexiCubes keeps the directed corner pairs in cube_edges. In
            // particular edges 8--11 are reverse-oriented. torch.unique
            // operates on those rows as-is, and the first endpoint is later
            // used for winding; canonicalizing the pair changes both output
            // ordering and face direction.
            const EdgeKey key{
                    surface.cell->vertices[kEdges[edge][0]],
                    surface.cell->vertices[kEdges[edge][1]],
            };
            auto [it, inserted] = edges.emplace(key, UniqueEdge{key});
            (void)inserted;
            ++it->second.count;
            surface.edges[edge] = &it->second;
        }
    }
    std::vector<UniqueEdge*> surface_edges;
    surface_edges.reserve(edges.size());
    for (auto& [key, edge] : edges) {
        const float first_sdf =
                vertex_component(*vertex_by_index[edge.key.first], 0);
        const float second_sdf =
                vertex_component(*vertex_by_index[edge.key.second], 0);
        if ((first_sdf < 0.0f) != (second_sdf < 0.0f)) {
            edge.surface_index = static_cast<int>(surface_edges.size());
            surface_edges.push_back(&edge);
        }
    }

    for (int number_of_duals = 1; number_of_duals <= 4; ++number_of_duals) {
        for (SurfaceCell& surface : surface_cells) {
            if (flexicubes_tables::kNumVdTable[surface.case_id] !=
                number_of_duals)
                continue;
            for (int dual = 0; dual < number_of_duals; ++dual) {
                Vec3 position_sum;
                std::array<float, 6> attribute_sum{};
                float beta_sum = 0.0f;
                for (int table_slot = 0; table_slot < 7; ++table_slot) {
                    const int edge =
                            flexicubes_tables::kDmcTable[surface.case_id][dual]
                                                        [table_slot];
                    if (edge < 0) continue;
                    const int first_corner = kEdges[edge][0];
                    const int second_corner = kEdges[edge][1];
                    const VertexData& first_vertex =
                            *vertex_by_index[surface.cell
                                                     ->vertices[first_corner]];
                    const VertexData& second_vertex =
                            *vertex_by_index[surface.cell
                                                     ->vertices[second_corner]];
                    const float first_weight =
                            vertex_component(first_vertex, 0) *
                            normalized_alpha(cube_features, *surface.cell,
                                             first_corner);
                    const float second_weight =
                            vertex_component(second_vertex, 0) *
                            normalized_alpha(cube_features, *surface.cell,
                                             second_corner);
                    const Vec3 first_position = deformed_position(
                            vertex_keys[surface.cell->vertices[first_corner]],
                            first_vertex, resolution);
                    const Vec3 second_position = deformed_position(
                            vertex_keys[surface.cell->vertices[second_corner]],
                            second_vertex, resolution);
                    const Vec3 crossing = interpolate_position(
                            first_position, second_position, first_weight,
                            second_weight);
                    std::array<float, 6> attributes{};
                    interpolate_attributes(first_vertex, second_vertex,
                                           first_weight, second_weight,
                                           attributes);
                    const float beta =
                            normalized_beta(cube_features, *surface.cell, edge);
                    position_sum = position_sum + crossing * beta;
                    for (int channel = 0; channel < 6; ++channel)
                        attribute_sum[channel] += attributes[channel] * beta;
                    beta_sum += beta;
                }
                if (!(beta_sum > 0.0f) || !std::isfinite(beta_sum)) {
                    error = "invalid FlexiCubes beta accumulation";
                    return false;
                }
                const uint32_t dual_index =
                        static_cast<uint32_t>(output.positions.size() / 3);
                output.positions.push_back(position_sum.x / beta_sum);
                output.positions.push_back(position_sum.y / beta_sum);
                output.positions.push_back(position_sum.z / beta_sum);
                for (float value : attribute_sum)
                    output.vertex_attributes.push_back(value / beta_sum);
                for (int table_slot = 0; table_slot < 7; ++table_slot) {
                    const int edge =
                            flexicubes_tables::kDmcTable[surface.case_id][dual]
                                                        [table_slot];
                    if (edge >= 0) surface.dual_vertices[edge] = dual_index;
                }
            }
        }
    }

    struct QuadEntry {
        int surface_edge = -1;
        uint32_t dual_vertex = 0;
    };
    std::vector<QuadEntry> quad_entries;
    for (const SurfaceCell& surface : surface_cells) {
        for (int edge = 0; edge < 12; ++edge) {
            const UniqueEdge* unique_edge = surface.edges[edge];
            if (unique_edge->count == 4 && unique_edge->surface_index >= 0) {
                quad_entries.push_back({unique_edge->surface_index,
                                        surface.dual_vertices[edge]});
            }
        }
    }
    std::stable_sort(quad_entries.begin(), quad_entries.end(),
                     [](const QuadEntry& left, const QuadEntry& right) {
                         return left.surface_edge < right.surface_edge;
                     });
    if (quad_entries.size() % 4 != 0) {
        error = "FlexiCubes surface-edge topology does not form four-cell "
                "quads";
        return false;
    }
    for (size_t first = 0; first < quad_entries.size(); first += 4) {
        if (quad_entries[first + 3].surface_edge !=
            quad_entries[first].surface_edge) {
            error = "FlexiCubes stable edge grouping is incomplete";
            return false;
        }
    }

    // The official code appends dual vertices grouped by `num_vd` (then sparse
    // cell order), so assign gamma in that same order before triangulating.
    std::vector<float> dual_gamma(output.positions.size() / 3, 1.0f);
    size_t dual_cursor = 0;
    for (int number_of_duals = 1; number_of_duals <= 4; ++number_of_duals) {
        for (const SurfaceCell& surface : surface_cells) {
            if (flexicubes_tables::kNumVdTable[surface.case_id] !=
                number_of_duals)
                continue;
            const float gamma = normalized_gamma(cube_features, *surface.cell);
            for (int dual = 0; dual < number_of_duals; ++dual)
                dual_gamma[dual_cursor++] = gamma;
        }
    }
    for (size_t first = 0; first < quad_entries.size(); first += 4) {
        const int edge_index = quad_entries[first].surface_edge;
        std::array<uint32_t, 4> quad{quad_entries[first].dual_vertex,
                                     quad_entries[first + 1].dual_vertex,
                                     quad_entries[first + 2].dual_vertex,
                                     quad_entries[first + 3].dual_vertex};
        const UniqueEdge& edge = *surface_edges[edge_index];
        if (vertex_component(*vertex_by_index[edge.key.first], 0) > 0.0f) {
            quad = {quad[0], quad[1], quad[3], quad[2]};
        } else {
            quad = {quad[2], quad[3], quad[1], quad[0]};
        }
        if (dual_gamma[quad[0]] * dual_gamma[quad[2]] >
            dual_gamma[quad[1]] * dual_gamma[quad[3]]) {
            append_triangle(output, quad[0], quad[1], quad[2]);
            append_triangle(output, quad[0], quad[2], quad[3]);
        } else {
            append_triangle(output, quad[0], quad[1], quad[3]);
            append_triangle(output, quad[3], quad[1], quad[2]);
        }
    }
    return true;
}

}  // namespace sam3d
