// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pointnet2_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace openpcdet::ext {

namespace {

float distance2(const float *a, const float *b) {
    const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

struct cell_coordinate {
    int32_t x;
    int32_t y;
    int32_t z;

    bool operator==(const cell_coordinate &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    bool operator<(const cell_coordinate &other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct cell_hash {
    size_t operator()(const cell_coordinate &cell) const {
        // Three independent odd multipliers keep nearby signed coordinates
        // from collapsing into the same buckets.
        const uint64_t x = static_cast<uint32_t>(cell.x);
        const uint64_t y = static_cast<uint32_t>(cell.y);
        const uint64_t z = static_cast<uint32_t>(cell.z);
        return static_cast<size_t>((x * 73856093ULL) ^ (y * 19349663ULL) ^
                                   (z * 83492791ULL));
    }
};

cell_coordinate point_cell(const float *point, float inverse_cell_size) {
    return {
            static_cast<int32_t>(std::floor(point[0] * inverse_cell_size)),
            static_cast<int32_t>(std::floor(point[1] * inverse_cell_size)),
            static_cast<int32_t>(std::floor(point[2] * inverse_cell_size)),
    };
}

bool linear_ball_query(const float *xyz,
                       int64_t point_count,
                       const float *centers,
                       int64_t center_count,
                       float radius,
                       int nsample,
                       std::vector<int32_t> &indices) {
    indices.assign(static_cast<size_t>(center_count) * nsample, -1);
    const float radius2 = radius * radius;
    for (int64_t center = 0; center < center_count; center++) {
        int count = 0;
        int32_t first = -1;
        for (int64_t point = 0; point < point_count && count < nsample;
             point++) {
            if (distance2(centers + center * 3, xyz + point * 3) <= radius2) {
                if (first < 0) first = static_cast<int32_t>(point);
                indices[center * nsample + count++] =
                        static_cast<int32_t>(point);
            }
        }
        for (; count < nsample && first >= 0; count++)
            indices[center * nsample + count] = first;
    }
    return true;
}

}  // namespace

struct ball_query_index::implementation {
    struct span {
        int32_t begin;
        int32_t end;
    };
    struct row {
        cell_coordinate cell;
        int32_t point;
    };

    const float *xyz = nullptr;
    int64_t point_count = 0;
    float cell_size = 0.0f;
    float inverse_cell_size = 0.0f;
    std::vector<int32_t> ordered_points;
    std::unordered_map<cell_coordinate, span, cell_hash> spans;
};

ball_query_index::ball_query_index()
    : impl_(std::make_unique<implementation>()) {}
ball_query_index::~ball_query_index() = default;
ball_query_index::ball_query_index(ball_query_index &&) noexcept = default;
ball_query_index &ball_query_index::operator=(ball_query_index &&) noexcept =
        default;

bool ball_query_index::build(const float *xyz,
                             int64_t point_count,
                             float cell_size,
                             std::string &error) {
    if (!xyz || point_count <= 0 || !std::isfinite(cell_size) ||
        cell_size <= 0.0f) {
        error = "bad ball-query index input";
        return false;
    }
    impl_->xyz = xyz;
    impl_->point_count = point_count;
    impl_->cell_size = cell_size;
    impl_->inverse_cell_size = 1.0f / cell_size;
    impl_->ordered_points.clear();
    impl_->spans.clear();

    std::vector<implementation::row> rows(static_cast<size_t>(point_count));
    for (int64_t point = 0; point < point_count; ++point) {
        rows[point] = {point_cell(xyz + point * 3, impl_->inverse_cell_size),
                       static_cast<int32_t>(point)};
    }
    // Sort once per source cloud. Point indices remain ordered inside each
    // cell, which lets queries reproduce the reference's first-nsample rule
    // with a bounded 27-way merge instead of sorting every candidate set.
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
        return a.cell < b.cell || (!(b.cell < a.cell) && a.point < b.point);
    });
    impl_->ordered_points.resize(static_cast<size_t>(point_count));
    impl_->spans.reserve(static_cast<size_t>(point_count) / 4 + 1);
    int32_t begin = 0;
    while (begin < point_count) {
        int32_t end = begin + 1;
        while (end < point_count && rows[end].cell == rows[begin].cell) ++end;
        impl_->spans.emplace(rows[begin].cell,
                             implementation::span{begin, end});
        for (int32_t row = begin; row < end; ++row)
            impl_->ordered_points[row] = rows[row].point;
        begin = end;
    }
    return true;
}

bool ball_query_index::query(const float *centers,
                             int64_t center_count,
                             float radius,
                             int nsample,
                             std::vector<int32_t> &indices,
                             std::string &error) const {
    indices.clear();
    if (!impl_->xyz || !centers || center_count < 0 || !std::isfinite(radius) ||
        radius < 0.0f || nsample <= 0) {
        error = "bad indexed ball-query input";
        return false;
    }
    // A radius no larger than the build cell size can touch only the center
    // cell and its 26 immediate neighbors. Fall back rather than risk an
    // incomplete search if a caller violates that build/query relationship.
    if (radius > impl_->cell_size) {
        return linear_ball_query(impl_->xyz, impl_->point_count, centers,
                                 center_count, radius, nsample, indices);
    }

    indices.assign(static_cast<size_t>(center_count) * nsample, -1);
    const float radius2 = radius * radius;
    struct cursor {
        int32_t position;
        int32_t end;
    };
    for (int64_t center = 0; center < center_count; ++center) {
        std::array<cursor, 27> cursors{};
        int cursor_count = 0;
        const cell_coordinate origin =
                point_cell(centers + center * 3, impl_->inverse_cell_size);
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const auto found = impl_->spans.find(
                            {origin.x + dx, origin.y + dy, origin.z + dz});
                    if (found != impl_->spans.end())
                        cursors[cursor_count++] = {found->second.begin,
                                                   found->second.end};
                }

        int count = 0;
        int32_t first = -1;
        while (count < nsample) {
            int best_cursor = -1;
            int32_t best_point = std::numeric_limits<int32_t>::max();
            for (int stream = 0; stream < cursor_count; ++stream) {
                if (cursors[stream].position < cursors[stream].end) {
                    const int32_t point =
                            impl_->ordered_points[cursors[stream].position];
                    if (point < best_point) {
                        best_point = point;
                        best_cursor = stream;
                    }
                }
            }
            if (best_cursor < 0) break;
            ++cursors[best_cursor].position;
            if (distance2(centers + center * 3,
                          impl_->xyz + static_cast<int64_t>(best_point) * 3) <=
                radius2) {
                if (first < 0) first = best_point;
                indices[center * nsample + count++] = best_point;
            }
        }
        for (; count < nsample && first >= 0; ++count)
            indices[center * nsample + count] = first;
    }
    return true;
}

bool farthest_point_sample(const float *xyz,
                           int64_t point_count,
                           int sample_count,
                           std::vector<int32_t> &indices,
                           std::string &error) {
    indices.clear();
    if (!xyz || point_count <= 0 || sample_count <= 0 ||
        sample_count > point_count) {
        error = "bad farthest-point-sampling input";
        return false;
    }
    std::vector<float> nearest(point_count,
                               std::numeric_limits<float>::infinity());
    indices.resize(sample_count);
    int32_t selected = 0;
    for (int sample = 0; sample < sample_count; sample++) {
        indices[sample] = selected;
        const float *center = xyz + static_cast<int64_t>(selected) * 3;
        float farthest_distance = -1.0f;
        int32_t farthest = selected;
        for (int64_t point = 0; point < point_count; point++) {
            nearest[point] = std::min(nearest[point],
                                      distance2(center, xyz + point * 3));
            if (nearest[point] > farthest_distance) {
                farthest_distance = nearest[point];
                farthest = static_cast<int32_t>(point);
            }
        }
        selected = farthest;
    }
    return true;
}

bool ball_query(const float *xyz,
                int64_t point_count,
                const float *centers,
                int64_t center_count,
                float radius,
                int nsample,
                std::vector<int32_t> &indices,
                std::string &error) {
    indices.clear();
    if (!xyz || !centers || point_count < 0 || center_count < 0 || radius < 0 ||
        nsample <= 0) {
        error = "bad ball-query input";
        return false;
    }
    return linear_ball_query(xyz, point_count, centers, center_count, radius,
                             nsample, indices);
}

bool group_points(const float *features,
                  int64_t point_count,
                  int channels,
                  const int32_t *indices,
                  int64_t group_count,
                  int samples_per_group,
                  std::vector<float> &output,
                  std::string &error) {
    output.clear();
    if (!features || !indices || point_count < 0 || channels <= 0 ||
        group_count < 0 || samples_per_group <= 0) {
        error = "bad group-points input";
        return false;
    }
    output.assign(
            static_cast<size_t>(group_count) * samples_per_group * channels,
            0.0f);
    for (int64_t group = 0; group < group_count; group++) {
        for (int sample = 0; sample < samples_per_group; sample++) {
            const int32_t source = indices[group * samples_per_group + sample];
            if (source < 0) continue;
            if (source >= point_count) {
                error = "group index outside point array";
                output.clear();
                return false;
            }
            std::copy_n(features + static_cast<int64_t>(source) * channels,
                        channels,
                        output.data() + (group * samples_per_group + sample) *
                                                channels);
        }
    }
    return true;
}

bool three_nearest_neighbors(const float *unknown,
                             int64_t unknown_count,
                             const float *known,
                             int64_t known_count,
                             std::vector<float> &squared_distances,
                             std::vector<int32_t> &indices,
                             std::string &error) {
    squared_distances.clear();
    indices.clear();
    if (!unknown || !known || unknown_count < 0 || known_count <= 0) {
        error = "bad three-nearest-neighbors input";
        return false;
    }
    squared_distances.resize(static_cast<size_t>(unknown_count) * 3);
    indices.resize(static_cast<size_t>(unknown_count) * 3);
    for (int64_t row = 0; row < unknown_count; row++) {
        std::array<std::pair<float, int32_t>, 3> best{
                {{std::numeric_limits<float>::infinity(), 0},
                 {std::numeric_limits<float>::infinity(), 0},
                 {std::numeric_limits<float>::infinity(), 0}}};
        for (int64_t point = 0; point < known_count; point++) {
            const std::pair<float, int32_t> candidate{
                    distance2(unknown + row * 3, known + point * 3),
                    static_cast<int32_t>(point)};
            if (candidate < best[2]) {
                best[2] = candidate;
                if (best[2] < best[1]) std::swap(best[2], best[1]);
                if (best[1] < best[0]) std::swap(best[1], best[0]);
            }
        }
        for (int slot = 0; slot < 3; slot++) {
            if (!std::isfinite(best[slot].first)) best[slot] = best[0];
            squared_distances[row * 3 + slot] = best[slot].first;
            indices[row * 3 + slot] = best[slot].second;
        }
    }
    return true;
}

bool three_interpolate(const float *known_features,
                       int64_t known_count,
                       int channels,
                       const int32_t *indices,
                       const float *weights,
                       int64_t output_count,
                       std::vector<float> &output,
                       std::string &error) {
    output.clear();
    if (!known_features || !indices || !weights || known_count <= 0 ||
        channels <= 0 || output_count < 0) {
        error = "bad three-interpolate input";
        return false;
    }
    output.assign(static_cast<size_t>(output_count) * channels, 0.0f);
    for (int64_t row = 0; row < output_count; row++) {
        for (int slot = 0; slot < 3; slot++) {
            const int32_t source = indices[row * 3 + slot];
            if (source < 0 || source >= known_count) {
                error = "interpolation index outside feature array";
                output.clear();
                return false;
            }
            const float weight = weights[row * 3 + slot];
            for (int channel = 0; channel < channels; channel++) {
                output[row * channels + channel] +=
                        known_features[static_cast<int64_t>(source) * channels +
                                       channel] *
                        weight;
            }
        }
    }
    return true;
}

bool sample_and_group(const float *xyz,
                      int64_t point_count,
                      const float *features,
                      int channels,
                      const int32_t *center_indices,
                      int64_t center_count,
                      float radius,
                      int nsample,
                      bool use_xyz,
                      std::vector<float> &center_xyz,
                      std::vector<int32_t> &group_indices,
                      std::vector<float> &grouped_features,
                      std::string &error,
                      const ball_query_index *query_index) {
    center_xyz.clear();
    group_indices.clear();
    grouped_features.clear();
    if (!xyz || !center_indices || point_count <= 0 || center_count < 0 ||
        radius < 0 || nsample <= 0 || channels < 0 ||
        (channels > 0 && !features) || (!use_xyz && channels == 0)) {
        error = "bad sample-and-group input";
        return false;
    }
    center_xyz.resize(static_cast<size_t>(center_count) * 3);
    for (int64_t center = 0; center < center_count; ++center) {
        const int32_t source = center_indices[center];
        if (source < 0 || source >= point_count) {
            error = "center index outside point array";
            center_xyz.clear();
            return false;
        }
        std::copy_n(xyz + static_cast<int64_t>(source) * 3, 3,
                    center_xyz.data() + center * 3);
    }
    if (query_index) {
        if (!query_index->query(center_xyz.data(), center_count, radius,
                                nsample, group_indices, error))
            return false;
    } else if (!ball_query(xyz, point_count, center_xyz.data(), center_count,
                           radius, nsample, group_indices, error)) {
        return false;
    }
    const int output_channels = (use_xyz ? 3 : 0) + channels;
    grouped_features.assign(
            static_cast<size_t>(center_count) * nsample * output_channels,
            0.0f);
    for (int64_t center = 0; center < center_count; ++center) {
        for (int sample = 0; sample < nsample; ++sample) {
            const int32_t point = group_indices[center * nsample + sample];
            if (point < 0) continue;
            float *destination = grouped_features.data() +
                                 (center * nsample + sample) * output_channels;
            int offset = 0;
            if (use_xyz) {
                for (int axis = 0; axis < 3; ++axis) {
                    destination[axis] =
                            xyz[static_cast<int64_t>(point) * 3 + axis] -
                            center_xyz[center * 3 + axis];
                }
                offset = 3;
            }
            if (channels > 0) {
                std::copy_n(features + static_cast<int64_t>(point) * channels,
                            channels, destination + offset);
            }
        }
    }
    return true;
}

bool feature_propagate(const float *unknown_xyz,
                       int64_t unknown_count,
                       const float *known_xyz,
                       int64_t known_count,
                       const float *known_features,
                       int known_channels,
                       const float *skip_features,
                       int skip_channels,
                       std::vector<float> &output,
                       std::string &error) {
    output.clear();
    if (!unknown_xyz || !known_xyz || !known_features || unknown_count < 0 ||
        known_count <= 0 || known_channels <= 0 || skip_channels < 0 ||
        (skip_channels > 0 && !skip_features)) {
        error = "bad feature-propagation input";
        return false;
    }
    std::vector<float> distances;
    std::vector<int32_t> indices;
    if (!three_nearest_neighbors(unknown_xyz, unknown_count, known_xyz,
                                 known_count, distances, indices, error))
        return false;
    std::vector<float> weights(static_cast<size_t>(unknown_count) * 3);
    for (int64_t row = 0; row < unknown_count; ++row) {
        float denominator = 0.0f;
        for (int slot = 0; slot < 3; ++slot) {
            const float distance =
                    std::sqrt(std::max(0.0f, distances[row * 3 + slot]));
            const float reciprocal = 1.0f / (distance + 1e-8f);
            weights[row * 3 + slot] = reciprocal;
            denominator += reciprocal;
        }
        for (int slot = 0; slot < 3; ++slot)
            weights[row * 3 + slot] /= denominator;
    }
    std::vector<float> interpolated;
    if (!three_interpolate(known_features, known_count, known_channels,
                           indices.data(), weights.data(), unknown_count,
                           interpolated, error))
        return false;
    const int output_channels = known_channels + skip_channels;
    output.resize(static_cast<size_t>(unknown_count) * output_channels);
    for (int64_t row = 0; row < unknown_count; ++row) {
        float *destination = output.data() + row * output_channels;
        std::copy_n(interpolated.data() + row * known_channels, known_channels,
                    destination);
        if (skip_channels > 0) {
            std::copy_n(skip_features + row * skip_channels, skip_channels,
                        destination + known_channels);
        }
    }
    return true;
}

bool vector_pool(const float *support_xyz,
                 int64_t support_count,
                 const float *support_features,
                 int channels,
                 const float *centers,
                 int64_t center_count,
                 const vector_pool_params &params,
                 std::vector<float> &output,
                 std::string &error) {
    output.clear();
    const int reduced_channels = params.reduced_channels;
    if (!support_xyz || !support_features || !centers || support_count < 0 ||
        center_count < 0 || channels <= 0 || reduced_channels <= 0 ||
        channels % reduced_channels != 0 || params.grid_x <= 0 ||
        params.grid_y <= 0 || params.grid_z <= 0 ||
        !std::isfinite(params.max_neighbor_distance) ||
        params.max_neighbor_distance <= 0.0f || params.nsample == 0 ||
        !std::isfinite(params.neighbor_distance_multiplier) ||
        params.neighbor_distance_multiplier <= 0.0f) {
        error = "bad vector-pool input";
        return false;
    }
    const int grid_count = params.grid_x * params.grid_y * params.grid_z;
    const bool interpolate =
            params.mode == vector_pool_mode::local_interpolation;
    const int grid_channels = reduced_channels + (interpolate ? 9 : 3);
    output.assign(
            static_cast<size_t>(center_count) * grid_count * grid_channels,
            0.0f);

    std::vector<float> reduced(
            static_cast<size_t>(support_count) * reduced_channels, 0.0f);
    for (int64_t point = 0; point < support_count; ++point) {
        for (int channel = 0; channel < channels; ++channel) {
            reduced[point * reduced_channels + channel % reduced_channels] +=
                    support_features[point * channels + channel];
        }
    }

    auto is_neighbor = [&](const float *point, const float *center,
                           float distance) {
        const float dx = point[0] - center[0];
        const float dy = point[1] - center[1];
        const float dz = point[2] - center[2];
        if (params.ball_neighbors)
            return dx * dx + dy * dy + dz * dz <= distance * distance;
        return std::fabs(dx) <= distance && std::fabs(dy) <= distance &&
               std::fabs(dz) <= distance;
    };

    std::vector<int32_t> neighbors;
    neighbors.reserve(std::min<int64_t>(support_count, 1000));
    const float radius = params.max_neighbor_distance;
    for (int64_t center_index = 0; center_index < center_count;
         ++center_index) {
        const float *center = centers + center_index * 3;
        neighbors.clear();
        const float query_distance =
                interpolate ? radius * params.neighbor_distance_multiplier
                            : radius;
        for (int64_t point = 0; point < support_count; ++point) {
            if (!is_neighbor(support_xyz + point * 3, center, query_distance))
                continue;
            neighbors.push_back(static_cast<int32_t>(point));
            if ((params.nsample > 0 &&
                 static_cast<int>(neighbors.size()) >= params.nsample) ||
                (interpolate && neighbors.size() >= 1000))
                break;
        }

        if (interpolate) {
            for (int gx = 0; gx < params.grid_x; ++gx) {
                const float offset_x =
                        -radius + (2.0f * gx + 1.0f) * radius / params.grid_x;
                for (int gy = 0; gy < params.grid_y; ++gy) {
                    const float offset_y = -radius + (2.0f * gy + 1.0f) *
                                                             radius /
                                                             params.grid_y;
                    for (int gz = 0; gz < params.grid_z; ++gz) {
                        const float offset_z = -radius + (2.0f * gz + 1.0f) *
                                                                 radius /
                                                                 params.grid_z;
                        const int grid =
                                (gx * params.grid_y + gy) * params.grid_z + gz;
                        float *destination =
                                output.data() + (center_index * grid_count +
                                                 grid) * grid_channels;
                        if (neighbors.empty()) continue;
                        const float grid_center[3] = {center[0] + offset_x,
                                                      center[1] + offset_y,
                                                      center[2] + offset_z};
                        std::array<float, 3> best_distance = {
                                std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity()};
                        std::array<int32_t, 3> best_index = {-1, -1, -1};
                        for (int32_t point : neighbors) {
                            const float candidate = distance2(
                                    grid_center,
                                    support_xyz +
                                            static_cast<int64_t>(point) * 3);
                            if (candidate < best_distance[0]) {
                                best_distance[2] = best_distance[1];
                                best_index[2] = best_index[1];
                                best_distance[1] = best_distance[0];
                                best_index[1] = best_index[0];
                                best_distance[0] = candidate;
                                best_index[0] = point;
                            } else if (candidate < best_distance[1]) {
                                best_distance[2] = best_distance[1];
                                best_index[2] = best_index[1];
                                best_distance[1] = candidate;
                                best_index[1] = point;
                            } else if (candidate < best_distance[2]) {
                                best_distance[2] = candidate;
                                best_index[2] = point;
                            }
                        }
                        if (best_index[1] < 0) {
                            best_index[1] = best_index[0];
                            best_distance[1] = best_distance[0];
                        }
                        if (best_index[2] < 0) {
                            best_index[2] = best_index[0];
                            best_distance[2] = best_distance[0];
                        }
                        float reciprocal[3];
                        float normalizer = 0.0f;
                        for (int slot = 0; slot < 3; ++slot) {
                            reciprocal[slot] =
                                    1.0f /
                                    (std::sqrt(best_distance[slot]) + 1e-8f);
                            normalizer += reciprocal[slot];
                        }
                        for (int channel = 0; channel < reduced_channels;
                             ++channel) {
                            for (int slot = 0; slot < 3; ++slot) {
                                destination[channel] +=
                                        reduced[static_cast<int64_t>(
                                                        best_index[slot]) *
                                                        reduced_channels +
                                                channel] *
                                        reciprocal[slot] / normalizer;
                            }
                        }
                        for (int slot = 0; slot < 3; ++slot) {
                            const float *point =
                                    support_xyz +
                                    static_cast<int64_t>(best_index[slot]) * 3;
                            for (int axis = 0; axis < 3; ++axis) {
                                destination[reduced_channels + slot * 3 +
                                            axis] =
                                        grid_center[axis] - point[axis];
                            }
                        }
                    }
                }
            }
            continue;
        }

        std::vector<int32_t> counts(grid_count, 0);
        int accepted = 0;
        for (int32_t point_index : neighbors) {
            const float *point =
                    support_xyz + static_cast<int64_t>(point_index) * 3;
            const float local[3] = {point[0] - center[0], point[1] - center[1],
                                    point[2] - center[2]};
            const int gx = std::clamp(static_cast<int>(std::floor(
                                              (local[0] + radius) *
                                              params.grid_x / (2.0f * radius))),
                                      0, params.grid_x - 1);
            const int gy = std::clamp(static_cast<int>(std::floor(
                                              (local[1] + radius) *
                                              params.grid_y / (2.0f * radius))),
                                      0, params.grid_y - 1);
            const int gz = std::clamp(static_cast<int>(std::floor(
                                              (local[2] + radius) *
                                              params.grid_z / (2.0f * radius))),
                                      0, params.grid_z - 1);
            const int grid = (gx * params.grid_y + gy) * params.grid_z + gz;
            if (params.mode == vector_pool_mode::voxel_first &&
                counts[grid] != 0)
                continue;
            float *destination = output.data() + (center_index * grid_count +
                                                  grid) * grid_channels;
            ++counts[grid];
            for (int axis = 0; axis < 3; ++axis)
                destination[axis] += local[axis];
            for (int channel = 0; channel < reduced_channels; ++channel) {
                destination[3 + channel] +=
                        reduced[static_cast<int64_t>(point_index) *
                                        reduced_channels +
                                channel];
            }
            ++accepted;
            if ((params.nsample > 0 && accepted >= params.nsample) ||
                (params.mode == vector_pool_mode::voxel_first &&
                 accepted >= grid_count))
                break;
        }
        for (int grid = 0; grid < grid_count; ++grid) {
            if (counts[grid] == 0 ||
                params.mode == vector_pool_mode::voxel_first)
                continue;
            float *destination = output.data() + (center_index * grid_count +
                                                  grid) * grid_channels;
            for (int channel = 0; channel < grid_channels; ++channel)
                destination[channel] /= counts[grid];
        }
    }
    return true;
}

}  // namespace openpcdet::ext
