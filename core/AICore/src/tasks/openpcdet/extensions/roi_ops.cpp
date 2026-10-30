#include "roi_ops.h"

#include "../ops/nms_3d.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace openpcdet::ext {

namespace {

bool validate_geometry(const float * points, int64_t point_count,
                       const float * boxes, int64_t box_count,
                       std::string & error) {
    if ((!points && point_count != 0) || point_count < 0 ||
        (!boxes && box_count != 0) || box_count < 0) {
        error = "bad RoI geometry input";
        return false;
    }
    return true;
}

bool local_point(const float * point, const float * box, float local[3]) {
    const float dx = point[0] - box[0];
    const float dy = point[1] - box[1];
    local[2] = point[2] - box[2];
    const float c = std::cos(box[6]);
    const float s = std::sin(box[6]);
    local[0] = dx * c + dy * s;
    local[1] = -dx * s + dy * c;
    const float epsilon = 1e-5f;
    return std::fabs(local[0]) <= box[3] * 0.5f + epsilon &&
           std::fabs(local[1]) <= box[4] * 0.5f + epsilon &&
           std::fabs(local[2]) <= box[5] * 0.5f + epsilon;
}

} // namespace

bool select_proposals(const float * boxes, int64_t box_count,
                      const float * class_scores, int class_count,
                      bool scores_are_normalized, int pre_max, int post_max,
                      float nms_threshold, proposal_result & output,
                      std::string & error) {
    output = {};
    if ((!boxes && box_count != 0) || (!class_scores && box_count != 0) ||
        box_count < 0 || class_count <= 0 || pre_max <= 0 || post_max <= 0 ||
        nms_threshold < 0.0f) {
        error = "bad proposal-selection input"; return false;
    }
    std::vector<int32_t> candidates(static_cast<size_t>(box_count));
    std::vector<float> best_scores(static_cast<size_t>(box_count));
    std::vector<int32_t> best_labels(static_cast<size_t>(box_count));
    for (int64_t box = 0; box < box_count; ++box) {
        float best = -std::numeric_limits<float>::infinity();
        int32_t label = 0;
        for (int cls = 0; cls < class_count; ++cls) {
            float score = class_scores[box * class_count + cls];
            if (!scores_are_normalized) score = 1.0f / (1.0f + std::exp(-score));
            if (score > best) { best = score; label = cls; }
        }
        candidates[box] = static_cast<int32_t>(box);
        best_scores[box] = best;
        best_labels[box] = label + 1;
    }
    const int keep_pre = std::min<int64_t>(box_count, pre_max);
    std::partial_sort(candidates.begin(), candidates.begin() + keep_pre, candidates.end(),
        [&](int32_t lhs, int32_t rhs) {
            if (best_scores[lhs] != best_scores[rhs]) return best_scores[lhs] > best_scores[rhs];
            return lhs < rhs;
        });
    candidates.resize(keep_pre);
    std::vector<float> candidate_boxes(static_cast<size_t>(keep_pre) * 7);
    std::vector<float> candidate_scores(keep_pre);
    for (int index = 0; index < keep_pre; ++index) {
        const int32_t source = candidates[index];
        std::copy_n(boxes + static_cast<int64_t>(source) * 7, 7,
                    candidate_boxes.data() + static_cast<int64_t>(index) * 7);
        candidate_scores[index] = best_scores[source];
    }
    const std::vector<int32_t> kept = openpcdet::nms_3d(
        candidate_boxes, candidate_scores, nms_threshold, post_max);
    output.indices.reserve(kept.size());
    output.scores.reserve(kept.size());
    output.labels.reserve(kept.size());
    for (int32_t local : kept) {
        const int32_t source = candidates[local];
        output.indices.push_back(source);
        output.scores.push_back(best_scores[source]);
        output.labels.push_back(best_labels[source]);
    }
    return true;
}

bool points_in_boxes(const float * points, int64_t point_count,
                     const float * boxes, int64_t box_count,
                     std::vector<uint8_t> & mask, std::string & error) {
    mask.clear();
    if (!validate_geometry(points, point_count, boxes, box_count, error)) return false;
    mask.assign(static_cast<size_t>(box_count * point_count), 0);
    for (int64_t box_idx = 0; box_idx < box_count; box_idx++) {
        const float * box = boxes + box_idx * 7;
        if (box[3] <= 0 || box[4] <= 0 || box[5] <= 0) {
            error = "RoI box dimensions must be positive";
            mask.clear();
            return false;
        }
        for (int64_t point_idx = 0; point_idx < point_count; point_idx++) {
            float local[3];
            mask[box_idx * point_count + point_idx] =
                local_point(points + point_idx * 3, box, local) ? 1 : 0;
        }
    }
    return true;
}

bool roi_aware_pool3d(const float * boxes, int64_t box_count,
                      const float * points, int64_t point_count,
                      const float * features, int channels,
                      int out_x, int out_y, int out_z,
                      int max_points_per_voxel, roi_pool_method method,
                      std::vector<float> & output, std::string & error) {
    output.clear();
    if (!validate_geometry(points, point_count, boxes, box_count, error) ||
        (!features && point_count != 0) || channels <= 0 || out_x <= 0 || out_y <= 0 ||
        out_z <= 0 || max_points_per_voxel <= 0) {
        if (error.empty()) error = "bad RoI-aware-pool input";
        return false;
    }
    const int64_t voxels_per_box = static_cast<int64_t>(out_x) * out_y * out_z;
    output.assign(static_cast<size_t>(box_count * voxels_per_box * channels), 0.0f);
    std::vector<int32_t> counts(static_cast<size_t>(box_count * voxels_per_box), 0);
    if (method == roi_pool_method::max) {
        std::fill(output.begin(), output.end(), -std::numeric_limits<float>::infinity());
    }

    for (int64_t box_idx = 0; box_idx < box_count; box_idx++) {
        const float * box = boxes + box_idx * 7;
        if (box[3] <= 0 || box[4] <= 0 || box[5] <= 0) {
            error = "RoI box dimensions must be positive";
            output.clear();
            return false;
        }
        for (int64_t point_idx = 0; point_idx < point_count; point_idx++) {
            float local[3];
            if (!local_point(points + point_idx * 3, box, local)) continue;
            const auto voxel_index = [](float coordinate, float extent, int bins) {
                const float normalized = (coordinate + extent * 0.5f) / extent;
                return std::min(bins - 1, std::max(0, static_cast<int>(normalized * bins)));
            };
            const int ix = voxel_index(local[0], box[3], out_x);
            const int iy = voxel_index(local[1], box[4], out_y);
            const int iz = voxel_index(local[2], box[5], out_z);
            const int64_t voxel = ((box_idx * out_x + ix) * out_y + iy) * out_z + iz;
            if (counts[voxel] >= max_points_per_voxel) continue;
            float * dst = output.data() + voxel * channels;
            const float * src = features + point_idx * channels;
            if (method == roi_pool_method::max) {
                for (int channel = 0; channel < channels; channel++)
                    dst[channel] = std::max(dst[channel], src[channel]);
            } else {
                for (int channel = 0; channel < channels; channel++) dst[channel] += src[channel];
            }
            counts[voxel]++;
        }
    }
    for (size_t voxel = 0; voxel < counts.size(); voxel++) {
        float * dst = output.data() + voxel * channels;
        if (counts[voxel] == 0) {
            std::fill_n(dst, channels, 0.0f);
        } else if (method == roi_pool_method::average) {
            for (int channel = 0; channel < channels; channel++) dst[channel] /= counts[voxel];
        }
    }
    return true;
}

bool roi_grid_points(const float * boxes, int64_t box_count, int grid_size,
                     std::vector<float> & local_points,
                     std::vector<float> & global_points,
                     std::string & error) {
    local_points.clear();
    global_points.clear();
    if ((!boxes && box_count != 0) || box_count < 0 || grid_size <= 0) {
        error = "bad RoI-grid input";
        return false;
    }
    const int64_t points_per_box = static_cast<int64_t>(grid_size) * grid_size * grid_size;
    local_points.reserve(static_cast<size_t>(box_count * points_per_box * 3));
    global_points.reserve(local_points.capacity());
    for (int64_t box_idx = 0; box_idx < box_count; box_idx++) {
        const float * box = boxes + box_idx * 7;
        if (box[3] <= 0 || box[4] <= 0 || box[5] <= 0) {
            error = "RoI box dimensions must be positive";
            local_points.clear();
            global_points.clear();
            return false;
        }
        const float c = std::cos(box[6]);
        const float s = std::sin(box[6]);
        for (int ix = 0; ix < grid_size; ix++) {
            for (int iy = 0; iy < grid_size; iy++) {
                for (int iz = 0; iz < grid_size; iz++) {
                    const float x = (static_cast<float>(ix) + 0.5f) / grid_size * box[3] - box[3] * 0.5f;
                    const float y = (static_cast<float>(iy) + 0.5f) / grid_size * box[4] - box[4] * 0.5f;
                    const float z = (static_cast<float>(iz) + 0.5f) / grid_size * box[5] - box[5] * 0.5f;
                    local_points.insert(local_points.end(), {x, y, z});
                    global_points.insert(global_points.end(),
                        {x * c - y * s + box[0], x * s + y * c + box[1], z + box[2]});
                }
            }
        }
    }
    return true;
}

bool rotated_bev_roi_align(const float * features, int height, int width,
                           int channels, const float * boxes, int64_t box_count,
                           int grid_size, float min_x, float min_y,
                           float voxel_size_x, float voxel_size_y,
                           int downsample_ratio, std::vector<float> & output,
                           std::string & error) {
    output.clear();
    if (!features || (!boxes && box_count != 0) || height <= 1 || width <= 1 ||
        channels <= 0 || box_count < 0 || grid_size <= 1 ||
        voxel_size_x <= 0.0f || voxel_size_y <= 0.0f || downsample_ratio <= 0) {
        error = "bad rotated BEV RoI-align input";
        return false;
    }
    output.assign(static_cast<size_t>(box_count) * channels * grid_size * grid_size, 0.0f);
    const float x_scale = voxel_size_x * downsample_ratio;
    const float y_scale = voxel_size_y * downsample_ratio;
    auto sample = [&](int y, int x, int channel) {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
        return features[(static_cast<size_t>(y) * width + x) * channels + channel];
    };
    for (int64_t box_index = 0; box_index < box_count; ++box_index) {
        const float * box = boxes + box_index * 7;
        if (box[3] <= 0.0f || box[4] <= 0.0f) {
            error = "rotated BEV RoI dimensions must be positive";
            output.clear();
            return false;
        }
        const float center_x = (box[0] - min_x) / x_scale;
        const float center_y = (box[1] - min_y) / y_scale;
        const float half_x = box[3] * 0.5f / x_scale;
        const float half_y = box[4] * 0.5f / y_scale;
        const float cosine = std::cos(box[6]);
        const float sine = std::sin(box[6]);
        for (int gy = 0; gy < grid_size; ++gy) {
            const float v = 2.0f * gy / (grid_size - 1) - 1.0f;
            for (int gx = 0; gx < grid_size; ++gx) {
                const float u = 2.0f * gx / (grid_size - 1) - 1.0f;
                const float source_x = center_x + half_x * (cosine * u - sine * v);
                const float source_y = center_y + half_y * (sine * u + cosine * v);
                const int x0 = static_cast<int>(std::floor(source_x));
                const int y0 = static_cast<int>(std::floor(source_y));
                const int x1 = x0 + 1, y1 = y0 + 1;
                const float wx = source_x - x0, wy = source_y - y0;
                for (int channel = 0; channel < channels; ++channel) {
                    const float value =
                        sample(y0, x0, channel) * (1.0f - wx) * (1.0f - wy) +
                        sample(y0, x1, channel) * wx * (1.0f - wy) +
                        sample(y1, x0, channel) * (1.0f - wx) * wy +
                        sample(y1, x1, channel) * wx * wy;
                    const size_t target = ((static_cast<size_t>(box_index) * channels + channel) *
                                           grid_size + gy) * grid_size + gx;
                    output[target] = value;
                }
            }
        }
    }
    return true;
}

bool roi_point_pool3d(const float * boxes, int64_t box_count,
                      const float * points, int64_t point_count,
                      const float * features, int channels,
                      const float extra_width[3], int sample_count,
                      std::vector<float> & output,
                      std::vector<uint8_t> & empty_flags,
                      std::string & error) {
    output.clear(); empty_flags.clear();
    if (!validate_geometry(points, point_count, boxes, box_count, error) ||
        (!features && point_count != 0) || !extra_width || channels <= 0 ||
        sample_count <= 0 || extra_width[0] < 0 || extra_width[1] < 0 ||
        extra_width[2] < 0) {
        if (error.empty()) error = "bad RoI-point-pool input";
        return false;
    }
    const int width = 3 + channels;
    output.assign(static_cast<size_t>(box_count) * sample_count * width, 0.0f);
    empty_flags.assign(static_cast<size_t>(box_count), 0);
    std::vector<int32_t> members;
    for (int64_t box_index = 0; box_index < box_count; ++box_index) {
        float enlarged[7];
        std::copy_n(boxes + box_index * 7, 7, enlarged);
        for (int axis = 0; axis < 3; ++axis) enlarged[3 + axis] += 2.0f * extra_width[axis];
        if (enlarged[3] <= 0 || enlarged[4] <= 0 || enlarged[5] <= 0) {
            error = "RoI box dimensions must be positive"; output.clear(); empty_flags.clear();
            return false;
        }
        members.clear();
        for (int64_t point = 0; point < point_count; ++point) {
            float local[3];
            if (local_point(points + point * 3, enlarged, local))
                members.push_back(static_cast<int32_t>(point));
        }
        if (members.empty()) { empty_flags[box_index] = 1; continue; }
        for (int sample = 0; sample < sample_count; ++sample) {
            const int32_t source = members[static_cast<size_t>(sample) % members.size()];
            float * destination = output.data() +
                (box_index * sample_count + sample) * width;
            std::copy_n(points + static_cast<int64_t>(source) * 3, 3, destination);
            std::copy_n(features + static_cast<int64_t>(source) * channels, channels,
                        destination + 3);
        }
    }
    return true;
}

bool canonicalize_roi_points(const float * boxes, int64_t box_count,
                             int sample_count, int feature_width,
                             std::vector<float> & pooled, std::string & error) {
    if ((!boxes && box_count != 0) || box_count < 0 || sample_count <= 0 ||
        feature_width < 3 || pooled.size() !=
        static_cast<size_t>(box_count) * sample_count * feature_width) {
        error = "bad canonical RoI input"; return false;
    }
    for (int64_t box_index = 0; box_index < box_count; ++box_index) {
        const float * box = boxes + box_index * 7;
        const float cosine = std::cos(box[6]);
        const float sine = std::sin(box[6]);
        for (int sample = 0; sample < sample_count; ++sample) {
            float * point = pooled.data() +
                (box_index * sample_count + sample) * feature_width;
            const float dx = point[0] - box[0];
            const float dy = point[1] - box[1];
            point[0] = dx * cosine + dy * sine;
            point[1] = -dx * sine + dy * cosine;
            point[2] -= box[2];
        }
    }
    return true;
}

bool decode_roi_residuals(const float * rois, const float * residuals,
                          int64_t roi_count, std::vector<float> & boxes,
                          std::string & error) {
    boxes.clear();
    if ((!rois && roi_count != 0) || (!residuals && roi_count != 0) || roi_count < 0) {
        error = "bad RoI residual input"; return false;
    }
    boxes.resize(static_cast<size_t>(roi_count) * 7);
    for (int64_t index = 0; index < roi_count; ++index) {
        const float * roi = rois + index * 7;
        const float * residual = residuals + index * 7;
        if (roi[3] <= 0 || roi[4] <= 0 || roi[5] <= 0) {
            error = "RoI dimensions must be positive"; boxes.clear(); return false;
        }
        const float diagonal = std::sqrt(roi[3] * roi[3] + roi[4] * roi[4]);
        const float local_x = residual[0] * diagonal;
        const float local_y = residual[1] * diagonal;
        const float cosine = std::cos(roi[6]);
        const float sine = std::sin(roi[6]);
        float * output_box = boxes.data() + index * 7;
        output_box[0] = local_x * cosine - local_y * sine + roi[0];
        output_box[1] = local_x * sine + local_y * cosine + roi[1];
        output_box[2] = residual[2] * roi[5] + roi[2];
        output_box[3] = std::exp(residual[3]) * roi[3];
        output_box[4] = std::exp(residual[4]) * roi[4];
        output_box[5] = std::exp(residual[5]) * roi[5];
        output_box[6] = residual[6] + roi[6];
    }
    return true;
}

} // namespace openpcdet::ext
