// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "center_head_ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "../ops/nms_3d.h"

namespace openpcdet::ext {

namespace {

float sigmoid(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

struct peak {
    float score;
    int32_t location;
    int32_t class_id;
};

}  // namespace

bool decode_center_head(const center_head_view& head,
                        const center_decode_params& params,
                        std::vector<detection>& output,
                        std::string& error) {
    output.clear();
    if (!head.heatmap || !head.center || !head.center_z || !head.dimensions ||
        !head.rotation || head.class_count <= 0 ||
        ((!head.coordinates && (head.height <= 0 || head.width <= 0)) ||
         (head.coordinates && head.active_count <= 0)) ||
        head.class_mapping.size() != static_cast<size_t>(head.class_count)) {
        error = "bad CenterHead tensor view";
        return false;
    }
    if (params.point_cloud_range.size() != 6 || params.voxel_size.size() != 3 ||
        params.post_center_limit_range.size() != 6 ||
        params.feature_map_stride <= 0 || params.max_objects <= 0 ||
        params.nms_pre_maxsize <= 0 || params.nms_post_maxsize <= 0) {
        error = "bad CenterHead decode parameters";
        return false;
    }
    if (head.iou &&
        (params.iou_rectifier.empty() ||
         params.class_nms_thresholds.size() != params.iou_rectifier.size() ||
         params.class_nms_pre_maxsizes.size() != params.iou_rectifier.size() ||
         params.class_nms_post_maxsizes.size() !=
                 params.iou_rectifier.size())) {
        error = "bad CenterHead IoU rectification parameters";
        return false;
    }

    const int32_t locations =
            head.coordinates ? head.active_count : head.height * head.width;
    // Pre-filter with the logit equivalent of the score threshold: sigmoid is
    // monotonic, so a peak rejected here could never pass the top-K decode
    // gate either. This collapses the sort set from the full dense grid
    // (e.g. 32k x classes per head) to the handful of real candidates.
    const float score_threshold = params.score_threshold;
    const float logit_threshold =
            head.heatmap_is_probability
                    ? score_threshold
                    : (score_threshold <= 0.0f
                               ? -std::numeric_limits<float>::infinity()
                               : std::log(score_threshold /
                                          (1.0f - score_threshold)));
    std::vector<peak> peaks;
    peaks.reserve(static_cast<size_t>(locations) * head.class_count);
    // Collect raw logits: sigmoid is monotonic, so the top-K order is identical
    // on logits, and deferring the expf to the small top-K loop below avoids
    // millions of exp() calls per head on dense grids. Iterate in layout order
    // so each pass is a single sequential walk (WHC: class-major, one full
    // contiguous sweep per class; HWC: location-major, one sweep per row),
    // which the prefetcher and the vectorizer can both exploit.
    if (head.whc_layout) {
        for (int32_t class_id = 0; class_id < head.class_count; class_id++) {
            const float* logits =
                    head.heatmap + static_cast<size_t>(class_id) * locations;
            for (int32_t location = 0; location < locations; location++) {
                const float logit = logits[location];
                if (logit < logit_threshold) continue;
                peaks.push_back({logit, location, class_id});
            }
        }
    } else {
        for (int32_t location = 0; location < locations; location++) {
            const float* logits = head.heatmap + static_cast<size_t>(location) *
                                                         head.class_count;
            for (int32_t class_id = 0; class_id < head.class_count;
                 class_id++) {
                const float logit = logits[class_id];
                if (logit < logit_threshold) continue;
                peaks.push_back({logit, location, class_id});
            }
        }
    }
    const size_t top_count =
            std::min(peaks.size(), static_cast<size_t>(params.max_objects));
    std::partial_sort(peaks.begin(), peaks.begin() + top_count, peaks.end(),
                      [](const peak& a, const peak& b) {
                          if (a.score != b.score) return a.score > b.score;
                          if (a.class_id != b.class_id)
                              return a.class_id < b.class_id;
                          return a.location < b.location;
                      });
    peaks.resize(top_count);

    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int32_t> labels;
    std::vector<float> ious;
    boxes.reserve(top_count * 7);
    scores.reserve(top_count);
    labels.reserve(top_count);
    for (const peak& current : peaks) {
        // Apply sigmoid only to the surviving top-K candidates.
        const float score = head.heatmap_is_probability
                                    ? current.score
                                    : sigmoid(current.score);
        if (!(score > params.score_threshold)) continue;
        const int32_t y = head.coordinates
                                  ? head.coordinates[current.location * 3 + 1]
                                  : current.location / head.width;
        const int32_t x = head.coordinates
                                  ? head.coordinates[current.location * 3 + 2]
                                  : current.location % head.width;
        // Channel lookup: HWC rows are channel-adjacent, WHC columns are
        // location-adjacent (contiguous walks in the dense case).
        auto at = [&](const float* data, int channels, int channel) {
            return head.whc_layout
                           ? data[current.location +
                                  static_cast<size_t>(channel) * locations]
                           : data[static_cast<size_t>(current.location) *
                                          channels +
                                  channel];
        };
        const float center_x = (x + at(head.center, 2, 0)) *
                                       params.feature_map_stride *
                                       params.voxel_size[0] +
                               params.point_cloud_range[0];
        const float center_y = (y + at(head.center, 2, 1)) *
                                       params.feature_map_stride *
                                       params.voxel_size[1] +
                               params.point_cloud_range[1];
        const float center_z = at(head.center_z, 1, 0);
        if (center_x < params.post_center_limit_range[0] ||
            center_y < params.post_center_limit_range[1] ||
            center_z < params.post_center_limit_range[2] ||
            center_x > params.post_center_limit_range[3] ||
            center_y > params.post_center_limit_range[4] ||
            center_z > params.post_center_limit_range[5])
            continue;
        boxes.insert(boxes.end(),
                     {
                             center_x,
                             center_y,
                             center_z,
                             head.dimensions_are_linear
                                     ? at(head.dimensions, 3, 0)
                                     : std::exp(at(head.dimensions, 3, 0)),
                             head.dimensions_are_linear
                                     ? at(head.dimensions, 3, 1)
                                     : std::exp(at(head.dimensions, 3, 1)),
                             head.dimensions_are_linear
                                     ? at(head.dimensions, 3, 2)
                                     : std::exp(at(head.dimensions, 3, 2)),
                             std::atan2(at(head.rotation, 2, 1),
                                        at(head.rotation, 2, 0)),
                     });
        scores.push_back(score);
        labels.push_back(head.class_mapping[current.class_id] + 1);
        if (head.iou) {
            ious.push_back(std::clamp(
                    (head.iou[current.location] + 1.0f) * 0.5f, 0.0f, 1.0f));
        }
    }

    if (head.iou) {
        for (size_t class_id = 0; class_id < params.iou_rectifier.size();
             ++class_id) {
            std::vector<int32_t> selected;
            std::vector<float> class_boxes;
            std::vector<float> class_scores;
            for (size_t i = 0; i < scores.size(); ++i) {
                if (labels[i] != static_cast<int32_t>(class_id + 1)) continue;
                selected.push_back(static_cast<int32_t>(i));
                class_boxes.insert(class_boxes.end(), boxes.begin() + i * 7,
                                   boxes.begin() + i * 7 + 7);
                const float r = params.iou_rectifier[class_id];
                class_scores.push_back(std::pow(scores[i], 1.0f - r) *
                                       std::pow(ious[i], r));
            }
            const int pre_max = params.class_nms_pre_maxsizes[class_id];
            if (static_cast<int>(selected.size()) > pre_max) {
                std::vector<int32_t> order(selected.size());
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + pre_max,
                                  order.end(), [&](int32_t a, int32_t b) {
                                      return class_scores[a] > class_scores[b];
                                  });
                std::vector<int32_t> trimmed_selected;
                std::vector<float> trimmed_boxes, trimmed_scores;
                for (int i = 0; i < pre_max; ++i) {
                    const int row = order[i];
                    trimmed_selected.push_back(selected[row]);
                    trimmed_boxes.insert(trimmed_boxes.end(),
                                         class_boxes.begin() + row * 7,
                                         class_boxes.begin() + row * 7 + 7);
                    trimmed_scores.push_back(class_scores[row]);
                }
                selected = std::move(trimmed_selected);
                class_boxes = std::move(trimmed_boxes);
                class_scores = std::move(trimmed_scores);
            }
            const std::vector<int32_t> kept =
                    nms_3d(class_boxes, class_scores,
                           params.class_nms_thresholds[class_id],
                           params.class_nms_post_maxsizes[class_id]);
            for (int32_t local : kept) {
                const int32_t index = selected[local];
                const float* box =
                        boxes.data() + static_cast<size_t>(index) * 7;
                detection result{};
                result.x = box[0];
                result.y = box[1];
                result.z = box[2];
                result.dx = box[3];
                result.dy = box[4];
                result.dz = box[5];
                result.heading = box[6];
                result.score = class_scores[local];
                result.label = labels[index];
                output.push_back(result);
            }
        }
        std::sort(output.begin(), output.end(),
                  [](const detection& a, const detection& b) {
                      return a.score > b.score;
                  });
        return true;
    }

    if (scores.size() > static_cast<size_t>(params.nms_pre_maxsize)) {
        boxes.resize(static_cast<size_t>(params.nms_pre_maxsize) * 7);
        scores.resize(params.nms_pre_maxsize);
        labels.resize(params.nms_pre_maxsize);
    }
    const std::vector<int32_t> kept = nms_3d(
            boxes, scores, params.nms_threshold, params.nms_post_maxsize);
    output.reserve(kept.size());
    for (int32_t index : kept) {
        const float* box = boxes.data() + static_cast<size_t>(index) * 7;
        detection result{};
        result.x = box[0];
        result.y = box[1];
        result.z = box[2];
        result.dx = box[3];
        result.dy = box[4];
        result.dz = box[5];
        result.heading = box[6];
        result.score = scores[index];
        result.label = labels[index];
        output.push_back(result);
    }
    return true;
}

}  // namespace openpcdet::ext
