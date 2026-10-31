// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "../backend.h"
#include "../model.h"
#include "anchor_head.h"
#include "anchor_head_multi.h"
#include "bev_backbone.h"
#include "center_head.h"

namespace openpcdet {

// Persistent execution state for run_dense_center_pipeline: the graph
// topology is identical across frames (fixed spatial grid), so the ctx +
// graph + gallocr allocation are built once and reused. Populated lazily on
// the first forward; destroyed when the owning model goes away. Non-copyable:
// the ctx owns all tensor memory, a copy would double-free it.
struct dense_center_cache {
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_tensor *spatial = nullptr;  // input upload target
    bool direct_whc_input = false;
    std::vector<center_single_outputs> outputs;
    // Reused device->host readback scratch (avoids per-frame reallocation).
    std::vector<std::vector<float>> readback_buffers;

    ~dense_center_cache();

    // Drops all references without freeing: the caller still owns the ctx
    // and must ggml_free it (avoids double-free when a frame fails mid-run).
    void invalidate();

    dense_center_cache() = default;
    dense_center_cache(const dense_center_cache &) = delete;
    dense_center_cache &operator=(const dense_center_cache &) = delete;
};

struct dense_anchor_result {
    std::vector<float> spatial_features_2d;  // optional [H,W,C]
    int feature_width = 0;
    int feature_height = 0;
    int feature_channels = 0;
    std::vector<float> boxes;  // [anchor_count,7]
    std::vector<float> scores;
    std::vector<int32_t> labels;  // one-based
};

bool run_dense_anchor_raw(engine_backend &backend,
                          const hparams &hp,
                          const bev_backbone_weights &backbone,
                          const anchor_head_weights &head,
                          const std::vector<float> &spatial_hwc,
                          int width,
                          int height,
                          bool keep_features,
                          dense_anchor_result &result,
                          const tap_sink &sink,
                          std::string &error);

bool postprocess_dense_anchors(const dense_anchor_result &input,
                               float score_threshold,
                               int pre_max,
                               int post_max,
                               float nms_threshold,
                               std::vector<detection> &detections,
                               std::string &error);

// Runs BaseBEVBackbone + AnchorHeadSingle + CPU decode/NMS. spatial_hwc is
// contiguous [height, width, num_bev_features].
bool run_dense_anchor_pipeline(engine_backend &backend,
                               const hparams &hp,
                               const bev_backbone_weights &backbone,
                               const anchor_head_weights &head,
                               const std::vector<float> &spatial_hwc,
                               int width,
                               int height,
                               std::vector<detection> &detections,
                               const tap_sink &sink,
                               std::string &error);

bool run_dense_center_pipeline(engine_backend &backend,
                               const hparams &hp,
                               const bev_backbone_weights &backbone,
                               const center_head_weights &head,
                               const std::vector<float> &spatial_hwc,
                               int width,
                               int height,
                               std::vector<detection> &detections,
                               const tap_sink &sink,
                               std::string &error,
                               dense_center_cache *cache = nullptr,
                               bool spatial_device_ready = false,
                               bool direct_whc_input = false);

bool run_dense_anchor_multi_pipeline(engine_backend &backend,
                                     const hparams &hp,
                                     const bev_backbone_weights &backbone,
                                     const anchor_multi_weights &head,
                                     const std::vector<float> &spatial_hwc,
                                     int width,
                                     int height,
                                     std::vector<detection> &detections,
                                     const tap_sink &sink,
                                     std::string &error);

}  // namespace openpcdet
