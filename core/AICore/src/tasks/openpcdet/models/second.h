// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../model.h"
#include "anchor_head.h"
#include "anchor_head_multi.h"
#include "bev_backbone.h"
#include "center_head.h"
#include "dense_anchor_pipeline.h"
#include "point_mlp.h"
#include "sparse_backbone.h"
#include "voxel_backbone_runner.h"

namespace openpcdet {

struct second : model {
    sparse_backbone_weights sparse_w;
    bev_backbone_weights backbone_w;
    anchor_head_weights head_w;
    anchor_multi_weights multi_head_w;
    center_head_weights center_head_w;
    point_mlp_weights roi_shared_w;
    point_mlp_weights roi_iou_w;
    // Reused CenterHead graph across frames (CenterPoint voxel runs here).
    dense_center_cache center_cache;
    sparse_graph_cache sparse_cache;

    bool load(const std::string& gguf_path,
              const std::string& device,
              int n_threads) override;
    bool forward(const float* points,
                 int32_t n_points,
                 int32_t point_dim,
                 std::vector<detection>& detections,
                 const tap_sink& sink = {}) override;
};

}  // namespace openpcdet
