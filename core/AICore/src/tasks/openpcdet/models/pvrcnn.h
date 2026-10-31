// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../model.h"
#include "anchor_head.h"
#include "bev_backbone.h"
#include "point_mlp.h"
#include "sparse_backbone.h"

namespace openpcdet {

struct pvrcnn final : model {
    sparse_backbone_weights sparse_w;
    bev_backbone_weights backbone_w;
    anchor_head_weights head_w;
    std::vector<std::vector<point_mlp_weights>> pfe_sa;
    point_mlp_weights pfe_fusion;
    point_mlp_weights point_cls;
    std::vector<point_mlp_weights> roi_pool;
    point_mlp_weights roi_shared;
    point_mlp_weights roi_cls;
    point_mlp_weights roi_reg;

    bool load(const std::string& path,
              const std::string& device,
              int threads) override;
    bool forward(const float* points,
                 int32_t count,
                 int32_t dimension,
                 std::vector<detection>& detections,
                 const tap_sink& sink = {}) override;
};

}  // namespace openpcdet
