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
#include "sparse_unet.h"

namespace openpcdet {

struct parta2 final : model {
    sparse_unet_weights unet;
    bev_backbone_weights backbone;
    anchor_head_weights anchor_head;
    point_mlp_weights point_cls;
    point_mlp_weights point_part;
    std::unordered_map<std::string, sparse_unet_layer> roi_sparse;
    point_mlp_weights roi_shared;
    point_mlp_weights roi_cls_hidden;
    point_mlp_weights roi_cls_output;
    point_mlp_weights roi_reg_hidden;
    point_mlp_weights roi_reg_output;

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
