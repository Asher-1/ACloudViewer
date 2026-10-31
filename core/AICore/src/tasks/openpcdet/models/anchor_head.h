// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AnchorHeadSingle — the dense head for PointPillar/SECOND (pcdet/models/
// dense_heads/anchor_head_single.py).
//
// Three 1x1 conv heads over the backbone's spatial_features_2d:
//   conv_cls:     [in, num_anchors_per_loc * num_class]      (with bias)
//   conv_box:     [in, num_anchors_per_loc * code_size]      (with bias)
//   conv_dir_cls: [in, num_anchors_per_loc * num_dir_bins]   (with bias,
//   optional)
// Outputs are permuted from [N, C, H, W] -> [N, H, W, C] (PyTorch does
// `.permute(0, 2, 3, 1).contiguous()`).
//
// Anchor decoding and direction classification happen on the CPU side after
// the graph (see anchor_decode()), since they depend on the per-scene anchor
// grid and are cheap point-wise ops.
#pragma once

#include <string>
#include <vector>

#include "../gguf_loader.h"
#include "../model.h"

namespace openpcdet {

struct anchor_head_weights {
    ggml_tensor* conv_cls_w = nullptr;  // [1, 1, in_C, A*num_class]
    ggml_tensor* conv_cls_b = nullptr;  // [A*num_class]
    ggml_tensor* conv_box_w = nullptr;  // [1, 1, in_C, A*code_size]
    ggml_tensor* conv_box_b = nullptr;  // [A*code_size]
    ggml_tensor* conv_dir_cls_w =
            nullptr;  // [1, 1, in_C, A*num_dir_bins] (or null)
    ggml_tensor* conv_dir_cls_b = nullptr;
};

bool load_anchor_head_weights(model_file& file,
                              const hparams& hp,
                              anchor_head_weights& w,
                              std::string& err);

// Build the head subgraph. spatial_2d is the backbone output [W, H, C, N].
// Outputs: cls / box / dir tensors, each [W, H, A*C', N] (already permuted
// to match PyTorch's NHWC layout). dir is nullptr if direction classifier
// is disabled.
struct head_outputs {
    ggml_tensor* cls = nullptr;
    ggml_tensor* box = nullptr;
    ggml_tensor* dir = nullptr;
};
head_outputs build_anchor_head_graph(ggml_context* ctx,
                                     const anchor_head_weights& w,
                                     const hparams& hp,
                                     ggml_tensor* spatial_2d);

// --- anchor generation + box decoding (CPU) ---
// Generate the anchor grid for one feature map. Matches AnchorGenerator in
// pcdet/models/dense_heads/target_assigner/anchor_generator.py. Returns
// anchors of shape [A_total, 7] (x,y,z,dx,dy,dz,heading) in the OpenPCDet
// convention (z already shifted to box center).
struct anchor_grid {
    std::vector<float> anchors;  // [A_total * 7] row-major
    int32_t total = 0;
    int32_t feature_map_w = 0;
    int32_t feature_map_h = 0;
};

// AnchorHeadSingle concatenates all configured anchor sets at every feature
// location. All sets must target this same feature-map size.
anchor_grid generate_anchors(const hparams& hp,
                             int feature_map_w,
                             int feature_map_h);

// AnchorHeadMulti uses anchor-major order per selected anchor set:
// [set][size][rotation][height][y][x].
anchor_grid generate_multihead_anchors(const hparams& hp,
                                       const std::vector<int32_t>& set_indices,
                                       int feature_map_w,
                                       int feature_map_h);

// Convert a contiguous ggml [W,H,C,N=1] buffer (W is the fastest dimension)
// to PyTorch's contiguous [H,W,C] order used by head view/decode.
std::vector<float> whc_to_hwc(const float* src, int W, int H, int C);

// Decode box encodings: applies ResidualCoder.decode_torch to [N, A, code_size]
// encodings against [A, 7] anchors. Returns [N, A, 7] boxes.
std::vector<float> decode_boxes(const hparams& hp,
                                const float* box_enc,
                                int32_t N,
                                int32_t A,
                                const float* anchors);

// Apply direction classifier: pick the bin per anchor, add the bin offset to
// the heading. Mutates `boxes` in place. dir_cls is [N, A, num_dir_bins].
void apply_direction_classifier(const hparams& hp,
                                const float* dir_cls,
                                int32_t N,
                                int32_t A,
                                std::vector<float>& boxes);

}  // namespace openpcdet
