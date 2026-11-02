// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// SimpleFPN conv tower for the HFRE vt branch (simple_fpn.py, 10B config:
// dim=4096, scale_factors=[8,4,2,1] on stride 32 -> level strides 4/8/16/32,
// out_channels=512, norm="LN" eps 1e-6).
//
// ONE single-output-per-consumer graph (the cross-graph expand trap lesson):
// every level's tail is ggml_cpy-ed into a dedicated ictx leaf whose address
// survives the whole-graph compute, so four levels are readable afterwards.
//
// Layout contract (all internal tensors token-major, ggml ne=(C, W, H):
// address = c + C*(w + W*h), i.e. the memory IS (C, H*W) channel-fastest and
// doubles as NCHW contiguous once the last dim is read as h*W+w order):
//   * Linear/1x1 conv : mul_mat directly on (C_in, H*W)          -> (C_out,
//   H*W)
//   * LayerNorm over C: reshape (C, H*W) -> ggml_norm -> *w + b
//   * conv3x3 pad1    : permute+cont to NCHW ne=(W,H,C) -> ggml_im2col
//                       (k order c*9+ky*3+kx == torch contiguous reshape,
//                       verified in ggml-cpu/ops.cpp) -> mul_mat -> the
//                       (OC, OW, OH) output IS token-major again, no permute.
//   * ConvT k2s2      : ONE mul_mat against the merged (4*out, in) weight
//                       (row k = 2a+b = W[:,:,a,b]^T, baked by the converter)
//                       + in-graph pixel shuffle:
//                         slice views (k=2a, k=2a+1) -> cat(0) -> reshape
//                         (b interleaved into W) -> cat(0) -> permute+cont
//                         -> reshape (a interleaved into H). Every step's
//                         address map was derived and verified by hand.
//   * GELU            : nn.GELU() = exact erf -> ggml_gelu_erf (NOT ggml_gelu).
#pragma once

struct ggml_tensor;  // CUDA RoIAlign device-resident output

#include <string>
#include <vector>

#include "vlx_types.h"

struct ggml_backend;

namespace vlx {

class GgufFile;

struct SimpleFpnConfig {
    int vt_dim = 4096;       // SimpleFPN input channels (merged tower feat)
    int out_channels = 512;  // per-level channels
    int strides[4] = {4, 8, 16, 32};  // finest first
    int roi_output_size = 14;
    bool roi_aligned = true;
};

class SimpleFpnGraph {
public:
    SimpleFpnGraph(ggml_backend* backend,
                   const SimpleFpnConfig& cfg,
                   const GgufFile* gguf);
    ~SimpleFpnGraph();

    // pooler: (N/4, vt_dim) f32 token-major bytes (the mmproj --dump-llm
    // stream); merged grid = (gh/2, gw/2) with row-major (h, w) token order.
    // Writes 4 FeatureLevels (NCHW float, finest first) in out_levels.
    bool run(const float* pooler,
             int grid_h,
             int grid_w,
             std::vector<FeatureLevel>* out_levels,
             std::string* err = nullptr,
             bool keep_on_device = false);

    // GPU-resident output level (persistent ictx leaf; on CUDA builds its
    // data pointer is a device pointer after run()). Lets the CUDA RoIAlign
    // consume the pyramid without a host round-trip. li in [0, 4).
    ggml_tensor* out_level_gpu(int li) const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
    void build_weights();
    void build_graph(int grid_h, int grid_w);
};

}  // namespace vlx
