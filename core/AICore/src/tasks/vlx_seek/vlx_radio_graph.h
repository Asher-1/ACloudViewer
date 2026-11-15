// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// C-RADIOv4-SO400M forward, split into 4 single-output segment graphs.
//
// Rationale (verified empirically): a single multi-output graph under
// ggml_gallocr does not reliably protect mid-graph tap buffers on the pinned
// v0.21 build, while a segment graph with exactly ONE terminal output
// (llama.cpp logits_all pattern) is safe. Segments are cut at out_indices;
// the segment output x (full token stream incl. cls/register prefix) is the
// next segment's input (cross-graph tensor reference; the previous segment's
// gallocr buffer keeps it alive and the next segment's gallocr skips it).
//
// In-graph token layout: channel on ne[0], tokens on ne[1]. All layout
// conventions below were verified by per-stage parity against torch
// (patchembed 1.4e-7, postpos/postcls 7e-11, blk0 interior 1.6e-6).
#pragma once

#include <string>
#include <vector>

#include "tasks/vlx_seek/vlx_types.h"

struct ggml_backend;
struct ggml_context;
struct ggml_cgraph;
struct ggml_gallocr;
struct ggml_tensor;
struct ggml_backend_buffer;

namespace vlx {

class GgufFile;

class RadioGraph {
public:
    RadioGraph(ggml_backend* backend,
               const RadioConfig& cfg,
               const GgufFile* gguf);
    ~RadioGraph();

    // img_hwc: row-major H*W*3 float, channels last, values in [0,1].
    // Conditioning (checkpoint mean/std, bf16 output) happens on the host in
    // run(), matching the official InputConditioner bit for bit.
    // out_levels receives the 4 intermediate levels as NCHW float.
    bool run(const float* img_hwc,
             int H,
             int W,
             std::vector<FeatureLevel>* out_levels,
             std::string* err = nullptr,
             bool keep_on_device = false);

    // GPU-resident output level (persistent leaf; device pointer on CUDA
    // builds after run()) for the fused RoIAlign kernel. li in [0, 4),
    // ordered finest-first like out_levels.
    ggml_tensor* out_level_gpu(int li) const;

    // Diagnostic mode: feed the verified postcls token stream (T rows of C,
    // token-major bytes) and read back ONE block-0 component. Single-output
    // graph per call — immune to gallocr reuse ambiguity. `point` is one of
    // "n1", "qkv", "attnbr", "mlpbr", "x".
    bool run_diag(const float* postcls_tokens,
                  const char* point,
                  int H,
                  int W,
                  std::vector<float>* out,
                  std::string* err = nullptr);

private:
    struct Impl;

    bool ensure_graph(int H, int W, std::string* err);
    void prepare_host_tables(int H, int W);
    void build_weights();
    bool ensure_weights(std::string* err);
    void build_graphs(int H, int W);
    ggml_tensor* build_block(ggml_context* ctx,
                             int i,
                             ggml_tensor* x,
                             float attn_scale);

    Impl* impl_ = nullptr;

    // Host-prepared per-resolution inputs.
    std::vector<int32_t> host_pids_;
    std::vector<float> host_pos_;
};

}  // namespace vlx
