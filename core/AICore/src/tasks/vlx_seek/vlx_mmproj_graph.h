// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Qwen3.5-VL main tower (mmproj) forward as ONE single-output graph.
//
// Faithful port of transformers qwen3_5 vision (modeling_qwen3_5.py +
// vision_utils.py):
//   patch GEMM (conv3d kt2 flattened, K=1536) + bias + interpolated learned
//   pos embed (48x48 table, bilinear align_corners=True, host-side) -> x (C, N)
//   27 pre-norm blocks (eps 1e-6): norm1 -> qkv -> axial 2D RoPE (rotate_half
//   form) -> flash_attn_ext (no mask, no q/k norm) -> proj -> residual;
//   norm2 -> fc1 -> GELU(tanh) -> fc2 -> residual                      -> x
//   merger: norm(1152) -> view(4608, N/4) (spatial-merge-block order makes
//           this a plain reshape) -> fc1 -> GELU(erf) -> fc2   -> (4096, N/4)
//
// ONE graph, not per-block segments: a cross-graph tensor reference carries
// src pointers into the producer's context, so build_forward_expand pulls the
// whole preceding chain into the consumer graph and every gallocr re-points
// the SHARED tensor objects' data into its own pool (last allocator wins —
// garbage inputs, O(depth^2) work). All inputs/taps live in a dedicated
// ictx buffer (gallocr never re-points them); the HFRE tap is ggml_cpy'd
// into an ictx leaf so its address survives the whole-graph compute.
//
// Layout conventions (parity-verified in radio_graph, same engine):
//   * token-major: a (C, T) ggml tensor stores channel fastest, so its memory
//     equals torch's (T, C) token-major dump byte for byte.
//   * cos/sin are precomputed on the host as (N, 72) token-major (axis layout
//     [cos_h(18) | cos_w(18)] x2), uploaded once and broadcast per block.
//   * RoPE rotation uses basic ops (view + neg + concat), NOT ggml_rope: the
//     axial cat layout pairs dim d with d+36 exactly like rotate_half.
#pragma once

#include <string>
#include <vector>

#include "vlx_types.h"

struct ggml_backend;

namespace vlx {

class GgufFile;

struct MmprojConfig {
    int patch_size = 16;
    int temporal = 2;
    int embed_dim = 1152;
    int depth = 27;
    int num_heads = 16;
    int mlp_hidden = 4304;
    int hidden_out = 4096;
    int merge = 2;
    int pos_side = 48;  // sqrt(num_position_embeddings)
    float rope_theta = 1000000.f;
    float ln_eps = 1e-6f;  // qwen3_5 vision norms (RADIO uses 1e-5!)

    int head_dim() const { return embed_dim / num_heads; }
    int kdim() const { return 3 * temporal * patch_size * patch_size; }
};

class MmprojGraph {
public:
    MmprojGraph(ggml_backend* backend,
                const MmprojConfig& cfg,
                const GgufFile* gguf);
    ~MmprojGraph();

    // patches: token-major (N, K) f32 bytes as written by dump_torch_mmproj
    // input.bin (block-major patch order, CLIP-normalized, temporal
    // duplicated). N must equal grid_h*grid_w.
    bool run(
            const float* patches,
            int grid_h,
            int grid_w,
            std::vector<float>* out_ll_tokens,  // (N/4, hidden_out) token-major
            std::vector<float>* out_final =
                    nullptr,  // (N, C) pre-merger (HFRE tap)
            std::string* err = nullptr);

    // Diagnostic mode: build a dedicated single-output graph containing
    // seg0 + block-0 up to the requested component. `point` is one of
    // "n1", "qkv", "q_rot", "attnbr", "mlpbr", "x". Immune to gallocr
    // reuse ambiguity (radio diag pattern).
    bool run_diag(const float* patches,
                  int grid_h,
                  int grid_w,
                  const char* point,
                  std::vector<float>* out,
                  std::string* err = nullptr);

private:
    struct Seg;  // per-segment graph state (defined in .cpp)
    struct Impl;
    Impl* impl_ = nullptr;
    void build_weights();
    void prepare_host_tables(int grid_h, int grid_w);
    void build_graphs(int grid_h, int grid_w);
};

}  // namespace vlx
