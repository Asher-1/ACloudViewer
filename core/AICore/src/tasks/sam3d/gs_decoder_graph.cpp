// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SLat Gaussian decoder graph - see gs_decoder_graph.hpp.
#include "gs_decoder_graph.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "common.hpp"
#include "graph_builder.hpp"

namespace sam3d {
namespace {

ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32
                   ? tensor
                   : ggml_cast(ctx, tensor, GGML_TYPE_F32);
}

ggml_tensor* as_f16(ggml_context* ctx, ggml_tensor* tensor) {
    // CUDA does not implement a direct Q4/Q8 -> F16 copy.  The F16 torso
    // still needs quantized GGUF deployment weights, so make the supported
    // dequantize-to-F32 then materialize-F16 boundary explicit.
    if (ggml_is_quantized(tensor->type))
        tensor = ggml_cast(ctx, tensor, GGML_TYPE_F32);
    return tensor->type == GGML_TYPE_F16
                   ? tensor
                   : ggml_cast(ctx, tensor, GGML_TYPE_F16);
}

// SLatGaussianDecoder converts only its Transformer blocks to F16.  CUDA
// Linear accumulates in F32 then materializes F16, including after the bias
// addition.  Keep that observable boundary instead of allowing ggml's
// backend-default storage or accumulation type to change the recurrence.
ggml_tensor* linear_fp16(ggml_context* ctx,
                         ggml_tensor* weight,
                         ggml_tensor* bias,
                         ggml_tensor* input) {
    ggml_tensor* output =
            ggml_mul_mat(ctx, as_f16(ctx, weight), as_f16(ctx, input));
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    if (bias != nullptr)
        output = ggml_add(ctx, output, as_f32(ctx, as_f16(ctx, bias)));
    return as_f16(ctx, output);
}

ggml_tensor* residual_fp16(ggml_context* ctx,
                           ggml_tensor* left,
                           ggml_tensor* right) {
    return as_f16(ctx, ggml_add(ctx, as_f32(ctx, left), as_f32(ctx, right)));
}

// ggml get_rows materializes a non-index gather as F32.  Sparse PyTorch
// indexing preserves the activation dtype, so restore the F16 boundary.
ggml_tensor* get_rows_preserving_type(ggml_context* ctx,
                                      ggml_tensor* source,
                                      ggml_tensor* indices) {
    ggml_tensor* rows = ggml_get_rows(ctx, source, indices);
    return source->type == GGML_TYPE_F16 ? as_f16(ctx, rows) : rows;
}

}  // namespace

std::vector<ggml_tensor*> GsDecoderGraph::build() {
    ggml_context* ctx = g->ctx();
    const int64_t C = m->i32(prefix + ".model_channels", 768);
    const int64_t n_blocks = m->i32(prefix + ".num_blocks", 12);
    const int64_t n_heads = m->i32(prefix + ".num_heads", 12);
    const int64_t Hd = C / n_heads;
    const int64_t in_ch = m->i32(prefix + ".latent_channels", 8);
    const int64_t out_ch = m->i32(prefix + ".out_channels", 448);
    const int64_t N = x->ne[1];
    // Legacy GGUFs predate this key, but the released Gaussian decoder YAML
    // unambiguously specifies use_fp16: true.  A future exporter writes the
    // key so models with a different official contract can opt out.
    const bool official_fp16 = m->i32(prefix + ".use_fp16", 1) != 0;
    GGML_ASSERT(x->ne[0] == in_ch);

    auto add_i32 = [&](const std::vector<int32_t>& v, const char* name,
                       std::initializer_list<int64_t> ne) {
        ggml_tensor* t = g->input_i32(name, ne);
        inputs.push_back(t);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(v));
        return t;
    };

    // ---- input layer + APE ----------------------------------------------
    // SparseTransformerBase leaves input_layer in F32, then adds a F32 APE.
    ggml_tensor* h =
            gb_linear(ctx, as_f32(ctx, m->get(prefix + ".input_layer.weight")),
                      as_f32(ctx, m->get(prefix + ".input_layer.bias")), x);
    if (debug_stage == "input_layer") return {h};
    {
        ggml_tensor* ape = g->input_f32("ape", {C, N});
        inputs.push_back(ape);
        // upload path: f32 data reinterpreted as the shared i32 vector storage
        // (same pattern as SlatFlowGraph)
        table_data.push_back(std::make_shared<std::vector<int32_t>>(
                (const int32_t*)tb->ape.data(),
                (const int32_t*)(tb->ape.data() + tb->ape.size())));
        h = ggml_add(ctx, h, ape);
    }
    if (debug_stage == "ape") return {h};
    if (official_fp16) h = as_f16(ctx, h);
    if (debug_stage == "torso_input") return {h};

    // The custom CUDA kernel takes one bucket-ordered token sequence and a
    // list of window lengths, exactly matching the block-diagonal mask used
    // by the official memory-efficient attention call.
    std::array<ggml_tensor*, 2> attention_sequence_lengths = {nullptr, nullptr};
    if (official_fp16 && use_pytorch_cuda_attention) {
        for (size_t shift = 0; shift < attention_sequence_lengths.size();
             ++shift) {
            const GsWindowTables& windows = tb->shifts[shift];
            std::vector<int32_t> lengths;
            lengths.reserve(windows.buckets.size());
            int64_t covered = 0;
            for (const GsWindowTables::Bucket& bucket : windows.buckets) {
                lengths.insert(lengths.end(), static_cast<size_t>(bucket.n_win),
                               bucket.len);
                covered += static_cast<int64_t>(bucket.len) * bucket.n_win;
            }
            GGML_ASSERT(covered == N && !lengths.empty());
            attention_sequence_lengths[shift] =
                    add_i32(lengths, "gs_attention_sequence_lengths",
                            {static_cast<int64_t>(lengths.size())});
        }
    }

    // ---- transformer blocks (swin windowed attention) -------------------
    for (int i = 0; i < (int)n_blocks; i++) {
        const std::string b = prefix + ".blocks." + std::to_string(i);
        const GsWindowTables& wt = tb->shifts[i % 2];
        ggml_tensor* h_in = h;  // block input (for the same-run debug dump)

        ggml_tensor* bkt_idx = add_i32(wt.bkt_idx, "gs_bkt_idx", {N});
        ggml_tensor* bwd_idx = add_i32(wt.bwd_idx, "gs_bwd_idx", {N});

        // LayerNorm32 normalizes in F32 but returns the input dtype.
        ggml_tensor* h1 = ggml_norm(ctx, as_f32(ctx, h), 1e-6f);
        if (official_fp16) h1 = as_f16(ctx, h1);
        if (i == 0 && debug_stage == "b0_norm1") return {h1};
        ggml_tensor* hb = get_rows_preserving_type(ctx, h1, bkt_idx);  // (C, N)

        // PyTorch performs one fused QKV F16 projection before splitting.
        // Three independent projections round differently on Tensor Cores.
        ggml_tensor* qkv_weight = m->get(b + ".attn.to_qkv.weight");
        ggml_tensor* qkv_bias = m->get(b + ".attn.to_qkv.bias");
        ggml_tensor* qkv = official_fp16
                                   ? linear_fp16(ctx, qkv_weight, qkv_bias, hb)
                                   : gb_linear(ctx, as_f32(ctx, qkv_weight),
                                               as_f32(ctx, qkv_bias), hb);
        const size_t qkv_element_size = ggml_type_size(qkv->type);
        auto qkv_view = [&](int index) {
            return ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1],
                                               static_cast<size_t>(index * C) *
                                                       qkv_element_size));
        };
        ggml_tensor* q = qkv_view(0);
        ggml_tensor* k = qkv_view(1);
        ggml_tensor* v = qkv_view(2);
        if (i == 0 && debug_stage == "b0_qkv")
            return {get_rows_preserving_type(ctx, q, bwd_idx),
                    get_rows_preserving_type(ctx, k, bwd_idx),
                    get_rows_preserving_type(ctx, v, bwd_idx)};

        // heads: (C, N) memory (n, (h,d)) -> ne (Hd, H, N)
        auto to_heads = [&](ggml_tensor* t) {
            return ggml_reshape_3d(ctx, t, Hd, n_heads, N);
        };
        ggml_tensor* qh = to_heads(q);
        ggml_tensor* kh = to_heads(k);
        ggml_tensor* vh = to_heads(v);
        const float scale = 1.0f / sqrtf((float)Hd);

        ggml_tensor* acc = nullptr;
        if (official_fp16 && use_pytorch_cuda_attention) {
            GGML_ASSERT(qh->type == GGML_TYPE_F16 &&
                        kh->type == GGML_TYPE_F16 && vh->type == GGML_TYPE_F16);
            acc = ggml_sam3d_f16_masked_attn(
                    ctx, qh, kh, vh, attention_sequence_lengths[i % 2], scale);
            acc = ggml_reshape_2d(ctx, acc, C, N);
        } else {
            // Portable backend formulation of the same block-diagonal
            // attention.  Its F32 reductions are rounded to F16 at the
            // official attention-output boundary.
            for (const auto& bk : wt.buckets) {
                const int64_t off = bk.off, len = bk.len, nw = bk.n_win;
                auto view_bkt = [&](ggml_tensor* t_heads3) {
                    return ggml_view_4d(ctx, t_heads3, Hd, len, n_heads, nw,
                                        t_heads3->nb[2], t_heads3->nb[1],
                                        len * t_heads3->nb[2],
                                        off * t_heads3->nb[2]);
                };
                ggml_tensor* qs = ggml_cont(ctx, view_bkt(qh));
                ggml_tensor* ks = ggml_cont(ctx, view_bkt(kh));
                ggml_tensor* vs = ggml_cont(ctx, view_bkt(vh));
                ggml_tensor* sc =
                        ggml_mul_mat(ctx, official_fp16 ? as_f32(ctx, ks) : ks,
                                     official_fp16 ? as_f32(ctx, qs) : qs);
                if (official_fp16) ggml_mul_mat_set_prec(sc, GGML_PREC_F32);
                sc = ggml_scale(ctx, sc, scale);
                sc = ggml_soft_max(ctx, sc);
                ggml_tensor* vt = ggml_cont(
                        ctx, ggml_transpose(ctx, official_fp16 ? as_f32(ctx, vs)
                                                               : vs));
                ggml_tensor* value = ggml_mul_mat(ctx, vt, sc);
                if (official_fp16) ggml_mul_mat_set_prec(value, GGML_PREC_F32);
                value = ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3));
                value = ggml_reshape_2d(ctx, value, C, len * nw);
                if (official_fp16) value = as_f16(ctx, value);
                acc = acc ? ggml_concat(ctx, acc, value, 1) : value;
            }
        }
        // Back to original token order + output projection + residual.
        ggml_tensor* ao =
                get_rows_preserving_type(ctx, acc, bwd_idx);  // (C, N)
        if (i == 0 && debug_stage == "b0_attn_values") return {ao};
        ggml_tensor* attn_weight = m->get(b + ".attn.to_out.weight");
        ggml_tensor* attn_bias = m->get(b + ".attn.to_out.bias");
        ao = official_fp16 ? linear_fp16(ctx, attn_weight, attn_bias, ao)
                           : gb_linear(ctx, as_f32(ctx, attn_weight),
                                       as_f32(ctx, attn_bias), ao);
        if (i == 0 && debug_stage == "b0_attn")
            return {ao};  // torch attn hook = post-to_out
        h = official_fp16 ? residual_fp16(ctx, h, ao) : ggml_add(ctx, h, ao);
        if (i == 0 && debug_stage == "b0_full")
            return {ao, h, acc};  // all block0 intermediates in ONE graph
        if (debug_stage == "b" + std::to_string(i) + "_attn") return {h};

        // norm2 -> MLP -> residual
        ggml_tensor* h2 = ggml_norm(ctx, as_f32(ctx, h), 1e-6f);
        if (official_fp16) h2 = as_f16(ctx, h2);
        if (i == 0 && debug_stage == "b0_norm2") return {h2};
        ggml_tensor* mlp0_weight = m->get(b + ".mlp.mlp.0.weight");
        ggml_tensor* mlp0_bias = m->get(b + ".mlp.mlp.0.bias");
        ggml_tensor* m1 = official_fp16
                                  ? linear_fp16(ctx, mlp0_weight, mlp0_bias, h2)
                                  : gb_linear(ctx, as_f32(ctx, mlp0_weight),
                                              as_f32(ctx, mlp0_bias), h2);
        if (i == 0 && debug_stage == "b0_mlp0") return {m1};
        ggml_tensor* g = ggml_gelu(ctx, m1);
        if (official_fp16) g = as_f16(ctx, g);
        if (i == 0 && debug_stage == "b0_gelu") return {g};
        ggml_tensor* mlp2_weight = m->get(b + ".mlp.mlp.2.weight");
        ggml_tensor* mlp2_bias = m->get(b + ".mlp.mlp.2.bias");
        ggml_tensor* m2 = official_fp16
                                  ? linear_fp16(ctx, mlp2_weight, mlp2_bias, g)
                                  : gb_linear(ctx, as_f32(ctx, mlp2_weight),
                                              as_f32(ctx, mlp2_bias), g);
        if (i == 0 && debug_stage == "b0_mlp2") return {m2};
        if (i == 0 && debug_stage == "b0_mlp") return {ao, h_in, h2, g, m2};
        h = official_fp16 ? residual_fp16(ctx, h, m2) : ggml_add(ctx, h, m2);
        if (debug_stage == "b" + std::to_string(i)) return {h};
    }

    // ---- final LayerNorm (F.layer_norm, eps 1e-5) + out layer -----------
    // decoder_gs.forward casts the F16 torso back to input dtype before this
    // final F.layer_norm and leaves out_layer in F32.
    h = ggml_norm(ctx, as_f32(ctx, h), 1e-5f);
    ggml_tensor* out =
            gb_linear(ctx, as_f32(ctx, m->get(prefix + ".out_layer.weight")),
                      as_f32(ctx, m->get(prefix + ".out_layer.bias")), h);
    GGML_ASSERT(out->ne[0] == out_ch);
    return {out};
}

}  // namespace sam3d
