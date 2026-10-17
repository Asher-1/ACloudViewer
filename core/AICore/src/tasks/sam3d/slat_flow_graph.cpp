// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SLat flow model single-step graph - see slat_flow_graph.hpp.
#include "slat_flow_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "common.hpp"
#include "graph_builder.hpp"

namespace sam3d {

static ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) t = ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// SLatFlowModel leaves input_layer, timestep embedding and output_layer in
// F32. Its input/output residual blocks and transformer torso are converted
// with convert_module_to_f16(). Preserve every observable F16 boundary: the
// Python LayerNorm32 implementation promotes only for the normalization and
// then converts back to the activation dtype.
static ggml_tensor* as_f16(ggml_context* ctx, ggml_tensor* t) {
    if (ggml_is_quantized(t->type)) t = ggml_cast(ctx, t, GGML_TYPE_F32);
    return t->type == GGML_TYPE_F16 ? t : ggml_cast(ctx, t, GGML_TYPE_F16);
}

static ggml_tensor* linear_fp16(ggml_context* ctx,
                                ggml_tensor* weight,
                                ggml_tensor* bias,
                                ggml_tensor* input,
                                bool strict_accumulation = true) {
    ggml_tensor* out =
            ggml_mul_mat(ctx, as_f16(ctx, weight), as_f16(ctx, input));
    // The explicit F32 accumulation matches the observed dense Linear
    // boundary more closely. Sparse convolution is the exception: official
    // spconv implicit-GEMM uses a different Tensor Core reduction topology
    // and opts out at its call site below.
    if (strict_accumulation) ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    if (bias) out = ggml_add(ctx, out, as_f32(ctx, as_f16(ctx, bias)));
    return as_f16(ctx, out);
}

static ggml_tensor* layer_norm32_fp16(ggml_context* ctx,
                                      ggml_tensor* input,
                                      ggml_tensor* weight,
                                      ggml_tensor* bias,
                                      float eps) {
    return as_f16(ctx, gb_layer_norm(ctx, as_f32(ctx, input),
                                     weight ? as_f32(ctx, weight) : nullptr,
                                     bias ? as_f32(ctx, bias) : nullptr, eps));
}

static ggml_tensor* silu_fp16(ggml_context* ctx, ggml_tensor* input) {
    return as_f16(ctx, ggml_silu(ctx, as_f16(ctx, input)));
}

static ggml_tensor* gelu_fp16(ggml_context* ctx, ggml_tensor* input) {
    // The official SparseFeedForwardNet uses GELU(approximate="tanh").
    return as_f16(ctx, ggml_gelu(ctx, as_f16(ctx, input)));
}

static ggml_tensor* modulate_fp16(ggml_context* ctx,
                                  ggml_tensor* input,
                                  ggml_tensor* scale,
                                  ggml_tensor* shift,
                                  ggml_tensor* one) {
    // Keep PyTorch's F16 operation order. Rewriting h * (1 + scale) as
    // h * scale + h is algebraically valid but rounds at a different point.
    one = as_f16(ctx, one);
    ggml_tensor* scale_plus_one =
            as_f16(ctx, ggml_add(ctx, as_f16(ctx, scale), one));
    ggml_tensor* value =
            as_f16(ctx, ggml_mul(ctx, as_f16(ctx, input), scale_plus_one));
    return as_f16(ctx, ggml_add(ctx, value, as_f16(ctx, shift)));
}

static ggml_tensor* residual_fp16(ggml_context* ctx,
                                  ggml_tensor* left,
                                  ggml_tensor* right) {
    return as_f16(ctx, ggml_add(ctx, as_f32(ctx, left), as_f32(ctx, right)));
}

static ggml_tensor* rms_norm_head_fp16(ggml_context* ctx,
                                       ggml_tensor* input,
                                       ggml_tensor* gamma,
                                       float head_dim) {
    return as_f16(ctx, gb_rms_norm_head(ctx, as_f32(ctx, input),
                                        as_f32(ctx, gamma), head_dim));
}

// The official SLatFlowModel keeps input_layer, the timestep embedder and
// output_layer in F32 modules for every checkpoint format; the same-GGUF
// oracle therefore dequantizes their weights before the F32 projection. A
// quantized GGML GEMM additionally quantizes the activation and is measurably
// not equivalent: with the quantized output_layer the Q8 single step reaches
// MAE 0.0060 against the oracle while the dequantized path lands at 0.00089,
// inside the oracle's own repeat-run band (0.0028). Dequantize by default;
// SAM3D_E2E_KEEP_QUANT_GEMM restores the compact-GGUF path for A/B bisects.
static ggml_tensor* matmul_weight(ggml_context* ctx,
                                  ggml_tensor* t,
                                  bool keep_quant_gemm) {
    if (ggml_is_quantized(t->type) && keep_quant_gemm) {
        return t;
    }
    return as_f32(ctx, t);
}

std::vector<ggml_tensor*> SlatFlowGraph::build() {
    ggml_context* ctx = g->ctx();
    const int64_t C = m->i32(prefix + ".model_channels", 1024);
    const int64_t n_blocks = m->i32(prefix + ".num_blocks", 24);
    const int64_t n_heads = m->i32(prefix + ".num_heads", 16);
    const int64_t Hd = C / n_heads;
    const int64_t Cin = m->i32(prefix + ".in_channels", 8);
    const int64_t io_ch = 128;  // io_block_channels[0]
    const int64_t nf = tb->nf;
    const int64_t nc = tb->nc;

    // ---------------- graph inputs ----------------
    t = g->input_f32("t", {1, 1});
    cond = g->input_f32("cond", {C, n_cond_tokens});
    x = g->input_f32("x", {Cin, nf});

    // host tables uploaded as int32 inputs; built once by the caller
    auto add_i32 = [&](const std::vector<int32_t>& v, const char* name,
                       std::initializer_list<int64_t> ne) {
        ggml_tensor* tidx = g->input_i32(name, ne);
        inputs.push_back(tidx);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(v));
        return tidx;
    };
    ggml_tensor* idx_f2c = add_i32(tb->fine_to_coarse, "f2c_idx", {nf});
    (void)idx_f2c;
    ggml_tensor* idx_conv_f =
            add_i32(tb->conv_fine, "conv_fine_idx", {27 * nf});
    ggml_tensor* idx_conv_c =
            add_i32(tb->conv_coarse, "conv_coarse_idx", {27 * nc});
    ggml_tensor* idx_spconv_f =
            add_i32(tb->conv_fine_spconv, "conv_fine_spconv_idx", {27, nf});
    ggml_tensor* idx_spconv_c =
            add_i32(tb->conv_coarse_spconv, "conv_coarse_spconv_idx", {27, nc});
    ggml_tensor* mask_spconv_f =
            add_i32(tb->conv_fine_mask, "conv_fine_spconv_mask", {nf});
    ggml_tensor* mask_spconv_c =
            add_i32(tb->conv_coarse_mask, "conv_coarse_spconv_mask", {nc});
    ggml_tensor* argsort_spconv_f =
            add_i32(tb->conv_fine_argsort, "conv_fine_spconv_argsort", {nf});
    ggml_tensor* argsort_spconv_c = add_i32(tb->conv_coarse_argsort,
                                            "conv_coarse_spconv_argsort", {nc});

    // per-channel-width zero sentinel rows for the gathers
    auto zero_row = [&](int64_t cw, const char* name) {
        ggml_tensor* z = g->input_f32(name, {cw, 1});
        inputs.push_back(z);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(cw, 0));
        return z;
    };
    ggml_tensor* z8 = zero_row(Cin, "zero8");
    ggml_tensor* z128 = zero_row(io_ch, "zero128");
    ggml_tensor* z256 = zero_row(2 * io_ch, "zero256");
    ggml_tensor* z1024 = zero_row(C, "zero1024");
    ggml_tensor* z2048 = zero_row(2 * C, "zero2048");
    // GraphContext deliberately uses no_alloc. Model-independent scalar
    // constants therefore follow the same explicit-input lifetime as tables
    // and sentinel rows rather than allocating host data while building.
    ggml_tensor* fp16_one = g->input_f32("fp16_one", {1, 1});
    inputs.push_back(fp16_one);
    auto one_data = std::make_shared<std::vector<int32_t>>(1);
    const float one_value = 1.0f;
    std::memcpy(one_data->data(), &one_value, sizeof(one_value));
    table_data.push_back(std::move(one_data));

    // SubM conv as one GEMM: gather 27 neighbour rows (missing -> sentinel);
    // the gathered layout (n, o, c) with c fastest == (27*C, N) ne, matching
    // the converter's (Cout, 27*Cin) row-format weight.
    auto zero_for = [&](int64_t cw) {
        return cw == Cin         ? z8
               : cw == io_ch     ? z128
               : cw == 2 * io_ch ? z256
               : cw == C         ? z1024
                                 : z2048;
    };
    auto conv = [&](ggml_tensor* feat, const std::string& wname,
                    const std::string& bname, ggml_tensor* idx, int64_t n) {
        const int64_t cw = feat->ne[0];
        ggml_tensor* zero = as_f16(ctx, zero_for(cw));
        ggml_tensor* weight = m->get(wname.c_str());
        ggml_tensor* bias = m->get(bname.c_str());

        if (use_cuda_spconv) {
            ggml_tensor* sp_idx = n == nf ? idx_spconv_f : idx_spconv_c;
            ggml_tensor* sp_mask = n == nf ? mask_spconv_f : mask_spconv_c;
            ggml_tensor* sp_argsort =
                    n == nf ? argsort_spconv_f : argsort_spconv_c;
            return ggml_sam3d_sparse_conv_f16(
                    ctx, as_f16(ctx, feat), as_f16(ctx, weight),
                    as_f16(ctx, bias), sp_idx, sp_mask, sp_argsort);
        }

        ggml_tensor* xc = ggml_concat(ctx, feat, zero, 1);  // (C, n+1)

        // Quantized CUDA GEMMs quantize their full activation matrix into a
        // temporary Q8 buffer. The first decoder convolution otherwise needs
        // one for 2048 x (27 * 26281) floats. Each token is independent, so
        // tile only oversized gathers and concatenate the original token order.
        constexpr int64_t kMaxGatherElements = 256LL * 1024 * 1024;
        const int64_t tile_tokens =
                std::max<int64_t>(1, kMaxGatherElements / (cw * 27));
        if (n <= tile_tokens) {
            ggml_tensor* gat = ggml_get_rows(ctx, xc, idx);  // (C, 27n)
            gat = ggml_reshape_2d(ctx, gat, cw * 27, n);
            return linear_fp16(ctx, weight, bias, gat, false);
        }

        const std::vector<int32_t>* host_idx =
                idx == idx_conv_f   ? &tb->conv_fine
                : idx == idx_conv_c ? &tb->conv_coarse
                                    : nullptr;
        GGML_ASSERT(host_idx != nullptr);
        ggml_tensor* result = nullptr;
        for (int64_t first = 0; first < n; first += tile_tokens) {
            const int64_t count = std::min(tile_tokens, n - first);
            // CUDA get_rows accepts strided source rows but not a view of an
            // index input reliably across a replayed gallocr graph. Make the
            // small I32 slice an explicit immutable input instead.
            auto slice = std::make_shared<std::vector<int32_t>>(
                    host_idx->begin() + 27 * first,
                    host_idx->begin() + 27 * (first + count));
            ggml_tensor* idx_tile =
                    g->input_i32("conv_index_tile", {27 * count});
            inputs.push_back(idx_tile);
            table_data.push_back(std::move(slice));
            ggml_tensor* gat = ggml_get_rows(ctx, xc, idx_tile);
            gat = ggml_reshape_2d(ctx, gat, cw * 27, count);
            ggml_tensor* tile = linear_fp16(ctx, weight, bias, gat, false);
            result = result ? ggml_concat(ctx, result, tile, 1) : tile;
        }
        return result;
    };

    // The reference performs a sparse F16 scatter_reduce(mean), including
    // the initial zero in each output row's denominator. A fixed gather/add
    // tree has a different rounding history, so pass the original fine-to-
    // coarse mapping to the dedicated operator instead. The divisor table
    // carries the count itself: the reference divides in F32 opmath rather
    // than multiplying a rounded reciprocal.
    auto downsample = [&](ggml_tensor* feat) {
        ggml_tensor* counts = g->input_f32("coarse_count", {1, nc});
        inputs.push_back(counts);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(
                (const int32_t*)tb->coarse_count.data(),
                (const int32_t*)(tb->coarse_count.data() + nc)));
        return ggml_sam3d_sparse_scatter_mean(ctx, as_f16(ctx, feat), idx_f2c,
                                              counts);
    };

    // 2x nearest upsample (fine_i = coarse[fine_to_coarse[i]]). idx_f2c only
    // addresses the nc real rows, so the zero-stuffing concat of the original
    // graph is dead input here - gather straight from `feat` (this also
    // avoids a Vulkan get_rows shape that reads back garbage on several
    // drivers). The gather runs in F32 to match the official spconv
    // accumulator contract.
    auto upsample = [&](ggml_tensor* feat) {
        ggml_tensor* gat =
                ggml_get_rows(ctx, as_f32(ctx, feat), idx_f2c);  // (C, nf)
        return as_f16(ctx, ggml_reshape_2d(ctx, gat, feat->ne[0], nf));
    };

    // ---------------- timestep embedding ----------------
    const int64_t half = 128;
    auto freq_buf = std::make_shared<std::vector<int32_t>>(half);
    float* ff = (float*)freq_buf->data();
    for (int i = 0; i < (int)half; i++)
        ff[i] = expf(-logf(10000.0f) * (float)i / (float)half);
    ggml_tensor* freqs = g->input_f32("t_freqs", {half, 1});
    inputs.push_back(freqs);
    table_data.push_back(freq_buf);
    ggml_tensor* tf = ggml_mul(ctx, ggml_repeat(ctx, t, freqs), freqs);
    ggml_tensor* emb =
            ggml_concat(ctx, ggml_cos(ctx, tf), ggml_sin(ctx, tf), 0);
    // The official SLatFlowModel keeps the timestep embedder as an F32 module
    // for every checkpoint format; the same-GGUF oracle therefore runs this
    // projection with dequantized F32 weights. A quantized GGML GEMV would
    // quantize the activation into Q8 and is measurably not equivalent: that
    // mismatch is the first Q8/Q4 divergence boundary (t=0 makes the identity
    // frequencies exact, so only the weight representation differs).
    ggml_tensor* t_emb = gb_linear(
            ctx, as_f32(ctx, m->get(prefix + ".t_embedder.mlp.0.weight")),
            as_f32(ctx, m->get(prefix + ".t_embedder.mlp.0.bias")), emb);
    t_emb = ggml_silu(ctx, t_emb);
    t_emb = gb_linear(ctx,
                      as_f32(ctx, m->get(prefix + ".t_embedder.mlp.2.weight")),
                      as_f32(ctx, m->get(prefix + ".t_embedder.mlp.2.bias")),
                      t_emb);  // (C, 1)
    const bool dump_block_outputs = this->dump_block_outputs;
    // SLatFlowModel casts the F32 embedding and external condition once at
    // the torso boundary, before either reaches a converted module.
    t_emb = as_f16(ctx, t_emb);
    if (debug_stage == "t_emb") return {t_emb};
    // Keep `cond` itself as the CLI-uploaded F32 graph input. Replacing this
    // member pointer with the cast node works only until cross-attention makes
    // that node live, at which point the uploader would misclassify it.
    ggml_tensor* cond_f16 = as_f16(ctx, cond);

    // sparse res block: [down/up] -> norm1(affine) -> silu -> conv1 ->
    // norm2(no affine) -> *(1+scale)+shift -> silu -> conv2 -> + skip
    auto res_block = [&](ggml_tensor* xin, const std::string& b, int64_t cout,
                         bool down, bool up, bool* dbg_early) {
        const bool trace_input_block1 =
                dump_block_outputs && b == prefix + ".input_blocks.1";
        const auto debug_endpoint = [&](const char* canonical_suffix,
                                        const char* short_suffix = nullptr) {
            if (debug_stage == b + "_" + canonical_suffix) return true;
            const std::string input_prefix = prefix + ".input_blocks.";
            if (b.rfind(input_prefix, 0) != 0) return false;
            const std::string index = b.substr(input_prefix.size());
            return debug_stage ==
                   "ib" + index + "_" +
                           (short_suffix ? short_suffix : canonical_suffix);
        };
        if (down) {
            xin = downsample(xin);
            if (trace_input_block1) debug_block_outputs.push_back(xin);
            if (debug_endpoint("down", "updown")) {
                if (dbg_early) *dbg_early = true;
                return xin;
            }
        }
        if (up) {
            xin = upsample(xin);
            if (debug_stage == b + "_up") {
                if (dbg_early) *dbg_early = true;
                return xin;
            }
        }
        const bool coarse = down || up;
        const int64_t n = coarse ? nc : nf;
        ggml_tensor* idx = coarse ? idx_conv_c : idx_conv_f;
        ggml_tensor* e = linear_fp16(ctx, m->get(b + ".emb_layers.1.weight"),
                                     m->get(b + ".emb_layers.1.bias"),
                                     silu_fp16(ctx, t_emb));  // (2*cout, 1)
        ggml_tensor* scale = ggml_view_2d(ctx, e, cout, 1, e->nb[1], 0);
        ggml_tensor* shift = ggml_view_2d(ctx, e, cout, 1, e->nb[1],
                                          cout * ggml_element_size(e));
        // Keep the pre-affine boundary observable for parity diagnosis. This
        // endpoint is not part of the normal graph and only exists when an
        // explicit SAM3D_DEBUG_STAGE request stops evaluation here.
        if (debug_endpoint("norm1_raw") || debug_endpoint("norm1_raw_f32")) {
            if (dbg_early) *dbg_early = true;
            ggml_tensor* raw_norm = ggml_norm(ctx, as_f32(ctx, xin), 1e-6f);
            return debug_endpoint("norm1_raw_f32") ? raw_norm
                                                   : as_f16(ctx, raw_norm);
        }
        ggml_tensor* hh =
                layer_norm32_fp16(ctx, xin, m->get(b + ".norm1.weight"),
                                  m->get(b + ".norm1.bias"), 1e-6f);
        if (debug_endpoint("norm1")) {
            if (dbg_early) *dbg_early = true;
            return hh;
        }
        hh = silu_fp16(ctx, hh);
        if (debug_endpoint("silu1")) {
            if (dbg_early) *dbg_early = true;
            return hh;
        }
        if (trace_input_block1) debug_block_outputs.push_back(hh);
        hh = conv(hh, b + ".conv1.conv.weight", b + ".conv1.conv.bias", idx, n);
        if (trace_input_block1) debug_block_outputs.push_back(hh);
        if (debug_endpoint("conv1")) {
            if (dbg_early) *dbg_early = true;
            return hh;
        }
        hh = layer_norm32_fp16(ctx, hh, nullptr, nullptr, 1e-6f);
        if (debug_endpoint("norm2")) {
            if (dbg_early) *dbg_early = true;
            return hh;
        }
        hh = modulate_fp16(ctx, hh, scale, shift, fp16_one);
        hh = silu_fp16(ctx, hh);
        if (debug_endpoint("silu2")) {
            if (dbg_early) *dbg_early = true;
            return hh;
        }
        if (trace_input_block1) debug_block_outputs.push_back(hh);
        hh = conv(hh, b + ".conv2.conv.weight", b + ".conv2.conv.bias", idx, n);
        if (trace_input_block1) debug_block_outputs.push_back(hh);
        if (debug_endpoint("conv2")) {
            if (dbg_early) *dbg_early = true;
            return hh;
        }
        ggml_tensor* skip = xin;
        if (xin->ne[0] != cout)
            skip = linear_fp16(ctx, m->get(b + ".skip_connection.weight"),
                               m->get(b + ".skip_connection.bias"), xin);
        if (trace_input_block1) debug_block_outputs.push_back(skip);
        ggml_tensor* out = residual_fp16(ctx, hh, skip);
        if (debug_endpoint("out")) {
            if (dbg_early) *dbg_early = true;
            return out;
        }
        if (trace_input_block1) debug_block_outputs.push_back(out);
        return out;
    };

    // ---------------- input stage ----------------
    ggml_tensor* h = gb_linear(
            ctx,
            matmul_weight(ctx, m->get(prefix + ".input_layer.weight"),
                          keep_quant_gemm),
            as_f32(ctx, m->get(prefix + ".input_layer.bias")), x);  // (128, nf)
    if (debug_stage == "input_layer") return {h};
    h = as_f16(ctx, h);
    if (dump_block_outputs) debug_block_outputs.push_back(h);

    bool early = false;
    ggml_tensor* skip0 = res_block(h, prefix + ".input_blocks.0", io_ch, false,
                                   false, &early);  // (128, nf)
    if (early) return {skip0};
    if (dump_block_outputs) debug_block_outputs.push_back(skip0);
    ggml_tensor* skip1 = res_block(skip0, prefix + ".input_blocks.1", C, true,
                                   false, &early);  // (1024, nc)
    if (early) return {skip1};
    if (dump_block_outputs) debug_block_outputs.push_back(skip1);

    // ---------------- APE + transformer blocks ----------------
    ggml_tensor* ape = g->input_f32("ape", {C, nc});
    inputs.push_back(ape);
    table_data.push_back(std::make_shared<std::vector<int32_t>>(
            (const int32_t*)tb->ape.data(),
            (const int32_t*)(tb->ape.data() + tb->ape.size())));
    h = as_f16(ctx, ggml_add(ctx, skip1, as_f16(ctx, ape)));
    if (debug_stage == "ape") return {h};
    if (debug_stage == "block_in") return {h};

    for (int i = 0; i < (int)n_blocks; i++) {
        const std::string b = prefix + ".blocks." + std::to_string(i);
        ggml_tensor* six = linear_fp16(
                ctx, m->get(b + ".adaLN_modulation.1.weight"),
                m->get(b + ".adaLN_modulation.1.bias"), silu_fp16(ctx, t_emb));
        if (i == 0 && debug_stage == "b0_adaln") return {six};
        ggml_tensor* shift_msa = ggml_view_2d(ctx, six, C, 1, six->nb[1], 0);
        ggml_tensor* scale_msa = ggml_view_2d(ctx, six, C, 1, six->nb[1],
                                              C * ggml_element_size(six));
        ggml_tensor* gate_msa =
                ggml_view_2d(ctx, six, C, 1, six->nb[1],
                             2 * (size_t)C * ggml_element_size(six));
        ggml_tensor* shift_mlp =
                ggml_view_2d(ctx, six, C, 1, six->nb[1],
                             3 * (size_t)C * ggml_element_size(six));
        ggml_tensor* scale_mlp =
                ggml_view_2d(ctx, six, C, 1, six->nb[1],
                             4 * (size_t)C * ggml_element_size(six));
        ggml_tensor* gate_mlp =
                ggml_view_2d(ctx, six, C, 1, six->nb[1],
                             5 * (size_t)C * ggml_element_size(six));

        ggml_tensor* hs = layer_norm32_fp16(ctx, h, nullptr, nullptr, 1e-6f);
        hs = modulate_fp16(ctx, hs, scale_msa, shift_msa, fp16_one);
        if (debug_stage == "b" + std::to_string(i) + "_attn_in") return {hs};
        // The reference torso applies one F16 QKV projection, then splits its
        // rounded result. This is also the compatibility path for Q8/Q4: the
        // same-GGUF Torch oracle dequantizes before loading its F16 module.
        ggml_tensor* wqkv = m->get(b + ".self_attn.to_qkv.weight");
        ggml_tensor* bqkv = m->get(b + ".self_attn.to_qkv.bias");
        ggml_tensor *q, *k, *v;
        ggml_tensor* qkv = linear_fp16(ctx, wqkv, bqkv, hs);
        if (debug_stage == "b" + std::to_string(i) + "_qpre") {
            return {ggml_cont(ctx,
                              ggml_view_2d(ctx, qkv, C, nc, qkv->nb[1], 0))};
        }
        if (i == 0 && debug_stage == "b0_qkv") return {qkv};
        gb_split_qkv(ctx, qkv, n_heads, &q, &k, &v);
        q = rms_norm_head_fp16(
                ctx, q, m->get(b + ".self_attn.q_rms_norm.gamma"), (float)Hd);
        if (i == 0 && debug_stage == "b0_qrms") {
            return {ggml_cont(ctx, q)};
        }
        k = rms_norm_head_fp16(
                ctx, k, m->get(b + ".self_attn.k_rms_norm.gamma"), (float)Hd);
        const float attn_scale = 1.0f / sqrtf((float)Hd);
        ggml_tensor* out =
                as_f16(ctx, gb_attention(ctx, q, k, v, attn_scale, true));
        out = ggml_reshape_2d(ctx, out, C, nc);
        out = linear_fp16(ctx, m->get(b + ".self_attn.to_out.weight"),
                          m->get(b + ".self_attn.to_out.bias"), out);
        if (i == 0 && debug_stage == "b0_attn_out") return {out};
        h = residual_fp16(ctx, h, as_f16(ctx, ggml_mul(ctx, out, gate_msa)));
        if (i == 0 && debug_stage == "b0_res") return {h};

        // cross attention (no rms norm on cross)
        ggml_tensor* hc = layer_norm32_fp16(ctx, h, m->get(b + ".norm2.weight"),
                                            m->get(b + ".norm2.bias"), 1e-6f);
        ggml_tensor* cq =
                linear_fp16(ctx, m->get(b + ".cross_attn.to_q.weight"),
                            m->get(b + ".cross_attn.to_q.bias"), hc);
        ggml_tensor* wkv = m->get(b + ".cross_attn.to_kv.weight");
        ggml_tensor* bkv = m->get(b + ".cross_attn.to_kv.bias");
        ggml_tensor *ck, *cv;
        gb_split_kv(ctx, linear_fp16(ctx, wkv, bkv, cond_f16), &ck, &cv);
        auto heads = [&](ggml_tensor* t2, int64_t n) {
            ggml_tensor* r = ggml_reshape_3d(ctx, t2, Hd, n_heads, n);
            return ggml_permute(ctx, r, 0, 2, 1, 3);
        };
        const int64_t n_q = cq->ne[1];
        const int64_t n_kv = ck->ne[1];
        cq = heads(cq, n_q);
        ck = heads(ck, n_kv);
        cv = heads(cv, n_kv);
        ggml_tensor* co =
                as_f16(ctx, gb_attention(ctx, cq, ck, cv, attn_scale, true));
        co = ggml_reshape_2d(ctx, co, C, n_q);
        co = linear_fp16(ctx, m->get(b + ".cross_attn.to_out.weight"),
                         m->get(b + ".cross_attn.to_out.bias"), co);
        if (i == 0 && debug_stage == "b0_cross_out") return {co};
        h = residual_fp16(ctx, h, co);
        if (i == 0 && debug_stage == "b0_cross_res") return {h};

        // MLP branch
        ggml_tensor* hm = layer_norm32_fp16(ctx, h, nullptr, nullptr, 1e-6f);
        hm = modulate_fp16(ctx, hm, scale_mlp, shift_mlp, fp16_one);
        hm = linear_fp16(ctx, m->get(b + ".mlp.mlp.0.weight"),
                         m->get(b + ".mlp.mlp.0.bias"), hm);
        hm = gelu_fp16(ctx, hm);
        hm = linear_fp16(ctx, m->get(b + ".mlp.mlp.2.weight"),
                         m->get(b + ".mlp.mlp.2.bias"), hm);
        if (i == 0 && debug_stage == "b0_mlp_out") return {hm};
        h = residual_fp16(ctx, h, as_f16(ctx, ggml_mul(ctx, hm, gate_mlp)));
        if (dump_block_outputs) debug_block_outputs.push_back(h);
        if (debug_stage == "block" + std::to_string(i)) return {h};
    }

    // ---------------- output stage ----------------
    // out_blocks[0]: in = concat([h(1024), skip1(1024)]) = 2048 coarse ->
    // upsample -> convs 2048->128, 128->128; skip = Linear(2048->128).
    // out_blocks[1]: in = concat([h(128), skip0(128)]) = 256 fine -> convs
    // 256->128, 128->128; skip = Linear(256->128).
    {
        const std::string b = prefix + ".out_blocks.0";
        ggml_tensor* xin =
                as_f16(ctx, ggml_concat(ctx, h, skip1, 0));  // (2048, nc)
        xin = upsample(xin);                                 // (2048, nf)
        if (debug_stage == "ob0_up_out") return {xin};
        if (dump_block_outputs) debug_block_outputs.push_back(xin);
        ggml_tensor* e = linear_fp16(ctx, m->get(b + ".emb_layers.1.weight"),
                                     m->get(b + ".emb_layers.1.bias"),
                                     silu_fp16(ctx, t_emb));
        ggml_tensor* scale = ggml_view_2d(ctx, e, io_ch, 1, e->nb[1], 0);
        ggml_tensor* shift = ggml_view_2d(ctx, e, io_ch, 1, e->nb[1],
                                          io_ch * ggml_element_size(e));
        ggml_tensor* hh =
                layer_norm32_fp16(ctx, xin, m->get(b + ".norm1.weight"),
                                  m->get(b + ".norm1.bias"), 1e-6f);
        if (debug_stage == "ob0_norm1") return {hh};
        hh = silu_fp16(ctx, hh);
        if (dump_block_outputs) debug_block_outputs.push_back(hh);
        hh = conv(hh, b + ".conv1.conv.weight", b + ".conv1.conv.bias",
                  idx_conv_f, nf);
        if (dump_block_outputs) debug_block_outputs.push_back(hh);
        if (debug_stage == "ob0_conv1") return {hh};
        hh = layer_norm32_fp16(ctx, hh, nullptr, nullptr, 1e-6f);
        if (debug_stage == "ob0_norm2") return {hh};
        hh = modulate_fp16(ctx, hh, scale, shift, fp16_one);
        if (debug_stage == "ob0_mod") return {hh};
        hh = silu_fp16(ctx, hh);
        if (debug_stage == "ob0_silu2") return {hh};
        hh = conv(hh, b + ".conv2.conv.weight", b + ".conv2.conv.bias",
                  idx_conv_f, nf);
        if (dump_block_outputs) debug_block_outputs.push_back(hh);
        if (debug_stage == "ob0_conv2") return {hh};
        ggml_tensor* skip =
                linear_fp16(ctx, m->get(b + ".skip_connection.weight"),
                            m->get(b + ".skip_connection.bias"), xin);
        if (dump_block_outputs) debug_block_outputs.push_back(skip);
        if (debug_stage == "ob0_skip") return {skip};
        if (debug_stage == "ob0_hh") return {hh};
        h = residual_fp16(ctx, hh, skip);  // (128, nf)
        if (dump_block_outputs)
            debug_block_outputs.push_back(h);  // out block 0
        if (debug_stage == "ob0_residual") return {h};
    }
    {
        const std::string b = prefix + ".out_blocks.1";
        ggml_tensor* xin =
                as_f16(ctx, ggml_concat(ctx, h, skip0, 0));  // (256, nf)
        if (debug_stage == "ob1_xin") return {xin};
        ggml_tensor* e = linear_fp16(ctx, m->get(b + ".emb_layers.1.weight"),
                                     m->get(b + ".emb_layers.1.bias"),
                                     silu_fp16(ctx, t_emb));
        ggml_tensor* scale = ggml_view_2d(ctx, e, io_ch, 1, e->nb[1], 0);
        ggml_tensor* shift = ggml_view_2d(ctx, e, io_ch, 1, e->nb[1],
                                          io_ch * ggml_element_size(e));
        ggml_tensor* hh =
                layer_norm32_fp16(ctx, xin, m->get(b + ".norm1.weight"),
                                  m->get(b + ".norm1.bias"), 1e-6f);
        if (debug_stage == "ob1_norm1") return {hh};
        hh = silu_fp16(ctx, hh);
        hh = conv(hh, b + ".conv1.conv.weight", b + ".conv1.conv.bias",
                  idx_conv_f, nf);
        if (debug_stage == "ob1_conv1") return {hh};
        hh = layer_norm32_fp16(ctx, hh, nullptr, nullptr, 1e-6f);
        hh = modulate_fp16(ctx, hh, scale, shift, fp16_one);
        hh = silu_fp16(ctx, hh);
        hh = conv(hh, b + ".conv2.conv.weight", b + ".conv2.conv.bias",
                  idx_conv_f, nf);
        if (debug_stage == "ob1_conv2") return {hh};
        ggml_tensor* skip =
                linear_fp16(ctx, m->get(b + ".skip_connection.weight"),
                            m->get(b + ".skip_connection.bias"), xin);
        h = residual_fp16(ctx, hh, skip);
        if (dump_block_outputs)
            debug_block_outputs.push_back(h);  // out block 1
    }

    if (debug_stage == "pre_final") return {h};
    // The final functional layer norm observes the F16 torso result; the
    // unconverted output linear then receives its explicit F32 promotion.
    h = layer_norm32_fp16(ctx, h, nullptr, nullptr, 1e-6f);
    if (dump_block_outputs)
        debug_block_outputs.push_back(h);  // final LayerNorm
    h = gb_linear(ctx,
                  matmul_weight(ctx, m->get(prefix + ".out_layer.weight"),
                                keep_quant_gemm),
                  as_f32(ctx, m->get(prefix + ".out_layer.bias")),
                  as_f32(ctx, h));
    return {h};
}

}  // namespace sam3d
