// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mesh_decoder_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "common.hpp"

namespace sam3d {
namespace {

ggml_tensor* as_f32(ggml_context* ctx, ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32
                   ? tensor
                   : ggml_cast(ctx, tensor, GGML_TYPE_F32);
}

ggml_tensor* as_f16(ggml_context* ctx, ggml_tensor* tensor) {
    // CUDA supports quantized decoder weights through the F32 dequantization
    // path, not a direct Q4/Q8 -> F16 copy.
    if (ggml_is_quantized(tensor->type))
        tensor = ggml_cast(ctx, tensor, GGML_TYPE_F32);
    return tensor->type == GGML_TYPE_F16
                   ? tensor
                   : ggml_cast(ctx, tensor, GGML_TYPE_F16);
}

// PyTorch's CUDA F16 Linear uses Tensor Core FP32 accumulation and then
// materializes an F16 tensor.  Request the same accumulation precision before
// the explicit F16 boundary, so this graph does not depend on ggml's backend
// default (which is F16 accumulation for F16 weights on CUDA).
ggml_tensor* linear_fp16(ggml_context* ctx,
                         ggml_tensor* weight,
                         ggml_tensor* bias,
                         ggml_tensor* input) {
    ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    if (bias != nullptr) output = ggml_add(ctx, output, as_f32(ctx, bias));
    return as_f16(ctx, output);
}

ggml_tensor* residual_fp16(ggml_context* ctx,
                           ggml_tensor* left,
                           ggml_tensor* right) {
    return as_f16(ctx, ggml_add(ctx, as_f32(ctx, left), as_f32(ctx, right)));
}

// The ggml get_rows API currently materializes non-I32 gathers as F32.  Sparse
// PyTorch gathers preserve the source dtype, which is a real precision boundary
// in SLatMeshDecoder's mixed-F16 path.  Restore that contract at the graph
// edge.
ggml_tensor* get_rows_preserving_type(ggml_context* ctx,
                                      ggml_tensor* source,
                                      ggml_tensor* indices) {
    ggml_tensor* rows = ggml_get_rows(ctx, source, indices);
    return source->type == GGML_TYPE_F16 ? as_f16(ctx, rows) : rows;
}

// SparseGroupNorm32 receives [tokens, channels] in PyTorch, permutes it to
// [1, channels, tokens], performs GroupNorm, then permutes it back.  ggml's
// group-norm kernel treats ne[2] as channels, so the equivalent logical layout
// is [tokens, 1, channels, batch], not [tokens, channels, 1].
ggml_tensor* sparse_group_norm(ggml_context* ctx,
                               ggml_tensor* features,
                               ggml_tensor* weight,
                               ggml_tensor* bias,
                               int groups) {
    const int64_t channels = features->ne[0];
    const int64_t tokens = features->ne[1];
    GGML_ASSERT(channels % groups == 0);
    // The official mixed-F16 decoder keeps GroupNorm32 in F32.  ggml CUDA
    // enforces the same type contract, so promote before the layout change
    // and restore the sparse activation precision at the module boundary.
    ggml_tensor* token_major =
            ggml_cont(ctx, ggml_transpose(ctx, as_f32(ctx, features)));
    ggml_tensor* normalized = ggml_group_norm(
            ctx, ggml_reshape_4d(ctx, token_major, tokens, 1, channels, 1),
            groups, 1e-5f);
    if (weight != nullptr) {
        normalized = ggml_mul(
                ctx, normalized,
                ggml_reshape_4d(ctx, as_f32(ctx, weight), 1, 1, channels, 1));
    }
    if (bias != nullptr) {
        normalized = ggml_add(
                ctx, normalized,
                ggml_reshape_4d(ctx, as_f32(ctx, bias), 1, 1, channels, 1));
    }
    ggml_tensor* output = ggml_cont(
            ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, normalized, tokens,
                                                     channels)));
    return features->type == GGML_TYPE_F16 ? as_f16(ctx, output) : output;
}

}  // namespace

std::vector<ggml_tensor*> MeshDecoderGraph::build() {
    GGML_ASSERT(g != nullptr && m != nullptr && tb != nullptr && x != nullptr);
    ggml_context* ctx = g->ctx();
    const int64_t channels = m->i32(prefix + ".model_channels", 768);
    const int64_t n_blocks = m->i32(prefix + ".num_blocks", 12);
    const int64_t n_heads = m->i32(prefix + ".num_heads", 12);
    const int64_t head_dim = channels / n_heads;
    const int64_t input_channels = m->i32(prefix + ".latent_channels", 8);
    const int64_t output_channels = m->i32(prefix + ".out_channels", 101);
    const int64_t n64 = tb->levels[0].n;
    const int64_t n256 = tb->levels[2].n;
    const bool official_fp16 =
            m->str(prefix + ".precision_contract") == "official-use-fp16-v1";
    GGML_ASSERT(x->ne[0] == input_channels && x->ne[1] == n64);
    GGML_ASSERT(channels % n_heads == 0);

    auto add_i32 = [&](const std::vector<int32_t>& data, const char* name,
                       std::initializer_list<int64_t> shape) {
        ggml_tensor* input = g->input_i32(name, shape);
        inputs.push_back(input);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(data));
        return input;
    };
    auto add_zero_row = [&](int64_t width, const char* name) {
        ggml_tensor* input = g->input_f32(name, {width, 1});
        inputs.push_back(input);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(
                static_cast<size_t>(width), 0));
        return input;
    };

    // ---- SparseTransformerBase torso ---------------------------------
    ggml_tensor* h =
            gb_linear(ctx, as_f32(ctx, m->get(prefix + ".input_layer.weight")),
                      as_f32(ctx, m->get(prefix + ".input_layer.bias")), x);
    if (debug_stage == "input_layer") {
        return {h};
    }
    {
        ggml_tensor* ape = g->input_f32("mesh_ape", {channels, n64});
        inputs.push_back(ape);
        table_data.push_back(std::make_shared<std::vector<int32_t>>(
                reinterpret_cast<const int32_t*>(tb->swin.ape.data()),
                reinterpret_cast<const int32_t*>(tb->swin.ape.data() +
                                                 tb->swin.ape.size())));
        h = ggml_add(ctx, h, ape);
    }
    if (official_fp16) h = as_f16(ctx, h);
    if (debug_stage == "ape" || debug_stage == "block0_ape") {
        return {h};
    }

    // PyTorch executes one variable-length attention call per shift, rather
    // than one call for each equal-length window bucket.  Its CUDA dispatch is
    // sensitive to the physical aligned dense-bias layout used by that single
    // call.  Keep just the window-length description in graph inputs; the
    // CUDA op materializes its ephemeral device-only bias buffer.
    std::array<ggml_tensor*, 2> attention_sequence_lengths = {nullptr, nullptr};
    if (official_fp16 && use_pytorch_cuda_attention) {
        for (size_t shift = 0; shift < attention_sequence_lengths.size();
             ++shift) {
            const GsWindowTables& windows = tb->swin.shifts[shift];
            std::vector<int32_t> lengths;
            lengths.reserve(windows.buckets.size());
            int64_t covered_tokens = 0;
            for (const GsWindowTables::Bucket& bucket : windows.buckets) {
                GGML_ASSERT(bucket.len > 0 && bucket.n_win > 0);
                lengths.insert(lengths.end(), static_cast<size_t>(bucket.n_win),
                               static_cast<int32_t>(bucket.len));
                covered_tokens += bucket.len * bucket.n_win;
            }
            GGML_ASSERT(covered_tokens == n64 && !lengths.empty());
            attention_sequence_lengths[shift] =
                    add_i32(lengths, "mesh_attention_sequence_lengths",
                            {static_cast<int64_t>(lengths.size())});
        }
    }

    for (int block = 0; block < n_blocks; ++block) {
        const std::string name = prefix + ".blocks." + std::to_string(block);
        const GsWindowTables& windows = tb->swin.shifts[block % 2];
        ggml_tensor* bucket_indices =
                add_i32(windows.bkt_idx, "mesh_bucket_indices", {n64});
        ggml_tensor* original_indices =
                add_i32(windows.bwd_idx, "mesh_original_indices", {n64});

        ggml_tensor* norm = ggml_norm(ctx, as_f32(ctx, h), 1e-6f);
        if (official_fp16) norm = as_f16(ctx, norm);
        if (debug_stage == "block" + std::to_string(block) + "_norm1") {
            return {norm};
        }
        ggml_tensor* bucketed =
                get_rows_preserving_type(ctx, norm, bucket_indices);
        ggml_tensor* qkv_weight = m->get(name + ".attn.to_qkv.weight");
        ggml_tensor* qkv_bias = m->get(name + ".attn.to_qkv.bias");
        // PyTorch executes to_qkv as one (3C, C) projection and only then
        // splits its result.  Keeping it fused is numerically significant for
        // FP16 Tensor Core GEMMs and avoids two otherwise independent launches.
        ggml_tensor* qkv =
                official_fp16 ? linear_fp16(ctx, qkv_weight, qkv_bias, bucketed)
                              : gb_linear(ctx, as_f32(ctx, qkv_weight),
                                          as_f32(ctx, qkv_bias), bucketed);
        const size_t qkv_element_size = ggml_type_size(qkv->type);
        auto qkv_view = [&](int index) {
            return ggml_cont(
                    ctx, ggml_view_2d(ctx, qkv, channels, n64, qkv->nb[1],
                                      static_cast<size_t>(index * channels) *
                                              qkv_element_size));
        };
        ggml_tensor* q = qkv_view(0);
        ggml_tensor* k = qkv_view(1);
        ggml_tensor* v = qkv_view(2);
        if (debug_stage == "block" + std::to_string(block) + "_qkv") {
            q = get_rows_preserving_type(ctx, q, original_indices);
            k = get_rows_preserving_type(ctx, k, original_indices);
            v = get_rows_preserving_type(ctx, v, original_indices);
            return {ggml_concat(ctx, ggml_concat(ctx, q, k, 0), v, 0)};
        }

        auto heads = [&](ggml_tensor* tensor) {
            return ggml_reshape_3d(ctx, tensor, head_dim, n_heads, n64);
        };
        ggml_tensor* qh = heads(q);
        ggml_tensor* kh = heads(k);
        ggml_tensor* vh = heads(v);
        ggml_tensor* attention = nullptr;
        const float attention_scale =
                1.0f / std::sqrt(static_cast<float>(head_dim));
        if (official_fp16 && use_pytorch_cuda_attention) {
            GGML_ASSERT(qh->type == GGML_TYPE_F16 &&
                        kh->type == GGML_TYPE_F16 && vh->type == GGML_TYPE_F16);
            attention = ggml_sam3d_f16_masked_attn(
                    ctx, qh, kh, vh, attention_sequence_lengths[block % 2],
                    attention_scale);
            attention = ggml_reshape_2d(ctx, attention, channels, n64);
        } else {
            for (const GsWindowTables::Bucket& bucket : windows.buckets) {
                const int64_t offset = bucket.off;
                const int64_t length = bucket.len;
                const int64_t window_count = bucket.n_win;
                auto window_view = [&](ggml_tensor* tensor) {
                    return ggml_view_4d(ctx, tensor, head_dim, length, n_heads,
                                        window_count, tensor->nb[2],
                                        tensor->nb[1], length * tensor->nb[2],
                                        offset * tensor->nb[2]);
                };
                ggml_tensor* query = ggml_cont(ctx, window_view(qh));
                ggml_tensor* key = ggml_cont(ctx, window_view(kh));
                ggml_tensor* value = ggml_cont(ctx, window_view(vh));
                // The portable graph deliberately retains explicit F32
                // reductions because it is used by CPU and Vulkan as well.
                ggml_tensor* attention_query =
                        official_fp16 ? as_f32(ctx, query) : query;
                ggml_tensor* attention_key =
                        official_fp16 ? as_f32(ctx, key) : key;
                ggml_tensor* attention_value =
                        official_fp16 ? as_f32(ctx, value) : value;
                ggml_tensor* scores =
                        ggml_mul_mat(ctx, attention_key, attention_query);
                if (official_fp16) ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                scores = ggml_scale(ctx, scores, attention_scale);
                scores = ggml_soft_max(ctx, scores);
                ggml_tensor* values_t =
                        ggml_cont(ctx, ggml_transpose(ctx, attention_value));
                ggml_tensor* result = ggml_mul_mat(ctx, values_t, scores);
                if (official_fp16) ggml_mul_mat_set_prec(result, GGML_PREC_F32);
                result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
                result = ggml_reshape_2d(ctx, result, channels,
                                         length * window_count);
                if (official_fp16) result = as_f16(ctx, result);
                attention = attention == nullptr
                                    ? result
                                    : ggml_concat(ctx, attention, result, 1);
            }
        }
        attention = get_rows_preserving_type(ctx, attention, original_indices);
        if (debug_stage ==
            "block" + std::to_string(block) + "_attention_values") {
            return {attention};
        }
        ggml_tensor* attn_weight = m->get(name + ".attn.to_out.weight");
        ggml_tensor* attn_bias = m->get(name + ".attn.to_out.bias");
        attention =
                official_fp16
                        ? linear_fp16(ctx, attn_weight, attn_bias, attention)
                        : gb_linear(ctx, as_f32(ctx, attn_weight),
                                    as_f32(ctx, attn_bias), attention);
        if (debug_stage == "block" + std::to_string(block) + "_attention_out") {
            return {attention};
        }
        h = official_fp16 ? residual_fp16(ctx, h, attention)
                          : ggml_add(ctx, h, attention);
        if (debug_stage == "block" + std::to_string(block) + "_attention") {
            return {h};
        }

        ggml_tensor* mlp = ggml_norm(ctx, as_f32(ctx, h), 1e-6f);
        if (official_fp16) mlp = as_f16(ctx, mlp);
        if (debug_stage == "block" + std::to_string(block) + "_norm2") {
            return {mlp};
        }
        ggml_tensor* mlp0_weight = m->get(name + ".mlp.mlp.0.weight");
        ggml_tensor* mlp0_bias = m->get(name + ".mlp.mlp.0.bias");
        mlp = official_fp16 ? linear_fp16(ctx, mlp0_weight, mlp0_bias, mlp)
                            : gb_linear(ctx, as_f32(ctx, mlp0_weight),
                                        as_f32(ctx, mlp0_bias), mlp);
        if (debug_stage == "block" + std::to_string(block) + "_mlp0") {
            return {mlp};
        }
        mlp = ggml_gelu(ctx, mlp);
        if (official_fp16) mlp = as_f16(ctx, mlp);
        if (debug_stage == "block" + std::to_string(block) + "_gelu") {
            return {mlp};
        }
        ggml_tensor* mlp2_weight = m->get(name + ".mlp.mlp.2.weight");
        ggml_tensor* mlp2_bias = m->get(name + ".mlp.mlp.2.bias");
        mlp = official_fp16 ? linear_fp16(ctx, mlp2_weight, mlp2_bias, mlp)
                            : gb_linear(ctx, as_f32(ctx, mlp2_weight),
                                        as_f32(ctx, mlp2_bias), mlp);
        if (debug_stage == "block" + std::to_string(block) + "_mlp2") {
            return {mlp};
        }
        h = official_fp16 ? residual_fp16(ctx, h, mlp) : ggml_add(ctx, h, mlp);
        if (debug_stage == "block" + std::to_string(block)) {
            return {h};
        }
    }

    // Mesh decoder Conv3d weights use the GGUF layout written by
    // convert_sam3d_to_gguf.py: logical dimensions (Cin * Cout, K^3), where
    // each K^3 column is a contiguous (Cout, Cin) matrix.  This is distinct
    // from the SLat-flow GGUF layout, which flattens 27 input offsets into a
    // single GEMM row.  Preserve the original spconv accumulation order by
    // applying each offset matrix separately, then summing the 27 results.
    auto sparse_conv = [&](ggml_tensor* features, const MeshConvLevel& level,
                           const std::string& name) {
        const int64_t input_width = features->ne[0];
        ggml_tensor* zero = add_zero_row(input_width, "mesh_conv_zero");
        if (features->type == GGML_TYPE_F16) zero = as_f16(ctx, zero);
        ggml_tensor* extended = ggml_concat(ctx, features, zero, 1);
        ggml_tensor* weights = m->get(name + ".weight");
        ggml_tensor* bias = m->get(name + ".bias");
        GGML_ASSERT(weights != nullptr && bias != nullptr);
        constexpr int64_t kernel_elements = 27;
        GGML_ASSERT(weights->ne[1] == kernel_elements);
        GGML_ASSERT(weights->ne[0] % input_width == 0);
        const int64_t output_width = weights->ne[0] / input_width;
        // Each offset is a logical [Cin, Cout] matrix.  A view into a
        // block-quantized [Cin*Cout, 27] source is not a valid independent
        // Q8/Q4 MMQ row layout: the block metadata spans the logical matrix
        // boundary.  Materialize the tensor once in F32 before creating the
        // 27 offset views, preserving the official offset accumulation order
        // without issuing an invalid quantized GEMM for a strided submatrix.
        ggml_tensor* view_weights = ggml_is_quantized(weights->type)
                                            ? as_f32(ctx, weights)
                                            : weights;
        constexpr int64_t max_gather_elements = 256LL * 1024 * 1024;
        const int64_t token_tile = std::max<int64_t>(
                1, max_gather_elements / (input_width * kernel_elements));
        ggml_tensor* output = nullptr;
        for (int64_t first = 0; first < level.n; first += token_tile) {
            const int64_t count = std::min(token_tile, level.n - first);
            const auto begin = level.conv.begin() + first * kernel_elements;
            const auto end = begin + count * kernel_elements;
            std::vector<int32_t> tile(begin, end);
            ggml_tensor* indices = add_i32(tile, "mesh_conv_indices",
                                           {count * kernel_elements});
            ggml_tensor* gathered = ggml_get_rows(ctx, extended, indices);
            gathered = ggml_reshape_3d(ctx, gathered, input_width,
                                       kernel_elements, count);
            ggml_tensor* result = nullptr;
            for (int64_t offset = 0; offset < kernel_elements; ++offset) {
                // gathered[:, offset, :] is strided across offsets, so make
                // it contiguous before feeding ggml_mul_mat.
                ggml_tensor* neighbour = ggml_view_3d(
                        ctx, gathered, input_width, 1, count, gathered->nb[1],
                        gathered->nb[2],
                        static_cast<size_t>(offset) * gathered->nb[1]);
                neighbour = ggml_reshape_2d(ctx, ggml_cont(ctx, neighbour),
                                            input_width, count);
                ggml_tensor* offset_weight = ggml_view_2d(
                        ctx, view_weights, input_width, output_width,
                        static_cast<size_t>(input_width) *
                                ggml_type_size(view_weights->type),
                        static_cast<size_t>(offset) * view_weights->nb[1]);
                ggml_tensor* term = ggml_mul_mat(ctx, offset_weight, neighbour);
                result = result == nullptr ? term : ggml_add(ctx, result, term);
            }
            result = ggml_add(ctx, result, as_f32(ctx, bias));
            if (features->type == GGML_TYPE_F16) result = as_f16(ctx, result);
            output = output == nullptr ? result
                                       : ggml_concat(ctx, output, result, 1);
        }
        return output;
    };

    auto sparse_conv_1x1 = [&](ggml_tensor* features, const std::string& name) {
        const int64_t input_width = features->ne[0];
        ggml_tensor* weights = m->get(name + ".weight");
        ggml_tensor* bias = m->get(name + ".bias");
        GGML_ASSERT(weights != nullptr && bias != nullptr);
        GGML_ASSERT(weights->ne[1] == 1);
        GGML_ASSERT(weights->ne[0] % input_width == 0);
        const int64_t output_width = weights->ne[0] / input_width;
        // The released spconv runtime has FILTER_HWIO=false.  Its Conv1x1
        // fast path executes weight.view(Cin, Cout), a raw-memory reshape of
        // the checkpoint's saved [Cout, 1, 1, 1, Cin] tensor.  Preserve that
        // observable official contract: first expose the saved buffer as
        // [Cout, Cin], then transpose it into ggml's [Cin, Cout] GEMM layout.
        // This differs intentionally from the 3x3 implicit-GEMM path, which
        // interprets KRSC weights conventionally per kernel offset.
        // A physical transpose of a block-quantized tensor requires a
        // quantized-to-quantized CUDA copy, which ggml deliberately does not
        // implement (the blocks no longer describe independent rows after a
        // transpose).  This 1x1 path is the only such layout conversion in
        // the decoder, so establish the F32 boundary before the transpose.
        // F16 weights retain their original layout and Tensor Core path.
        ggml_tensor* layout_source = ggml_is_quantized(weights->type)
                                             ? as_f32(ctx, weights)
                                             : weights;
        ggml_tensor* saved_layout =
                ggml_reshape_2d(ctx, layout_source, output_width, input_width);
        ggml_tensor* matrix = ggml_cont(ctx, ggml_transpose(ctx, saved_layout));
        // ggml's CUDA F32 ADD kernel requires both operands to be F32.  This
        // branch is used by the official decoder's F32 input/normalization
        // path, while the checkpoint stores this Conv3d weight and bias in
        // F16.  Make the precision boundary explicit rather than relying on
        // an unsupported F32 + F16 broadcast.
        return features->type == GGML_TYPE_F16
                       ? linear_fp16(ctx, matrix, bias, features)
                       : gb_linear(ctx, as_f32(ctx, matrix), as_f32(ctx, bias),
                                   as_f32(ctx, features));
    };

    auto subdivide = [&](ggml_tensor* features, int64_t output_tokens) {
        std::vector<int32_t> parents(static_cast<size_t>(output_tokens));
        for (int64_t token = 0; token < output_tokens; ++token) {
            parents[static_cast<size_t>(token)] =
                    static_cast<int32_t>(token / 8);
        }
        ggml_tensor* indices =
                add_i32(parents, "mesh_subdivide_indices", {output_tokens});
        return get_rows_preserving_type(ctx, features, indices);
    };

    struct SubdivideBlockResult {
        ggml_tensor* output = nullptr;
        ggml_tensor* debug_output = nullptr;
    };
    auto subdivide_block = [&](int block, ggml_tensor* input,
                               const MeshConvLevel& level) {
        const std::string name = prefix + ".upsample." + std::to_string(block);
        const int64_t out_channels = block == 0 ? channels / 4 : channels / 8;
        const std::string debug_prefix =
                "upsample" + std::to_string(block) + "_";
        SubdivideBlockResult result;
        auto capture_debug = [&](const char* suffix, ggml_tensor* value) {
            if (debug_stage == debug_prefix + suffix) {
                result.debug_output = value;
            }
        };
        ggml_tensor* h_block = sparse_group_norm(
                ctx, input, m->get(name + ".act_layers.0.weight"),
                m->get(name + ".act_layers.0.bias"), 32);
        capture_debug("act_norm", h_block);
        h_block = ggml_silu(ctx, h_block);
        capture_debug("act", h_block);
        h_block = subdivide(h_block, level.n);
        capture_debug("main_sub", h_block);
        ggml_tensor* skip = subdivide(input, level.n);
        capture_debug("skip_sub", skip);
        h_block = sparse_conv(h_block, level, name + ".out_layers.0.conv");
        capture_debug("conv1", h_block);
        h_block = sparse_group_norm(ctx, h_block,
                                    m->get(name + ".out_layers.1.weight"),
                                    m->get(name + ".out_layers.1.bias"), 32);
        capture_debug("norm", h_block);
        h_block = ggml_silu(ctx, h_block);
        capture_debug("silu", h_block);
        // SparseGroupNorm32 computes in F32 but the official mixed-precision
        // ResBlock materializes F16 before each sparse convolution.
        if (official_fp16) h_block = as_f16(ctx, h_block);
        h_block = sparse_conv(h_block, level, name + ".out_layers.3.conv");
        capture_debug("conv2", h_block);
        skip = sparse_conv_1x1(skip, name + ".skip_connection.conv");
        capture_debug("skip", skip);
        GGML_ASSERT(h_block->ne[0] == out_channels &&
                    skip->ne[0] == out_channels);
        result.output = official_fp16 ? residual_fp16(ctx, h_block, skip)
                                      : ggml_add(ctx, h_block, skip);
        return result;
    };

    SubdivideBlockResult upsample0 = subdivide_block(0, h, tb->levels[1]);
    if (upsample0.debug_output != nullptr) {
        return {upsample0.debug_output};
    }
    h = upsample0.output;
    if (debug_stage == "upsample0") {
        return {h};
    }
    SubdivideBlockResult upsample1 = subdivide_block(1, h, tb->levels[2]);
    if (upsample1.debug_output != nullptr) {
        return {upsample1.debug_output};
    }
    h = upsample1.output;
    if (debug_stage == "upsample1") {
        return {h};
    }
    if (official_fp16) h = as_f32(ctx, h);
    h = gb_linear(ctx, as_f32(ctx, m->get(prefix + ".out_layer.weight")),
                  as_f32(ctx, m->get(prefix + ".out_layer.bias")), h);
    GGML_ASSERT(h->ne[0] == output_channels && h->ne[1] == n256);
    return {h};
}

}  // namespace sam3d
