// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/wevisone/wevisone_vision_taps_graph.h"

#include <cmath>
#include <cstring>

#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace wwv {

namespace {

ggml_tensor* layer_norm(ggml_context* ctx0,
                        ggml_tensor* x,
                        ggml_tensor* w,
                        ggml_tensor* b,
                        float eps) {
    ggml_tensor* cur = ggml_norm(ctx0, x, eps);
    if (w != nullptr) cur = ggml_mul(ctx0, cur, w);
    if (b != nullptr) cur = ggml_add(ctx0, cur, b);
    return cur;
}

// Full attention with vision m-rope, following the mtmd qwen3vl sequence.
// q_rope/k_rope/v_out are optional bisect probes (may be nullptr); the
// probe cpy nodes expand into gf so they execute with the main graph.
ggml_tensor* vision_attention(ggml_context* ctx0,
                              ggml_cgraph* gf,
                              ggml_tensor* qkv_out,
                              ggml_tensor* o_w,
                              ggml_tensor* o_b,
                              int64_t n_embd,
                              int64_t n_head,
                              int64_t n_pos,
                              float kq_scale,
                              ggml_tensor* positions,
                              ggml_tensor* q_rope_out = nullptr,
                              ggml_tensor* k_rope_out = nullptr,
                              ggml_tensor* v_out = nullptr) {
    const int64_t d_head = n_embd / n_head;
    ggml_tensor* q = ggml_view_3d(ctx0, qkv_out, d_head, n_head, n_pos,
                                  ggml_row_size(qkv_out->type, d_head),
                                  qkv_out->nb[1], 0);
    ggml_tensor* k =
            ggml_view_3d(ctx0, qkv_out, d_head, n_head, n_pos,
                         ggml_row_size(qkv_out->type, d_head), qkv_out->nb[1],
                         ggml_row_size(qkv_out->type, n_embd));
    ggml_tensor* v =
            ggml_view_3d(ctx0, qkv_out, d_head, n_head, n_pos,
                         ggml_row_size(qkv_out->type, d_head), qkv_out->nb[1],
                         ggml_row_size(qkv_out->type, 2 * n_embd));

    int32_t sections[4] = {
            static_cast<int32_t>(d_head / 4), static_cast<int32_t>(d_head / 4),
            static_cast<int32_t>(d_head / 4), static_cast<int32_t>(d_head / 4)};
    q = ggml_rope_multi(ctx0, q, positions, nullptr, d_head / 2, sections,
                        GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
    k = ggml_rope_multi(ctx0, k, positions, nullptr, d_head / 2, sections,
                        GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
    if (q_rope_out != nullptr) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, q, q_rope_out));
    }
    if (k_rope_out != nullptr) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, k, k_rope_out));
    }
    if (v_out != nullptr) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, v, v_out));
    }

    // flash_attn_ext wants (hd, T, nh); the views above are (hd, nh, T).
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    // mtmd contract (clip.cpp build_attn): the fattn kernels take K/V in
    // F16 and the V-weighted accumulation must be pinned to F32 — without
    // the acc-precision pin the 6936-term V sum drifts in f16 and the
    // block outputs diverge from the torch reference (round-10 bisect:
    // the attention stage alone was corr 0.89 with the std 12% low).
    k = ggml_cast(ctx0, k, GGML_TYPE_F16);
    v = ggml_cast(ctx0, v, GGML_TYPE_F16);
    ggml_tensor* kq =
            ggml_flash_attn_ext(ctx0, q, k, v, nullptr, kq_scale, 0.0f, 0.0f);
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    // fattn output ne is (v.ne0, q.ne2, q.ne1) = (hd, nh, T) — already the
    // permute(0,2,1,3) view of (hd, T, nh) (ggml.c's own comment); mtmd
    // reshapes straight to (n_embd, T). An extra permute here interleaves
    // channels with tokens: round-11 root cause #4 (blk0 attn corr 0.89 vs
    // torch, std 12% low, CPU == CUDA so it was graph construction, not a
    // kernel issue).
    ggml_tensor* flat = ggml_cont_2d(ctx0, kq, n_embd, n_pos);
    ggml_tensor* cur = ggml_mul_mat(ctx0, o_w, flat);
    if (o_b != nullptr) cur = ggml_add(ctx0, cur, o_b);
    return cur;
}

}  // namespace

std::vector<int32_t> vision_mrope_positions(int32_t pixel_w,
                                            int32_t pixel_h,
                                            int32_t patch_size,
                                            int32_t merge_size) {
    const int pw = pixel_w / patch_size;
    const int ph = pixel_h / patch_size;
    const int n_pos = pw * ph;
    std::vector<int32_t> positions(static_cast<size_t>(n_pos) * 4, 0);
    int ptr = 0;
    for (int y = 0; y < ph; y += merge_size) {
        for (int x = 0; x < pw; x += merge_size) {
            for (int dy = 0; dy < merge_size; ++dy) {
                for (int dx = 0; dx < merge_size; ++dx) {
                    positions[static_cast<size_t>(ptr)] = y + dy;
                    positions[static_cast<size_t>(n_pos) + ptr] = x + dx;
                    positions[static_cast<size_t>(2 * n_pos) + ptr] = y + dy;
                    positions[static_cast<size_t>(3 * n_pos) + ptr] = x + dx;
                    ++ptr;
                }
            }
        }
    }
    return positions;
}

void unshuffle_tokens_host(const float* tokens,
                           int64_t n_embd,
                           int64_t n_tokens,
                           int64_t map_w,
                           int64_t map_h,
                           int32_t merge_size,
                           float* out_map) {
    // tokens: ggml (E, T) flat — element (e, t) lives at tokens[t*E + e]
    // (channel fastest), NOT a (E, T) C-order matrix; reading it with a
    // per-channel row pointer was round-11 root cause #5 (level maps
    // permuted into noise: corr 0.0003 vs the numpy recompute, while the
    // same tap re-read as tokens[t*E+e] gives corr 0.991 vs torch).
    // out_map: (E, W', H') row-major, x fastest; W' = m*Wb, H' = m*Hb.
    const int64_t m = merge_size;
    const int64_t wb = map_w;  // blocks per row of the map
    const int64_t hb = map_h;
    const int64_t w = map_w * m;  // full map width (post-merge grid)
    for (int64_t c = 0; c < n_embd; ++c) {
        for (int64_t by = 0; by < hb; ++by) {
            for (int64_t bx = 0; bx < wb; ++bx) {
                for (int64_t iy = 0; iy < m; ++iy) {
                    for (int64_t ix = 0; ix < m; ++ix) {
                        const int64_t t = ((by * wb + bx) * m + iy) * m + ix;
                        const int64_t dst = (by * m + iy) * w + bx * m + ix;
                        out_map[static_cast<size_t>(c) * (w * map_h * m) +
                                dst] =
                                tokens[static_cast<size_t>(t) * n_embd + c];
                    }
                }
            }
        }
    }
}

ggml_cgraph* build_vision_taps_graph(
        ggml_context* ctx0,
        const VisionTapsParams& params,
        const std::vector<int32_t>& taps_layers,
        const std::vector<ggml_tensor*>& tap_leaves,
        ggml_tensor* pixels,
        ggml_tensor* positions,
        const std::vector<ggml_tensor*>& patch_embd,
        ggml_tensor* patch_bias,
        ggml_tensor* position_embd,
        ggml_tensor* pre_ln_w,
        ggml_tensor* pre_ln_b,
        const std::vector<int>& layer_indexes,
        const std::vector<std::vector<ggml_tensor*>>& block_weights,
        VisionTapsResult* out,
        std::string* err) {
    const int64_t p = params.patch_size;
    const int64_t m = params.merge_size;
    if (params.grid_w % (p * m) != 0 || params.grid_h % (p * m) != 0) {
        if (err != nullptr) {
            *err = "smart-resize grid not divisible by patch*merge";
        }
        return nullptr;
    }
    if (patch_embd.size() < 2 || patch_embd[0] == nullptr ||
        patch_embd[1] == nullptr) {
        if (err != nullptr) {
            *err = "qwen3vl requires the dual patch-embed convs";
        }
        return nullptr;
    }
    if (tap_leaves.size() != taps_layers.size()) {
        if (err != nullptr) *err = "tap leaf count mismatch";
        return nullptr;
    }
    const int64_t npx = params.grid_w / p;  // full patch grid
    const int64_t npy = params.grid_h / p;
    const int64_t n_embd = patch_embd[0]->ne[3];
    if (params.heads <= 0 || n_embd % params.heads != 0) {
        if (err != nullptr) *err = "invalid head count";
        return nullptr;
    }

    // Patch embed: the kt=2 temporal conv split into two 2D convs, summed,
    // then the 2x2 spatial merge (qwen3vl.cpp build_inp_with_temporal_merge
    // + spatial-merge block): [x, y, c, b] -> [c, x, y, b] -> merged tokens
    // in (by, bx, iy, ix) order.
    ggml_tensor* inp = ggml_add(
            ctx0,
            ggml_conv_2d(ctx0, patch_embd[0], pixels, static_cast<int>(p),
                         static_cast<int>(p), 0, 0, 1, 1),
            ggml_conv_2d(ctx0, patch_embd[1], pixels, static_cast<int>(p),
                         static_cast<int>(p), 0, 0, 1, 1));
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 2, 0, 3));
    inp = ggml_cont_4d(ctx0, inp, n_embd * 2, npx / 2, npy, 1);
    inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, npx / 2, 2, npy / 2);
    inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
    inp = ggml_cont_3d(ctx0, inp, n_embd, npx * npy, 1);
    if (patch_bias != nullptr) inp = ggml_add(ctx0, inp, patch_bias);
    ggml_tensor* inp_pb = inp;  // bisect probe point: patch-embed + bias

    // Learned position embedding, resized bilinear/align_corners to the
    // target patch grid (clip_graph::resize_position_embeddings), then the
    // same 2x2 merge shuffle so the token order matches the input.
    ggml_tensor* pos = position_embd;  // (C, PW*PH) flat token table
    const int64_t n_per_side =
            static_cast<int64_t>(std::sqrt(static_cast<double>(pos->ne[1])));
    if (n_per_side * n_per_side != pos->ne[1]) {
        if (err != nullptr) *err = "position embedding is not square";
        return nullptr;
    }
    if (n_per_side != npx || n_per_side != npy) {
        // GGUF pos table (C, N) keeps n = y*x_side + x, so the (C, s, s)
        // view is (C, x, y). Bring it to (y, x, C) BEFORE interpolating:
        // the h axis (npy) is y and the w axis (npx) is x — the previous
        // chain permuted to (x, y, C) and swapped the resize targets,
        // transposing every position embedding (the round-10 bisect: pos
        // was the first divergent stage, corr 0.695).
        pos = ggml_reshape_3d(ctx0, pos, n_embd, n_per_side, n_per_side);
        pos = ggml_cont(ctx0,
                        ggml_permute(ctx0, pos, 2, 1, 0, 3));  // (y, x, C)
        pos = ggml_interpolate(
                ctx0, pos, npy, npx, n_embd, 1,
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
        // back to (C, x, y) — the same flat layout the patch-embed chain
        // feeds into the 2x2 merge shuffle below
        pos = ggml_cont(ctx0, ggml_permute(ctx0, pos, 2, 1, 0, 3));
        pos = ggml_cont_2d(ctx0, pos, n_embd, npx * npy);
    }
    {
        ggml_tensor* cur = ggml_cont_4d(ctx0, pos, n_embd * 2, npx / 2, npy, 1);
        cur = ggml_reshape_4d(ctx0, cur, n_embd * 2, npx / 2, 2, npy / 2);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_3d(ctx0, cur, n_embd, npx * npy, 1);
        inp = ggml_add(ctx0, inp, cur);
    }
    ggml_tensor* inp_pp = inp;  // bisect probe point: + position embed

    ggml_tensor* inp_l = inp;
    if (pre_ln_w != nullptr) {
        inp_l = layer_norm(ctx0, inp_l, pre_ln_w, pre_ln_b, params.eps);
    }

    const int64_t n_pos = npx * npy;
    const float kq_scale =
            1.0f / std::sqrt(static_cast<float>(n_embd / params.heads));

    // real-size graphs exceed the default 2048-node limit
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // sentinel bisect probes: -1 taps the patch-embed+bias output, -2 the
    // position-embed output (both pre pre-ln / pre blocks) into their
    // dedicated leaves so the runtime can dump raw (C, T) flats
    for (size_t ti = 0; ti < taps_layers.size(); ++ti) {
        ggml_tensor* src = taps_layers[ti] == -1   ? inp_pb
                           : taps_layers[ti] == -2 ? inp_pp
                                                   : nullptr;
        if (src != nullptr) {
            ggml_tensor* tap = ggml_cpy(ctx0, src, tap_leaves[ti]);
            ggml_build_forward_expand(gf, tap);
        }
    }

    const size_t n_blocks = block_weights.size();
    for (size_t il = 0; il < n_blocks; ++il) {
        const auto& w = block_weights[il];  // 14 slots, see the header.
        ggml_tensor* cur = inp_l;
        cur = layer_norm(ctx0, cur, w[0], w[1], params.eps);
        // block-0 sub-stage sentinels (-3 ln1 out = attn input, -4 attn
        // out post o_proj, -5 ffn out pre residual) for the torch-side
        // bisect of the first block
        ggml_tensor* s3 = il == 0 ? cur : nullptr;
        cur = ggml_mul_mat(ctx0, w[2], cur);
        if (w[3] != nullptr) cur = ggml_add(ctx0, cur, w[3]);
        ggml_tensor* s6 = nullptr;  // -6: qkv projection output (pre rope)
        ggml_tensor* s7 = nullptr;  // -7: rope'd Q
        ggml_tensor* s8 = nullptr;  // -8: V (never roped)
        if (il == 0) {
            for (size_t ti = 0; ti < taps_layers.size(); ++ti) {
                if (taps_layers[ti] == -6)
                    s6 = tap_leaves[ti];
                else if (taps_layers[ti] == -7)
                    s7 = tap_leaves[ti];
                else if (taps_layers[ti] == -8)
                    s8 = tap_leaves[ti];
            }
            if (s6 != nullptr) {
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, s6));
            }
        }
        cur = vision_attention(ctx0, gf, cur, w[4], w[5], n_embd, params.heads,
                               n_pos, kq_scale, positions, s7, nullptr, s8);
        ggml_tensor* s4 = il == 0 ? cur : nullptr;
        cur = ggml_add(ctx0, cur, inp_l);
        inp_l = cur;
        cur = layer_norm(ctx0, cur, w[6], w[7], params.eps);
        // FFN: up (+ gate with SiLU? qwen3vl vision uses GELU gate per the
        // mtmd ffn_op), then down.
        ggml_tensor* up = ggml_mul_mat(ctx0, w[8], cur);
        if (w[9] != nullptr) up = ggml_add(ctx0, up, w[9]);
        if (w[10] != nullptr) {
            ggml_tensor* gate = ggml_mul_mat(ctx0, w[10], cur);
            if (w[11] != nullptr) gate = ggml_add(ctx0, gate, w[11]);
            gate = ggml_gelu(ctx0, gate);
            up = ggml_mul(ctx0, up, gate);
        } else {
            up = ggml_gelu(ctx0, up);
        }
        cur = ggml_mul_mat(ctx0, w[12], up);
        if (w[13] != nullptr) cur = ggml_add(ctx0, cur, w[13]);
        ggml_tensor* s5 = il == 0 ? cur : nullptr;
        cur = ggml_add(ctx0, inp_l, cur);
        inp_l = cur;

        if (il == 0) {
            for (size_t ti = 0; ti < taps_layers.size(); ++ti) {
                ggml_tensor* src = taps_layers[ti] == -3   ? s3
                                   : taps_layers[ti] == -4 ? s4
                                   : taps_layers[ti] == -5 ? s5
                                                           : nullptr;
                if (src != nullptr) {
                    ggml_tensor* tap = ggml_cpy(ctx0, src, tap_leaves[ti]);
                    ggml_build_forward_expand(gf, tap);
                }
            }
        }

        for (size_t ti = 0; ti < taps_layers.size(); ++ti) {
            if (taps_layers[ti] == static_cast<int32_t>(il)) {
                ggml_build_forward_expand(gf, cur);
                ggml_tensor* tap = ggml_cpy(ctx0, cur, tap_leaves[ti]);
                ggml_build_forward_expand(gf, tap);
            }
        }
    }

    out->taps.assign(tap_leaves.begin(), tap_leaves.end());
    out->n_tokens = n_pos;
    out->token_grid_w = npx;
    out->token_grid_h = npy;
    out->n_embd = n_embd;
    return gf;
}

}  // namespace wwv
