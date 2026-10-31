// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "swin_transformer.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

namespace openpcdet {

namespace {

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
ggml_tensor *layer_norm(ggml_context *ctx,
                        ggml_tensor *x,
                        ggml_tensor *gamma,
                        ggml_tensor *beta,
                        float eps = 1e-5f) {
    ggml_tensor *n = ggml_norm(ctx, x, eps);
    n = ggml_add(ctx, ggml_mul(ctx, n, gamma), beta);
    return n;
}

// ---- Relative-position-index table (host, deterministic) ----
// Matches the NON-standard OpenPCDet/mmcv Swin implementation, which builds
// the index with double_step_seq + a flip(1) (NOT the standard
// (h1-h2+Wh-1)*(2Ww-1)+(w1-w2+Ww-1) formula):
//   coords[k] = h_k*(2Ww-1) + w_k                       (k = h_k*Ww + w_k)
//   rel_position_index[i, j] = coords[i] + coords[j]
//   rel_position_index = rel_position_index.flip(1)     (column j -> N-1-j)
// so  B[i, j] = (h_i + h_{N-1-j})*(2Ww-1) + (w_i + w_{N-1-j}).
std::vector<int> rel_pos_index_table(int Wh, int Ww) {
    int N = Wh * Ww;
    int span = 2 * Ww - 1;
    std::vector<int> idx(N * N);
    for (int i = 0; i < N; i++) {
        int hi = i / Ww, wi = i % Ww;
        for (int j = 0; j < N; j++) {
            int m = (N - 1 - j);  // flipped column index
            int hm = m / Ww, wm = m % Ww;
            idx[i * N + j] = (hi + hm) * span + (wi + wm);
        }
    }
    return idx;
}

// ---- Combined cyclic-shift + window-partition FORWARD indices ----
// The reference does torch.roll(-shift, -shift) then window_partition on
// [H, W]. Window layout:
//   win_idx  = (h_win * nWw + w_win), h_win=h/Ws, w_win=w/Ws
//   pos_in_win = (h%Ws)*Ws + (w%Ws)
//   flat_win_idx = pos_in_win + Ws² * win_idx
// A single gather maps each window-flat position (p + Ws²*wi) directly to the
// original (un-rolled) spatial row n = h*sW+w, folding in the SW-MSA roll:
//   rolled (h',w') = (wh*Ws+lh, ww*Ws+lw); original = rolled + shift (mod).
std::vector<int32_t> compute_fwd_partition_indices(int sH,
                                                   int sW,
                                                   int Ws,
                                                   int shift) {
    int nWw = sW / Ws;
    int nw = (sH / Ws) * nWw, Ws2 = Ws * Ws;
    std::vector<int32_t> idx((size_t)Ws2 * nw);
    for (int wi = 0; wi < nw; wi++) {
        int wh = wi / nWw, ww = wi % nWw;
        for (int p = 0; p < Ws2; p++) {
            int lh = p / Ws, lw = p % Ws;
            int h = wh * Ws + lh;       // rolled H
            int w = ww * Ws + lw;       // rolled W
            int oh = (h + shift) % sH;  // original H
            int ow = (w + shift) % sW;  // original W
            idx[p + Ws2 * wi] = oh * sW + ow;
        }
    }
    return idx;
}

// ---- Reverse of the above: original spatial row -> window-flat position ----
// For original (h, w), the rolled position holding its value is
// r = ((h-shift)%sH)*sW + ((w-shift)%sW); map that to its window-flat index.
std::vector<int32_t> compute_rev_partition_indices(int sH,
                                                   int sW,
                                                   int Ws,
                                                   int shift) {
    int nWw = sW / Ws;
    int Ws2 = Ws * Ws;
    std::vector<int32_t> idx((size_t)sH * sW);
    for (int h = 0; h < sH; h++)
        for (int w = 0; w < sW; w++) {
            int rh = (h - shift + sH) % sH;  // rolled H
            int rw = (w - shift + sW) % sW;  // rolled W
            int wh = rh / Ws, lh = rh % Ws;
            int ww = rw / Ws, lw = rw % Ws;
            int win = wh * nWw + ww;
            idx[h * sW + w] = lh * Ws + lw + Ws2 * win;
        }
    return idx;
}

// ---- Attention mask for shifted-window (SW-MSA) ----
// Returns [nW, N, N] row-major, N=Ws*Ws, nW=nH*nW_w.
std::vector<float> build_shift_mask(int H_pad, int W_pad, int Ws, int shift) {
    int nH = H_pad / Ws, nW = W_pad / Ws;
    int nw = nH * nW, N = Ws * Ws;
    std::vector<int> img_mask(H_pad * W_pad, 0);
    int cnt = 0;
    auto h_slice = [&](int i) -> std::pair<int, int> {
        if (nH <= 1) return {0, H_pad};
        if (i == 0) return {0, H_pad - Ws};
        if (i == 1) return {H_pad - Ws, H_pad - shift};
        return {H_pad - shift, H_pad};
    };
    auto w_slice = [&](int i) -> std::pair<int, int> {
        if (nW <= 1) return {0, W_pad};
        if (i == 0) return {0, W_pad - Ws};
        if (i == 1) return {W_pad - Ws, W_pad - shift};
        return {W_pad - shift, W_pad};
    };
    for (int hi = 0; hi < 3; hi++)
        for (int wi = 0; wi < 3; wi++) {
            auto [hs, he] = h_slice(hi);
            auto [ws, we] = w_slice(wi);
            for (int h = hs; h < he; h++)
                for (int w = ws; w < we; w++) img_mask[h * W_pad + w] = cnt;
            cnt++;
        }
    std::vector<float> mask(nw * N * N, 0.0f);
    for (int win = 0; win < nw; win++) {
        int wh = win / nW, ww = win % nW;
        std::vector<int> idx(N);
        for (int h = 0; h < Ws; h++)
            for (int w = 0; w < Ws; w++)
                idx[h * Ws + w] =
                        img_mask[(wh * Ws + h) * W_pad + (ww * Ws + w)];
        float *out = mask.data() + win * N * N;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                out[i * N + j] = (idx[i] != idx[j]) ? -100.0f : 0.0f;
    }
    return mask;
}

}  // namespace

// ---- Weight loading ----

bool load_swin_weights(model_file &file,
                       const hparams &hp,
                       swin_weights &w,
                       std::string &error) {
    const int embed_dims = hp.image_swin_embed_dims;
    const auto &depths = hp.image_swin_depths;
    const auto &out_indices = hp.image_swin_out_indices;
    const int num_stages = (int)depths.size();

    w.patch_embed_w =
            file.require("image_backbone.patch_embed.projection.weight");
    w.patch_embed_b =
            file.require("image_backbone.patch_embed.projection.bias");
    w.patch_norm_w = file.require("image_backbone.patch_embed.norm.weight");
    w.patch_norm_b = file.require("image_backbone.patch_embed.norm.bias");
    if (!file.error.empty()) {
        error = file.error;
        return false;
    }

    int in_channels = embed_dims;
    w.stages.resize(num_stages);
    for (int si = 0; si < num_stages; si++) {
        auto &st = w.stages[si];
        st.blocks.resize(depths[si]);
        for (int bi = 0; bi < depths[si]; bi++) {
            auto &blk = st.blocks[bi];
            char buf[256];
#define PFX(fmt, ...)                                   \
    do {                                                \
        snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
    } while (0)

            PFX("image_backbone.stages.%d.blocks.%d.norm1.weight", si, bi);
            blk.norm1_w = file.require(buf);
            PFX("image_backbone.stages.%d.blocks.%d.norm1.bias", si, bi);
            blk.norm1_b = file.require(buf);

            PFX("image_backbone.stages.%d.blocks.%d.attn.w_msa.qkv.weight", si,
                bi);
            blk.attn.qkv_w = file.require(buf);
            PFX("image_backbone.stages.%d.blocks.%d.attn.w_msa.qkv.bias", si,
                bi);
            blk.attn.qkv_b = file.require(buf);
            PFX("image_backbone.stages.%d.blocks.%d.attn.w_msa.proj.weight", si,
                bi);
            blk.attn.proj_w = file.require(buf);
            PFX("image_backbone.stages.%d.blocks.%d.attn.w_msa.proj.bias", si,
                bi);
            blk.attn.proj_b = file.require(buf);
            // rel_bias: precomputed relative position bias [nH, Ws2*Ws2]
            PFX("image_backbone.stages.%d.blocks.%d.attn.w_msa.rel_bias", si,
                bi);
            blk.attn.rel_pos_bias = file.require(buf);

            PFX("image_backbone.stages.%d.blocks.%d.norm2.weight", si, bi);
            blk.norm2_w = file.require(buf);
            PFX("image_backbone.stages.%d.blocks.%d.norm2.bias", si, bi);
            blk.norm2_b = file.require(buf);

            PFX("image_backbone.stages.%d.blocks.%d.ffn.layers.0.0.weight", si,
                bi);
            blk.ffn.fc1_w = file.require(buf);
            PFX("image_backbone.stages.%d.blocks.%d.ffn.layers.0.0.bias", si,
                bi);
            blk.ffn.fc1_b = file.require(buf);
            PFX("image_backbone.stages.%d.blocks.%d.ffn.layers.1.weight", si,
                bi);
            blk.ffn.fc2_w = file.require(buf);
            PFX("image_backbone.stages.%d.blocks.%d.ffn.layers.1.bias", si, bi);
            blk.ffn.fc2_b = file.require(buf);
#undef PFX
            if (!file.error.empty()) {
                error = file.error;
                return false;
            }
        }
        if (si < num_stages - 1) {
            st.has_downsample = true;
            char pfx[256];
            snprintf(pfx, sizeof(pfx), "image_backbone.stages.%d.downsample",
                     si);
            std::string p(pfx);
            st.downsample.norm_w = file.require((p + ".norm.weight").c_str());
            st.downsample.norm_b = file.require((p + ".norm.bias").c_str());
            st.downsample.reduction_w =
                    file.require((p + ".reduction.weight").c_str());
            if (!file.error.empty()) {
                error = file.error;
                return false;
            }
            in_channels *= 2;
        }
    }

    w.out_norm_w.resize(out_indices.size());
    w.out_norm_b.resize(out_indices.size());
    for (size_t i = 0; i < out_indices.size(); i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "image_backbone.norm%d.weight",
                 out_indices[i]);
        w.out_norm_w[i] = file.require(buf);
        snprintf(buf, sizeof(buf), "image_backbone.norm%d.bias",
                 out_indices[i]);
        w.out_norm_b[i] = file.require(buf);
        if (!file.error.empty()) {
            error = file.error;
            return false;
        }
    }
    return true;
}

// ---- Forward ----

// Cross-frame graph cache: the swin graph structure is identical every frame
// (fixed W/H/B/hp), but rebuilding it costs 102-104 ms of pure CPU (see
// REPORT r13). Cache ctx+graph+outputs on success and reuse them; input data
// is overwritten per frame. Not used in dump mode (taps alter the graph).
struct swin_graph_cache {
    bool valid = false;
    ggml_context *ctx = nullptr;
    ggml_cgraph *gf = nullptr;
    ggml_tensor *input = nullptr;
    int W = 0, H = 0, B = 0;
    std::vector<ggml_tensor *> outputs;
    std::vector<int> out_W, out_H, out_C;
    // Constants (window masks/indices) uploaded during build; the shared
    // gallocr re-plans buffers between frames and drops their data, so the
    // hit path must re-feed them.
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> const_sets;
};
static swin_graph_cache g_swin_cache;

bool run_swin_transformer(engine_backend &backend,
                          const hparams &hp,
                          const swin_weights &w,
                          const float *images,
                          int batch_size,
                          int image_height,
                          int image_width,
                          swin_output &result,
                          const tap_sink &sink,
                          std::string &error) {
    const int embed_dims = hp.image_swin_embed_dims;
    const int Ws = hp.image_swin_window_size;
    const int mlp_ratio = hp.image_swin_mlp_ratio;
    const auto &depths = hp.image_swin_depths;
    const auto &num_heads = hp.image_swin_num_heads;
    const auto &out_indices = hp.image_swin_out_indices;
    const int num_stages = (int)depths.size();

    // Generous node count: all blocks for all cameras in one graph
    const size_t graph_nodes = 32768;
    const bool dump_mode = false;  // AICore port note: geom dump dropped
    // ---- Graph-cache hit path: skip the 102 ms rebuild entirely ----
    // DISABLED by default: gallocr re-planning between frames drops the
    // constant tensor data (masks/indices uploaded during build); re-feeding
    // them on the hit path is pending. Opt in for testing only.
    const bool swin_cache_on =
            false;  // AICore port note: env opt-in cache dropped
    if (swin_cache_on && !dump_mode && g_swin_cache.valid &&
        g_swin_cache.W == image_width && g_swin_cache.H == image_height &&
        g_swin_cache.B == batch_size) {
        ggml_context *ctx = g_swin_cache.ctx;
        ggml_cgraph *gf = g_swin_cache.gf;
        ggml_tensor *input = g_swin_cache.input;
        {
            std::vector<float> input_buf((size_t)image_width * image_height *
                                         3 * batch_size);
            for (int b = 0; b < batch_size; b++)
                for (int y = 0; y < image_height; y++)
                    for (int x = 0; x < image_width; x++)
                        for (int c = 0; c < 3; c++)
                            input_buf[(size_t)x +
                                      image_width * ((size_t)y +
                                                     image_height *
                                                             ((size_t)c +
                                                              3 * (size_t)b))] =
                                    images[((size_t)b * image_height + y) *
                                                   image_width * 3 +
                                           (size_t)x * 3 + c];
            ggml_backend_tensor_set(input, input_buf.data(), 0,
                                    input_buf.size() * sizeof(float));
        }
        // Re-allocate through the shared gallocr: other graphs (fpn/head/...)
        // re-plan it between frames and would otherwise leave this graph's
        // tensors dangling.
        if (!ggml_gallocr_alloc_graph(backend.galloc, gf)) {
            error = "swin cached graph alloc failed";
            g_swin_cache = {};
            ggml_free(ctx);
            return false;
        }
        // gallocr re-planning drops constant data (masks/indices) — re-feed.
        for (auto &ds : g_swin_cache.const_sets)
            ggml_backend_tensor_set(ds.first, ds.second.data(), 0,
                                    ds.second.size());
        if (ggml_backend_graph_compute(backend.be, gf) != GGML_STATUS_SUCCESS) {
            error = "swin graph compute failed (cached)";
            g_swin_cache = {};
            ggml_free(ctx);
            return false;
        }
        result.features.resize(g_swin_cache.outputs.size());
        result.widths.resize(g_swin_cache.outputs.size());
        result.heights.resize(g_swin_cache.outputs.size());
        result.channels.resize(g_swin_cache.outputs.size());
        for (size_t i = 0; i < g_swin_cache.outputs.size(); i++) {
            const int out_C = g_swin_cache.out_C[i];
            const int oW = g_swin_cache.out_W[i], oH = g_swin_cache.out_H[i];
            const size_t n = (size_t)out_C * oH * oW * batch_size;
            std::vector<float> buf(n);
            ggml_backend_tensor_get(g_swin_cache.outputs[i], buf.data(), 0,
                                    n * sizeof(float));
            result.widths[i] = oW;
            result.heights[i] = oH;
            result.channels[i] = out_C;
            result.features[i].resize(n);
            const size_t HW = (size_t)oH * oW;
            for (int b = 0; b < batch_size; b++)
                for (int c = 0; c < out_C; c++)
                    for (int h = 0; h < oH; h++)
                        for (int w = 0; w < oW; w++) {
                            size_t s = (size_t)c +
                                       (size_t)out_C * ((size_t)h * oW + w) +
                                       (size_t)out_C * HW * b;
                            result.features[i][(size_t)b * out_C * HW +
                                               (size_t)c * HW + (size_t)h * oW +
                                               w] = buf[s];
                        }
        }
        return true;
    }

    ggml_init_params gp = {
            ggml_tensor_overhead() * graph_nodes +
                    ggml_graph_overhead_custom(graph_nodes, false),
            nullptr,
            /*no_alloc=*/true,
    };
    ggml_context *ctx = ggml_init(gp);
    if (!ctx) {
        error = "graph ctx init failed";
        return false;
    }
    const bool prof_sw = false;
    auto sw_lap = [&](const char *nm,
                      std::chrono::steady_clock::time_point &t0) {
        if (!prof_sw) return;
        auto t1 = std::chrono::steady_clock::now();
        std::fprintf(
                stderr, "[swin] %-14s %7.2f ms\n", nm,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        t0 = t1;
    };
    auto sw_t0 = std::chrono::steady_clock::now();

    // Input: [W, H, 3, B]
    ggml_tensor *input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, image_width,
                                            image_height, 3, batch_size);
    ggml_set_name(input, "swin_input");
    ggml_set_input(input);

    // ---- Patch Embedding ----
    int H = image_height / 4, W = image_width / 4;
    fprintf(stderr,
            "[SWIN-PE] conv w ne=(%ld,%ld,%ld,%ld) input "
            "ne=(%ld,%ld,%ld,%ld)\n",
            w.patch_embed_w->ne[0], w.patch_embed_w->ne[1],
            w.patch_embed_w->ne[2], w.patch_embed_w->ne[3], input->ne[0],
            input->ne[1], input->ne[2], input->ne[3]);
    ggml_tensor *x =
            ggml_conv_2d_direct(ctx, w.patch_embed_w, input, 4, 4, 0, 0, 1, 1);
    {
        ggml_tensor *b =
                ggml_reshape_4d(ctx, w.patch_embed_b, 1, 1, embed_dims, 1);
        x = ggml_add(ctx, x, b);
    }
    ggml_tensor *conv_raw = x;  // [W',H',C,B] debug
    // ggml_conv_2d_direct emits [W', H', C, B] (W fastest). The rest of the
    // Swin pipeline operates on [C, Nc, B] with C fastest and a W-inner spatial
    // flatten (n = h*W + w, matching PyTorch [N,C,H,W]). permute(1,2,0,3)
    // reorders the (W,H,C,B) output to (C,W,H,B) so after cont + reshape the
    // merged spatial index is W-inner (verified against ggml_permute convention
    // ne[axis_i]=ne[i]).
    x = ggml_permute(ctx, x, 1, 2, 0, 3);
    x = ggml_cont(ctx, x);
    x = ggml_reshape_3d(ctx, x, embed_dims, H * W, batch_size);
    ggml_tensor *x_before_norm = x;  // pre-norm raw conv [embed_dims, H*W, B]
    if (w.patch_norm_w) x = layer_norm(ctx, x, w.patch_norm_w, w.patch_norm_b);

    int cur_H = H, cur_W = W;
    ggml_tensor *patch_embed_conv_out = x_before_norm;

    struct StageOut {
        ggml_tensor *feat;
        int H, W, C;
    };
    std::vector<StageOut> stage_outs;
    ggml_tensor *patch_embed_out = nullptr;  // [embed_dims, H*W, B] tap
    int pe_H = H, pe_W = W;
    patch_embed_out = x;  // [embed_dims, H*W, B], contiguous
    ggml_set_name(patch_embed_out, "patch_embed_out");

    // Auxiliary constant tensors (window indices / shift masks) are written to
    // the backend only after the graph is allocated by gallocr.
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> deferred_sets;

    // Per-block outputs (cropped back to unpadded) for divergence localization.
    struct BlockTap {
        ggml_tensor *t;
        int si, bi, C, H, W;
    };
    std::vector<BlockTap> block_taps;
    // Window-partitioned norm1 output and attention output (block-0 focus).
    std::vector<ggml_tensor *> win_in_batch;
    std::vector<ggml_tensor *> attn_out_batch;
    std::vector<std::pair<std::string, ggml_tensor *>> win_in_taps;
    std::vector<std::pair<std::string, ggml_tensor *>> attn_out_taps;
    ggml_tensor *qkv_tap = nullptr;
    ggml_tensor *attn_raw_tap = nullptr;
    ggml_tensor *Q_tap = nullptr;
    ggml_tensor *K_tap = nullptr;
    ggml_tensor *ds_tap = nullptr;
    ggml_tensor *norm1_before_win =
            nullptr;  // LayerNorm output BEFORE window partition
    ggml_tensor *identity_tap = nullptr;  // block input (before LayerNorm)

    for (int si = 0; si < num_stages; si++) {
        int C = embed_dims * (1 << si);
        fprintf(stderr, "[SWIN] stage %d C=%d curH=%d curW=%d\n", si, C, cur_H,
                cur_W);
        int nH = num_heads[si];
        int head_dim = C / nH;
        float scale_val = 1.0f / std::sqrt((float)head_dim);

        int H_pad = ((cur_H + Ws - 1) / Ws) * Ws;
        int W_pad = ((cur_W + Ws - 1) / Ws) * Ws;
        bool pad_needed = (H_pad != cur_H || W_pad != cur_W);
        int sH = H_pad, sW = W_pad;

        // Keep stage_x UNPADDED — padding happens INSIDE each block after
        // norm1, matching the mmcv reference: norm1(unpadded) → pad → window →
        // attn → unpad.
        ggml_tensor *stage_x = x;

        for (int bi = 0; bi < depths[si]; bi++) {
            bool shifted = (bi % 2 == 1);
            int shift = shifted ? Ws / 2 : 0;
            auto &blk = w.stages[si].blocks[bi];
            int N = sH * sW;
            int nw = (sH / Ws) * (sW / Ws);
            int Ws2 = Ws * Ws;

            // Pre-compute shift mask if needed (shared across batches)
            ggml_tensor *mask_t = nullptr;
            if (shifted) {
                auto mask_data = build_shift_mask(sH, sW, Ws, shift);
                // Store as [Ws2, Ws2, nH*nw] in GGML column-major.
                // mask_data is [nw, Ws2, Ws2] row-major.
                // Layout: element at logical [j, i, h*nWw+w] stored at
                // j + Ws2*(i + Ws2*(h*nWw+w))
                int total = Ws2 * Ws2 * nH * nw;
                mask_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Ws2, Ws2,
                                            nH * nw);
                std::vector<float> mask_buf(total, 0.0f);
                for (int wi = 0; wi < nw; wi++)
                    for (int i = 0; i < Ws2; i++)
                        for (int j = 0; j < Ws2; j++) {
                            float v = mask_data[wi * Ws2 * Ws2 + i * Ws2 + j];
                            for (int hh = 0; hh < nH; hh++) {
                                size_t dst =
                                        j +
                                        (size_t)Ws2 *
                                                (i +
                                                 (size_t)Ws2 * (hh + nH * wi));
                                mask_buf[dst] = v;
                            }
                        }
                deferred_sets.emplace_back(
                        mask_t,
                        std::vector<uint8_t>(
                                (uint8_t *)mask_buf.data(),
                                (uint8_t *)mask_buf.data() +
                                        mask_buf.size() * sizeof(float)));
            }

            // Pre-compute PAD indices for norm1→pad (once per block, per-batch
            // below)
            ggml_tensor *pad_zero_col = nullptr;
            if (pad_needed) {
                pad_zero_col = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, 1);
                std::vector<float> z(C, 0.0f);
                deferred_sets.emplace_back(
                        pad_zero_col,
                        std::vector<uint8_t>((uint8_t *)z.data(),
                                             (uint8_t *)z.data() +
                                                     z.size() * sizeof(float)));
            }
            int N_unpad = cur_H * cur_W;
            std::vector<int32_t> pad_idx_vec;
            if (pad_needed) {
                pad_idx_vec.resize((size_t)N);
                for (int np = 0; np < N; np++) {
                    int hp = np / sW, wp = np % sW;
                    if (hp < cur_H && wp < cur_W)
                        pad_idx_vec[np] = (int32_t)(hp * (int64_t)cur_W + wp);
                    else
                        pad_idx_vec[np] = (int32_t)N_unpad;  // zero column
                }
            }
            // Pre-compute CROP indices for attn_out→unpad
            std::vector<int32_t> crop_idx_vec;
            if (pad_needed) {
                crop_idx_vec.resize((size_t)N_unpad);
                for (int n = 0; n < N_unpad; n++) {
                    int h = n / cur_W, w = n % cur_W;
                    crop_idx_vec[n] = h * sW + w;
                }
            }

            // Process each batch element separately (ggml_permute is 4D-only)
            win_in_batch.clear();
            attn_out_batch.clear();
            std::vector<ggml_tensor *> batch_outs(batch_size);

            for (int b = 0; b < batch_size; b++) {
                // Extract single batch: [C, N_unpad, 1] (UNPADDED)
                ggml_tensor *bx = ggml_view_3d(
                        ctx, stage_x, C, N_unpad, 1, (size_t)C * sizeof(float),
                        (size_t)C * N_unpad * sizeof(float),
                        (size_t)b * C * N_unpad * sizeof(float));
                ggml_tensor *identity = bx;

                // ---- Norm1 on UNPADDED input (matches mmcv reference) ----
                ggml_tensor *normed =
                        layer_norm(ctx, bx, blk.norm1_w, blk.norm1_b);

                // Tap: norm1 output BEFORE padding/window [C, N_unpad, 1]
                if (sink && si == 0 && bi == 0 && b == 0) {
                    ggml_tensor *n1tap =
                            ggml_reshape_2d(ctx, normed, C, N_unpad);
                    ggml_set_output(n1tap);
                    norm1_before_win = n1tap;
                }
                // Tap: block input [C, N_unpad, 1]
                if (sink && si == 0 && bi == 0 && b == 0) {
                    ggml_tensor *idtap =
                            ggml_reshape_2d(ctx, identity, C, N_unpad);
                    ggml_set_output(idtap);
                    identity_tap = idtap;
                }

                // ---- Pad norm1 output (matches REF: norm1→pad→window) ----
                ggml_tensor *normed_padded = normed;
                if (pad_needed) {
                    ggml_tensor *flat_n =
                            ggml_reshape_2d(ctx, normed, C, N_unpad);
                    ggml_tensor *src_b =
                            ggml_concat(ctx, flat_n, pad_zero_col, 1);
                    ggml_tensor *pidx_t = ggml_new_tensor_1d(
                            ctx, GGML_TYPE_I32, pad_idx_vec.size());
                    deferred_sets.emplace_back(
                            pidx_t, std::vector<uint8_t>(
                                            (uint8_t *)pad_idx_vec.data(),
                                            (uint8_t *)pad_idx_vec.data() +
                                                    pad_idx_vec.size() *
                                                            sizeof(int32_t)));
                    normed_padded = ggml_get_rows(ctx, src_b, pidx_t);
                    normed_padded =
                            ggml_reshape_3d(ctx, normed_padded, C, N, 1);
                }

                // ---- Forward window partition on PADDED norm1 (SW-MSA cyclic
                // shift fused) ----
                {
                    auto fwd = compute_fwd_partition_indices(sH, sW, Ws, shift);
                    ggml_tensor *fwd_t = ggml_new_tensor_1d(
                            ctx, GGML_TYPE_I32, (int64_t)fwd.size());
                    deferred_sets.emplace_back(
                            fwd_t,
                            std::vector<uint8_t>(
                                    (uint8_t *)fwd.data(),
                                    (uint8_t *)fwd.data() +
                                            fwd.size() * sizeof(int32_t)));
                    normed_padded = ggml_reshape_2d(ctx, normed_padded, C, N);
                    normed_padded = ggml_get_rows(ctx, normed_padded,
                                                  fwd_t);  // [C, Ws2*nw]
                    normed_padded = ggml_reshape_3d(ctx, normed_padded, C, Ws2,
                                                    nw);  // [C, Ws2, nw]
                }
                if (sink) win_in_batch.push_back(normed_padded);

                // ---- Window Attention ----
                fprintf(stderr,
                        "[ATTN] bi=%d normed ne=(%ld,%ld,%ld) qkv_w "
                        "ne=(%ld,%ld) nh=%d nw=%d\n",
                        bi, normed_padded->ne[0], normed_padded->ne[1],
                        normed_padded->ne[2], blk.attn.qkv_w->ne[0],
                        blk.attn.qkv_w->ne[1], nH, nw);
                ggml_tensor *qkv =
                        ggml_mul_mat(ctx, blk.attn.qkv_w, normed_padded);
                fprintf(stderr, "[QKV] bi=%d qkv ne=(%ld,%ld,%ld) OK\n", bi,
                        qkv->ne[0], qkv->ne[1], qkv->ne[2]);
                if (blk.attn.qkv_b)
                    qkv = ggml_add(
                            ctx, qkv,
                            ggml_reshape_2d(ctx, blk.attn.qkv_b, 3 * C, 1));
                if (sink && si == 0 && bi == 0 && b == 0) {
                    ggml_set_output(qkv);
                    qkv_tap = qkv;
                }

                // Split Q, K, V from [3*C, Ws2, nw].
                // qkv is contiguous along the channel dim (stride
                // sizeof(float)), so the K/V channel blocks start at byte
                // offsets C and 2*C floats.
                size_t qkv_nb1 = (size_t)(3 * C) * sizeof(float);
                size_t qkv_nb2 = (size_t)(3 * C) * Ws2 * sizeof(float);
                size_t qkv_off_k = (size_t)C * sizeof(float);
                size_t qkv_off_v = (size_t)(2 * C) * sizeof(float);
                ggml_tensor *Q =
                        ggml_view_3d(ctx, qkv, C, Ws2, nw, qkv_nb1, qkv_nb2, 0);
                ggml_tensor *K = ggml_view_3d(ctx, qkv, C, Ws2, nw, qkv_nb1,
                                              qkv_nb2, qkv_off_k);
                ggml_tensor *V = ggml_view_3d(ctx, qkv, C, Ws2, nw, qkv_nb1,
                                              qkv_nb2, qkv_off_v);

                // Reshape to multi-head: [C, Ws2, nw] → [head_dim, Ws2, nH*nw]
                auto to_heads = [&](ggml_tensor *t) -> ggml_tensor * {
                    t = ggml_cont(ctx, t);
                    t = ggml_reshape_4d(ctx, t, head_dim, nH, Ws2, nw);
                    t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
                    return ggml_reshape_3d(ctx, t, head_dim, Ws2, nH * nw);
                };
                Q = to_heads(Q);
                K = to_heads(K);
                V = to_heads(V);
                if (sink && si == 0 && bi == 0 && b == 0) {
                    ggml_set_output(Q);
                    ggml_set_output(K);
                    Q_tap = Q;
                    K_tap = K;
                }
                fprintf(stderr,
                        "[HEADS] Q ne=(%ld,%ld,%ld) K ne=(%ld,%ld,%ld) V "
                        "ne=(%ld,%ld,%ld)\n",
                        Q->ne[0], Q->ne[1], Q->ne[2], K->ne[0], K->ne[1],
                        K->ne[2], V->ne[0], V->ne[1], V->ne[2]);

                Q = ggml_scale(ctx, Q, scale_val);
                ggml_tensor *attn = ggml_mul_mat(ctx, K, Q);
                fprintf(stderr, "[ATTNM] bi=%d attn ne=(%ld,%ld,%ld) OK\n", bi,
                        attn->ne[0], attn->ne[1], attn->ne[2]);
                // attn: [Ws2, Ws2, nH*nw]

                // ---- Relative position bias ----
                // blk.attn.rel_pos_bias: [nH, (2*Ws-1)^2] — the bias table
                // rel_pos_index_table gives indices in [0, (2*Ws-1)^2 - 1]
                {
                    auto indices = rel_pos_index_table(Ws, Ws);
                    ggml_tensor *idx_t =
                            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Ws2 * Ws2);
                    std::vector<int32_t> idx_data(Ws2 * Ws2);
                    for (int i = 0; i < Ws2 * Ws2; i++)
                        idx_data[i] = indices[i];
                    deferred_sets.emplace_back(
                            idx_t,
                            std::vector<uint8_t>(
                                    (uint8_t *)idx_data.data(),
                                    (uint8_t *)idx_data.data() +
                                            idx_data.size() * sizeof(int32_t)));
                    // Gather: rel_pos_bias [nH, table] → [nH, Ws2*Ws2]
                    // (get_rows output has head as the FASTEST dim: memory = h
                    // + nH*f).
                    ggml_tensor *gathered =
                            ggml_get_rows(ctx, blk.attn.rel_pos_bias, idx_t);
                    // Transpose [nH, f] → [f, nH] so the head becomes the
                    // SLOWEST dim (memory = f + nH*Ws2*Ws2*h). Without this,
                    // the following reshape scrambles the head/position axes
                    // (bias lands on wrong positions).
                    gathered =
                            ggml_reshape_4d(ctx, gathered, nH, Ws2 * Ws2, 1, 1);
                    gathered = ggml_cont(
                            ctx, ggml_permute(ctx, gathered, 1, 0, 2, 3));
                    // Reshape to [Ws2, Ws2, nH]: gathered_3d[i, j, h] =
                    // rel_bias[h, idx_data[i*Ws2 + j]]. Added to attn[ne0=key,
                    // ne1=query] (dim0=i=key, dim1=j=query) gives
                    // attn[key,query] += rel_bias[h, index(query,key)] —
                    // matches ref.
                    gathered = ggml_reshape_3d(ctx, gathered, Ws2, Ws2, nH);
                    gathered = ggml_cont(ctx, gathered);
                    // Broadcast: [Ws2, Ws2, nH] → [Ws2, Ws2, nH*nw] (tile per
                    // head)
                    gathered = ggml_repeat(ctx, gathered, attn);
                    attn = ggml_add(ctx, attn, gathered);
                }

                // ---- Shift mask ----
                if (shifted && mask_t) {
                    attn = ggml_add(ctx, attn, mask_t);
                }
                if (sink && si == 0 && bi == 0 && b == 0) {
                    ggml_set_output(attn);
                    attn_raw_tap = attn;
                }

                attn = ggml_soft_max(ctx, attn);
                // out[h,i] = sum_j V[h,j] * attn[j,i]  → mul_mat(V^T, attn)
                ggml_tensor *Vt =
                        ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
                ggml_tensor *out = ggml_mul_mat(ctx, Vt, attn);
                fprintf(stderr,
                        "[R417] out ne=(%ld,%ld,%ld) cont=%d Vcont=%d\n",
                        out->ne[0], out->ne[1], out->ne[2],
                        ggml_is_contiguous(out), ggml_is_contiguous(V));
                // [head_dim, Ws2, nH*nw] → [C, Ws2, nw]
                out = ggml_reshape_4d(ctx, out, head_dim, Ws2, nH, nw);
                fprintf(stderr,
                        "[R4] after reshape_4d(%ld,%ld,%ld,%ld) "
                        "ne=(%ld,%ld,%ld,%ld) cont=%d\n",
                        (long)head_dim, (long)Ws2, (long)nH, (long)nw,
                        out->ne[0], out->ne[1], out->ne[2], out->ne[3],
                        ggml_is_contiguous(out));
                out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
                fprintf(stderr,
                        "[RR3] pre reshape_3d(%ld,%ld,%ld) cont=%d ne=%ld\n",
                        (long)C, (long)Ws2, (long)nw, ggml_is_contiguous(out),
                        (long)ggml_nelements(out));
                out = ggml_reshape_3d(ctx, out, C, Ws2, nw);

                // ---- Output projection ----
                out = ggml_mul_mat(ctx, blk.attn.proj_w, out);
                if (blk.attn.proj_b)
                    out = ggml_add(ctx, out,
                                   ggml_reshape_2d(ctx, blk.attn.proj_b, C, 1));

                // ---- Reverse window partition (SW-MSA reverse shift fused)
                // ---- A single gather maps each original spatial row back to
                // the window-flat position that held its value, restoring
                // spatial row-major order directly (no separate roll-back
                // needed).
                {
                    auto rev = compute_rev_partition_indices(sH, sW, Ws, shift);
                    ggml_tensor *rev_t = ggml_new_tensor_1d(
                            ctx, GGML_TYPE_I32, (int64_t)rev.size());
                    deferred_sets.emplace_back(
                            rev_t,
                            std::vector<uint8_t>(
                                    (uint8_t *)rev.data(),
                                    (uint8_t *)rev.data() +
                                            rev.size() * sizeof(int32_t)));
                    ggml_tensor *flat =
                            ggml_reshape_2d(ctx, out, C, N);  // [C, Ws2*nw]
                    out = ggml_get_rows(ctx, flat, rev_t);    // [C, N]
                    out = ggml_reshape_3d(ctx, out, C, N, 1);
                }
                // ---- Crop attention output back to UNPADDED ----
                if (pad_needed) {
                    ggml_tensor *crop_t = ggml_new_tensor_1d(
                            ctx, GGML_TYPE_I32, crop_idx_vec.size());
                    deferred_sets.emplace_back(
                            crop_t, std::vector<uint8_t>(
                                            (uint8_t *)crop_idx_vec.data(),
                                            (uint8_t *)crop_idx_vec.data() +
                                                    crop_idx_vec.size() *
                                                            sizeof(int32_t)));
                    ggml_tensor *flat_out = ggml_reshape_2d(ctx, out, C, N);
                    out = ggml_get_rows(ctx, flat_out, crop_t);  // [C, N_unpad]
                    out = ggml_reshape_3d(ctx, out, C, N_unpad, 1);
                }
                if (sink) attn_out_batch.push_back(out);

                // ---- Residual (both unpadded) ----
                out = ggml_add(ctx, out, identity);

                // ---- FFN: Norm2 → FC1 → GELU → FC2 ----
                ggml_tensor *ffn_in =
                        layer_norm(ctx, out, blk.norm2_w, blk.norm2_b);
                ggml_tensor *ffn_out = ggml_mul_mat(ctx, blk.ffn.fc1_w, ffn_in);
                if (blk.ffn.fc1_b)
                    ffn_out =
                            ggml_add(ctx, ffn_out,
                                     ggml_reshape_2d(ctx, blk.ffn.fc1_b,
                                                     blk.ffn.fc1_w->ne[1], 1));
                ffn_out = ggml_gelu_erf(ctx, ffn_out);
                ffn_out = ggml_mul_mat(ctx, blk.ffn.fc2_w, ffn_out);
                if (blk.ffn.fc2_b)
                    ffn_out =
                            ggml_add(ctx, ffn_out,
                                     ggml_reshape_2d(ctx, blk.ffn.fc2_b,
                                                     blk.ffn.fc2_w->ne[1], 1));

                // ---- Residual ----
                out = ggml_add(ctx, out, ffn_out);

                batch_outs[b] = out;  // [C, N_unpad, 1]
            }  // end batch loop

            // Concatenate all batch results: [C, N_unpad, B]
            if (batch_size == 1) {
                stage_x = batch_outs[0];
            } else {
                stage_x = batch_outs[0];
                for (int b = 1; b < batch_size; b++)
                    stage_x = ggml_concat(ctx, stage_x, batch_outs[b], 2);
            }
            // Concat output is non-contiguous; downstream reshape requires it.
            stage_x = ggml_cont(ctx, stage_x);

            // Block-level tap: already unpadded [C, N_unpad, B]
            if (sink) {
                ggml_tensor *bt = stage_x;
                ggml_set_output(bt);
                block_taps.push_back({bt, si, bi, C, cur_H, cur_W});
            }

            // ---- win_in / attn_out taps for this block (divergence
            // localization) ----
            if (sink && !win_in_batch.empty()) {
                std::string nm = "bevfusion_win_in_s" + std::to_string(si) +
                                 "_b" + std::to_string(bi);
                ggml_tensor *wi = win_in_batch[0];
                for (int bb = 1; bb < batch_size; bb++)
                    wi = ggml_concat(ctx, wi, win_in_batch[bb],
                                     2);  // [C, Ws2, nw*B]
                wi = ggml_cont(ctx, wi);
                ggml_set_output(wi);
                win_in_taps.emplace_back(nm, wi);

                ggml_tensor *ao = attn_out_batch[0];
                for (int bb = 1; bb < batch_size; bb++)
                    ao = ggml_concat(ctx, ao, attn_out_batch[bb],
                                     2);  // [C, N_unpad, B]
                ao = ggml_cont(ctx, ao);
                ggml_set_output(ao);
                attn_out_taps.emplace_back("bevfusion_attn_out_s" +
                                                   std::to_string(si) + "_b" +
                                                   std::to_string(bi),
                                           ao);
            }
        }  // end blocks

        // stage_x is already unpadded [C, cur_H*cur_W, B] — blocks crop
        // internally
        stage_outs.push_back({stage_x, cur_H, cur_W, C});

        // ---- Downsample (PatchMerging) ----
        if (w.stages[si].has_downsample) {
            int oH = cur_H / 2, oW = cur_W / 2;
            int M = oH * oW;  // output spatial positions
            int CC = 4 * C;   // merged channel dim (Unfold c*4+k)
            // stage_x: [C, Nc, B] (Nc = cur_H*cur_W, contiguous). Flatten to
            // [1, C*Nc*B] so ggml_get_rows can gather by flat index t = c + C*n
            // + C*Nc*b.
            int Nc = cur_H * cur_W;
            ggml_tensor *flat =
                    ggml_reshape_2d(ctx, stage_x, 1, C * Nc * batch_size);
            // Build gather index over (cc, m, b): i = cc + CC*(m + M*b).
            // cc = c*4 + k with k order TL,TR,BL,BR (matches nn.Unfold channel
            // order).
            std::vector<int32_t> ds_idx((size_t)CC * M * batch_size);
            for (int b = 0; b < batch_size; b++)
                for (int m = 0; m < M; m++) {
                    int oh = m / oW, ow = m % oW;
                    // input flat positions for the 2x2 patch, k order
                    // TL,TR,BL,BR
                    int nk[4] = {
                            (2 * oh) * cur_W + (2 * ow),         // k0 TL
                            (2 * oh) * cur_W + (2 * ow + 1),     // k1 TR
                            (2 * oh + 1) * cur_W + (2 * ow),     // k2 BL
                            (2 * oh + 1) * cur_W + (2 * ow + 1)  // k3 BR
                    };
                    for (int cc = 0; cc < CC; cc++) {
                        int c = cc / 4, k = cc % 4;
                        ds_idx[cc + CC * (m + M * b)] =
                                c + C * nk[k] + C * Nc * b;
                    }
                }
            ggml_tensor *ds_idx_t =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ds_idx.size());
            deferred_sets.emplace_back(
                    ds_idx_t, std::vector<uint8_t>(
                                      (uint8_t *)ds_idx.data(),
                                      (uint8_t *)ds_idx.data() +
                                              ds_idx.size() * sizeof(int32_t)));
            // Gather → [1, CC*M*B] → reshape [CC, M*B]; LayerNorm + Linear
            // reduction
            ggml_tensor *ds = ggml_get_rows(ctx, flat, ds_idx_t);
            ds = ggml_reshape_2d(ctx, ds, CC, M * batch_size);
            ds = layer_norm(ctx, ds, w.stages[si].downsample.norm_w,
                            w.stages[si].downsample.norm_b);
            ds = ggml_mul_mat(ctx, w.stages[si].downsample.reduction_w, ds);
            // [out_ch, M*B] → [out_ch, M, B] for the next stage
            ds = ggml_reshape_3d(ctx, ds, ds->ne[0], M, batch_size);
            if (sink && si == 0) {
                ggml_set_output(ds);
                ds_tap = ds;
            }
            x = ds;  // feed next stage
            stage_x = ds;
            cur_H = oH;
            cur_W = oW;
        }
    }  // end stages

    // ---- Output norms ----
    std::vector<ggml_tensor *> outputs(out_indices.size());
    for (size_t i = 0; i < out_indices.size(); i++) {
        int idx = out_indices[i];
        auto &so = stage_outs[idx];
        int out_C = embed_dims * (1 << idx);
        // so.feat is [out_C, Nc, B] with row-major spatial (n = h*W + w).
        // LayerNorm over out_C, keep that layout as the graph output so we can
        // read it back and rearrange into [B,C,H,W] for the FPN.
        ggml_tensor *normed =
                layer_norm(ctx, so.feat, w.out_norm_w[i], w.out_norm_b[i]);
        char nm[64];
        snprintf(nm, sizeof(nm), "swin_out_%d", idx);
        ggml_set_name(normed, nm);
        ggml_set_output(normed);
        outputs[i] = normed;
    }

    // ---- Patch-embed tap (set as output for readback) ----
    ggml_set_output(patch_embed_out);
    if (patch_embed_conv_out) ggml_set_output(patch_embed_conv_out);
    if (conv_raw) ggml_set_output(conv_raw);

    // ---- Build & compute graph ----
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, graph_nodes, false);
    for (auto *t : outputs) ggml_build_forward_expand(gf, t);
    ggml_build_forward_expand(gf, patch_embed_out);
    if (patch_embed_conv_out)
        ggml_build_forward_expand(gf, patch_embed_conv_out);
    if (conv_raw) ggml_build_forward_expand(gf, conv_raw);
    for (auto &btk : block_taps) ggml_build_forward_expand(gf, btk.t);
    for (auto &tp : win_in_taps) ggml_build_forward_expand(gf, tp.second);
    for (auto &tp : attn_out_taps) ggml_build_forward_expand(gf, tp.second);
    if (qkv_tap) ggml_build_forward_expand(gf, qkv_tap);
    if (attn_raw_tap) ggml_build_forward_expand(gf, attn_raw_tap);
    if (Q_tap) ggml_build_forward_expand(gf, Q_tap);
    if (K_tap) ggml_build_forward_expand(gf, K_tap);
    if (ds_tap) ggml_build_forward_expand(gf, ds_tap);
    if (norm1_before_win) ggml_build_forward_expand(gf, norm1_before_win);
    if (identity_tap) ggml_build_forward_expand(gf, identity_tap);

    if (!ggml_gallocr_alloc_graph(backend.galloc, gf)) {
        error = "swin graph alloc failed";
        ggml_free(ctx);
        return false;
    }

    // Upload the precomputed constant tensors now that gallocr has allocated
    // them.
    for (auto &ds : deferred_sets)
        ggml_backend_tensor_set(ds.first, ds.second.data(), 0,
                                ds.second.size());

    // Scene images are stored NHWC [B, H, W, C] (row-major), but the ggml input
    // tensor is declared [W, H, C, B] (column-major, W fastest). Re-index so
    // each ggml element (x, y, c, b) holds the NHWC value at (b, y, x, c).
    {
        std::vector<float> input_buf((size_t)image_width * image_height * 3 *
                                     batch_size);
        for (int b = 0; b < batch_size; b++)
            for (int y = 0; y < image_height; y++)
                for (int x = 0; x < image_width; x++)
                    for (int c = 0; c < 3; c++) {
                        const float v = images[(((size_t)b * image_height + y) *
                                                        image_width +
                                                x) * 3 +
                                               c];
                        input_buf[(size_t)x +
                                  (size_t)image_width *
                                          (y + (size_t)image_height *
                                                       (c + 3 * b))] = v;
                    }
        ggml_backend_tensor_set(input, input_buf.data(), 0,
                                input_buf.size() * sizeof(float));
    }
    if (sink && false) {  // AICore port note: dump dropped
        std::vector<float> inbuf((size_t)image_width * image_height * 3 *
                                 batch_size);
        ggml_backend_tensor_get(input, inbuf.data(), 0,
                                inbuf.size() * sizeof(float));
        FILE *f = std::fopen("swin_input_dump.bin", "wb");
        if (f) {
            std::fwrite(inbuf.data(), sizeof(float), inbuf.size(), f);
            std::fclose(f);
        }
        fprintf(stderr, "[SWIN] dumped input ne=(%ld,%ld,%ld,%ld)\n",
                input->ne[0], input->ne[1], input->ne[2], input->ne[3]);
    }

    sw_lap("build", sw_t0);
    if (ggml_backend_graph_compute(backend.be, gf) != GGML_STATUS_SUCCESS) {
        error = "swin graph compute failed";
        ggml_free(ctx);
        return false;
    }
    sw_lap("compute", sw_t0);

    // ---- Read results ----
    result.features.resize(out_indices.size());
    result.widths.resize(out_indices.size());
    result.heights.resize(out_indices.size());
    result.channels.resize(out_indices.size());
    for (size_t i = 0; i < out_indices.size(); i++) {
        int idx = out_indices[i];
        auto &so = stage_outs[idx];
        int out_C = embed_dims * (1 << idx);
        result.widths[i] = so.W;
        result.heights[i] = so.H;
        result.channels[i] = out_C;
        size_t n = (size_t)out_C * so.H * so.W * batch_size;
        std::vector<float> buf(n);
        ggml_backend_tensor_get(outputs[i], buf.data(), 0, n * sizeof(float));
        result.features[i].resize(n);
        // outputs[i] buffer is [out_C, Nc, B] with row-major spatial n = h*W+w:
        //   idx = c + out_C*(h*W+w) + out_C*HW*b.
        // Rearrange to [B, C, H, W] row-major (as the FPN laterals expect):
        //   dst = b*(C*HW) + c*HW + (h*W+w).
        size_t HW = (size_t)so.H * so.W;
        for (int b = 0; b < batch_size; b++)
            for (int c = 0; c < out_C; c++)
                for (int h = 0; h < so.H; h++)
                    for (int w = 0; w < so.W; w++) {
                        size_t src = (size_t)c +
                                     (size_t)out_C * ((size_t)h * so.W + w) +
                                     (size_t)out_C * HW * b;
                        size_t dst = (size_t)b * out_C * HW + (size_t)c * HW +
                                     (size_t)h * so.W + w;
                        result.features[i][dst] = buf[src];
                    }

        // Dump tap in reference layout [B*H*W, C] (row-major HWC per batch).
        // result.features[i] is [B,C,H,W]; tap[b,h,w,c] = feat[b,c,h,w].
        if (sink) {
            int64_t rows = (int64_t)batch_size * so.H * so.W;
            std::vector<float> tap((size_t)rows * out_C);
            for (int b = 0; b < batch_size; b++)
                for (int h = 0; h < so.H; h++)
                    for (int w = 0; w < so.W; w++)
                        for (int c = 0; c < out_C; c++) {
                            size_t src = (size_t)b * out_C * HW +
                                         (size_t)c * HW + (size_t)h * so.W + w;
                            size_t dst = ((size_t)b * HW + (size_t)h * so.W +
                                          w) * out_C +
                                         c;
                            tap[dst] = result.features[i][src];
                        }
            char nm[64];
            snprintf(nm, sizeof(nm), "bevfusion.swin.%zu", i);
            sink(nm, tap.data(), rows, out_C);
        }
    }

    // ---- Patch-embed tap readback ----
    if (patch_embed_out && sink) {
        int64_t rows = (int64_t)batch_size * pe_H * pe_W;
        std::vector<float> buf((size_t)embed_dims * pe_H * pe_W * batch_size);
        ggml_backend_tensor_get(patch_embed_out, buf.data(), 0,
                                buf.size() * sizeof(float));
        // buf is [embed_dims, H*W, B] (C fastest); reorder to [B*H*W, C]
        std::vector<float> tap((size_t)rows * embed_dims);
        size_t HW = (size_t)pe_H * pe_W;
        for (int b = 0; b < batch_size; b++)
            for (int h = 0; h < pe_H; h++)
                for (int w = 0; w < pe_W; w++)
                    for (int c = 0; c < embed_dims; c++) {
                        size_t n = (size_t)h * pe_W + w;
                        size_t src = c + (size_t)embed_dims * (n + HW * b);
                        size_t dst = ((size_t)b * HW + n) * embed_dims + c;
                        tap[dst] = buf[src];
                    }
        sink("bevfusion.swin.patch_embed", tap.data(), rows, embed_dims);
    }
    if (patch_embed_conv_out && sink) {
        // raw pre-norm conv output [embed_dims, H*W, B] (C fastest) -> [B*H*W,
        // C]
        int64_t rows = (int64_t)batch_size * pe_H * pe_W;
        std::vector<float> buf((size_t)embed_dims * pe_H * pe_W * batch_size);
        ggml_backend_tensor_get(patch_embed_conv_out, buf.data(), 0,
                                buf.size() * sizeof(float));
        std::vector<float> tap((size_t)rows * embed_dims);
        size_t HW = (size_t)pe_H * pe_W;
        for (int b = 0; b < batch_size; b++)
            for (int n = 0; n < (int)HW; n++)
                for (int c = 0; c < embed_dims; c++) {
                    size_t src = c + (size_t)embed_dims * (n + HW * b);
                    size_t dst = ((size_t)b * HW + n) * embed_dims + c;
                    tap[dst] = buf[src];
                }
        sink("bevfusion.swin.patch_embed_conv", tap.data(), rows, embed_dims);
    }
    if (conv_raw && sink && false) {  // AICore port note: dump dropped
        size_t n = (size_t)W * H * embed_dims * batch_size;  // [W,H,C,B]
        std::vector<float> buf(n);
        ggml_backend_tensor_get(conv_raw, buf.data(), 0, n * sizeof(float));
        FILE *f = std::fopen("swin_conv_raw.bin", "wb");
        if (f) {
            std::fwrite(buf.data(), sizeof(float), n, f);
            std::fclose(f);
        }
        fprintf(stderr, "[SWIN] dumped conv_raw ne=(%ld,%ld,%ld,%ld)\n",
                conv_raw->ne[0], conv_raw->ne[1], conv_raw->ne[2],
                conv_raw->ne[3]);
    }

    // ---- Block-level taps readback (localize divergence) ----
    for (auto &btk : block_taps) {
        size_t HW = (size_t)btk.H * btk.W;
        size_t ntotal = (size_t)btk.C * HW * batch_size;
        std::vector<float> buf(ntotal);
        ggml_backend_tensor_get(btk.t, buf.data(), 0, ntotal * sizeof(float));
        // buf is [C, Nc*B] (C fastest); reorder to [B*Nc, C]
        std::vector<float> tap((size_t)batch_size * HW * btk.C);
        for (int b = 0; b < batch_size; b++)
            for (size_t n = 0; n < HW; n++)
                for (int c = 0; c < btk.C; c++) {
                    size_t src = c + (size_t)btk.C * (n + HW * b);
                    size_t dst = ((size_t)b * HW + n) * btk.C + c;
                    tap[dst] = buf[src];
                }
        char nm[64];
        snprintf(nm, sizeof(nm), "bevfusion.swinblock.s%d.b%d", btk.si, btk.bi);
        sink(nm, tap.data(), (int64_t)batch_size * HW, btk.C);
    }

    // ---- win_in (per-block window-partitioned norm1) readback ----
    for (auto &tp : win_in_taps) {
        ggml_tensor *wt = tp.second;
        int nw = wt->ne[2] / batch_size;  // windows per batch
        int Ws2 = (int)wt->ne[1];
        std::vector<float> buf((size_t)wt->ne[0] * Ws2 * wt->ne[2]);
        ggml_backend_tensor_get(wt, buf.data(), 0, buf.size() * sizeof(float));
        // buf[c + C*(p + Ws2*m)], m = b*nw + wi  -> tap[(m*Ws2+p)*C + c]
        std::vector<float> tap(buf.size());
        for (size_t m = 0; m < wt->ne[2]; m++)
            for (int p = 0; p < Ws2; p++)
                for (int c = 0; c < wt->ne[0]; c++)
                    tap[(m * Ws2 + p) * wt->ne[0] + c] =
                            buf[c + wt->ne[0] * (p + (size_t)Ws2 * m)];
        sink(("bevfusion." + tp.first).c_str(), tap.data(),
             (int64_t)wt->ne[2] * Ws2, (int64_t)wt->ne[0]);
    }

    // ---- attn_out (per-block attention output, unpadded) readback ----
    for (auto &tp : attn_out_taps) {
        ggml_tensor *at = tp.second;
        int C0 = (int)at->ne[0];
        int Nc = (int)at->ne[1];
        std::vector<float> buf((size_t)C0 * Nc * batch_size);
        ggml_backend_tensor_get(at, buf.data(), 0, buf.size() * sizeof(float));
        std::vector<float> tap((size_t)batch_size * Nc * C0);
        for (int b = 0; b < batch_size; b++)
            for (int n = 0; n < Nc; n++)
                for (int c = 0; c < C0; c++) {
                    size_t src = c + (size_t)C0 * (n + (size_t)Nc * b);
                    size_t dst = ((size_t)b * Nc + n) * C0 + c;
                    tap[dst] = buf[src];
                }
        sink(("bevfusion." + tp.first).c_str(), tap.data(),
             (int64_t)batch_size * Nc, C0);
    }

    // ---- qkv (block-0 batch-0) readback: [3C, Ws2, nw] -> [nw, Ws2, 3C] ----
    if (qkv_tap && sink) {
        int qC = (int)qkv_tap->ne[0], qWs2 = (int)qkv_tap->ne[1],
            qnw = (int)qkv_tap->ne[2];
        std::vector<float> buf(qkv_tap->ne[0] * qkv_tap->ne[1] *
                               qkv_tap->ne[2]);
        ggml_backend_tensor_get(qkv_tap, buf.data(), 0,
                                buf.size() * sizeof(float));
        std::vector<float> tap((size_t)qnw * qWs2 * qC);
        for (int wi = 0; wi < qnw; wi++)
            for (int p = 0; p < qWs2; p++)
                for (int c = 0; c < qC; c++)
                    tap[((size_t)wi * qWs2 + p) * qC + c] =
                            buf[c + (size_t)qC * (p + (size_t)qWs2 * wi)];
        sink("bevfusion.qkv.s0.b0.b0", tap.data(), (int64_t)qnw * qWs2, qC);
    }

    // ---- attn pre-softmax (block-0 batch-0): [Ws2, Ws2, nH*nw] -> [nH*nw,
    // Ws2, Ws2] ----
    if (attn_raw_tap && sink) {
        int a0 = (int)attn_raw_tap->ne[0], a1 = (int)attn_raw_tap->ne[1],
            a2 = (int)attn_raw_tap->ne[2];
        std::vector<float> buf(attn_raw_tap->ne[0] * attn_raw_tap->ne[1] *
                               attn_raw_tap->ne[2]);
        ggml_backend_tensor_get(attn_raw_tap, buf.data(), 0,
                                buf.size() * sizeof(float));
        std::vector<float> tap((size_t)a0 * a1 * a2);
        // buf is column-major [ne0=key, ne1=query, ne2=m]; reorder to [m,
        // query, key]
        for (int m = 0; m < a2; m++)
            for (int q = 0; q < a1; q++)
                for (int k = 0; k < a0; k++)
                    tap[((size_t)m * a1 + q) * a0 + k] =
                            buf[k + (size_t)a0 * (q + (size_t)a1 * m)];
        sink("bevfusion.attn_raw.s0.b0.b0", tap.data(), (int64_t)a2 * a1, a0);
    }

    // ---- Q,K (block-0 batch-0): [head_dim, Ws2, nH*nw] -> [nH*nw, Ws2,
    // head_dim] ----
    if (Q_tap && sink) {
        int hd0 = (int)Q_tap->ne[0], W0 = (int)Q_tap->ne[1],
            M = (int)Q_tap->ne[2];
        std::vector<float> buf(hd0 * W0 * M);
        ggml_backend_tensor_get(Q_tap, buf.data(), 0,
                                buf.size() * sizeof(float));
        std::vector<float> tap((size_t)M * W0 * hd0);
        for (int m = 0; m < M; m++)
            for (int p = 0; p < W0; p++)
                for (int d = 0; d < hd0; d++)
                    tap[((size_t)m * W0 + p) * hd0 + d] =
                            buf[d + (size_t)hd0 * (p + (size_t)W0 * m)];
        sink("bevfusion.qhead.s0.b0.b0", tap.data(), (int64_t)M * W0, hd0);
    }
    if (K_tap && sink) {
        int hd0 = (int)K_tap->ne[0], W0 = (int)K_tap->ne[1],
            M = (int)K_tap->ne[2];
        std::vector<float> buf(hd0 * W0 * M);
        ggml_backend_tensor_get(K_tap, buf.data(), 0,
                                buf.size() * sizeof(float));
        std::vector<float> tap((size_t)M * W0 * hd0);
        for (int m = 0; m < M; m++)
            for (int p = 0; p < W0; p++)
                for (int d = 0; d < hd0; d++)
                    tap[((size_t)m * W0 + p) * hd0 + d] =
                            buf[d + (size_t)hd0 * (p + (size_t)W0 * m)];
        sink("bevfusion.khead.s0.b0.b0", tap.data(), (int64_t)M * W0, hd0);
    }

    // ---- norm1 before window partition: [C, N] -> [N, C] HWC ----
    if (norm1_before_win && sink) {
        int nC = (int)norm1_before_win->ne[0],
            nN = (int)norm1_before_win->ne[1];
        std::vector<float> buf((size_t)nC * nN);
        ggml_backend_tensor_get(norm1_before_win, buf.data(), 0,
                                buf.size() * sizeof(float));
        std::vector<float> tap((size_t)nN * nC);
        for (int n = 0; n < nN; n++)
            for (int c = 0; c < nC; c++)
                tap[(size_t)n * nC + c] = buf[(size_t)c + (size_t)nC * n];
        sink("bevfusion.norm1_before_win_s0b0", tap.data(), (int64_t)nN, nC);
    }
    // ---- block input: [C, N] -> [N, C] HWC ----
    if (identity_tap && sink) {
        int nC = (int)identity_tap->ne[0], nN = (int)identity_tap->ne[1];
        std::vector<float> buf((size_t)nC * nN);
        ggml_backend_tensor_get(identity_tap, buf.data(), 0,
                                buf.size() * sizeof(float));
        std::vector<float> tap((size_t)nN * nC);
        for (int n = 0; n < nN; n++)
            for (int c = 0; c < nC; c++)
                tap[(size_t)n * nC + c] = buf[(size_t)c + (size_t)nC * n];
        sink("bevfusion.identity_s0b0", tap.data(), (int64_t)nN, nC);
    }

    // ---- downsample-0 output: [out_ch, M, B] -> [B*M, out_ch] ----
    if (ds_tap && sink) {
        int oc = (int)ds_tap->ne[0], M = (int)ds_tap->ne[1],
            Bt = (int)ds_tap->ne[2];
        std::vector<float> buf(oc * M * Bt);
        ggml_backend_tensor_get(ds_tap, buf.data(), 0,
                                buf.size() * sizeof(float));
        std::vector<float> tap((size_t)Bt * M * oc);
        for (int b = 0; b < Bt; b++)
            for (int m = 0; m < M; m++)
                for (int c = 0; c < oc; c++)
                    tap[((size_t)b * M + m) * oc + c] =
                            buf[c + (size_t)oc * (m + (size_t)M * b)];
        sink("bevfusion.ds0", tap.data(), (int64_t)Bt * M, oc);
    }

    // Persist the graph for reuse next frame (structure is identical; the
    // 102 ms rebuild is skipped). Dump mode keeps the original
    // free-every-frame.
    if (!dump_mode && swin_cache_on) {
        g_swin_cache.valid = true;
        g_swin_cache.ctx = ctx;
        g_swin_cache.gf = gf;
        g_swin_cache.input = input;
        g_swin_cache.W = image_width;
        g_swin_cache.H = image_height;
        g_swin_cache.B = batch_size;
        g_swin_cache.outputs = outputs;
        g_swin_cache.out_W.assign(result.widths.begin(), result.widths.end());
        g_swin_cache.out_H.assign(result.heights.begin(), result.heights.end());
        g_swin_cache.out_C.assign(result.channels.begin(),
                                  result.channels.end());
        g_swin_cache.const_sets = deferred_sets;
    } else {
        ggml_free(ctx);
    }
    return true;
}

}  // namespace openpcdet
