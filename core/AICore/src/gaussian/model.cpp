// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "model.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#ifdef AICore_HAS_CVLOG
#include <CVLog.h>
#endif

// Parallel-for over [0,n): splits into contiguous chunks, one per hardware
// thread, and joins. body(begin,end) must own disjoint output ranges (the
// post-processing loops below write one output row per index -> race-free).
template <class F>
static void parallel_for(int64_t n, F &&body) {
    if (n <= 0) return;
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    nt = (unsigned)std::min<int64_t>(nt, n);
    if (nt <= 1) {
        body(int64_t{0}, n);
        return;
    }
    const int64_t chunk = (n + nt - 1) / nt;
    std::vector<std::thread> ths;
    ths.reserve(nt);
    for (int64_t a = 0; a < n; a += chunk)
        ths.emplace_back(
                [&body, a, b = std::min(n, a + chunk)] { body(a, b); });
    for (auto &t : ths) t.join();
}

namespace aicore {
namespace gaussian {

bool model::load(const std::string &gguf_path,
                 const std::string &device,
                 int n_threads) {
    release();

    if (!file.open(gguf_path, /*with_data =*/true)) {
        error = file.error;
        return false;
    }
    if (!be.init(device, n_threads)) {
        error = be.error;
        return false;
    }
    if (!map_tensors() || !realize_weights()) {
        return false;
    }
    return true;
}

void model::release() {
    // Drop tensor pointers before freeing backing storage (matches depth
    // ModelLoader).
    patch_embed = pos_embed = view_ref = view_src = nullptr;
    output_norm = unpatch_w = unpatch_b = nullptr;
    layers.clear();

    if (weights_buf) {
        ggml_backend_buffer_free(weights_buf);
        weights_buf = nullptr;
    }
    if (device_ctx) {
        ggml_free(device_ctx);
        device_ctx = nullptr;
    }
    be.release();
    file.close();
    error.clear();
}

bool model::map_tensors() {
    auto t = [&](const std::string &name) {
        return file.require(name.c_str());
    };

    patch_embed = t("patch_embed.weight");
    pos_embed = t("pos_embed");
    view_ref = t("view_embed.ref");
    view_src = t("view_embed.src");
    output_norm = t("output_norm.weight");
    unpatch_w = t("unpatch.weight");
    unpatch_b = t("unpatch.bias");

    layers.resize(file.hp.n_layer);
    for (int i = 0; i < file.hp.n_layer; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        layer_weights &l = layers[i];
        l.attn_norm = t(p + "attn_norm.weight");
        l.wq = t(p + "attn_q.weight");
        l.wk = t(p + "attn_k.weight");
        l.wv = t(p + "attn_v.weight");
        l.wo = t(p + "attn_out.weight");
        l.ffn_norm = t(p + "ffn_norm.weight");
        l.ffn_up = t(p + "ffn_up.weight");
        l.ffn_down = t(p + "ffn_down.weight");
    }
    if (!file.error.empty()) {
        error = file.error;
        return false;
    }
    return true;
}

bool model::realize_weights() {
    if (be.is_cpu()) {
        // Zero-copy: the GGUF loaded with no_alloc=false, so all tensor data
        // lives in one contiguous ctx buffer. Wrap it as a CPU backend buffer
        // so graphs can reference loader tensors directly as leaves.
        void *base = ggml_get_mem_buffer(file.ctx);
        size_t size = ggml_get_mem_size(file.ctx);
        weights_buf = ggml_backend_cpu_buffer_from_ptr(base, size);
        if (!weights_buf) {
            error = "cpu weight buffer wrap failed";
            return false;
        }
        for (ggml_tensor *t = ggml_get_first_tensor(file.ctx); t;
             t = ggml_get_next_tensor(file.ctx, t)) {
            t->buffer = weights_buf;
        }
        return true;
    }

    // Device path: mirror weights into a no_alloc ctx, allocate on the backend,
    // upload, and repoint the weight pointers at the device tensors.
    size_t n_tensors = 0;
    for (ggml_tensor *t = ggml_get_first_tensor(file.ctx); t;
         t = ggml_get_next_tensor(file.ctx, t)) {
        n_tensors++;
    }
    ggml_init_params dp = {ggml_tensor_overhead() * (n_tensors + 8), nullptr,
                           /*no_alloc =*/true};
    device_ctx = ggml_init(dp);
    if (!device_ctx) {
        error = "device ctx init failed";
        return false;
    }
    for (ggml_tensor *s = ggml_get_first_tensor(file.ctx); s;
         s = ggml_get_next_tensor(file.ctx, s)) {
        ggml_tensor *d =
                ggml_new_tensor(device_ctx, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, ggml_get_name(s));
    }
    weights_buf = ggml_backend_alloc_ctx_tensors(device_ctx, be.be);
    if (!weights_buf) {
        error = "device weight alloc failed (out of device memory?)";
        return false;
    }
    for (ggml_tensor *s = ggml_get_first_tensor(file.ctx); s;
         s = ggml_get_next_tensor(file.ctx, s)) {
        ggml_tensor *d = ggml_get_tensor(device_ctx, ggml_get_name(s));
        ggml_backend_tensor_set(d, s->data, 0, ggml_nbytes(s));
    }
    auto remap = [&](ggml_tensor *&tref) {
        if (tref) tref = ggml_get_tensor(device_ctx, ggml_get_name(tref));
    };
    remap(patch_embed);
    remap(pos_embed);
    remap(view_ref);
    remap(view_src);
    remap(output_norm);
    remap(unpatch_w);
    remap(unpatch_b);
    for (auto &l : layers) {
        remap(l.attn_norm);
        remap(l.wq);
        remap(l.wk);
        remap(l.wv);
        remap(l.wo);
        remap(l.ffn_norm);
        remap(l.ffn_up);
        remap(l.ffn_down);
    }
    return true;
}

bool model::forward(const float *images,
                    int32_t n_views,
                    std::vector<float> &out,
                    const tap_sink &sink) {
    const hparams &h = file.hp;
    const int64_t N = n_views;
    const int64_t D = h.n_embd;                       // 1024
    const int64_t P = h.patch_size;                   // 8
    const int64_t IMG = h.image_size;                 // 512
    const int64_t C = h.in_channels;                  // 3
    const int64_t G = IMG / P;                        // 64 patches per side
    const int64_t TPV = G * G;                        // 4096 tokens / view
    const int64_t S = N * TPV;                        // global sequence length
    const int64_t nh = h.n_head;                      // 16
    const int64_t dh = h.head_dim;                    // 64
    const float scale = 1.0f / std::sqrt((float)dh);  // 1/8
    const float eps = h.ln_eps;

    // Optional host-phase timing (FREE_SPLATTER_PROFILE=1) to stderr.
    const bool prof = std::getenv("FREE_SPLATTER_PROFILE") != nullptr;
    auto t_clk = std::chrono::steady_clock::now();
    auto lap = [&](const char *name) {
        if (!prof) return;
        const auto now = std::chrono::steady_clock::now();
        const double ms =
                std::chrono::duration<double, std::milli>(now - t_clk).count();
#ifdef AICore_HAS_CVLOG
        CVLog::PrintDebug("[FS profile] %-12s %7.2f ms", name, ms);
#else
        std::fprintf(stderr, "[profile] %-12s %7.2f ms\n", name, ms);
#endif
        t_clk = now;
    };

    const size_t graph_nodes = 8192;
    ggml_init_params gp = {
            ggml_tensor_overhead() * graph_nodes +
                    ggml_graph_overhead_custom(graph_nodes, false),
            nullptr,
            /*no_alloc =*/true,
    };
    ggml_context *ctx = ggml_init(gp);
    if (!ctx) {
        error = "graph ctx init failed";
        return false;
    }

    std::vector<ggml_tensor *> tap_list;
    auto tap = [&](ggml_tensor *t, const std::string &name) {
        ggml_set_name(t, name.c_str());
        if (sink) {
            ggml_set_output(t);
            if (t->view_src) ggml_set_output(t->view_src);
            tap_list.push_back(t);
        }
        return t;
    };
    auto layernorm = [&](ggml_tensor *x, ggml_tensor *w) {
        return ggml_mul(ctx, ggml_norm(ctx, x, eps), w);
    };

    // input image, view-major NCHW -> ne=[W,H,C,N] (w fastest), data set after
    // alloc
    ggml_tensor *img = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IMG, IMG, C, N);
    ggml_set_input(img);

    // --- Piece 1: patch tokenizer ---
    // im2col (F32, stride==kernel) -> [IC*KH*KW, OW, OH, N]; matmul with the
    // conv kernel reshaped to [IC*KH*KW, OC]; reshape to tokens [D, TPV, N].
    ggml_tensor *cols = ggml_im2col(ctx, patch_embed, img, P, P, 0, 0, 1, 1,
                                    true, GGML_TYPE_F32);
    ggml_tensor *wflat = ggml_reshape_2d(ctx, patch_embed, C * P * P, D);
    ggml_tensor *conv = ggml_mul_mat(ctx, wflat, cols);  // [D, OW, OH, N]
    ggml_tensor *toks =
            ggml_reshape_3d(ctx, conv, D, TPV, N);  // [D, t=oh*G+ow, N]
    tap(ggml_reshape_2d(ctx, toks, D, S), "patch_embed");

    // + positional embedding (shared across views): pos_embed ne=[D,TPV]
    // broadcast over N
    toks = ggml_add(ctx, toks, pos_embed);
    tap(ggml_reshape_2d(ctx, toks, D, S), "tokens_pos");

    // + view embedding: ref to view 0, src to views 1..N-1. Build [D,1,N] bias.
    ggml_tensor *vbias = ggml_reshape_3d(ctx, view_ref, D, 1, 1);
    for (int64_t v = 1; v < N; v++) {
        vbias = ggml_concat(ctx, vbias, ggml_reshape_3d(ctx, view_src, D, 1, 1),
                            2);
    }
    toks = ggml_add(ctx, toks, vbias);  // broadcast over t
    ggml_tensor *x = tap(ggml_reshape_2d(ctx, toks, D, S), "tokens_in");

    // --- Piece 2: transformer blocks ---
    // FREE_SPLATTER_MAX_BLOCKS caps the block count (debug/parity iteration;
    // e.g. 1 to validate just block 0 against an M1 fixture). 0/unset = all
    // blocks.
    size_t n_blocks = layers.size();
    if (const char *env = std::getenv("FREE_SPLATTER_MAX_BLOCKS")) {
        if (int v = std::atoi(env)) n_blocks = std::min(n_blocks, (size_t)v);
    }
    for (size_t il = 0; il < n_blocks; il++) {
        const layer_weights &l = layers[il];
        const std::string L = "l" + std::to_string(il);

        ggml_tensor *an = tap(layernorm(x, l.attn_norm), L + ".attn_norm");
        ggml_tensor *q = tap(ggml_mul_mat(ctx, l.wq, an), L + ".q");  // [D, S]
        ggml_tensor *k = tap(ggml_mul_mat(ctx, l.wk, an), L + ".k");
        ggml_tensor *v = tap(ggml_mul_mat(ctx, l.wv, an), L + ".v");

        // self-attention (global, no mask). Flash attn keeps memory O(S) — the
        // explicit [S,S] scores would be ~4GB at S=8192.
        ggml_tensor *q3 = ggml_reshape_3d(ctx, q, dh, nh, S);
        ggml_tensor *k3 = ggml_reshape_3d(ctx, k, dh, nh, S);
        ggml_tensor *v3 = ggml_reshape_3d(ctx, v, dh, nh, S);
        ggml_tensor *qp =
                ggml_permute(ctx, q3, 0, 2, 1,
                             3);  // [dh, S, nh] (f32 query, required by FA)
        ggml_tensor *kp = ggml_permute(ctx, k3, 0, 2, 1, 3);
        ggml_tensor *vp = ggml_permute(ctx, v3, 0, 2, 1, 3);
        // On GPU, cast K/V to f16 so the Vulkan coopmat2 (tensor-core)
        // flash-attn path is selected — the f32 K/V the projections produce
        // falls off it
        // (~4x slower FA). softmax still accumulates in f32 (GGML_PREC_F32
        // below): this is f16 *inputs*, not an f16 softmax. CPU has no tensor
        // cores and its tiled FA handles f32 fine, so CPU keeps f32 K/V —
        // leaving the CPU-f32 strict gate and the CPU-f16 head check
        // byte-for-byte unchanged; only the (already --scale-widened) GPU path
        // moves.
        if (!be.is_cpu() && l.wk->type == GGML_TYPE_F16) {
            kp = ggml_cast(ctx, kp, GGML_TYPE_F16);
            vp = ggml_cast(ctx, vp, GGML_TYPE_F16);
        } else {
            vp = ggml_cont(ctx, vp);  // FA requires V contiguous
        }
        ggml_tensor *fa = ggml_flash_attn_ext(ctx, qp, kp, vp, nullptr, scale,
                                              0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        ggml_tensor *attn = ggml_reshape_2d(ctx, fa, nh * dh, S);  // [D, S]

        ggml_tensor *ao = tap(ggml_mul_mat(ctx, l.wo, attn), L + ".attn_out");
        x = tap(ggml_add(ctx, x, ao), L + ".attn_resid");

        ggml_tensor *fn = tap(layernorm(x, l.ffn_norm), L + ".ffn_norm");
        ggml_tensor *up =
                tap(ggml_mul_mat(ctx, l.ffn_up, fn), L + ".ffn_up");  // [4D, S]
        ggml_tensor *ge = tap(ggml_gelu_erf(ctx, up), L + ".gelu");
        ggml_tensor *dn = tap(ggml_mul_mat(ctx, l.ffn_down, ge),
                              L + ".ffn_down");  // [D, S]
        x = tap(ggml_add(ctx, x, dn), L + ".l_out");
    }

    ggml_tensor *rn = tap(layernorm(x, output_norm), "result_norm");

    // --- Piece 3: head (unpatchify) ---
    // Linear(D -> P*P*Gc) + bias -> [U, S]. The (p,q,c) pixel-unshuffle, SH
    // residual and per-channel activations run on the host (no clean ggml
    // primitive for the 6-D reshuffle; the rasterizer is downstream anyway).
    const int64_t Gc = h.gaussian_channels;  // 23
    const int64_t U = P * P * Gc;            // 1472
    ggml_tensor *logits =
            tap(ggml_add(ctx, ggml_mul_mat(ctx, unpatch_w, rn), unpatch_b),
                "head_logits");  // [U,S]
    ggml_set_output(logits);

    ggml_cgraph *gf = ggml_new_graph_custom(ctx, graph_nodes, false);
    ggml_build_forward_expand(gf, logits);
    for (ggml_tensor *t : tap_list) ggml_build_forward_expand(gf, t);
    lap("graph_build");

    if (!be.alloc_graph(gf, graph_nodes)) {
        error = "graph alloc failed";
        ggml_free(ctx);
        return false;
    }
    lap("alloc");

    ggml_backend_tensor_set(img, images, 0,
                            (size_t)N * C * IMG * IMG * sizeof(float));
    lap("upload");

    if (be.compute_graph(gf) != GGML_STATUS_SUCCESS) {
        error = "graph compute failed";
        ggml_free(ctx);
        return false;
    }
    lap("compute");

    // Read head logits [U,S] (memory j + U*s) and unshuffle on the host.
    std::vector<float> hl((size_t)U * S);
    ggml_backend_tensor_get(logits, hl.data(), 0, hl.size() * sizeof(float));
    lap("readback");

    // out: render-ready gaussians, row-major [N*IMG*IMG, Gc], row =
    // ((n*IMG + hh)*IMG + ww). gaussians_raw is the same but pre-activation.
    const int64_t PIX = N * IMG * IMG;
    std::vector<float> raw((size_t)PIX * Gc);
    auto img_at = [&](int64_t n, int64_t ch, int64_t hh, int64_t ww) {
        return images[(((n * C) + ch) * IMG + hh) * IMG + ww];
    };
    const float C0 = 0.28209479177387814f;
    parallel_for(S, [&](int64_t s0, int64_t s1) {
        for (int64_t s = s0; s < s1; s++) {
            const int64_t n = s / TPV, t = s % TPV, hp = t / G, wp = t % G;
            for (int64_t j = 0; j < U; j++) {
                const int64_t p = j / (P * Gc), q = (j / Gc) % P, c = j % Gc;
                const int64_t hh = hp * P + p, ww = wp * P + q;
                float val = hl[(size_t)j + (size_t)U * s];
                if (h.sh_residual && c >= 3 &&
                    c < 6) {  // RGB2SH into SH-DC term
                    val += (img_at(n, c - 3, hh, ww) - 0.5f) / C0;
                }
                raw[(size_t)((n * IMG + hh) * IMG + ww) * Gc + c] = val;
            }
        }
    });
    lap("unshuffle");

    // gaussians_raw tap (pre-activation), emitted before we mutate `raw`; then
    // activate in place and move into `out` (no 48 MB copy).
    if (sink) sink("gaussians_raw", raw.data(), PIX, Gc);
    const float smin = h.scale_min_act, smax = h.scale_max_act;
    auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    parallel_for(PIX, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) {
            float *g = raw.data() + (size_t)r * Gc;
            // xyz (0:3) and SH (3:15) pass through; opacity sigmoid; scale
            // mapped sigmoid; rotation (19:23) L2-normalized (quat order
            // w,x,y,z).
            g[15] = sigmoid(g[15]);
            for (int c = 16; c < 19; c++)
                g[c] = smin + (smax - smin) * sigmoid(g[c]);
            float nrm = 0.0f;
            for (int c = 19; c < 23; c++) nrm += g[c] * g[c];
            nrm = std::sqrt(nrm) + 1e-12f;
            for (int c = 19; c < 23; c++) g[c] /= nrm;
        }
    });
    lap("activation");
    if (sink) sink("gaussians", raw.data(), PIX, Gc);
    out = std::move(raw);

    // Stream graph taps to the sink one at a time (read into a reused temp,
    // emit, free) so host memory never holds all activations at once.
    if (sink) {
        std::vector<float> tmp;
        for (ggml_tensor *t : tap_list) {
            const int64_t rows = t->ne[ggml_n_dims(t) - 1];
            const int64_t cols = ggml_nelements(t) / rows;
            tmp.resize(ggml_nelements(t));
            if (ggml_is_contiguous(t)) {
                ggml_backend_tensor_get(t, tmp.data(), 0,
                                        ggml_nelements(t) * sizeof(float));
            } else {
                for (int64_t r = 0; r < t->ne[1]; r++) {
                    ggml_backend_tensor_get(t, tmp.data() + r * t->ne[0],
                                            r * t->nb[1],
                                            t->ne[0] * sizeof(float));
                }
            }
            sink(ggml_get_name(t), tmp.data(), rows, cols);
        }
    }

    ggml_free(ctx);
    return true;
}

}  // namespace gaussian
}  // namespace aicore
