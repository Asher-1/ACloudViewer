// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "querysplat_lpips_graph.hpp"

#include <cmath>
#include <cstring>
#include <map>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "map_anything_common.hpp"
#include "map_anything_gguf_loader.hpp"
#include "querysplat_log.hpp"

namespace qs {

// Task-identity logging (see querysplat_log.hpp): querysplat messages must
// not carry the shared runtime's [map_anything] tag.
using qs::log_error;
using qs::log_info;

namespace {

// torchvision vgg16().features up to relu5_3: conv channel widths and the
// slice boundaries lpips.pretrained_networks.vgg16 taps (after convs 1, 3,
// 6, 9, 12); a 2x2 max-pool opens every slice but the first.
constexpr int kConvs = 13;
constexpr int kTapAfter[5] = {1, 3, 6, 9, 12};
// lpips ScalingLayer buffers (f32)
constexpr float kShift[3] = {-.030f, -.088f, -.188f};
constexpr float kScale[3] = {.458f, .448f, .450f};

}  // namespace

struct LpipsVgg::Impl {
    map_anything::Backend be;
    std::unique_ptr<map_anything::GGUFModel> model;
    ggml_context* wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor*> wt;

    // graph for one (H, W) pair batch
    int H = 0, W = 0;
    ggml_context* gctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor* in = nullptr;
    ggml_tensor* out = nullptr;

    ~Impl() {
        free_graph();
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }
    void free_graph() {
        if (galloc) ggml_gallocr_free(galloc);
        if (gctx) ggml_free(gctx);
        galloc = nullptr;
        gctx = nullptr;
        graph = nullptr;
    }

    bool stage_weights() {
        const size_t n = model->tensors.size();
        ggml_init_params wp{ggml_tensor_overhead() * n + (1u << 16), nullptr,
                            true};
        wctx = ggml_init(wp);
        if (!wctx) return false;
        for (const auto& [name, ht] : model->tensors) {
            if (ht.type != GGML_TYPE_F32) {
                log_error(
                        "lpips: %s is not F32 (convert with "
                        "convert_lpips_to_gguf.py)",
                        name.c_str());
                return false;
            }
            ggml_tensor* t = ggml_new_tensor(wctx, GGML_TYPE_F32, 4, ht.ne);
            ggml_set_name(t, name.c_str());
            wt[name] = t;
        }
        for (int i = 0; i < kConvs; i++)
            for (const char* s : {".weight", ".bias"})
                if (!wt.count("conv" + std::to_string(i) + s)) {
                    log_error("lpips: missing conv%d%s", i, s);
                    return false;
                }
        for (int l = 0; l < 5; l++)
            if (!wt.count("lin" + std::to_string(l) + ".weight")) {
                log_error("lpips: missing lin%d.weight", l);
                return false;
            }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, be.handle);
        if (!wbuf) return false;
        for (const auto& [name, ht] : model->tensors)
            ggml_backend_tensor_set(wt[name], ht.data.data(), 0,
                                    ht.data.size());
        return true;
    }

    // 3x3 / stride 1 / pad 1 conv + bias + ReLU on {W, H, Cin, N}: F32 im2col
    // {Cin*9, W, H, N} x kernel {9*Cin, Cout} -> {W, H, Cout, N}.
    ggml_tensor* conv_relu(ggml_context* ctx, ggml_tensor* x, int i) {
        ggml_tensor* k = wt["conv" + std::to_string(i) + ".weight"];
        ggml_tensor* b = wt["conv" + std::to_string(i) + ".bias"];
        ggml_tensor* col =
                ggml_im2col(ctx, k, x, 1, 1, 1, 1, 1, 1, true, GGML_TYPE_F32);
        ggml_tensor* r = ggml_mul_mat(
                ctx,
                ggml_reshape_2d(ctx, col, col->ne[0],
                                col->ne[1] * col->ne[2] * col->ne[3]),
                ggml_reshape_2d(ctx, k, k->ne[0] * k->ne[1] * k->ne[2],
                                k->ne[3]));
        r = ggml_reshape_4d(ctx, r, col->ne[1], col->ne[2], col->ne[3],
                            k->ne[3]);
        r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
        r = ggml_add(ctx, r, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
        return ggml_relu(ctx, r);
    }

    // one LPIPS layer term {1, 1} for the pair batch {w, h, C, 2}
    ggml_tensor* layer_term(ggml_context* ctx, ggml_tensor* f, int l) {
        const int64_t w = f->ne[0], h = f->ne[1], C = f->ne[2];
        // channels to ne[0]: {C, w, h, 2}
        ggml_tensor* x = ggml_cont(ctx, ggml_permute(ctx, f, 1, 2, 0, 3));
        ggml_tensor* nrm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, x)));
        x = ggml_div(ctx, x, ggml_scale_bias(ctx, nrm, 1.0f, 1e-10f));
        const size_t half = (size_t)x->nb[3];
        ggml_tensor* a = ggml_view_4d(ctx, x, C, w, h, 1, x->nb[1], x->nb[2],
                                      x->nb[3], 0);
        ggml_tensor* b = ggml_view_4d(ctx, x, C, w, h, 1, x->nb[1], x->nb[2],
                                      x->nb[3], half);
        ggml_tensor* d2 = ggml_sqr(ctx, ggml_sub(ctx, a, b));
        ggml_tensor* lw =
                wt["lin" + std::to_string(l) + ".weight"];  // {1,1,C,1}
        ggml_tensor* r =
                ggml_mul_mat(ctx, ggml_reshape_2d(ctx, lw, C, 1),
                             ggml_reshape_2d(ctx, d2, C, w * h));  // {1, w*h}
        return ggml_mean(ctx,
                         ggml_reshape_2d(ctx, r, w * h, 1));  // spatial_average
    }

    bool build(int h, int w) {
        free_graph();
        H = h;
        W = w;
        constexpr size_t kNodes = 512;
        ggml_init_params gp{ggml_tensor_overhead() * kNodes +
                                    ggml_graph_overhead_custom(kNodes, false),
                            nullptr, true};
        gctx = ggml_init(gp);
        if (!gctx) return false;
        in = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, W, H, 3, 2);
        ggml_set_input(in);
        ggml_tensor* x = in;
        ggml_tensor* val = nullptr;
        int tap = 0;
        for (int i = 0; i < kConvs; i++) {
            if (tap > 0 && i == kTapAfter[tap - 1] + 1)
                x = ggml_pool_2d(gctx, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
            x = conv_relu(gctx, x, i);
            if (i == kTapAfter[tap]) {
                ggml_tensor* t = layer_term(gctx, x, tap);
                val = val ? ggml_add(gctx, val, t)
                          : t;  // val = 0; val += res[l]
                tap++;
            }
        }
        out = val;
        ggml_set_output(out);
        graph = ggml_new_graph_custom(gctx, kNodes, false);
        ggml_build_forward_expand(graph, out);
        galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(be.handle));
        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            log_error("lpips: graph allocation failed");
            return false;
        }
        return true;
    }
};

LpipsVgg::LpipsVgg(const std::string& gguf_path, map_anything::Backend backend)
    : impl_(new Impl()) {
    impl_->be = backend;
    impl_->model = map_anything::load_gguf(gguf_path);
    if (!impl_->model || !impl_->stage_weights()) {
        impl_.reset();
        return;
    }
    log_info("lpips: staged %zu tensors from %s", impl_->model->tensors.size(),
             gguf_path.c_str());
}

LpipsVgg::~LpipsVgg() = default;

bool LpipsVgg::valid() const { return impl_ != nullptr; }

// ---- host-side forward + backward (TTO t3) -------------------------------
//
// lpips(in0=target, in1=prediction, normalize=False) exactly as
// QuerySplat._tto_losses calls it: NO 2x-1 in front of the ScalingLayer.
//   v   = (img - shift) / scale                     (ScalingLayer)
//   f_l = VGG16 relu features (13 convs, pools before convs 2,4,7,10)
//   n_l = f_l / (sqrt(sum_c f_l^2) + 1e-10)         (normalize_tensor)
//   LPIPS = sum_l mean_hw( lin_l . (n_l(in0) - n_l(in1))^2 )
// The backward w.r.t. in1 ports convTranspose (flipped kernel), the ReLU
// mask, 2x2 maxpool argmax routing (torch routes to the FIRST max) and the
// normalize_tensor VJP. Every pixel-dependent factor stays OUTSIDE the
// transposed convolutions (the fused_ssim lesson, see tto_loss.cpp).

namespace {

struct LpipsHost {
    std::vector<float> w[13], b[13];  // conv: {OC,IC,3,3} + {OC}
    std::vector<float> lin[5];        // {C}
    int cout[13];
};

// 3x3 stride-1 pad-1 conv + bias + ReLU, CHW in/out
void host_conv_relu(const float* in,
                    int C,
                    int H,
                    int W,
                    const float* w,
                    int OC,
                    const float* bias,
                    float* out) {
#ifdef QS_USE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int oc = 0; oc < OC; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float acc = bias[oc];
                for (int ic = 0; ic < C; ic++) {
                    const float* ip = in + ((size_t)ic * H + y) * W + x;
                    const float* wp = w + ((size_t)oc * C + ic) * 9;
                    float accic = 0.f;
                    for (int ky = 0; ky < 3; ky++) {
                        const int yy = y + ky - 1;
                        if (yy < 0 || yy >= H) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            const int xx = x + kx - 1;
                            if (xx < 0 || xx >= W) continue;
                            accic += ip[(ky - 1) * W + (kx - 1)] *
                                     wp[ky * 3 + kx];
                        }
                    }
                    acc += accic;
                }
                out[((size_t)oc * H + y) * W + x] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

// conv backward w.r.t. the conv INPUT (flipped kernel); writes (not adds)
void host_conv_input_grad(const float* dout,
                          int OC,
                          int H,
                          int W,
                          const float* w,
                          int C,
                          float* din) {
#ifdef QS_USE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int ic = 0; ic < C; ic++) {
                float acc = 0.f;
                for (int oc = 0; oc < OC; oc++) {
                    const float* wp = w + ((size_t)oc * C + ic) * 9;
                    for (int ky = 0; ky < 3; ky++) {
                        const int yy = y - ky + 1;
                        if (yy < 0 || yy >= H) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            const int xx = x - kx + 1;
                            if (xx < 0 || xx >= W) continue;
                            acc += dout[((size_t)oc * H + yy) * W + xx] *
                                   wp[ky * 3 + kx];
                        }
                    }
                }
                din[((size_t)ic * H + y) * W + x] = acc;
            }
        }
    }
}

constexpr float kNromEps = 1e-10f;  // normalize_tensor's +eps

}  // namespace

bool LpipsVgg::value_and_grad(const float* prediction,
                              const float* target,
                              int V,
                              int H,
                              int W,
                              std::vector<float>* out,
                              std::vector<float>* grad_prediction) {
    if (!impl_) return false;
    Impl& m = *impl_;
    static LpipsHost hw;
    static bool hw_ready = false;
    if (!hw_ready) {
        auto bytes_to_f32 = [](const std::vector<unsigned char>& bytes) {
            std::vector<float> out(bytes.size() / 4);
            std::memcpy(out.data(), bytes.data(), out.size() * 4);
            return out;
        };
        for (int i = 0; i < kConvs; i++) {
            const auto& hw_t =
                    m.model->tensors.at("conv" + std::to_string(i) + ".weight");
            const auto& hb_t =
                    m.model->tensors.at("conv" + std::to_string(i) + ".bias");
            hw.cout[i] = (int)hw_t.ne[3];
            hw.w[i] = bytes_to_f32(hw_t.data);
            hw.b[i] = bytes_to_f32(hb_t.data);
        }
        for (int l = 0; l < 5; l++)
            hw.lin[l] = bytes_to_f32(
                    m.model->tensors.at("lin" + std::to_string(l) + ".weight")
                            .data);
        hw_ready = true;
    }
    // torchvision vgg16 features channels for convs 0..12
    // (VGG_CONV_INDICES = features {0,2,5,7,10,12,14,17,19,21,24,26,28})
    static const int kCh[13] = {64,  64,  128, 128, 256, 256, 256,
                                512, 512, 512, 512, 512, 512};
    for (int i = 0; i < kConvs; i++)
        if (hw.cout[i] != kCh[i]) {
            log_error("lpips: conv%d channel mismatch (%d vs %d)", i,
                      hw.cout[i], kCh[i]);
            return false;
        }
    // per-layer sizes; pools sit before convs 2, 4, 7, 10
    int lh[13], lw[13];
    int ph = H, pw = W;
    for (int i = 0; i < kConvs; i++) {
        if (i == 2 || i == 4 || i == 7 || i == 10) {
            ph = (ph + 1) / 2;
            pw = (pw + 1) / 2;
        }
        lh[i] = ph;
        lw[i] = pw;
    }
    // pool p (0..3) feeds conv 2,4,7,10 and pools the output of conv 1,3,6,9
    static const int kPoolConv[4] = {2, 4, 7, 10};
    out->assign((size_t)V, 0.0f);
    grad_prediction->assign((size_t)V * 3 * H * W, 0.0f);

    for (int v = 0; v < V; v++) {
        const size_t in_plane = (size_t)3 * H * W;
        const float* tgt = target + (size_t)v * in_plane;
        const float* prd = prediction + (size_t)v * in_plane;
        // ---- scaling layer, normalize=False (no 2x-1) ----
        std::vector<float> f0((size_t)3 * H * W), f1((size_t)3 * H * W);
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < H * W; i++) {
                f0[(size_t)c * H * W + i] =
                        (tgt[(size_t)c * H * W + i] - kShift[c]) / kScale[c];
                f1[(size_t)c * H * W + i] =
                        (prd[(size_t)c * H * W + i] - kShift[c]) / kScale[c];
            }
        // ---- forward: 13 convs, saving the in1 side for the backward ----
        std::vector<std::vector<float>> a0(kConvs), a1(kConvs);  // relu outputs
        std::vector<std::vector<float>> pre1(4);                 // pre-pool in1
        std::vector<std::vector<int32_t>> parg(4);               // pool argmax
        int pre_ci[4], pre_ch[4], pre_cw[4];  // pre-pool shape
        int ci = 3, ch = H, cw = W;
        int pool_i = 0;
        for (int i = 0; i < kConvs; i++) {
            if (pool_i < 4 && i == kPoolConv[pool_i]) {
                const int och = (ch + 1) / 2, ocw = (cw + 1) / 2;
                pre1[pool_i] = f1;
                pre_ci[pool_i] = ci;
                pre_ch[pool_i] = ch;
                pre_cw[pool_i] = cw;
                std::vector<float> pooled((size_t)ci * och * ocw);
                std::vector<int32_t> arg((size_t)ci * och * ocw);
                for (int c = 0; c < ci; c++)
                    for (int py = 0; py < och; py++)
                        for (int px = 0; px < ocw; px++) {
                            float best = -1e30f;
                            int ba = 0;
                            for (int dy = 0; dy < 2; dy++)
                                for (int dx = 0; dx < 2; dx++) {
                                    const int sy = 2 * py + dy,
                                              sx = 2 * px + dx;
                                    if (sy >= ch || sx >= cw) continue;
                                    const float val =
                                            f1[((size_t)c * ch + sy) * cw + sx];
                                    if (val > best) {  // FIRST max wins
                                        best = val;
                                        ba = dy * 2 + dx;
                                    }
                                }
                            pooled[((size_t)c * och + py) * ocw + px] = best;
                            arg[((size_t)c * och + py) * ocw + px] = ba;
                        }
                f1.swap(pooled);
                parg[pool_i] = std::move(arg);
                // the target branch pools too (no argmax needed — frozen)
                {
                    std::vector<float> pooled0((size_t)ci * och * ocw);
                    for (int c = 0; c < ci; c++)
                        for (int py = 0; py < och; py++)
                            for (int px = 0; px < ocw; px++) {
                                float best = -1e30f;
                                for (int dy = 0; dy < 2; dy++)
                                    for (int dx = 0; dx < 2; dx++) {
                                        const int sy = 2 * py + dy,
                                                  sx = 2 * px + dx;
                                        if (sy >= ch || sx >= cw) continue;
                                        const float val =
                                                f0[((size_t)c * ch + sy) * cw +
                                                   sx];
                                        if (val > best) best = val;
                                    }
                                pooled0[((size_t)c * och + py) * ocw + px] =
                                        best;
                            }
                    f0.swap(pooled0);
                }
                pool_i++;
                ch = och;
                cw = ocw;
            }
            std::vector<float> o0((size_t)hw.cout[i] * lh[i] * lw[i]);
            a1[i].resize((size_t)hw.cout[i] * lh[i] * lw[i]);
            host_conv_relu(f0.data(), ci, ch, cw, hw.w[i].data(), hw.cout[i],
                           hw.b[i].data(), o0.data());
            host_conv_relu(f1.data(), ci, ch, cw, hw.w[i].data(), hw.cout[i],
                           hw.b[i].data(), a1[i].data());
            f0.swap(o0);
            f1.swap(a1[i]);
            std::vector<float>().swap(a1[i]);
            a1[i] = f1;  // copy of the relu output (backward + tap feature)
            a0[i] = f0;  // copy of the in0-side relu output (tap feature)
            ci = hw.cout[i];
            ch = lh[i];
            cw = lw[i];
        }
        // ---- taps: 1, 3, 6, 9, 12 — normalize, lin, spatial mean ----
        static const int kTapAfter[5] = {1, 3, 6, 9, 12};
        float value = 0.0f;
        std::vector<std::vector<float>> dn1_tap(5);  // dLPIPS/dn1 at each tap
        std::vector<std::vector<float>> n1_tap(5);   // normalized in1 features
        std::vector<std::vector<float>> N1_tap(5);   // per-pixel norm of in1
        int tap = 0;
        for (int i = 0; i < kConvs; i++) {
            if (i != kTapAfter[tap]) continue;
            const int C = kCh[i];
            const size_t plane = (size_t)C * lh[i] * lw[i];
            const float inv_hw = 1.0f / ((float)lh[i] * (float)lw[i]);
            n1_tap[tap].resize(plane);
            N1_tap[tap].resize((size_t)lh[i] * lw[i]);
            dn1_tap[tap].assign(plane, 0.0f);
            for (int y = 0; y < lh[i]; y++)
                for (int x = 0; x < lw[i]; x++) {
                    const size_t px = (size_t)y * lw[i] + x;
                    double n0 = 0.0, n1n = 0.0;
                    for (int c = 0; c < C; c++) {
                        n0 += (double)a0[i][(size_t)c * lh[i] * lw[i] + px] *
                              a0[i][(size_t)c * lh[i] * lw[i] + px];
                        n1n += (double)a1[i][(size_t)c * lh[i] * lw[i] + px] *
                               a1[i][(size_t)c * lh[i] * lw[i] + px];
                    }
                    const float N0 = (float)std::sqrt(n0);
                    const float N1 = (float)std::sqrt(n1n);
                    N1_tap[tap][px] = N1;
                    float acc = 0.0f;
                    for (int c = 0; c < C; c++) {
                        const size_t o = (size_t)c * lh[i] * lw[i] + px;
                        const float q0 = a0[i][o] / (N0 + kNromEps);
                        const float q1 = a1[i][o] / (N1 + kNromEps);
                        n1_tap[tap][o] = q1;
                        const float dd = q0 - q1;
                        acc += hw.lin[tap][c] * dd * dd;
                        // dLPIPS/dn1 = -2 (n0 - n1) lin_c / (H W)
                        dn1_tap[tap][o] = -2.0f * dd * hw.lin[tap][c] * inv_hw;
                    }
                    value += acc * inv_hw;
                }
            tap++;
        }
        (*out)[(size_t)v] = value;
        // ---- backward: layer-by-layer from relu5_3 down to the input ----
        // Every conv needs its own relu mask + transposed convolution (the
        // taps sit on convs 1,3,6,9,12 but the chain between two pools runs
        // over ALL layers: pool3 -> conv10 -> 11 -> 12 etc.). The tap VJPs
        // (normalize_tensor + lin + spatial mean) add on their layers.
        std::vector<std::vector<float>> df_tap(5);  // per-tap normalize VJP
        for (int t = 0; t < 5; t++) {
            const int L = kTapAfter[t];
            const int C = kCh[L];
            const size_t plane = (size_t)C * lh[L] * lw[L];
            df_tap[t].resize(plane);
            for (int y = 0; y < lh[L]; y++)
                for (int x = 0; x < lw[L]; x++) {
                    const size_t px = (size_t)y * lw[L] + x;
                    const float N1 = N1_tap[t][px];
                    const float Ne = N1 + kNromEps;
                    float dot = 0.0f;
                    for (int c = 0; c < C; c++)
                        dot += dn1_tap[t][(size_t)c * lh[L] * lw[L] + px] *
                               a1[L][(size_t)c * lh[L] * lw[L] + px];
                    const float beta = dot / (Ne * Ne * N1);
                    for (int c = 0; c < C; c++) {
                        const size_t o = (size_t)c * lh[L] * lw[L] + px;
                        df_tap[t][o] = dn1_tap[t][o] / Ne - a1[L][o] * beta;
                    }
                }
        }
        std::vector<float> d;  // gradient at the current relu-output level
        int t = 4;             // next tap whose VJP joins the chain
        for (int L = 12; L >= 0; --L) {
            const int C = kCh[L];
            const size_t plane = (size_t)C * lh[L] * lw[L];
            if (t >= 0 && L == kTapAfter[t]) {
                // join this tap's VJP (first tap initializes the chain)
                if (L == 12) {
                    d = df_tap[t];
                } else {
                    for (size_t i = 0; i < plane; i++) d[i] += df_tap[t][i];
                }
                t--;
            }
            // relu mask on this layer's output
            for (size_t i = 0; i < plane; i++)
                if (a1[L][i] <= 0.0f) d[i] = 0.0f;
            // transposed conv to this conv's input
            const int cin = L == 0 ? 3 : kCh[L - 1];
            std::vector<float> din((size_t)cin * lh[L] * lw[L], 0.0f);
            host_conv_input_grad(d.data(), C, lh[L], lw[L], hw.w[L].data(), cin,
                                 din.data());
            if (L == 0) {
                // through the scaling layer to [0,1] prediction space
                for (int c = 0; c < 3; c++)
                    for (int i = 0; i < H * W; i++)
                        (*grad_prediction)[(size_t)v * in_plane +
                                           (size_t)c * H * W + i] =
                                din[(size_t)c * H * W + i] / kScale[c];
                break;
            }
            // conv L's input is pool p's output when L in {10,7,4,2}: route
            // the gradient through the saved argmax; otherwise it is the
            // previous conv's output at the same resolution
            const int p = (L == 10  ? 3
                           : L == 7 ? 2
                           : L == 4 ? 1
                           : L == 2 ? 0
                                    : -1);
            if (p < 0) {
                d = std::move(din);
            } else {
                const int sch = pre_ci[p], sh_ = pre_ch[p], sw_ = pre_cw[p];
                std::vector<float> dpre((size_t)sch * sh_ * sw_, 0.0f);
                const int oh = lh[L], ow = lw[L];
                for (int c = 0; c < sch; c++)
                    for (int py = 0; py < oh; py++)
                        for (int px = 0; px < ow; px++) {
                            const int ba =
                                    parg[p][((size_t)c * oh + py) * ow + px];
                            const int sy = 2 * py + (ba >> 1),
                                      sx = 2 * px + (ba & 1);
                            if (sy < sh_ && sx < sw_)
                                dpre[((size_t)c * sh_ + sy) * sw_ + sx] =
                                        din[((size_t)c * oh + py) * ow + px];
                        }
                d = std::move(dpre);
            }
        }
    }
    return true;
}

bool LpipsVgg::score(const float* prediction,
                     const float* target,
                     int V,
                     int H,
                     int W,
                     std::vector<float>* out) {
    if (!impl_) return false;
    Impl& m = *impl_;
    if ((m.H != H || m.W != W || !m.graph) && !m.build(H, W)) return false;
    const size_t plane = (size_t)H * W, n = 3 * plane;
    std::vector<float> buf(2 * n);
    out->assign((size_t)V, 0.0f);
    for (int v = 0; v < V; v++) {
        // in0 = target, in1 = prediction; 2*x - 1 then (x - shift) / scale,
        // each an f32 op as in torch
        const float* src[2] = {target + (size_t)v * n,
                               prediction + (size_t)v * n};
        for (int s = 0; s < 2; s++)
            for (int c = 0; c < 3; c++)
                for (size_t i = 0; i < plane; i++) {
                    const float x = 2.0f * src[s][c * plane + i] - 1.0f;
                    buf[s * n + c * plane + i] = (x - kShift[c]) / kScale[c];
                }
        if (!ggml_gallocr_alloc_graph(m.galloc, m.graph)) {
            log_error("lpips: gallocr alloc failed");
            return false;
        }
        ggml_backend_tensor_set(m.in, buf.data(), 0,
                                buf.size() * sizeof(float));
        if (ggml_backend_graph_compute(m.be.handle, m.graph) !=
            GGML_STATUS_SUCCESS) {
            log_error("lpips: graph compute failed");
            return false;
        }
        ggml_backend_tensor_get(m.out, &(*out)[(size_t)v], 0, sizeof(float));
    }
    return true;
}

}  // namespace qs
