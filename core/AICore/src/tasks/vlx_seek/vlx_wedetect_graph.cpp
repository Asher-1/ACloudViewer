// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "vlx_wedetect_graph.h"
#include "vlx_gguf_loader.h"
#include "vlx_ggml_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>

namespace vlx {

namespace {

constexpr int kImg = 640;
constexpr int kStrides[3] = {8, 16, 32};
constexpr int kTopk = 30000;        // filter_scores_and_topk(scores, 0.0, 30000)
constexpr double kNmsIou = 0.7;     // torchvision.ops.batched_nms(..., 0.7)

}  // namespace

struct WeDetectGraph::Impl {
    ggml_backend* backend = nullptr;
    const GgufFile* gguf = nullptr;
    int reg_max = 16, num_prompts = 256, num_proposals = 100;
    int depths[4] = {3, 3, 27, 3};

    ggml_context* wctx = nullptr;
    ggml_backend_buffer* wbuf = nullptr;
    std::map<std::string, ggml_tensor*> w;

    ggml_context* gctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr* allocr = nullptr;
    ggml_context* ictx = nullptr;
    ggml_backend_buffer* ibuf = nullptr;
    ggml_tensor* in_img = nullptr;     // NCHW ne=(640, 640, 3)
    ggml_tensor* out_cls = nullptr;    // (256, 8400)
    ggml_tensor* out_reg = nullptr;    // (64, 8400)
    std::vector<ggml_tensor*> out_taps;
    bool taps_built = false;
    double graph_ms = 0.0;

    ggml_tensor* W(const std::string& n) const {
        auto it = w.find(n);
        // load_weights enumerated every GGUF tensor, so a miss is a converter
        // bug (invariant), not a runtime condition.
        if (it == w.end()) GGML_ABORT("wedetect: missing weight %s", n.c_str());
        return it->second;
    }
    bool has(const std::string& n) const { return w.count(n) != 0; }

    void free_graph() {
        if (allocr) ggml_gallocr_free(allocr);
        if (gctx) ggml_free(gctx);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (ictx) ggml_free(ictx);
        allocr = nullptr; gctx = nullptr; ibuf = nullptr; ictx = nullptr; graph = nullptr;
        out_taps.clear();
    }
    ~Impl() {
        free_graph();
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }

    bool load_weights(std::string* err);
    void build_graph(bool taps);
};

bool WeDetectGraph::Impl::load_weights(std::string* err) {
    reg_max = int(gguf->i32("wd.reg_max", reg_max));
    num_prompts = int(gguf->i32("wd.num_prompts", num_prompts));
    num_proposals = int(gguf->i32("wd.num_proposals", num_proposals));
    const auto& all = gguf->tensors();
    struct ggml_init_params ip = {ggml_tensor_overhead() * (all.size() + 8), nullptr, true};
    wctx = ggml_init(ip);
    std::vector<std::pair<ggml_tensor*, const TensorInfo*>> bind;
    std::map<ggml_tensor*, std::vector<uint8_t>> cwhn_bytes;
    for (const TensorInfo& ti : all) {
        const ggml_type t = to_ggml_type(ti.dtype);
        const int64_t ne[4] = {int64_t(ti.ne[0]), int64_t(ti.ne[1]), int64_t(ti.ne[2]),
                               int64_t(ti.ne[3])};
        ggml_tensor* x = ggml_new_tensor(wctx, t, int(ti.n_dims), ne);
        ggml_set_name(x, ti.name.c_str());
        // Depthwise kernels feed the CWHN conv_2d_dw path, whose CPU / CUDA /
        // Vulkan kernels all read k[(ky*KW + kx)*C + c] (channel fastest) and
        // the CPU op asserts the matching strides. Re-pack once on the host
        // ([c][ky][kx] -> [ky][kx][c]) and describe it with CWHN strides; the
        // logical shape ne=(KW, KH, 1, C) is unchanged. O(49*C) once.
        const bool dw = ti.name.size() > 5 && ti.name.compare(ti.name.size() - 5, 5, ".dw.w") == 0;
        if (dw) {
            GGML_ASSERT(t == GGML_TYPE_F32 && ne[2] == 1);
            const int64_t KW = ne[0], KH = ne[1], C = ne[3];
            const float* src = reinterpret_cast<const float*>(ti.bytes.data());
            std::vector<uint8_t> buf(ti.bytes.size());
            float* dst = reinterpret_cast<float*>(buf.data());
            for (int64_t c = 0; c < C; ++c)
                for (int64_t k = 0; k < KW * KH; ++k) dst[k * C + c] = src[c * KW * KH + k];
            x->nb[0] = size_t(C) * sizeof(float);
            x->nb[1] = size_t(KW) * x->nb[0];
            x->nb[2] = sizeof(float);
            x->nb[3] = sizeof(float);
            cwhn_bytes[x] = std::move(buf);
        }
        w[ti.name] = x;
        bind.emplace_back(x, &ti);
    }
    // stage depths from the tensor names (the gguf array KV is not parsed)
    for (int s = 0; s < 4; ++s) {
        int n = 0;
        while (has("bb.s" + std::to_string(s) + "." + std::to_string(n) + ".dw.w")) ++n;
        depths[s] = n;
    }
    wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    for (auto& [x, ti] : bind) {
        auto it = cwhn_bytes.find(x);
        const std::vector<uint8_t>& b = it == cwhn_bytes.end() ? ti->bytes : it->second;
        ggml_backend_tensor_set(x, b.data(), 0, b.size());
    }
    return true;
}

void WeDetectGraph::Impl::build_graph(bool taps) {
    free_graph();
    // ~3k nodes (36 ConvNeXt blocks, 4 BepC3 x 6 BottleRep, 3-level head)
    const size_t n_nodes = 8192;
    struct ggml_init_params ip = {ggml_tensor_overhead() * n_nodes +
                                      ggml_graph_overhead_custom(n_nodes, false),
                                  nullptr, true};
    gctx = ggml_init(ip);
    ggml_context* ctx = gctx;

    struct ggml_init_params iip = {ggml_tensor_overhead() * 16, nullptr, true};
    ictx = ggml_init(iip);
    in_img = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, kImg, kImg, 3);
    int n_anchor = 0;
    for (int s : kStrides) n_anchor += (kImg / s) * (kImg / s);
    out_cls = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, num_prompts, n_anchor);
    out_reg = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, 4 * reg_max, n_anchor);
    if (taps) {
        const int tap_c[7] = {128, 256, 512, 1024, 128, 256, 512};
        const int tap_s[7] = {4, 8, 16, 32, 8, 16, 32};
        for (int i = 0; i < 7; ++i)
            out_taps.push_back(ggml_new_tensor_3d(ictx, GGML_TYPE_F32, tap_c[i],
                                                  kImg / tap_s[i], kImg / tap_s[i]));
    }
    ibuf = ggml_backend_alloc_ctx_tensors(ictx, backend);

    // ---- op vocabulary (token-major (C, W, H) in and out) ----
    auto lnorm = [&](ggml_tensor* x, const std::string& p) {
        ggml_tensor* r = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
        r = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, r, 1e-6f), W(p + ".w")), W(p + ".b"));
        return ggml_reshape_3d(ctx, r, x->ne[0], x->ne[1], x->ne[2]);
    };
    auto linear = [&](ggml_tensor* x, ggml_tensor* wt, ggml_tensor* b) {
        ggml_tensor* r = mul_mat_w(ctx, wt, ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]));
        if (b) r = ggml_add(ctx, r, b);
        return ggml_reshape_3d(ctx, r, wt->ne[1], x->ne[1], x->ne[2]);
    };
    // kxk conv via im2col. The kernel operand of ggml_im2col only supplies its
    // shape (KW, KH, IC), so a tiny F32 shape leaf stands in for it — the real
    // weight (IC*k*k, OC) may be q8_0, whose rows cannot be reshaped to k.
    auto conv = [&](ggml_tensor* x, const std::string& p, int k, int s, int pad) {
        ggml_tensor* wt = W(p + ".w");
        ggml_tensor* b = has(p + ".b") ? W(p + ".b") : nullptr;
        if (k == 1 && s == 1) return linear(x, wt, b);
        const int64_t IC = x->ne[0];
        GGML_ASSERT(wt->ne[0] == IC * k * k);
        ggml_tensor* shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, k, IC, 1);
        ggml_tensor* xn = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));   // (W, H, C)
        ggml_tensor* cols = ggml_im2col(ctx, shape, xn, s, s, pad, pad, 1, 1, true, GGML_TYPE_F32);
        ggml_tensor* r = mul_mat_w(ctx, wt, cols);    // (OC, OW, OH): token-major
        return b ? ggml_add(ctx, r, b) : r;
    };
    auto relu = [&](ggml_tensor* x) { return ggml_relu(ctx, x); };
    auto silu = [&](ggml_tensor* x) { return ggml_silu(ctx, x); };
    auto cat0 = [&](ggml_tensor* a, ggml_tensor* b) { return ggml_concat(ctx, a, b, 0); };
    // ConvTranspose k2s2 (merged 4-slice GEMM + pixel shuffle; derivation in
    // simplefpn_graph.cpp — identical weight layout from the converter).
    auto convt = [&](ggml_tensor* x, const std::string& p) {
        ggml_tensor* w_all = W(p + ".wt");
        const int64_t C_in = x->ne[0], Wd = x->ne[1], H = x->ne[2];
        const int64_t C_out = w_all->ne[1] / 4;
        ggml_tensor* r = mul_mat_w(ctx, w_all, ggml_reshape_2d(ctx, x, C_in, Wd * H));
        auto slice = [&](int kk) {
            return ggml_view_3d(ctx, r, C_out, Wd, H, 4 * C_out * sizeof(float),
                                4 * C_out * Wd * sizeof(float), kk * C_out * sizeof(float));
        };
        ggml_tensor* g0 = ggml_reshape_3d(ctx, cat0(slice(0), slice(1)), C_out, 2 * Wd, H);
        ggml_tensor* g1 = ggml_reshape_3d(ctx, cat0(slice(2), slice(3)), C_out, 2 * Wd, H);
        ggml_tensor* c = cat0(g0, g1);
        ggml_tensor* v4 = ggml_view_4d(ctx, c, C_out, 2 * Wd, 2, H, c->nb[1],
                                       C_out * sizeof(float), c->nb[2], 0);
        ggml_tensor* sh = ggml_reshape_3d(ctx, ggml_cont(ctx, v4), C_out, 2 * Wd, 2 * H);
        return ggml_add(ctx, sh, W(p + ".b"));
    };
    auto tap = [&](ggml_tensor* x, int i) -> ggml_tensor* {
        return taps ? ggml_cpy(ctx, x, out_taps[size_t(i)]) : nullptr;
    };
    std::vector<ggml_tensor*> roots;

    // ---------------- backbone: ConvNeXt-B ----------------
    ggml_tensor* x;
    {
        ggml_tensor* shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 3, 1);
        ggml_tensor* cols = ggml_im2col(ctx, shape, in_img, 4, 4, 0, 0, 1, 1, true, GGML_TYPE_F32);
        x = ggml_add(ctx, mul_mat_w(ctx, W("bb.stem.w"), cols), W("bb.stem.b"));
        x = lnorm(x, "bb.stem.ln");   // channels_first LN, eps 1e-6
    }
    ggml_tensor* feats[4];
    for (int s = 0; s < 4; ++s) {
        if (s > 0) {
            const std::string p = "bb.ds" + std::to_string(s);
            x = lnorm(x, p + ".ln");
            x = conv(x, p, 2, 2, 0);
        }
        for (int bi = 0; bi < depths[s]; ++bi) {
            const std::string p = "bb.s" + std::to_string(s) + "." + std::to_string(bi);
            ggml_tensor* in = x;
            // depthwise 7x7 on the channels-contiguous permuted view: ggml
            // keeps the CWHN strides, so permuting back is a free view.
            ggml_tensor* d = ggml_conv_2d_dw_direct(ctx, W(p + ".dw.w"),
                                                    ggml_permute(ctx, x, 2, 0, 1, 3),
                                                    1, 1, 3, 3, 1, 1);
            d = ggml_permute(ctx, d, 1, 2, 0, 3);           // -> (C, W, H) contiguous
            d = ggml_add(ctx, d, W(p + ".dw.b"));
            d = lnorm(d, p + ".ln");
            d = ggml_gelu_erf(ctx, linear(d, W(p + ".pw1.w"), W(p + ".pw1.b")));
            d = linear(d, W(p + ".pw2.w"), W(p + ".pw2.b"));   // gamma folded
            x = ggml_add(ctx, in, d);
        }
        feats[s] = x;
        if (ggml_tensor* t = tap(x, s)) roots.push_back(t);
    }

    // ---------------- neck: CSPRepBiFPANNeck ----------------
    auto cbr = [&](ggml_tensor* v, const std::string& p, int k, int s) {
        return relu(conv(v, "neck." + p, k, s, k / 2));
    };
    auto cbs = [&](ggml_tensor* v, const std::string& p, int k, int s) {
        return silu(conv(v, "neck." + p, k, s, k / 2));
    };
    auto bottlerep = [&](ggml_tensor* v, const std::string& p) {
        ggml_tensor* o = cbs(cbs(v, p + ".conv1", 3, 1), p + ".conv2", 3, 1);
        const float alpha = gguf->tensor_f32("neck." + p + ".alpha")[0];
        return ggml_add(ctx, o, ggml_scale(ctx, v, alpha));   // in == out: shortcut
    };
    auto bepc3 = [&](ggml_tensor* v, const std::string& p) {
        ggml_tensor* m = bottlerep(cbs(v, p + ".cv1", 1, 1), p + ".m.conv1");
        for (int i = 0; has("neck." + p + ".m.block." + std::to_string(i) + ".conv1.w"); ++i)
            m = bottlerep(m, p + ".m.block." + std::to_string(i));
        return cbs(cat0(m, cbs(v, p + ".cv2", 1, 1)), p + ".cv3", 1, 1);
    };
    auto bifusion = [&](ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, const std::string& p) {
        ggml_tensor* x0 = convt(a, "neck." + p + ".upsample");
        ggml_tensor* x1 = cbr(b, p + ".cv1", 1, 1);
        ggml_tensor* x2 = cbr(cbr(c, p + ".cv2", 1, 1), p + ".downsample", 3, 2);
        return cbr(cat0(cat0(x0, x1), x2), p + ".cv3", 1, 1);
    };
    // (x3, x2, x1, x0) = (c1, c2, c3, c4)
    ggml_tensor* fpn_out0 = cbr(feats[3], "reduce_layer0", 1, 1);
    ggml_tensor* f_out0 = bepc3(bifusion(fpn_out0, feats[2], feats[1], "Bifusion0"), "Rep_p4");
    ggml_tensor* fpn_out1 = cbr(f_out0, "reduce_layer1", 1, 1);
    ggml_tensor* pan_out2 = bepc3(bifusion(fpn_out1, feats[1], feats[0], "Bifusion1"), "Rep_p3");
    ggml_tensor* pan_out1 = bepc3(cat0(cbr(pan_out2, "downsample2", 3, 2), fpn_out1), "Rep_n3");
    ggml_tensor* pan_out0 = bepc3(cat0(cbr(pan_out1, "downsample1", 3, 2), fpn_out0), "Rep_n4");
    ggml_tensor* neck[3] = {pan_out2, pan_out1, pan_out0};
    for (int i = 0; i < 3; ++i)
        if (ggml_tensor* t = tap(neck[i], 4 + i)) roots.push_back(t);

    // ---------------- head (BN / logit_scale folded) ----------------
    ggml_tensor* cls_all = nullptr;
    ggml_tensor* reg_all = nullptr;
    for (int li = 0; li < 3; ++li) {
        const std::string p = "head." + std::to_string(li);
        auto branch = [&](const std::string& kind) {
            ggml_tensor* v = silu(conv(neck[li], p + "." + kind + ".0", 3, 1, 1));
            v = silu(conv(v, p + "." + kind + ".1", 3, 1, 1));
            return conv(v, p + "." + kind + ".2", 1, 1, 0);
        };
        ggml_tensor* reg = branch("reg");                       // (64, W, H)
        ggml_tensor* ce = branch("cls");                        // (768, W, H), BN folded
        ggml_tensor* cl = linear(ce, W(p + ".emb.w"), W(p + ".emb.b"));   // (256, W, H)
        const int64_t hw = reg->ne[1] * reg->ne[2];
        reg = ggml_reshape_2d(ctx, reg, reg->ne[0], hw);         // anchor = h*W + w
        cl = ggml_reshape_2d(ctx, cl, cl->ne[0], hw);
        reg_all = reg_all ? ggml_concat(ctx, reg_all, reg, 1) : reg;
        cls_all = cls_all ? ggml_concat(ctx, cls_all, cl, 1) : cl;
    }
    roots.push_back(ggml_cpy(ctx, cls_all, out_cls));
    roots.push_back(ggml_cpy(ctx, reg_all, out_reg));

    graph = ggml_new_graph_custom(ctx, n_nodes, false);
    for (ggml_tensor* r : roots) ggml_build_forward_expand(graph, r);
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);
    taps_built = taps;
}

WeDetectGraph::WeDetectGraph(ggml_backend* backend, const GgufFile* gguf) : impl_(new Impl) {
    impl_->backend = backend;
    impl_->gguf = gguf;
}

WeDetectGraph::~WeDetectGraph() { delete impl_; }

double WeDetectGraph::last_graph_ms() const { return impl_->graph_ms; }

namespace {

// torchvision nms CPU kernel (nms_kernel_impl) on one class: candidates are
// already score-descending (stable), IoU in float, suppress iff ovr > thr.
void nms_class(const std::vector<int>& idx, const std::vector<std::array<float, 4>>& bx,
               std::vector<char>* keep) {
    const size_t n = idx.size();
    std::vector<float> area(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& b = bx[size_t(idx[i])];
        area[i] = (b[2] - b[0]) * (b[3] - b[1]);
    }
    std::vector<char> sup(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (sup[i]) continue;
        (*keep)[size_t(idx[i])] = 1;
        const auto& a = bx[size_t(idx[i])];
        for (size_t j = i + 1; j < n; ++j) {   // O(n_c^2) per class (n_c ~ 1e2)
            if (sup[j]) continue;
            const auto& b = bx[size_t(idx[j])];
            const float xx1 = std::max(a[0], b[0]), yy1 = std::max(a[1], b[1]);
            const float xx2 = std::min(a[2], b[2]), yy2 = std::min(a[3], b[3]);
            const float ww = std::max(0.f, xx2 - xx1), hh = std::max(0.f, yy2 - yy1);
            const float inter = ww * hh;
            const float ovr = inter / (area[i] + area[j] - inter);
            if (double(ovr) > kNmsIou) sup[j] = 1;
        }
    }
}

}  // namespace

bool WeDetectGraph::run(const ImageU8& img, WeDetectResult* out, bool want_taps,
                        std::string* err) {
    Impl* m = impl_;
    if (!m->wctx && !m->load_weights(err)) return false;
    if (!m->graph || m->taps_built != want_taps) m->build_graph(want_taps);

    // ---- letterbox (PIL BILINEAR, 114 pad) + /255 -> planar NCHW ----
    const Letterbox lb = letterbox(img, kImg);
    {
        std::vector<float> x(size_t(3) * kImg * kImg);
        const size_t hw = size_t(kImg) * kImg;
        for (size_t i = 0; i < hw; ++i)
            for (int c = 0; c < 3; ++c) x[size_t(c) * hw + i] = float(lb.canvas.rgb[i * 3 + c]) / 255.f;
        ggml_backend_tensor_set(m->in_img, x.data(), 0, ggml_nbytes(m->in_img));
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(m->backend, m->graph) != GGML_STATUS_SUCCESS) {
        if (err) *err = "wedetect graph compute failed";
        return false;
    }
    m->graph_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const int K = m->num_prompts, R = m->reg_max;
    const int A = int(m->out_cls->ne[1]);
    std::vector<float> cls(size_t(A) * K), reg(size_t(A) * 4 * R);
    ggml_backend_tensor_get(m->out_cls, cls.data(), 0, ggml_nbytes(m->out_cls));
    ggml_backend_tensor_get(m->out_reg, reg.data(), 0, ggml_nbytes(m->out_reg));
    if (want_taps) {
        out->taps.clear();
        for (ggml_tensor* t : m->out_taps) {
            std::vector<float> v(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
            out->taps.push_back(std::move(v));
        }
    }

    // ---- head_predict: DFL decode (softmax over reg_max . arange) * stride ----
    std::vector<std::array<float, 4>> bx(static_cast<size_t>(A));
    std::vector<float> dist(size_t(A) * 4);
    {
        int a = 0;
        for (int s : kStrides) {
            const int g = kImg / s;
            for (int yy = 0; yy < g; ++yy)
                for (int xx = 0; xx < g; ++xx, ++a) {
                    const float* r = &reg[size_t(a) * 4 * R];   // channel = side*R + bin
                    float d[4];
                    for (int side = 0; side < 4; ++side) {
                        const float* q = r + side * R;
                        float mx = q[0];
                        for (int b = 1; b < R; ++b) mx = std::max(mx, q[b]);
                        float e[64], sum = 0.f;
                        for (int b = 0; b < R; ++b) { e[b] = std::exp(q[b] - mx); sum += e[b]; }
                        float acc = 0.f;
                        for (int b = 0; b < R; ++b) acc += (e[b] / sum) * float(b);
                        d[side] = acc * float(s);
                        dist[size_t(a) * 4 + side] = d[side];
                    }
                    const float px = (float(xx) + 0.5f) * float(s);
                    const float py = (float(yy) + 0.5f) * float(s);
                    bx[size_t(a)] = {px - d[0], py - d[1], px + d[2], py + d[3]};
                }
        }
    }

    // ---- filter_scores_and_topk(sigmoid, 0.0, 30000): flat idx = a*K + k ----
    std::vector<float> sc(cls.size());
    for (size_t i = 0; i < cls.size(); ++i) sc[i] = 1.f / (1.f + std::exp(-cls[i]));
    std::vector<uint32_t> order;
    order.reserve(sc.size());
    for (uint32_t i = 0; i < uint32_t(sc.size()); ++i)
        if (sc[i] > 0.f) order.push_back(i);
    auto by_score = [&](uint32_t a, uint32_t b) { return sc[a] > sc[b] || (sc[a] == sc[b] && a < b); };
    const size_t nk = std::min<size_t>(kTopk, order.size());
    // O(N + k log k): selection then sort of the top-k only (N = 2.15M)
    std::nth_element(order.begin(), order.begin() + nk, order.end(), by_score);
    order.resize(nk);
    std::sort(order.begin(), order.end(), by_score);

    // ---- batched_nms (vanilla per-class path: numel > 4000) ----
    std::vector<std::array<float, 4>> cb(nk);
    std::vector<std::vector<int>> per_class(static_cast<size_t>(K));
    for (size_t i = 0; i < nk; ++i) {
        cb[i] = bx[order[i] / uint32_t(K)];
        per_class[order[i] % uint32_t(K)].push_back(int(i));
    }
    std::vector<char> keep(nk, 0);
    for (int k = 0; k < K; ++k)
        if (!per_class[size_t(k)].empty()) nms_class(per_class[size_t(k)], cb, &keep);
    std::vector<int> kept;
    for (size_t i = 0; i < nk; ++i)
        if (keep[i]) kept.push_back(int(i));
    // keep order = score descending (candidates were sorted); [:num_proposals]
    if (int(kept.size()) > m->num_proposals) kept.resize(size_t(m->num_proposals));

    // ---- forward(): undo letterbox, clamp, drop degenerate ----
    out->boxes.clear();
    out->scores.clear();
    const float ratio = float(lb.ratio);
    for (int i : kept) {
        std::array<float, 4> b = cb[size_t(i)];
        b[0] -= lb.off_x; b[1] -= lb.off_y; b[2] -= lb.off_x; b[3] -= lb.off_y;
        for (float& v : b) v /= ratio;
        b[0] = std::min(std::max(b[0], 0.f), float(img.width));
        b[2] = std::min(std::max(b[2], 0.f), float(img.width));
        b[1] = std::min(std::max(b[1], 0.f), float(img.height));
        b[3] = std::min(std::max(b[3], 0.f), float(img.height));
        if (!(b[2] > b[0] && b[3] > b[1])) continue;
        out->boxes.push_back({b[0], b[1], b[2], b[3]});
        out->scores.push_back(sc[order[size_t(i)]]);
    }
    if (want_taps) {
        out->cls_logits = std::move(cls);
        out->bbox_preds = std::move(dist);
    }
    return true;
}

}  // namespace vlx
