// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "querysplat_graph.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "map_anything_common.hpp"
#include "map_anything_gguf_loader.hpp"
#include "querysplat_activations.hpp"

namespace qs {

using map_anything::log_error;
using map_anything::log_info;
using map_anything::log_warn;

namespace {

constexpr float kEpsLN = 1e-5f;  // torch nn.LayerNorm default

// Exact (non-flash) attention over contiguous {dh, T, nh, B} q/k/v: identical
// math to ggml_flash_attn_ext but F32 softmax/matmul throughout. Needed on
// CUDA, whose flash kernels convert F32 K/V to F16; CPU/Vulkan flash is
// F32-exact. Returns head-major {dim, T, B} — drop-in for fa + reshape.
ggml_tensor* exact_attention_core(ggml_context* ctx,
                                  ggml_tensor* q,
                                  ggml_tensor* k,
                                  ggml_tensor* v,
                                  float scale) {
    const int64_t dh = q->ne[0], T = q->ne[1], nh = q->ne[2], B = q->ne[3];
    ggml_tensor* qf = ggml_reshape_3d(ctx, q, dh, T, nh * B);
    ggml_tensor* kf =
            ggml_reshape_3d(ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);
    ggml_tensor* vf =
            ggml_reshape_3d(ctx, v, v->ne[0], v->ne[1], v->ne[2] * v->ne[3]);
    ggml_tensor* scores = ggml_mul_mat(ctx, kf, qf);  // {Tq, Tk, nh*B}
    scores = ggml_scale(ctx, scores, scale);
    scores = ggml_soft_max(ctx, scores);  // normalize along keys
    ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vf));  // {Tk, dh, m}
    ggml_tensor* o = ggml_mul_mat(ctx, vt, scores);  // {dh, Tq, nh*B}
    o = ggml_reshape_4d(ctx, o, dh, T, nh, B);
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));  // {dh, nh, T, B}
    return ggml_reshape_3d(ctx, o, dh * nh, T, B);
}

// Read a staged HostTensor (f32 or f16) into an f32 vector (host constant
// folding needs plain floats; the converter keeps these small parameters
// f32 in every tier, so quantized types are rejected loudly).
std::vector<float> host_f32(const map_anything::HostTensor& ht) {
    const int64_t n = ht.ne[0] * ht.ne[1] * ht.ne[2] * ht.ne[3];
    std::vector<float> out((size_t)n);
    if (ht.type == GGML_TYPE_F32) {
        std::memcpy(out.data(), ht.data.data(), (size_t)n * sizeof(float));
    } else if (ht.type == GGML_TYPE_F16) {
        const ggml_fp16_t* src =
                reinterpret_cast<const ggml_fp16_t*>(ht.data.data());
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = ggml_fp16_to_fp32(src[i]);
    } else {
        log_error("host_f32: unsupported type %d for %s", (int)ht.type,
                  ht.name.c_str());
    }
    return out;
}

inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x / 1.4142135623730951f));
}

// Constant folding of layer_weights = softmax(MLP(eye_token)) — the exact
// torch op sequence in VGGTEncoder._tokens_from_layers (f32).
void fold_layer_weights(const map_anything::HostTensor& eye,
                        const map_anything::HostTensor& ln_w,
                        const map_anything::HostTensor& ln_b,
                        const map_anything::HostTensor& fc1_w,
                        const map_anything::HostTensor& fc1_b,
                        const map_anything::HostTensor& fc2_w,
                        const map_anything::HostTensor& fc2_b,
                        float out[4]) {
    const std::vector<float> e = host_f32(eye);    // {C,1,1,1}
    const std::vector<float> w0 = host_f32(ln_w);  // {C}
    const std::vector<float> b0 = host_f32(ln_b);
    const int C = (int)eye.ne[0];
    const int H = (int)fc1_w.ne[1];            // ggml ne {C, H}
    const int L = (int)fc2_w.ne[1];            // ggml ne {H, L}
    std::vector<float> f1w = host_f32(fc1_w);  // row-major {C, H} rows=C
    std::vector<float> f1b = host_f32(fc1_b);
    std::vector<float> f2w = host_f32(fc2_w);  // {H, L}
    std::vector<float> f2b = host_f32(fc2_b);
    // LayerNorm(eps 1e-5) over C
    double mean = 0.0;
    for (int i = 0; i < C; i++) mean += e[(size_t)i];
    mean /= C;
    double var = 0.0;
    for (int i = 0; i < C; i++)
        var += (e[(size_t)i] - mean) * (e[(size_t)i] - mean);
    var /= C;
    const float inv = (float)(1.0 / std::sqrt(var + kEpsLN));
    std::vector<float> h((size_t)C);
    for (int i = 0; i < C; i++)
        h[(size_t)i] = (e[(size_t)i] - (float)mean) * inv * w0[(size_t)i] +
                       b0[(size_t)i];
    // Linear 1 {C -> H}, GELU(erf), Linear 2 {H -> L}, softmax
    // torch (out,in) C-order flat == out*in_features + in == ggml
    // ne{in,out} flat (ne0 fastest) — index [j*in + i] below.
    std::vector<float> hid((size_t)H, 0.0f);
    for (int j = 0; j < H; j++) {
        double acc = 0.0;
        for (int i = 0; i < C; i++)
            acc += (double)f1w[(size_t)j * C + i] * h[(size_t)i];
        hid[(size_t)j] = gelu_erf((float)acc + f1b[(size_t)j]);
    }
    double logits[4];
    double mx = -1e30;
    for (int l = 0; l < L; l++) {
        double acc = 0.0;
        for (int j = 0; j < H; j++)
            acc += (double)f2w[(size_t)l * H + j] * hid[(size_t)j];
        logits[l] = acc + f2b[(size_t)l];
        mx = std::max(mx, logits[l]);
    }
    double sum = 0.0;
    for (int l = 0; l < L; l++) {
        logits[l] = std::exp(logits[l] - mx);
        sum += logits[l];
    }
    for (int l = 0; l < L; l++) out[l] = (float)(logits[l] / sum);
}

}  // namespace

struct QuerySplatDecoder::Impl {
    map_anything::Backend be;
    std::unique_ptr<map_anything::GGUFModel> model;
    DecoderConfig* cfg = nullptr;

    // Exact-F32 attention only where flash is not F32-exact (CUDA converts
    // F32 K/V to F16 inside the flash kernels).
    bool f32_accuracy = false;
    bool f32_attn() const {
        return f32_accuracy && be.name.rfind("CUDA", 0) == 0;
    }

    // staged weights
    ggml_context* wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor*> wt;

    // graph
    ggml_context* gctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph* graph = nullptr;
    // inputs
    ggml_tensor* in_layers[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* in_rgb = nullptr;      // {W, H, 3, S} normalized
    ggml_tensor* in_plucker = nullptr;  // {W, H, 6, S}
    // outputs
    ggml_tensor* out_deconv = nullptr;  // {640, Ngs, 1}
    ggml_tensor* out_color = nullptr;   // {post_out*64, Ngs, 1}
    // dump seams (set_output only when dumping)
    ggml_tensor* out_fused = nullptr;  // post-LN fused tokens {C, Nv, 1}
    ggml_tensor* out_fused_patches =
            nullptr;  // pre-concat patch fusion {C, P*S, 1}
    ggml_tensor* out_rgb_tokens =
            nullptr;                     // pre original_encoder_norm {C, Nr, 1}
    ggml_tensor* out_kvproj = nullptr;   // pre-split geometry kv {2C, Nv, 1}
    ggml_tensor* out_pe_conv = nullptr;  // patch_embed conv pre-LN {C, P, S}
    ggml_tensor* out_ppe_conv = nullptr;  // plucker conv pre-LN {C, P, S}
    ggml_tensor* out_kv_k = nullptr;
    ggml_tensor* out_kv_v = nullptr;
    ggml_tensor* out_post_k = nullptr;
    ggml_tensor* out_post_v = nullptr;
    ggml_tensor* out_geom = nullptr;

    // shapes
    int S = 0, H = 0, W = 0, Ncr = 0;
    int C = 0, nh = 0, dh = 0;
    int Ngs = 0, gpt = 0, P = 0, Nagg = 0, Nv = 0, Nr = 0, post_out = 0;
    bool built = false;
    std::string dump_dir;
    // TTO session state (t4). The session owns a separate graph pair with
    // the four KV memories as trainable inputs; see docs/tto_design.md.
    bool seams_kv_requested = false;  // run() also fetches the 4 KV seams
    bool tto_active = false;
    int tto_S = 0, tto_H = 0, tto_W = 0, tto_Ncr = 0;
    ggml_context* tctx = nullptr;
    ggml_context* tbuf_ctx = nullptr;
    ggml_backend_buffer_t tbuf = nullptr;
    ggml_gallocr_t tgalloc = nullptr;
    ggml_cgraph* tf_graph = nullptr;  // forward: mem -> raw heads
    ggml_cgraph* tb_graph = nullptr;  // backward: mem,d_raw -> mem update
    ggml_tensor* mem[4] = {nullptr, nullptr, nullptr, nullptr};  // K,V,pK,pV
    ggml_tensor* d_deconv_in = nullptr;
    ggml_tensor* d_color_in = nullptr;
    ggml_tensor* adamw_params = nullptr;
    float lr_cur = 5e-3f;
    int tto_T = 0, tto_Nv = 0;
    // TTO graph F/B tensors
    ggml_tensor* tf_out_deconv = nullptr;
    ggml_tensor* tf_out_color = nullptr;
    ggml_tensor* probe_dmem[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_gallocr_t tf_galloc = nullptr;
    ggml_gallocr_t tb_galloc = nullptr;
    ggml_tensor* adam_m[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* adam_v[4] = {nullptr, nullptr, nullptr, nullptr};
    ggml_tensor* tb_out[4] = {nullptr, nullptr, nullptr, nullptr};

    ggml_tensor* w(const char* name) const {
        auto it = wt.find(name);
        if (it == wt.end()) {
            log_error("missing weight tensor: %s", name);
            return nullptr;
        }
        return it->second;
    }

    // Same contract as map-anything-ggml's lin(): 2-D mul_mat, bias broadcast
    // {Cout,1,1,1}. Under f32_accuracy a quantized weight is dequantized to
    // F32 per call on CPU/CUDA: their quantized GEMMs also quantize the
    // ACTIVATION (CPU vec_dot -> q8_0, CUDA MMQ -> q8_1), which measured
    // ~1 dB render loss vs GT on the q8_0 tier, while the weight rounding
    // alone (Vulkan, integer-dot off) costs nothing measurable (q8_0 is the
    // only quantized decoder tier). The cast is a
    // graph temp (largest: fc1 2048x8192 f32 = 64 MB, reused by gallocr), so
    // the resident weights stay q8_0.
    ggml_tensor* lin(ggml_context* ctx,
                     ggml_tensor* x,
                     ggml_tensor* w_,
                     ggml_tensor* b) const {
        ggml_tensor* w32 = w_;
        if (w_->type == GGML_TYPE_F16 && be.name == "CPU") {
            w32 = ggml_cast(ctx, w_, GGML_TYPE_F32);
        } else if (f32_accuracy && w_->type == GGML_TYPE_Q8_0 &&
                   (be.name == "CPU" || be.name.rfind("CUDA", 0) == 0)) {
            w32 = ggml_cast(ctx, w_, GGML_TYPE_F32);
        }
        const int64_t kn = x->ne[1] * x->ne[2] * x->ne[3];
        ggml_tensor* x2 = ggml_reshape_2d(ctx, x, x->ne[0], kn);
        if (w32->ne[0] != x2->ne[0] || w32->ne[2] != x2->ne[2] ||
            w32->ne[3] != x2->ne[3]) {
            log_error(
                    "lin shape mismatch: w=%s {w0=%d,w1=%d} x=%s "
                    "{x0=%d,x1=%d,x2=%d,x3=%d}",
                    ggml_get_name(w_), (int)w32->ne[0], (int)w32->ne[1],
                    ggml_get_name(x) ? ggml_get_name(x) : "(anon)",
                    (int)x->ne[0], (int)x->ne[1], (int)x->ne[2], (int)x->ne[3]);
        }
        ggml_tensor* r = ggml_mul_mat(ctx, w32, x2);  // {Cout, kn}
        ggml_tensor* out = ggml_reshape_4d(ctx, r, w32->ne[1], x->ne[1],
                                           x->ne[2], x->ne[3]);
        if (b) {
            out = ggml_add(ctx, out,
                           ggml_reshape_4d(ctx, b, w32->ne[1], 1, 1, 1));
        }
        return out;
    }

    ggml_tensor* ln(ggml_context* ctx,
                    ggml_tensor* x,
                    ggml_tensor* nw,
                    ggml_tensor* nb,
                    float eps = kEpsLN) const {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), nw), nb);
    }

    // q {dh,nh,T,B} -> {dh,T,nh,B} contiguous for flash_attn_ext
    ggml_tensor* fa_permute(ggml_context* ctx, ggml_tensor* x) const {
        return ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    }

    // strided view over a packed qkv linear output {3C, T, 1, 1}: rows
    // [row0, row0+C) reinterpreted as {dh, nh, T, 1}. Element (c, h, t) sits
    // at t*rows + h*dh + c, so nb1 (h) = dh*4 and nb2 (t) = the t-stride of
    // the source (nb[1]).
    ggml_tensor* head_view(ggml_context* ctx,
                           ggml_tensor* lin_out,
                           int row0,
                           int T) const {
        return ggml_view_4d(ctx, lin_out, dh, nh, T, 1, dh * sizeof(float),
                            lin_out->nb[1], lin_out->nb[2],
                            (size_t)row0 * sizeof(float));
    }

    // Pre-norm DecoderBlock (enc_dec.py): cross -> self -> mlp, each with a
    // LayerScale gamma. K/V for cross-attn come pre-permuted from the KV
    // builders. tok: {C, T, 1}. Keys are the short GGUF names from
    // convert_querysplat_to_gguf.py rename_key().
    ggml_tensor* decoder_block(ggml_context* ctx,
                               ggml_tensor* tok,
                               const std::string& p,
                               ggml_tensor* Kc,
                               ggml_tensor* Vc,
                               int T) const {
        // ---- cross attention (q_norm only; keys/values pre-normalized) ----
        ggml_tensor* h = ln(ctx, tok, w((p + ".cross_norm.weight").c_str()),
                            w((p + ".cross_norm.bias").c_str()));
        ggml_tensor* q = lin(ctx, h, w((p + ".cross_qp.weight").c_str()),
                             w((p + ".cross_qp.bias").c_str()));
        q = ln(ctx, ggml_reshape_4d(ctx, q, dh, nh, T, 1),
               w((p + ".cross_qn.weight").c_str()),
               w((p + ".cross_qn.bias").c_str()));
        q = fa_permute(ctx, q);
        ggml_tensor* o =
                f32_attn() ? exact_attention_core(ctx, q, Kc, Vc,
                                                  1.0f / std::sqrt((float)dh))
                           : ggml_flash_attn_ext(ctx, q, Kc, Vc, nullptr,
                                                 1.0f / std::sqrt((float)dh),
                                                 0.0f, 0.0f);
        o = ggml_reshape_3d(ctx, o, C, T, 1);  // head-major token vectors
        o = lin(ctx, o, w((p + ".cross_op.weight").c_str()),
                w((p + ".cross_op.bias").c_str()));
        tok = ggml_add(ctx, tok,
                       ggml_mul(ctx, o, w((p + ".ls_cross").c_str())));

        // ---- self attention (packed qkv, qk-norm) ----
        h = ln(ctx, tok, w((p + ".self_norm.weight").c_str()),
               w((p + ".self_norm.bias").c_str()));
        ggml_tensor* qkv = lin(ctx, h, w((p + ".attn.qkv.weight").c_str()),
                               w((p + ".attn.qkv.bias").c_str()));
        ggml_tensor* qv = ggml_cont(ctx, head_view(ctx, qkv, 0, T));
        ggml_tensor* kv_ = ggml_cont(ctx, head_view(ctx, qkv, C, T));
        ggml_tensor* vv = ggml_cont(ctx, head_view(ctx, qkv, 2 * C, T));
        qv = ln(ctx, qv, w((p + ".attn.qn.weight").c_str()),
                w((p + ".attn.qn.bias").c_str()));
        kv_ = ln(ctx, kv_, w((p + ".attn.kn.weight").c_str()),
                 w((p + ".attn.kn.bias").c_str()));
        qv = fa_permute(ctx, qv);
        ggml_tensor* kp = fa_permute(ctx, kv_);
        ggml_tensor* vp = fa_permute(ctx, vv);
        o = f32_attn() ? exact_attention_core(ctx, qv, kp, vp,
                                              1.0f / std::sqrt((float)dh))
                       : ggml_flash_attn_ext(ctx, qv, kp, vp, nullptr,
                                             1.0f / std::sqrt((float)dh), 0.0f,
                                             0.0f);
        o = ggml_reshape_3d(ctx, o, C, T, 1);
        o = lin(ctx, o, w((p + ".attn.proj.weight").c_str()),
                w((p + ".attn.proj.bias").c_str()));
        tok = ggml_add(ctx, tok, ggml_mul(ctx, o, w((p + ".ls_self").c_str())));

        // ---- mlp ----
        h = ln(ctx, tok, w((p + ".mlp_norm.weight").c_str()),
               w((p + ".mlp_norm.bias").c_str()));
        ggml_tensor* f = lin(ctx, h, w((p + ".fc1.weight").c_str()),
                             w((p + ".fc1.bias").c_str()));
        f = ggml_gelu_erf(ctx, f);
        f = lin(ctx, f, w((p + ".fc2.weight").c_str()),
                w((p + ".fc2.bias").c_str()));
        return ggml_add(ctx, tok, ggml_mul(ctx, f, w((p + ".ls_mlp").c_str())));
    }

    // K from a {2C, N, 1} kv projection: rows [0,C) -> per-(h,n) LayerNorm ->
    // {dh, N, nh, 1}; V: rows [C,2C) -> {dh, N, nh, 1}. Both flash-layout.
    // Element (c, h, n) of the K block sits at n*2C + h*dh + c, so nb2 (n)
    // = 2C*4. The views are materialized with ggml_cont BEFORE the LayerNorm
    // (a strided-LN once misbehaved; cont costs 2 MB here).
    void build_kv(ggml_context* ctx,
                  ggml_tensor* kv,
                  const char* k_norm_w,
                  const char* k_norm_b,
                  ggml_tensor** K,
                  ggml_tensor** V,
                  int N) const {
        ggml_tensor* k = ggml_cont(
                ctx, ggml_view_4d(ctx, kv, dh, nh, N, 1, dh * sizeof(float),
                                  2 * C * sizeof(float),
                                  2 * C * N * sizeof(float), 0));
        ggml_tensor* v = ggml_cont(
                ctx,
                ggml_view_4d(ctx, kv, dh, nh, N, 1, dh * sizeof(float),
                             2 * C * sizeof(float), 2 * C * N * sizeof(float),
                             (size_t)C * sizeof(float)));
        k = ln(ctx, k, w(k_norm_w), w(k_norm_b));
        *K = fa_permute(ctx, k);
        *V = fa_permute(ctx, v);
    }

    // ggml_conv_2d hardwires its im2col activations to F16, which costs
    // ~5e-4 relative on the patch tokens and amplified to several percent
    // through the post-rgb branch. This variant keeps im2col in F32 (the
    // weight is cast to F32 on CPU — same rule as lin()).
    ggml_tensor* conv_2d_f32(ggml_context* ctx,
                             ggml_tensor* w_,
                             ggml_tensor* in,
                             int ps) const {
        ggml_tensor* w32 = w_;
        if (w_->type == GGML_TYPE_F16 && be.name == "CPU") {
            w32 = ggml_cast(ctx, w_, GGML_TYPE_F32);
        }
        ggml_tensor* im2col = ggml_im2col(ctx, w32, in, ps, ps, 0, 0, 1, 1,
                                          true, GGML_TYPE_F32);
        ggml_tensor* r = ggml_mul_mat(
                ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                                im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                ggml_reshape_2d(ctx, w32, w32->ne[0] * w32->ne[1] * w32->ne[2],
                                w32->ne[3]));
        r = ggml_reshape_4d(ctx, r, im2col->ne[1], im2col->ne[2], im2col->ne[3],
                            w32->ne[3]);
        return ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
    }

    bool stage_weights() {
        const size_t n = model->tensors.size();
        ggml_init_params wp{ggml_tensor_overhead() * n + (1u << 20), nullptr,
                            true};
        wctx = ggml_init(wp);
        if (!wctx) return false;
        // F16-stored weights are exactly representable in F32: staging them
        // as F32 makes the f16 GGUF tier bit-identical to the f32 tier on
        // every backend (the CUDA f16-weight GEMM kernels otherwise quantize
        // activations internally; the CPU backend already cast per call).
        // q8_0 keeps its native type — the quant IS the tier.
        auto staged_type = [](ggml_type t) {
            return t == GGML_TYPE_F16 ? GGML_TYPE_F32 : t;
        };
        for (const auto& [name, ht] : model->tensors) {
            ggml_tensor* t =
                    ggml_new_tensor(wctx, staged_type(ht.type), 4, ht.ne);
            ggml_set_name(t, name.c_str());
            wt[name] = t;
        }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, be.handle);
        if (!wbuf) return false;
        for (const auto& [name, ht] : model->tensors) {
            if (ht.type == GGML_TYPE_F16) {
                const int64_t nels = ht.ne[0] * ht.ne[1] * ht.ne[2] * ht.ne[3];
                std::vector<float> f32((size_t)nels);
                const auto* h16 =
                        reinterpret_cast<const ggml_fp16_t*>(ht.data.data());
                for (int64_t i = 0; i < nels; i++)
                    f32[(size_t)i] = ggml_fp16_to_fp32(h16[i]);
                ggml_backend_tensor_set(wt[name], f32.data(), 0,
                                        nels * sizeof(float));
            } else {
                ggml_backend_tensor_set(wt[name], ht.data.data(), 0,
                                        ht.data.size());
            }
        }
        log_info("querysplat: staged %zu tensors (%.2f GB)", n,
                 double(wbuf ? ggml_backend_buffer_get_size(wbuf) : 0) / 1e9);
        return true;
    }

    bool build(int s, int h, int w_, int patch_token_start) {
        S = s;
        H = h;
        W = w_;
        Ncr = patch_token_start;
        C = cfg->enc_embed_dim;
        nh = cfg->enc_num_heads;
        dh = C / nh;
        Ngs = cfg->num_gs_tokens;
        gpt = cfg->num_gaussians_per_token;
        P = (H / cfg->patch_size) * (W / cfg->patch_size);
        Nagg = Ncr + P;
        Nv = Nagg * S;
        Nr = P * S;
        post_out = 3 * (cfg->gaussian_sh_degree + 1) *
                           (cfg->gaussian_sh_degree + 1) +
                   1;

        const auto t0 = std::chrono::steady_clock::now();
        // node budget: 18 blocks x ~45 + fusion/kv ~80
        constexpr size_t kNodes = 4096;
        ggml_init_params gp{ggml_tensor_overhead() * kNodes +
                                    ggml_graph_overhead_custom(kNodes, false),
                            nullptr, true};
        gctx = ggml_init(gp);
        if (!gctx) return false;

        // ---- inputs ----
        for (int i = 0; i < 4; i++) {
            in_layers[i] = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, C, Nagg, S);
            ggml_set_input(in_layers[i]);
        }
        in_rgb = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, W, H, 3, S);
        in_plucker = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, W, H, 6, S);
        ggml_set_input(in_rgb);
        ggml_set_input(in_plucker);

        const bool dump = !dump_dir.empty() || seams_kv_requested;

        // ---- layer fusion (output_norm included) ----
        ggml_tensor* final_layer = in_layers[3];
        ggml_tensor* cam =
                ggml_cont(ctx_of(gctx), ggml_view_3d(gctx, final_layer, C, 1, S,
                                                     final_layer->nb[1],
                                                     final_layer->nb[2], 0));
        ggml_tensor* scene =
                ggml_cont(ctx_of(gctx),
                          ggml_view_3d(gctx, final_layer, C, Ncr - 1, S,
                                       final_layer->nb[1], final_layer->nb[2],
                                       (size_t)C * sizeof(float)));
        ggml_tensor* fused = nullptr;
        for (int l = 0; l < 4; l++) {
            ggml_tensor* patch_l = ggml_cont(
                    ctx_of(gctx),
                    ggml_view_3d(gctx, in_layers[l], C, P, S,
                                 in_layers[l]->nb[1], in_layers[l]->nb[2],
                                 (size_t)Ncr * C * sizeof(float)));
            ggml_tensor* term =
                    ggml_scale(ctx_of(gctx), patch_l, cfg->layer_weights[l]);
            fused = (l == 0) ? term : ggml_add(ctx_of(gctx), fused, term);
        }
        out_fused_patches = fused;
        if (dump) ggml_set_output(out_fused_patches);
        ggml_tensor* tokens = ggml_concat(
                ctx_of(gctx), ggml_concat(ctx_of(gctx), cam, scene, 1), fused,
                1);  // {C, Nagg, S}
        tokens = ln(ctx_of(gctx), tokens, w("onorm.weight"), w("onorm.bias"));
        tokens = ggml_reshape_3d(ctx_of(gctx), tokens, C, Nv, 1);  // view-major
        out_fused = tokens;  // primary seam == torch output.tokens
        if (dump) ggml_set_output(out_fused);

        // ---- geometry KV ----
        // torch _encode_to_kv applies EncDecBackbone.encoder_norm BEFORE
        // kv_proj (a SECOND LayerNorm on top of the encoder-side
        // output_norm) — skipping it diverges the whole decoder chain.
        tokens = ln(ctx_of(gctx), tokens, w("gen.weight"), w("gen.bias"));
        ggml_tensor* kv =
                lin(ctx_of(gctx), tokens, w("gkv.weight"), w("gkv.bias"));
        out_kvproj = kv;
        if (dump) ggml_set_output(out_kvproj);
        build_kv(ctx_of(gctx), kv, "gkn.weight", "gkn.bias", &out_kv_k,
                 &out_kv_v, Nv);
        if (dump) {
            ggml_set_output(out_kv_k);
            ggml_set_output(out_kv_v);
        }
        ggml_tensor* Kg = out_kv_k;
        ggml_tensor* Vg = out_kv_v;

        // ---- rgb/plucker patch tokens -> post_rgb KV ----
        auto patch_embed = [&](ggml_tensor* in, const char* prefix) {
            ggml_tensor* x = conv_2d_f32(
                    ctx_of(gctx),
                    w((std::string(prefix) + ".proj.weight").c_str()), in,
                    cfg->patch_size);
            x = ggml_add(
                    ctx_of(gctx), x,
                    ggml_reshape_4d(
                            ctx_of(gctx),
                            w((std::string(prefix) + ".proj.bias").c_str()), 1,
                            1, C, 1));
            // conv out ne = {Wp, Hp, C, S}; permute (1,2,0,3) gives
            // {C, OW, OH, S} so the flattened token grid is x-fastest,
            // matching torch patch_embed's flatten(2).transpose(1,2)
            x = ggml_cont(ctx_of(gctx),
                          ggml_permute(ctx_of(gctx), x, 1, 2, 0, 3));
            x = ggml_reshape_3d(ctx_of(gctx), x, C, P, S);
            if (dump) {
                if (std::string(prefix) == "pe") {
                    out_pe_conv = x;
                    ggml_set_output(out_pe_conv);
                } else {
                    out_ppe_conv = x;
                    ggml_set_output(out_ppe_conv);
                }
            }
            return ln(ctx_of(gctx), x,
                      w((std::string(prefix) + ".norm.weight").c_str()),
                      w((std::string(prefix) + ".norm.bias").c_str()));
        };
        ggml_tensor* rgb_tok = patch_embed(in_rgb, "pe");
        ggml_tensor* plk_tok = patch_embed(in_plucker, "ppe");
        ggml_tensor* rgb_tokens = ggml_add(ctx_of(gctx), rgb_tok, plk_tok);
        rgb_tokens = ggml_reshape_3d(ctx_of(gctx), rgb_tokens, C, Nr, 1);
        out_rgb_tokens =
                rgb_tokens;  // seam: torch _original_tokens_to_kv input
        if (dump) ggml_set_output(out_rgb_tokens);
        rgb_tokens =
                ln(ctx_of(gctx), rgb_tokens, w("oen.weight"), w("oen.bias"));
        ggml_tensor* post_kv =
                lin(ctx_of(gctx), rgb_tokens, w("okv.weight"), w("okv.bias"));
        build_kv(ctx_of(gctx), post_kv, "okn.weight", "okn.bias", &out_post_k,
                 &out_post_v, Nr);
        if (dump) {
            ggml_set_output(out_post_k);
            ggml_set_output(out_post_v);
        }
        ggml_tensor* Kp = out_post_k;
        ggml_tensor* Vp = out_post_v;

        // ---- decoder tokens: [eye | gs_tokens] ----
        ggml_tensor* eye = w("eye");  // {C,1,1,1}
        ggml_tensor* gsw = w("gst");  // {C,Ngs,1,1}
        ggml_tensor* tok =
                ggml_concat(ctx_of(gctx), eye,
                            ggml_reshape_3d(ctx_of(gctx), gsw, C, Ngs, 1), 1);
        const int T = Ngs + 1;
        for (int i = 0; i < cfg->dec_depth; i++) {
            tok = decoder_block(ctx_of(gctx), tok, "dec" + std::to_string(i),
                                Kg, Vg, T);
        }
        ggml_tensor* geom = ggml_cont(
                ctx_of(gctx),
                ggml_view_3d(ctx_of(gctx), tok, C, Ngs, 1, tok->nb[1],
                             tok->nb[2],
                             (size_t)C * sizeof(float)));  // drop the eye row
        out_geom = geom;
        if (dump) ggml_set_output(out_geom);

        // ---- appearance branch ----
        ggml_tensor* ctok =
                ggml_add(ctx_of(gctx), geom,
                         ggml_reshape_3d(ctx_of(gctx), w("prgs"), C, Ngs, 1));
        for (int i = 0; i < cfg->post_decoder_depth; i++) {
            ctok = decoder_block(ctx_of(gctx), ctok, "pdec" + std::to_string(i),
                                 Kp, Vp, Ngs);
        }

        // ---- raw heads (activations run host-side) ----
        out_deconv =
                lin(ctx_of(gctx), geom, w("deconv.weight"), w("deconv.bias"));
        out_color = lin(ctx_of(gctx), ctok, w("dh.weight"), w("dh.bias"));
        ggml_set_output(out_deconv);
        ggml_set_output(out_color);

        graph = ggml_new_graph_custom(gctx, kNodes, false);
        ggml_build_forward_expand(graph, out_deconv);
        ggml_build_forward_expand(graph, out_color);
        if (dump) {
            ggml_build_forward_expand(graph, out_fused);
            ggml_build_forward_expand(graph, out_fused_patches);
            ggml_build_forward_expand(graph, out_rgb_tokens);
            ggml_build_forward_expand(graph, out_kvproj);
            ggml_build_forward_expand(graph, out_pe_conv);
            ggml_build_forward_expand(graph, out_ppe_conv);
            ggml_build_forward_expand(graph, out_kv_k);
            ggml_build_forward_expand(graph, out_kv_v);
            ggml_build_forward_expand(graph, out_post_k);
            ggml_build_forward_expand(graph, out_post_v);
            ggml_build_forward_expand(graph, out_geom);
        }
        galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(be.handle));
        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            log_error("querysplat: graph allocation failed");
            return false;
        }
        built = true;
        const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
        log_info(
                "querysplat: graph built in %.0f ms (S=%d P=%d Nagg=%d T=%d "
                "post_out=%d)",
                ms, S, P, Nagg, T, post_out);
        return true;
    }

    // tiny helper: gctx typed use (keeps the lambdas above readable)
    static ggml_context* ctx_of(ggml_context* c) { return c; }

    bool run(const float* layer_tokens,
             const float* rgb,
             const float* plucker,
             DecoderOutputs& out) {
        if (!built) return false;
        const size_t per_layer = (size_t)C * Nagg * S;
        for (int i = 0; i < 4; i++) {
            ggml_backend_tensor_set(in_layers[i], layer_tokens + i * per_layer,
                                    0, per_layer * sizeof(float));
        }
        ggml_backend_tensor_set(in_rgb, rgb, 0,
                                (size_t)S * 3 * H * W * sizeof(float));
        ggml_backend_tensor_set(in_plucker, plucker, 0,
                                (size_t)S * 6 * H * W * sizeof(float));
        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            log_error("querysplat: gallocr alloc failed in run()");
            return false;
        }
        enum ggml_status st = ggml_backend_graph_compute(be.handle, graph);
        if (st != GGML_STATUS_SUCCESS) {
            log_error("querysplat: backend graph compute failed: %s",
                      ggml_status_to_string(st));
            return false;
        }

        auto fetch = [&](ggml_tensor* t, std::vector<float>& dst) {
            dst.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, dst.data(), 0,
                                    dst.size() * sizeof(float));
        };
        std::vector<float> deconv_raw, color_raw;
        std::vector<float> kv_k_data, kv_v_data, pk_data, pv_data;
        fetch(out_deconv, deconv_raw);
        fetch(out_color, color_raw);
        const bool fetch_kv_only = seams_kv_requested;
        if (!dump_dir.empty() || fetch_kv_only) {
            const char* dir = dump_dir.c_str();
            auto maybe_write = [&](const char* name,
                                   const std::vector<float>& v) {
                if (dump_dir.empty()) return;  // TTO mode: seams only
                FILE* fp = fopen(
                        (std::string(dir) + "/" + name + ".bin").c_str(), "wb");
                if (fp) {
                    fwrite(v.data(), 4, v.size(), fp);
                    fclose(fp);
                }
            };
            auto write_bin = [&](const char* name,
                                 const std::vector<float>& v) {
                FILE* fp = fopen(
                        (std::string(dir) + "/" + name + ".bin").c_str(), "wb");
                if (fp) {
                    fwrite(v.data(), 4, v.size(), fp);
                    fclose(fp);
                }
            };
            std::vector<float> tmp;
            // NOTE: kv_k_data etc. are the FUNCTION-LEVEL vectors declared
            // above (line ~632): fetch fills them here and the fetch_kv_only
            // block after activate_final() re-moves them into out.seams (the
            // assignment `out = activate_final(...)` clears the map). Shadow-
            // ing them here would re-fill the seams with empty vectors and
            // seed the TTO memory with all zeros.
            fetch(out_fused, tmp);
            out.seams["cpp_tokens"] = tmp;
            maybe_write("cpp_tokens", tmp);
            fetch(out_fused_patches, tmp);
            out.seams["cpp_fused_patches"] = tmp;
            maybe_write("cpp_fused_patches", tmp);
            fetch(out_rgb_tokens, tmp);
            out.seams["cpp_rgb_tokens"] = tmp;
            maybe_write("cpp_rgb_tokens", tmp);
            fetch(out_kvproj, tmp);
            out.seams["cpp_geom_kvproj"] = tmp;
            maybe_write("cpp_geom_kvproj", tmp);
            fetch(out_pe_conv, tmp);
            out.seams["cpp_pe_conv"] = tmp;
            maybe_write("cpp_pe_conv", tmp);
            fetch(out_ppe_conv, tmp);
            out.seams["cpp_ppe_conv"] = tmp;
            maybe_write("cpp_ppe_conv", tmp);
            fetch(out_kv_k, kv_k_data);
            out.seams["cpp_geom_kv_k"] = kv_k_data;
            maybe_write("cpp_geom_kv_k", kv_k_data);
            fetch(out_kv_v, kv_v_data);
            out.seams["cpp_geom_kv_v"] = kv_v_data;
            maybe_write("cpp_geom_kv_v", kv_v_data);
            fetch(out_post_k, pk_data);
            out.seams["cpp_post_kv_k"] = pk_data;
            maybe_write("cpp_post_kv_k", pk_data);
            fetch(out_post_v, pv_data);
            out.seams["cpp_post_kv_v"] = pv_data;
            maybe_write("cpp_post_kv_v", pv_data);
            fetch(out_geom, tmp);
            out.seams["cpp_geom_tokens"] = tmp;
            maybe_write("cpp_geom_tokens", tmp);
            if (!dump_dir.empty()) {
                maybe_write("cpp_deconv_raw", deconv_raw);
                maybe_write("cpp_color_raw", color_raw);
            }
        }

        // host activation heads -> final gaussians
        const int64_t N = (int64_t)Ngs * gpt;
        out = activate_final(deconv_raw, color_raw, N);
        // re-fill the KV seams (activate_final's assignment cleared them)
        if (fetch_kv_only) {
            out.seams["cpp_geom_kv_k"] = kv_k_data;
            out.seams["cpp_geom_kv_v"] = kv_v_data;
            out.seams["cpp_post_kv_k"] = pk_data;
            out.seams["cpp_post_kv_v"] = pv_data;
        }
        return true;
    }

    DecoderOutputs activate_final(const std::vector<float>& deconv_raw,
                                  const std::vector<float>& color_raw,
                                  int64_t N) {
        DecoderOutputs out;
        GaussianOutput g = activate_gaussians(
                deconv_raw.data(), color_raw.data(), Ngs, gpt, post_out,
                cfg->gaussian_scale_cap, cfg->gaussian_z_offset,
                cfg->gaussian_sh_degree, cfg->gaussian_sh_rest_scale);
        out.gaussians = std::move(g.data);
        out.N = g.N;
        out.color_dim = g.color_dim;
        return out;
    }
};

// ---- public class ----

bool read_decoder_config(const std::string& gguf_path, DecoderConfig* out) {
    gguf_init_params p{true, nullptr};
    gguf_context* meta = gguf_init_from_file(gguf_path.c_str(), p);
    if (!meta) {
        log_error("querysplat: cannot open %s", gguf_path.c_str());
        return false;
    }
    DecoderConfig cfg;
    auto get_u32 = [&](const char* k, int def) {
        const int id = gguf_find_key(meta, k);
        return id >= 0 ? (int)gguf_get_val_u32(meta, id) : def;
    };
    auto get_f32 = [&](const char* k, float def) {
        const int id = gguf_find_key(meta, k);
        return id >= 0 ? gguf_get_val_f32(meta, id) : def;
    };
    cfg.enc_embed_dim = get_u32("querysplat.enc_embed_dim", cfg.enc_embed_dim);
    cfg.enc_num_heads = get_u32("querysplat.enc_num_heads", cfg.enc_num_heads);
    cfg.dec_depth = get_u32("querysplat.dec_depth", cfg.dec_depth);
    cfg.post_decoder_depth =
            get_u32("querysplat.vggt_original_encoder_post_decoder_depth",
                    cfg.post_decoder_depth);
    cfg.num_gs_tokens = get_u32("querysplat.num_gs_tokens", cfg.num_gs_tokens);
    cfg.num_gaussians_per_token = get_u32("querysplat.num_gaussians_per_token",
                                          cfg.num_gaussians_per_token);
    cfg.patch_size = get_u32("querysplat.vggt_original_encoder_patch_size",
                             cfg.patch_size);
    cfg.mlp_ratio = get_f32("querysplat.mlp_ratio", cfg.mlp_ratio);
    cfg.gaussian_scale_cap =
            get_f32("querysplat.gaussian_scale_cap", cfg.gaussian_scale_cap);
    cfg.gaussian_z_offset =
            get_f32("querysplat.gaussian_z_offset", cfg.gaussian_z_offset);
    cfg.gaussian_sh_degree =
            get_u32("querysplat.gaussian_sh_degree", cfg.gaussian_sh_degree);
    cfg.gaussian_sh_rest_scale = get_f32("querysplat.gaussian_sh_rest_scale",
                                         cfg.gaussian_sh_rest_scale);
    cfg.gaussian_save_opacity_threshold =
            get_f32("querysplat.gaussian_save_opacity_threshold",
                    cfg.gaussian_save_opacity_threshold);
    cfg.znear = get_f32("querysplat.znear", cfg.znear);
    cfg.zfar = get_f32("querysplat.zfar", cfg.zfar);
    bool ok = true;
    const int is = gguf_find_key(meta, "querysplat.img_size");
    if (is >= 0 && gguf_get_arr_type(meta, is) == GGUF_TYPE_INT32 &&
        gguf_get_arr_n(meta, is) == 2) {
        const int32_t* v =
                static_cast<const int32_t*>(gguf_get_arr_data(meta, is));
        cfg.img_size[0] = v[0];
        cfg.img_size[1] = v[1];
    }
    const int bg = gguf_find_key(meta, "querysplat.bg_color");
    if (bg >= 0) {
        // QuerySplat._background_color
        const std::string c = gguf_get_val_str(meta, bg);
        const float v = c == "white"   ? 1.0f
                        : c == "black" ? 0.0f
                        : c == "grey"  ? 0.5f
                                       : -1.0f;
        if (v < 0) {
            log_error("querysplat: invalid background color: %s", c.c_str());
            ok = false;
        }
        for (float& b : cfg.bg_color) b = v;
    }
    gguf_free(meta);
    if (ok && cfg.enc_embed_dim % cfg.enc_num_heads) {
        log_error("querysplat: embed_dim %% num_heads != 0");
        ok = false;
    }
    *out = cfg;
    return ok;
}

QuerySplatDecoder::QuerySplatDecoder(const std::string& gguf_path,
                                     map_anything::Backend backend,
                                     const DecoderRuntimeOptions& opts)
    : impl_(new Impl()) {
    impl_->be = backend;
    impl_->f32_accuracy = opts.f32_accuracy;
    DecoderConfig cfg;
    if (!read_decoder_config(gguf_path, &cfg)) {
        impl_.reset();
        return;
    }
    impl_->model = map_anything::load_gguf(gguf_path);
    if (!impl_->model) {
        impl_.reset();  // valid() must be false after a failed load
        return;
    }
    cfg_ = cfg;
    impl_->cfg = &cfg_;

    if (!impl_->stage_weights()) {
        log_error("querysplat: weight staging failed");
        impl_.reset();
        return;
    }

    // constant-fold the layer weights from eye_token/eye_layer_mlp
    const auto& t = impl_->model->tensors;
    auto ht = [&](const char* k) -> const map_anything::HostTensor& {
        return t.at(k);
    };
    fold_layer_weights(ht("eye"), ht("elm.0.weight"), ht("elm.0.bias"),
                       ht("elm.1.weight"), ht("elm.1.bias"), ht("elm.3.weight"),
                       ht("elm.3.bias"), cfg_.layer_weights);
    log_info("querysplat: layer_weights = %.4f %.4f %.4f %.4f",
             cfg_.layer_weights[0], cfg_.layer_weights[1],
             cfg_.layer_weights[2], cfg_.layer_weights[3]);
    // explicit runtime switches (no environment variables — see
    // DecoderRuntimeOptions in querysplat_graph.hpp)
    impl_->dump_dir = opts.dump_dir;
    impl_->seams_kv_requested = opts.fetch_kv;
}

QuerySplatDecoder::~QuerySplatDecoder() = default;

bool QuerySplatDecoder::valid() const { return impl_ != nullptr; }

bool QuerySplatDecoder::ensure_shape(int S,
                                     int H,
                                     int W,
                                     int patch_token_start) {
    if (impl_->built && impl_->S == S && impl_->H == H && impl_->W == W &&
        impl_->Ncr == patch_token_start) {
        return true;
    }
    // release the previous graph
    if (impl_->galloc) ggml_gallocr_free(impl_->galloc);
    if (impl_->gctx) ggml_free(impl_->gctx);
    impl_->galloc = nullptr;
    impl_->gctx = nullptr;
    impl_->graph = nullptr;
    impl_->built = false;
    return impl_->build(S, H, W, patch_token_start);
}

bool QuerySplatDecoder::run(const float* layer_tokens,
                            const float* rgb,
                            const float* plucker,
                            DecoderOutputs& out) {
    return impl_->run(layer_tokens, rgb, plucker, out);
}

// ===== TTO session (t4) — see docs/tto_design.md ===========================
//
// Fine-tunes the four KV memories (geom K/V + post_rgb K/V) against the
// input views' pixels, mirroring QuerySplat.forward_tto_memory (kv target,
// sh0). Two graphs share one tensor context:
//   graph F: mem -> raw heads                        (tto_forward)
//   graph B: mem + d(raw) -> forward recompute, block VJPs, AdamW in-place
// Attention runs EXPLICIT and CHUNKED over the query dim; P is recomputed
// inside the backward (torch's mem-efficient SDPA does the same) so nothing
// T-sized persists across layers. mul_mat(A,B): ne[0] is the contraction,
// result.ne[0] = A.ne[1], result.ne[1] = B.ne[1] (A.B^T over row vectors).

namespace {

constexpr int kTtoChunk = 512;

// A&S 7.1.26 erf approximation (|eps| <= 1.5e-7), mirrored for z < 0;
// ggml has no erf op.
struct TtoConsts {
    ggml_tensor *erf_c, *erf_a1, *erf_a2, *erf_a3, *erf_a4, *erf_a5, *half,
            *inv_sqrt2pi, *one;
};

ggml_tensor* tto_erf(ggml_context* ctx, ggml_tensor* z, const TtoConsts& cs) {
    ggml_tensor* one = cs.one;
    ggml_tensor* az = ggml_abs(ctx, z);
    ggml_tensor* t = ggml_div(ctx, ggml_repeat(ctx, one, az),
                              ggml_add(ctx, ggml_scale(ctx, az, 0.3275911f),
                                       ggml_repeat(ctx, one, az)));
    ggml_tensor* t2 = ggml_mul(ctx, t, t);
    ggml_tensor* t3 = ggml_mul(ctx, t2, t);
    ggml_tensor* t4 = ggml_mul(ctx, t2, t2);
    ggml_tensor* t5 = ggml_mul(ctx, t4, t);
    ggml_tensor* poly = ggml_add(ctx, ggml_mul(ctx, t, cs.erf_a1),
                                 ggml_mul(ctx, t2, cs.erf_a2));
    poly = ggml_add(ctx, poly, ggml_mul(ctx, t3, cs.erf_a3));
    poly = ggml_add(ctx, poly, ggml_mul(ctx, t4, cs.erf_a4));
    poly = ggml_add(ctx, poly, ggml_mul(ctx, t5, cs.erf_a5));
    ggml_tensor* e =
            ggml_mul(ctx, poly,
                     ggml_exp(ctx, ggml_scale(ctx, ggml_sqr(ctx, az), -1.0f)));
    return ggml_mul(ctx, ggml_sgn(ctx, z),
                    ggml_sub(ctx,
                             // cs.one: a fresh graph-scratch 1-element tensor
                             // would hold stale memory and corrupt erf/gelu'
                             ggml_repeat(ctx, cs.one, e), e));
}

// d(gelu_erf(x))/dx = Phi(x) + x*phi(x)
ggml_tensor* tto_gelu_deriv(ggml_context* ctx,
                            ggml_tensor* x,
                            const TtoConsts& cs) {
    ggml_tensor* Phi = ggml_scale(
            ctx,
            ggml_add(ctx,
                     tto_erf(ctx, ggml_scale(ctx, x, 0.70710678118654752f), cs),
                     ggml_repeat(ctx, cs.one, x)),
            0.5f);
    ggml_tensor* phi = ggml_scale(
            ctx, ggml_exp(ctx, ggml_scale(ctx, ggml_sqr(ctx, x), -0.5f)),
            0.3989422804014327f);
    return ggml_add(ctx, Phi, ggml_mul(ctx, x, phi));
}

// linear backward: d_x = W^T . d_y. The contraction axis is Cout on both
// sides: A = cont(transpose(w)) {Cout,Cin} (cont clears the transposed flag
// — mul_mat asserts on it), B = dy {Cout,T} -> {Cin,T}.
ggml_tensor* tto_lin_bwd(ggml_context* ctx, ggml_tensor* w, ggml_tensor* dy) {
    return ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, w)), dy);
}

// LayerNorm backward; x = the LN input (feature axis ne[0]); eps 1e-5.
// dy may arrive with a different rank (e.g. 3D {C,T} vs a 4D head-view x):
// it is reshaped to x's shape first so the w broadcast lines up per-dim.
ggml_tensor* tto_lnback(ggml_context* ctx,
                        ggml_tensor* dy,
                        ggml_tensor* x,
                        ggml_tensor* w,
                        ggml_tensor* one) {
    if (ggml_nelements(dy) == ggml_nelements(x) && dy->ne[0] != x->ne[0]) {
        std::vector<int64_t> ne(4);
        for (int d = 0; d < 4; d++) ne[d] = x->ne[d];
        dy = ggml_reshape(ctx, dy,
                          ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data()));
    }
    const float C = (float)x->ne[0];
    ggml_tensor* mean = ggml_scale(ctx, ggml_sum_rows(ctx, x), 1.0f / C);
    ggml_tensor* ctr = ggml_sub(ctx, x, mean);
    ggml_tensor* var = ggml_scale(
            ctx, ggml_sum_rows(ctx, ggml_mul(ctx, ctr, ctr)), 1.0f / C);
    // one MUST be a properly initialized 1.0 constant (TtoConsts.one in the
    // persistent tbuf): a fresh graph-scratch tensor holds stale/unset memory,
    // which zeroes or rescales rstd and silently kills the whole VJP.
    ggml_tensor* rstd =
            ggml_div(ctx, ggml_repeat(ctx, one, var),
                     ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0f, 1e-5f)));
    ggml_tensor* xhat = ggml_mul(ctx, ctr, rstd);
    ggml_tensor* dyw = ggml_mul(ctx, dy, w);
    ggml_tensor* m1 = ggml_scale(ctx, ggml_sum_rows(ctx, dyw), 1.0f / C);
    ggml_tensor* m2 = ggml_scale(
            ctx, ggml_sum_rows(ctx, ggml_mul(ctx, dyw, xhat)), 1.0f / C);
    return ggml_mul(
            ctx, ggml_sub(ctx, ggml_sub(ctx, dyw, m1), ggml_mul(ctx, xhat, m2)),
            rstd);
}

}  // namespace

bool QuerySplatDecoder::tto_begin(int S,
                                  int H,
                                  int W,
                                  int patch_token_start,
                                  const float* layer_tokens,
                                  const float* rgb,
                                  const float* plucker,
                                  float lr) {
    if (!impl_ || !impl_->built || impl_->cfg->gaussian_sh_degree != 0) {
        log_error("tto: requires a built sh0 decoder");
        return false;
    }
    if (impl_->tto_active) tto_end();
    Impl& m = *impl_;
    m.lr_cur = lr;
    m.seams_kv_requested = true;
    // the graph was built without the KV seams (fetch_kv off at build time):
    // force a rebuild so out_kv_k/v etc. become graph outputs
    impl_->built = false;
    if (!ensure_shape(S, H, W, patch_token_start)) {
        log_error("tto: rebuild with KV seams failed");
        return false;
    }
    DecoderOutputs seed;
    if (!m.run(layer_tokens, rgb, plucker, seed)) {
        log_error("tto: seeding inference failed");
        return false;
    }
    const auto& kv = seed.seams;
    if (!kv.count("cpp_geom_kv_k") || !kv.count("cpp_post_kv_k")) {
        log_error("tto: KV seams missing");
        return false;
    }
    const int C = m.cfg->enc_embed_dim, nh = m.cfg->enc_num_heads, dh = C / nh;
    const int P = (H / m.cfg->patch_size) * (W / m.cfg->patch_size);
    const int Nagg = patch_token_start + P;
    const int Nv = Nagg * S, Ngs = m.cfg->num_gs_tokens;
    m.tto_S = S;
    m.tto_H = H;
    m.tto_W = W;
    m.tto_Ncr = patch_token_start;
    m.tto_T = Ngs + 1;
    m.tto_Nv = Nv;

    constexpr size_t kTtoNodes = 65536;
    ggml_init_params gp{
            ggml_tensor_overhead() * kTtoNodes +
                    2 * ggml_graph_overhead_custom(kTtoNodes, false),
            nullptr, true};
    m.tctx = ggml_init(gp);
    if (!m.tctx) return false;
    ggml_context* ctx = m.tctx;
    const float sc = 1.0f / std::sqrt((float)dh);

    // trainable tensors live in a SEPARATE context with a persistent backend
    // buffer: ggml_gallocr skips tensors that already have a buffer (see
    // ggml_gallocr_init_tensor), so the AdamW in-place update lands in the
    // same storage the forward graph reads.
    ggml_init_params tp{ggml_tensor_overhead() * 16 + (1u << 20), nullptr,
                        true};
    m.tbuf_ctx = ggml_init(tp);
    if (!m.tbuf_ctx) return false;
    ggml_context* tbuf = m.tbuf_ctx;

    const char* mem_names[4] = {"mem_K", "mem_V", "mem_pK", "mem_pV"};
    const char* seed_names[4] = {"cpp_geom_kv_k", "cpp_geom_kv_v",
                                 "cpp_post_kv_k", "cpp_post_kv_v"};
    // post_rgb KV tokens are the per-view rgb patches (Nr = P*S), NOT the
    // 8192 gaussian tokens — the pdec cross-attn reads the same Nr keys the
    // run() graph builds (build_kv(..., Nr)).
    const int Npost = P * S;
    const int64_t mem_n[4] = {Nv, Nv, Npost, Npost};
    for (int i = 0; i < 4; i++) {
        m.mem[i] = ggml_new_tensor_4d(tbuf, GGML_TYPE_F32, dh, mem_n[i], nh, 1);
        ggml_set_name(m.mem[i], mem_names[i]);
        ggml_set_input(m.mem[i]);
        ggml_set_param(m.mem[i]);
        m.adam_m[i] =
                ggml_new_tensor_4d(tbuf, GGML_TYPE_F32, dh, mem_n[i], nh, 1);
        m.adam_v[i] =
                ggml_new_tensor_4d(tbuf, GGML_TYPE_F32, dh, mem_n[i], nh, 1);
    }
    m.d_deconv_in = ggml_new_tensor_2d(
            tbuf, GGML_TYPE_F32, 10 * m.cfg->num_gaussians_per_token, Ngs);
    m.d_color_in = ggml_new_tensor_2d(
            tbuf, GGML_TYPE_F32, m.post_out * m.cfg->num_gaussians_per_token,
            Ngs);
    ggml_set_input(m.d_deconv_in);
    ggml_set_input(m.d_color_in);
    m.adamw_params = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 7);
    ggml_set_input(m.adamw_params);
    // scalar constants for the erf/gelu backward (erf_c = 1/(1+0.3275911)
    // is built in-graph; the A&S 7.1.26 coefficients + 1/2 + 1/sqrt(2*pi))
    TtoConsts cs;
    cs.erf_a1 = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 1);
    cs.erf_a2 = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 1);
    cs.erf_a3 = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 1);
    cs.erf_a4 = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 1);
    cs.erf_a5 = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 1);
    cs.half = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 1);
    cs.inv_sqrt2pi = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 1);
    cs.one = ggml_new_tensor_1d(tbuf, GGML_TYPE_F32, 1);
    m.tbuf = ggml_backend_alloc_ctx_tensors(tbuf, m.be.handle);
    if (!m.tbuf) {
        log_error("tto: trainable buffer allocation failed");
        return false;
    }
    ggml_backend_buffer_clear(m.tbuf, 0);  // adam m/v start at zero
    for (int i = 0; i < 4; i++) {
        const auto& sd = kv.at(seed_names[i]);
        // seeding invariant: the seam dump must fill the memory exactly —
        // a partial fill would silently leave the tail of the KV memory at
        // zero and desynchronize the TTO graph from the run() graph.
        if ((int64_t)sd.size() != ggml_nelements(m.mem[i])) {
            log_error("tto: seed %s has %zu floats, mem expects %lld",
                      seed_names[i], sd.size(),
                      (long long)ggml_nelements(m.mem[i]));
            return false;
        }
        ggml_backend_tensor_set(m.mem[i], sd.data(), 0,
                                sd.size() * sizeof(float));
    }
    {
        const float cvals[] = {0.254829592f,        -0.284496736f, 1.421413741f,
                               -1.453152027f,       1.061405429f,  0.5f,
                               0.3989422804014327f, 1.0f};
        ggml_tensor* ct[8] = {cs.erf_a1, cs.erf_a2, cs.erf_a3,      cs.erf_a4,
                              cs.erf_a5, cs.half,   cs.inv_sqrt2pi, cs.one};
        for (int i = 0; i < 8; i++)
            ggml_backend_tensor_set(ct[i], &cvals[i], 0, sizeof(float));
    }

    // ---- shared graph builders ----
    // chunked explicit attention forward; returns o {dh, Tq, nh, 1}
    auto attn_fwd = [&](ggml_tensor* q, ggml_tensor* K, ggml_tensor* Vm,
                        int Tq) {
        ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, Vm));  // {N,dh,nh}
        ggml_tensor* o = nullptr;
        for (int t0 = 0; t0 < Tq; t0 += kTtoChunk) {
            const int tc = std::min(kTtoChunk, Tq - t0);
            ggml_tensor* q_c =
                    ggml_view_4d(ctx, q, dh, tc, nh, 1, q->nb[1], q->nb[2],
                                 q->nb[3], (size_t)t0 * q->nb[1]);
            ggml_tensor* s_c = ggml_mul_mat(ctx, K, q_c);  // {N, tc, nh}
            ggml_tensor* p_c = ggml_soft_max_ext(ctx, s_c, nullptr, sc, 0.0f);
            ggml_tensor* o_c = ggml_mul_mat(ctx, vt, p_c);  // {dh, tc, nh}
            o = o ? ggml_concat(ctx, o, o_c, 1) : o_c;
        }
        return ggml_cont(ctx, o);
    };
    // attention backward for one chunked layer: dO {dh,Tq,nh,1}; q/K/V as in
    // the forward; P recomputed per chunk (never persisted). dK/dV accumulate
    // in the mem layout {dh,N,nh,1}; returns dq {dh,Tq,nh,1}.
    auto attn_bwd = [&](ggml_tensor* dO, ggml_tensor* q, ggml_tensor* K,
                        ggml_tensor* Vm, ggml_tensor** dq_out,
                        ggml_tensor** dK_out, ggml_tensor** dV_out) {
        if (!dO || !q || !K || !Vm)
            log_error("attn_bwd: null tensor dO=%p q=%p K=%p V=%p", (void*)dO,
                      (void*)q, (void*)K, (void*)Vm);
        const int Tq = dO->ne[1];
        ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, Vm));
        ggml_tensor* dq = nullptr;
        ggml_tensor* dK = nullptr;
        ggml_tensor* dV = nullptr;
        for (int t0 = 0; t0 < Tq; t0 += kTtoChunk) {
            const int tc = std::min(kTtoChunk, Tq - t0);
            ggml_tensor* q_c =
                    ggml_view_4d(ctx, q, dh, tc, nh, 1, q->nb[1], q->nb[2],
                                 q->nb[3], (size_t)t0 * q->nb[1]);
            ggml_tensor* dO_c =
                    ggml_view_4d(ctx, dO, dh, tc, nh, 1, dO->nb[1], dO->nb[2],
                                 dO->nb[3], (size_t)t0 * dO->nb[1]);
            ggml_tensor* s_c = ggml_mul_mat(ctx, K, q_c);
            ggml_tensor* p_c = ggml_soft_max_ext(ctx, s_c, nullptr, sc, 0.0f);
            // dP = dL/dp[t,n] = dO[t,:]·v[n,:] — the V of o = p·v, NOT the
            // key: using K here corrupts dS and with it dK/dV/dq (a token-level
            // typo that reads like the correct dP = Kᵀq forward pattern).
            ggml_tensor* dP = ggml_mul_mat(ctx, Vm, dO_c);  // {N,tc,nh}
            ggml_tensor* dot = ggml_sum_rows(ctx, ggml_mul(ctx, dP, p_c));
            ggml_tensor* dS = ggml_mul(ctx, p_c, ggml_sub(ctx, dP, dot));
            ggml_tensor* dq_c = ggml_scale(
                    ctx,
                    ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, K)),
                                 dS),
                    sc);  // {dh,tc,nh}: K^T contraction over N
            dq = dq ? ggml_concat(ctx, dq, dq_c, 1) : dq_c;
            ggml_tensor* dO_t = ggml_cont(ctx, ggml_transpose(ctx, dO_c));
            ggml_tensor* P_t = ggml_cont(ctx, ggml_transpose(ctx, p_c));
            ggml_tensor* dv_inc = ggml_mul_mat(ctx, dO_t, P_t);  // {dh,N,nh}
            ggml_tensor* q_t = ggml_cont(ctx, ggml_transpose(ctx, q_c));
            ggml_tensor* dS_t = ggml_cont(ctx, ggml_transpose(ctx, dS));
            ggml_tensor* dk_inc = ggml_scale(ctx, ggml_mul_mat(ctx, q_t, dS_t),
                                             sc);  // {dh,N,nh}
            dV = dV ? ggml_add(ctx, dV, dv_inc) : dv_inc;
            dK = dK ? ggml_add(ctx, dK, dk_inc) : dk_inc;
        }
        *dq_out = ggml_cont(ctx, dq);
        *dK_out = dK;
        *dV_out = dV;
    };

    // decoder block forward (explicit attention). Records the anchors the
    // backward walks back through.
    struct BlockActs {
        ggml_tensor *tok_in, *tok_mid, *tok_mid2, *tok_out;
        ggml_tensor *q_ln_in, *q_cross_fa;
        ggml_tensor *qv_ln_in, *kv_ln_in, *q_self_fa;
        ggml_tensor *kp_fa, *vp_fa;
        ggml_tensor* fc1_in;
    };
    auto block_fwd = [&](ggml_tensor* tok, const std::string& p,
                         ggml_tensor* Kc, ggml_tensor* Vc, int Tq,
                         BlockActs* acts) {
        ggml_tensor* h = m.ln(ctx, tok, m.w((p + ".cross_norm.weight").c_str()),
                              m.w((p + ".cross_norm.bias").c_str()));
        ggml_tensor* q = m.lin(ctx, h, m.w((p + ".cross_qp.weight").c_str()),
                               m.w((p + ".cross_qp.bias").c_str()));
        ggml_tensor* q4 = ggml_reshape_4d(ctx, q, dh, nh, Tq, 1);
        ggml_tensor* q4_pre =
                q4;  // cross_qn LN input (lnback recomputes stats)
        q4 = m.ln(ctx, q4, m.w((p + ".cross_qn.weight").c_str()),
                  m.w((p + ".cross_qn.bias").c_str()));
        ggml_tensor* qfa = m.fa_permute(ctx, q4);
        acts->tok_in = tok;
        ggml_tensor* o_fa = attn_fwd(qfa, Kc, Vc, Tq);
        ggml_tensor* o = ggml_reshape_3d(
                ctx, ggml_cont(ctx, ggml_permute(ctx, o_fa, 0, 2, 1, 3)), C, Tq,
                1);
        o = m.lin(ctx, o, m.w((p + ".cross_op.weight").c_str()),
                  m.w((p + ".cross_op.bias").c_str()));
        tok = ggml_add(ctx, tok,
                       ggml_mul(ctx, o, m.w((p + ".ls_cross").c_str())));
        acts->tok_mid = tok;
        acts->q_ln_in = q4_pre;
        acts->q_cross_fa = qfa;
        h = m.ln(ctx, tok, m.w((p + ".self_norm.weight").c_str()),
                 m.w((p + ".self_norm.bias").c_str()));
        ggml_tensor* qkv = m.lin(ctx, h, m.w((p + ".attn.qkv.weight").c_str()),
                                 m.w((p + ".attn.qkv.bias").c_str()));
        ggml_tensor* qv = ggml_cont(ctx, m.head_view(ctx, qkv, 0, Tq));
        ggml_tensor* kv_ = ggml_cont(ctx, m.head_view(ctx, qkv, C, Tq));
        ggml_tensor* vv = ggml_cont(ctx, m.head_view(ctx, qkv, 2 * C, Tq));
        ggml_tensor* qv_pre = qv;   // qn LN input (lnback recomputes stats)
        ggml_tensor* kv_pre = kv_;  // kn LN input
        qv = m.ln(ctx, qv, m.w((p + ".attn.qn.weight").c_str()),
                  m.w((p + ".attn.qn.bias").c_str()));
        kv_ = m.ln(ctx, kv_, m.w((p + ".attn.kn.weight").c_str()),
                   m.w((p + ".attn.kn.bias").c_str()));
        ggml_tensor* qfa2 = m.fa_permute(ctx, qv);
        ggml_tensor* kp = m.fa_permute(ctx, kv_);
        ggml_tensor* vp = m.fa_permute(ctx, vv);
        ggml_tensor* o2_fa = attn_fwd(qfa2, kp, vp, Tq);
        ggml_tensor* o2 = ggml_reshape_3d(
                ctx, ggml_cont(ctx, ggml_permute(ctx, o2_fa, 0, 2, 1, 3)), C,
                Tq, 1);
        o2 = m.lin(ctx, o2, m.w((p + ".attn.proj.weight").c_str()),
                   m.w((p + ".attn.proj.bias").c_str()));
        tok = ggml_add(ctx, tok,
                       ggml_mul(ctx, o2, m.w((p + ".ls_self").c_str())));
        acts->tok_mid2 = tok;
        acts->qv_ln_in = qv_pre;
        acts->kv_ln_in = kv_pre;
        acts->q_self_fa = qfa2;
        acts->kp_fa = kp;
        acts->vp_fa = vp;
        h = m.ln(ctx, tok, m.w((p + ".mlp_norm.weight").c_str()),
                 m.w((p + ".mlp_norm.bias").c_str()));
        ggml_tensor* fc1 = m.lin(ctx, h, m.w((p + ".fc1.weight").c_str()),
                                 m.w((p + ".fc1.bias").c_str()));
        ggml_tensor* g = ggml_gelu_erf(ctx, fc1);
        ggml_tensor* f = m.lin(ctx, g, m.w((p + ".fc2.weight").c_str()),
                               m.w((p + ".fc2.bias").c_str()));
        tok = ggml_add(ctx, tok,
                       ggml_mul(ctx, f, m.w((p + ".ls_mlp").c_str())));
        acts->fc1_in = fc1;
        acts->tok_out = tok;
    };
    // block backward: d_out {C,T} -> d_in {C,T}; cross-attn accumulates
    // dK/dV (mem layout); self-attn routes its K/V grads back through the
    // qkv linear (kn LN -> qkv columns).
    auto block_bwd = [&](ggml_tensor* d_out, const std::string& p,
                         const BlockActs& acts, int Tq, ggml_tensor* memK,
                         ggml_tensor* memV, ggml_tensor** dK_acc,
                         ggml_tensor** dV_acc) {
        // ---- mlp ----
        ggml_tensor* d_f = ggml_mul(ctx, d_out, m.w((p + ".ls_mlp").c_str()));
        ggml_tensor* d_g =
                tto_lin_bwd(ctx, m.w((p + ".fc2.weight").c_str()), d_f);
        ggml_tensor* d_fc1 =
                ggml_mul(ctx, d_g, tto_gelu_deriv(ctx, acts.fc1_in, cs));
        ggml_tensor* d_h_mlp =
                tto_lin_bwd(ctx, m.w((p + ".fc1.weight").c_str()), d_fc1);
        ggml_tensor* d_tok = ggml_add(
                ctx, d_out,
                tto_lnback(ctx, d_h_mlp, acts.tok_mid2,
                           m.w((p + ".mlp_norm.weight").c_str()), cs.one));
        // ---- self attention ----
        ggml_tensor* d_o2 = ggml_mul(ctx, d_tok, m.w((p + ".ls_self").c_str()));
        // d_attn = proj^T . d_o2 — the gradient of the ATTENTION OUTPUT
        // {C,T}; its head-major form feeds attn_bwd as dO.
        ggml_tensor* d_attn = tto_lin_bwd(
                ctx, m.w((p + ".attn.proj.weight").c_str()), d_o2);  // {C,T}
        ggml_tensor* d_qv4 = ggml_reshape_4d(ctx, d_attn, dh, nh, Tq, 1);
        ggml_tensor* d_qfa = m.fa_permute(ctx, d_qv4);
        // self-attention: q/k/v all derive from tok. dK/dV are the gradients
        // of kp/vp — they flow back through kn-LN into the qkv rows.
        ggml_tensor* dK_self = nullptr;
        ggml_tensor* dV_self = nullptr;
        ggml_tensor* d_qfa_s = nullptr;
        attn_bwd(d_qfa, acts.q_self_fa, acts.kp_fa, acts.vp_fa, &d_qfa_s,
                 &dK_self, &dV_self);
        ggml_tensor* d_qv = ggml_cont(
                ctx, ggml_permute(ctx, d_qfa_s, 0, 2, 1, 3));  // {dh,nh,T}
        ggml_tensor* d_kv = ggml_cont(
                ctx, ggml_permute(ctx, dK_self, 0, 2, 1, 3));  // {dh,T,nh}
        ggml_tensor* d_vv =
                ggml_cont(ctx, ggml_permute(ctx, dV_self, 0, 2, 1, 3));
        d_qv = tto_lnback(ctx, ggml_reshape_3d(ctx, d_qv, C, Tq, 1),
                          acts.qv_ln_in, m.w((p + ".attn.qn.weight").c_str()),
                          cs.one);
        d_kv = tto_lnback(ctx, ggml_reshape_3d(ctx, d_kv, C, Tq, 1),
                          acts.kv_ln_in, m.w((p + ".attn.kn.weight").c_str()),
                          cs.one);
        // reassemble the qkv linear-output gradient {3C,T}: rows [0,C) = q,
        // [C,2C) = k, [2C,3C) = v
        ggml_tensor* d_qkv_full = ggml_concat(
                ctx,
                ggml_concat(ctx, ggml_reshape_3d(ctx, d_qv, C, Tq, 1),
                            ggml_reshape_3d(ctx, d_kv, C, Tq, 1), 0),
                ggml_reshape_3d(ctx, d_vv, C, Tq, 1), 0);
        ggml_tensor* d_h_self = tto_lin_bwd(
                ctx, m.w((p + ".attn.qkv.weight").c_str()), d_qkv_full);
        d_tok = ggml_add(
                ctx, d_tok,
                tto_lnback(ctx, d_h_self, acts.tok_mid,
                           m.w((p + ".self_norm.weight").c_str()), cs.one));
        // ---- cross attention (K/V are the trainable memory) ----
        ggml_tensor* d_o = ggml_mul(ctx, d_tok, m.w((p + ".ls_cross").c_str()));
        ggml_tensor* d_op = tto_lin_bwd(
                ctx, m.w((p + ".cross_op.weight").c_str()), d_o);  // {C,T}
        ggml_tensor* d_q4 = ggml_reshape_4d(ctx, d_op, dh, nh, Tq, 1);
        ggml_tensor* d_qfa_x = m.fa_permute(ctx, d_q4);
        ggml_tensor* d_q_cross = nullptr;
        ggml_tensor* dK_cross = nullptr;
        ggml_tensor* dV_cross = nullptr;
        attn_bwd(d_qfa_x, acts.q_cross_fa, memK, memV, &d_q_cross, &dK_cross,
                 &dV_cross);
        *dK_acc = *dK_acc ? ggml_add(ctx, *dK_acc, dK_cross) : dK_cross;
        *dV_acc = *dV_acc ? ggml_add(ctx, *dV_acc, dV_cross) : dV_cross;
        ggml_tensor* d_q4b = ggml_cont(
                ctx, ggml_permute(ctx, d_q_cross, 0, 2, 1, 3));  // {dh,nh,T}
        d_q4b = tto_lnback(ctx, d_q4b, acts.q_ln_in,
                           m.w((p + ".cross_qn.weight").c_str()), cs.one);
        ggml_tensor* d_qp = ggml_reshape_3d(ctx, d_q4b, C, Tq, 1);
        ggml_tensor* d_h_cross =
                tto_lin_bwd(ctx, m.w((p + ".cross_qp.weight").c_str()), d_qp);
        d_tok = ggml_add(
                ctx, d_tok,
                tto_lnback(ctx, d_h_cross, acts.tok_in,
                           m.w((p + ".cross_norm.weight").c_str()), cs.one));
        return d_tok;
    };

    // full forward chain builder shared by both graphs (graph B recomputes)
    auto fwd_chain = [&](ggml_tensor* memK, ggml_tensor* memV,
                         ggml_tensor* memPK, ggml_tensor* memPV,
                         std::vector<BlockActs>& acts, ggml_tensor** out_deconv,
                         ggml_tensor** out_color) {
        ggml_tensor* eye = m.w("eye");
        ggml_tensor* gsw = m.w("gst");
        ggml_tensor* tok = ggml_concat(
                ctx, eye, ggml_reshape_3d(ctx, gsw, C, Ngs, 1), 1);  // {C, T}
        const int T = Ngs + 1;
        acts.resize(m.cfg->dec_depth);
        for (int i = 0; i < m.cfg->dec_depth; i++) {
            block_fwd(tok, "dec" + std::to_string(i), memK, memV, T, &acts[i]);
            tok = acts[i].tok_out;
        }
        ggml_tensor* geom = ggml_cont(
                ctx,
                ggml_view_3d(ctx, tok, C, Ngs, 1, tok->nb[1], tok->nb[2],
                             (size_t)C * sizeof(float)));  // drop the eye row
        ggml_tensor* ctok = ggml_add(
                ctx, geom, ggml_reshape_3d(ctx, m.w("prgs"), C, Ngs, 1));
        const int pd = m.cfg->post_decoder_depth;
        acts.resize(m.cfg->dec_depth + pd);
        for (int i = 0; i < pd; i++) {
            block_fwd(ctok, "pdec" + std::to_string(i), memPK, memPV, Ngs,
                      &acts[m.cfg->dec_depth + i]);
            ctok = acts[m.cfg->dec_depth + i].tok_out;
        }
        *out_deconv =
                m.lin(ctx, geom, m.w("deconv.weight"), m.w("deconv.bias"));
        *out_color = m.lin(ctx, ctok, m.w("dh.weight"), m.w("dh.bias"));
    };

    // ---- graph F ----
    {
        std::vector<BlockActs> acts;
        ggml_tensor *od = nullptr, *oc = nullptr;
        fwd_chain(m.mem[0], m.mem[1], m.mem[2], m.mem[3], acts, &od, &oc);
        ggml_set_output(od);
        ggml_set_output(oc);
        m.tf_out_deconv = od;
        m.tf_out_color = oc;
        m.tf_graph = ggml_new_graph_custom(ctx, kTtoNodes, false);
        ggml_build_forward_expand(m.tf_graph, od);
        ggml_build_forward_expand(m.tf_graph, oc);
        m.tf_galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(m.be.handle));
        if (!ggml_gallocr_alloc_graph(m.tf_galloc, m.tf_graph)) {
            log_error("tto: graph F allocation failed");
            return false;
        }
    }
    // ---- graph B: re-forward + block backward + AdamW ----
    {
        std::vector<BlockActs> acts;
        ggml_tensor *od = nullptr, *oc = nullptr;
        fwd_chain(m.mem[0], m.mem[1], m.mem[2], m.mem[3], acts, &od, &oc);
        const int T = Ngs + 1;
        // head backward: gradient at the pdec chain OUTPUT (dh head input).
        ggml_tensor* d_ctok = tto_lin_bwd(ctx, m.w("dh.weight"), m.d_color_in);
        ggml_tensor* d_geom =
                tto_lin_bwd(ctx, m.w("deconv.weight"), m.d_deconv_in);
        // pdec backward: ctok has no eye row. The chain entry gradient is the
        // head gradient ALONE — the deconv head reads geom directly, so its
        // gradient joins d_geom only AFTER the loop (ctok = geom + prgs,
        // prgs frozen). Feeding g+h into the chain or adding h to d_geom here
        // contaminates both the memory gradients and the dec-chain entry.
        ggml_tensor* dKp = nullptr;
        ggml_tensor* dVp = nullptr;
        ggml_tensor* d_tok = d_ctok;
        for (int i = m.cfg->post_decoder_depth - 1; i >= 0; i--) {
            d_tok = block_bwd(d_tok, "pdec" + std::to_string(i),
                              acts[m.cfg->dec_depth + i], Ngs, m.mem[2],
                              m.mem[3], &dKp, &dVp);
        }
        // ctok = geom + prgs (prgs frozen): d_geom += d_tok
        d_geom = ggml_add(ctx, d_geom, d_tok);
        // dec backward: tok = concat(eye, geom) — the eye row's incoming
        // gradient is exactly zero (no head reads it). Build it as a scaled
        // view: a fresh graph-scratch tensor would hold stale/unset memory.
        ggml_tensor* zero_row = ggml_scale(
                ctx,
                ggml_cont(ctx, ggml_view_3d(ctx, d_geom, C, 1, 1, d_geom->nb[1],
                                            d_geom->nb[1], 0)),
                0.0f);
        ggml_tensor* d_tok_full = ggml_concat(ctx, zero_row, d_geom, 1);
        ggml_tensor* dK = nullptr;
        ggml_tensor* dV = nullptr;
        for (int i = m.cfg->dec_depth - 1; i >= 0; i--) {
            d_tok_full = block_bwd(d_tok_full, "dec" + std::to_string(i),
                                   acts[i], T, m.mem[0], m.mem[1], &dK, &dV);
        }
        // grads are {dh,N,nh} views into d via the accumulators' layout —
        // they already match the mem layout
        ggml_tensor* dmem[4] = {dK, dV, dKp, dVp};
        for (int i = 0; i < 4; i++) {
            if (!dmem[i] || dmem[i]->ne[0] != m.mem[i]->ne[0] ||
                dmem[i]->ne[1] != m.mem[i]->ne[1] ||
                dmem[i]->ne[2] != m.mem[i]->ne[2]) {
                log_error("tto: d_mem[%d] layout mismatch (%p)", i,
                          (void*)dmem[i]);
                return false;
            }
            m.probe_dmem[i] = dmem[i];  // fd parity anchor (graph B output)
            m.tb_out[i] =
                    ggml_opt_step_adamw(ctx, m.mem[i], dmem[i], m.adam_m[i],
                                        m.adam_v[i], m.adamw_params);
        }
        m.tb_graph = ggml_new_graph_custom(ctx, kTtoNodes, false);
        for (int i = 0; i < 4; i++) {
            // fd parity: keep the gradient tensors alive past the AdamW step
            ggml_set_output(m.probe_dmem[i]);
            ggml_build_forward_expand(m.tb_graph, m.probe_dmem[i]);
            ggml_build_forward_expand(m.tb_graph, m.tb_out[i]);
        }
        m.tb_galloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(m.be.handle));
        if (!ggml_gallocr_alloc_graph(m.tb_galloc, m.tb_graph)) {
            log_error("tto: graph B allocation failed");
            return false;
        }
        m.tto_active = true;
        log_info("tto: session built (S=%d T=%d Nv=%d)", S, m.tto_T, Nv);
        return true;
    }
}

bool QuerySplatDecoder::tto_mem_get(int idx, std::vector<float>& out) {
    Impl& m = *impl_;
    if (!m.tto_active || idx < 0 || idx > 3) return false;
    out.resize(ggml_nelements(m.mem[idx]));
    ggml_backend_tensor_get(m.mem[idx], out.data(), 0,
                            out.size() * sizeof(float));
    return true;
}

bool QuerySplatDecoder::tto_mem_set(int idx, const std::vector<float>& in) {
    Impl& m = *impl_;
    if (!m.tto_active || idx < 0 || idx > 3) return false;
    if (in.size() != (size_t)ggml_nelements(m.mem[idx])) return false;
    ggml_backend_tensor_set(m.mem[idx], in.data(), 0,
                            in.size() * sizeof(float));
    return true;
}

void QuerySplatDecoder::tto_end() {
    Impl& m = *impl_;
    if (!m.tto_active) return;
    if (m.tf_galloc) ggml_gallocr_free(m.tf_galloc);
    if (m.tb_galloc) ggml_gallocr_free(m.tb_galloc);
    if (m.tbuf) ggml_backend_buffer_free(m.tbuf);
    if (m.tbuf_ctx) ggml_free(m.tbuf_ctx);
    if (m.tgalloc) ggml_gallocr_free(m.tgalloc);
    if (m.tctx) ggml_free(m.tctx);
    m.tf_galloc = m.tb_galloc = nullptr;
    m.tbuf = nullptr;
    m.tbuf_ctx = nullptr;
    m.tgalloc = nullptr;
    m.tctx = nullptr;
    m.tf_graph = m.tb_graph = nullptr;
    m.tto_active = false;
}

bool QuerySplatDecoder::tto_forward(std::vector<float>& deconv_raw,
                                    std::vector<float>& color_raw) {
    Impl& m = *impl_;
    if (!m.tto_active) return false;
    if (!ggml_gallocr_alloc_graph(m.tf_galloc, m.tf_graph)) return false;
    if (ggml_backend_graph_compute(m.be.handle, m.tf_graph) !=
        GGML_STATUS_SUCCESS)
        return false;
    auto fetch = [&](ggml_tensor* t, std::vector<float>& dst) {
        dst.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    };
    fetch(m.tf_out_deconv, deconv_raw);
    fetch(m.tf_out_color, color_raw);
    return true;
}

bool QuerySplatDecoder::tto_backward(const float* d_deconv,
                                     const float* d_color,
                                     int step,
                                     std::vector<float>* dmem_out[4]) {
    Impl& m = *impl_;
    if (!m.tto_active) return false;
    ggml_backend_tensor_set(m.d_deconv_in, d_deconv, 0,
                            ggml_nbytes(m.d_deconv_in));
    ggml_backend_tensor_set(m.d_color_in, d_color, 0,
                            ggml_nbytes(m.d_color_in));
    // AdamW schedule: [lr, beta1, beta2, eps, wd, beta1h, beta2h]; the
    // bias-correction terms follow torch.optim.AdamW with the official
    // defaults (beta1 0.9, beta2 0.999, eps 1e-8, wd 0.01? — the official
    // chain uses torch defaults except lr; wd = 0 (only the memory trains,
    // torch.optim.Adam(memory, lr) has NO weight decay))
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const float b1h = 1.0f / (1.0f - std::pow(b1, (float)(step + 1)));
    const float b2h = 1.0f / (1.0f - std::pow(b2, (float)(step + 1)));
    const float params[7] = {m.lr_cur, b1, b2, eps, 0.0f, b1h, b2h};
    ggml_backend_tensor_set(m.adamw_params, params, 0, sizeof(params));
    if (!ggml_gallocr_alloc_graph(m.tb_galloc, m.tb_graph)) return false;
    if (ggml_backend_graph_compute(m.be.handle, m.tb_graph) !=
        GGML_STATUS_SUCCESS)
        return false;
    // TEMP-DIAG(NaN-bisect): mem/params state ON ENTRY to graph B
    {
        std::vector<float> tm0(ggml_nelements(m.mem[0]));
        ggml_backend_tensor_get(m.mem[0], tm0.data(), 0,
                                tm0.size() * sizeof(float));
        double m0 = 0;
        bool mnan = false;
        for (float x : tm0) {
            if (!std::isfinite(x)) mnan = true;
            m0 += x;
        }
        float ap[7];
        ggml_backend_tensor_get(m.adamw_params, ap, 0, sizeof(ap));
        log_info(
                "tto-entry: mem0_sum=%.6f mem0_nan=%d "
                "adamw_params=[%g,%g,%g,%g,%g,%g,%g]",
                mnan ? 0.0 : m0, mnan ? 1 : 0, ap[0], ap[1], ap[2], ap[3],
                ap[4], ap[5], ap[6]);
    }
    // TEMP-DIAG(NaN-bisect): report the FIRST graph-B node whose output is
    // non-finite (topological order => the origin). Remove after the fix.
    for (int ni = 0; ni < ggml_graph_n_nodes(m.tb_graph); ni++) {
        ggml_tensor* node = ggml_graph_node(m.tb_graph, ni);
        if (node->type != GGML_TYPE_F32 || ggml_nelements(node) == 0) continue;
        std::vector<float> v(ggml_nelements(node));
        ggml_backend_tensor_get(node, v.data(), 0, v.size() * sizeof(float));
        bool bad = false;
        for (float x : v) {
            if (!std::isfinite(x)) {
                bad = true;
                break;
            }
        }
        if (!bad) continue;
        auto src_state = [](ggml_tensor* t) {
            if (!t) return std::string("none");
            if (t->type != GGML_TYPE_F32) return std::string("non-f32");
            std::vector<float> sv(ggml_nelements(t));
            ggml_backend_tensor_get(t, sv.data(), 0, sv.size() * sizeof(float));
            for (float x : sv)
                if (!std::isfinite(x)) return std::string("NONFINITE");
            return std::string("finite");
        };
        log_error(
                "tto-diag: first non-finite node #%d op=%s name=%s "
                "ne=%lldx%lldx%lldx%lld src0=%s src1=%s src2=%s",
                ni, ggml_op_name(node->op),
                node->name[0] ? node->name : "(unnamed)",
                (long long)node->ne[0], (long long)node->ne[1],
                (long long)node->ne[2], (long long)node->ne[3],
                src_state(node->src[0]).c_str(),
                src_state(node->src[1]).c_str(),
                src_state(node->src[2]).c_str());
        break;
    }
    // Safety invariant: the TTO memory must stay finite. A non-finite
    // gradient here means the decoder-backward graph misbehaves on this
    // ggml build (upstream t4 passes on ggml 0.24; a 0.24 -> 0.27 primitive
    // behavior change produces NaN in the hand-written backward). Failing
    // loudly beats silently optimizing NaN memory (which renders the
    // background forever — the observed "frozen loss" symptom).
    for (int i = 0; i < 4; i++) {
        std::vector<float> probe(ggml_nelements(m.mem[i]));
        ggml_backend_tensor_get(m.mem[i], probe.data(), 0,
                                probe.size() * sizeof(float));
        for (float x : probe) {
            if (!std::isfinite(x)) {
                log_error(
                        "tto: non-finite memory after the AdamW step "
                        "(decoder-backward graph produced NaN gradients "
                        "on this ggml build); failing the TTO session");
                return false;
            }
        }
    }
    if (dmem_out) {
        for (int i = 0; i < 4; i++) {
            dmem_out[i]->resize(ggml_nelements(m.probe_dmem[i]));
            ggml_backend_tensor_get(m.probe_dmem[i], dmem_out[i]->data(), 0,
                                    dmem_out[i]->size() * sizeof(float));
        }
    }
    return true;
}
bool QuerySplatDecoder::tto_gaussians(DecoderOutputs& out) {
    Impl& m = *impl_;
    if (!m.tto_active) return false;
    std::vector<float> deconv_raw(ggml_nelements(m.tf_out_deconv));
    std::vector<float> color_raw(ggml_nelements(m.tf_out_color));
    ggml_backend_tensor_get(m.tf_out_deconv, deconv_raw.data(), 0,
                            deconv_raw.size() * sizeof(float));
    ggml_backend_tensor_get(m.tf_out_color, color_raw.data(), 0,
                            color_raw.size() * sizeof(float));
    out = m.activate_final(deconv_raw, color_raw, (int64_t)m.Ngs * m.gpt);
    return true;
}

DecoderOutputs QuerySplatDecoder::tto_gaussians_from(
        const std::vector<float>& deconv_raw,
        const std::vector<float>& color_raw) {
    Impl& m = *impl_;
    return m.activate_final(deconv_raw, color_raw, (int64_t)m.Ngs * m.gpt);
}

}  // namespace qs
