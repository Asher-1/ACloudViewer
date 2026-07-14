#include "dpt_head.hpp"
#include "dpt_blocks.hpp"
#include "uv_posembed.hpp"
#include "ggml_extend.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

namespace da {

// Build the *0.1-scaled UV positional embedding as a ggml graph input in
// [W,H,C] (w fastest, then h, then c) memory layout, matching feature maps.
// uv_pos_embed returns (y,x,c) flat = (y*W + x)*C + c (RAW, no ratio).
static ggml_tensor* add_uv_input(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                                 int W, int H, int C, float aspect, float ratio) {
    // The UV positional embedding is input-INDEPENDENT: it depends only on the
    // feature-map geometry (W,H,C,aspect,ratio), so it is identical on every
    // forward at a given resolution. Recomputing it (scalar sin/cos over W*H*C;
    // the full-res call alone is ~90ms single-threaded) on every forward dominated
    // graph build. Cache the finished [C,H,W] buffer; recompute only on new geometry.
    static std::mutex uv_mu;
    static std::map<std::tuple<int,int,int,uint32_t,uint32_t>, std::vector<float>> uv_cache;
    uint32_t ab, rb; std::memcpy(&ab, &aspect, 4); std::memcpy(&rb, &ratio, 4);
    const auto key = std::make_tuple(W, H, C, ab, rb);
    std::lock_guard<std::mutex> lk(uv_mu);
    auto it = uv_cache.find(key);
    if (it == uv_cache.end()) {
        std::vector<float> uv = uv_pos_embed(/*pw=*/W, /*ph=*/H, C, aspect);
        std::vector<float> buf((size_t)W * H * C);
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    buf[(size_t)c * H * W + (size_t)h * W + w] =
                        ratio * uv[((size_t)h * W + w) * C + c];
        it = uv_cache.emplace(key, std::move(buf)).first;
    }
    const int64_t ne[4] = { W, H, C, 1 };
    // Borrow the cached buffer (it outlives this compute) -> skip the per-forward
    // memcpy of the full-res UV embedding (~tens of MB) into the upload pool.
    (void)pool;
    return be.add_graph_input_nd_borrow(ctx, it->second.data(), ne, 4);
}

// Channels-last LayerNorm over the channel dim (ne2) of a feature map [W,H,C,N],
// matching the reference Permute(NCHW->NHWC) -> LayerNorm(C) -> Permute(NHWC->NCHW).
// ggml_norm normalizes over ne0, so bring C to ne0, normalize+affine, then restore.
// w,b are [C] vectors (broadcast over the W,H dims). eps = torch LayerNorm default.
static ggml_tensor* layernorm_channels_last(ggml_context* ctx, ggml_tensor* x,
                                            ggml_tensor* w, ggml_tensor* b, float eps) {
    // [W,H,C,N] -> [C,W,H,N]
    ggml_tensor* y = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    y = ggml_norm(ctx, y, eps);          // over ne0 = C
    y = ggml_mul(ctx, y, w);             // affine scale ([C] broadcasts over W,H)
    if (b) y = ggml_add(ctx, y, b);
    // [C,W,H,N] -> [W,H,C,N]
    y = ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
    return y;
}

// Shared DPT depth graph builder. Consumes the 4 backbone out-layer feat tensors
// (each [C=2*embed (or embed), N=Npatch], ne0=channel fastest) — whether uploaded
// as graph inputs (unfused run()) or produced in-graph by the backbone (fused
// single-image path) — and returns the logits tensor [W,H,output_dim,1]. Optional
// capture dests: stage_caps (array[4]) post-resize stages, fused_cap post-conv1,
// sky_cap metric sky head; nullptr to skip. Math is identical to the old run()
// build lambda, so depth/conf are unchanged.
ggml_tensor* DptHead::build_depth_graph(ggml_context* ctx, ggml_tensor* const feat[4],
                                        int H, int W, GraphInputPool& pool,
                                        std::vector<float>* stage_caps,
                                        std::vector<float>* fused_cap,
                                        std::vector<float>* sky_cap,
                                        std::vector<float>* aux_cap) {
    const Config& cfg = ml_.config();
    const int patch = (int)cfg.patch_size;
    const int pw = W / patch, ph = H / patch;
    const int C = (cfg.cat_token ? 2 : 1) * (int)cfg.embed_dim;
    int oc[4] = { 96, 192, 384, 768 };
    if (cfg.head_out_channels.size() == 4)
        for (int s = 0; s < 4; ++s) oc[s] = cfg.head_out_channels[s];
    const int feat_half = cfg.head_features ? (int)cfg.head_features / 2 : 64;
    const float aspect = (float)W / (float)H;
    const float ratio = 0.1f;
    const float eps = 1e-5f;                          // head.norm is nn.LayerNorm default
    auto t = [&](const std::string& n) { return ml_.tensor(n); };
    const bool has_head_norm = t("head.norm.weight") != nullptr;
    const bool want_sky = sky_cap && t("head.scratch.sky_out2b.weight") != nullptr;
    ggml_tensor* norm_w = has_head_norm ? t("head.norm.weight") : nullptr;
    ggml_tensor* norm_b = has_head_norm ? t("head.norm.bias")   : nullptr;

    ggml_tensor* l[4];
    for (int s = 0; s < 4; ++s) {
        // feat[s] = [C, N=256] (ne0=channel fastest, ne1=token)
        ggml_tensor* x = feat[s];
        // LayerNorm over channel dim (ne0); skipped when norm_type is "idt".
        if (has_head_norm) x = layernorm(ctx, x, norm_w, norm_b, eps);
        // [C,N] -> [N,C] -> [W,H,C,1]  (token = h*pw + w => ne0=w, ne1=h)
        x = ggml_cont(ctx, ggml_transpose(ctx, x));      // [N,C]
        x = ggml_reshape_4d(ctx, x, pw, ph, C, 1);       // [W,H,C,1]
        // projects[s]: 1x1 conv 1536 -> oc[s]
        x = conv2d(ctx, t("head.proj." + std::to_string(s) + ".weight"),
                   t("head.proj." + std::to_string(s) + ".bias"), x, 1, 0);
        // + 0.1 * UV pos-embed at [pw,ph,oc] (only when the head uses pos_embed;
        // the metric DPT has pos_embed=False -> skip).
        if (cfg.head_pos_embed) {
            ggml_tensor* pe = add_uv_input(ctx, be_, pool, pw, ph, oc[s], aspect, ratio);
            x = ggml_add(ctx, x, pe);
        }
        // resize_layers[s]
        if (s == 0)
            x = conv_transpose2d_p0(ctx, t("head.resize.0.weight"),
                                    t("head.resize.0.bias"), x, 4);   // ->64x64
        else if (s == 1)
            x = conv_transpose2d_p0(ctx, t("head.resize.1.weight"),
                                    t("head.resize.1.bias"), x, 2);   // ->32x32
        else if (s == 3)
            x = conv2d(ctx, t("head.resize.3.weight"),
                       t("head.resize.3.bias"), x, 2, 1);            // ->8x8
        // s==2 Identity
        l[s] = x;
        if (stage_caps) be_.capture(x, &stage_caps[s]);
    }

    // _fuse: layer{i}_rn = 3x3 pad1 conv (NO bias), out_channels[i-1] -> 128
    ggml_tensor* l1_rn = conv2d(ctx, t("head.scratch.layer1_rn.weight"), nullptr, l[0], 1, 1);
    ggml_tensor* l2_rn = conv2d(ctx, t("head.scratch.layer2_rn.weight"), nullptr, l[1], 1, 1);
    ggml_tensor* l3_rn = conv2d(ctx, t("head.scratch.layer3_rn.weight"), nullptr, l[2], 1, 1);
    ggml_tensor* l4_rn = conv2d(ctx, t("head.scratch.layer4_rn.weight"), nullptr, l[3], 1, 1);

    // Fusion target sizes are the lateral-skip spatial sizes, derived from the
    // patch grid (DPT._fuse: size=l{i}_rn.shape[2:]). The resize layers put the
    // skips at l3=(pw,ph) [identity], l2=(2pw,2ph) [stride-2 transpose], l1=
    // (4pw,4ph) [stride-4 transpose]. At the 224 square fixture this is the old
    // 16/32/64; at native non-square it tracks pw,ph (e.g. 504x336 -> 36,24).
    // refinenet4 (no residual / no rc1): top=l4_rn, size=l3 (pw,ph)
    ggml_tensor* out = feature_fusion(ctx, l4_rn, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        t("head.scratch.rn4.rc2.c1.weight"), t("head.scratch.rn4.rc2.c1.bias"),
        t("head.scratch.rn4.rc2.c2.weight"), t("head.scratch.rn4.rc2.c2.bias"),
        t("head.scratch.rn4.out.weight"), t("head.scratch.rn4.out.bias"), pw, ph);
    // refinenet3: lateral=l3_rn, size=l2 (2pw,2ph)
    out = feature_fusion(ctx, out, l3_rn,
        t("head.scratch.rn3.rc1.c1.weight"), t("head.scratch.rn3.rc1.c1.bias"),
        t("head.scratch.rn3.rc1.c2.weight"), t("head.scratch.rn3.rc1.c2.bias"),
        t("head.scratch.rn3.rc2.c1.weight"), t("head.scratch.rn3.rc2.c1.bias"),
        t("head.scratch.rn3.rc2.c2.weight"), t("head.scratch.rn3.rc2.c2.bias"),
        t("head.scratch.rn3.out.weight"), t("head.scratch.rn3.out.bias"), 2*pw, 2*ph);
    // refinenet2: lateral=l2_rn, size=l1 (4pw,4ph)
    out = feature_fusion(ctx, out, l2_rn,
        t("head.scratch.rn2.rc1.c1.weight"), t("head.scratch.rn2.rc1.c1.bias"),
        t("head.scratch.rn2.rc1.c2.weight"), t("head.scratch.rn2.rc1.c2.bias"),
        t("head.scratch.rn2.rc2.c1.weight"), t("head.scratch.rn2.rc2.c1.bias"),
        t("head.scratch.rn2.rc2.c2.weight"), t("head.scratch.rn2.rc2.c2.bias"),
        t("head.scratch.rn2.out.weight"), t("head.scratch.rn2.out.bias"), 4*pw, 4*ph);
    // refinenet1: lateral=l1_rn, scale_factor 2 -> 128x128
    out = feature_fusion(ctx, out, l1_rn,
        t("head.scratch.rn1.rc1.c1.weight"), t("head.scratch.rn1.rc1.c1.bias"),
        t("head.scratch.rn1.rc1.c2.weight"), t("head.scratch.rn1.rc1.c2.bias"),
        t("head.scratch.rn1.rc2.c1.weight"), t("head.scratch.rn1.rc2.c1.bias"),
        t("head.scratch.rn1.rc2.c2.weight"), t("head.scratch.rn1.rc2.c2.bias"),
        t("head.scratch.rn1.out.weight"), t("head.scratch.rn1.out.bias"), 0, 0);
    // output_conv1: 3x3 pad1, 128 -> 64
    out = conv2d(ctx, t("head.scratch.out1.weight"), t("head.scratch.out1.bias"), out, 1, 1);
    if (fused_cap) be_.capture(out, fused_cap);

    // upsample to (H,W), + 0.1*UV(64). This shared feature `feat_map` drives BOTH the
    // main head and (metric) the parallel sky head (DPT._forward_impl: feat=fused).
    out = interp_bilinear_ac(ctx, out, W, H);             // [W,H,feat_half]
    ggml_tensor* feat_map = out;
    if (cfg.head_pos_embed) {
        ggml_tensor* pe2 = add_uv_input(ctx, be_, pool, W, H, feat_half, aspect, ratio);
        feat_map = ggml_add(ctx, out, pe2);
    }
    // Sky head (metric): conv feat/2->32 (3x3 pad1) -> relu -> conv 32->1 (1x1).
    if (want_sky) {
        ggml_tensor* sk = conv2d(ctx, t("head.scratch.sky_out2a.weight"),
                                 t("head.scratch.sky_out2a.bias"), feat_map, 1, 1);
        sk = ggml_relu(ctx, sk);
        sk = conv2d(ctx, t("head.scratch.sky_out2b.weight"),
                    t("head.scratch.sky_out2b.bias"), sk, 1, 0);
        be_.capture(sk, sky_cap);
    }
    // output_conv2: conv feat/2->32 (3x3 pad1) -> relu -> conv 32->output_dim (1x1)
    out = conv2d(ctx, t("head.scratch.out2a.weight"), t("head.scratch.out2a.bias"), feat_map, 1, 1);
    out = ggml_relu(ctx, out);
    out = conv2d(ctx, t("head.scratch.out2b.weight"), t("head.scratch.out2b.bias"), out, 1, 0);

    // ===== DualDPT AUXILIARY ray head (opt-in; present only in --with-aux GGUF) =====
    // A fully independent pyramid sharing only l1_rn..l4_rn with the main path
    // (DualDPT._fuse / _forward_impl, aux branch). Only the finest level is used.
    // refinenet4_aux has_residual=False (no rc1). Mirrors the main refinenet sizes.
    if (aux_cap && t("head.scratch.rn1_aux.out.weight")) {
        ggml_tensor* a = feature_fusion(ctx, l4_rn, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            t("head.scratch.rn4_aux.rc2.c1.weight"), t("head.scratch.rn4_aux.rc2.c1.bias"),
            t("head.scratch.rn4_aux.rc2.c2.weight"), t("head.scratch.rn4_aux.rc2.c2.bias"),
            t("head.scratch.rn4_aux.out.weight"), t("head.scratch.rn4_aux.out.bias"), pw, ph);
        a = feature_fusion(ctx, a, l3_rn,
            t("head.scratch.rn3_aux.rc1.c1.weight"), t("head.scratch.rn3_aux.rc1.c1.bias"),
            t("head.scratch.rn3_aux.rc1.c2.weight"), t("head.scratch.rn3_aux.rc1.c2.bias"),
            t("head.scratch.rn3_aux.rc2.c1.weight"), t("head.scratch.rn3_aux.rc2.c1.bias"),
            t("head.scratch.rn3_aux.rc2.c2.weight"), t("head.scratch.rn3_aux.rc2.c2.bias"),
            t("head.scratch.rn3_aux.out.weight"), t("head.scratch.rn3_aux.out.bias"), 2*pw, 2*ph);
        a = feature_fusion(ctx, a, l2_rn,
            t("head.scratch.rn2_aux.rc1.c1.weight"), t("head.scratch.rn2_aux.rc1.c1.bias"),
            t("head.scratch.rn2_aux.rc1.c2.weight"), t("head.scratch.rn2_aux.rc1.c2.bias"),
            t("head.scratch.rn2_aux.rc2.c1.weight"), t("head.scratch.rn2_aux.rc2.c1.bias"),
            t("head.scratch.rn2_aux.rc2.c2.weight"), t("head.scratch.rn2_aux.rc2.c2.bias"),
            t("head.scratch.rn2_aux.out.weight"), t("head.scratch.rn2_aux.out.bias"), 4*pw, 4*ph);
        a = feature_fusion(ctx, a, l1_rn,
            t("head.scratch.rn1_aux.rc1.c1.weight"), t("head.scratch.rn1_aux.rc1.c1.bias"),
            t("head.scratch.rn1_aux.rc1.c2.weight"), t("head.scratch.rn1_aux.rc1.c2.bias"),
            t("head.scratch.rn1_aux.rc2.c1.weight"), t("head.scratch.rn1_aux.rc2.c1.bias"),
            t("head.scratch.rn1_aux.rc2.c2.weight"), t("head.scratch.rn1_aux.rc2.c2.bias"),
            t("head.scratch.rn1_aux.out.weight"), t("head.scratch.rn1_aux.out.bias"), 0, 0);
        // output_conv1_aux[last]: 5 sequential 3x3 pad1 convs (128->64->128->64->128->64),
        // NO activations between (DualDPT._make_aux_out1_block, aux_out1_conv_num=5).
        for (int i = 0; i < 5; ++i) {
            std::string p = "head.scratch.out1_aux." + std::to_string(i) + ".";
            a = conv2d(ctx, t(p + "weight"), t(p + "bias"), a, 1, 1);
        }
        // pos_embed on the finest aux feature (64ch) at its own resolution.
        if (cfg.head_pos_embed) {
            const int aw = (int)a->ne[0], ah = (int)a->ne[1], ac = (int)a->ne[2];
            ggml_tensor* pea = add_uv_input(ctx, be_, pool, aw, ah, ac, aspect, ratio);
            a = ggml_add(ctx, a, pea);
        }
        // output_conv2_aux[last]: Conv 64->32 (3x3 pad1) -> channels-last LayerNorm(32)
        // -> ReLU -> Conv 32->7 (1x1). The LN is the single instance shared by all
        // aux levels (out2_aux_ln).
        a = conv2d(ctx, t("head.scratch.out2a_aux.weight"), t("head.scratch.out2a_aux.bias"), a, 1, 1);
        a = layernorm_channels_last(ctx, a, t("head.scratch.out2_aux_ln.weight"),
                                    t("head.scratch.out2_aux_ln.bias"), eps);
        a = ggml_relu(ctx, a);
        a = conv2d(ctx, t("head.scratch.out2b_aux.weight"), t("head.scratch.out2b_aux.bias"), a, 1, 0);
        be_.capture(a, aux_cap);                          // [W,H,7,1] aux logits
    }
    return out;                                            // [W,H,output_dim,1] logits
}

bool DptHead::run(const std::vector<std::vector<float>>& feats, int H, int W,
                  std::vector<float>& depth_out, std::vector<float>& conf_out,
                  std::vector<std::vector<float>>* stages, std::vector<float>* fused,
                  std::vector<float>* sky_out) {
    if (feats.size() != 4) return false;
    const Config& cfg = ml_.config();
    const int patch = (int)cfg.patch_size;           // 14
    const int pw = W / patch, ph = H / patch;        // 16,16
    const int N = ph * pw;                            // 256
    // dim_in: cat_token true -> cat([local,norm]) = 2*embed (base 1536, giant 3072);
    // cat_token false (metric ViT-L) -> norm(x) only = embed (1024).
    const int C = (cfg.cat_token ? 2 : 1) * (int)cfg.embed_dim;

    for (const auto& f : feats)
        if ((int)f.size() != N * C) return false;

    auto t = [&](const std::string& n) { return ml_.tensor(n); };

    // output_dim from out2b out-channels (base 2 = depth+conf; metric 1 = depth only).
    ggml_tensor* out2b_w = t("head.scratch.out2b.weight");
    const int output_dim = out2b_w ? (int)out2b_w->ne[3] : 2;
    // sky head present only on the metric DPT (norm_type idt, single depth head).
    const bool want_sky = sky_out && t("head.scratch.sky_out2b.weight") != nullptr;

    std::vector<float> stage_caps[4];
    std::vector<float> fused_cap;
    std::vector<float> sky_cap;

    GraphInputPool pool;
    std::vector<float> logits;
    bool ok = be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        // Upload the 4 backbone feats as graph inputs [C, N] (ne0=channel fastest,
        // ne1=token), then build the shared depth graph from them.
        ggml_tensor* feat[4];
        for (int s = 0; s < 4; ++s) {
            const int64_t fne[2] = { C, N };
            feat[s] = be_.add_graph_input_nd(ctx, pool, feats[s].data(), fne, 2);
        }
        return build_depth_graph(ctx, feat, H, W, pool,
                                 stages  ? stage_caps : nullptr,
                                 fused   ? &fused_cap : nullptr,
                                 sky_out ? &sky_cap   : nullptr);
    }, logits);
    if (!ok) return false;

    const size_t HW = (size_t)H * W;
    if (logits.size() != (size_t)output_dim * HW) return false;
    // channel 0 = depth = exp(logits); channel 1 (if present) = conf = exp(logits)+1.
    depth_out.resize(HW);
    for (size_t i = 0; i < HW; ++i) depth_out[i] = std::exp(logits[i]);
    if (output_dim >= 2) {
        conf_out.resize(HW);
        for (size_t i = 0; i < HW; ++i) conf_out[i] = std::exp(logits[HW + i]) + 1.0f;
    } else {
        conf_out.clear();
    }
    if (want_sky) {
        if (sky_cap.size() != HW) return false;
        sky_out->resize(HW);
        for (size_t i = 0; i < HW; ++i) sky_out->operator[](i) = std::max(0.0f, sky_cap[i]); // relu
    }
    if (stages) {
        stages->resize(4);
        for (int s = 0; s < 4; ++s) (*stages)[s] = std::move(stage_caps[s]);
    }
    if (fused) *fused = std::move(fused_cap);
    return true;
}

bool DptHead::rays(const std::vector<std::vector<float>>& feats, int H, int W,
                   std::vector<float>& ray_out, std::vector<float>& ray_conf_out,
                   int& ray_h, int& ray_w) {
    if (feats.size() != 4) return false;
    const Config& cfg = ml_.config();
    const int patch = (int)cfg.patch_size;
    const int pw = W / patch, ph = H / patch;
    const int N = ph * pw;
    const int C = (cfg.cat_token ? 2 : 1) * (int)cfg.embed_dim;
    for (const auto& f : feats)
        if ((int)f.size() != N * C) return false;
    if (!ml_.tensor("head.scratch.rn1_aux.out.weight")) return false;  // not an aux GGUF

    // aux head resolution: refinenet1_aux upsamples the (4pw,4ph) level by 2.
    ray_w = 8 * pw;
    ray_h = 8 * ph;
    const size_t HWa = (size_t)ray_h * ray_w;

    std::vector<float> aux_cap;
    std::vector<float> logits;   // main logits (unused, but build_depth_graph returns it)
    GraphInputPool pool;
    bool ok = be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* feat[4];
        for (int s = 0; s < 4; ++s) {
            const int64_t fne[2] = { C, N };
            feat[s] = be_.add_graph_input_nd(ctx, pool, feats[s].data(), fne, 2);
        }
        return build_depth_graph(ctx, feat, H, W, pool, nullptr, nullptr, nullptr, &aux_cap);
    }, logits);
    if (!ok) return false;
    if (aux_cap.size() != 7 * HWa) return false;          // [W,H,7] ggml layout

    // ggml aux_cap layout: index = c*HWa + h*ray_w + w (w fastest, c slowest).
    // ray (reference order): (h*ray_w + w)*6 + c, identity activation.
    // ray_conf: h*ray_w + w, conf_activation expp1 = exp(x)+1.
    ray_out.resize(6 * HWa);
    ray_conf_out.resize(HWa);
    for (size_t p = 0; p < HWa; ++p) {
        for (int c = 0; c < 6; ++c)
            ray_out[p * 6 + c] = aux_cap[(size_t)c * HWa + p];
        ray_conf_out[p] = std::exp(aux_cap[(size_t)6 * HWa + p]) + 1.0f;
    }
    return true;
}

bool DptHead::depth(const std::vector<std::vector<float>>& feats, int H, int W,
                    std::vector<float>& depth_out, std::vector<float>& conf_out) {
    return run(feats, H, W, depth_out, conf_out, nullptr, nullptr);
}

bool DptHead::depth_sky(const std::vector<std::vector<float>>& feats, int H, int W,
                        std::vector<float>& depth_out, std::vector<float>& sky_out) {
    std::vector<float> conf_unused;
    return run(feats, H, W, depth_out, conf_unused, nullptr, nullptr, &sky_out);
}

bool DptHead::depth_relative(const std::vector<std::vector<float>>& feats, int H, int W,
                             float max_depth, std::vector<float>& depth_out) {
    if (feats.size() != 4) return false;
    const Config& cfg = ml_.config();
    const int pw = W / (int)cfg.patch_size, ph = H / (int)cfg.patch_size;
    const int N = ph * pw;
    const int C = (cfg.cat_token ? 2 : 1) * (int)cfg.embed_dim;
    for (const auto& f : feats) if ((int)f.size() != N * C) return false;

    GraphInputPool pool;
    std::vector<float> logits;
    bool ok = be_.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* feat[4];
        for (int sidx = 0; sidx < 4; ++sidx) {
            const int64_t fne[2] = { C, N };
            feat[sidx] = be_.add_graph_input_nd(ctx, pool, feats[sidx].data(), fne, 2);
        }
        return build_depth_graph(ctx, feat, H, W, pool, nullptr, nullptr, nullptr);
    }, logits);
    if (!ok) return false;

    const size_t HW = (size_t)H * W;
    if (logits.size() != HW) return false;                 // output_dim == 1
    // The build_depth_graph output stops at output_conv2's final conv (the trailing
    // activation is not a tensor in our graph), so we apply it here. The two DA2 heads
    // differ in that activation:
    //   relative: output_conv2 ends ReLU->Identity, then outer F.relu  -> relu(logit)
    //   metric:   output_conv2 ends Sigmoid, then depth * max_depth     -> sigmoid(logit)*max_depth
    depth_out.resize(HW);
    if (max_depth > 0.f) {                                  // metric
        for (size_t i = 0; i < HW; ++i)
            depth_out[i] = (1.0f / (1.0f + std::exp(-logits[i]))) * max_depth;
    } else {                                                // relative
        for (size_t i = 0; i < HW; ++i)
            depth_out[i] = std::max(0.0f, logits[i]);
    }
    return true;
}

bool DptHead::depth_debug(const std::vector<std::vector<float>>& feats, int H, int W,
                          std::vector<float>& depth_out, std::vector<float>& conf_out,
                          std::vector<std::vector<float>>& stages, std::vector<float>& fused) {
    return run(feats, H, W, depth_out, conf_out, &stages, &fused);
}

} // namespace da
