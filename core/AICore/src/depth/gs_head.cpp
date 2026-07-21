// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "gs_head.hpp"

#include <cmath>
#include <string>

#include "common.hpp"
#include "dpt_blocks.hpp"
#include "ggml_extend.hpp"
#include "uv_posembed.hpp"

namespace aicore {
namespace depth {

// *0.1-scaled UV positional embedding as a ggml graph input in [W,H,C] (w
// fastest, then h, then c) memory layout, matching feature maps. (Same helper
// as in dpt_head.cpp.)
static ggml_tensor* gs_add_uv_input(ggml_context* ctx,
                                    Backend& be,
                                    GraphInputPool& pool,
                                    int W,
                                    int H,
                                    int C,
                                    float aspect,
                                    float ratio) {
    std::vector<float> uv = uv_pos_embed(/*pw=*/W, /*ph=*/H, C, aspect);
    std::vector<float> buf((size_t)W * H * C);
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                buf[(size_t)c * H * W + (size_t)h * W + w] =
                        ratio * uv[((size_t)h * W + w) * C + c];
    const int64_t ne[4] = {W, H, C, 1};
    return be.add_graph_input_nd(ctx, pool, buf.data(), ne, 4);
}

bool GsHead::raw_gaussians(const std::vector<std::vector<float>>& feats,
                           const std::vector<float>& image_chw,
                           int H,
                           int W,
                           std::vector<float>& raw_gs,
                           std::vector<float>& gs_conf) {
    if (feats.size() != 4) return false;
    const Config& cfg = ml_.config();
    const int patch = (int)cfg.patch_size;     // 14
    const int pw = W / patch, ph = H / patch;  // 16,16
    const int N = ph * pw;                     // 256
    const int C = 2 * (int)cfg.embed_dim;      // dim_in = 3072 (giant)
    // GSDPT fixed config (gs.features=256, gs.out_channels=[256,512,1024,1024],
    // output_dim=38). norm_type="idt" -> NO LayerNorm at the input.
    const int oc[4] = {256, 512, 1024, 1024};
    const int feat_half = 128;                 // gs.features/2
    const int out_dim = 38;                    // 37 xyz + 1 conf
    const float aspect = (float)W / (float)H;  // 1.0
    const float ratio = 0.1f;

    for (const auto& f : feats)
        if ((int)f.size() != N * C) return false;
    if ((int)image_chw.size() != 3 * H * W) return false;

    auto t = [&](const std::string& n) { return ml_.tensor(n); };

    if (!t("gs.proj.0.weight") || !t("gs.scratch.rn4.out.weight") ||
        !t("gs.scratch.out2b.weight")) {
        DA_ERR("GsHead: model has no gs.* weights (GIANT model required)");
        return false;
    }

    GraphInputPool pool;
    std::vector<float> logits;
    bool ok = be_.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* l[4];
                for (int s = 0; s < 4; ++s) {
                    // feat input [C=3072, N=256] (ne0=channel fastest,
                    // ne1=token).
                    const int64_t fne[2] = {C, N};
                    ggml_tensor* x = be_.add_graph_input_nd(
                            ctx, pool, feats[s].data(), fne, 2);
                    // norm = Identity (norm_type="idt") -> no LayerNorm.
                    // [C,N] -> [N,C] -> [W,H,C,1]  (token = h*pw + w => ne0=w,
                    // ne1=h)
                    x = ggml_cont(ctx, ggml_transpose(ctx, x));  // [N,C]
                    x = ggml_reshape_4d(ctx, x, pw, ph, C, 1);   // [W,H,C,1]
                    // projects[s]: 1x1 conv 3072 -> oc[s]
                    x = conv2d(ctx,
                               t("gs.proj." + std::to_string(s) + ".weight"),
                               t("gs.proj." + std::to_string(s) + ".bias"), x,
                               1, 0);
                    // + 0.1 * UV pos-embed at [pw,ph,oc]
                    ggml_tensor* pe = gs_add_uv_input(ctx, be_, pool, pw, ph,
                                                      oc[s], aspect, ratio);
                    x = ggml_add(ctx, x, pe);
                    // resize_layers[s]
                    if (s == 0)
                        x = conv_transpose2d_p0(ctx, t("gs.resize.0.weight"),
                                                t("gs.resize.0.bias"), x,
                                                4);  // ->64x64
                    else if (s == 1)
                        x = conv_transpose2d_p0(ctx, t("gs.resize.1.weight"),
                                                t("gs.resize.1.bias"), x,
                                                2);  // ->32x32
                    else if (s == 3)
                        x = conv2d(ctx, t("gs.resize.3.weight"),
                                   t("gs.resize.3.bias"), x, 2, 1);  // ->8x8
                    // s==2 Identity
                    l[s] = x;
                }

                // _fuse: layer{i}_rn = 3x3 pad1 conv (NO bias),
                // out_channels[i-1] -> 256
                ggml_tensor* l1_rn =
                        conv2d(ctx, t("gs.scratch.layer1_rn.weight"), nullptr,
                               l[0], 1, 1);
                ggml_tensor* l2_rn =
                        conv2d(ctx, t("gs.scratch.layer2_rn.weight"), nullptr,
                               l[1], 1, 1);
                ggml_tensor* l3_rn =
                        conv2d(ctx, t("gs.scratch.layer3_rn.weight"), nullptr,
                               l[2], 1, 1);
                ggml_tensor* l4_rn =
                        conv2d(ctx, t("gs.scratch.layer4_rn.weight"), nullptr,
                               l[3], 1, 1);

                // Compute refinenet target sizes dynamically from actual patch
                // grid. resize_layers produce: l[0]=4pw x 4ph, l[1]=2pw x 2ph,
                // l[2]=pw x ph, l[3]=(pw+1)/2 x (ph+1)/2. RefineNet fuses
                // bottom-up:
                //   rn4 outputs at l[2] size (pw x ph)
                //   rn3 outputs at l[1] size (2pw x 2ph)
                //   rn2 outputs at l[0] size (4pw x 4ph)
                //   rn1 uses scale_factor=2 (0,0 means 2x in feature_fusion)
                const int rn4_w = pw, rn4_h = ph;
                const int rn3_w = pw * 2, rn3_h = ph * 2;
                const int rn2_w = pw * 4, rn2_h = ph * 4;

                // refinenet4 (no residual / no rc1): top=l4_rn, upsample to
                // l[2] size
                ggml_tensor* out = feature_fusion(
                        ctx, l4_rn, nullptr, nullptr, nullptr, nullptr, nullptr,
                        t("gs.scratch.rn4.rc2.c1.weight"),
                        t("gs.scratch.rn4.rc2.c1.bias"),
                        t("gs.scratch.rn4.rc2.c2.weight"),
                        t("gs.scratch.rn4.rc2.c2.bias"),
                        t("gs.scratch.rn4.out.weight"),
                        t("gs.scratch.rn4.out.bias"), rn4_w, rn4_h);
                // refinenet3: lateral=l3_rn, upsample to l[1] size
                out = feature_fusion(
                        ctx, out, l3_rn, t("gs.scratch.rn3.rc1.c1.weight"),
                        t("gs.scratch.rn3.rc1.c1.bias"),
                        t("gs.scratch.rn3.rc1.c2.weight"),
                        t("gs.scratch.rn3.rc1.c2.bias"),
                        t("gs.scratch.rn3.rc2.c1.weight"),
                        t("gs.scratch.rn3.rc2.c1.bias"),
                        t("gs.scratch.rn3.rc2.c2.weight"),
                        t("gs.scratch.rn3.rc2.c2.bias"),
                        t("gs.scratch.rn3.out.weight"),
                        t("gs.scratch.rn3.out.bias"), rn3_w, rn3_h);
                // refinenet2: lateral=l2_rn, upsample to l[0] size
                out = feature_fusion(
                        ctx, out, l2_rn, t("gs.scratch.rn2.rc1.c1.weight"),
                        t("gs.scratch.rn2.rc1.c1.bias"),
                        t("gs.scratch.rn2.rc1.c2.weight"),
                        t("gs.scratch.rn2.rc1.c2.bias"),
                        t("gs.scratch.rn2.rc2.c1.weight"),
                        t("gs.scratch.rn2.rc2.c1.bias"),
                        t("gs.scratch.rn2.rc2.c2.weight"),
                        t("gs.scratch.rn2.rc2.c2.bias"),
                        t("gs.scratch.rn2.out.weight"),
                        t("gs.scratch.rn2.out.bias"), rn2_w, rn2_h);
                // refinenet1: lateral=l1_rn, scale_factor 2
                out = feature_fusion(ctx, out, l1_rn,
                                     t("gs.scratch.rn1.rc1.c1.weight"),
                                     t("gs.scratch.rn1.rc1.c1.bias"),
                                     t("gs.scratch.rn1.rc1.c2.weight"),
                                     t("gs.scratch.rn1.rc1.c2.bias"),
                                     t("gs.scratch.rn1.rc2.c1.weight"),
                                     t("gs.scratch.rn1.rc2.c1.bias"),
                                     t("gs.scratch.rn1.rc2.c2.weight"),
                                     t("gs.scratch.rn1.rc2.c2.bias"),
                                     t("gs.scratch.rn1.out.weight"),
                                     t("gs.scratch.rn1.out.bias"), 0, 0);
                // output_conv1: 3x3 pad1, 256 -> 128
                out = conv2d(ctx, t("gs.scratch.out1.weight"),
                             t("gs.scratch.out1.bias"), out, 1, 1);

                // upsample to (H,W)
                out = interp_bilinear_ac(ctx, out, W, H);  // [224,224,128]

                // images_merger(images): 3->32->64->128, 3x3 pad1, EXACT GELU
                // after each conv (incl. the last). The image is the
                // ImageNet-normalized input.
                const int64_t ine[4] = {W, H, 3, 1};
                ggml_tensor* img = be_.add_graph_input_nd(
                        ctx, pool, image_chw.data(), ine, 4);
                ggml_tensor* m = conv2d(ctx, t("gs.merger.0.weight"),
                                        t("gs.merger.0.bias"), img, 1, 1);
                m = ggml_gelu_erf(ctx, m);
                m = conv2d(ctx, t("gs.merger.1.weight"), t("gs.merger.1.bias"),
                           m, 1, 1);
                m = ggml_gelu_erf(ctx, m);
                m = conv2d(ctx, t("gs.merger.2.weight"), t("gs.merger.2.bias"),
                           m, 1, 1);
                m = ggml_gelu_erf(ctx, m);
                out = ggml_add(ctx, out, m);

                // + 0.1 * UV(128)
                ggml_tensor* pe2 = gs_add_uv_input(ctx, be_, pool, W, H,
                                                   feat_half, aspect, ratio);
                out = ggml_add(ctx, out, pe2);

                // output_conv2: conv 128->32 (3x3 pad1) -> relu -> conv 32->38
                // (1x1)
                out = conv2d(ctx, t("gs.scratch.out2a.weight"),
                             t("gs.scratch.out2a.bias"), out, 1, 1);
                out = ggml_relu(ctx, out);
                out = conv2d(ctx, t("gs.scratch.out2b.weight"),
                             t("gs.scratch.out2b.bias"), out, 1, 0);
                return out;  // [W,H,38,1] logits
            },
            logits);
    if (!ok) return false;

    const size_t HW = (size_t)H * W;
    if (logits.size() != (size_t)out_dim * HW) return false;
    // logits flat: channel c at offset c*HW, pixel (h,w) at h*W+w.
    // activate_head_gs: xyz (ch 0..36) linear; conf (ch 37) sigmoid.
    // channels-last.
    raw_gs.resize(HW * 37);
    gs_conf.resize(HW);
    for (size_t i = 0; i < HW; ++i) {
        for (int c = 0; c < 37; ++c)
            raw_gs[i * 37 + c] = logits[(size_t)c * HW + i];
        gs_conf[i] = 1.0f / (1.0f + std::exp(-logits[(size_t)37 * HW + i]));
    }
    return true;
}

}  // namespace depth
}  // namespace aicore
