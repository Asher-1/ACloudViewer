// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/rmbg/birefnet_decoder.hpp"

#include "tasks/rmbg/nn_ops.hpp"

namespace rmbg {

bool BiRefNetDecoderForward::init(const char* gguf_path, std::string& err) {
    weights_ = WeightMap{};
    if (!weights_.load_gguf(gguf_path, err)) return false;
    loaded_ = true;
    return true;
}

bool BiRefNetDecoderForward::forward_squeeze(const std::vector<float>& x4,
                                             int H,
                                             int W,
                                             std::vector<float>& out,
                                             std::string& err) const {
    if (!loaded_) {
        err = "decoder not loaded";
        return false;
    }
    const int C = (int)(x4.size() / ((size_t)H * W));
    if (C <= 0 || x4.size() != (size_t)C * H * W) {
        err = "bad x4 feature shape";
        return false;
    }
    return squeeze_module_forward(x4, 1, C, H, W, weights_, out, err);
}

static bool ipt_cat(const std::vector<float>& image,
                    int H,
                    int W,
                    const std::vector<float>& feat,
                    int Cf,
                    int fH,
                    int fW,
                    const WeightMap& w,
                    const std::string& ipt_prefix,
                    std::vector<float>& out,
                    std::string& err) {
    std::vector<float> patches;
    image2patches_split_nchw(image, H, W, fH, fW, patches);
    const int patch_c = (int)(patches.size() / ((size_t)fH * fW));
    std::vector<float> ipt;
    if (!simple_convs_forward(patches, 1, patch_c, fH, fW, w, ipt_prefix, ipt,
                              err))
        return false;
    const int Ci = (int)(ipt.size() / ((size_t)fH * fW));
    concat_nchw_channel(feat, Cf, ipt, Ci, fH, fW, out);
    return true;
}

bool forward_decoder_eval(const std::vector<float>& nchw_in,
                          int H,
                          int W,
                          const Encoder4ScaleOutput& enc,
                          const std::vector<float>& x4_sq,
                          const WeightMap& w,
                          std::vector<float>& alpha_logits,
                          std::string& err) {
    const int h1 = H / 4, w1 = W / 4;
    const int h2 = H / 8, w2 = W / 8;
    const int h3 = H / 16, w3 = W / 16;
    const int h4 = H / 32, w4 = W / 32;

    std::vector<float> x4_in;
    if (!ipt_cat(nchw_in, H, W, x4_sq, 3072, h4, w4, w, "ipt5_", x4_in, err))
        return false;

    std::vector<float> p4;
    if (!basic_dec_blk_forward(x4_in, 1,
                               (int)(x4_in.size() / ((size_t)h4 * w4)), h4, w4,
                               w, "db4_", p4, err))
        return false;
    if (!gdt_attn_forward(p4, 1, 1536, h4, w4, w, "gdt4_", "gdta4_", p4, err))
        return false;

    std::vector<float> p4_up, lat3, fused3;
    bilinear_resize_nchw(p4, 1, 1536, h4, w4, h3, w3, p4_up);
    if (!lateral_block_forward(enc.x3, 1, 1536, h3, w3, w, "lat4_", lat3, err))
        return false;
    add_nchw(p4_up, lat3, 1536, h3, w3, fused3);

    std::vector<float> p3_in;
    if (!ipt_cat(nchw_in, H, W, fused3, 1536, h3, w3, w, "ipt4_", p3_in, err))
        return false;
    std::vector<float> p3;
    if (!basic_dec_blk_forward(p3_in, 1,
                               (int)(p3_in.size() / ((size_t)h3 * w3)), h3, w3,
                               w, "db3_", p3, err))
        return false;
    if (!gdt_attn_forward(p3, 1, 768, h3, w3, w, "gdt3_", "gdta3_", p3, err))
        return false;

    std::vector<float> p3_up, lat2, fused2;
    bilinear_resize_nchw(p3, 1, 768, h3, w3, h2, w2, p3_up);
    if (!lateral_block_forward(enc.x2, 1, 768, h2, w2, w, "lat3_", lat2, err))
        return false;
    add_nchw(p3_up, lat2, 768, h2, w2, fused2);

    std::vector<float> p2_in;
    if (!ipt_cat(nchw_in, H, W, fused2, 768, h2, w2, w, "ipt3_", p2_in, err))
        return false;
    std::vector<float> p2;
    if (!basic_dec_blk_forward(p2_in, 1,
                               (int)(p2_in.size() / ((size_t)h2 * w2)), h2, w2,
                               w, "db2_", p2, err))
        return false;
    if (!gdt_attn_forward(p2, 1, 384, h2, w2, w, "gdt2_", "gdta2_", p2, err))
        return false;

    std::vector<float> p2_up, lat1, fused1;
    bilinear_resize_nchw(p2, 1, 384, h2, w2, h1, w1, p2_up);
    if (!lateral_block_forward(enc.x1, 1, 384, h1, w1, w, "lat2_", lat1, err))
        return false;
    add_nchw(p2_up, lat1, 384, h1, w1, fused1);

    std::vector<float> p1_in;
    if (!ipt_cat(nchw_in, H, W, fused1, 384, h1, w1, w, "ipt2_", p1_in, err))
        return false;
    std::vector<float> p1;
    if (!basic_dec_blk_forward(p1_in, 1,
                               (int)(p1_in.size() / ((size_t)h1 * w1)), h1, w1,
                               w, "db1_", p1, err))
        return false;

    std::vector<float> p1_up;
    bilinear_resize_nchw(p1, 1, 192, h1, w1, H, W, p1_up);

    std::vector<float> out_in;
    if (!ipt_cat(nchw_in, H, W, p1_up, 192, H, W, w, "ipt1_", out_in, err))
        return false;

    Conv2dParams head;
    if (!load_conv2d(w, "out1_", head, err)) return false;
    conv2d_nchw(out_in, 1, (int)(out_in.size() / ((size_t)H * W)), H, W, head,
                1, 0, alpha_logits);
    return true;
}

bool forward_alpha(const std::vector<float>& nchw_in,
                   int H,
                   int W,
                   SwinBackboneForward& bb,
                   const WeightMap& w,
                   std::vector<float>& alpha_logits,
                   std::string& err) {
    Encoder4ScaleOutput enc;
    if (!forward_encoder_4scale(nchw_in, H, W, bb, enc, err)) return false;

    const int h4 = H / 32, w4 = W / 32;
    std::vector<float> x4_sq;
    if (!squeeze_module_forward(enc.x4, 1, 5760, h4, w4, w, x4_sq, err))
        return false;

    return forward_decoder_eval(nchw_in, H, W, enc, x4_sq, w, alpha_logits,
                                err);
}

bool BiRefNetDecoderForward::forward(
        const std::vector<std::vector<float>>& enc_feats,
        int H,
        int W,
        std::vector<float>& alpha_logits,
        std::string& err) const {
    (void)enc_feats;
    err = "use forward_alpha with full RGB input";
    return false;
}

}  // namespace rmbg
