#include "genderage_graph.hpp"
#include "align.hpp"
#include "backend.hpp"
#include "detect.hpp"
#include "graph_ops.hpp"
#include "preprocess.hpp"
#include "model_loader.hpp"
#include "common.hpp"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fd {

namespace {

// ONNX BatchNormalization epsilon for genderage.onnx (mxnet export: every bn node
// carries epsilon = 1e-3, NOT the 1e-5 the ArcFace export uses).
constexpr float kGaBnEps = 1e-3f;

// Per-channel BatchNorm in inference mode using the genderage naming convention
// (prefix + _gamma / _beta / _moving_mean / _moving_var); the ArcFace-flavoured
// fd::batchnorm in graph_ops expects .weight/.bias/.running_mean/.running_var, so
// this head needs its own loader. Math is identical:
//   y = (x - mean)/sqrt(var+eps)*gamma + beta, broadcast per channel (ne[2]).
ggml_tensor* ga_batchnorm(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                          const std::string& prefix,
                          std::vector<std::vector<float>>& keep) {
    ggml_tensor* gamma = clone_weight(ctx, ml, (prefix + "_gamma").c_str());
    ggml_tensor* beta  = clone_weight(ctx, ml, (prefix + "_beta").c_str());
    ggml_tensor* mean  = clone_weight(ctx, ml, (prefix + "_moving_mean").c_str());
    ggml_tensor* var   = clone_weight(ctx, ml, (prefix + "_moving_var").c_str());

    keep.emplace_back(1, kGaBnEps);
    const int64_t ne1[4] = {1, 1, 1, 1};
    ggml_tensor* eps_t = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne1,
                                            keep.back().data(), sizeof(float));
    ggml_tensor* inv_std = ggml_sqrt(ctx, ggml_add1(ctx, var, eps_t));
    ggml_tensor* scale   = ggml_div(ctx, gamma, inv_std);
    ggml_tensor* shift   = ggml_sub(ctx, beta, ggml_mul(ctx, mean, scale));

    const int64_t C = x->ne[2];
    ggml_tensor* s4 = ggml_reshape_4d(ctx, scale, 1, 1, C, 1);
    ggml_tensor* b4 = ggml_reshape_4d(ctx, shift, 1, 1, C, 1);
    return ggml_add(ctx, ggml_mul(ctx, x, s4), b4);
}

// Standard (group=1) conv -> BN -> ReLU. The conv is bias-less (the following BN
// is a separate node, not folded), so conv2d is called with b=nullptr.
ggml_tensor* conv_bn_relu(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                          const std::string& conv_w, const std::string& bn_prefix,
                          int stride, int pad, std::vector<std::vector<float>>& keep) {
    x = conv2d(ctx, ml, x, conv_w.c_str(), nullptr, stride, pad, /*relu=*/false);
    x = ga_batchnorm(ctx, ml, x, bn_prefix, keep);
    return ggml_relu(ctx, x);
}

// Depthwise (group=C) conv -> BN -> ReLU. The dw kernel is ggml ne [KW,KH,1,C];
// ggml_conv_2d_dw_direct yields the same WHCN output layout the rest of the graph
// uses.
ggml_tensor* dwconv_bn_relu(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                            const std::string& conv_w, const std::string& bn_prefix,
                            int stride, int pad, std::vector<std::vector<float>>& keep) {
    ggml_tensor* W = clone_weight(ctx, ml, conv_w.c_str());
    x = ggml_conv_2d_dw_direct(ctx, W, x, stride, stride, pad, pad, 1, 1);
    x = ga_batchnorm(ctx, ml, x, bn_prefix, keep);
    return ggml_relu(ctx, x);
}

// One depthwise-separable block conv_N: depthwise (k3, stride `dw_stride`, pad 1)
// then pointwise (1x1, stride 1, pad 0), each conv followed by its own BN + ReLU.
ggml_tensor* sep_block(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                       int n, int dw_stride, std::vector<std::vector<float>>& keep) {
    const std::string p = "ga.conv_" + std::to_string(n);
    x = dwconv_bn_relu(ctx, ml, x, p + "_dw_conv2d_weight", p + "_dw_batchnorm",
                       dw_stride, 1, keep);
    x = conv_bn_relu(ctx, ml, x, p + "_conv2d_weight", p + "_batchnorm", 1, 0, keep);
    return x;
}

// One task branch (t0 = gender, t1 = age): conv_13_t* (dw stride 2 + pw 1x1) ->
// conv_14_t* (dw stride 1 + pw 1x1) -> GlobalAveragePool -> FC. Returns the FC
// output [out_dim, 1] (gender: 2 logits, age: 1 scalar). Both branches consume
// the same conv_12 trunk output.
ggml_tensor* task_branch(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* trunk,
                         const std::string& tag, const std::string& fc,
                         std::vector<std::vector<float>>& keep) {
    const std::string c13 = "ga.conv_13";
    const std::string c14 = "ga.conv_14";
    // conv_13_t*: depthwise k3 stride 2 pad 1, then pointwise 1x1 (128 -> 256).
    ggml_tensor* x = dwconv_bn_relu(ctx, ml, trunk,
                                    c13 + "_dw_" + tag + "_conv2d_weight",
                                    c13 + "_dw_" + tag + "_batchnorm", 2, 1, keep);
    x = conv_bn_relu(ctx, ml, x, c13 + "_" + tag + "_conv2d_weight",
                     c13 + "_" + tag + "_batchnorm", 1, 0, keep);
    // conv_14_t*: depthwise k3 stride 1 pad 1, then pointwise 1x1 (256 -> 256).
    x = dwconv_bn_relu(ctx, ml, x, c14 + "_dw_" + tag + "_conv2d_weight",
                       c14 + "_dw_" + tag + "_batchnorm", 1, 1, keep);
    x = conv_bn_relu(ctx, ml, x, c14 + "_" + tag + "_conv2d_weight",
                     c14 + "_" + tag + "_batchnorm", 1, 0, keep);

    // GlobalAveragePool over the full spatial extent -> [1,1,C,1], flattened to
    // [C,1] for the Gemm. Kernel size is the live spatial size (fixed by the 96x96
    // input: 3x3 here).
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
    x = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, (int)W, (int)H, (int)W, (int)H, 0, 0);
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, x), C, 1);

    // Gemm(transB=1): out = x @ Wfc^T + b. Wfc ggml ne [256, out_dim] = [in, out].
    ggml_tensor* Wfc = clone_weight(ctx, ml, (fc + "_weight").c_str());
    x = ggml_mul_mat(ctx, Wfc, x);                       // -> [out_dim, 1]
    return ggml_add(ctx, x, clone_weight(ctx, ml, (fc + "_bias").c_str()));
}

} // namespace

std::vector<float> genderage_forward(const ModelLoader& ml, const Image& crop96,
                                     Backend& be) {
    const int sz = (int)ml.config().genderage_input_size;  // 96
    // The crop is already RGB; genderage's reference blob is
    // blobFromImage(BGR, swapRB=True) == R,G,B planes, mean 0 / std 1 (the net's
    // own (x-127.5)/128 lives inside the graph as scalar_op1/op2). swap_rb=false
    // lands on those R,G,B planes - the same convention as arcface_embed.
    std::vector<float> blob = to_blob(crop96, sz, 0.0f, 1.0f, /*swap_rb=*/false);

    std::vector<std::vector<float>> keep;  // BN-eps host buffers; must outlive compute
    std::vector<float> out;
    const bool ok = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ne[4] = {sz, sz, 3, 1};
        ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_F32, 4, ne,
                                            blob.data(), blob.size() * sizeof(float));
        // Built-in input normalization: y = (x - scalar_op1) * scalar_op2.
        ggml_tensor* s1 = ggml_reshape_4d(ctx, clone_weight(ctx, ml, "ga.scalar_op1"),
                                          1, 1, 1, 1);
        ggml_tensor* s2 = ggml_reshape_4d(ctx, clone_weight(ctx, ml, "ga.scalar_op2"),
                                          1, 1, 1, 1);
        x = ggml_mul(ctx, ggml_sub(ctx, x, s1), s2);

        // Stem: standard conv 3->16, k3 stride 2 pad 1, + BN + ReLU.
        x = conv_bn_relu(ctx, ml, x, "ga.conv_1_conv2d_weight", "ga.conv_1_batchnorm",
                         2, 1, keep);
        // 11 depthwise-separable blocks conv_2..conv_12. Depthwise stride is 2 at
        // conv_3 / conv_5 / conv_7 (the spatial downsamples), 1 elsewhere.
        static const int kDwStride[13] = {0, 0, /*2*/1, /*3*/2, /*4*/1, /*5*/2,
                                          /*6*/1, /*7*/2, /*8*/1, /*9*/1, /*10*/1,
                                          /*11*/1, /*12*/1};
        for (int n = 2; n <= 12; ++n)
            x = sep_block(ctx, ml, x, n, kDwStride[n], keep);

        // Two task branches off conv_12, then concat to fc1 = [g0, g1, age].
        ggml_tensor* gender = task_branch(ctx, ml, x, "t0", "ga.fullyconnected0", keep);
        ggml_tensor* age    = task_branch(ctx, ml, x, "t1", "ga.fullyconnected1", keep);
        return ggml_concat(ctx, gender, age, /*dim=*/0);  // [3, 1]
    }, out);
    if (!ok) throw std::runtime_error("facedetect: genderage graph compute failed");
    return out;
}

std::pair<int, int> genderage(const ModelLoader& ml, const Image& img,
                              const Detection& d, Backend& be) {
    const int sz = (int)ml.config().genderage_input_size;  // 96
    // face_align.transform: scale-about-center the detection box (expanded 1.5x)
    // to fit the square model input, centered. M maps source -> output:
    //   s = sz / (max(w,h) * 1.5);  M = [[s,0, -s*cx + sz/2],[0,s, -s*cy + sz/2]]
    const float w = d.x2 - d.x1, h = d.y2 - d.y1;
    const float cx = (d.x1 + d.x2) * 0.5f, cy = (d.y1 + d.y2) * 0.5f;
    const float s = (float)sz / (std::max(w, h) * 1.5f);
    const std::array<float, 6> M{ s, 0.0f, -s * cx + sz * 0.5f,
                                  0.0f, s, -s * cy + sz * 0.5f };
    Image crop;
    if (!warp_affine(img, M, crop, sz, sz))
        throw std::runtime_error("facedetect: genderage crop failed");

    std::vector<float> out = genderage_forward(ml, crop, be);
    FD_ASSERT(out.size() == 3);
    // gender = argmax(out[:2]) (np.argmax: first max on a tie -> 0 = Woman).
    const int gender = (out[1] > out[0]) ? 1 : 0;
    const int age = (int)std::lround(out[2] * 100.0f);
    return {gender, age};
}

} // namespace fd
