#include "scrfd_graph.hpp"
#include "antispoof_graph.hpp"
#include "backend.hpp"
#include "graph_ops.hpp"
#include "model_loader.hpp"
#include "preprocess.hpp"
#include "common.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace fd {

namespace {

// ---- Conv(+bias)(+relu) helper ---------------------------------------------
// The SCRFD ONNX export has BN already folded into the conv weight+bias, so a
// "block" is just conv -> (+ per-channel bias) -> (optional relu). Thin wrapper
// over the shared fd::conv2d (also used by the ArcFace graph).
ggml_tensor* conv(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                  const char* w, const char* b, int stride, int pad, bool relu) {
    return fd::conv2d(ctx, ml, x, w, b, stride, pad, relu);
}

// A 3x3 stride-1 pad-1 conv (the SCRFD backbone/neck default).
ggml_tensor* conv3(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                   const char* w, const char* b, bool relu) {
    return conv(ctx, ml, x, w, b, 1, 1, relu);
}

ggml_tensor* relu(ggml_context* ctx, ggml_tensor* x) { return ggml_relu(ctx, x); }
ggml_tensor* add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) { return ggml_add(ctx, a, b); }

// 2x2 stride-2 pool (no padding). SCRFD's downsample shortcuts use AveragePool;
// the stem uses MaxPool. Even feature sizes here, so ceil/floor agree.
ggml_tensor* pool2(ggml_context* ctx, ggml_tensor* x, ggml_op_pool op) {
    return ggml_pool_2d(ctx, x, op, 2, 2, 2, 2, 0, 0);
}

// Flatten a captured head map [W,H,Ctot,1] (ggml contiguous) into insightface's
// (H*W*num_anchors, C) row order. ct = a*C + c (anchor-major within a location).
std::vector<float> flatten_head(const std::vector<float>& src, int W, int H,
                                int A, int C) {
    std::vector<float> dst((size_t)H * W * A * C);
    const size_t plane = (size_t)W * H;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            const size_t loc = (size_t)h * W + w;
            for (int a = 0; a < A; ++a)
                for (int c = 0; c < C; ++c)
                    dst[(loc * A + a) * C + c] = src[(size_t)(a * C + c) * plane + loc];
        }
    return dst;
}

// cv2.resize INTER_LINEAR for uint8, 11-bit fixed-point (OpenCV's
// HResizeLinear/VResizeLinear 8U path). INTER_RESIZE_COEF_BITS = 11. Matches
// OpenCV to within 1 LSB: OpenCV's SSE/AVX path narrows the horizontal int32
// result to int16 (>>4) before the vertical multiply, so this scalar version
// differs by at most 1 on a minority of pixels. This is off the Task 3.1 graph
// gate (which feeds the reference's exact letterbox pixels); it is the
// production preprocess used by the decode path.
void cv_resize_linear_u8(const uint8_t* src, int sw, int sh,
                         uint8_t* dst, int dw, int dh, int cn) {
    const int BITS = 11;
    const float SCALE = (float)(1 << BITS);  // 2048
    auto sat_short = [](float v) -> int {
        int i = (int)std::lrint(v);
        return i < -32768 ? -32768 : (i > 32767 ? 32767 : i);
    };
    std::vector<int> xofs(dw);
    std::vector<int> ialpha(dw * 2);
    std::vector<int> yofs(dh);
    std::vector<int> ibeta(dh * 2);
    const double sxs = (double)sw / dw, sys = (double)sh / dh;
    for (int dx = 0; dx < dw; ++dx) {
        float fx = (float)((dx + 0.5) * sxs - 0.5);
        int sx = (int)std::floor(fx);
        fx -= sx;
        if (sx < 0) { sx = 0; fx = 0.f; }
        if (sx >= sw - 1) { sx = sw - 1; fx = 0.f; }
        xofs[dx] = sx;
        ialpha[dx * 2] = sat_short((1.f - fx) * SCALE);
        ialpha[dx * 2 + 1] = sat_short(fx * SCALE);
    }
    for (int dy = 0; dy < dh; ++dy) {
        float fy = (float)((dy + 0.5) * sys - 0.5);
        int sy = (int)std::floor(fy);
        fy -= sy;
        if (sy < 0) { sy = 0; fy = 0.f; }
        if (sy >= sh - 1) { sy = sh - 1; fy = 0.f; }
        yofs[dy] = sy;
        ibeta[dy * 2] = sat_short((1.f - fy) * SCALE);
        ibeta[dy * 2 + 1] = sat_short(fy * SCALE);
    }
    for (int dy = 0; dy < dh; ++dy) {
        const int sy0 = yofs[dy], sy1 = std::min(sy0 + 1, sh - 1);
        const int b0 = ibeta[dy * 2], b1 = ibeta[dy * 2 + 1];
        const uint8_t* r0 = src + (size_t)sy0 * sw * cn;
        const uint8_t* r1 = src + (size_t)sy1 * sw * cn;
        uint8_t* drow = dst + (size_t)dy * dw * cn;
        for (int dx = 0; dx < dw; ++dx) {
            const int sx = xofs[dx], sx1 = std::min(sx + 1, sw - 1);
            const int a0 = ialpha[dx * 2], a1 = ialpha[dx * 2 + 1];
            for (int c = 0; c < cn; ++c) {
                const int p0 = r0[sx * cn + c] * a0 + r0[sx1 * cn + c] * a1;
                const int p1 = r1[sx * cn + c] * a0 + r1[sx1 * cn + c] * a1;
                int v = ((int64_t)p0 * b0 + (int64_t)p1 * b1 + (1 << 21)) >> 22;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                drow[dx * cn + c] = (uint8_t)v;
            }
        }
    }
}

} // namespace

void scrfd_letterbox(const Image& src, int size, Image& out, float& det_scale) {
    const int ow = src.width, oh = src.height;
    const float im_ratio = (float)oh / (float)ow;
    const float model_ratio = 1.0f;  // square detector input
    int new_w, new_h;
    if (im_ratio > model_ratio) {
        new_h = size;
        new_w = (int)((float)size / im_ratio);
    } else {
        new_w = size;
        new_h = (int)((float)size * im_ratio);
    }
    det_scale = (float)new_h / (float)oh;

    std::vector<uint8_t> resized((size_t)new_w * new_h * 3);
    cv_resize_linear_u8(src.rgb.data(), ow, oh, resized.data(), new_w, new_h, 3);

    out.width = size;
    out.height = size;
    out.rgb.assign((size_t)size * size * 3, 0);  // top-left zero pad
    for (int y = 0; y < new_h; ++y) {
        std::memcpy(&out.rgb[((size_t)y * size) * 3],
                    &resized[((size_t)y * new_w) * 3], (size_t)new_w * 3);
    }
}

namespace {

// det_2.5g / det_500m (buffalo_m / buffalo_s) path: replay the embedded detector
// ONNX topology through the shared metadata-driven interpreter and map its 9 named
// per-stride heads onto ScrfdRawOut. The interpreter already flattens each head to
// the (H*W*num_anchors, C) row layout the host decode reads (via the head
// Transpose+Reshape), and the score heads end in Sigmoid in-graph, so the outputs
// are drop-in for the existing anchor decode + NMS - exactly the det_10g layout,
// just produced metadata-driven instead of hand-mapped. Output order is the ONNX
// graph-output order: scores s8/16/32, bbox s8/16/32, kps s8/16/32.
std::vector<ScrfdRawOut> scrfd_forward_interp(const ModelLoader& ml,
                                              const Image& blob_img, Backend& be) {
    const FaceConfig& cfg = ml.config();
    const int S = (int)cfg.det_input_size;
    const size_t nstride = cfg.det_strides.size();
    std::vector<ScrfdRawOut> outs(nstride);
    if (cfg.det_graph_outputs.size() != nstride * 3) {
        FD_LOG("scrfd_forward_interp: expected %zu outputs, graph has %zu",
               nstride * 3, cfg.det_graph_outputs.size());
        return outs;
    }
    std::vector<float> blob = to_blob(blob_img, S, 127.5f, 128.0f, /*swap_rb=*/false);
    std::vector<std::vector<float>> heads;
    try {
        heads = run_onnx_graph_multi(ml, "det.", cfg.det_graph, cfg.det_graph_input,
                                     cfg.det_graph_outputs, blob, S, be, 1e-5f);
    } catch (const std::exception& e) {
        FD_LOG("scrfd_forward_interp: %s", e.what());
        return outs;
    }
    for (size_t i = 0; i < nstride; ++i) {
        outs[i].score = std::move(heads[i]);
        outs[i].bbox  = std::move(heads[nstride + i]);
        outs[i].kps   = std::move(heads[2 * nstride + i]);
    }
    return outs;
}

} // namespace

std::vector<ScrfdRawOut> scrfd_forward(const ModelLoader& ml,
                                       const Image& blob_img640, Backend& be) {
    // buffalo_m (det_2.5g) / buffalo_s (det_500m) carry their detector topology as
    // a GGUF KV graph; replay it metadata-driven. buffalo_l (det_10g) has no such
    // KV and stays on the hand-mapped path below (zero regression).
    if (!ml.config().det_graph.empty())
        return scrfd_forward_interp(ml, blob_img640, be);

    const int S = (int)ml.config().det_input_size;            // 640
    const int A = (int)ml.config().det_num_anchors;           // 2

    // Detector blob: R,G,B plane order, (px-127.5)/128 (== cv2 blobFromImage with
    // swapRB on a BGR image; our Image is already RGB, so no plane swap).
    std::vector<float> blob = to_blob(blob_img640, S, 127.5f, 128.0f, /*swap_rb=*/false);

    // Captured raw head maps, [W,H,Ctot,1] ggml-contiguous, per stride.
    std::vector<float> cap_score[3], cap_bbox[3], cap_kps[3];
    int feat_w[3] = {0, 0, 0}, feat_h[3] = {0, 0, 0};

    std::vector<float> dummy;
    const bool ok = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        // Permit F(4x4,3x3) Winograd on this detector's large early 3x3 s1p1
        // convs (the 320/160/80 maps). Scoped to this graph build only: the guard
        // clears the flag on return so the recognizer/other graphs keep F(2x2).
        WinogradF4Scope f4scope(true);
        const int64_t ne_in[4] = {S, S, 3, 1};
        ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_F32, 4, ne_in,
                                            blob.data(), blob.size() * sizeof(float));
        // ---- Backbone (ResNet-style stem + 4 stages) ------------------------
        x = conv(ctx, ml, x, "det.547", "det.549", 2, 1, true);   // 279 stem s2
        x = conv3(ctx, ml, x, "det.551", "det.553", true);        // 282
        x = conv3(ctx, ml, x, "det.555", "det.557", true);        // 285
        ggml_tensor* p = pool2(ctx, x, GGML_OP_POOL_MAX);          // 286 maxpool

        // layer1 (56ch, 3 residual blocks)
        ggml_tensor* y;
        y = conv3(ctx, ml, p, "det.559", "det.561", true);
        y = conv3(ctx, ml, y, "det.563", "det.565", false);
        ggml_tensor* l1 = relu(ctx, add(ctx, y, p));              // 293
        y = conv3(ctx, ml, l1, "det.567", "det.569", true);
        y = conv3(ctx, ml, y, "det.571", "det.573", false);
        l1 = relu(ctx, add(ctx, y, l1));                          // 300
        y = conv3(ctx, ml, l1, "det.575", "det.577", true);
        y = conv3(ctx, ml, y, "det.579", "det.581", false);
        ggml_tensor* c2 = relu(ctx, add(ctx, y, l1));            // 307 (56ch, /4)

        // layer2 (88ch): downsample block + 3 residual blocks -> stride 8 (C3)
        y = conv(ctx, ml, c2, "det.583", "det.585", 2, 1, true);
        y = conv3(ctx, ml, y, "det.587", "det.589", false);
        ggml_tensor* sc = pool2(ctx, c2, GGML_OP_POOL_AVG);
        sc = conv(ctx, ml, sc, "det.591", "det.593", 1, 0, false);
        ggml_tensor* s8 = relu(ctx, add(ctx, y, sc));            // 317
        y = conv3(ctx, ml, s8, "det.595", "det.597", true);
        y = conv3(ctx, ml, y, "det.599", "det.601", false);
        s8 = relu(ctx, add(ctx, y, s8));                         // 324
        y = conv3(ctx, ml, s8, "det.603", "det.605", true);
        y = conv3(ctx, ml, y, "det.607", "det.609", false);
        s8 = relu(ctx, add(ctx, y, s8));                         // 331
        y = conv3(ctx, ml, s8, "det.611", "det.613", true);
        y = conv3(ctx, ml, y, "det.615", "det.617", false);
        ggml_tensor* c3 = relu(ctx, add(ctx, y, s8));           // 338 (88ch, /8)

        // layer3 (88ch): downsample block + 1 residual block -> stride 16 (C4)
        y = conv(ctx, ml, c3, "det.619", "det.621", 2, 1, true);
        y = conv3(ctx, ml, y, "det.623", "det.625", false);
        sc = pool2(ctx, c3, GGML_OP_POOL_AVG);
        sc = conv(ctx, ml, sc, "det.627", "det.629", 1, 0, false);
        ggml_tensor* s16 = relu(ctx, add(ctx, y, sc));          // 348
        y = conv3(ctx, ml, s16, "det.631", "det.633", true);
        y = conv3(ctx, ml, y, "det.635", "det.637", false);
        ggml_tensor* c4 = relu(ctx, add(ctx, y, s16));          // 355 (88ch, /16)

        // layer4 (224ch): downsample block + 2 residual blocks -> stride 32 (C5)
        y = conv(ctx, ml, c4, "det.639", "det.641", 2, 1, true);
        y = conv3(ctx, ml, y, "det.643", "det.645", false);
        sc = pool2(ctx, c4, GGML_OP_POOL_AVG);
        sc = conv(ctx, ml, sc, "det.647", "det.649", 1, 0, false);
        ggml_tensor* s32 = relu(ctx, add(ctx, y, sc));          // 365
        y = conv3(ctx, ml, s32, "det.651", "det.653", true);
        y = conv3(ctx, ml, y, "det.655", "det.657", false);
        s32 = relu(ctx, add(ctx, y, s32));                      // 372
        y = conv3(ctx, ml, s32, "det.659", "det.661", true);
        y = conv3(ctx, ml, y, "det.663", "det.665", false);
        ggml_tensor* c5 = relu(ctx, add(ctx, y, s32));          // 379 (224ch, /32)

        // ---- PAFPN neck -----------------------------------------------------
        ggml_tensor* l0 = conv(ctx, ml, c3, "det.neck.lateral_convs.0.conv.weight",
                               "det.neck.lateral_convs.0.conv.bias", 1, 0, false);  // 380
        ggml_tensor* la1 = conv(ctx, ml, c4, "det.neck.lateral_convs.1.conv.weight",
                                "det.neck.lateral_convs.1.conv.bias", 1, 0, false); // 381
        ggml_tensor* la2 = conv(ctx, ml, c5, "det.neck.lateral_convs.2.conv.weight",
                                "det.neck.lateral_convs.2.conv.bias", 1, 0, false); // 382

        // top-down: nearest 2x upsample + add
        ggml_tensor* up = ggml_upscale(ctx, la2, 2, GGML_SCALE_MODE_NEAREST);       // 401
        ggml_tensor* p1m = add(ctx, la1, up);                                       // 402
        ggml_tensor* up2 = ggml_upscale(ctx, p1m, 2, GGML_SCALE_MODE_NEAREST);      // 421
        ggml_tensor* p0m = add(ctx, l0, up2);                                       // 422

        ggml_tensor* f0 = conv3(ctx, ml, p0m, "det.neck.fpn_convs.0.conv.weight",
                                "det.neck.fpn_convs.0.conv.bias", false);           // 423
        // NOTE: the ONNX export feeds fpn_convs.1/.2's bias from downsample_convs
        // 0/1 (fpn_convs.1/.2 carry no own bias). Match that exactly.
        ggml_tensor* f1 = conv3(ctx, ml, p1m, "det.neck.fpn_convs.1.conv.weight",
                                "det.neck.downsample_convs.0.conv.bias", false);    // 424
        ggml_tensor* f2 = conv3(ctx, ml, la2, "det.neck.fpn_convs.2.conv.weight",
                                "det.neck.downsample_convs.1.conv.bias", false);    // 425

        // bottom-up: strided downsample + add
        ggml_tensor* d0 = conv(ctx, ml, f0, "det.neck.downsample_convs.0.conv.weight",
                               "det.neck.downsample_convs.0.conv.bias", 2, 1, false); // 426
        ggml_tensor* n1 = add(ctx, f1, d0);                                          // 427
        ggml_tensor* d1 = conv(ctx, ml, n1, "det.neck.downsample_convs.1.conv.weight",
                               "det.neck.downsample_convs.1.conv.bias", 2, 1, false); // 428
        ggml_tensor* n2 = add(ctx, f2, d1);                                          // 429

        ggml_tensor* pa0 = conv3(ctx, ml, n1, "det.neck.pafpn_convs.0.conv.weight",
                                 "det.neck.pafpn_convs.0.conv.bias", false);         // 430
        ggml_tensor* pa1 = conv3(ctx, ml, n2, "det.neck.pafpn_convs.1.conv.weight",
                                 "det.neck.pafpn_convs.1.conv.bias", false);         // 431

        // ---- Three shared stride heads (stem 3x3 x3, then cls/reg/kps) -------
        struct Head {
            ggml_tensor* feat;
            const char* sw[3]; const char* sb[3];   // stem convs
            const char* cls_w; const char* cls_b;
            const char* reg_w; const char* reg_b;
            const char* kps_w; const char* kps_b;
            const char* scale;
        };
        const Head heads[3] = {
            { f0,
              {"det.667","det.671","det.675"}, {"det.669","det.673","det.677"},
              "det.bbox_head.stride_cls.(8, 8).weight",  "det.bbox_head.stride_cls.(8, 8).bias",
              "det.bbox_head.stride_reg.(8, 8).weight",  "det.bbox_head.stride_reg.(8, 8).bias",
              "det.bbox_head.stride_kps.(8, 8).weight",  "det.bbox_head.stride_kps.(8, 8).bias",
              "det.bbox_head.scales.0.scale" },
            { pa0,
              {"det.679","det.683","det.687"}, {"det.681","det.685","det.689"},
              "det.bbox_head.stride_cls.(16, 16).weight","det.bbox_head.stride_cls.(16, 16).bias",
              "det.bbox_head.stride_reg.(16, 16).weight","det.bbox_head.stride_reg.(16, 16).bias",
              "det.bbox_head.stride_kps.(16, 16).weight","det.bbox_head.stride_kps.(16, 16).bias",
              "det.bbox_head.scales.1.scale" },
            { pa1,
              {"det.691","det.695","det.699"}, {"det.693","det.697","det.701"},
              "det.bbox_head.stride_cls.(32, 32).weight","det.bbox_head.stride_cls.(32, 32).bias",
              "det.bbox_head.stride_reg.(32, 32).weight","det.bbox_head.stride_reg.(32, 32).bias",
              "det.bbox_head.stride_kps.(32, 32).weight","det.bbox_head.stride_kps.(32, 32).bias",
              "det.bbox_head.scales.2.scale" },
        };

        ggml_tensor* last = nullptr;
        for (int i = 0; i < 3; ++i) {
            const Head& hd = heads[i];
            ggml_tensor* h = conv3(ctx, ml, hd.feat, hd.sw[0], hd.sb[0], true);
            h = conv3(ctx, ml, h, hd.sw[1], hd.sb[1], true);
            h = conv3(ctx, ml, h, hd.sw[2], hd.sb[2], true);

            ggml_tensor* cls = conv3(ctx, ml, h, hd.cls_w, hd.cls_b, false);
            ggml_tensor* score = ggml_sigmoid(ctx, cls);

            ggml_tensor* reg = conv3(ctx, ml, h, hd.reg_w, hd.reg_b, false);
            ggml_tensor* scale = clone_weight(ctx, ml, hd.scale);  // ne [1]
            reg = ggml_mul(ctx, reg, scale);

            ggml_tensor* kps = conv3(ctx, ml, h, hd.kps_w, hd.kps_b, false);

            feat_w[i] = (int)score->ne[0];
            feat_h[i] = (int)score->ne[1];
            capture_graph_output(score, &cap_score[i]);
            capture_graph_output(reg, &cap_bbox[i]);
            capture_graph_output(kps, &cap_kps[i]);
            last = kps;
        }
        return last;
    }, dummy);

    std::vector<ScrfdRawOut> outs(3);
    if (!ok) {
        FD_LOG("scrfd_forward: graph compute failed");
        return outs;
    }
    for (int i = 0; i < 3; ++i) {
        const int W = feat_w[i], H = feat_h[i];
        outs[i].score = flatten_head(cap_score[i], W, H, A, 1);
        outs[i].bbox  = flatten_head(cap_bbox[i],  W, H, A, 4);
        outs[i].kps   = flatten_head(cap_kps[i],   W, H, A, 10);
    }
    return outs;
}

} // namespace fd
