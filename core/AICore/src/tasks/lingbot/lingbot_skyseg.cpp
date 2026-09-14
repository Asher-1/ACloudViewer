// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "lingbot_skyseg.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <unordered_map>

namespace lingbot {

namespace {

// Official preprocessing (lingbot_map/vis/sky_segmentation.py): the image is
// rounded to u8, resized to 320x320 with OpenCV's u8 INTER_LINEAR (replicated
// bit-exactly by resize_u8_exact below), normalized with ImageNet statistics
// and fed as CHW RGB. All of it is reproduced here in plain C++ so the CLI
// needs nothing but this translation unit.
constexpr int kInput = 320;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

// Bit-exact replica of the official preprocessing resize: cv2.resize on u8
// with INTER_LINEAR (OpenCV 5.x CPU path, modules/imgproc/src/resize.cpp).
// Matching cv2 bitwise matters because the official mask postprocess
// (min-max normalize, truncate to u8, keep rule "resized u8 == 0") amplifies
// sub-lsb input differences into visible mask flips (the courthouse tent
// regions flipped 2% of pixels before this was exact - see
// benchmarks/validation_report.md, "skyseg preprocess alignment").
// Properties that must be preserved, per resize.cpp:
//   - coefficients are 11-bit fixed point, quantized with two INDEPENDENT
//     cvRound calls (round-half-to-even), so a0+a1 is not always 2048;
//   - the source coordinate is computed in double and rounded to float:
//     fx = (float)((d+0.5)*scale - 0.5), then fx -= sx in float;
//   - horizontally out-of-range taps clamp sx and zero fx (pure copy),
//     while vertical rows are clamped with the fraction kept (replicate);
//   - the horizontal pass is a lossless int accumulation;
//   - the vertical pass truncates twice:
//     dst = ((b0*(h0>>4)>>16) + (b1*(h1>>4)>>16) + 2) >> 2.
void resize_u8_exact(const unsigned char *src,
                     int sw,
                     int sh,
                     int nc,
                     unsigned char *dst,
                     int dw,
                     int dh) {
    std::vector<int> xofs(dw), xa0(dw), xa1(dw);
    const double scale_x = (double)sw / dw;
    for (int dx = 0; dx < dw; ++dx) {
        float fx = (float)((dx + 0.5) * scale_x - 0.5);
        int ix = (int)std::floor(fx);
        fx = fx - (float)ix;
        if (ix < 0) {
            fx = 0.f;
            ix = 0;
        }
        if (ix >= sw - 1) {
            fx = 0.f;
            ix = sw - 1;
        }
        xofs[dx] = ix;
        xa0[dx] = (int)std::lrintf((1.f - fx) * 2048.f);
        xa1[dx] = (int)std::lrintf(fx * 2048.f);
    }
    std::vector<int> yofs(dh), yb0(dh), yb1(dh);
    const double scale_y = (double)sh / dh;
    for (int dy = 0; dy < dh; ++dy) {
        float fy = (float)((dy + 0.5) * scale_y - 0.5);
        int iy = (int)std::floor(fy);
        fy = fy - (float)iy;
        yofs[dy] = iy;
        yb0[dy] = (int)std::lrintf((1.f - fy) * 2048.f);
        yb1[dy] = (int)std::lrintf(fy * 2048.f);
    }
    // lossless horizontal pass on every source row
    std::vector<int> hall((size_t)sh * dw * nc);
    for (int r = 0; r < sh; ++r) {
        const unsigned char *s = src + (size_t)r * sw * nc;
        int *h = &hall[(size_t)r * dw * nc];
        for (int dx = 0; dx < dw; ++dx) {
            const int x0 = xofs[dx], x1 = std::min(x0 + 1, sw - 1);
            for (int c = 0; c < nc; ++c)
                h[(size_t)dx * nc + c] = xa0[dx] * s[(size_t)x0 * nc + c] +
                                         xa1[dx] * s[(size_t)x1 * nc + c];
        }
    }
    for (int dy = 0; dy < dh; ++dy) {
        const int r0 = yofs[dy] < 0 ? 0 : std::min(yofs[dy], sh - 1);
        const int r1 = std::min(yofs[dy] + 1, sh - 1);
        const int *g0 = &hall[(size_t)r0 * dw * nc];
        const int *g1 = &hall[(size_t)r1 * dw * nc];
        unsigned char *o = dst + (size_t)dy * dw * nc;
        const int b0 = yb0[dy], b1 = yb1[dy];
        for (int i = 0; i < dw * nc; ++i)
            o[i] = (unsigned char)(((b0 * (g0[i] >> 4) >> 16) +
                                    (b1 * (g1[i] >> 4) >> 16) + 2) >>
                                   2);
    }
}

std::vector<std::string> split(const std::string &line) {
    std::vector<std::string> out;
    std::istringstream ss(line);
    std::string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
}

}  // namespace

struct skyseg::impl {
    gguf_context *guf = nullptr;
    ggml_context *wctx = nullptr;          // weight tensors with CPU data
    ggml_context *dctx = nullptr;          // device-side weight mirrors
    ggml_backend_buffer_t wbuf = nullptr;  // device weight buffer
    ggml_backend_t be = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_context *input_ctx = nullptr;  // owns the input tensor (private
    ggml_backend_buffer_t input_buf = nullptr;  // buffer so galloc cannot
    ggml_tensor *input_keep = nullptr;          // recycle it mid-graph)
    ggml_context *graph_ctx = nullptr;          // owns the graph + its tensors
    ggml_cgraph *graph = nullptr;               // static graph, built once
    ggml_tensor *input = nullptr;   // graph input node {320,320,3,1}
    ggml_tensor *output = nullptr;  // graph output node {320,320,1,1}
    std::vector<std::string> graph_lines;
    std::string input_name = "input";
    std::string output_name;
};

skyseg::skyseg() : p_(new impl) {}
skyseg::~skyseg() { release(); }

void skyseg::release() {
    if (!p_) return;
    if (p_->galloc) ggml_gallocr_free(p_->galloc);
    if (p_->graph_ctx) ggml_free(p_->graph_ctx);
    if (p_->input_buf) ggml_backend_buffer_free(p_->input_buf);
    if (p_->input_ctx) ggml_free(p_->input_ctx);
    if (p_->wbuf) ggml_backend_buffer_free(p_->wbuf);
    if (p_->dctx) ggml_free(p_->dctx);
    if (p_->guf) gguf_free(p_->guf);
    if (p_->wctx) ggml_free(p_->wctx);
    *p_ = impl();
    error_.clear();
}

bool skyseg::load(const std::string &gguf_path, ggml_backend_t backend) {
    release();
    p_->be = backend;
    if (!backend) {
        error_ = "skyseg: no backend";
        return false;
    }

    gguf_init_params params = {false, &p_->wctx};
    p_->guf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!p_->guf) {
        error_ = "skyseg: failed to open GGUF: " + gguf_path;
        return false;
    }

    // Read the instruction list emitted by convert_skyseg.py.
    const int64_t gid = gguf_find_key(p_->guf, "skyseg.graph");
    if (gid < 0) {
        error_ = "skyseg: missing skyseg.graph metadata";
        return false;
    }
    std::istringstream graph_ss(gguf_get_val_str(p_->guf, gid));
    std::string line;
    while (std::getline(graph_ss, line))
        if (!line.empty()) p_->graph_lines.push_back(line);
    const auto kv_str = [&](const char *key) {
        const int64_t id = gguf_find_key(p_->guf, key);
        return id >= 0 ? std::string(gguf_get_val_str(p_->guf, id))
                       : std::string();
    };
    p_->input_name = kv_str("skyseg.input");
    p_->output_name = kv_str("skyseg.output");

    // Vulkan (v0.21) capability boundary, verified by experiment: its mul_mat
    // has no Q8_0 x F32 shader (auto-quantizing the activation operand needs
    // the integer-dot-product extension, which reports unsupported here), and
    // neither im2col nor cpy can emit Q8_1 activations. Q8_0 kernels are
    // therefore dequantized to F16 at load time on Vulkan; CUDA and CPU run
    // the native Q8_0 path. The dequant error is bounded by the f16 rounding
    // of d*q, ~5e-4 relative - see benchmarks/skyseg_ggml.md.
    const char *bname = ggml_backend_name(backend);
    const bool vk_backend = bname && std::strstr(bname, "Vulkan") != nullptr;

    // Bind the weight tensors to the backend buffer (same pattern as
    // model.cpp: mirror the tensor defs in a device context, allocate, copy,
    // then point the CPU-side tensors at the device memory so graph building
    // can use them directly).
    size_t n_tensors = 0;
    for (ggml_tensor *t = ggml_get_first_tensor(p_->wctx); t;
         t = ggml_get_next_tensor(p_->wctx, t))
        ++n_tensors;
    ggml_init_params dp = {ggml_tensor_overhead() * (n_tensors + 8), nullptr,
                           true};
    p_->dctx = ggml_init(dp);
    if (!p_->dctx) {
        error_ = "skyseg: failed to create device weight context";
        return false;
    }
    for (ggml_tensor *s = ggml_get_first_tensor(p_->wctx); s;
         s = ggml_get_next_tensor(p_->wctx, s)) {
        ggml_tensor *d = (vk_backend && s->type == GGML_TYPE_Q8_0)
                                 ? ggml_new_tensor(p_->dctx, GGML_TYPE_F16,
                                                   ggml_n_dims(s), s->ne)
                                 : ggml_dup_tensor(p_->dctx, s);
        ggml_set_name(d, ggml_get_name(s));
    }
    p_->wbuf = ggml_backend_alloc_ctx_tensors(p_->dctx, backend);
    if (!p_->wbuf) {
        error_ = "skyseg: failed to allocate weight buffer";
        return false;
    }
    for (ggml_tensor *s = ggml_get_first_tensor(p_->wctx); s;
         s = ggml_get_next_tensor(p_->wctx, s)) {
        auto *d = ggml_get_tensor(p_->dctx, ggml_get_name(s));
        if (d->type == GGML_TYPE_F16 && s->type == GGML_TYPE_Q8_0) {
            // host-side dequant q8_0 -> f16. Block layout (ggml-common.h,
            // ABI-stable): ggml_half d; int8_t qs[32]; value = d * qs[i].
            const int64_t n = ggml_nelements(s);
            const uint8_t *bytes = (const uint8_t *)s->data;
            const int64_t nblk = n / 32;
            std::vector<float> tmp(n);
            for (int64_t b = 0; b < nblk; ++b) {
                float scale = 0.f;
                scale = ggml_fp16_to_fp32(
                        *(const ggml_fp16_t *)(bytes + b * 34));
                const int8_t *qs = (const int8_t *)(bytes + b * 34 + 2);
                for (int j = 0; j < 32; ++j) tmp[b * 32 + j] = scale * qs[j];
            }
            std::vector<ggml_fp16_t> h16(n);
            for (int64_t i = 0; i < n; ++i)
                h16[static_cast<size_t>(i)] =
                        ggml_fp32_to_fp16(tmp[static_cast<size_t>(i)]);
            ggml_backend_tensor_set(d, h16.data(), 0, n * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(d, s->data, 0, ggml_nbytes(s));
        }
        s->buffer = d->buffer;
        s->data = d->data;
        s->type = d->type;
        for (int k = 0; k < GGML_MAX_DIMS; ++k) s->nb[k] = d->nb[k];
    }
    ggml_backend_synchronize(backend);
    return build_graph();
}

bool skyseg::build_graph() {
    // Node lookup: weight tensors from the weight context plus the input.
    std::unordered_map<std::string, ggml_tensor *> nodes;
    for (ggml_tensor *t = ggml_get_first_tensor(p_->wctx); t;
         t = ggml_get_next_tensor(p_->wctx, t)) {
        nodes[ggml_get_name(t)] = t;
    }

    ggml_init_params ip = {256ull * 1024 * 1024, nullptr, true /* no_alloc */};
    p_->graph_ctx = ggml_init(ip);
    if (!p_->graph_ctx) {
        error_ = "skyseg: failed to create graph context";
        return false;
    }
    ggml_context *gctx = p_->graph_ctx;

    // The input lives in its OWN buffer: as a leaf it would otherwise be
    // recycled by the graph allocator for later nodes' outputs, clobbering
    // it right after tensor_set (the first im2col then reads garbage).
    ggml_init_params ipi = {ggml_tensor_overhead() * 4, nullptr, true};
    p_->input_ctx = ggml_init(ipi);
    p_->input = ggml_new_tensor_4d(p_->input_ctx, GGML_TYPE_F32, kInput, kInput,
                                   3, 1);
    ggml_set_name(p_->input, p_->input_name.c_str());
    ggml_set_input(p_->input);
    p_->input_buf = ggml_backend_alloc_ctx_tensors(p_->input_ctx, p_->be);
    if (!p_->input_buf) {
        error_ = "skyseg: failed to allocate input buffer";
        return false;
    }
    nodes[p_->input_name] = p_->input;

    for (const std::string &ln : p_->graph_lines) {
        const auto tok = split(ln);
        const std::string &op = tok[0];
        ggml_tensor *cur = nullptr;
        if (op == "conv") {
            // conv <out> <in> <k> <ph> <pw> <dh> <dw> <oh> <ow> <has_bias>
            const std::string &out = tok[1], &in = tok[2];
            const int k = std::stoi(tok[3]), ph = std::stoi(tok[4]),
                      pw = std::stoi(tok[5]), dh = std::stoi(tok[6]),
                      dw = std::stoi(tok[7]);
            const bool has_bias = tok.size() > 10 && tok[10] == "1";
            ggml_tensor *w = nodes.at(out + ".weight");
            ggml_tensor *x = nodes.at(in);
            int64_t OC = 0;
            if (w->type == GGML_TYPE_F32) {
                // 4-D kernel {KW,KH,IC,OC} (f32/f16): native conv_2d
                // step-by-step expansion of ggml_conv_2d for fault isolation
                ggml_tensor *im2c = ggml_im2col(gctx, w, x, 1, 1, ph, pw, dh,
                                                dw, true, GGML_TYPE_F32);
                ggml_tensor *mm = ggml_mul_mat(
                        gctx,
                        ggml_reshape_2d(
                                gctx, im2c, im2c->ne[0],
                                im2c->ne[3] * im2c->ne[2] * im2c->ne[1]),
                        ggml_reshape_2d(gctx, w, w->ne[0] * w->ne[1] * w->ne[2],
                                        w->ne[3]));
                ggml_tensor *r4 =
                        ggml_reshape_4d(gctx, mm, im2c->ne[1], im2c->ne[2],
                                        im2c->ne[3], w->ne[3]);
                cur = ggml_cont(gctx, ggml_permute(gctx, r4, 0, 1, 3, 2));
            } else {
                // 2-D kernel {K,OC} (f16/q8_0): im2col + mul_mat with the
                // kernel as src0 (CPU mul_mat requires an f32 src1).
                const int64_t K = w->ne[0];
                OC = w->ne[1];
                const int64_t IC = K / (static_cast<int64_t>(k) * k);
                if (x->ne[2] != IC) {
                    error_ = "conv " + out + ": input channels mismatch";
                    return false;
                }
                ggml_tensor *carrier = ggml_reshape_4d(gctx, w, k, k, IC, OC);
                ggml_tensor *cols = ggml_im2col(gctx, carrier, x, 1, 1, ph, pw,
                                                dh, dw, true, GGML_TYPE_F32);
                ggml_tensor *cur2 = ggml_mul_mat(
                        gctx, w,
                        ggml_reshape_2d(gctx, cols, K,
                                        cols->ne[1] * cols->ne[2]));
                // Q8_0 kernels are intentionally quantized inference: opt out
                // of the graph-wide GGML_PREC_F32 contract so Vulkan may
                // quantize the activation operand to Q8_1 and hit its native
                // q8_0 x q8_1 matmul shader (the F32-activation contract
                // exists for the reconstruction model's cache feedback loop,
                // which this single-pass mask network does not have).
                if (w->type == GGML_TYPE_Q8_0) {
                    ggml_mul_mat_set_prec(cur2, GGML_PREC_DEFAULT);
                }
                const int64_t oh = std::stoll(tok[8]), ow = std::stoll(tok[9]);
                cur2 = ggml_reshape_2d(gctx, cur2, OC, oh * ow);
                cur2 = ggml_reshape_3d(gctx, cur2, OC, ow, oh);
                cur = ggml_cont(gctx, ggml_permute(gctx, cur2, 2, 0, 1, 3));
            }
            const int64_t oh = std::stoll(tok[8]), ow = std::stoll(tok[9]);
            if (has_bias) {
                ggml_tensor *b = nodes.at(out + ".bias");
                cur = ggml_add(gctx, cur,
                               ggml_reshape_4d(gctx, b, 1, 1, b->ne[0], 1));
            }
        } else if (op == "relu") {
            cur = ggml_relu(gctx, nodes.at(tok[2]));
        } else if (op == "maxpool") {
            cur = ggml_pool_2d(gctx, nodes.at(tok[2]), GGML_OP_POOL_MAX, 2, 2,
                               2, 2, 0.f, 0.f);
        } else if (op == "resize") {
            // resize <out> <in> <oh> <ow> — bilinear, pytorch_half_pixel
            const int oh = std::stoi(tok[3]), ow = std::stoi(tok[4]);
            ggml_tensor *x = nodes.at(tok[2]);
            cur = ggml_interpolate(gctx, x, ow, oh, x->ne[2], x->ne[3],
                                   GGML_SCALE_MODE_BILINEAR);
        } else if (op == "concat") {
            // both inputs join along C (ggml dim 2), as in the ONNX axis=1
            cur = ggml_concat(gctx, nodes.at(tok[2]), nodes.at(tok[3]), 2);
        } else if (op == "add") {
            cur = ggml_add(gctx, nodes.at(tok[2]), nodes.at(tok[3]));
        } else if (op == "sigmoid") {
            cur = ggml_sigmoid(gctx, nodes.at(tok[2]));
        } else {
            error_ = "skyseg: unknown op in graph: " + op;
            return false;
        }
        nodes[tok[1]] = cur;
    }
    p_->output = nodes.at(p_->output_name);
    ggml_set_name(p_->output, "output");

    p_->graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(p_->graph, p_->output);
    p_->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(p_->be));
    return true;
}

bool skyseg::infer(const float *rgb_hwc,
                   int height,
                   int width,
                   std::vector<float> &sky_map) {
    if (!p_->graph) {
        error_ = "skyseg: model not loaded";
        return false;
    }

    // Official preprocessing: u8 round-trip, the bit-exact cv2 u8 bilinear
    // resize, then ImageNet norm.
    std::vector<unsigned char> u8(static_cast<size_t>(height) * width * 3);
    for (size_t i = 0; i < u8.size(); ++i)
        u8[i] = static_cast<unsigned char>(
                std::lround(std::clamp(rgb_hwc[i], 0.f, 1.f) * 255.f));
    std::vector<unsigned char> small(static_cast<size_t>(kInput) * kInput * 3);
    resize_u8_exact(u8.data(), width, height, 3, small.data(), kInput, kInput);

    std::vector<float> chw(3 * kInput * kInput);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < kInput * kInput; ++i)
            chw[c * kInput * kInput + i] =
                    (small[i * 3 + c] / 255.f - kMean[c]) / kStd[c];

    // Same graph every frame: allocate + compute + read the sigmoid map.
    if (!ggml_gallocr_alloc_graph(p_->galloc, p_->graph)) {
        error_ = "skyseg: graph allocation failed";
        return false;
    }
    ggml_backend_tensor_set(p_->input, chw.data(), 0,
                            chw.size() * sizeof(float));
    ggml_backend_graph_compute(p_->be, p_->graph);
    sky_map.assign(static_cast<size_t>(kInput) * kInput, 0.f);
    ggml_backend_tensor_get(p_->output, sky_map.data(), 0,
                            sky_map.size() * sizeof(float));
    return true;
}

// Convenience wrapper used by the CLI: brings the 320x320 u8 mask back to
// the frame size with the same bit-exact cv2 u8 INTER_LINEAR semantics as
// the official segment_sky_from_array postprocess.
void skyseg_resize_mask_u8(const unsigned char *src,
                           int sw,
                           int sh,
                           unsigned char *dst,
                           int dw,
                           int dh) {
    resize_u8_exact(src, sw, sh, 1, dst, dw, dh);
}

}  // namespace lingbot
