// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "deeplsd.hpp"
#include "deeplsd_line_detect.hpp"
#include "ggml_backend_registry.hpp"
#include "gguf_loader.hpp"

namespace deeplsd {
namespace {

ggml_tensor *ConvReluBn(ggml_context *ctx,
                        ggml_tensor *input,
                        ggml_tensor *weight,
                        ggml_tensor *conv_bias,
                        ggml_tensor *bn_scale,
                        ggml_tensor *bn_shift,
                        int32_t stride,
                        int32_t pad) {
    ggml_tensor *conv =
            ggml_conv_2d(ctx, weight, input, stride, stride, pad, pad, 1, 1);
    if (conv_bias != nullptr) {
        conv = ggml_add(ctx, conv, conv_bias);
    }
    conv = ggml_relu(ctx, conv);
    if (bn_scale != nullptr && bn_shift != nullptr) {
        conv = ggml_mul(ctx, conv, bn_scale);
        conv = ggml_add(ctx, conv, bn_shift);
    }
    return conv;
}

struct PendingUpload {
    ggml_tensor *tensor = nullptr;
    std::vector<float> data;
};

std::vector<float> ToGgmlConvWeight(
        const std::vector<float> &pytorch_oc_ic_kh_kw,
        int32_t ic,
        int32_t oc,
        int32_t kh,
        int32_t kw) {
    std::vector<float> out(static_cast<size_t>(kw) * kh * ic * oc, 0.0f);
    for (int32_t o = 0; o < oc; ++o) {
        for (int32_t i = 0; i < ic; ++i) {
            for (int32_t ky = 0; ky < kh; ++ky) {
                for (int32_t kx = 0; kx < kw; ++kx) {
                    const size_t pt_idx =
                            static_cast<size_t>(o) * ic * kh * kw +
                            static_cast<size_t>(i) * kh * kw +
                            static_cast<size_t>(ky) * kw + kx;
                    const size_t ggml_idx =
                            static_cast<size_t>(kx) +
                            static_cast<size_t>(ky) * kw +
                            static_cast<size_t>(i) * kh * kw +
                            static_cast<size_t>(o) * ic * kh * kw;
                    out[ggml_idx] = pytorch_oc_ic_kh_kw[pt_idx];
                }
            }
        }
    }
    return out;
}

ggml_tensor *NewTensor4d(ggml_context *ctx, int64_t w, int64_t h, int64_t c) {
    return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, c, 1);
}

ggml_tensor *MakeKernel(ggml_context *ctx,
                        const std::vector<float> &nchw,
                        int64_t kw,
                        int64_t kh,
                        int32_t ic,
                        int32_t oc,
                        std::vector<PendingUpload> *pending) {
    ggml_tensor *t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kw, kh, ic, oc);
    pending->push_back(
            {t, ToGgmlConvWeight(nchw, ic, oc, static_cast<int32_t>(kh),
                                 static_cast<int32_t>(kw))});
    return t;
}

ggml_tensor *MakeBias(ggml_context *ctx,
                      const std::vector<float> &bias,
                      int64_t oc,
                      std::vector<PendingUpload> *pending) {
    ggml_tensor *t = NewTensor4d(ctx, 1, 1, oc);
    pending->push_back({t, bias});
    return t;
}

void UploadPending(const std::vector<PendingUpload> &pending) {
    for (const PendingUpload &item : pending) {
        if (item.tensor == nullptr) {
            continue;
        }
        ggml_backend_tensor_set(item.tensor, item.data.data(), 0,
                                item.data.size() * sizeof(float));
    }
}

ggml_tensor *UpsampleTo(ggml_context *ctx,
                        ggml_tensor *input,
                        ggml_tensor *ref) {
    return ggml_interpolate(ctx, input, ref->ne[0], ref->ne[1], input->ne[2],
                            input->ne[3], GGML_SCALE_MODE_BILINEAR);
}

void BilinearUpsamplePlane(const std::vector<float> &src,
                           int32_t src_w,
                           int32_t src_h,
                           int32_t dst_w,
                           int32_t dst_h,
                           std::vector<float> *dst) {
    if (dst == nullptr || src_w <= 0 || src_h <= 0 || dst_w <= 0 ||
        dst_h <= 0) {
        return;
    }
    dst->assign(static_cast<size_t>(dst_w) * dst_h, 0.0f);
    if (src_w == dst_w && src_h == dst_h) {
        *dst = src;
        return;
    }
    for (int32_t y = 0; y < dst_h; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) *
                                    static_cast<float>(src_h) /
                                    static_cast<float>(dst_h) -
                            0.5f;
        const int32_t y0 = std::clamp(static_cast<int32_t>(std::floor(src_y)),
                                      0, src_h - 1);
        const int32_t y1 = std::min(y0 + 1, src_h - 1);
        const float wy = src_y - static_cast<float>(y0);
        for (int32_t x = 0; x < dst_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) *
                                        static_cast<float>(src_w) /
                                        static_cast<float>(dst_w) -
                                0.5f;
            const int32_t x0 = std::clamp(
                    static_cast<int32_t>(std::floor(src_x)), 0, src_w - 1);
            const int32_t x1 = std::min(x0 + 1, src_w - 1);
            const float wx = src_x - static_cast<float>(x0);
            const auto sample = [&](int32_t sy, int32_t sx) {
                return src[static_cast<size_t>(sy) * src_w + sx];
            };
            const float v00 = sample(y0, x0);
            const float v01 = sample(y0, x1);
            const float v10 = sample(y1, x0);
            const float v11 = sample(y1, x1);
            const float v0 = v00 * (1.0f - wx) + v01 * wx;
            const float v1 = v10 * (1.0f - wx) + v11 * wx;
            (*dst)[static_cast<size_t>(y) * dst_w + x] =
                    v0 * (1.0f - wy) + v1 * wy;
        }
    }
}

bool Require(const TensorMap &map, const char *key, std::string *error) {
    if (map.count(key) == 0) {
        if (error) {
            *error = std::string("missing tensor: ") + key;
        }
        return false;
    }
    return true;
}

class DeepLSDExtractorImpl : public DeepLSDExtractor {
public:
    explicit DeepLSDExtractorImpl(DeepLSDOptions options)
        : options_(std::move(options)) {
        if (!LoadGguf(options_.model_path, &weights_, &init_error_)) {
            return;
        }
        backend_lease_ = aicore::runtime::acquire_backend_lease(
                options_.device, options_.num_threads, &init_error_);
        backend_ = backend_lease_.handle();
        if (backend_ == nullptr) {
            if (init_error_.empty()) init_error_ = "failed to init backend";
            return;
        }
        device_ = backend_lease_.device();
        const auto backend_lock = backend_lease_.lock();
        galloc_ = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr && init_error_.empty()) {
            init_error_ = "gallocr init failed";
        }
    }

    ~DeepLSDExtractorImpl() override {
        if (galloc_ != nullptr) {
            ggml_gallocr_free(galloc_);
        }
        backend_ = nullptr;
        backend_lease_.reset();
    }

    bool ExtractFromGray(const uint8_t *gray,
                         int32_t width,
                         int32_t height,
                         int32_t row_stride,
                         DeepLSDResult *result) override {
        error_.clear();
        if (result == nullptr) {
            error_ = "null result";
            return false;
        }
        if (!init_error_.empty() || backend_ == nullptr) {
            error_ = init_error_.empty() ? "extractor not initialized"
                                         : init_error_;
            return false;
        }

        std::vector<float> input(static_cast<size_t>(width) * height);
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                input[static_cast<size_t>(y) * width + x] =
                        static_cast<float>(
                                gray[static_cast<size_t>(y) * row_stride + x]) /
                        255.0f;
            }
        }

        std::vector<float> df;
        std::vector<float> angle;
        if (!RunGraph(input, width, height, &df, &angle, &error_)) {
            return false;
        }

        result->width = width;
        result->height = height;
        result->distance_field = std::move(df);
        result->angle_field = std::move(angle);
        std::string line_err;
        if (!DetectAfmLines(gray, width, height, row_stride,
                            result->distance_field.data(),
                            result->angle_field.data(), &result->segments,
                            &line_err)) {
            error_ = line_err.empty() ? "line detection failed" : line_err;
            return false;
        }
        return true;
    }

    const std::string &Device() const override { return device_; }
    const std::string &Error() const override {
        return error_.empty() ? init_error_ : error_;
    }

private:
    bool RunGraph(const std::vector<float> &input,
                  int32_t w,
                  int32_t h,
                  std::vector<float> *df,
                  std::vector<float> *angle,
                  std::string *error) {
        const auto backend_lock = backend_lease_.lock();
        if (!Require(weights_, "backbone_block1_0_weight", error)) {
            return false;
        }

        ggml_init_params params{128 * 1024 * 1024, nullptr, true};
        ggml_context *ctx = ggml_init(params);
        if (ctx == nullptr) {
            if (error) {
                *error = "ggml_init failed";
            }
            return false;
        }

        std::vector<PendingUpload> pending;

        auto convrb = [&](const char *prefix, ggml_tensor *in, int32_t ic,
                          int32_t oc, int32_t stride,
                          int32_t pad) -> ggml_tensor * {
            const std::string wkey = std::string(prefix) + "_weight";
            const std::string bkey = std::string(prefix) + "_conv_bias";
            const std::string skey = std::string(prefix) + "_scale";
            const std::string tkey = std::string(prefix) + "_shift";
            if (!Require(weights_, wkey.c_str(), error) ||
                !Require(weights_, skey.c_str(), error) ||
                !Require(weights_, tkey.c_str(), error)) {
                return nullptr;
            }
            ggml_tensor *weight =
                    MakeKernel(ctx, weights_.at(wkey), 3, 3, ic, oc, &pending);
            ggml_tensor *conv_bias = nullptr;
            if (weights_.count(bkey)) {
                conv_bias = MakeBias(ctx, weights_.at(bkey), oc, &pending);
            }
            ggml_tensor *scale = MakeBias(ctx, weights_.at(skey), oc, &pending);
            ggml_tensor *shift = MakeBias(ctx, weights_.at(tkey), oc, &pending);
            return ConvReluBn(ctx, in, weight, conv_bias, scale, shift, stride,
                              pad);
        };

        ggml_tensor *inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, 1, 1);

        ggml_tensor *e1 = convrb("backbone_block1_0", inp, 1, 64, 1, 1);
        if (e1 == nullptr) {
            ggml_free(ctx);
            return false;
        }
        e1 = convrb("backbone_block1_3", e1, 64, 64, 1, 1);
        ggml_tensor *p1 =
                ggml_pool_2d(ctx, e1, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

        ggml_tensor *e2 = convrb("backbone_block2_0", p1, 64, 128, 1, 1);
        e2 = convrb("backbone_block2_3", e2, 128, 128, 1, 1);
        ggml_tensor *p2 =
                ggml_pool_2d(ctx, e2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

        ggml_tensor *e3 = convrb("backbone_block3_0", p2, 128, 256, 1, 1);
        e3 = convrb("backbone_block3_3", e3, 256, 256, 1, 1);
        ggml_tensor *p3 =
                ggml_pool_2d(ctx, e3, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

        ggml_tensor *e4 = convrb("backbone_block4_0", p3, 256, 512, 1, 1);
        e4 = convrb("backbone_block4_3", e4, 512, 512, 1, 1);

        ggml_tensor *d4 = convrb("backbone_deblock4_0", e4, 512, 256, 1, 1);
        d4 = convrb("backbone_deblock4_3", d4, 256, 256, 1, 1);

        ggml_tensor *up3 = UpsampleTo(ctx, d4, e3);
        ggml_tensor *c3 = ggml_concat(ctx, up3, e3, 2);
        ggml_tensor *d3 = convrb("backbone_deblock3_0", c3, 512, 256, 1, 1);
        d3 = convrb("backbone_deblock3_3", d3, 256, 128, 1, 1);

        ggml_tensor *up2 = UpsampleTo(ctx, d3, e2);
        ggml_tensor *c2 = ggml_concat(ctx, up2, e2, 2);
        ggml_tensor *d2 = convrb("backbone_deblock2_0", c2, 256, 128, 1, 1);
        d2 = convrb("backbone_deblock2_3", d2, 128, 64, 1, 1);

        ggml_tensor *up1 = UpsampleTo(ctx, d2, e1);
        ggml_tensor *c1 = ggml_concat(ctx, up1, e1, 2);
        ggml_tensor *d1 = convrb("backbone_deblock1_0", c1, 128, 64, 1, 1);
        d1 = convrb("backbone_deblock1_3", d1, 64, 64, 1, 1);

        ggml_tensor *df_head = convrb("df_head_0", d1, 64, 64, 1, 1);
        if (df_head == nullptr) {
            ggml_free(ctx);
            return false;
        }
        df_head = convrb("df_head_3", df_head, 64, 64, 1, 1);
        if (df_head == nullptr ||
            !Require(weights_, "df_head_6_weight", error)) {
            ggml_free(ctx);
            return false;
        }
        df_head = ggml_conv_2d(ctx,
                               MakeKernel(ctx, weights_.at("df_head_6_weight"),
                                          1, 1, 64, 1, &pending),
                               df_head, 1, 1, 0, 0, 1, 1);
        if (weights_.count("df_head_6_bias")) {
            df_head = ggml_add(
                    ctx, df_head,
                    MakeBias(ctx, weights_.at("df_head_6_bias"), 1, &pending));
        }
        df_head = ggml_relu(ctx, df_head);

        ggml_tensor *ang_head = convrb("angle_head_0", d1, 64, 64, 1, 1);
        if (ang_head == nullptr) {
            ggml_free(ctx);
            return false;
        }
        ang_head = convrb("angle_head_3", ang_head, 64, 64, 1, 1);
        if (ang_head == nullptr ||
            !Require(weights_, "angle_head_6_weight", error)) {
            ggml_free(ctx);
            return false;
        }
        ang_head =
                ggml_conv_2d(ctx,
                             MakeKernel(ctx, weights_.at("angle_head_6_weight"),
                                        1, 1, 64, 1, &pending),
                             ang_head, 1, 1, 0, 0, 1, 1);
        if (weights_.count("angle_head_6_bias")) {
            ang_head = ggml_add(ctx, ang_head,
                                MakeBias(ctx, weights_.at("angle_head_6_bias"),
                                         1, &pending));
        }
        ang_head = ggml_sigmoid(ctx, ang_head);

        ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, df_head);
        ggml_build_forward_expand(gf, ang_head);

        if (galloc_ == nullptr || !ggml_gallocr_alloc_graph(galloc_, gf)) {
            ggml_free(ctx);
            if (error) {
                *error = "graph alloc failed";
            }
            return false;
        }

        UploadPending(pending);
        ggml_backend_tensor_set(inp, input.data(), 0,
                                input.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            if (error) {
                *error = "graph compute failed";
            }
            return false;
        }

        const int32_t ow = static_cast<int32_t>(df_head->ne[0]);
        const int32_t oh = static_cast<int32_t>(df_head->ne[1]);
        const size_t out_plane = static_cast<size_t>(ow) * oh;
        std::vector<float> df_low(out_plane);
        std::vector<float> ang_low(out_plane);
        ggml_backend_tensor_get(df_head, df_low.data(), 0,
                                out_plane * sizeof(float));
        ggml_backend_tensor_get(ang_head, ang_low.data(), 0,
                                out_plane * sizeof(float));

        ggml_free(ctx);

        if (ow == w && oh == h) {
            *df = std::move(df_low);
            *angle = std::move(ang_low);
        } else {
            BilinearUpsamplePlane(df_low, ow, oh, w, h, df);
            BilinearUpsamplePlane(ang_low, ow, oh, w, h, angle);
        }
        return true;
    }

    DeepLSDOptions options_;
    TensorMap weights_;
    aicore::runtime::BackendLease backend_lease_;
    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    std::string device_;
    std::string init_error_;
    std::string error_;
};

}  // namespace

std::unique_ptr<DeepLSDExtractor> CreateDeepLSDExtractor(
        const DeepLSDOptions &options, std::string *error) {
    auto impl = std::make_unique<DeepLSDExtractorImpl>(options);
    if (!impl->Error().empty() && error != nullptr) {
        *error = impl->Error();
        return nullptr;
    }
    return impl;
}

}  // namespace deeplsd
