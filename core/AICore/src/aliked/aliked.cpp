// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "lightglue/aliked.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

#include "aliked_gpu_ops.hpp"
#include "backend.h"
#include "deform_conv.hpp"
#include "ggml_cnn.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_pipeline_cache.hpp"
#include "gpu_postprocess.hpp"
#include "model_weights.hpp"
#include "postprocess.hpp"
#include "tensor_ops.hpp"

namespace lightglue {
namespace {

using namespace aliked_internal;

struct InputPadder {
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    int32_t pad_top = 0;
    int32_t pad_bottom = 0;
    int32_t padded_h = 0;
    int32_t padded_w = 0;

    explicit InputPadder(int32_t h, int32_t w, int32_t divisor) {
        const int32_t pad_h = ((h + divisor - 1) / divisor) * divisor - h;
        const int32_t pad_w = ((w + divisor - 1) / divisor) * divisor - w;
        pad_left = pad_w / 2;
        pad_right = pad_w - pad_left;
        pad_top = pad_h / 2;
        pad_bottom = pad_h - pad_top;
        padded_h = h + pad_h;
        padded_w = w + pad_w;
    }

    std::vector<float> Pad(const std::vector<float> &input,
                           int32_t c,
                           int32_t h,
                           int32_t w) const {
        std::vector<float> output(static_cast<size_t>(c) * padded_h * padded_w,
                                  0.0f);
        for (int32_t ch = 0; ch < c; ++ch) {
            for (int32_t y = 0; y < padded_h; ++y) {
                const int32_t src_y = std::min(std::max(y - pad_top, 0), h - 1);
                for (int32_t x = 0; x < padded_w; ++x) {
                    const int32_t src_x =
                            std::min(std::max(x - pad_left, 0), w - 1);
                    output[static_cast<size_t>(ch) * padded_h * padded_w +
                           y * padded_w + x] =
                            input[static_cast<size_t>(ch) * h * w + src_y * w +
                                  src_x];
                }
            }
        }
        return output;
    }

    void Unpad(std::vector<float> *tensor,
               int32_t c,
               int32_t orig_h,
               int32_t orig_w) const {
        std::vector<float> cropped(static_cast<size_t>(c) * orig_h * orig_w,
                                   0.0f);
        for (int32_t ch = 0; ch < c; ++ch) {
            for (int32_t y = 0; y < orig_h; ++y) {
                for (int32_t x = 0; x < orig_w; ++x) {
                    cropped[static_cast<size_t>(ch) * orig_h * orig_w +
                            y * orig_w + x] =
                            (*tensor)[static_cast<size_t>(ch) * padded_h *
                                              padded_w +
                                      (y + pad_top) * padded_w +
                                      (x + pad_left)];
                }
            }
        }
        *tensor = std::move(cropped);
    }
};

void ResizeLongEdge(const uint8_t *rgb,
                    int32_t width,
                    int32_t height,
                    int32_t row_stride,
                    int32_t long_edge,
                    std::vector<float> *image,
                    int32_t *out_w,
                    int32_t *out_h) {
    const float scale = static_cast<float>(long_edge) /
                        static_cast<float>(std::max(width, height));
    *out_w = std::max(1, static_cast<int32_t>(std::lround(width * scale)));
    *out_h = std::max(1, static_cast<int32_t>(std::lround(height * scale)));
    image->assign(static_cast<size_t>(*out_w) * *out_h * 3, 0.0f);
    for (int32_t y = 0; y < *out_h; ++y) {
        const float src_y = (*out_h > 1)
                                    ? y * static_cast<float>(height - 1) /
                                              static_cast<float>(*out_h - 1)
                                    : 0.0f;
        for (int32_t x = 0; x < *out_w; ++x) {
            const float src_x = (*out_w > 1)
                                        ? x * static_cast<float>(width - 1) /
                                                  static_cast<float>(*out_w - 1)
                                        : 0.0f;
            const int32_t x0 = static_cast<int32_t>(std::floor(src_x));
            const int32_t y0 = static_cast<int32_t>(std::floor(src_y));
            const int32_t x1 = std::min(x0 + 1, width - 1);
            const int32_t y1 = std::min(y0 + 1, height - 1);
            const float lx = src_x - static_cast<float>(x0);
            const float ly = src_y - static_cast<float>(y0);
            for (int32_t c = 0; c < 3; ++c) {
                const float v00 = rgb[y0 * row_stride + 3 * x0 + c] / 255.0f;
                const float v01 = rgb[y0 * row_stride + 3 * x1 + c] / 255.0f;
                const float v10 = rgb[y1 * row_stride + 3 * x0 + c] / 255.0f;
                const float v11 = rgb[y1 * row_stride + 3 * x1 + c] / 255.0f;
                (*image)[static_cast<size_t>(c) * (*out_h) * (*out_w) +
                         y * (*out_w) + x] = (1.0f - ly) * (1.0f - lx) * v00 +
                                             (1.0f - ly) * lx * v01 +
                                             ly * (1.0f - lx) * v10 +
                                             ly * lx * v11;
            }
        }
    }
}

void ConvBn(const std::vector<float> &input,
            int32_t ic,
            int32_t ih,
            int32_t iw,
            const std::vector<float> &weight,
            int32_t oc,
            int32_t kh,
            int32_t kw,
            const std::vector<float> &gamma,
            const std::vector<float> &beta,
            const std::vector<float> &mean,
            const std::vector<float> &var,
            std::vector<float> *output,
            int32_t *oh,
            int32_t *ow) {
    std::vector<float> conv;
    Conv2d(input, ic, ih, iw, weight, oc, kh, kw, nullptr, 1, 1, &conv, oh, ow);
    BatchNorm2d(conv, oc, *oh, *ow, gamma, beta, mean, var, output);
}

void ConvBnSelu(const std::vector<float> &input,
                int32_t ic,
                int32_t ih,
                int32_t iw,
                const std::vector<float> &weight,
                int32_t oc,
                int32_t kh,
                int32_t kw,
                const std::vector<float> &gamma,
                const std::vector<float> &beta,
                const std::vector<float> &mean,
                const std::vector<float> &var,
                std::vector<float> *output,
                int32_t *oh,
                int32_t *ow) {
    ConvBn(input, ic, ih, iw, weight, oc, kh, kw, gamma, beta, mean, var,
           output, oh, ow);
    ApplySelu(output);
}

void DcnConvBn(const std::vector<float> &input,
               int32_t ic,
               int32_t ih,
               int32_t iw,
               const std::vector<float> &offset_w,
               const std::vector<float> &offset_b,
               const std::vector<float> &regular_w,
               int32_t oc,
               const std::vector<float> &gamma,
               const std::vector<float> &beta,
               const std::vector<float> &mean,
               const std::vector<float> &var,
               std::vector<float> *output,
               int32_t *oh,
               int32_t *ow) {
    std::vector<float> offset;
    int32_t offset_h = 0;
    int32_t offset_wd = 0;
    Conv2d(input, ic, ih, iw, offset_w, 18, 3, 3, &offset_b, 1, 1, &offset,
           &offset_h, &offset_wd);
    const float max_offset = std::max(ih, iw) / 4.0f;
    for (float &value : offset) {
        value = std::max(-max_offset, std::min(max_offset, value));
    }
    std::vector<float> conv;
    DeformConv2d(input, ic, ih, iw, offset, 1, regular_w, oc, 3, 3, nullptr, 1,
                 &conv, oh, ow);
    BatchNorm2d(conv, oc, *oh, *ow, gamma, beta, mean, var, output);
}

void ConvBnGgml(GgmlConvRunner *runner,
                const std::vector<float> &input,
                int32_t ic,
                int32_t ih,
                int32_t iw,
                const std::vector<float> &weight,
                int32_t oc,
                int32_t kh,
                int32_t kw,
                const std::vector<float> &gamma,
                const std::vector<float> &beta,
                const std::vector<float> &mean,
                const std::vector<float> &var,
                int32_t pad,
                int32_t stride,
                std::vector<float> *output,
                int32_t *oh,
                int32_t *ow,
                std::string *error,
                const char *cache_key = nullptr) {
    const FusedConv2d fused =
            FuseConvBn(weight, oc, ic, kh, kw, nullptr, gamma, beta, mean, var);
    runner->Run(fused, input, ih, iw, pad, stride, output, oh, ow, error,
                cache_key);
}

void ConvBnSeluGgml(GgmlConvRunner *runner,
                    const std::vector<float> &input,
                    int32_t ic,
                    int32_t ih,
                    int32_t iw,
                    const std::vector<float> &weight,
                    int32_t oc,
                    int32_t kh,
                    int32_t kw,
                    const std::vector<float> &gamma,
                    const std::vector<float> &beta,
                    const std::vector<float> &mean,
                    const std::vector<float> &var,
                    int32_t pad,
                    int32_t stride,
                    std::vector<float> *output,
                    int32_t *oh,
                    int32_t *ow,
                    std::string *error,
                    const char *cache_key = nullptr) {
    ConvBnGgml(runner, input, ic, ih, iw, weight, oc, kh, kw, gamma, beta, mean,
               var, pad, stride, output, oh, ow, error, cache_key);
    if (error != nullptr && !error->empty()) {
        return;
    }
    ApplySelu(output);
}

void ConvGgml(GgmlConvRunner *runner,
              const std::vector<float> &input,
              int32_t ic,
              int32_t ih,
              int32_t iw,
              const std::vector<float> &weight,
              int32_t oc,
              int32_t kh,
              int32_t kw,
              const std::vector<float> *bias,
              int32_t pad,
              int32_t stride,
              std::vector<float> *output,
              int32_t *oh,
              int32_t *ow,
              std::string *error,
              const char *cache_key = nullptr) {
    std::vector<float> ones(static_cast<size_t>(oc), 1.0f);
    std::vector<float> zeros(static_cast<size_t>(oc), 0.0f);
    const FusedConv2d fused =
            FuseConvBn(weight, oc, ic, kh, kw, bias, ones, zeros, zeros, ones);
    runner->Run(fused, input, ih, iw, pad, stride, output, oh, ow, error,
                cache_key);
}

void ConvSeluGgml(GgmlConvRunner *runner,
                  const std::vector<float> &input,
                  int32_t ic,
                  int32_t ih,
                  int32_t iw,
                  const std::vector<float> &weight,
                  int32_t oc,
                  int32_t kh,
                  int32_t kw,
                  const std::vector<float> *bias,
                  int32_t pad,
                  int32_t stride,
                  std::vector<float> *output,
                  int32_t *oh,
                  int32_t *ow,
                  std::string *error,
                  const char *cache_key = nullptr) {
    ConvGgml(runner, input, ic, ih, iw, weight, oc, kh, kw, bias, pad, stride,
             output, oh, ow, error, cache_key);
    if (error != nullptr && !error->empty()) {
        return;
    }
    ApplySelu(output);
}

void ResBlockForward(const std::vector<float> &input,
                     int32_t ic,
                     int32_t oc,
                     int32_t ih,
                     int32_t iw,
                     const TensorMap &tensors,
                     const std::string &prefix,
                     bool dcn,
                     GgmlConvRunner *ggml,
                     std::vector<float> *output,
                     int32_t *oh,
                     int32_t *ow,
                     std::string *error) {
    std::vector<float> conv1;
    int32_t h1 = 0;
    int32_t w1 = 0;
    if (dcn) {
        DcnConvBn(input, ic, ih, iw,
                  RequireTensor(tensors, prefix + "_conv1_offset_conv_weight",
                                error),
                  RequireTensor(tensors, prefix + "_conv1_offset_conv_bias",
                                error),
                  RequireTensor(tensors, prefix + "_conv1_regular_conv_weight",
                                error),
                  oc, RequireTensor(tensors, prefix + "_bn1_weight", error),
                  RequireTensor(tensors, prefix + "_bn1_bias", error),
                  RequireTensor(tensors, prefix + "_bn1_running_mean", error),
                  RequireTensor(tensors, prefix + "_bn1_running_var", error),
                  &conv1, &h1, &w1);
    } else if (ggml != nullptr) {
        ConvBnGgml(ggml, input, ic, ih, iw,
                   RequireTensor(tensors, prefix + "_conv1_weight", error), oc,
                   3, 3, RequireTensor(tensors, prefix + "_bn1_weight", error),
                   RequireTensor(tensors, prefix + "_bn1_bias", error),
                   RequireTensor(tensors, prefix + "_bn1_running_mean", error),
                   RequireTensor(tensors, prefix + "_bn1_running_var", error),
                   1, 1, &conv1, &h1, &w1, error, (prefix + ".conv1").c_str());
    } else {
        ConvBn(input, ic, ih, iw,
               RequireTensor(tensors, prefix + "_conv1_weight", error), oc, 3,
               3, RequireTensor(tensors, prefix + "_bn1_weight", error),
               RequireTensor(tensors, prefix + "_bn1_bias", error),
               RequireTensor(tensors, prefix + "_bn1_running_mean", error),
               RequireTensor(tensors, prefix + "_bn1_running_var", error),
               &conv1, &h1, &w1);
    }
    ApplySelu(&conv1);

    std::vector<float> conv2;
    int32_t h2 = 0;
    int32_t w2 = 0;
    if (dcn) {
        DcnConvBn(conv1, oc, h1, w1,
                  RequireTensor(tensors, prefix + "_conv2_offset_conv_weight",
                                error),
                  RequireTensor(tensors, prefix + "_conv2_offset_conv_bias",
                                error),
                  RequireTensor(tensors, prefix + "_conv2_regular_conv_weight",
                                error),
                  oc, RequireTensor(tensors, prefix + "_bn2_weight", error),
                  RequireTensor(tensors, prefix + "_bn2_bias", error),
                  RequireTensor(tensors, prefix + "_bn2_running_mean", error),
                  RequireTensor(tensors, prefix + "_bn2_running_var", error),
                  &conv2, &h2, &w2);
    } else if (ggml != nullptr) {
        ConvBnGgml(ggml, conv1, oc, h1, w1,
                   RequireTensor(tensors, prefix + "_conv2_weight", error), oc,
                   3, 3, RequireTensor(tensors, prefix + "_bn2_weight", error),
                   RequireTensor(tensors, prefix + "_bn2_bias", error),
                   RequireTensor(tensors, prefix + "_bn2_running_mean", error),
                   RequireTensor(tensors, prefix + "_bn2_running_var", error),
                   1, 1, &conv2, &h2, &w2, error, (prefix + ".conv2").c_str());
    } else {
        ConvBn(conv1, oc, h1, w1,
               RequireTensor(tensors, prefix + "_conv2_weight", error), oc, 3,
               3, RequireTensor(tensors, prefix + "_bn2_weight", error),
               RequireTensor(tensors, prefix + "_bn2_bias", error),
               RequireTensor(tensors, prefix + "_bn2_running_mean", error),
               RequireTensor(tensors, prefix + "_bn2_running_var", error),
               &conv2, &h2, &w2);
    }

    std::vector<float> identity = input;
    int32_t identity_h = ih;
    int32_t identity_w = iw;
    if (tensors.count(prefix + "_downsample_weight") > 0) {
        const std::vector<float> &down_w =
                RequireTensor(tensors, prefix + "_downsample_weight", error);
        const std::vector<float> &down_b =
                RequireTensor(tensors, prefix + "_downsample_bias", error);
        if (ggml != nullptr) {
            std::vector<float> ones(static_cast<size_t>(oc), 1.0f);
            std::vector<float> zeros(static_cast<size_t>(oc), 0.0f);
            const FusedConv2d fused = FuseConvBn(down_w, oc, ic, 1, 1, &down_b,
                                                 ones, zeros, zeros, ones);
            if (!ggml->Run(fused, input, ih, iw, 0, 1, &identity, &identity_h,
                           &identity_w, error,
                           (prefix + ".downsample").c_str())) {
                return;
            }
        } else {
            Conv2d(input, ic, ih, iw, down_w, oc, 1, 1, &down_b, 0, 1,
                   &identity, &identity_h, &identity_w);
        }
    } else if (ic != oc) {
        error->assign("residual channel mismatch without downsample for " +
                      prefix);
        return;
    }

    output->resize(conv2.size());
    for (size_t i = 0; i < conv2.size(); ++i) {
        (*output)[i] = Selu(conv2[i] + identity[i]);
    }
    *oh = h2;
    *ow = w2;
}

bool ExtractDenseMap(const TensorMap &tensors,
                     const std::vector<float> &image,
                     int32_t width,
                     int32_t height,
                     int32_t orig_h,
                     int32_t orig_w,
                     std::vector<float> *feature_map,
                     std::vector<float> *score_map,
                     std::string *error,
                     internal::Backend *ggml_backend,
                     bool use_ggml_cnn) {
    GgmlConvRunner ggml_runner(ggml_backend);
    const bool ggml_on_device =
            use_ggml_cnn && ggml_backend != nullptr &&
            (ggml_backend->IsCpu() || ggml_backend->IsVulkan());
    GgmlConvRunner *ggml = ggml_on_device ? &ggml_runner : nullptr;

    // Full VRAM pipeline: CUDA (custom kernels) or Vulkan (GGML + DCN CPU
    // bridge).
    if (use_ggml_cnn && ggml_backend != nullptr && ggml_backend->IsGpu()) {
        return ExtractDenseMapGpu(tensors, image, width, height, orig_h, orig_w,
                                  feature_map, score_map, error, ggml_backend);
    }

    InputPadder padder(height, width, 32);
    std::vector<float> padded = padder.Pad(image, 3, height, width);

    std::vector<float> x1;
    int32_t h1 = 0;
    int32_t w1 = 0;
    if (ggml != nullptr) {
        ConvBnSeluGgml(ggml, padded, 3, padder.padded_h, padder.padded_w,
                       RequireTensor(tensors, "block1_conv1_weight", error), 16,
                       3, 3, RequireTensor(tensors, "block1_bn1_weight", error),
                       RequireTensor(tensors, "block1_bn1_bias", error),
                       RequireTensor(tensors, "block1_bn1_running_mean", error),
                       RequireTensor(tensors, "block1_bn1_running_var", error),
                       1, 1, &x1, &h1, &w1, error, "block1.conv1");
        if (!error->empty()) {
            return false;
        }
        ConvBnSeluGgml(ggml, x1, 16, h1, w1,
                       RequireTensor(tensors, "block1_conv2_weight", error), 16,
                       3, 3, RequireTensor(tensors, "block1_bn2_weight", error),
                       RequireTensor(tensors, "block1_bn2_bias", error),
                       RequireTensor(tensors, "block1_bn2_running_mean", error),
                       RequireTensor(tensors, "block1_bn2_running_var", error),
                       1, 1, &x1, &h1, &w1, error, "block1.conv2");
        if (!error->empty()) {
            return false;
        }
    } else {
        ConvBnSelu(padded, 3, padder.padded_h, padder.padded_w,
                   RequireTensor(tensors, "block1_conv1_weight", error), 16, 3,
                   3, RequireTensor(tensors, "block1_bn1_weight", error),
                   RequireTensor(tensors, "block1_bn1_bias", error),
                   RequireTensor(tensors, "block1_bn1_running_mean", error),
                   RequireTensor(tensors, "block1_bn1_running_var", error), &x1,
                   &h1, &w1);
        ConvBnSelu(x1, 16, h1, w1,
                   RequireTensor(tensors, "block1_conv2_weight", error), 16, 3,
                   3, RequireTensor(tensors, "block1_bn2_weight", error),
                   RequireTensor(tensors, "block1_bn2_bias", error),
                   RequireTensor(tensors, "block1_bn2_running_mean", error),
                   RequireTensor(tensors, "block1_bn2_running_var", error), &x1,
                   &h1, &w1);
    }

    std::vector<float> x2;
    int32_t h2 = 0;
    int32_t w2 = 0;
    AvgPool2d(x1, 16, h1, w1, 2, 2, 2, &x2, &h2, &w2);
    ResBlockForward(x2, 16, 32, h2, w2, tensors, "block2", false, ggml, &x2,
                    &h2, &w2, error);
    if (!error->empty()) {
        return false;
    }

    std::vector<float> x3;
    int32_t h3 = 0;
    int32_t w3 = 0;
    AvgPool2d(x2, 32, h2, w2, 4, 4, 4, &x3, &h3, &w3);
    ResBlockForward(x3, 32, 64, h3, w3, tensors, "block3", true, nullptr, &x3,
                    &h3, &w3, error);
    if (!error->empty()) {
        return false;
    }

    std::vector<float> x4;
    int32_t h4 = 0;
    int32_t w4 = 0;
    AvgPool2d(x3, 64, h3, w3, 4, 4, 4, &x4, &h4, &w4);
    ResBlockForward(x4, 64, 128, h4, w4, tensors, "block4", true, nullptr, &x4,
                    &h4, &w4, error);
    if (!error->empty()) {
        return false;
    }

    auto project = [&](const std::vector<float> &src, int32_t ic, int32_t ih,
                       int32_t iw, const char *weight_name,
                       std::vector<float> *dst) {
        int32_t oh = 0;
        int32_t ow = 0;
        if (ggml != nullptr) {
            ConvSeluGgml(ggml, src, ic, ih, iw,
                         RequireTensor(tensors, weight_name, error), 32, 1, 1,
                         nullptr, 0, 1, dst, &oh, &ow, error, weight_name);
        } else {
            Conv2d(src, ic, ih, iw, RequireTensor(tensors, weight_name, error),
                   32, 1, 1, nullptr, 0, 1, dst, &oh, &ow);
            ApplySelu(dst);
        }
        return std::make_tuple(oh, ow);
    };

    int32_t fh = 0;
    int32_t fw = 0;
    std::vector<float> f1;
    project(x1, 16, h1, w1, "conv1_weight", &f1);
    fh = h1;
    fw = w1;
    std::vector<float> f2;
    auto [h2p, w2p] = project(x2, 32, h2, w2, "conv2_weight", &f2);
    (void)h2p;
    (void)w2p;
    UpsampleBilinear(f2, 32, h2, w2, fh, fw, &f2);
    std::vector<float> f3;
    project(x3, 64, h3, w3, "conv3_weight", &f3);
    UpsampleBilinear(f3, 32, h3, w3, fh, fw, &f3);
    std::vector<float> f4;
    project(x4, 128, h4, w4, "conv4_weight", &f4);
    UpsampleBilinear(f4, 32, h4, w4, fh, fw, &f4);

    std::vector<float> fused;
    ConcatChannel(f1, 32, f2, 32, fh, fw, &fused);
    std::vector<float> fused2;
    ConcatChannel(fused, 64, f3, 32, fh, fw, &fused2);
    ConcatChannel(fused2, 96, f4, 32, fh, fw, feature_map);

    std::vector<float> score;
    if (ggml != nullptr) {
        ConvSeluGgml(ggml, *feature_map, 128, fh, fw,
                     RequireTensor(tensors, "score_head_0_weight", error), 8, 1,
                     1, nullptr, 0, 1, &score, &fh, &fw, error, "score_head_0");
        if (!error->empty()) {
            return false;
        }
        ConvSeluGgml(ggml, score, 8, fh, fw,
                     RequireTensor(tensors, "score_head_2_weight", error), 4, 3,
                     3, nullptr, 1, 1, &score, &fh, &fw, error, "score_head_2");
        if (!error->empty()) {
            return false;
        }
        ConvSeluGgml(ggml, score, 4, fh, fw,
                     RequireTensor(tensors, "score_head_4_weight", error), 4, 3,
                     3, nullptr, 1, 1, &score, &fh, &fw, error, "score_head_4");
        if (!error->empty()) {
            return false;
        }
        ConvGgml(ggml, score, 4, fh, fw,
                 RequireTensor(tensors, "score_head_6_weight", error), 1, 3, 3,
                 nullptr, 1, 1, score_map, &fh, &fw, error, "score_head_6");
        if (!error->empty()) {
            return false;
        }
    } else {
        Conv2d(*feature_map, 128, fh, fw,
               RequireTensor(tensors, "score_head_0_weight", error), 8, 1, 1,
               nullptr, 0, 1, &score, &fh, &fw);
        ApplySelu(&score);
        Conv2d(score, 8, fh, fw,
               RequireTensor(tensors, "score_head_2_weight", error), 4, 3, 3,
               nullptr, 1, 1, &score, &fh, &fw);
        ApplySelu(&score);
        Conv2d(score, 4, fh, fw,
               RequireTensor(tensors, "score_head_4_weight", error), 4, 3, 3,
               nullptr, 1, 1, &score, &fh, &fw);
        ApplySelu(&score);
        Conv2d(score, 4, fh, fw,
               RequireTensor(tensors, "score_head_6_weight", error), 1, 3, 3,
               nullptr, 1, 1, score_map, &fh, &fw);
    }
    Sigmoid(score_map);
    L2NormalizeChannels(feature_map, 128, fh, fw);

    padder.Unpad(feature_map, 128, orig_h, orig_w);
    padder.Unpad(score_map, 1, orig_h, orig_w);
    return error->empty();
}

class AlikedFeatureExtractorImpl final : public AlikedFeatureExtractor {
public:
    AlikedFeatureExtractorImpl(AlikedExtractionOptions options,
                               TensorMap tensors,
                               int32_t descriptor_dim)
        : options_(std::move(options)),
          tensors_(std::move(tensors)),
          descriptor_dim_(descriptor_dim) {
        if (options_.use_ggml_cnn) {
            if (!backend_.Init(options_.device, options_.num_threads)) {
                init_error_ = backend_.error;
            } else {
                device_ = backend_.device;
                if (backend_.IsGpu()) {
                    gpu_cache_ = std::make_unique<GpuPipelineCache>(&backend_);
                    if (!gpu_cache_->Warmup(tensors_, &init_error_)) {
                        gpu_cache_.reset();
                    }
                }
            }
        } else {
            device_ = "cpu-ref";
        }
    }

    bool ExtractFromRgb(const uint8_t *rgb,
                        int32_t width,
                        int32_t height,
                        int32_t row_stride,
                        Features *features) override {
        error_.clear();
        if (!init_error_.empty()) {
            error_ = init_error_;
            return false;
        }
        if (features == nullptr) {
            error_ = "null features output";
            return false;
        }

        std::vector<float> resized;
        int32_t resized_w = 0;
        int32_t resized_h = 0;
        ResizeLongEdge(rgb, width, height, row_stride,
                       options_.resize_long_edge, &resized, &resized_w,
                       &resized_h);

        std::vector<float> feature_map;
        std::vector<float> score_map;
        GpuDenseMaps gpu_maps;
        GpuKeypointResult gpu_kpts;
        GpuTensor gpu_desc;

        const bool gpu_path = options_.use_ggml_cnn && backend_.IsGpu() &&
                              init_error_.empty() && gpu_cache_ != nullptr;

        DkdOutput dkd_out;
        std::vector<float> descriptors;

        if (gpu_path) {
            if (!ExtractDenseMapGpuVram(tensors_, resized, resized_w, resized_h,
                                        resized_h, resized_w, &gpu_maps,
                                        &error_, &backend_, gpu_cache_.get())) {
                return false;
            }

            DkdOptions dkd;
            dkd.radius = options_.nms_radius;
            dkd.top_k = options_.max_keypoints;
            dkd.scores_th = options_.detection_threshold;
            dkd.n_limit =
                    options_.max_keypoints > 0 ? options_.max_keypoints : 20000;

            if (!RunDkdDispatch(gpu_maps.score, gpu_maps.height, gpu_maps.width,
                                dkd, &backend_, &gpu_kpts, &error_,
                                gpu_cache_.get())) {
                return false;
            }

            std::vector<float> kpts_host(static_cast<size_t>(gpu_kpts.count) *
                                         2);
            ggml_backend_tensor_get(gpu_kpts.keypoints_norm.tensor,
                                    kpts_host.data(), 0,
                                    kpts_host.size() * sizeof(float));
            if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
                const size_t n = kpts_host.size();
                size_t bad = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (!std::isfinite(kpts_host[i])) {
                        ++bad;
                    }
                }
                std::cerr << "dkd kpts count=" << gpu_kpts.count
                          << " non_finite=" << bad << "\n";
            }

            if (!RunSddhDispatch(
                        gpu_maps.feature, descriptor_dim_, gpu_maps.feature.h,
                        gpu_maps.feature.w, kpts_host, gpu_kpts.count, 3, 16,
                        RequireTensor(tensors_,
                                      "desc_head_offset_conv_0_weight",
                                      &error_),
                        RequireTensor(tensors_, "desc_head_offset_conv_0_bias",
                                      &error_),
                        RequireTensor(tensors_,
                                      "desc_head_offset_conv_2_weight",
                                      &error_),
                        RequireTensor(tensors_, "desc_head_offset_conv_2_bias",
                                      &error_),
                        RequireTensor(tensors_, "desc_head_sf_conv_weight",
                                      &error_),
                        RequireTensor(tensors_, "desc_head_agg_weights",
                                      &error_),
                        &backend_, &gpu_desc, &error_, gpu_cache_.get())) {
                return false;
            }

            const int32_t count = gpu_kpts.count;
            dkd_out.scores.resize(static_cast<size_t>(count));
            dkd_out.keypoints_norm.resize(static_cast<size_t>(count) * 2);
            descriptors.resize(static_cast<size_t>(count) * descriptor_dim_);
            ggml_backend_tensor_get(gpu_kpts.scores.tensor,
                                    dkd_out.scores.data(), 0,
                                    static_cast<size_t>(count) * sizeof(float));
            ggml_backend_tensor_get(
                    gpu_kpts.keypoints_norm.tensor,
                    dkd_out.keypoints_norm.data(), 0,
                    static_cast<size_t>(count) * 2 * sizeof(float));
            ggml_backend_tensor_get(gpu_desc.tensor, descriptors.data(), 0,
                                    static_cast<size_t>(count) *
                                            descriptor_dim_ * sizeof(float));
        } else if (!ExtractDenseMap(tensors_, resized, resized_w, resized_h,
                                    resized_h, resized_w, &feature_map,
                                    &score_map, &error_,
                                    options_.use_ggml_cnn ? &backend_ : nullptr,
                                    options_.use_ggml_cnn)) {
            return false;
        } else {
            DkdOptions dkd;
            dkd.radius = options_.nms_radius;
            dkd.top_k = options_.max_keypoints;
            dkd.scores_th = options_.detection_threshold;
            dkd.n_limit =
                    options_.max_keypoints > 0 ? options_.max_keypoints : 20000;

            dkd_out = RunDkd(score_map, resized_h, resized_w, dkd, resized_w,
                             resized_h);

            descriptors = RunSddh(
                    feature_map, descriptor_dim_, resized_h, resized_w,
                    dkd_out.keypoints_norm, 3, 16,
                    RequireTensor(tensors_, "desc_head_offset_conv_0_weight",
                                  &error_),
                    RequireTensor(tensors_, "desc_head_offset_conv_0_bias",
                                  &error_),
                    RequireTensor(tensors_, "desc_head_offset_conv_2_weight",
                                  &error_),
                    RequireTensor(tensors_, "desc_head_offset_conv_2_bias",
                                  &error_),
                    RequireTensor(tensors_, "desc_head_sf_conv_weight",
                                  &error_),
                    RequireTensor(tensors_, "desc_head_agg_weights", &error_));
        }
        if (!error_.empty()) {
            return false;
        }

        const int32_t count = static_cast<int32_t>(dkd_out.scores.size());
        features->keypoints.resize(count);
        features->descriptors.resize(static_cast<size_t>(count) *
                                     descriptor_dim_);
        features->descriptor_dim = descriptor_dim_;
        features->image_width = width;
        features->image_height = height;

        const float wh_x = static_cast<float>(resized_w - 1);
        const float wh_y = static_cast<float>(resized_h - 1);
        const float scale_x =
                static_cast<float>(resized_w) / static_cast<float>(width);
        const float scale_y =
                static_cast<float>(resized_h) / static_cast<float>(height);

        for (int32_t i = 0; i < count; ++i) {
            const float x_norm =
                    dkd_out.keypoints_norm[static_cast<size_t>(i) * 2 + 0];
            const float y_norm =
                    dkd_out.keypoints_norm[static_cast<size_t>(i) * 2 + 1];
            const float x_resized = (x_norm + 1.0f) * 0.5f * wh_x;
            const float y_resized = (y_norm + 1.0f) * 0.5f * wh_y;
            features->keypoints[static_cast<size_t>(i)] = {
                    x_resized / scale_x, y_resized / scale_y, 1.0f, 0.0f};
            for (int32_t d = 0; d < descriptor_dim_; ++d) {
                features->descriptors[static_cast<size_t>(i) * descriptor_dim_ +
                                      d] =
                        descriptors[static_cast<size_t>(i) * descriptor_dim_ +
                                    d];
            }
        }
        return true;
    }

    const std::string &Error() const override { return error_; }
    const std::string &Device() const override { return device_; }

private:
    AlikedExtractionOptions options_;
    TensorMap tensors_;
    int32_t descriptor_dim_ = 128;
    internal::Backend backend_;
    std::unique_ptr<GpuPipelineCache> gpu_cache_;
    std::string device_ = "cpu-ref";
    std::string init_error_;
    std::string error_;
};

}  // namespace

std::unique_ptr<AlikedFeatureExtractor> CreateAlikedFeatureExtractor(
        const AlikedExtractionOptions &options, std::string *error) {
    std::string local_error;
    if (options.model_path.empty()) {
        local_error = "model_path is required";
        if (error != nullptr) {
            *error = local_error;
        }
        return nullptr;
    }

    TensorMap tensors;
    int32_t descriptor_dim = 128;
    if (!aliked_internal::LoadAlikedTensors(options.model_path, &tensors,
                                            &descriptor_dim, &local_error)) {
        if (error != nullptr) {
            *error = local_error;
        }
        return nullptr;
    }

    return std::make_unique<AlikedFeatureExtractorImpl>(
            options, std::move(tensors), descriptor_dim);
}

bool QuantizeAlikedModel(const std::string &input_gguf,
                         const std::string &output_gguf,
                         const std::string &type,
                         std::string *error) {
    (void)input_gguf;
    (void)output_gguf;
    (void)type;
    if (error != nullptr) {
        *error = "ALIKED quantize is not implemented yet; use "
                 "convert_aliked_to_gguf.py";
    }
    return false;
}

bool DumpAlikedDcnParity(const AlikedExtractionOptions &options,
                         const uint8_t *rgb,
                         int32_t width,
                         int32_t height,
                         int32_t row_stride,
                         const std::string &output_dump,
                         std::string *error) {
    ClearAlikedDcnParityEntries();
#if defined(_WIN32)
    _putenv_s("LIGHTGLUE_ALIKED_VULKAN_COMPUTE", "1");
#else
    setenv("LIGHTGLUE_ALIKED_VULKAN_COMPUTE", "1", 1);
    if (std::getenv("LIGHTGLUE_ALIKED_DCN_DEBUG") == nullptr &&
        std::getenv("LIGHTGLUE_ALIKED_FORCE_DCN_DEBUG") != nullptr) {
        setenv("LIGHTGLUE_ALIKED_DCN_DEBUG", "1", 1);
    }
#endif

    AlikedExtractionOptions opts = options;
    opts.use_ggml_cnn = true;
    if (opts.device.empty()) {
        opts.device = "vulkan";
    }

    TensorMap tensors;
    int32_t descriptor_dim = 128;
    if (!LoadAlikedTensors(opts.model_path, &tensors, &descriptor_dim, error)) {
        return false;
    }

    internal::Backend backend;
    if (!backend.Init(opts.device, opts.num_threads)) {
        if (error != nullptr) {
            *error = backend.error;
        }
        return false;
    }
    if (!backend.IsGpu()) {
        if (error != nullptr) {
            *error = "DumpAlikedDcnParity requires a GPU backend (vulkan)";
        }
        return false;
    }

    GpuPipelineCache cache(&backend);
    if (!cache.Warmup(tensors, error)) {
        return false;
    }

    std::vector<float> resized;
    int32_t resized_w = 0;
    int32_t resized_h = 0;
    ResizeLongEdge(rgb, width, height, row_stride, opts.resize_long_edge,
                   &resized, &resized_w, &resized_h);

    GpuDenseMaps maps;
    if (!ExtractDenseMapGpuVram(tensors, resized, resized_w, resized_h,
                                resized_h, resized_w, &maps, error, &backend,
                                &cache)) {
        return false;
    }

    return WriteAlikedDcnParityDump(output_dump, error);
}

}  // namespace lightglue
