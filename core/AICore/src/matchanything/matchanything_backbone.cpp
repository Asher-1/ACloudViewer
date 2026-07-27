// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "matchanything_backbone.hpp"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "simple_gguf_io.hpp"

namespace matchanything {
namespace {

std::string Lower(std::string value) {
    for (char &c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::vector<float> ToGgmlConvWeight(const std::vector<float> &nchw,
                                    int32_t ic,
                                    int32_t oc,
                                    int32_t kh,
                                    int32_t kw) {
    std::vector<float> out(static_cast<size_t>(ic) * oc * kh * kw);
    for (int32_t o = 0; o < oc; ++o) {
        for (int32_t i = 0; i < ic; ++i) {
            for (int32_t y = 0; y < kh; ++y) {
                for (int32_t x = 0; x < kw; ++x) {
                    const size_t pt_idx =
                            static_cast<size_t>(o) * ic * kh * kw +
                            static_cast<size_t>(i) * kh * kw +
                            static_cast<size_t>(y) * kw +
                            static_cast<size_t>(x);
                    const size_t ggml_idx =
                            static_cast<size_t>(x) +
                            static_cast<size_t>(y) * kw +
                            static_cast<size_t>(i) * kh * kw +
                            static_cast<size_t>(o) * ic * kh * kw;
                    out[ggml_idx] = nchw[pt_idx];
                }
            }
        }
    }
    return out;
}

struct PendingUpload {
    ggml_tensor *tensor = nullptr;
    std::vector<float> data;
};

ggml_tensor *MakeKernel(ggml_context *ctx,
                        const std::vector<float> &nchw,
                        int32_t kw,
                        int32_t kh,
                        int32_t ic,
                        int32_t oc,
                        std::vector<PendingUpload> *pending) {
    ggml_tensor *t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kw, kh, ic, oc);
    pending->push_back({t, ToGgmlConvWeight(nchw, ic, oc, kh, kw)});
    return t;
}

ggml_tensor *RepBlock(ggml_context *ctx,
                      ggml_tensor *input,
                      const TensorMap &weights,
                      const std::string &prefix,
                      int32_t ic,
                      int32_t oc,
                      int32_t stride,
                      std::vector<PendingUpload> *pending,
                      std::string *error) {
    const std::string wp = prefix + "_rbr_reparam_weight";
    const std::string bp = prefix + "_rbr_reparam_bias";
    if (weights.count(wp) == 0 || weights.count(bp) == 0) {
        if (error) {
            *error = "missing tensor " + wp;
        }
        return nullptr;
    }
    ggml_tensor *weight =
            MakeKernel(ctx, weights.at(wp), 3, 3, ic, oc, pending);
    ggml_tensor *bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, oc, 1);
    pending->push_back({bias, weights.at(bp)});
    ggml_tensor *y =
            ggml_conv_2d(ctx, weight, input, stride, stride, 1, 1, 1, 1);
    y = ggml_add(ctx, y, ggml_repeat(ctx, bias, y));
    return ggml_relu(ctx, y);
}

ggml_backend_t CreateBackend(const std::string &device, std::string *error) {
    ggml_backend_load_all();
    const std::string name = Lower(device);
    if (name.empty() || name == "cpu") {
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (name == "cuda" || name == "vulkan" || name == "gpu") {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            if (name != "gpu") {
                const char *registry = ggml_backend_reg_name(
                        ggml_backend_dev_backend_reg(dev));
                if (registry == nullptr || Lower(registry) != name) {
                    continue;
                }
            }
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend != nullptr) {
                return backend;
            }
        }
        if (error) {
            *error = "no usable backend: " + device;
        }
        return nullptr;
    }
    if (error) {
        *error = "unknown device: " + device;
    }
    return nullptr;
}

void FetchNchw(ggml_backend_t backend,
               ggml_tensor *t,
               std::vector<float> *out) {
    const int64_t w = t->ne[0];
    const int64_t h = t->ne[1];
    const int64_t c = t->ne[2];
    out->resize(static_cast<size_t>(w * h * c));
    ggml_backend_tensor_get(t, out->data(), 0, out->size() * sizeof(float));
}

}  // namespace

bool LoadMatchAnythingGguf(const std::string &path,
                           TensorMap *tensors,
                           std::string *error) {
    if (tensors == nullptr) {
        return false;
    }
    return aicore::common::load_simple_gguf_f32(path, tensors, error);
}

bool RunRepVggInterBackbone(const TensorMap &weights,
                            const std::vector<float> &input_nchw,
                            int32_t h,
                            int32_t w,
                            const std::string &device,
                            BackboneOutput *out,
                            std::string *error) {
    if (out == nullptr) {
        return false;
    }
    ggml_backend_t backend = CreateBackend(device, error);
    if (backend == nullptr) {
        return false;
    }

    ggml_init_params params{128 * 1024 * 1024, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) {
        ggml_backend_free(backend);
        if (error) {
            *error = "ggml_init failed";
        }
        return false;
    }

    std::vector<PendingUpload> pending;
    ggml_tensor *inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, 1, 1);
    ggml_tensor *x = RepBlock(ctx, inp, weights, "matcher_backbone_layer0", 1,
                              64, 2, &pending, error);
    if (x == nullptr) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor *x1 = RepBlock(ctx, x, weights, "matcher_backbone_layer1_0", 64,
                               64, 1, &pending, error);
    x1 = RepBlock(ctx, x1, weights, "matcher_backbone_layer1_1", 64, 64, 1,
                  &pending, error);

    ggml_tensor *x2 = RepBlock(ctx, x1, weights, "matcher_backbone_layer2_0",
                               64, 128, 2, &pending, error);
    for (int b = 1; b <= 3; ++b) {
        x2 = RepBlock(ctx, x2, weights,
                      "matcher_backbone_layer2_" + std::to_string(b), 128, 128,
                      1, &pending, error);
    }

    ggml_tensor *xc = RepBlock(ctx, x2, weights, "matcher_backbone_layer3_0",
                               128, 256, 2, &pending, error);
    for (int b = 1; b <= 13; ++b) {
        xc = RepBlock(ctx, xc, weights,
                      "matcher_backbone_layer3_" + std::to_string(b), 256, 256,
                      1, &pending, error);
    }
    if (xc == nullptr) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, xc);
    ggml_build_forward_expand(gf, x2);
    ggml_build_forward_expand(gf, x1);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        if (error) {
            *error = "backend alloc failed";
        }
        return false;
    }
    for (const PendingUpload &item : pending) {
        ggml_backend_tensor_set(item.tensor, item.data.data(), 0,
                                item.data.size() * sizeof(float));
    }
    ggml_backend_tensor_set(inp, input_nchw.data(), 0,
                            input_nchw.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        if (error) {
            *error = "graph compute failed";
        }
        return false;
    }

    FetchNchw(backend, xc, &out->feat_c);
    FetchNchw(backend, x2, &out->feat_x2);
    FetchNchw(backend, x1, &out->feat_x1);
    out->wc = static_cast<int32_t>(xc->ne[0]);
    out->hc = static_cast<int32_t>(xc->ne[1]);
    out->cc = static_cast<int32_t>(xc->ne[2]);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

}  // namespace matchanything
