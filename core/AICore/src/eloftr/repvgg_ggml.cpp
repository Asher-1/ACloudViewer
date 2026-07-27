// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "eloftr.hpp"
#include "eloftr_backbone.hpp"

namespace eloftr {
namespace {

using TensorMap = std::unordered_map<std::string, std::vector<float>>;

std::string Lower(std::string value) {
    for (char &c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

bool LoadGguf(const std::string &path, TensorMap *tensors, std::string *error) {
    RepVggTensorMap mapped;
    if (!LoadRepVggGguf(path, &mapped, error)) {
        return false;
    }
    *tensors = TensorMap(mapped.begin(), mapped.end());
    return true;
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

// Coarse matching skeleton: softmax(sim(f0,f1)/T) + mutual nearest (CPU for
// Phase 6b).
struct CoarseMatchResult {
    std::vector<int32_t> idx0;
    std::vector<int32_t> idx1;
    std::vector<float> scores;
};

bool CoarseMatchCpu(const std::vector<float> &f0,
                    const std::vector<float> &f1,
                    int32_t n0,
                    int32_t n1,
                    int32_t c,
                    float temperature,
                    CoarseMatchResult *out,
                    std::string *error) {
    if (n0 <= 0 || n1 <= 0 || c <= 0) {
        if (error) {
            *error = "invalid coarse feature shape";
        }
        return false;
    }
    out->idx0.clear();
    out->idx1.clear();
    out->scores.clear();
    std::vector<float> sim(static_cast<size_t>(n0) * n1, 0.0f);
    for (int32_t i = 0; i < n0; ++i) {
        for (int32_t j = 0; j < n1; ++j) {
            float dot = 0.0f;
            for (int32_t k = 0; k < c; ++k) {
                dot += f0[static_cast<size_t>(k) * n0 + i] *
                       f1[static_cast<size_t>(k) * n1 + j];
            }
            sim[static_cast<size_t>(i) * n1 + j] = dot / temperature;
        }
    }
    for (int32_t i = 0; i < n0; ++i) {
        int32_t best_j = 0;
        float best = sim[static_cast<size_t>(i) * n1];
        for (int32_t j = 1; j < n1; ++j) {
            const float v = sim[static_cast<size_t>(i) * n1 + j];
            if (v > best) {
                best = v;
                best_j = j;
            }
        }
        out->idx0.push_back(i);
        out->idx1.push_back(best_j);
        out->scores.push_back(best);
    }
    return true;
}

class EfficientLoFTRMatcherImpl : public EfficientLoFTRMatcher {
public:
    explicit EfficientLoFTRMatcherImpl(EfficientLoFTROptions options)
        : options_(std::move(options)) {
        if (!LoadGguf(options_.model_path, &weights_, &init_error_)) {
            return;
        }
        backend_ = CreateBackend(options_.device, &init_error_);
    }

    bool MatchGray(const uint8_t *img0,
                   const uint8_t *img1,
                   int32_t w,
                   int32_t h,
                   int32_t stride,
                   EfficientLoFTRResult *result) override {
        error_.clear();
        if (result == nullptr || backend_ == nullptr) {
            error_ = init_error_.empty() ? "matcher not initialized"
                                         : init_error_;
            return false;
        }

        std::vector<float> in0(static_cast<size_t>(w) * h);
        std::vector<float> in1(static_cast<size_t>(w) * h);
        for (int32_t y = 0; y < h; ++y) {
            for (int32_t x = 0; x < w; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                in0[idx] = static_cast<float>(
                                   img0[static_cast<size_t>(y) * stride + x]) /
                           255.0f;
                in1[idx] = static_cast<float>(
                                   img1[static_cast<size_t>(y) * stride + x]) /
                           255.0f;
            }
        }

        std::vector<float> f0;
        std::vector<float> f1;
        RepVggTensorMap mapped(weights_.begin(), weights_.end());
        int32_t oh = 0;
        int32_t ow = 0;
        int32_t oc = 0;
        if (!RunRepVggBackbone(mapped, in0, h, w, options_.device, &f0, &oh,
                               &ow, &oc, &error_) ||
            !RunRepVggBackbone(mapped, in1, h, w, options_.device, &f1, &oh,
                               &ow, &oc, &error_)) {
            return false;
        }
        const int32_t c = oc;
        const int32_t spatial = ow * oh;
        CoarseMatchResult coarse;
        if (!CoarseMatchCpu(f0, f1, spatial, spatial, c, 0.1f, &coarse,
                            &error_)) {
            return false;
        }

        const float scale_x = static_cast<float>(w) / static_cast<float>(ow);
        const float scale_y = static_cast<float>(h) / static_cast<float>(oh);
        result->matches.clear();
        result->matches.reserve(coarse.idx0.size());
        for (size_t i = 0; i < coarse.idx0.size(); ++i) {
            const int32_t i0 = coarse.idx0[i];
            const int32_t i1 = coarse.idx1[i];
            const int32_t gx0 = i0 % ow;
            const int32_t gy0 = i0 / ow;
            const int32_t gx1 = i1 % ow;
            const int32_t gy1 = i1 / ow;
            MatchPair m;
            m.x0 = (static_cast<float>(gx0) + 0.5f) * scale_x;
            m.y0 = (static_cast<float>(gy0) + 0.5f) * scale_y;
            m.x1 = (static_cast<float>(gx1) + 0.5f) * scale_x;
            m.y1 = (static_cast<float>(gy1) + 0.5f) * scale_y;
            m.score = coarse.scores[i];
            result->matches.push_back(m);
        }
        return true;
    }

    const std::string &Device() const override { return options_.device; }
    const std::string &Error() const override {
        return error_.empty() ? init_error_ : error_;
    }

private:
    EfficientLoFTROptions options_;
    TensorMap weights_;
    ggml_backend_t backend_ = nullptr;
    std::string init_error_;
    std::string error_;
};

}  // namespace

std::unique_ptr<EfficientLoFTRMatcher> CreateEfficientLoFTRMatcher(
        const EfficientLoFTROptions &options, std::string *error) {
    auto impl = std::make_unique<EfficientLoFTRMatcherImpl>(options);
    if (!impl->Error().empty() && error != nullptr) {
        *error = impl->Error();
        return nullptr;
    }
    return impl;
}

}  // namespace eloftr
