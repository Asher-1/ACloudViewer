// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "matchanything.hpp"
#include "matchanything_backbone.hpp"

namespace matchanything {
namespace {

void L2NormalizeChannels(std::vector<float> *feat, int32_t c, int32_t spatial) {
    if (feat == nullptr || feat->empty()) {
        return;
    }
    for (int32_t s = 0; s < spatial; ++s) {
        float norm = 0.0f;
        for (int32_t ch = 0; ch < c; ++ch) {
            const float v = (*feat)[static_cast<size_t>(ch) * spatial + s];
            norm += v * v;
        }
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int32_t ch = 0; ch < c; ++ch) {
            (*feat)[static_cast<size_t>(ch) * spatial + s] /= norm;
        }
    }
}

struct CoarseMatchResult {
    std::vector<int32_t> idx0;
    std::vector<int32_t> idx1;
    std::vector<float> scores;
};

bool CoarseDualSoftmax(const std::vector<float> &f0,
                       const std::vector<float> &f1,
                       int32_t n0,
                       int32_t n1,
                       int32_t c,
                       float temperature,
                       float thr,
                       CoarseMatchResult *out,
                       std::string *error) {
    if (n0 <= 0 || n1 <= 0 || c <= 0 || out == nullptr) {
        if (error) {
            *error = "invalid coarse shape";
        }
        return false;
    }
    std::vector<float> sim(static_cast<size_t>(n0) * n1);
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

    // Row softmax
    std::vector<float> conf(static_cast<size_t>(n0) * n1);
    for (int32_t i = 0; i < n0; ++i) {
        float maxv = sim[static_cast<size_t>(i) * n1];
        for (int32_t j = 1; j < n1; ++j) {
            maxv = std::max(maxv, sim[static_cast<size_t>(i) * n1 + j]);
        }
        float sum = 0.0f;
        for (int32_t j = 0; j < n1; ++j) {
            const float e =
                    std::exp(sim[static_cast<size_t>(i) * n1 + j] - maxv);
            conf[static_cast<size_t>(i) * n1 + j] = e;
            sum += e;
        }
        for (int32_t j = 0; j < n1; ++j) {
            conf[static_cast<size_t>(i) * n1 + j] /= sum;
        }
    }

    // Mutual nearest with threshold (force_nearest path)
    std::vector<int32_t> nn01(n0, -1);
    std::vector<float> nn01_score(n0, -1.0f);
    for (int32_t i = 0; i < n0; ++i) {
        int32_t best_j = 0;
        float best = conf[static_cast<size_t>(i) * n1];
        for (int32_t j = 1; j < n1; ++j) {
            const float v = conf[static_cast<size_t>(i) * n1 + j];
            if (v > best) {
                best = v;
                best_j = j;
            }
        }
        if (best >= thr) {
            nn01[i] = best_j;
            nn01_score[i] = best;
        }
    }

    std::vector<int32_t> nn10(n1, -1);
    for (int32_t i = 0; i < n0; ++i) {
        if (nn01[i] < 0) {
            continue;
        }
        const int32_t j = nn01[i];
        if (nn10[j] < 0 || nn01_score[i] > nn01_score[nn10[j]]) {
            nn10[j] = i;
        }
    }

    out->idx0.clear();
    out->idx1.clear();
    out->scores.clear();
    for (int32_t j = 0; j < n1; ++j) {
        const int32_t i = nn10[j];
        if (i >= 0 && nn01[i] == j) {
            out->idx0.push_back(i);
            out->idx1.push_back(j);
            out->scores.push_back(nn01_score[i]);
        }
    }
    return true;
}

class MatchAnythingElotrMatcher : public MatchAnythingMatcher {
public:
    explicit MatchAnythingElotrMatcher(MatchAnythingOptions options)
        : options_(std::move(options)) {
        if (!LoadMatchAnythingGguf(options_.model_path, &weights_,
                                   &init_error_)) {
            return;
        }
    }

    bool MatchGray(const uint8_t *img0,
                   const uint8_t *img1,
                   int32_t w,
                   int32_t h,
                   int32_t stride,
                   MatchAnythingResult *result) override {
        error_.clear();
        if (result == nullptr || !init_error_.empty()) {
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

        BackboneOutput b0;
        BackboneOutput b1;
        if (!RunRepVggInterBackbone(weights_, in0, h, w, options_.device, &b0,
                                    &error_) ||
            !RunRepVggInterBackbone(weights_, in1, h, w, options_.device, &b1,
                                    &error_)) {
            return false;
        }

        const int32_t spatial = b0.hc * b0.wc;
        L2NormalizeChannels(&b0.feat_c, b0.cc, spatial);
        L2NormalizeChannels(&b1.feat_c, b1.cc, spatial);

        // Phase 7b: PAN coarse transformer weights loaded; CPU passthrough
        // until PAN/XFormer ggml graph lands. Coarse matching uses normalized
        // backbone feats.
        CoarseMatchResult coarse;
        if (!CoarseDualSoftmax(b0.feat_c, b1.feat_c, spatial, spatial, b0.cc,
                               0.1f, 0.1f, &coarse, &error_)) {
            return false;
        }

        const float scale_x = static_cast<float>(w) / static_cast<float>(b0.wc);
        const float scale_y = static_cast<float>(h) / static_cast<float>(b0.hc);
        result->matches.clear();
        result->matches.reserve(coarse.idx0.size());
        for (size_t k = 0; k < coarse.idx0.size(); ++k) {
            const int32_t i0 = coarse.idx0[k];
            const int32_t i1 = coarse.idx1[k];
            MatchPair m;
            m.x0 = (static_cast<float>(i0 % b0.wc) + 0.5f) * scale_x;
            m.y0 = (static_cast<float>(i0 / b0.wc) + 0.5f) * scale_y;
            m.x1 = (static_cast<float>(i1 % b0.wc) + 0.5f) * scale_x;
            m.y1 = (static_cast<float>(i1 / b0.wc) + 0.5f) * scale_y;
            m.score = coarse.scores[k];
            result->matches.push_back(m);
        }
        return true;
    }

    const std::string &Device() const override { return options_.device; }
    const std::string &VariantName() const override { return variant_name_; }
    const std::string &Error() const override {
        return error_.empty() ? init_error_ : error_;
    }

private:
    MatchAnythingOptions options_;
    TensorMap weights_;
    std::string init_error_;
    std::string error_;
    std::string variant_name_{"eloftr"};
};

class MatchAnythingRomaStub : public MatchAnythingMatcher {
public:
    explicit MatchAnythingRomaStub(MatchAnythingOptions options)
        : options_(std::move(options)) {
        init_error_ =
                "matchanything_roma ggml inference is Phase 7e (not yet "
                "implemented)";
    }

    bool MatchGray(const uint8_t *,
                   const uint8_t *,
                   int32_t,
                   int32_t,
                   int32_t,
                   MatchAnythingResult *) override {
        error_ = init_error_;
        return false;
    }

    const std::string &Device() const override { return options_.device; }
    const std::string &VariantName() const override { return variant_name_; }
    const std::string &Error() const override {
        return error_.empty() ? init_error_ : error_;
    }

private:
    MatchAnythingOptions options_;
    std::string init_error_;
    std::string error_;
    std::string variant_name_{"roma"};
};

}  // namespace

std::unique_ptr<MatchAnythingMatcher> CreateMatchAnythingMatcher(
        const MatchAnythingOptions &options, std::string *error) {
    std::unique_ptr<MatchAnythingMatcher> impl;
    if (options.variant == Variant::kRoma) {
        impl = std::make_unique<MatchAnythingRomaStub>(options);
    } else {
        impl = std::make_unique<MatchAnythingElotrMatcher>(options);
    }
    if (!impl->Error().empty() && error != nullptr &&
        options.variant != Variant::kRoma) {
        *error = impl->Error();
        return nullptr;
    }
    return impl;
}

}  // namespace matchanything
