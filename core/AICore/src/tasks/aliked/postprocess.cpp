// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "tasks/aliked/deform_conv.hpp"
#include "tasks/aliked/tensor_ops.hpp"

namespace lightglue::aliked_internal {
namespace {

constexpr float kDkdTemperature = 0.1f;

#if defined(_OPENMP)
int CpuPostprocessThreads() { return std::min(8, omp_get_max_threads()); }
#endif

inline int32_t IndexNchw(
        int32_t c, int32_t y, int32_t x, int32_t h, int32_t w) {
    return c * h * w + y * w + x;
}

void MaxPool2d(const std::vector<float> &input,
               int32_t h,
               int32_t w,
               int32_t k,
               int32_t pad,
               std::vector<float> *output) {
    output->assign(static_cast<size_t>(h) * w, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(CpuPostprocessThreads())
#endif
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float best = -std::numeric_limits<float>::infinity();
            for (int32_t ky = -pad; ky <= pad; ++ky) {
                for (int32_t kx = -pad; kx <= pad; ++kx) {
                    const int32_t iy = y + ky;
                    const int32_t ix = x + kx;
                    if (iy < 0 || ix < 0 || iy >= h || ix >= w) {
                        continue;
                    }
                    best = std::max(best,
                                    input[static_cast<size_t>(iy) * w + ix]);
                }
            }
            (*output)[static_cast<size_t>(y) * w + x] = best;
        }
    }
}

std::vector<float> SimpleNms(const std::vector<float> &scores,
                             int32_t h,
                             int32_t w,
                             int32_t radius) {
    std::vector<float> max_mask;
    MaxPool2d(scores, h, w, radius * 2 + 1, radius, &max_mask);
    const int64_t count = static_cast<int64_t>(scores.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(CpuPostprocessThreads())
#endif
    for (int64_t i = 0; i < count; ++i) {
        max_mask[static_cast<size_t>(i)] =
                scores[static_cast<size_t>(i)] ==
                                max_mask[static_cast<size_t>(i)]
                        ? 1.0f
                        : 0.0f;
    }

    std::vector<float> result = scores;
    for (int iter = 0; iter < 2; ++iter) {
        std::vector<float> supp_mask;
        MaxPool2d(max_mask, h, w, radius * 2 + 1, radius, &supp_mask);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(CpuPostprocessThreads())
#endif
        for (int64_t i = 0; i < count; ++i) {
            float &value = supp_mask[static_cast<size_t>(i)];
            value = value > 0.0f ? 1.0f : 0.0f;
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(CpuPostprocessThreads())
#endif
        for (int64_t i = 0; i < count; ++i) {
            const size_t index = static_cast<size_t>(i);
            result[index] = supp_mask[index] > 0.0f ? 0.0f : result[index];
        }
        std::vector<float> new_max;
        MaxPool2d(result, h, w, radius * 2 + 1, radius, &new_max);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(CpuPostprocessThreads())
#endif
        for (int64_t i = 0; i < count; ++i) {
            const size_t index = static_cast<size_t>(i);
            if (result[index] == new_max[index]) {
                max_mask[index] = 1.0f;
            } else if (supp_mask[index] <= 0.0f) {
                max_mask[index] = 0.0f;
            }
        }
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(CpuPostprocessThreads())
#endif
    for (int64_t i = 0; i < count; ++i) {
        const size_t index = static_cast<size_t>(i);
        result[index] = max_mask[index] > 0.0f ? scores[index] : 0.0f;
    }
    return result;
}

float GridSampleBilinearNorm(const std::vector<float> &scores,
                             int32_t h,
                             int32_t w,
                             float x_norm,
                             float y_norm) {
    const float x = (x_norm + 1.0f) * 0.5f * static_cast<float>(w - 1);
    const float y = (y_norm + 1.0f) * 0.5f * static_cast<float>(h - 1);
    return BilinearSample(scores, 1, h, w, 0, y, x);
}

}  // namespace

DkdOutput RunDkd(const std::vector<float> &score_map,
                 int32_t h,
                 int32_t w,
                 const DkdOptions &options,
                 int32_t image_width,
                 int32_t image_height) {
    (void)image_width;
    (void)image_height;
    DkdOutput output;
    if (h <= 0 || w <= 0 ||
        score_map.size() < static_cast<size_t>(h) * static_cast<size_t>(w)) {
        return output;
    }
    std::vector<float> nms = SimpleNms(score_map, h, w, options.radius);

    for (int32_t y = 0; y < options.radius; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            nms[static_cast<size_t>(y) * w + x] = 0.0f;
        }
    }
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < options.radius; ++x) {
            nms[static_cast<size_t>(y) * w + x] = 0.0f;
        }
    }
    for (int32_t y = std::max(0, h - options.radius); y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            nms[static_cast<size_t>(y) * w + x] = 0.0f;
        }
    }
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = std::max(0, w - options.radius); x < w; ++x) {
            nms[static_cast<size_t>(y) * w + x] = 0.0f;
        }
    }

    std::vector<int32_t> indices;
    if (options.top_k > 0) {
        std::vector<std::pair<float, int32_t>> scored;
        scored.reserve(static_cast<size_t>(h) * w);
        for (int32_t i = 0; i < h * w; ++i) {
            const float score = nms[static_cast<size_t>(i)];
            if (score > options.scores_th) {
                scored.emplace_back(score, i);
            }
        }
        const int32_t keep =
                std::min(options.top_k, static_cast<int32_t>(scored.size()));
        std::partial_sort(
                scored.begin(), scored.begin() + keep, scored.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });
        indices.reserve(static_cast<size_t>(keep));
        for (int32_t i = 0; i < keep; ++i) {
            indices.push_back(scored[static_cast<size_t>(i)].second);
        }
    } else {
        float threshold = options.scores_th;
        if (options.scores_th > 0.0f) {
            bool any = false;
            for (float value : nms) {
                if (value > threshold) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                threshold = 0.0f;
                for (float value : score_map) {
                    threshold += value;
                }
                threshold /= static_cast<float>(score_map.size());
            }
        } else {
            threshold = 0.0f;
            for (float value : score_map) {
                threshold += value;
            }
            threshold /= static_cast<float>(score_map.size());
        }
        std::vector<std::pair<float, int32_t>> scored;
        for (int32_t i = 0; i < h * w; ++i) {
            if (nms[static_cast<size_t>(i)] > threshold) {
                scored.emplace_back(score_map[static_cast<size_t>(i)], i);
            }
        }
        std::sort(
                scored.begin(), scored.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });
        if (static_cast<int32_t>(scored.size()) > options.n_limit) {
            scored.resize(static_cast<size_t>(options.n_limit));
        }
        indices.reserve(scored.size());
        for (const auto &entry : scored) {
            indices.push_back(entry.second);
        }
    }

    const float wh_x = static_cast<float>(w - 1);
    const float wh_y = static_cast<float>(h - 1);
    const int32_t kernel = options.radius * 2 + 1;
    const int32_t kernel_area = kernel * kernel;
    std::vector<float> hw_grid;
    hw_grid.reserve(static_cast<size_t>(kernel_area) * 2);
    for (int32_t ky = -options.radius; ky <= options.radius; ++ky) {
        for (int32_t kx = -options.radius; kx <= options.radius; ++kx) {
            hw_grid.push_back(static_cast<float>(kx));
            hw_grid.push_back(static_cast<float>(ky));
        }
    }

    output.keypoints_norm.reserve(indices.size() * 2);
    output.scores.reserve(indices.size());
    for (int32_t index : indices) {
        const int32_t x_nms = index % w;
        const int32_t y_nms = index / w;

        std::vector<float> patch_scores;
        patch_scores.reserve(static_cast<size_t>(kernel_area));
        for (int32_t ky = -options.radius; ky <= options.radius; ++ky) {
            for (int32_t kx = -options.radius; kx <= options.radius; ++kx) {
                const int32_t py = std::min(std::max(y_nms + ky, 0), h - 1);
                const int32_t px = std::min(std::max(x_nms + kx, 0), w - 1);
                patch_scores.push_back(
                        score_map[static_cast<size_t>(py) * w + px]);
            }
        }

        float max_v =
                *std::max_element(patch_scores.begin(), patch_scores.end());
        std::vector<float> x_exp(kernel_area, 0.0f);
        float exp_sum = 0.0f;
        for (int32_t i = 0; i < kernel_area; ++i) {
            x_exp[static_cast<size_t>(i)] =
                    std::exp((patch_scores[static_cast<size_t>(i)] - max_v) /
                             kDkdTemperature);
            exp_sum += x_exp[static_cast<size_t>(i)];
        }
        float residual_x = 0.0f;
        float residual_y = 0.0f;
        for (int32_t i = 0; i < kernel_area; ++i) {
            const float weight =
                    exp_sum > 0.0f ? x_exp[static_cast<size_t>(i)] / exp_sum
                                   : 0.0f;
            residual_x += weight * hw_grid[static_cast<size_t>(i) * 2 + 0];
            residual_y += weight * hw_grid[static_cast<size_t>(i) * 2 + 1];
        }

        const float x =
                (static_cast<float>(x_nms) + residual_x) / wh_x * 2.0f - 1.0f;
        const float y =
                (static_cast<float>(y_nms) + residual_y) / wh_y * 2.0f - 1.0f;
        output.keypoints_norm.push_back(x);
        output.keypoints_norm.push_back(y);
        output.scores.push_back(GridSampleBilinearNorm(score_map, h, w, x, y));
    }
    return output;
}

std::vector<float> RunSddh(const std::vector<float> &feature_map,
                           int32_t dim,
                           int32_t h,
                           int32_t w,
                           const std::vector<float> &keypoints_norm,
                           int32_t kernel_size,
                           int32_t n_pos,
                           const std::vector<float> &offset_0_w,
                           const std::vector<float> &offset_0_b,
                           const std::vector<float> &offset_2_w,
                           const std::vector<float> &offset_2_b,
                           const std::vector<float> &sf_conv_w,
                           const std::vector<float> &agg_weights) {
    const int32_t num_kpts = static_cast<int32_t>(keypoints_norm.size() / 2);
    std::vector<float> descriptors(static_cast<size_t>(num_kpts) * dim, 0.0f);
    const float wh_x = static_cast<float>(w - 1);
    const float wh_y = static_cast<float>(h - 1);
    const float max_offset = std::max(h, w) / 4.0f;
    const int32_t pad = kernel_size / 2;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(CpuPostprocessThreads())
#endif
    for (int32_t k = 0; k < num_kpts; ++k) {
        const float x_norm = keypoints_norm[static_cast<size_t>(k) * 2 + 0];
        const float y_norm = keypoints_norm[static_cast<size_t>(k) * 2 + 1];
        const float x_wh = (x_norm / 2.0f + 0.5f) * wh_x;
        const float y_wh = (y_norm / 2.0f + 0.5f) * wh_y;

        std::vector<float> patch;
        if (kernel_size > 1) {
            const int32_t x0 = std::min(
                    std::max(static_cast<int32_t>(std::lround(x_wh)) - pad, 0),
                    w - kernel_size);
            const int32_t y0 = std::min(
                    std::max(static_cast<int32_t>(std::lround(y_wh)) - pad, 0),
                    h - kernel_size);
            patch.assign(static_cast<size_t>(dim) * kernel_size * kernel_size,
                         0.0f);
            for (int32_t c = 0; c < dim; ++c) {
                for (int32_t ky = 0; ky < kernel_size; ++ky) {
                    for (int32_t kx = 0; kx < kernel_size; ++kx) {
                        patch[IndexNchw(c, ky, kx, kernel_size, kernel_size)] =
                                feature_map[IndexNchw(c, y0 + ky, x0 + kx, h,
                                                      w)];
                    }
                }
            }
        } else {
            patch.assign(static_cast<size_t>(dim), 0.0f);
            const int32_t xi = std::min(
                    std::max(static_cast<int32_t>(std::lround(x_wh)), 0),
                    w - 1);
            const int32_t yi = std::min(
                    std::max(static_cast<int32_t>(std::lround(y_wh)), 0),
                    h - 1);
            for (int32_t c = 0; c < dim; ++c) {
                patch[static_cast<size_t>(c)] =
                        feature_map[IndexNchw(c, yi, xi, h, w)];
            }
        }

        std::vector<float> offset_raw;
        int32_t oh = 0;
        int32_t ow = 0;
        Conv2d(patch, dim, kernel_size, kernel_size, offset_0_w, 32,
               kernel_size, kernel_size, &offset_0_b, 0, 1, &offset_raw, &oh,
               &ow);
        ApplySelu(&offset_raw);
        Conv2d(offset_raw, 32, oh, ow, offset_2_w, 32, 1, 1, &offset_2_b, 0, 1,
               &offset_raw, &oh, &ow);

        std::vector<float> desc(dim, 0.0f);
        std::vector<float> sampled(dim, 0.0f);
        std::vector<float> transformed(dim, 0.0f);
        for (int32_t p = 0; p < n_pos; ++p) {
            const float off_x = std::max(
                    -max_offset,
                    std::min(max_offset, offset_raw[static_cast<size_t>(p)]));
            const float off_y = std::max(
                    -max_offset,
                    std::min(max_offset,
                             offset_raw[static_cast<size_t>(n_pos + p)]));
            const float sample_x = (x_wh + off_x) / wh_x * 2.0f - 1.0f;
            const float sample_y = (y_wh + off_y) / wh_y * 2.0f - 1.0f;
            const float px = (sample_x + 1.0f) * 0.5f * wh_x;
            const float py = (sample_y + 1.0f) * 0.5f * wh_y;

            for (int32_t c = 0; c < dim; ++c) {
                sampled[static_cast<size_t>(c)] =
                        BilinearSample(feature_map, dim, h, w, c, py, px);
            }

            for (int32_t c = 0; c < dim; ++c) {
                float value = 0.0f;
                for (int32_t ic = 0; ic < dim; ++ic) {
                    value += sampled[static_cast<size_t>(ic)] *
                             sf_conv_w[static_cast<size_t>(c) * dim + ic];
                }
                transformed[static_cast<size_t>(c)] = Selu(value);
            }

            const size_t agg_base = static_cast<size_t>(p) * dim * dim;
            for (int32_t ic = 0; ic < dim; ++ic) {
                const float value = transformed[static_cast<size_t>(ic)];
                const float *weights = agg_weights.data() + agg_base +
                                       static_cast<size_t>(ic) * dim;
#if defined(_OPENMP)
#pragma omp simd
#endif
                for (int32_t c = 0; c < dim; ++c) {
                    desc[static_cast<size_t>(c)] += value * weights[c];
                }
            }
        }

        float norm = 0.0f;
        for (float value : desc) {
            norm += value * value;
        }
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int32_t c = 0; c < dim; ++c) {
            descriptors[static_cast<size_t>(k) * dim + c] =
                    desc[static_cast<size_t>(c)] / norm;
        }
    }
    return descriptors;
}

bool RunSddhStages(const std::vector<float> &feature_map,
                   int32_t dim,
                   int32_t h,
                   int32_t w,
                   const std::vector<float> &keypoints_norm,
                   int32_t key_index,
                   int32_t kernel_size,
                   int32_t n_pos,
                   const std::vector<float> &offset_0_w,
                   const std::vector<float> &offset_0_b,
                   const std::vector<float> &offset_2_w,
                   const std::vector<float> &offset_2_b,
                   const std::vector<float> &sf_conv_w,
                   const std::vector<float> &agg_weights,
                   SddhStageDump *out) {
    if (out == nullptr || key_index < 0 ||
        key_index >= static_cast<int32_t>(keypoints_norm.size() / 2)) {
        return false;
    }
    const float wh_x = static_cast<float>(w - 1);
    const float wh_y = static_cast<float>(h - 1);
    const float max_offset = std::max(h, w) / 4.0f;
    const int32_t pad = kernel_size / 2;

    out->key_index = key_index;
    const float x_norm = keypoints_norm[static_cast<size_t>(key_index) * 2 + 0];
    const float y_norm = keypoints_norm[static_cast<size_t>(key_index) * 2 + 1];
    out->x_wh = (x_norm / 2.0f + 0.5f) * wh_x;
    out->y_wh = (y_norm / 2.0f + 0.5f) * wh_y;

    if (kernel_size > 1) {
        const int32_t x0 = std::min(
                std::max(static_cast<int32_t>(std::lround(out->x_wh)) - pad, 0),
                w - kernel_size);
        const int32_t y0 = std::min(
                std::max(static_cast<int32_t>(std::lround(out->y_wh)) - pad, 0),
                h - kernel_size);
        out->patch.assign(static_cast<size_t>(dim) * kernel_size * kernel_size,
                          0.0f);
        for (int32_t c = 0; c < dim; ++c) {
            for (int32_t ky = 0; ky < kernel_size; ++ky) {
                for (int32_t kx = 0; kx < kernel_size; ++kx) {
                    out->patch[IndexNchw(c, ky, kx, kernel_size, kernel_size)] =
                            feature_map[IndexNchw(c, y0 + ky, x0 + kx, h, w)];
                }
            }
        }
    } else {
        out->patch.assign(static_cast<size_t>(dim), 0.0f);
        const int32_t xi = std::min(
                std::max(static_cast<int32_t>(std::lround(out->x_wh)), 0),
                w - 1);
        const int32_t yi = std::min(
                std::max(static_cast<int32_t>(std::lround(out->y_wh)), 0),
                h - 1);
        for (int32_t c = 0; c < dim; ++c) {
            out->patch[static_cast<size_t>(c)] =
                    feature_map[IndexNchw(c, yi, xi, h, w)];
        }
    }

    int32_t oh = 0;
    int32_t ow = 0;
    Conv2d(out->patch, dim, kernel_size, kernel_size, offset_0_w, 32,
           kernel_size, kernel_size, &offset_0_b, 0, 1, &out->offset_raw, &oh,
           &ow);
    ApplySelu(&out->offset_raw);
    Conv2d(out->offset_raw, 32, oh, ow, offset_2_w, 32, 1, 1, &offset_2_b, 0, 1,
           &out->offset_final, &oh, &ow);

    out->desc_pre_norm.assign(static_cast<size_t>(dim), 0.0f);
    out->desc.assign(static_cast<size_t>(dim), 0.0f);
    out->sampled.assign(static_cast<size_t>(dim), 0.0f);
    out->transformed.assign(static_cast<size_t>(dim), 0.0f);

    for (int32_t p = 0; p < n_pos; ++p) {
        const float off_x =
                std::max(-max_offset,
                         std::min(max_offset,
                                  out->offset_final[static_cast<size_t>(p)]));
        const float off_y = std::max(
                -max_offset,
                std::min(max_offset,
                         out->offset_final[static_cast<size_t>(n_pos + p)]));
        const float sample_x = (out->x_wh + off_x) / wh_x * 2.0f - 1.0f;
        const float sample_y = (out->y_wh + off_y) / wh_y * 2.0f - 1.0f;
        const float px = (sample_x + 1.0f) * 0.5f * wh_x;
        const float py = (sample_y + 1.0f) * 0.5f * wh_y;

        for (int32_t c = 0; c < dim; ++c) {
            out->sampled[static_cast<size_t>(c)] =
                    BilinearSample(feature_map, dim, h, w, c, py, px);
        }

        for (int32_t c = 0; c < dim; ++c) {
            float value = 0.0f;
            for (int32_t ic = 0; ic < dim; ++ic) {
                value += out->sampled[static_cast<size_t>(ic)] *
                         sf_conv_w[static_cast<size_t>(c) * dim + ic];
            }
            out->transformed[static_cast<size_t>(c)] = Selu(value);
        }

        for (int32_t c = 0; c < dim; ++c) {
            for (int32_t ic = 0; ic < dim; ++ic) {
                out->desc_pre_norm[static_cast<size_t>(c)] +=
                        out->transformed[static_cast<size_t>(ic)] *
                        agg_weights[static_cast<size_t>(p) * dim * dim +
                                    static_cast<size_t>(ic) * dim + c];
            }
        }
    }

    float norm = 0.0f;
    for (float value : out->desc_pre_norm) {
        norm += value * value;
    }
    norm = std::sqrt(std::max(norm, 1e-12f));
    for (int32_t c = 0; c < dim; ++c) {
        out->desc[static_cast<size_t>(c)] =
                out->desc_pre_norm[static_cast<size_t>(c)] / norm;
    }
    return true;
}

}  // namespace lightglue::aliked_internal
