// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "session.hpp"

namespace aicore {
namespace tagging {
std::vector<float> preprocess(const img::Image& image, int threads) {
    if (image.empty()) throw std::runtime_error("invalid image");
    constexpr int size = 384, precision = 22;
    using Table = std::vector<std::vector<std::pair<int, int32_t>>>;
    const auto coefficients = [](int input_size) {
        Table table(size);
        const double scale = double(input_size) / size;
        const double support = std::max(1.0, scale);
        for (int out = 0; out < size; ++out) {
            const double center = (out + .5) * scale;
            const int left =
                    std::max(0, int(std::floor(center - support + .5)));
            const int right = std::min(input_size,
                                       int(std::floor(center + support + .5)));
            double total = 0;
            std::vector<double> weights;
            for (int in = left; in < right; ++in) {
                double w = std::max(
                        0.0, 1.0 - std::abs((in + .5 - center) / support));
                weights.push_back(w);
                total += w;
            }
            for (int in = left; in < right; ++in)
                table[out].emplace_back(in, int32_t(weights[in - left] / total *
                                                            (1 << precision) +
                                                    .5));
        }
        return table;
    };
    struct Cache {
        int input = -1;
        Table table;
    };
    static thread_local std::array<Cache, 2> cache;
    if (cache[0].input != image.width)
        cache[0] = {image.width, coefficients(image.width)};
    if (cache[1].input != image.height)
        cache[1] = {image.height, coefficients(image.height)};
    const auto& horizontal_coefficients = cache[0].table;
    const auto& vertical_coefficients = cache[1].table;
    // Pillow's separable antialias filter uses 22-bit coefficients and rounds
    // to uint8 after EACH axis. A float/cv resize changes threshold decisions.
    std::vector<uint8_t> horizontal(size_t(image.height) * size * 3);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int y = 0; y < image.height; ++y)
        for (int x = 0; x < size; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t sum = 1 << (precision - 1);
                for (auto [in, w] : horizontal_coefficients[x])
                    sum += image.channel(in, y, c) * w;
                horizontal[(size_t(y) * size + x) * 3 + c] =
                        uint8_t(std::clamp<int64_t>(sum >> precision, 0, 255));
            }
    std::vector<float> pixels(size * size * 3);
    constexpr float mean[] = {.485f, .456f, .406f},
                    stddev[] = {.229f, .224f, .225f};
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t sum = 1 << (precision - 1);
                for (auto [in, w] : vertical_coefficients[y])
                    sum += horizontal[(size_t(in) * size + x) * 3 + c] * w;
                const auto v = std::clamp<int64_t>(sum >> precision, 0, 255);
                pixels[(c * size + y) * size + x] =
                        (v / 255.f - mean[c]) / stddev[c];
            }
    return pixels;
}
}  // namespace tagging
}  // namespace aicore
