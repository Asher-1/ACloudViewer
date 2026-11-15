// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Parallel.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "tasks/aliked/deform_conv.hpp"
#include "tasks/aliked/tensor_ops.hpp"
#include "tasks/openpcdet/extensions/deterministic_sparse_conv.h"

namespace {
void equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() ||
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0)
        throw std::runtime_error("Host kernel changed its per-cell arithmetic");
}

void check_aliked() {
    namespace ops = lightglue::aliked_internal;
    constexpr int channels = 2, height = 5, width = 7, outputs = 3;
    std::vector<float> input(channels * height * width);
    std::vector<float> weights(outputs * channels * 9), bias(outputs);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = (static_cast<int>(i % 13) - 6) / 16.f;
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = (static_cast<int>(i % 7) - 3) / 8.f;
    for (int i = 0; i < outputs; ++i) bias[i] = i / 32.f;
    std::vector<float> conv, deform, pooled, upsampled;
    int h, w;
    ops::Conv2d(input, channels, height, width, weights, outputs, 3, 3, &bias,
                1, 1, &conv, &h, &w);
    if (h != height || w != width)
        throw std::runtime_error("Invalid conv shape");
    std::vector<float> offsets(18 * height * width, 0.f);
    ops::DeformConv2d(input, channels, height, width, offsets, 1, weights,
                      outputs, 3, 3, &bias, 1, &deform, &h, &w);
    // Zero offsets reduce the deformable kernel to the ordinary convolution.
    equal(conv, deform);
    ops::AvgPool2d(input, channels, height, width, 2, 2, 1, &pooled, &h, &w);
    ops::UpsampleBilinear(input, channels, height, width, 9, 11, &upsampled);
    // A second execution checks stable ownership of all output cells.
    std::vector<float> repeated;
    ops::Conv2d(input, channels, height, width, weights, outputs, 3, 3, &bias,
                1, 1, &repeated, &h, &w);
    equal(conv, repeated);
    ops::AvgPool2d(input, channels, height, width, 2, 2, 1, &repeated, &h, &w);
    equal(pooled, repeated);
    ops::UpsampleBilinear(input, channels, height, width, 9, 11, &repeated);
    equal(upsampled, repeated);
}

void check_sparse_conv() {
    namespace det = openpcdet::det;
    constexpr int in_rows = 19, in_channels = 16, out_channels = 32;
    constexpr int out_rows = 17, kernel = 27;
    std::vector<float> input(in_rows * in_channels);
    std::vector<float> weights(kernel * in_channels * out_channels);
    std::vector<float> bias(out_channels);
    std::vector<int32_t> indices(out_rows * kernel);
    for (size_t i = 0; i < input.size(); ++i) input[i] = (i % 23) / 32.f;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (i % 17) / 64.f;
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i < kernel || i % 5 == 0 ? -1 : i % in_rows;
    std::vector<float> actual(out_rows * out_channels), expected(actual.size());
    det::det_sparse_conv_cpu(input.data(), in_channels, indices.data(),
                             weights.data(), bias.data(), actual.data(),
                             out_rows, out_channels, kernel);
    det::det_sparse_conv_rows(input.data(), in_channels, indices.data(),
                              weights.data(), bias.data(), expected.data(),
                              out_rows, out_channels, kernel, det::det_fma);
    equal(actual, expected);
}
}  // namespace

int main() {
    try {
        check_aliked();
        check_sparse_conv();
        std::puts("ALIKED host kernels and deterministic sparse conv passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
