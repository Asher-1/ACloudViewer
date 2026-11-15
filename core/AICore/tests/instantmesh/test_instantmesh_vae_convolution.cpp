// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <memory>
#include <vector>

#include "src/common/ggml_backend_registry.hpp"
#ifdef AICORE_TEST_UNET_CONV
#include "src/tasks/instantmesh/unet.cpp"
#else
#include "src/tasks/instantmesh/vae.cpp"
#endif

int main(int argc, char** argv) {
    std::string error;
    auto lease = aicore::runtime::acquire_backend_lease(
            argc > 1 ? argv[1] : "cpu", 4, &error);
    if (!lease) {
        std::fprintf(stderr, "FAIL: backend unavailable: %s\n", error.c_str());
        return 1;
    }
    auto lock = lease.lock();
    constexpr int W = 7, H = 5, B = 2, K = 3;
    int failures = 0;
    for (bool f32 : {true, false}) {
        constexpr int IC = 16, OC = 16;
        std::vector<float> image(W * H * IC * B), weights(K * K * IC * OC);
        for (size_t i = 0; i < image.size(); ++i)
            image[i] = f32 ? (int(i % 37) - 18) / 16.0f : 1.0f;
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = f32 ? (int(i % 23) - 11) * 0.1234567f
                             : (i % (K * K * IC) == 0 ? 1.0f : 0x1p-14f);
        std::vector<double> expected(W * H * OC * B);
        for (int batch = 0; batch < B; ++batch)
            for (int oc = 0; oc < OC; ++oc)
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        double value = 0;
                        for (int ic = 0; ic < IC; ++ic)
                            for (int ky = 0; ky < K; ++ky)
                                for (int kx = 0; kx < K; ++kx) {
                                    const int iy = y + ky - 1, ix = x + kx - 1;
                                    if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                        continue;
                                    value += double(image[((batch * IC + ic) *
                                                                   H +
                                                           iy) * W +
                                                          ix]) *
                                             weights[((oc * IC + ic) * K + ky) *
                                                             K +
                                                     kx];
                                }
                        expected[((batch * OC + oc) * H + y) * W + x] = value;
                    }
        ggml_init_params params{16 * 1024 * 1024, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
                ggml_init(params), ggml_free);
        auto* input =
                ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, W, H, IC, B);
        auto* kernel =
                ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, K, K, IC, OC);
        ggml_set_input(input);
        ggml_set_input(kernel);
        auto* output = f32 ? instantmesh::conv_f32(context.get(), input, kernel,
                                                   nullptr, 1, 1)
                           : instantmesh::conv(context.get(), input, kernel,
                                               nullptr, 1, 1);
        ggml_set_output(output);
        auto* graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, output);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(
                        ggml_backend_get_default_buffer_type(lease.handle())),
                ggml_gallocr_free);
        if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) return 1;
        ggml_backend_tensor_set(input, image.data(), 0,
                                image.size() * sizeof(float));
        ggml_backend_tensor_set(kernel, weights.data(), 0,
                                weights.size() * sizeof(float));
        if (ggml_backend_graph_compute(lease.handle(), graph) !=
            GGML_STATUS_SUCCESS)
            return 1;
        if (ggml_nelements(output) != int64_t(expected.size())) return 1;
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(float));
        double maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) return 1;
            maximum = std::max(maximum,
                               std::abs(double(actual[i]) - expected[i]));
        }
        std::printf(
                "%s max_abs=%.9g actual_backend=%s\n", f32 ? "f32" : "mixed",
                maximum,
                ggml_backend_dev_name(ggml_backend_get_device(lease.handle())));
        if (!(maximum < 2e-5)) ++failures;
    }
    return failures ? 1 : 0;
}
