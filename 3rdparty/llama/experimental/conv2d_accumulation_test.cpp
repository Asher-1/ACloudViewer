// F16 input/output must not lose representable residuals through inter-block F16 accumulation; independently derived cancellation sequence.
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool run(ggml_backend_t backend, int channels, int output_channels, int kernel) {
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) return false;
    auto* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 17, 15, channels, 2);
    auto* w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kernel, kernel, channels, output_channels);
    auto* y = ggml_conv_2d_direct(ctx, w, x, 1, 1, kernel / 2, kernel / 2, 1, 1);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<ggml_fp16_t> pixels(ggml_nelements(x));
    std::vector<ggml_fp16_t> weights(ggml_nelements(w));
    for (int batch = 0; batch < 2; ++batch) for (int channel = 0; channel < channels; ++channel) {
        const float value = channel == 0 ? 1024 : channel == channels - 1 ? -1024 : 0.25f;
        for (int i = 0; i < 17 * 15; ++i)
            pixels[(batch * channels + channel) * 17 * 15 + i] = ggml_fp32_to_fp16(value);
    }
    for (int oc = 0; oc < output_channels; ++oc) for (int ic = 0; ic < channels; ++ic)
        weights[(oc * channels + ic) * kernel * kernel + kernel * kernel / 2] =
                ggml_fp32_to_fp16(oc % 2 ? -1.0f : 1.0f);
    ggml_backend_tensor_set(x, pixels.data(), 0, pixels.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(ggml_fp16_t));
    bool valid = true;
    for (int repeat = 0; repeat < 3; ++repeat) {
        valid &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        std::vector<ggml_fp16_t> actual(ggml_nelements(y));
        ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < actual.size(); ++i) {
            // +/-1024 cancel exactly; the remaining (C-2) quarters sum exactly in F32 and F16.
            const int oc = (i / (17 * 15)) % output_channels;
            const float expected = (oc % 2 ? -1 : 1) * (channels - 2) * 0.25f;
            const auto expected_half = ggml_fp32_to_fp16(expected);
            if (actual[i] != expected_half) {
                std::fprintf(stderr, "F16 accumulation C=%d OC=%d K=%d repeat=%d index=%zu actual=%.9g expected=%.9g\n",
                             channels, output_channels, kernel, repeat, i, ggml_fp16_to_fp32(actual[i]), expected);
                valid = false;
                break;
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return valid;
}

// Detect F16 write-out truncation, incorrect tie rounding and premature fused-epilogue narrowing.
// 1.5 * (1 + 2^-10) is halfway between halves; the even mantissa selects 1.501953125.
static bool run_rounding(ggml_backend_t backend, bool fused) {
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) return false;
    auto* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 17, 15, 3, 2);
    auto* w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, 3, 16);
    auto* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    auto* y = fused ? ggml_conv_2d_direct_bias_silu(ctx, w, x, b, 1, 1, 0, 0, 1, 1)
                    : ggml_conv_2d_direct(ctx, w, x, 1, 1, 0, 0, 1, 1);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<ggml_fp16_t> pixels(ggml_nelements(x), 0), weights(ggml_nelements(w), 0);
    std::vector<float> bias(16);
    for (int batch = 0; batch < 2; ++batch)
        for (int i = 0; i < 17 * 15; ++i)
            pixels[batch * 3 * 17 * 15 + i] = ggml_fp32_to_fp16(1.0009765625f);
    for (int oc = 0; oc < 16; ++oc) {
        const float sign = oc % 2 ? -1.0f : 1.0f;
        weights[oc * 3] = ggml_fp32_to_fp16(sign * (oc % 4 < 2 ? 1.5f : 1.5009765625f));
        bias[oc] = sign * (0.0001220703125f * (oc + 1));
    }
    ggml_backend_tensor_set(x, pixels.data(), 0, pixels.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
    bool valid = true;
    for (int repeat = 0; repeat < 3; ++repeat) {
        ggml_backend_tensor_memset(y, 0x55, 0, ggml_nbytes(y));
        valid &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        std::vector<ggml_fp16_t> actual(ggml_nelements(y));
        ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < actual.size(); ++i) {
            const int oc = (i / (17 * 15)) % 16;
            const double sign = oc % 2 ? -1.0 : 1.0;
            float expected = float(sign * (oc % 4 < 2 ? 1.501953125 : 1.5029296875));
            if (fused) {
                // Independent double formula; canonical F16 rounding occurs only at final output.
                double value = sign * (oc % 4 < 2 ? 1.5 : 1.5009765625) * 1.0009765625 + bias[oc];
                expected = float(value / (1.0 + std::exp(-value)));
            }
            const auto half = ggml_fp32_to_fp16(expected);
            if (actual[i] != half) {
                std::fprintf(stderr, "F16 store rounding fused=%d repeat=%d index=%zu actual=%.9g expected=%.9g\n",
                             fused, repeat, i, ggml_fp16_to_fp32(actual[i]), ggml_fp16_to_fp32(half));
                valid = false;
                break;
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return valid;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    ggml_backend_load_all_from_path(argv[1]);
    auto* device = ggml_backend_dev_by_name(argv[2]);
    if (!device) return 77;
    auto* backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return 1;
    int errors = 0;
    for (bool fused : {false, true}) errors += !run_rounding(backend, fused);
    for (int channels : {64, 128, 256}) for (int output_channels : {16, 32})
    for (int kernel : {1, 3})
        errors += !run(backend, channels, output_channels, kernel);
    ggml_backend_free(backend);
    std::printf("F16 convolution accumulation/rounding: 14 cases, %d failures\n", errors);
    return errors ? 1 : 0;
}
