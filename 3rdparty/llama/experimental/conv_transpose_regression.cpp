// Independent scalar oracle for batch indexing and input-channel reduction.
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static bool check(ggml_backend_t backend, ggml_type type, int channels,
                  int batches) {
    constexpr int iw = 5, ih = 4, kw = 3, kh = 2, oc = 3, stride = 2;
    constexpr int ow = (iw - 1) * stride + kw;
    constexpr int oh = (ih - 1) * stride + kh;
    ggml_init_params init{2 * 1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(init);
    if (!ctx) return false;
    ggml_tensor* kernel = ggml_new_tensor_4d(ctx, type, kw, kh, oc, channels);
    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, iw, ih, channels,
                                           batches);
    ggml_tensor* output = ggml_conv_transpose_2d_p0(ctx, kernel, input, stride);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr, "conv_transpose_2d unsupported by requested backend\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float> weights(kw * kh * oc * channels);
    std::vector<float> values(iw * ih * channels * batches);
    // Dyadic products and modest sums are exact in F32, independent of the
    // reduction order. Distinct batch/channel patterns expose missing terms.
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = (static_cast<int>(i % 11) - 5) * (1.0f / 64);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = (static_cast<int>((i * 7 + i / (iw * ih)) % 13) - 6) *
                    (1.0f / 32);
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(weights.size());
        for (size_t i = 0; i < half.size(); ++i)
            half[i] = ggml_fp32_to_fp16(weights[i]);
        ggml_backend_tensor_set(kernel, half.data(), 0, half.size() * sizeof(half[0]));
    } else {
        ggml_backend_tensor_set(kernel, weights.data(), 0,
                                weights.size() * sizeof(weights[0]));
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(values[0]));
    bool valid = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(ow * oh * oc * batches);
    if (valid)
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(actual[0]));
    std::vector<float> expected(actual.size(), 0.0f);
    // Scatter formulation: deliberately independent of the shader's gather
    // loops, lane partition and threadgroup reduction.
    for (int batch = 0; batch < batches; ++batch)
        for (int ic = 0; ic < channels; ++ic)
            for (int y = 0; y < ih; ++y)
                for (int x = 0; x < iw; ++x)
                    for (int outc = 0; outc < oc; ++outc)
                        for (int ky = 0; ky < kh; ++ky)
                            for (int kx = 0; kx < kw; ++kx) {
                                const int src = ((batch * channels + ic) * ih + y) * iw + x;
                                const int weight = ((ic * oc + outc) * kh + ky) * kw + kx;
                                const int dst = ((batch * oc + outc) * oh + y * stride + ky) *
                                                ow + x * stride + kx;
                                expected[dst] += values[src] * weights[weight];
                            }
    float max_error = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        valid &= std::isfinite(actual[i]);
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
        mismatches += actual[i] != expected[i];
    }
    valid &= mismatches == 0;
    std::printf("{\"backend\":\"%s\",\"type\":\"%s\",\"channels\":%d,"
                "\"batches\":%d,\"max_error\":%.9g,\"mismatches\":%zu,\"pass\":%s}\n",
                ggml_backend_name(backend), ggml_type_name(type), channels, batches,
                max_error, mismatches, valid ? "true" : "false");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return valid;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <backend-module-directory> <cpu|metal>\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t backend = ggml_backend_init_by_name(argv[2], nullptr);
    if (!backend) return 3;
    bool valid = true;
    for (const ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (const int channels : {3, 48, 192, 257})
            for (const int batches : {1, 2})
                valid = check(backend, type, channels, batches) && valid;
    ggml_backend_free(backend);
    return valid ? 0 : 1;
}
