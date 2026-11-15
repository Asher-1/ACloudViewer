// Native-output cuBLAS stride regression: single, batched and broadcast GEMM.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static bool check(ggml_backend_t backend, ggml_type type, int batches,
                  bool broadcast) {
    constexpr int k = 32, m = 48, n = 19;
    const int weight_batches = broadcast ? 1 : batches;
    ggml_init_params init{1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(init);
    ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, weight_batches);
    ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, n, batches);
    ggml_tensor* c = ggml_mul_mat_cast(ctx, a, b, type);
    ggml_set_name(c, "sam3_native_stride_regression");
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    if (!ggml_backend_supports_op(backend, c)) {
        std::fprintf(stderr, "requested native GEMM unsupported\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    std::vector<ggml_fp16_t> av(k * m * weight_batches), bv(k * n * batches);
    for (size_t i = 0; i < av.size(); ++i)
        av[i] = ggml_fp32_to_fp16((static_cast<int>(i % 7) - 3) * 0.125f);
    for (size_t i = 0; i < bv.size(); ++i)
        bv[i] = ggml_fp32_to_fp16((static_cast<int>(i % 5) - 2) * 0.125f);
    ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(av[0]));
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(bv[0]));
    bool valid = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> output(m * n * batches);
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(output.size());
        ggml_backend_tensor_get(c, half.data(), 0, half.size() * sizeof(half[0]));
        for (size_t i = 0; i < output.size(); ++i) output[i] = ggml_fp16_to_fp32(half[i]);
    } else {
        ggml_backend_tensor_get(c, output.data(), 0, output.size() * sizeof(output[0]));
    }
    float max_error = 0;
    // O(batch*m*n*k) independent scalar oracle; these small dyadic inputs
    // have exact F16 products/sums and reveal row or batch overlap directly.
    for (int batch = 0; batch < batches; ++batch)
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < m; ++row) {
                float expected = 0;
                for (int inner = 0; inner < k; ++inner) {
                    const int ab = broadcast ? 0 : batch;
                    expected += ggml_fp16_to_fp32(av[(ab * m + row) * k + inner]) *
                                ggml_fp16_to_fp32(bv[(batch * n + col) * k + inner]);
                }
                const float actual = output[(batch * n + col) * m + row];
                valid &= std::isfinite(actual);
                max_error = std::max(max_error, std::fabs(actual - expected));
            }
    valid &= max_error == 0;
    std::printf("{\"type\":\"%s\",\"batches\":%d,\"broadcast\":%s,"
                "\"max_error\":%.9g,\"pass\":%s}\n",
                ggml_type_name(type), batches, broadcast ? "true" : "false",
                max_error, valid ? "true" : "false");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return valid;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) return 1;
    bool valid = true;
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        valid &= check(backend, type, 1, false);
        valid &= check(backend, type, 3, false);
        valid &= check(backend, type, 3, true);
    }
    ggml_backend_free(backend);
    return valid ? 0 : 1;
}
