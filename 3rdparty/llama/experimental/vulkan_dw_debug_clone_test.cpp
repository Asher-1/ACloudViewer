// Vulkan 结果检查器的 CPU clone 必须保留真实 DW 节点的可选 bias。
#include "ggml-vulkan-common.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

extern void* comp_result;
extern size_t comp_size;

static bool run(float bias_scale) {
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    auto* cpu = ggml_backend_cpu_init();
    if (!ctx || !cpu) return false;
    auto* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 2, 2);
    auto* weights = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 2);
    auto* bias = bias_scale == 0 ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    auto* output = bias ? ggml_conv_2d_dw_direct_bias(ctx, weights, input, bias, 1, 1, 0, 0, 1, 1)
                        : ggml_conv_2d_dw_direct(ctx, weights, input, 1, 1, 0, 0, 1, 1);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) { ggml_free(ctx); ggml_backend_free(cpu); return false; }
    const float pixels[] = {1, 2, 3, 4};
    const float filter[] = {2, -1};
    const float offsets[] = {3 * bias_scale, -7 * bias_scale};
    const float expected[] = {2 + offsets[0], -2 + offsets[1], 6 + offsets[0], -4 + offsets[1]};
    ggml_backend_tensor_set(input, pixels, 0, sizeof(pixels));
    ggml_backend_tensor_set(weights, filter, 0, sizeof(filter));
    if (bias) ggml_backend_tensor_set(bias, offsets, 0, sizeof(offsets));
    bool valid = ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS;
    float actual[4];
    ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
    valid &= std::memcmp(actual, expected, sizeof(expected)) == 0;
    // 调用真正的检查器，不复制其分支或用替身替代 clone；此处消费 host buffer 路径。
    ggml_backend_vk_context checker{};
    ggml_vk_check_results_0(&checker, graph, ggml_graph_n_nodes(graph) - 1);
    valid &= comp_result && comp_size == sizeof(expected);
    if (comp_result && comp_size == sizeof(expected)) {
        for (int i = 0; i < 4; ++i) {
            const float cloned = static_cast<float*>(comp_result)[i];
            if (std::memcmp(&cloned, &expected[i], sizeof(float))) {
                std::fprintf(stderr, "DW checker bias=%.0f index=%d clone=%.9g expected=%.9g\n",
                             bias_scale, i, cloned, expected[i]);
                valid = false;
            }
        }
    }
    std::free(comp_result);
    comp_result = nullptr;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    return valid;
}

int main() {
    int errors = 0;
    for (float scale : {0.0f, 1.0f, -1.0f}) errors += !run(scale);
    std::printf("DW debug clone: 3 cases, %d failures\n", errors);
    return errors ? 1 : 0;
}
