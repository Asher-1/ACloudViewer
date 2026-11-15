// Checks the dependency entry's installed headers, transitive links and dynamic
// CPU module. This intentionally makes no model accuracy or speed claim.
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>
#include <mtmd.h>

#include <cmath>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    const bool cuda = argc == 2 && std::strcmp(argv[1], "cuda") == 0;
    const bool vulkan = argc == 2 && std::strcmp(argv[1], "vulkan") == 0;
    if (argc > 2 || (argc == 2 && !cuda && !vulkan)) {
        return 1;
    }
    ggml_backend_load_all_from_path(ACV_GGML_MODULE_DIR);
    llama_backend_init();
    const auto model = llama_model_default_params();
    const auto multimodal = mtmd_context_params_default();
    const char* info = llama_print_system_info();
    if (!info || !*info || model.n_gpu_layers < -1 || multimodal.n_threads < 1) {
        return 1;
    }
    ggml_backend_t backend = cuda ? ggml_backend_init_by_name("CUDA0", nullptr)
                        : vulkan ? ggml_backend_init_by_name("Vulkan0", nullptr)
                                 : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        return 2;
    }
    ggml_context* context = ggml_init({1024 * 1024, nullptr, true});
    if (!context) {
        ggml_backend_free(backend);
        return 3;
    }
    // The downstream strict CUDA kernel implements the model's 512-wide rows.
    auto* input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 512);
    auto* output = ggml_l2_norm(context, input, 1e-6f);
    // Resolves an ACloudViewer operator from the patched embedded ggml.
    ggml_l2_norm_set_hint(output, GGML_HINT_STRICT_F32);
    auto* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) {
        ggml_free(context);
        ggml_backend_free(backend);
        return 4;
    }
    const float values[512] = {3.0f, 4.0f};
    ggml_backend_tensor_set(input, values, 0, sizeof(values));
    float normalized[512] = {};
    const auto status = ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, normalized, 0, sizeof(normalized));
    }
    bool correct = status == GGML_STATUS_SUCCESS &&
                         std::isfinite(normalized[0]) && std::isfinite(normalized[1]) &&
                         std::abs(normalized[0] - 0.6f) < 1e-5f &&
                         std::abs(normalized[1] - 0.8f) < 1e-5f;
    for (int i = 2; i < 512; ++i) {
        correct = correct && std::isfinite(normalized[i]) && normalized[i] == 0.0f;
    }
    std::printf("backend=%s strict_l2_norm=[%.8f, %.8f] correct=%d\n",
                ggml_backend_name(backend), normalized[0], normalized[1], correct);
    std::puts(info);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    llama_backend_free();
    return correct ? 0 : 5;
}
