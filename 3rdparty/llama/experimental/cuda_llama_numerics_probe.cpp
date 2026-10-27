// 输出真实 CUDA 算术逐元素位模式；由 runner 对照 pristine llama 与冻结 baseline。
#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool fresh_backend = false;

static bool upload(ggml_tensor* tensor, int seed) {
    std::vector<float> values(ggml_nelements(tensor));
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = float(int((i * seed + 17) % 991) - 495) / 997.0f;
    std::vector<uint8_t> packed(ggml_nbytes(tensor));
    const size_t written = ggml_quantize_chunk(tensor->type, values.data(), packed.data(),
                                              0, ggml_nrows(tensor), tensor->ne[0], nullptr);
    if (written != packed.size()) return false;
    ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
    return true;
}

static bool run(ggml_backend_t backend, const char* id, bool llama,
                int mode, ggml_type weight_type, const char* weight_name,
                const char* input_name, const char* result_name) {
    ggml_backend_t isolated = fresh_backend ? ggml_backend_init_by_name("CUDA0", nullptr) : nullptr;
    if (fresh_backend && !isolated) return false;
    if (isolated) backend = isolated;
    ggml_context* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    if (!ctx) { ggml_backend_free(isolated); return false; }
    std::vector<ggml_tensor*> inputs;
    ggml_tensor* result = nullptr;
    if (mode == 0) {
        auto* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 32, 48, 3);
        auto* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 32, 19, 3);
        ggml_set_name(a, weight_name);
        ggml_set_name(b, input_name);
        inputs = {a, b};
        result = ggml_mul_mat(ctx, a, b);
    } else if (mode == 1) {
        auto* gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        auto* up = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        inputs = {gate, up};
        result = ggml_swiglu_split(ctx, gate, up);
    } else {
        auto* up = ggml_new_tensor_2d(ctx, weight_type, 128, 64);
        auto* gate = ggml_new_tensor_2d(ctx, weight_type, 128, 64);
        auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
        ggml_set_name(up, weight_name);
        ggml_set_name(gate, weight_name);
        inputs = {up, gate, x};
        result = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, gate, x), ggml_mul_mat(ctx, up, x));
    }
    ggml_set_name(result, result_name);
    ggml_set_output(result);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool valid = buffer != nullptr;
    if (valid) {
        int seed = 37;
        for (auto* input : inputs) { valid &= upload(input, seed); seed += 16; }
        valid &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        std::vector<float> values(ggml_nelements(result));
        ggml_backend_tensor_get(result, values.data(), 0, values.size() * sizeof(float));
        std::printf("{\"id\":\"%s\",\"llama\":%s,\"bits\":[", id, llama ? "true" : "false");
        for (size_t i = 0; i < values.size(); ++i) {
            valid &= std::isfinite(values[i]);
            uint32_t bits;
            std::memcpy(&bits, &values[i], sizeof(bits));
            std::printf("%s%u", i ? "," : "", bits);
        }
        std::printf("]}\n");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (isolated) ggml_backend_free(isolated);
    return valid;
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    fresh_backend = argc == 3 && std::strcmp(argv[2], "--fresh-backend") == 0;
    ggml_backend_load_all_from_path(argv[1]);
    auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) return 77;
    bool valid = true;
    valid &= run(backend, "gemm-vision-anonymous", true, 0, GGML_TYPE_F16, "v.blk.0.attn.weight", "", "anonymous");
    valid &= run(backend, "gemm-projector-input", true, 0, GGML_TYPE_F16, "", "mm.projector", "anonymous");
    valid &= run(backend, "gemm-text", true, 0, GGML_TYPE_F16, "blk.0.attn.weight", "", "anonymous");
    valid &= run(backend, "gemm-task-default", false, 0, GGML_TYPE_F16, "rmbg.weight", "", "rmbg_scalar_proj");
    valid &= run(backend, "gemm-sam3d", false, 0, GGML_TYPE_F16, "sam3d.weight", "", "sam3d_proj");
    valid &= run(backend, "swiglu-text", true, 1, GGML_TYPE_F32, "", "", "ffn_swiglu-0");
    valid &= run(backend, "swiglu-task-default", false, 1, GGML_TYPE_F32, "", "", "sam3d_swiglu");
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        valid &= run(backend, type == GGML_TYPE_F16 ? "fused-f16-text" : "fused-q8-text",
                     true, 2, type, "blk.0.ffn_up.weight", "", "ffn_swiglu-0");
        valid &= run(backend, type == GGML_TYPE_F16 ? "fused-f16-task" : "fused-q8-task",
                     false, 2, type, "sam3d.ffn_up.weight", "", "sam3d_swiglu");
    }
    ggml_backend_free(backend);
    return valid ? 0 : 1;
}
