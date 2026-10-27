// 显式 epilogue 必须保持独立 ADD/GELU 的算术及可观察的中间输出。
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct Output {
    std::vector<float> final;
    std::vector<float> add;
    bool valid = false;
};

static Output run(ggml_backend_t backend, ggml_backend_t cpu, int width, int rows, bool hint,
                  bool observe_add, bool multiple_users, bool strided,
                  bool equal_bias) {
    Output result;
    ggml_context* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    if (!ctx) return result;
    // 峰值约 5×width×rows×4 字节；仅保留当前 case 的结果。
    const int stride = width + (strided ? 7 : 0);
    ggml_tensor* storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                            stride * rows + 3);
    ggml_tensor* x = ggml_view_2d(ctx, storage, width, rows,
                                  stride * sizeof(float), 3 * sizeof(float));
    ggml_tensor* bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width,
                                         equal_bias ? rows : 1);
    ggml_set_input(storage);
    ggml_set_input(bias);
    ggml_tensor* add = ggml_add(ctx, x, bias);
    if (hint) {
        ggml_elementwise_set_epilogue_hint(add, GGML_EPILOGUE_GELU);
        if (ggml_elementwise_get_epilogue_hint(add) != GGML_EPILOGUE_GELU)
            goto cleanup;
    } else if (ggml_elementwise_get_epilogue_hint(add) != 0) {
        goto cleanup;
    }
    {
        ggml_tensor* gelu = ggml_gelu_erf(ctx, add);
        ggml_tensor* final = multiple_users ? ggml_add(ctx, gelu, add) : gelu;
        ggml_set_output(final);
        if (observe_add) ggml_set_output(add);
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, final);
        // 同一个 backend 的前后对比，不把 CPU/GPU 差异当作融合误差。
        ggml_backend_t backends[] = {backend, cpu};
        ggml_backend_sched_t sched = ggml_backend_sched_new(
                backends, nullptr, backend == cpu ? 1 : 2,
                GGML_DEFAULT_GRAPH_SIZE, false, true);
        if (!sched) goto cleanup;
        // scheduler 要求末尾 CPU。显式指派目标设备，避免小张量默认
        // 留在 CPU 而得到并未执行 Vulkan 融合的假阳性。
        ggml_backend_sched_set_tensor_backend(sched, storage, backend);
        ggml_backend_sched_set_tensor_backend(sched, bias, backend);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(graph, i), backend);
        if (ggml_backend_sched_alloc_graph(sched, graph)) {
            std::vector<float> data(stride * rows + 3);
            std::vector<float> offsets(ggml_nelements(bias));
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = (int((i * 37 + i / 11) % 2047) - 1023) / 256.0f;
            for (size_t i = 0; i < offsets.size(); ++i)
                offsets[i] = (int(i % 23) - 11) / 8.0f;
            result.valid = true;
            for (int repeat = 0; repeat < 3; ++repeat) {
                // gallocr 可在最后一次使用后复用输入 view 的存储；每次
                // forward 都重新提交 caller 输入，不把复用后的数据当新帧。
                ggml_backend_tensor_set(storage, data.data(), 0,
                                         data.size() * sizeof(float));
                ggml_backend_tensor_set(bias, offsets.data(), 0,
                                         offsets.size() * sizeof(float));
                result.valid &= ggml_backend_sched_graph_compute(sched, graph) ==
                                GGML_STATUS_SUCCESS;
                std::vector<float> values(width * rows);
                ggml_backend_tensor_get(final, values.data(), 0,
                                         values.size() * sizeof(float));
                for (float value : values) result.valid &= std::isfinite(value);
                if (repeat > 0)
                    result.valid &= result.final.size() == values.size() &&
                                    std::memcmp(result.final.data(), values.data(),
                                                values.size() * sizeof(float)) == 0;
                result.final = std::move(values);
            }
            if (observe_add) {
                result.add.resize(width * rows);
                ggml_backend_tensor_get(add, result.add.data(), 0,
                                         result.add.size() * sizeof(float));
                for (int row = 0; row < rows; ++row)
                    for (int column = 0; column < width; ++column) {
                        const float expected = data[3 + row * stride + column] +
                                offsets[(equal_bias ? row * width : 0) + column];
                        result.valid &= result.add[row * width + column] == expected;
                    }
            }
        }
        ggml_backend_sched_free(sched);
    }
cleanup:
    ggml_free(ctx);
    return result;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_t backend = ggml_backend_init_by_name(argv[2], nullptr);
    if (!backend) return 1;
    const bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) ==
                        GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_t cpu = is_cpu ? backend : ggml_backend_init_by_name("CPU", nullptr);
    if (!cpu) { ggml_backend_free(backend); return 1; }
    bool pass = true;
    int cases = 0;
    for (int width : {1, 31, 128, 512})
        for (int rows : {1, 17})
            for (bool observe_add : {false, true})
                for (bool multiple_users : {false, true})
                    for (bool strided : {false, true})
                        for (bool equal_bias : {false, true}) {
                            const Output plain = run(backend, cpu, width, rows, false,
                                    observe_add, multiple_users, strided, equal_bias);
                            const Output fused = run(backend, cpu, width, rows, true,
                                    observe_add, multiple_users, strided, equal_bias);
                            const bool same = plain.valid && fused.valid &&
                                    plain.final.size() == fused.final.size() &&
                                    std::memcmp(plain.final.data(), fused.final.data(),
                                                plain.final.size() * sizeof(float)) == 0 &&
                                    plain.add == fused.add;
                            pass &= same;
                            ++cases;
                            if (!same)
                                std::fprintf(stderr, "epilogue mismatch: %d,%d output=%d users=%d stride=%d bias=%d\n",
                                             width, rows, observe_add, multiple_users,
                                             strided, equal_bias);
                        }
    if (cpu != backend) ggml_backend_free(cpu);
    ggml_backend_free(backend);
    std::printf("{\"cases\":%d,\"pass\":%s}\n", cases, pass ? "true" : "false");
    return pass ? 0 : 1;
}
