// POOL_2D 必须覆盖完整输出域；超过一个 workgroup 后不能留下未写元素。
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static bool run(ggml_backend_t backend, ggml_type type, ggml_op_pool op,
                int width, int height, int channels, int kernel, int stride) {
    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) return false;
    auto* input = ggml_new_tensor_4d(ctx, type, width, height, channels, 2);
    const int pad = kernel / 2;
    auto* output = ggml_pool_2d(ctx, input, op, kernel, kernel, stride, stride, pad, pad);
    ggml_set_input(input);
    ggml_set_output(output);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> pixels(ggml_nelements(input));
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = float(int(i % 97) - 48);
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(pixels.size());
        ggml_fp32_to_fp16_row(pixels.data(), half.data(), pixels.size());
        ggml_backend_tensor_set(input, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    } else ggml_backend_tensor_set(input, pixels.data(), 0, pixels.size() * sizeof(float));
    bool valid = true;
    for (int repeat = 0; repeat < 3; ++repeat) {
        // 用非零 sentinel 证明每次都写出完整域，不依赖新分配缓冲区碰巧为零。
        ggml_backend_tensor_memset(output, 0x55, 0, ggml_nbytes(output));
        valid &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        std::vector<float> actual(ggml_nelements(output));
        if (type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(actual.size());
            ggml_backend_tensor_get(output, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(half.data(), actual.data(), half.size());
        } else ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        // O(outputs * kernel^2)：独立枚举窗口，验证 padding 和跨通道/batch 地址。
        for (size_t i = 0; i < actual.size(); ++i) {
            const int plane = output->ne[0] * output->ne[1];
            const int nc = i / plane, x = i % output->ne[0];
            const int y = (i % plane) / output->ne[0];
            float expected = op == GGML_OP_POOL_MAX ? -std::numeric_limits<float>::max() : 0;
            for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) {
                const int ix = x * stride + kx - pad, iy = y * stride + ky - pad;
                if (ix < 0 || iy < 0 || ix >= width || iy >= height) continue;
                const float value = pixels[(nc * height + iy) * width + ix];
                expected = op == GGML_OP_POOL_MAX ? std::max(expected, value) : expected + value;
            }
            if (op == GGML_OP_POOL_AVG) expected /= float(kernel * kernel);
            if (type == GGML_TYPE_F16) expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
            // AVG 的除法和乘倒数允许两次 F32 舍入；MAX/F16 仍要求逐位一致。
            const float tolerance = op == GGML_OP_POOL_AVG && type == GGML_TYPE_F32
                    ? 2 * std::numeric_limits<float>::epsilon() * std::max(1.0f, std::abs(expected))
                    : 0.0f;
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected) > tolerance ||
                (tolerance == 0 && std::memcmp(&actual[i], &expected, sizeof(float)) != 0)) {
                std::fprintf(stderr, "POOL_2D type=%s op=%d shape=%dx%dx%d k=%d s=%d repeat=%d index=%zu actual=%.9g expected=%.9g\n",
                             ggml_type_name(type), op, width, height, channels, kernel, stride,
                             repeat, i, actual[i], expected);
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
    auto* dev = ggml_backend_dev_by_name(argv[2]);
    if (!dev) return 77;
    auto* backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 1;
    const bool cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    int errors = 0, cases = 0;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        if (cpu && type == GGML_TYPE_F16) continue;
        for (ggml_op_pool op : {GGML_OP_POOL_MAX, GGML_OP_POOL_AVG})
        // 7/9 同时覆盖 CUDA 的 warp 分派，防止只修小窗口的 scalar 路径。
        for (int channels : {1, 32}) for (int kernel : {1, 3, 5, 7, 9}) for (int stride : {1, 2}) {
            ++cases;
            errors += !run(backend, type, op, 17, 15, channels, kernel, stride);
        }
    }
    ggml_backend_free(backend);
    std::printf("POOL_2D: %d cases, %d failures\n", cases, errors);
    return errors == 0 ? 0 : 1;
}
