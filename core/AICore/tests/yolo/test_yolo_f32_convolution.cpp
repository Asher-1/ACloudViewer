// YOLO F32 模型不能在通用卷积 lowering 中静默量化成 F16。
#include <QTemporaryDir>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "tasks/yolo/yolo_graph.hpp"

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)

int run(bool depthwise, const char* device, int size) {
    QTemporaryDir temp;
    CHECK(temp.isValid());
    const auto path = temp.filePath("f32-conv.gguf").toStdString();
    ggml_context* weights = ggml_init({1024 * 1024, nullptr, false});
    CHECK(weights);
    const int output_channels = depthwise ? 3 : size == 1 ? 1 : 16;
    ggml_tensor* kernel = ggml_new_tensor_4d(weights, GGML_TYPE_F32, size, size,
                                            depthwise ? 1 : 3, output_channels);
    ggml_set_name(kernel, "op.0.w");
    std::vector<float> coefficients(ggml_nelements(kernel));
    for (size_t i = 0; i < coefficients.size(); ++i)
        coefficients[i] = float((int(i * 29 % 127) - 63) / 97.0);
    std::copy(coefficients.begin(), coefficients.end(), static_cast<float*>(kernel->data));
    gguf_context* file = gguf_init_empty();
    CHECK(file);
    gguf_set_val_str(file, "yolo.task", "classify");
    gguf_set_val_str(file, "yolo.dtype", "f32");
    gguf_set_val_u32(file, "yolo.nc", 1);
    gguf_set_val_u32(file, "yolo.imgsz", 32);
    gguf_set_val_u32(file, "yolo.op_graph_version", 1);
    gguf_set_val_u32(file, "yolo.op.count", 2);
    gguf_set_val_str(file, "op.0.type", depthwise ? "dwconv" : "conv");
    gguf_set_val_str(file, "op.1.type", "classify");
    const int32_t input = -1, tail = 0;
    gguf_set_arr_data(file, "op.0.inputs", GGUF_TYPE_INT32, &input, 1);
    gguf_set_arr_data(file, "op.1.inputs", GGUF_TYPE_INT32, &tail, 1);
    const uint32_t one[] = {1, 1};
    const uint32_t sizes[] = {uint32_t(size), uint32_t(size)};
    const uint32_t padding[] = {uint32_t(size / 2), uint32_t(size / 2)};
    for (const char* key : {"op.0.s", "op.0.d"})
        gguf_set_arr_data(file, key, GGUF_TYPE_UINT32, one, 2);
    gguf_set_arr_data(file, "op.0.k", GGUF_TYPE_UINT32, sizes, 2);
    gguf_set_arr_data(file, "op.0.p", GGUF_TYPE_UINT32, padding, 2);
    gguf_add_tensor(file, kernel);
    CHECK(gguf_write_to_file(file, path.c_str(), false));
    gguf_free(file);
    ggml_free(weights);

    yolo::SessionOptions options;
    options.threads = 1;
    std::unique_ptr<yolo::Session, decltype(&yolo::free_session)> session(
            yolo::create_session(path, device, options), &yolo::free_session);
    CHECK(session);
    // 相同权重跨 canvas 重建，输入含大量不可由 F16 表达的数。
    for (int width : {32, 16, 32}) {
        CHECK(yolo::session_ensure_canvas(session.get(), width, 32));
        const int plane = width * 32;
        std::vector<float> pixels(plane * 3);
        for (size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = float((int(i * 37 % 991) - 495) / 997.0);
        CHECK(yolo::session_run(session.get(), pixels.data()));
        CHECK(session->output->type == GGML_TYPE_F32);
        std::vector<float> actual(plane * output_channels);
        ggml_backend_tensor_get(session->output, actual.data(), 0,
                                actual.size() * sizeof(float));
        float error = 0;
        for (int i = 0; i < int(actual.size()); ++i) {
            // 独立的 NCHW 卷积解析式；不调用 ggml 或生产 helper。
            const int oc = i / plane, x = i % width, y = (i % plane) / width;
            double sum = 0;
            const int channels = depthwise ? 1 : 3;
            for (int ic = 0; ic < channels; ++ic) {
                for (int ky = 0; ky < size; ++ky) {
                    for (int kx = 0; kx < size; ++kx) {
                        const int sx = x + kx - size / 2;
                        const int sy = y + ky - size / 2;
                        if (sx < 0 || sx >= width || sy < 0 || sy >= 32) continue;
                        const int channel = depthwise ? oc : ic;
                        const size_t wi = ((oc * channels + ic) * size + ky) * size + kx;
                        sum += double(pixels[channel * plane + sy * width + sx]) * coefficients[wi];
                    }
                }
            }
            const float expected = float(sum);
            error = std::max(error, std::abs(actual[i] - expected));
        }
        std::printf("F32 %s k=%d canvas=%dx32 max_error=%.9g\n",
                     depthwise ? "depthwise" : "convolution", size, width, error);
        CHECK(error < 2e-6f);
    }
    return 0;
}

// F16 卷积图仍须消费 GGUF 中原样保存的 F32 bias，不能先窄化 bias 再 ADD。
// 若生产 lowering 将 ±2048.25 舍入为 ±2048，这个单点解析式会从 ±0.25 变成 0。
int run_f16_bias(const char* device) {
    QTemporaryDir temp;
    CHECK(temp.isValid());
    const auto path = temp.filePath("f16-bias.gguf").toStdString();
    auto* weights = ggml_init({1024 * 1024, nullptr, false});
    CHECK(weights);
    constexpr int channels = 16;
    auto* kernel = ggml_new_tensor_4d(weights, GGML_TYPE_F16, 1, 1, 3, channels);
    auto* bias = ggml_new_tensor_1d(weights, GGML_TYPE_F32, channels);
    ggml_set_name(kernel, "op.0.w");
    ggml_set_name(bias, "op.0.b");
    for (int oc = 0; oc < channels; ++oc) {
        auto* values = static_cast<ggml_fp16_t*>(kernel->data) + oc * 3;
        values[0] = ggml_fp32_to_fp16(oc % 2 ? 1.0f : -1.0f);
        values[1] = values[2] = ggml_fp32_to_fp16(0.0f);
        static_cast<float*>(bias->data)[oc] = oc % 2 ? -2048.25f : 2048.25f;
    }
    auto* file = gguf_init_empty();
    CHECK(file);
    gguf_set_val_str(file, "yolo.task", "classify");
    gguf_set_val_str(file, "yolo.dtype", "f16");
    gguf_set_val_u32(file, "yolo.nc", channels);
    gguf_set_val_u32(file, "yolo.imgsz", 32);
    gguf_set_val_u32(file, "yolo.op_graph_version", 1);
    gguf_set_val_u32(file, "yolo.op.count", 2);
    gguf_set_val_str(file, "op.0.type", "conv");
    gguf_set_val_str(file, "op.0.act", "none");
    gguf_set_val_str(file, "op.1.type", "classify");
    const int32_t input = -1, tail = 0;
    gguf_set_arr_data(file, "op.0.inputs", GGUF_TYPE_INT32, &input, 1);
    gguf_set_arr_data(file, "op.1.inputs", GGUF_TYPE_INT32, &tail, 1);
    const uint32_t one[] = {1, 1}, zero[] = {0, 0};
    for (const char* key : {"op.0.k", "op.0.s", "op.0.d"})
        gguf_set_arr_data(file, key, GGUF_TYPE_UINT32, one, 2);
    gguf_set_arr_data(file, "op.0.p", GGUF_TYPE_UINT32, zero, 2);
    gguf_add_tensor(file, kernel);
    gguf_add_tensor(file, bias);
    CHECK(gguf_write_to_file(file, path.c_str(), false));
    gguf_free(file);
    ggml_free(weights);
    yolo::SessionOptions options;
    options.threads = 1;
    std::unique_ptr<yolo::Session, decltype(&yolo::free_session)> session(
            yolo::create_session(path, device, options), &yolo::free_session);
    CHECK(session);
    for (int width : {32, 16, 32}) {
        CHECK(yolo::session_ensure_canvas(session.get(), width, 32));
        const size_t plane = width * 32;
        std::vector<float> pixels(plane * 3, 0.0f);
        std::fill_n(pixels.begin(), plane, 2048.0f);
        CHECK(yolo::session_run(session.get(), pixels.data()));
        std::vector<float> actual(plane * channels);
        CHECK(session->output->type == GGML_TYPE_F32);
        ggml_backend_tensor_get(session->output, actual.data(), 0, actual.size() * sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            const float expected = (i / plane) % 2 ? -0.25f : 0.25f;
            if (actual[i] != expected) {
                std::fprintf(stderr, "F16 weight/F32 bias device=%s canvas=%dx32 index=%zu actual=%.9g expected=%.9g\n",
                             device, width, i, actual[i], expected);
                return 1;
            }
        }
    }
    return 0;
}

// 固定 tensor 地址/形状，模拟 context 销毁后地址被新模型复用；不同 owner 必须重新暂存权重。
// 此解析式只证明 CUDA 缓存身份，不代表真实 checkpoint 精度或性能资格。
int run_f16_plan_owner(const char* device) {
    auto backend = yolo::init_backend_ctx(1, device);
    if (!backend.is_cuda) {
        yolo::free_backend_ctx(backend);
        return 0;
    }
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({1024 * 1024, nullptr, true}), &ggml_free);
    CHECK(context);
    constexpr int width = 16, channels = 16;
    auto* weights = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, 1, 1, 3, channels);
    auto* input = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, width, width, 3, 1);
    auto* output = ggml_conv_2d_direct(context.get(), weights, input, 1, 1, 0, 0, 1, 1);
    CHECK(output && output->type == GGML_TYPE_F16);
    auto* graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(context.get(), backend.gpu), &ggml_backend_buffer_free);
    CHECK(buffer);
    std::vector<ggml_fp16_t> pixels(width * width * 3, ggml_fp32_to_fp16(1.0f));
    ggml_backend_tensor_set(input, pixels.data(), 0, pixels.size() * sizeof(ggml_fp16_t));
    std::vector<ggml_fp16_t> coefficients(channels * 3), actual(width * width * channels);
    for (uint64_t owner : {uint64_t(101), uint64_t(102), uint64_t(103)}) {
        const float weight = owner == 101 ? 1.0f : owner == 102 ? 2.0f : -1.0f;
        std::fill(coefficients.begin(), coefficients.end(), ggml_fp32_to_fp16(weight));
        ggml_backend_tensor_set(weights, coefficients.data(), 0,
                                coefficients.size() * sizeof(ggml_fp16_t));
        std::memcpy(&output->op_params[8], &owner, sizeof(owner));
        CHECK(ggml_backend_graph_compute(backend.gpu, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend.gpu);
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(ggml_fp16_t));
        for (const auto value : actual) {
            const float expected = 3.0f * weight;
            if (ggml_fp16_to_fp32(value) != expected) {
                std::fprintf(stderr, "plan owner=%llu actual=%.9g expected=%.9g\n",
                             static_cast<unsigned long long>(owner), ggml_fp16_to_fp32(value), expected);
                return 1;
            }
        }
    }
    buffer.reset();
    context.reset();
    yolo::free_backend_ctx(backend);
    return 0;
}

int main(int argc, char** argv) {
    const char* device = argc > 1 ? argv[1] : "cpu";
    int status = run_f16_bias(device);
    status |= run_f16_plan_owner(device);
    for (int size : {1, 3}) {
        status |= run(false, device, size);
        status |= run(true, device, size);
    }
    return status;
}
