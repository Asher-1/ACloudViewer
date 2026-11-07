// DW src[2] supplies bias at write-out without changing separate ADD arithmetic.
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void upload(ggml_tensor* tensor, const std::vector<float>& values) {
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(values.size());
        ggml_fp32_to_fp16_row(values.data(), half.data(), values.size());
        ggml_backend_tensor_set(tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
}

static bool run(ggml_backend_t backend, ggml_backend_t cpu, int kernel_size,
                int channels, int stride, ggml_type input_type, ggml_type weight_type,
                bool fused, std::vector<float>& output, float bias_scale,
                bool cwhn, int dilation, bool views) {
    ggml_context* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    if (!ctx) return false;
    const size_t prefix = views ? 1 : 0;
    ggml_tensor* weight_storage = ggml_new_tensor_1d(
            ctx, weight_type, kernel_size * kernel_size * channels + prefix);
    ggml_tensor* pixel_storage = ggml_new_tensor_1d(
            ctx, input_type, 17 * 15 * channels * 2 + prefix);
    const size_t wt = ggml_type_size(weight_type), xt = ggml_type_size(input_type);
    ggml_tensor* weights = cwhn
            ? ggml_permute(ctx, ggml_view_4d(ctx, weight_storage, channels, kernel_size,
                    kernel_size, 1, channels * wt, channels * kernel_size * wt,
                    channels * kernel_size * kernel_size * wt, prefix * wt), 3, 0, 1, 2)
            : ggml_view_4d(ctx, weight_storage, kernel_size, kernel_size, 1, channels,
                    kernel_size * wt, kernel_size * kernel_size * wt,
                    kernel_size * kernel_size * wt, prefix * wt);
    // ne[2] is singleton; the CPU CWHN contract uses this stride for channel spacing.
    if (cwhn) weights->nb[2] = wt;
    ggml_tensor* pixels = cwhn
            ? ggml_permute(ctx, ggml_view_4d(ctx, pixel_storage, channels, 17, 15, 2,
                    channels * xt, channels * 17 * xt, channels * 17 * 15 * xt,
                    prefix * xt), 2, 0, 1, 3)
            : ggml_view_4d(ctx, pixel_storage, 17, 15, channels, 2,
                    17 * xt, 17 * 15 * xt, 17 * 15 * channels * xt, prefix * xt);
    // Unaligned views detect ignored descriptor offsets in Vulkan.
    ggml_tensor* bias_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels + 1);
    ggml_tensor* bias = ggml_view_1d(ctx, bias_storage, channels, sizeof(float));
    ggml_set_input(weight_storage);
    ggml_set_input(pixel_storage);
    ggml_set_input(bias_storage);
    ggml_tensor* convolution = ggml_conv_2d_dw_direct(
            ctx, weights, pixels, stride, stride,
            dilation * (kernel_size / 2), dilation * (kernel_size / 2), dilation, dilation);
    ggml_tensor* final;
    if (fused) {
        final = ggml_conv_2d_dw_direct_bias(
                ctx, weights, pixels, bias, stride, stride,
                dilation * (kernel_size / 2), dilation * (kernel_size / 2), dilation, dilation);
    } else {
        // CPU ADD requires contiguous elements; layout conversion copies bits without changing convolution rounding.
        if (!ggml_is_contiguous(convolution)) convolution = ggml_cont(ctx, convolution);
        final = ggml_add(ctx, convolution, ggml_reshape_4d(ctx, bias, 1, 1, channels, 1));
    }
    if (!ggml_backend_supports_op(backend, final)) {
        std::fprintf(stderr, "The requested backend does not support this test's computation path\n");
        ggml_free(ctx);
        return false;
    }
    // ADD outputs WHCN; reorder the CWHN candidate to identical logical element order before comparison.
    if (!ggml_is_contiguous(final)) final = ggml_cont(ctx, final);
    ggml_set_output(final);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final);
    ggml_backend_t backends[] = {backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends, nullptr, backend == cpu ? 1 : 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) { ggml_free(ctx); return false; }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(graph, i), backend);
    for (ggml_tensor* tensor : {pixel_storage, weight_storage, bias_storage})
        ggml_backend_sched_set_tensor_backend(sched, tensor, backend);
    bool valid = ggml_backend_sched_alloc_graph(sched, graph);
    if (valid) {
        std::vector<float> x(ggml_nelements(pixels)), w(ggml_nelements(weights));
        std::vector<float> b(channels + 1, -17.0f);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i * 37 % 991) - 495) / 997.0f;
        for (size_t i = 0; i < w.size(); ++i) w[i] = float(int(i * 19 % 127) - 63) / 257.0f;
        for (int i = 0; i < channels; ++i) b[i + 1] = bias_scale * (i + 1) / 8.0f;
        std::vector<float> stored_x = x, stored_w = w;
        if (views) {
            stored_x.insert(stored_x.begin(), -17.0f);
            stored_w.insert(stored_w.begin(), -17.0f);
        }
        std::vector<float> first;
        for (int repeat = 0; repeat < 3; ++repeat) {
            upload(pixel_storage, stored_x);
            upload(weight_storage, stored_w);
            upload(bias_storage, b);
            valid &= ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
            output.resize(ggml_nelements(final));
            if (final->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(output.size());
                ggml_backend_tensor_get(final, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(half.data(), output.data(), half.size());
            } else {
                ggml_backend_tensor_get(final, output.data(), 0, output.size() * sizeof(float));
            }
            for (float value : output) valid &= std::isfinite(value);
            if (input_type == GGML_TYPE_F16 && kernel_size == 1 && stride == 1 && repeat == 0) {
                // Independent scalar-convolution rounding oracle: convolution writes F16 before ADD.
                for (size_t i = 0; i < output.size(); ++i) {
                    const size_t channel = (i / (17 * 15)) % channels;
                    const size_t pixel_index = cwhn
                            ? (i / (17 * 15 * channels)) * (17 * 15 * channels) +
                              (i % (17 * 15)) * channels + channel
                            : i;
                    const float px = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x[pixel_index]));
                    const float kw = ggml_fp16_to_fp32(ggml_fp32_to_fp16(w[channel]));
                    const float conv = ggml_fp16_to_fp32(ggml_fp32_to_fp16(px * kw));
                    const float expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(conv + b[channel + 1]));
                    if (output[i] != expected) {
                        std::printf("F16 oracle fused=%d index=%zu actual=%.9g expected=%.9g conv=%.9g raw=%.9g bias=%.9g\n",
                                    fused, i, output[i], expected, conv, px * kw, b[channel + 1]);
                        valid = false;
                        break;
                    }
                }
            }
            if (repeat == 0) first = output;
            else valid &= first == output;
        }
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return valid;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_dev_t requested = ggml_backend_dev_by_name(argv[2]);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_name("CPU");
    if (!requested || !cpu_dev) return 77;
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t backend = requested == cpu_dev ? cpu : ggml_backend_dev_init(requested, nullptr);
    if (!cpu || !backend) return 1;
    int errors = 0;
    for (ggml_type input_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
    // CPU DW currently supports only F32 input; include F16 weights in the CPU matrix.
    if (backend == cpu && input_type == GGML_TYPE_F16) continue;
    for (ggml_type weight_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
    if (input_type == GGML_TYPE_F16 && weight_type != GGML_TYPE_F16) continue;
    for (int size : {1, 3, 7}) for (int channels : {1, 8, 32}) for (int stride : {1, 2})
    for (float bias_scale : {0.0f, 1.0f, -1.0f, 0.24f})
    for (bool cwhn : {false, true}) for (int dilation : {1, 2})
    for (bool views : {false, true}) {
        std::vector<float> reference, candidate;
        if (!run(backend, cpu, size, channels, stride, input_type, weight_type, false,
                 reference, bias_scale, cwhn, dilation, views) ||
            !run(backend, cpu, size, channels, stride, input_type, weight_type, true,
                 candidate, bias_scale, cwhn, dilation, views) ||
            reference.size() != candidate.size()) {
            ++errors;
            continue;
        }
        float max_error = 0;
        for (size_t i = 0; i < reference.size(); ++i)
            max_error = std::max(max_error, std::abs(reference[i] - candidate[i]));
        const bool equal = std::memcmp(reference.data(), candidate.data(),
                                       reference.size() * sizeof(float)) == 0;
        std::printf("DW+bias x=%s w=%s k=%d c=%d s=%d bias_scale=%.3g layout=%s d=%d views=%d max_error=%.9g bitwise=%d\n",
                    ggml_type_name(input_type), ggml_type_name(weight_type),
                    size, channels, stride, bias_scale, cwhn ? "CWHN" : "WHCN",
                    dilation, views, max_error, equal);
        if (!equal) ++errors;
    }
    }
    }
    if (backend != cpu) ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return errors == 0 ? 0 : 1;
}
