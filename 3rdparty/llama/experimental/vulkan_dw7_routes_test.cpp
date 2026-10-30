#include "ggml-vulkan.cpp"

static bool route(ggml_backend_vk_context * backend, ggml_type weight_type,
                  int channels, int stride, int dilation, bool cwhn,
                  size_t input_offset, size_t output_offset, bool bias,
                  bool specialized) {
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor * input = cwhn
            ? ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, channels, 17, 13, 2), 2, 0, 1, 3)
            : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 17, 13, channels, 2);
    ggml_tensor * kernel = cwhn
            ? ggml_permute(ctx, ggml_new_tensor_4d(ctx, weight_type, channels, 7, 7, 1), 3, 0, 1, 2)
            : ggml_new_tensor_4d(ctx, weight_type, 7, 7, 1, channels);
    if (cwhn) kernel->nb[2] = ggml_type_size(weight_type);
    ggml_tensor * b = bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels) : nullptr;
    ggml_tensor * out = bias
            ? ggml_conv_2d_dw_direct_bias(ctx, kernel, input, b, stride, stride, 3, 3, dilation, dilation)
            : ggml_conv_2d_dw_direct(ctx, kernel, input, stride, stride, 3, 3, dilation, dilation);
    input->view_src = nullptr;
    input->data = vk_ptr_base;
    input->view_offs = input_offset;
    out->data = vk_ptr_base;
    out->view_offs = output_offset;
    const vk_pipeline selected = ggml_vk_op_get_pipeline(backend, kernel, input, b, out, out->op);
    const std::string name = selected ? selected->name : "missing";
    const bool actual = name.find("conv2d_dw7_cwhn_") == 0;
    std::printf("DW7 route w=%s c=%d s=%d d=%d cwhn=%d input_offset=%zu output_offset=%zu bias=%d pipeline=%s expected_specialized=%d\n",
                ggml_type_name(weight_type), channels, stride, dilation, cwhn,
                input_offset, output_offset, bias, name.c_str(), specialized);
    ggml_free(ctx);
    return selected && actual == specialized;
}

int main() {
    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (!backend) return 77;
    auto * context = static_cast<ggml_backend_vk_context *>(backend->context);
    int errors = 0;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (bool bias : {false, true}) {
            errors += !route(context, type, 12, 1, 1, true, 0, 0, bias, true);
            errors += !route(context, type, 12, 1, 1, true, 16, 16, bias, true);
            errors += !route(context, type, 12, 1, 1, true, 4, 0, bias, false);
            errors += !route(context, type, 12, 1, 1, true, 0, 4, bias, false);
            errors += !route(context, type, 6, 1, 1, true, 0, 0, bias, false);
            errors += !route(context, type, 12, 2, 1, true, 0, 0, bias, false);
            errors += !route(context, type, 12, 1, 2, true, 0, 0, bias, false);
            errors += !route(context, type, 12, 1, 1, false, 0, 0, bias, false);
        }
    }
    ggml_backend_free(backend);
    std::printf("DW7 Vulkan selector: failures=%d cases=32\n", errors);
    return errors ? 1 : 0;
}
