#include "sam3d_spconv_bridge.cuh"

#include <sam3d_spconv/sam3d/spconv/ConvMainUnitTest.h>
#include <tensorview/tensor.h>

namespace spconv = sam3d_spconv::sam3d::spconv;

namespace {

template <typename T>
tv::Tensor tensor_view(T * pointer, std::initializer_list<int64_t> shape,
                       tv::DType dtype) {
    return tv::from_blob(pointer, tv::TensorShape(shape), dtype, 0);
}

}  // namespace

extern "C" void sam3d_spconv_f16_forward(
    cudaStream_t stream,
    const void * input,
    const void * weight,
    const void * bias,
    void * output,
    const int32_t * indices,
    const uint32_t * mask,
    const int32_t * mask_argsort,
    int64_t tokens,
    int64_t channels,
    int64_t output_channels) {
    constexpr int64_t kernel_volume = 27;
    tv::gemm::ConvParams params(3, tv::gemm::ConvOpType::kForward,
                                tv::CUDAKernelTimer(false));
    params.conv_algo_desp = spconv::ConvMainUnitTest::get_all_conv_algo_desp().at(0);
    params.input = tensor_view(const_cast<void *>(input), {tokens, channels}, tv::float16);
    params.weight = tensor_view(const_cast<void *>(weight),
                                {output_channels, kernel_volume, channels}, tv::float16);
    params.output = tensor_view(output, {tokens, output_channels}, tv::float16);
    params.bias = tensor_view(const_cast<void *>(bias), {output_channels}, tv::float16);
    params.indices = tensor_view(const_cast<int32_t *>(indices),
                                 {kernel_volume, tokens}, tv::int32);
    params.mask = tensor_view(const_cast<uint32_t *>(mask), {tokens, 1}, tv::uint32);
    params.mask_argsort = tensor_view(const_cast<int32_t *>(mask_argsort),
                                      {tokens}, tv::int32);
    params.mask_width = -1;
    params.mask_filter = 0xffffffffu;
    params.mask_int_count = 1;
    params.beta = tv::half_t(1);
    params.stream = reinterpret_cast<std::uintptr_t>(stream);
    spconv::ConvMainUnitTest::matmul_split_Ampere_f16f16f16_0(params);
}
