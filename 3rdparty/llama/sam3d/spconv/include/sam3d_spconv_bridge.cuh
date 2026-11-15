#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

// This bridge is intentionally a small C ABI so the ggml CUDA backend does
// not need to include Cumm/spconv implementation headers.  The implementation
// calls the exact Ampere F16 algorithm selected by spconv 2.3.8 for this model.
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
    int64_t output_channels);
