// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <ggml-backend.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "aliked_cuda.hpp"
#include "ggml_backend_util.hpp"

namespace lightglue::aliked_internal {

constexpr int32_t kTopKBlockCount = 256;
constexpr int32_t kTopKLocal = 32;

struct AlikedDkdScratch::Impl {
    thrust::device_vector<float> keys;
    thrust::device_vector<int32_t> indices;
    thrust::device_vector<float> block_keys;
    thrust::device_vector<int32_t> block_indices;
    float *nms = nullptr;
    float *tmp_a = nullptr;
    float *tmp_b = nullptr;
    float *tmp_c = nullptr;
    int32_t capacity = 0;
};

struct AlikedSddhScratch::Impl {
    float *workspace = nullptr;
    int32_t capacity_count = 0;
    int32_t dim = 0;
    int32_t kernel_size = 0;
};

AlikedDkdScratch::AlikedDkdScratch() : impl(std::make_unique<Impl>()) {}

AlikedSddhScratch::AlikedSddhScratch() : impl(std::make_unique<Impl>()) {}

AlikedDkdScratch::~AlikedDkdScratch() { Release(); }

AlikedSddhScratch::~AlikedSddhScratch() { Release(); }

void AlikedDkdScratch::Ensure(int32_t count) {
    if (count <= impl->capacity) {
        return;
    }
    Release();
    impl->keys.resize(static_cast<size_t>(count));
    impl->indices.resize(static_cast<size_t>(count));
    impl->block_keys.resize(static_cast<size_t>(kTopKBlockCount * kTopKLocal));
    impl->block_indices.resize(
            static_cast<size_t>(kTopKBlockCount * kTopKLocal));
    cudaMalloc(&impl->nms, static_cast<size_t>(count) * sizeof(float));
    cudaMalloc(&impl->tmp_a, static_cast<size_t>(count) * sizeof(float));
    cudaMalloc(&impl->tmp_b, static_cast<size_t>(count) * sizeof(float));
    cudaMalloc(&impl->tmp_c, static_cast<size_t>(count) * sizeof(float));
    impl->capacity = count;
}

size_t SddhWorkspaceFloats(int32_t count, int32_t dim, int32_t kernel_size) {
    const size_t per = static_cast<size_t>(dim) *
                               static_cast<size_t>(kernel_size) *
                               static_cast<size_t>(kernel_size) +
                       64 + static_cast<size_t>(dim) * 3;
    return per * static_cast<size_t>(count);
}

void AlikedSddhScratch::Ensure(int32_t count,
                               int32_t dim,
                               int32_t kernel_size) {
    if (count <= impl->capacity_count && dim == impl->dim &&
        kernel_size == impl->kernel_size) {
        return;
    }
    Release();
    const size_t floats = SddhWorkspaceFloats(count, dim, kernel_size);
    cudaMalloc(&impl->workspace, floats * sizeof(float));
    impl->capacity_count = count;
    impl->dim = dim;
    impl->kernel_size = kernel_size;
}

void AlikedSddhScratch::Release() {
    if (impl->workspace != nullptr) {
        cudaFree(impl->workspace);
        impl->workspace = nullptr;
    }
    impl->capacity_count = 0;
    impl->dim = 0;
    impl->kernel_size = 0;
}

void AlikedDkdScratch::Release() {
    if (impl->nms != nullptr) {
        cudaFree(impl->nms);
        impl->nms = nullptr;
    }
    if (impl->tmp_a != nullptr) {
        cudaFree(impl->tmp_a);
        impl->tmp_a = nullptr;
    }
    if (impl->tmp_b != nullptr) {
        cudaFree(impl->tmp_b);
        impl->tmp_b = nullptr;
    }
    if (impl->tmp_c != nullptr) {
        cudaFree(impl->tmp_c);
        impl->tmp_c = nullptr;
    }
    impl->capacity = 0;
    impl->keys.clear();
    impl->indices.clear();
    impl->block_keys.clear();
    impl->block_indices.clear();
}

namespace {

constexpr float kSeluScale = 1.050700987f;
constexpr float kSeluAlpha = 1.67326324f;

struct DescendingByFirst {
    __host__ __device__ bool operator()(
            const thrust::tuple<float, int32_t> &a,
            const thrust::tuple<float, int32_t> &b) const {
        return thrust::get<0>(a) > thrust::get<0>(b);
    }
};

bool CudaStreamSync() {
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

__device__ void InsertLocalTopK(
        float value, int32_t index, float *keys, int32_t *indices, int32_t k) {
    if (value <= keys[k - 1]) {
        return;
    }
    keys[k - 1] = value;
    indices[k - 1] = index;
    for (int32_t i = k - 1; i > 0 && keys[i] > keys[i - 1]; --i) {
        const float tk = keys[i - 1];
        keys[i - 1] = keys[i];
        keys[i] = tk;
        const int32_t ti = indices[i - 1];
        indices[i - 1] = indices[i];
        indices[i] = ti;
    }
}

__global__ void BlockTopKKernel(const float *nms,
                                int32_t count,
                                int32_t k_local,
                                float *block_keys,
                                int32_t *block_indices) {
    const int32_t block = static_cast<int32_t>(blockIdx.x);
    const int32_t num_blocks = static_cast<int32_t>(gridDim.x);
    float keys[32];
    int32_t indices[32];
    for (int32_t i = 0; i < k_local; ++i) {
        keys[i] = -INFINITY;
        indices[i] = -1;
    }
    for (int32_t i = block; i < count; i += num_blocks) {
        InsertLocalTopK(nms[i], i, keys, indices, k_local);
    }
    for (int32_t i = 0; i < k_local; ++i) {
        const int32_t out = block * k_local + i;
        block_keys[out] = keys[i];
        block_indices[out] = indices[i];
    }
}

[[maybe_unused]] bool PartialSortTopK(
        const float *nms,
        thrust::device_vector<float> *block_keys,
        thrust::device_vector<int32_t> *block_indices,
        thrust::device_vector<float> &keys,
        thrust::device_vector<int32_t> &indices,
        int32_t count,
        int32_t keep) {
    const int32_t candidates = kTopKBlockCount * kTopKLocal;
    BlockTopKKernel<<<kTopKBlockCount, 1>>>(
            nms, count, kTopKLocal,
            thrust::raw_pointer_cast(block_keys->data()),
            thrust::raw_pointer_cast(block_indices->data()));
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    cudaMemcpy(thrust::raw_pointer_cast(keys.data()),
               thrust::raw_pointer_cast(block_keys->data()),
               static_cast<size_t>(candidates) * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(thrust::raw_pointer_cast(indices.data()),
               thrust::raw_pointer_cast(block_indices->data()),
               static_cast<size_t>(candidates) * sizeof(int32_t),
               cudaMemcpyDeviceToDevice);

    const auto zip_begin = thrust::make_zip_iterator(
            thrust::make_tuple(keys.begin(), indices.begin()));
    thrust::sort(zip_begin, zip_begin + candidates, DescendingByFirst());
    (void)keep;
    return CudaStreamSync();
}

__device__ __host__ inline int32_t IndexNchw(
        int32_t c, int32_t y, int32_t x, int32_t h, int32_t w) {
    return c * h * w + y * w + x;
}

__device__ float BilinearSampleDeformDevice(const float *tensor,
                                            int32_t c,
                                            int32_t h,
                                            int32_t w,
                                            float y,
                                            float x) {
    if (y <= -1.0f || y >= static_cast<float>(h) || x <= -1.0f ||
        x >= static_cast<float>(w)) {
        return 0.0f;
    }

    const int32_t y0 = static_cast<int32_t>(floorf(y));
    const int32_t x0 = static_cast<int32_t>(floorf(x));
    const int32_t y1 = y0 + 1;
    const int32_t x1 = x0 + 1;
    const float ly = y - static_cast<float>(y0);
    const float lx = x - static_cast<float>(x0);
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;

    auto at = [&](int32_t yy, int32_t xx) -> float {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) {
            return 0.0f;
        }
        return tensor[IndexNchw(c, yy, xx, h, w)];
    };

    return hy * hx * at(y0, x0) + hy * lx * at(y0, x1) + ly * hx * at(y1, x0) +
           ly * lx * at(y1, x1);
}

__global__ void DeformConv2dKernel(const float *input,
                                   const float *offset,
                                   const float *weight,
                                   const float *bias,
                                   float *output,
                                   int32_t ic,
                                   int32_t ih,
                                   int32_t iw,
                                   int32_t oc,
                                   int32_t kh,
                                   int32_t kw,
                                   int32_t pad) {
    const int32_t ox =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t oy =
            static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const int32_t o = static_cast<int32_t>(blockIdx.z);
    if (ox >= iw || oy >= ih || o >= oc) {
        return;
    }

    float sum = bias != nullptr ? bias[o] : 0.0f;
    for (int32_t i = 0; i < ic; ++i) {
        for (int32_t ky = 0; ky < kh; ++ky) {
            for (int32_t kx = 0; kx < kw; ++kx) {
                const int32_t k_idx = ky * kw + kx;
                const int32_t off_c_y = k_idx * 2 + 0;
                const int32_t off_c_x = k_idx * 2 + 1;
                const float sample_y =
                        static_cast<float>(oy - pad + ky) +
                        offset[IndexNchw(off_c_y, oy, ox, ih, iw)];
                const float sample_x =
                        static_cast<float>(ox - pad + kx) +
                        offset[IndexNchw(off_c_x, oy, ox, ih, iw)];
                const size_t widx = static_cast<size_t>(o) * ic * kh * kw +
                                    static_cast<size_t>(i) * kh * kw + ky * kw +
                                    kx;
                sum += BilinearSampleDeformDevice(input, i, ih, iw, sample_y,
                                                  sample_x) *
                       weight[widx];
            }
        }
    }
    output[IndexNchw(o, oy, ox, ih, iw)] = sum;
}

__global__ void SeluKernel(float *data, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float x = data[i];
    data[i] = x > 0.0f ? kSeluScale * x
                       : kSeluScale * kSeluAlpha * (expf(x) - 1.0f);
}

__global__ void ClampKernel(float *data,
                            size_t count,
                            float min_value,
                            float max_value) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    data[i] = fminf(fmaxf(data[i], min_value), max_value);
}

__global__ void SigmoidKernel(float *data, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    data[i] = 1.0f / (1.0f + expf(-data[i]));
}

__global__ void L2NormalizeChannelsKernel(float *data,
                                          int32_t c,
                                          int32_t h,
                                          int32_t w) {
    const int32_t spatial = h * w;
    const int32_t idx =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= spatial) {
        return;
    }
    float norm = 0.0f;
    for (int32_t ch = 0; ch < c; ++ch) {
        const float v = data[ch * spatial + idx];
        norm += v * v;
    }
    norm = sqrtf(fmaxf(norm, 1e-12f));
    for (int32_t ch = 0; ch < c; ++ch) {
        data[ch * spatial + idx] /= norm;
    }
}

__global__ void WhcnToNchwKernel(
        const float *whcn, float *nchw, int32_t c, int32_t h, int32_t w) {
    const int32_t x =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y);
    const int32_t ch = static_cast<int32_t>(blockIdx.z);
    if (x >= w || y >= h || ch >= c) {
        return;
    }
    nchw[IndexNchw(ch, y, x, h, w)] = whcn[x + y * w + ch * h * w];
}

__global__ void NchwToWhcnKernel(
        const float *nchw, float *whcn, int32_t c, int32_t h, int32_t w) {
    const int32_t x =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y);
    const int32_t ch = static_cast<int32_t>(blockIdx.z);
    if (x >= w || y >= h || ch >= c) {
        return;
    }
    whcn[x + y * w + ch * h * w] = nchw[IndexNchw(ch, y, x, h, w)];
}

__global__ void ConcatChannelKernel(const float *a,
                                    const float *b,
                                    float *out,
                                    int32_t ca,
                                    int32_t cb,
                                    int32_t h,
                                    int32_t w) {
    const int32_t x =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y);
    const int32_t ch = static_cast<int32_t>(blockIdx.z);
    const int32_t total_c = ca + cb;
    if (x >= w || y >= h || ch >= total_c) {
        return;
    }
    const int32_t whcn_idx = x + y * w + ch * h * w;
    if (ch < ca) {
        out[whcn_idx] = a[x + y * w + ch * h * w];
    } else {
        const int32_t bch = ch - ca;
        out[whcn_idx] = b[x + y * w + bch * h * w];
    }
}

}  // namespace

bool AlikedCudaDeformConv2d(ggml_backend_t backend,
                            const float *input,
                            int32_t ic,
                            int32_t ih,
                            int32_t iw,
                            const float *offset,
                            const float *weight,
                            const float *bias,
                            int32_t oc,
                            int32_t kh,
                            int32_t kw,
                            int32_t pad,
                            float *output) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }

    dim3 block(16, 16, 1);
    dim3 grid((static_cast<unsigned>(iw) + block.x - 1) / block.x,
              (static_cast<unsigned>(ih) + block.y - 1) / block.y,
              static_cast<unsigned>(oc));
    DeformConv2dKernel<<<grid, block>>>(input, offset, weight, bias, output, ic,
                                        ih, iw, oc, kh, kw, pad);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaWhcnToNchw(ggml_backend_t backend,
                          const float *whcn,
                          float *nchw,
                          int32_t c,
                          int32_t h,
                          int32_t w) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    dim3 block(16, 1, 1);
    dim3 grid((static_cast<unsigned>(w) + block.x - 1) / block.x,
              static_cast<unsigned>(h), static_cast<unsigned>(c));
    WhcnToNchwKernel<<<grid, block>>>(whcn, nchw, c, h, w);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaNchwToWhcn(ggml_backend_t backend,
                          const float *nchw,
                          float *whcn,
                          int32_t c,
                          int32_t h,
                          int32_t w) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    dim3 block(16, 1, 1);
    dim3 grid((static_cast<unsigned>(w) + block.x - 1) / block.x,
              static_cast<unsigned>(h), static_cast<unsigned>(c));
    NchwToWhcnKernel<<<grid, block>>>(nchw, whcn, c, h, w);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaConcatChannel(ggml_backend_t backend,
                             const float *a,
                             int32_t ca,
                             const float *b,
                             int32_t cb,
                             int32_t h,
                             int32_t w,
                             float *output) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    const int32_t total_c = ca + cb;
    dim3 block(16, 1, 1);
    dim3 grid((static_cast<unsigned>(w) + block.x - 1) / block.x,
              static_cast<unsigned>(h), static_cast<unsigned>(total_c));
    ConcatChannelKernel<<<grid, block>>>(a, b, output, ca, cb, h, w);
    return cudaGetLastError() == cudaSuccess;
}

__device__ inline int32_t WhcnIndex(
        int32_t x, int32_t y, int32_t ch, int32_t h, int32_t w) {
    return x + y * w + ch * h * w;
}

__global__ void AvgPoolKernel(const float *input,
                              float *output,
                              int32_t ic,
                              int32_t ih,
                              int32_t iw,
                              int32_t kh,
                              int32_t kw,
                              int32_t stride,
                              int32_t oh,
                              int32_t ow) {
    const int32_t ox =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t oy = static_cast<int32_t>(blockIdx.y);
    const int32_t ch = static_cast<int32_t>(blockIdx.z);
    if (ox >= ow || oy >= oh || ch >= ic) {
        return;
    }

    float sum = 0.0f;
    for (int32_t ky = 0; ky < kh; ++ky) {
        for (int32_t kx = 0; kx < kw; ++kx) {
            const int32_t iy = oy * stride + ky;
            const int32_t ix = ox * stride + kx;
            if (iy < ih && ix < iw) {
                sum += input[WhcnIndex(ix, iy, ch, ih, iw)];
            }
        }
    }
    output[WhcnIndex(ox, oy, ch, oh, ow)] = sum / static_cast<float>(kh * kw);
}

__global__ void UpsampleBilinearKernel(const float *input,
                                       float *output,
                                       int32_t ic,
                                       int32_t ih,
                                       int32_t iw,
                                       int32_t out_h,
                                       int32_t out_w) {
    const int32_t ox =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t oy = static_cast<int32_t>(blockIdx.y);
    const int32_t ch = static_cast<int32_t>(blockIdx.z);
    if (ox >= out_w || oy >= out_h || ch >= ic) {
        return;
    }

    const float scale_y = out_h > 1 ? static_cast<float>(ih - 1) /
                                              static_cast<float>(out_h - 1)
                                    : 0.0f;
    const float scale_x = out_w > 1 ? static_cast<float>(iw - 1) /
                                              static_cast<float>(out_w - 1)
                                    : 0.0f;
    const float in_y = oy * scale_y;
    const float in_x = ox * scale_x;
    const int32_t y0 = static_cast<int32_t>(floorf(in_y));
    const int32_t x0 = static_cast<int32_t>(floorf(in_x));
    const int32_t y1 = min(y0 + 1, ih - 1);
    const int32_t x1 = min(x0 + 1, iw - 1);
    const float ly = in_y - static_cast<float>(y0);
    const float lx = in_x - static_cast<float>(x0);
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;

    auto at = [&](int32_t yy, int32_t xx) -> float {
        return input[WhcnIndex(xx, yy, ch, ih, iw)];
    };
    output[WhcnIndex(ox, oy, ch, out_h, out_w)] =
            hy * hx * at(y0, x0) + hy * lx * at(y0, x1) + ly * hx * at(y1, x0) +
            ly * lx * at(y1, x1);
}

__global__ void CropWhcnKernel(const float *input,
                               float *output,
                               int32_t ic,
                               int32_t padded_h,
                               int32_t padded_w,
                               int32_t pad_top,
                               int32_t pad_left,
                               int32_t out_h,
                               int32_t out_w) {
    const int32_t ox =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t oy = static_cast<int32_t>(blockIdx.y);
    const int32_t ch = static_cast<int32_t>(blockIdx.z);
    if (ox >= out_w || oy >= out_h || ch >= ic) {
        return;
    }
    output[WhcnIndex(ox, oy, ch, out_h, out_w)] = input[WhcnIndex(
            ox + pad_left, oy + pad_top, ch, padded_h, padded_w)];
}

__global__ void MaxPoolKernel(
        const float *input, float *output, int32_t h, int32_t w, int32_t pad) {
    const int32_t x =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y);
    if (x >= w || y >= h) {
        return;
    }
    float best = -INFINITY;
    for (int32_t ky = -pad; ky <= pad; ++ky) {
        for (int32_t kx = -pad; kx <= pad; ++kx) {
            const int32_t iy = y + ky;
            const int32_t ix = x + kx;
            if (iy < 0 || ix < 0 || iy >= h || ix >= w) {
                continue;
            }
            best = fmaxf(best, input[static_cast<size_t>(iy) * w + ix]);
        }
    }
    output[static_cast<size_t>(y) * w + x] = best;
}

__global__ void CompareEqualMaskKernel(const float *scores,
                                       const float *max_vals,
                                       float *mask,
                                       int32_t count) {
    const int32_t i =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) {
        return;
    }
    mask[i] = fabsf(scores[i] - max_vals[i]) <= 1e-6f ? 1.0f : 0.0f;
}

__global__ void ThresholdPositiveKernel(const float *input,
                                        float *output,
                                        int32_t count) {
    const int32_t i =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) {
        return;
    }
    output[i] = input[i] > 0.0f ? 1.0f : 0.0f;
}

__global__ void SuppressScoresKernel(const float *supp,
                                     float *result,
                                     int32_t count) {
    const int32_t i =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) {
        return;
    }
    if (supp[i] > 0.0f) {
        result[i] = 0.0f;
    }
}

__global__ void UpdateMaxMaskKernel(const float *result,
                                    const float *new_max,
                                    const float *supp,
                                    float *max_mask,
                                    int32_t count) {
    const int32_t i =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) {
        return;
    }
    if (fabsf(result[i] - new_max[i]) <= 1e-6f) {
        max_mask[i] = 1.0f;
    } else if (supp[i] <= 0.0f) {
        max_mask[i] = 0.0f;
    }
}

__global__ void ApplyMaxMaskKernel(const float *scores,
                                   const float *max_mask,
                                   float *result,
                                   int32_t count) {
    const int32_t i =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) {
        return;
    }
    result[i] = max_mask[i] > 0.0f ? scores[i] : 0.0f;
}

__global__ void ZeroBorderKernel(float *data,
                                 int32_t h,
                                 int32_t w,
                                 int32_t radius) {
    const int32_t x =
            static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t y = static_cast<int32_t>(blockIdx.y);
    if (x >= w || y >= h) {
        return;
    }
    if (y < radius || x < radius || y >= h - radius || x >= w - radius) {
        data[static_cast<size_t>(y) * w + x] = 0.0f;
    }
}

constexpr float kDkdTemperature = 0.1f;

__device__ inline bool IsValidSampleCoord(float y,
                                          float x,
                                          int32_t h,
                                          int32_t w) {
    return isfinite(y) && isfinite(x) && y > -1.0f &&
           y < static_cast<float>(h) && x > -1.0f && x < static_cast<float>(w);
}

__device__ float BilinearSampleScoreDevice(
        const float *scores, int32_t h, int32_t w, float y, float x) {
    if (!IsValidSampleCoord(y, x, h, w)) {
        return 0.0f;
    }
    const int32_t y0 = static_cast<int32_t>(floorf(y));
    const int32_t x0 = static_cast<int32_t>(floorf(x));
    const int32_t y1 = min(y0 + 1, h - 1);
    const int32_t x1 = min(x0 + 1, w - 1);
    const float ly = y - static_cast<float>(y0);
    const float lx = x - static_cast<float>(x0);
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;
    auto at = [&](int32_t yy, int32_t xx) -> float {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) {
            return 0.0f;
        }
        return scores[static_cast<size_t>(yy) * w + xx];
    };
    return hy * hx * at(y0, x0) + hy * lx * at(y0, x1) + ly * hx * at(y1, x0) +
           ly * lx * at(y1, x1);
}

__global__ void RefineKeypointsKernel(const float *score_map,
                                      const int32_t *indices,
                                      int32_t count,
                                      int32_t h,
                                      int32_t w,
                                      int32_t radius,
                                      float *keypoints_norm,
                                      float *scores_out) {
    const int32_t k = static_cast<int32_t>(blockIdx.x);
    if (k >= count) {
        return;
    }

    const int32_t index = indices[k];
    const int32_t x_nms = index % w;
    const int32_t y_nms = index / w;
    const int32_t kernel = radius * 2 + 1;
    const int32_t kernel_area = kernel * kernel;

    float patch_scores[25];
    float hw_grid[50];
    int32_t idx = 0;
    for (int32_t ky = -radius; ky <= radius; ++ky) {
        for (int32_t kx = -radius; kx <= radius; ++kx) {
            const int32_t py = min(max(y_nms + ky, 0), h - 1);
            const int32_t px = min(max(x_nms + kx, 0), w - 1);
            patch_scores[idx] = score_map[static_cast<size_t>(py) * w + px];
            hw_grid[idx * 2 + 0] = static_cast<float>(kx);
            hw_grid[idx * 2 + 1] = static_cast<float>(ky);
            ++idx;
        }
    }

    float max_v = patch_scores[0];
    for (int32_t i = 1; i < kernel_area; ++i) {
        max_v = fmaxf(max_v, patch_scores[i]);
    }

    float exp_sum = 0.0f;
    float x_exp[25];
    for (int32_t i = 0; i < kernel_area; ++i) {
        x_exp[i] = expf((patch_scores[i] - max_v) / kDkdTemperature);
        exp_sum += x_exp[i];
    }

    exp_sum = fmaxf(exp_sum, 1e-12f);
    float residual_x = 0.0f;
    float residual_y = 0.0f;
    for (int32_t i = 0; i < kernel_area; ++i) {
        const float weight = x_exp[i] / exp_sum;
        residual_x += weight * hw_grid[i * 2 + 0];
        residual_y += weight * hw_grid[i * 2 + 1];
    }

    const float wh_x = static_cast<float>(w - 1);
    const float wh_y = static_cast<float>(h - 1);
    if (!isfinite(wh_x) || !isfinite(wh_y) || wh_x <= 0.0f || wh_y <= 0.0f) {
        keypoints_norm[k * 2 + 0] = 0.0f;
        keypoints_norm[k * 2 + 1] = 0.0f;
        scores_out[k] = 0.0f;
        return;
    }
    const float x =
            (static_cast<float>(x_nms) + residual_x) / wh_x * 2.0f - 1.0f;
    const float y =
            (static_cast<float>(y_nms) + residual_y) / wh_y * 2.0f - 1.0f;
    if (!isfinite(x) || !isfinite(y)) {
        keypoints_norm[k * 2 + 0] = 0.0f;
        keypoints_norm[k * 2 + 1] = 0.0f;
        scores_out[k] = 0.0f;
        return;
    }
    keypoints_norm[k * 2 + 0] = x;
    keypoints_norm[k * 2 + 1] = y;

    const float sample_x = (x + 1.0f) * 0.5f * wh_x;
    const float sample_y = (y + 1.0f) * 0.5f * wh_y;
    scores_out[k] =
            BilinearSampleScoreDevice(score_map, h, w, sample_y, sample_x);
}

__device__ float SeluDevice(float x) {
    return x > 0.0f ? kSeluScale * x
                    : kSeluScale * kSeluAlpha * (expf(x) - 1.0f);
}

__device__ float BilinearSampleFeatureDevice(const float *feature,
                                             int32_t dim,
                                             int32_t h,
                                             int32_t w,
                                             int32_t c,
                                             float py,
                                             float px) {
    if (!IsValidSampleCoord(py, px, h, w)) {
        return 0.0f;
    }
    const int32_t y0 = static_cast<int32_t>(floorf(py));
    const int32_t x0 = static_cast<int32_t>(floorf(px));
    const int32_t y1 = min(y0 + 1, h - 1);
    const int32_t x1 = min(x0 + 1, w - 1);
    const float ly = py - static_cast<float>(y0);
    const float lx = px - static_cast<float>(x0);
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;
    auto read_whcn = [&](int32_t yy, int32_t xx) -> float {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) {
            return 0.0f;
        }
        return feature[WhcnIndex(xx, yy, c, h, w)];
    };
    return hy * hx * read_whcn(y0, x0) + hy * lx * read_whcn(y0, x1) +
           ly * hx * read_whcn(y1, x0) + ly * lx * read_whcn(y1, x1);
}

__global__ void SddhKernel(const float *feature_map,
                           int32_t dim,
                           int32_t h,
                           int32_t w,
                           const float *keypoints_norm,
                           int32_t count,
                           int32_t kernel_size,
                           int32_t n_pos,
                           const float *offset_0_w,
                           const float *offset_0_b,
                           const float *offset_2_w,
                           const float *offset_2_b,
                           const float *sf_conv_w,
                           const float *agg_weights,
                           float *workspace,
                           float *descriptors) {
    const int32_t k = static_cast<int32_t>(blockIdx.x);
    if (k >= count) {
        return;
    }

    const size_t patch_size = static_cast<size_t>(dim) *
                              static_cast<size_t>(kernel_size) *
                              static_cast<size_t>(kernel_size);
    const size_t stride = patch_size + 64 + static_cast<size_t>(dim) * 3;
    float *base = workspace + static_cast<size_t>(k) * stride;
    float *patch = base;
    float *offset_raw = patch + patch_size;
    float *offset_final = offset_raw + 32;
    float *sampled = offset_final + 32;
    float *transformed = sampled + dim;
    float *desc = transformed + dim;

    const float wh_x = fmaxf(static_cast<float>(w - 1), 1.0f);
    const float wh_y = fmaxf(static_cast<float>(h - 1), 1.0f);
    const float max_offset = static_cast<float>(max(h, w)) / 4.0f;
    const int32_t pad = kernel_size / 2;

    const float x_norm = keypoints_norm[k * 2 + 0];
    const float y_norm = keypoints_norm[k * 2 + 1];
    if (!isfinite(x_norm) || !isfinite(y_norm)) {
        for (int32_t c = 0; c < dim; ++c) {
            descriptors[static_cast<size_t>(k) * dim + c] = 0.0f;
        }
        return;
    }

    const float x_wh = fminf(
            fmaxf((x_norm / 2.0f + 0.5f) * static_cast<float>(w - 1), 0.0f),
            static_cast<float>(w - 1));
    const float y_wh = fminf(
            fmaxf((y_norm / 2.0f + 0.5f) * static_cast<float>(h - 1), 0.0f),
            static_cast<float>(h - 1));

    if (kernel_size > 1) {
        const int32_t x0 =
                min(max(static_cast<int32_t>(lroundf(x_wh)) - pad, 0),
                    max(w - kernel_size, 0));
        const int32_t y0 =
                min(max(static_cast<int32_t>(lroundf(y_wh)) - pad, 0),
                    max(h - kernel_size, 0));
        for (int32_t c = 0; c < dim; ++c) {
            for (int32_t ky = 0; ky < kernel_size; ++ky) {
                for (int32_t kx = 0; kx < kernel_size; ++kx) {
                    patch[c * kernel_size * kernel_size + ky * kernel_size +
                          kx] =
                            feature_map[WhcnIndex(x0 + kx, y0 + ky, c, h, w)];
                }
            }
        }
    } else {
        const int32_t xi =
                min(max(static_cast<int32_t>(lroundf(x_wh)), 0), w - 1);
        const int32_t yi =
                min(max(static_cast<int32_t>(lroundf(y_wh)), 0), h - 1);
        for (int32_t c = 0; c < dim; ++c) {
            patch[c] = feature_map[WhcnIndex(xi, yi, c, h, w)];
        }
    }

    for (int32_t oc = 0; oc < 32; ++oc) {
        float sum = offset_0_b[oc];
        for (int32_t ic = 0; ic < dim; ++ic) {
            for (int32_t ky = 0; ky < kernel_size; ++ky) {
                for (int32_t kx = 0; kx < kernel_size; ++kx) {
                    const size_t widx = static_cast<size_t>(oc) * dim *
                                                kernel_size * kernel_size +
                                        static_cast<size_t>(ic) * kernel_size *
                                                kernel_size +
                                        ky * kernel_size + kx;
                    sum += patch[ic * kernel_size * kernel_size +
                                 ky * kernel_size + kx] *
                           offset_0_w[widx];
                }
            }
        }
        offset_raw[oc] = SeluDevice(sum);
    }

    for (int32_t oc = 0; oc < 32; ++oc) {
        float sum = offset_2_b[oc];
        for (int32_t ic = 0; ic < 32; ++ic) {
            sum += offset_raw[ic] *
                   offset_2_w[static_cast<size_t>(oc) * 32 + ic];
        }
        offset_final[oc] = sum;
    }

    for (int32_t c = 0; c < dim; ++c) {
        desc[c] = 0.0f;
    }

    for (int32_t p = 0; p < n_pos; ++p) {
        const float raw_off_x = offset_final[p];
        const float raw_off_y = offset_final[n_pos + p];
        if (!isfinite(raw_off_x) || !isfinite(raw_off_y)) {
            continue;
        }
        const float off_x = fminf(fmaxf(raw_off_x, -max_offset), max_offset);
        const float off_y = fminf(fmaxf(raw_off_y, -max_offset), max_offset);
        const float sample_x = (x_wh + off_x) / wh_x * 2.0f - 1.0f;
        const float sample_y = (y_wh + off_y) / wh_y * 2.0f - 1.0f;
        if (!isfinite(sample_x) || !isfinite(sample_y)) {
            continue;
        }
        const float px = (sample_x + 1.0f) * 0.5f * wh_x;
        const float py = (sample_y + 1.0f) * 0.5f * wh_y;

        for (int32_t c = 0; c < dim; ++c) {
            sampled[c] = BilinearSampleFeatureDevice(feature_map, dim, h, w, c,
                                                     py, px);
        }

        for (int32_t c = 0; c < dim; ++c) {
            float value = 0.0f;
            for (int32_t ic = 0; ic < dim; ++ic) {
                value += sampled[ic] *
                         sf_conv_w[static_cast<size_t>(c) * dim + ic];
            }
            transformed[c] = SeluDevice(value);
        }

        for (int32_t c = 0; c < dim; ++c) {
            for (int32_t ic = 0; ic < dim; ++ic) {
                desc[c] += transformed[ic] *
                           agg_weights[static_cast<size_t>(p) * dim * dim +
                                       static_cast<size_t>(ic) * dim + c];
            }
        }
    }

    float norm = 0.0f;
    for (int32_t c = 0; c < dim; ++c) {
        norm += desc[c] * desc[c];
    }
    norm = sqrtf(fmaxf(norm, 1e-12f));
    for (int32_t c = 0; c < dim; ++c) {
        descriptors[static_cast<size_t>(k) * dim + c] = desc[c] / norm;
    }
}

bool LaunchMaxPool(
        const float *input, float *output, int32_t h, int32_t w, int32_t pad) {
    dim3 block(16, 16, 1);
    dim3 grid((static_cast<unsigned>(w) + block.x - 1) / block.x,
              static_cast<unsigned>(h), 1);
    MaxPoolKernel<<<grid, block>>>(input, output, h, w, pad);
    return cudaGetLastError() == cudaSuccess;
}

bool SimpleNmsGpu(const float *scores,
                  float *nms,
                  int32_t h,
                  int32_t w,
                  int32_t radius,
                  float *tmp_a,
                  float *tmp_b,
                  float *tmp_c) {
    const int32_t count = h * w;
    const int32_t threads = 256;
    const int32_t blocks = (count + threads - 1) / threads;

    if (!LaunchMaxPool(scores, tmp_a, h, w, radius)) {
        return false;
    }
    CompareEqualMaskKernel<<<blocks, threads>>>(scores, tmp_a, tmp_b, count);
    cudaMemcpy(nms, scores, static_cast<size_t>(count) * sizeof(float),
               cudaMemcpyDeviceToDevice);

    for (int iter = 0; iter < 2; ++iter) {
        if (!LaunchMaxPool(tmp_b, tmp_a, h, w, radius)) {
            return false;
        }
        ThresholdPositiveKernel<<<blocks, threads>>>(tmp_a, tmp_c, count);
        SuppressScoresKernel<<<blocks, threads>>>(tmp_c, nms, count);
        if (!LaunchMaxPool(nms, tmp_a, h, w, radius)) {
            return false;
        }
        UpdateMaxMaskKernel<<<blocks, threads>>>(nms, tmp_a, tmp_c, tmp_b,
                                                 count);
    }
    ApplyMaxMaskKernel<<<blocks, threads>>>(scores, tmp_b, nms, count);
    return cudaGetLastError() == cudaSuccess;
}

__global__ void AddInPlaceKernel(float *dst, const float *src, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    dst[i] += src[i];
}

bool AlikedCudaAddInPlace(ggml_backend_t backend,
                          float *dst,
                          const float *src,
                          size_t count) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    AddInPlaceKernel<<<blocks, threads>>>(dst, src, count);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaApplySelu(ggml_backend_t backend, float *data, size_t count) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    SeluKernel<<<blocks, threads>>>(data, count);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaClampInPlace(ggml_backend_t backend,
                            float *data,
                            size_t count,
                            float min_value,
                            float max_value) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    ClampKernel<<<blocks, threads>>>(data, count, min_value, max_value);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaSigmoidInPlace(ggml_backend_t backend,
                              float *data,
                              size_t count) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    SigmoidKernel<<<blocks, threads>>>(data, count);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaL2NormalizeChannels(
        ggml_backend_t backend, float *data, int32_t c, int32_t h, int32_t w) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    const int32_t spatial = h * w;
    const int threads = 256;
    const int blocks = (spatial + threads - 1) / threads;
    L2NormalizeChannelsKernel<<<blocks, threads>>>(data, c, h, w);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaAvgPool2d(ggml_backend_t backend,
                         const float *input,
                         int32_t ic,
                         int32_t ih,
                         int32_t iw,
                         int32_t kh,
                         int32_t kw,
                         int32_t stride,
                         float *output,
                         int32_t oh,
                         int32_t ow) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    dim3 block(16, 16, 1);
    dim3 grid((static_cast<unsigned>(ow) + block.x - 1) / block.x,
              static_cast<unsigned>(oh), static_cast<unsigned>(ic));
    AvgPoolKernel<<<grid, block>>>(input, output, ic, ih, iw, kh, kw, stride,
                                   oh, ow);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaUpsampleBilinear(ggml_backend_t backend,
                                const float *input,
                                int32_t ic,
                                int32_t ih,
                                int32_t iw,
                                int32_t out_h,
                                int32_t out_w,
                                float *output) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    dim3 block(16, 16, 1);
    dim3 grid((static_cast<unsigned>(out_w) + block.x - 1) / block.x,
              static_cast<unsigned>(out_h), static_cast<unsigned>(ic));
    UpsampleBilinearKernel<<<grid, block>>>(input, output, ic, ih, iw, out_h,
                                            out_w);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaCropWhcn(ggml_backend_t backend,
                        const float *input,
                        int32_t ic,
                        int32_t padded_h,
                        int32_t padded_w,
                        int32_t pad_top,
                        int32_t pad_left,
                        int32_t out_h,
                        int32_t out_w,
                        float *output) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }
    dim3 block(16, 16, 1);
    dim3 grid((static_cast<unsigned>(out_w) + block.x - 1) / block.x,
              static_cast<unsigned>(out_h), static_cast<unsigned>(ic));
    CropWhcnKernel<<<grid, block>>>(input, output, ic, padded_h, padded_w,
                                    pad_top, pad_left, out_h, out_w);
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaRunDkd(ggml_backend_t backend,
                      const float *score_map,
                      int32_t h,
                      int32_t w,
                      int32_t radius,
                      int32_t top_k,
                      float scores_th,
                      int32_t n_limit,
                      float *keypoints_norm,
                      float *scores,
                      int32_t *out_count,
                      AlikedDkdScratch *scratch) {
    if (!aicore::common::ggml_backend_is_cuda(backend)) {
        return false;
    }

    const int32_t count = h * w;
    float *nms = nullptr;
    float *tmp_a = nullptr;
    float *tmp_b = nullptr;
    float *tmp_c = nullptr;
    if (scratch != nullptr) {
        scratch->Ensure(count);
        nms = scratch->impl->nms;
        tmp_a = scratch->impl->tmp_a;
        tmp_b = scratch->impl->tmp_b;
        tmp_c = scratch->impl->tmp_c;
    } else {
        cudaMalloc(&nms, static_cast<size_t>(count) * sizeof(float));
        cudaMalloc(&tmp_a, static_cast<size_t>(count) * sizeof(float));
        cudaMalloc(&tmp_b, static_cast<size_t>(count) * sizeof(float));
        cudaMalloc(&tmp_c, static_cast<size_t>(count) * sizeof(float));
    }

    if (!SimpleNmsGpu(score_map, nms, h, w, radius, tmp_a, tmp_b, tmp_c)) {
        if (scratch == nullptr) {
            cudaFree(nms);
            cudaFree(tmp_a);
            cudaFree(tmp_b);
            cudaFree(tmp_c);
        }
        return false;
    }

    dim3 zblock(16, 16, 1);
    dim3 zgrid((static_cast<unsigned>(w) + zblock.x - 1) / zblock.x,
               static_cast<unsigned>(h), 1);
    ZeroBorderKernel<<<zgrid, zblock>>>(nms, h, w, radius);

    std::vector<int32_t> indices;
    if (top_k > 0) {
        std::vector<float> nms_host(static_cast<size_t>(count));
        cudaMemcpy(nms_host.data(), nms,
                   static_cast<size_t>(count) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        std::vector<std::pair<float, int32_t>> scored;
        scored.reserve(static_cast<size_t>(count));
        for (int32_t i = 0; i < count; ++i) {
            if (nms_host[static_cast<size_t>(i)] > scores_th) {
                scored.emplace_back(nms_host[static_cast<size_t>(i)], i);
            }
        }
        const int32_t keep =
                std::min(top_k, static_cast<int32_t>(scored.size()));
        std::partial_sort(
                scored.begin(), scored.begin() + keep, scored.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });
        indices.reserve(static_cast<size_t>(keep));
        for (int32_t i = 0; i < keep; ++i) {
            indices.push_back(scored[static_cast<size_t>(i)].second);
        }
    } else {
        std::vector<float> nms_host(static_cast<size_t>(count));
        cudaMemcpy(nms_host.data(), nms,
                   static_cast<size_t>(count) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        float threshold = scores_th;
        if (scores_th > 0.0f) {
            bool any = false;
            for (float value : nms_host) {
                if (value > threshold) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                threshold = 0.0f;
                for (float value : nms_host) {
                    threshold += value;
                }
                threshold /= static_cast<float>(count);
            }
        } else {
            threshold = 0.0f;
            for (float value : nms_host) {
                threshold += value;
            }
            threshold /= static_cast<float>(count);
        }
        std::vector<std::pair<float, int32_t>> scored;
        std::vector<float> score_host(static_cast<size_t>(count));
        cudaMemcpy(score_host.data(), score_map,
                   static_cast<size_t>(count) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (int32_t i = 0; i < count; ++i) {
            if (nms_host[static_cast<size_t>(i)] > threshold) {
                scored.emplace_back(score_host[static_cast<size_t>(i)], i);
            }
        }
        std::sort(
                scored.begin(), scored.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });
        if (static_cast<int32_t>(scored.size()) > n_limit) {
            scored.resize(static_cast<size_t>(n_limit));
        }
        indices.reserve(scored.size());
        for (const auto &entry : scored) {
            indices.push_back(entry.second);
        }
    }

    const int32_t kpt_count = static_cast<int32_t>(indices.size());
    *out_count = kpt_count;
    if (kpt_count == 0) {
        if (scratch == nullptr) {
            cudaFree(nms);
            cudaFree(tmp_a);
            cudaFree(tmp_b);
            cudaFree(tmp_c);
        }
        return true;
    }

    for (int32_t index : indices) {
        if (index < 0 || index >= count) {
            if (scratch == nullptr) {
                cudaFree(nms);
                cudaFree(tmp_a);
                cudaFree(tmp_b);
                cudaFree(tmp_c);
            }
            return false;
        }
    }

    int32_t *indices_dev = nullptr;
    cudaMalloc(&indices_dev, static_cast<size_t>(kpt_count) * sizeof(int32_t));
    cudaMemcpy(indices_dev, indices.data(),
               static_cast<size_t>(kpt_count) * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    RefineKeypointsKernel<<<kpt_count, 1>>>(score_map, indices_dev, kpt_count,
                                            h, w, radius, keypoints_norm,
                                            scores);
    cudaFree(indices_dev);
    if (scratch == nullptr) {
        cudaFree(nms);
        cudaFree(tmp_a);
        cudaFree(tmp_b);
        cudaFree(tmp_c);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool AlikedCudaRunSddh(ggml_backend_t backend,
                       const float *feature_map,
                       int32_t dim,
                       int32_t h,
                       int32_t w,
                       const float *keypoints_norm,
                       int32_t count,
                       int32_t kernel_size,
                       int32_t n_pos,
                       const float *offset_0_w,
                       const float *offset_0_b,
                       const float *offset_2_w,
                       const float *offset_2_b,
                       const float *sf_conv_w,
                       const float *agg_weights,
                       float *descriptors,
                       AlikedSddhScratch *scratch) {
    if (!aicore::common::ggml_backend_is_cuda(backend) || count <= 0) {
        return false;
    }

    float *workspace = nullptr;
    bool owned_workspace = false;
    if (scratch != nullptr) {
        scratch->Ensure(count, dim, kernel_size);
        workspace = scratch->impl->workspace;
    } else {
        const size_t floats = SddhWorkspaceFloats(count, dim, kernel_size);
        if (cudaMalloc(&workspace, floats * sizeof(float)) != cudaSuccess) {
            return false;
        }
        owned_workspace = true;
    }

    SddhKernel<<<count, 1>>>(feature_map, dim, h, w, keypoints_norm, count,
                             kernel_size, n_pos, offset_0_w, offset_0_b,
                             offset_2_w, offset_2_b, sf_conv_w, agg_weights,
                             workspace, descriptors);
    if (owned_workspace) {
        cudaFree(workspace);
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace lightglue::aliked_internal
