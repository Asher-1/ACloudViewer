// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/gpu_postprocess.hpp"

#include "tasks/aliked/gpu_pipeline_cache.hpp"
#include "tasks/aliked/gpu_sync.hpp"

#if defined(AICORE_CUDA_ALIKED)
#include <cuda_runtime.h>

#include "common/ggml_backend_util.hpp"
#include "tasks/aliked/cuda/aliked_cuda.hpp"

namespace lightglue::aliked_internal {
namespace {

float *DevPtr(const GpuTensor &tensor) {
    return reinterpret_cast<float *>(tensor.tensor->data);
}

bool UploadWeights(internal::Backend *backend,
                   const std::vector<float> &weights,
                   float **device_ptr) {
    if (!aicore::common::ggml_backend_is_cuda(backend->handle)) {
        return false;
    }
    cudaMalloc(device_ptr, weights.size() * sizeof(float));
    cudaMemcpy(*device_ptr, weights.data(), weights.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    return cudaGetLastError() == cudaSuccess;
}

void FreeWeights(float *ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

}  // namespace

bool RunDkdGpu(const GpuTensor &score_map,
               int32_t h,
               int32_t w,
               const DkdOptions &options,
               internal::Backend *backend,
               GpuKeypointResult *output,
               std::string *error,
               AlikedDkdScratch *scratch) {
    if (backend == nullptr || !backend->IsGpu()) {
        if (error) {
            *error = "RunDkdGpu requires a GPU backend";
        }
        return false;
    }

    const int32_t max_kpts =
            options.top_k > 0 ? options.top_k
                              : (options.n_limit > 0 ? options.n_limit : 20000);
    if (!GpuTensor::Allocate(backend, max_kpts * 2, 1, 1,
                             &output->keypoints_norm, error)) {
        return false;
    }
    if (!GpuTensor::Allocate(backend, max_kpts, 1, 1, &output->scores, error)) {
        return false;
    }

    int32_t count = 0;
    if (!AlikedCudaRunDkd(backend->handle, DevPtr(score_map), h, w,
                          options.radius, options.top_k, options.scores_th,
                          options.n_limit, DevPtr(output->keypoints_norm),
                          DevPtr(output->scores), &count, scratch)) {
        if (error) {
            *error = "CUDA DKD failed";
        }
        return false;
    }
    output->count = count;
    SyncGpuPipeline(backend);
    return true;
}

bool RunSddhGpu(const GpuTensor &feature_map,
                int32_t dim,
                int32_t h,
                int32_t w,
                const GpuTensor &keypoints_norm,
                int32_t count,
                int32_t kernel_size,
                int32_t n_pos,
                const std::vector<float> &offset_0_w,
                const std::vector<float> &offset_0_b,
                const std::vector<float> &offset_2_w,
                const std::vector<float> &offset_2_b,
                const std::vector<float> &sf_conv_w,
                const std::vector<float> &agg_weights,
                internal::Backend *backend,
                GpuTensor *descriptors,
                std::string *error,
                GpuPipelineCache *cache) {
    if (backend == nullptr || !backend->IsGpu() || count <= 0) {
        if (error) {
            *error = "RunSddhGpu requires a GPU backend and keypoints";
        }
        return false;
    }

    const float *d_offset_0_w = nullptr;
    const float *d_offset_0_b = nullptr;
    const float *d_offset_2_w = nullptr;
    const float *d_offset_2_b = nullptr;
    const float *d_sf_conv_w = nullptr;
    const float *d_agg_weights = nullptr;

    float *temp_offset_0_w = nullptr;
    float *temp_offset_0_b = nullptr;
    float *temp_offset_2_w = nullptr;
    float *temp_offset_2_b = nullptr;
    float *temp_sf_conv_w = nullptr;
    float *temp_agg_weights = nullptr;

    const bool use_cache =
            cache != nullptr &&
            cache->EnsureSddhWeights(offset_0_w, offset_0_b, offset_2_w,
                                     offset_2_b, sf_conv_w, agg_weights, error);
    if (use_cache) {
        d_offset_0_w = cache->SddhOffset0WPtr();
        d_offset_0_b = cache->SddhOffset0BPtr();
        d_offset_2_w = cache->SddhOffset2WPtr();
        d_offset_2_b = cache->SddhOffset2BPtr();
        d_sf_conv_w = cache->SddhSfConvWPtr();
        d_agg_weights = cache->SddhAggWeightsPtr();
    } else if (!UploadWeights(backend, offset_0_w, &temp_offset_0_w) ||
               !UploadWeights(backend, offset_0_b, &temp_offset_0_b) ||
               !UploadWeights(backend, offset_2_w, &temp_offset_2_w) ||
               !UploadWeights(backend, offset_2_b, &temp_offset_2_b) ||
               !UploadWeights(backend, sf_conv_w, &temp_sf_conv_w) ||
               !UploadWeights(backend, agg_weights, &temp_agg_weights)) {
        FreeWeights(temp_offset_0_w);
        FreeWeights(temp_offset_0_b);
        FreeWeights(temp_offset_2_w);
        FreeWeights(temp_offset_2_b);
        FreeWeights(temp_sf_conv_w);
        FreeWeights(temp_agg_weights);
        if (error) {
            *error = "failed to upload SDDH weights";
        }
        return false;
    } else {
        d_offset_0_w = temp_offset_0_w;
        d_offset_0_b = temp_offset_0_b;
        d_offset_2_w = temp_offset_2_w;
        d_offset_2_b = temp_offset_2_b;
        d_sf_conv_w = temp_sf_conv_w;
        d_agg_weights = temp_agg_weights;
    }

    if (!GpuTensor::Allocate(backend, count * dim, 1, 1, descriptors, error)) {
        FreeWeights(temp_offset_0_w);
        FreeWeights(temp_offset_0_b);
        FreeWeights(temp_offset_2_w);
        FreeWeights(temp_offset_2_b);
        FreeWeights(temp_sf_conv_w);
        FreeWeights(temp_agg_weights);
        return false;
    }

    SyncGpuPipeline(backend);
    AlikedSddhScratch *sddh_scratch =
            cache != nullptr ? cache->sddh_scratch() : nullptr;
    const bool ok = AlikedCudaRunSddh(
            backend->handle, DevPtr(feature_map), dim, h, w,
            DevPtr(keypoints_norm), count, kernel_size, n_pos, d_offset_0_w,
            d_offset_0_b, d_offset_2_w, d_offset_2_b, d_sf_conv_w,
            d_agg_weights, DevPtr(*descriptors), sddh_scratch);

    FreeWeights(temp_offset_0_w);
    FreeWeights(temp_offset_0_b);
    FreeWeights(temp_offset_2_w);
    FreeWeights(temp_offset_2_b);
    FreeWeights(temp_sf_conv_w);
    FreeWeights(temp_agg_weights);

    if (!ok && error) {
        *error = "CUDA SDDH failed";
        return false;
    }
    SyncGpuPipeline(backend);
    return true;
}

}  // namespace lightglue::aliked_internal

#else

namespace lightglue::aliked_internal {

bool RunDkdGpu(const GpuTensor &,
               int32_t,
               int32_t,
               const DkdOptions &,
               internal::Backend *,
               GpuKeypointResult *,
               std::string *,
               void *) {
    return false;
}

bool RunSddhGpu(const GpuTensor &,
                int32_t,
                int32_t,
                int32_t,
                const GpuTensor &,
                int32_t,
                int32_t,
                int32_t,
                const std::vector<float> &,
                const std::vector<float> &,
                const std::vector<float> &,
                const std::vector<float> &,
                const std::vector<float> &,
                const std::vector<float> &,
                internal::Backend *,
                GpuTensor *,
                std::string *,
                GpuPipelineCache *) {
    return false;
}

}  // namespace lightglue::aliked_internal

#endif
