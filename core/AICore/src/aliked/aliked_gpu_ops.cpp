// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aliked_gpu_ops.hpp"

#include "deform_conv.hpp"
#include "ggml_gpu_ops.hpp"
#include "postprocess.hpp"

#if defined(LIGHTGLUE_HAS_CUDA)
#include <cuda_runtime.h>

#include "aliked_cuda.hpp"
#endif

#if defined(LIGHTGLUE_HAS_VULKAN)
#include <ggml-vulkan-aliked.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace lightglue::aliked_internal {
namespace {

struct DcnParityEntry {
    std::string name;
    int32_t c = 0;
    int32_t h = 0;
    int32_t w = 0;
    std::vector<float> cpu;
    std::vector<float> vulkan;
};

std::vector<DcnParityEntry> &DcnParityEntries() {
    static std::vector<DcnParityEntry> entries;
    return entries;
}

void RecordDcnParity(internal::Backend *backend,
                     const std::string &name,
                     const GpuTensor &cpu_out,
                     const GpuTensor &vulkan_out,
                     std::string *error) {
    DcnParityEntry entry;
    entry.name = name;
    entry.c = vulkan_out.c;
    entry.h = vulkan_out.h;
    entry.w = vulkan_out.w;
    if (!cpu_out.DownloadNchw(backend, &entry.cpu, entry.c, entry.h, entry.w,
                              error) ||
        !vulkan_out.DownloadNchw(backend, &entry.vulkan, entry.c, entry.h,
                                 entry.w, error)) {
        return;
    }
    DcnParityEntries().push_back(std::move(entry));
}

void RecordDcnNchwParity(const std::string &name,
                         const std::vector<float> &cpu_nchw,
                         const ggml_tensor *gpu_nchw,
                         int32_t c,
                         int32_t h,
                         int32_t w) {
    DcnParityEntry entry;
    entry.name = name;
    entry.c = c;
    entry.h = h;
    entry.w = w;
    entry.cpu = cpu_nchw;
    entry.vulkan.resize(static_cast<size_t>(c) * h * w);
    ggml_backend_tensor_get(gpu_nchw, entry.vulkan.data(), 0,
                            entry.vulkan.size() * sizeof(float));
    DcnParityEntries().push_back(std::move(entry));
}

void WriteTensorRecord(std::ofstream &out,
                       const std::string &name,
                       const std::vector<float> &nchw,
                       int32_t c,
                       int32_t h,
                       int32_t w) {
    const uint32_t name_len = static_cast<uint32_t>(name.size());
    out.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
    out.write(name.data(), static_cast<std::streamsize>(name.size()));
    out.write(reinterpret_cast<const char *>(&c), sizeof(c));
    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    out.write(reinterpret_cast<const char *>(&w), sizeof(w));
    out.write(reinterpret_cast<const char *>(nchw.data()),
              static_cast<std::streamsize>(nchw.size() * sizeof(float)));
}

}  // namespace

void ClearAlikedDcnParityEntries() { DcnParityEntries().clear(); }

bool WriteAlikedDcnParityDump(const std::string &path, std::string *error) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (error) {
            *error = "failed to open DCN dump: " + path;
        }
        return false;
    }
    out.write("DCNDMP01", 8);
    const uint32_t count = static_cast<uint32_t>(DcnParityEntries().size() * 3);
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const DcnParityEntry &entry : DcnParityEntries()) {
        WriteTensorRecord(out, entry.name + ".cpu", entry.cpu, entry.c, entry.h,
                          entry.w);
        WriteTensorRecord(out, entry.name + ".vulkan", entry.vulkan, entry.c,
                          entry.h, entry.w);
        std::vector<float> diff(entry.cpu.size());
        for (size_t i = 0; i < diff.size(); ++i) {
            diff[i] = entry.vulkan[i] - entry.cpu[i];
        }
        WriteTensorRecord(out, entry.name + ".diff", diff, entry.c, entry.h,
                          entry.w);
    }
    if (!out) {
        if (error) {
            *error = "failed to write DCN dump: " + path;
        }
        return false;
    }
    return true;
}

namespace {

float *DevPtr(const GpuTensor &tensor) {
    return reinterpret_cast<float *>(tensor.tensor->data);
}

void SyncGpuPipeline(internal::Backend *backend) {
    if (backend != nullptr && backend->handle != nullptr) {
        ggml_backend_synchronize(backend->handle);
#if defined(LIGHTGLUE_HAS_CUDA)
        if (backend->IsCuda()) {
            cudaDeviceSynchronize();
        }
#endif
    }
}

bool UseCudaCustomKernels(internal::Backend *backend) {
#if defined(LIGHTGLUE_HAS_CUDA)
    return backend != nullptr && backend->IsCuda();
#else
    (void)backend;
    return false;
#endif
}

bool ConvGpu(GgmlConvRunner *runner,
             const GpuTensor &input,
             const std::vector<float> &weight,
             int32_t oc,
             int32_t kh,
             int32_t kw,
             const std::vector<float> *bias,
             int32_t pad,
             int32_t stride,
             GpuTensor *output,
             const char *cache_key,
             std::string *error) {
    std::vector<float> ones(static_cast<size_t>(oc), 1.0f);
    std::vector<float> zeros(static_cast<size_t>(oc), 0.0f);
    const FusedConv2d fused = FuseConvBn(weight, oc, input.c, kh, kw, bias,
                                         ones, zeros, zeros, ones);
    return runner->RunDevice(fused, input, output, pad, stride, error,
                             cache_key);
}

bool DcnConvBnCpuBridge(GpuPipelineCache *cache,
                        const GpuTensor &input,
                        const std::vector<float> &offset_w,
                        const std::vector<float> &offset_b,
                        const std::vector<float> &regular_w,
                        int32_t oc,
                        const std::vector<float> &gamma,
                        const std::vector<float> &beta,
                        const std::vector<float> &mean,
                        const std::vector<float> &var,
                        const std::string &cache_prefix,
                        GpuTensor *output,
                        std::string *error) {
    GpuTensor offset;
    if (!ConvGpu(cache->ggml()->runner(), input, offset_w, 18, 3, 3, &offset_b,
                 1, 1, &offset, (cache_prefix + ".offset").c_str(), error)) {
        return false;
    }

    const float max_offset =
            static_cast<float>(std::max(input.h, input.w)) / 4.0f;
    if (!RunClampGpu(cache->backend(), &offset, -max_offset, max_offset,
                     error)) {
        return false;
    }

    std::vector<float> input_nchw;
    std::vector<float> offset_nchw;
    if (!input.DownloadNchw(cache->backend(), &input_nchw, input.c, input.h,
                            input.w, error) ||
        !offset.DownloadNchw(cache->backend(), &offset_nchw, 18, input.h,
                             input.w, error)) {
        return false;
    }

    const FusedConv2dNchw fused = FuseConvBnNchw(
            regular_w, oc, input.c, 3, 3, nullptr, gamma, beta, mean, var);
    std::vector<float> out_nchw;
    int32_t oh = 0;
    int32_t ow = 0;
    DeformConv2d(input_nchw, input.c, input.h, input.w, offset_nchw, 1,
                 fused.kernel, oc, 3, 3, &fused.bias, 1, &out_nchw, &oh, &ow);

    if (!GpuTensor::Allocate(cache->backend(), ow, oh, oc, output, error)) {
        return false;
    }
    return output->UploadNchw(cache->backend(), out_nchw, oc, oh, ow, error);
}

#if defined(LIGHTGLUE_HAS_CUDA)
bool DcnConvBnCuda(GpuPipelineCache *cache,
                   const GpuTensor &input,
                   const std::vector<float> &offset_w,
                   const std::vector<float> &offset_b,
                   const std::vector<float> &regular_w,
                   int32_t oc,
                   const std::vector<float> &gamma,
                   const std::vector<float> &beta,
                   const std::vector<float> &mean,
                   const std::vector<float> &var,
                   const std::string &cache_prefix,
                   GpuTensor *output,
                   std::string *error) {
    GpuTensor offset;
    if (!ConvGpu(cache->ggml()->runner(), input, offset_w, 18, 3, 3, &offset_b,
                 1, 1, &offset, (cache_prefix + ".offset").c_str(), error)) {
        return false;
    }

    SyncGpuPipeline(cache->backend());
    const float max_offset =
            static_cast<float>(std::max(input.h, input.w)) / 4.0f;
    if (!AlikedCudaClampInPlace(cache->backend()->handle, DevPtr(offset),
                                offset.ElementCount(), -max_offset,
                                max_offset)) {
        if (error) {
            *error = "offset clamp CUDA failed";
        }
        return false;
    }

    const FusedConv2dNchw fused = FuseConvBnNchw(
            regular_w, oc, input.c, 3, 3, nullptr, gamma, beta, mean, var);
    const std::string weight_key = cache_prefix + ".deform";
    if (!cache->EnsureDcnWeight(weight_key, fused, error)) {
        return false;
    }
    if (!cache->EnsureDcnWorkspace(input.w, input.h, input.c, oc, error)) {
        return false;
    }

    if (!AlikedCudaWhcnToNchw(cache->backend()->handle, DevPtr(input),
                              cache->DcnNchwInPtr(), input.c, input.h,
                              input.w)) {
        if (error) {
            *error = "WHCN->NCHW CUDA failed";
        }
        return false;
    }
    if (!AlikedCudaWhcnToNchw(cache->backend()->handle, DevPtr(offset),
                              cache->DcnNchwOffsetPtr(), 18, input.h,
                              input.w)) {
        if (error) {
            *error = "offset WHCN->NCHW CUDA failed";
        }
        return false;
    }
    if (!AlikedCudaDeformConv2d(
                cache->backend()->handle, cache->DcnNchwInPtr(), input.c,
                input.h, input.w, cache->DcnNchwOffsetPtr(),
                cache->DcnWeightPtr(weight_key), cache->DcnBiasPtr(weight_key),
                oc, 3, 3, 1, cache->DcnNchwOutPtr())) {
        if (error) {
            *error = "deform conv CUDA failed";
        }
        return false;
    }

    if (!GpuTensor::Allocate(cache->backend(), input.w, input.h, oc, output,
                             error)) {
        return false;
    }
    if (!AlikedCudaNchwToWhcn(cache->backend()->handle, cache->DcnNchwOutPtr(),
                              DevPtr(*output), oc, input.h, input.w)) {
        if (error) {
            *error = "NCHW->WHCN CUDA failed";
        }
        return false;
    }
    return true;
}
#endif

#if defined(LIGHTGLUE_HAS_VULKAN)
bool UseVulkanCompute(internal::Backend *backend) {
    if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_COMPUTE") == nullptr) {
        return false;
    }
    return backend != nullptr && backend->IsVulkan() &&
           ggml_vulkan_aliked_available(backend->handle);
}

bool DcnConvBnVulkan(GpuPipelineCache *cache,
                     const GpuTensor &input,
                     const std::vector<float> &offset_w,
                     const std::vector<float> &offset_b,
                     const std::vector<float> &regular_w,
                     int32_t oc,
                     const std::vector<float> &gamma,
                     const std::vector<float> &beta,
                     const std::vector<float> &mean,
                     const std::vector<float> &var,
                     const std::string &cache_prefix,
                     GpuTensor *output,
                     std::string *error) {
    if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
        std::cerr << "dcn_vk begin " << cache_prefix << " ic=" << input.c
                  << " oc=" << oc << "\n";
    }
    SyncGpuPipeline(cache->backend());
    GpuTensor offset;
    if (!ConvGpu(cache->ggml()->runner(), input, offset_w, 18, 3, 3, &offset_b,
                 1, 1, &offset, (cache_prefix + ".offset").c_str(), error)) {
        return false;
    }

    if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
        std::cerr << "dcn_vk offset conv done " << cache_prefix << "\n";
    }

    SyncGpuPipeline(cache->backend());
    const float max_offset =
            static_cast<float>(std::max(input.h, input.w)) / 4.0f;
    if (!ggml_vulkan_aliked_clamp_inplace(cache->backend()->handle,
                                          offset.tensor, offset.ElementCount(),
                                          -max_offset, max_offset)) {
        if (error) {
            *error = "offset clamp Vulkan failed";
        }
        return false;
    }
    if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
        std::cerr << "dcn_vk clamp done " << cache_prefix << "\n";
    }

    const FusedConv2dNchw fused = FuseConvBnNchw(
            regular_w, oc, input.c, 3, 3, nullptr, gamma, beta, mean, var);
    const std::string weight_key = cache_prefix + ".deform";
    if (!cache->EnsureDcnWeight(weight_key, fused, error)) {
        return false;
    }
    if (!cache->EnsureDcnWorkspace(input.w, input.h, input.c, oc, error)) {
        return false;
    }
    internal::Backend *backend = cache->backend();
    const GpuPipelineCache::CachedDcnWeights *weights =
            cache->FindDcnWeight(weight_key);
    if (weights == nullptr) {
        if (error) {
            *error = "DCN weight cache miss";
        }
        return false;
    }
    if (!ggml_vulkan_aliked_whcn_to_nchw(backend->handle, input.tensor,
                                         cache->DcnNchwInTensor(), input.c,
                                         input.h, input.w)) {
        if (error) {
            *error = "DCN input WHCN->NCHW failed";
        }
        return false;
    }
    if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
        std::cerr << "dcn_vk in layout done " << cache_prefix << "\n";
    }
    if (std::getenv("LIGHTGLUE_ALIKED_DCN_DEBUG") != nullptr) {
        std::vector<float> input_nchw;
        if (input.DownloadNchw(backend, &input_nchw, input.c, input.h, input.w,
                               error)) {
            RecordDcnNchwParity(cache_prefix + ".input_nchw", input_nchw,
                                cache->DcnNchwInTensor(), input.c, input.h,
                                input.w);
        }
    }
    if (!ggml_vulkan_aliked_whcn_to_nchw(backend->handle, offset.tensor,
                                         cache->DcnNchwOffsetTensor(), 18,
                                         input.h, input.w)) {
        if (error) {
            *error = "DCN offset WHCN->NCHW failed";
        }
        return false;
    }
    if (std::getenv("LIGHTGLUE_ALIKED_DCN_DEBUG") != nullptr) {
        std::vector<float> offset_nchw;
        if (offset.DownloadNchw(backend, &offset_nchw, 18, input.h, input.w,
                                error)) {
            RecordDcnNchwParity(cache_prefix + ".offset_nchw", offset_nchw,
                                cache->DcnNchwOffsetTensor(), 18, input.h,
                                input.w);
        }
    }
    SyncGpuPipeline(backend);
    if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
        std::cerr << "dcn_vk pre deform " << cache_prefix << "\n";
    }
    if (!GpuTensor::Allocate(backend, input.w, input.h, oc, output, error)) {
        return false;
    }
    if (!ggml_vulkan_aliked_deform_conv2d(
                backend->handle, cache->DcnNchwInTensor(),
                cache->DcnNchwOffsetTensor(), weights->weight.tensor,
                weights->bias.tensor, cache->DcnNchwOutTensor(), input.c,
                input.h, input.w, oc, 3, 3, 1, 0)) {
        if (error) {
            *error = "deform conv Vulkan failed";
        }
        return false;
    }
    if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
        std::cerr << "dcn_vk deform done " << cache_prefix << "\n";
    }
    if (!ggml_vulkan_aliked_nchw_to_whcn(
                backend->handle, cache->DcnNchwOutTensor(), output->tensor, oc,
                input.h, input.w)) {
        if (error) {
            *error = "DCN output NCHW->WHCN failed";
        }
        return false;
    }
    SyncGpuPipeline(backend);
    return true;
}

bool RunDkdVulkan(const GpuTensor &score_map,
                  int32_t h,
                  int32_t w,
                  const DkdOptions &options,
                  internal::Backend *backend,
                  GpuKeypointResult *result,
                  std::string *error,
                  GpuPipelineCache *cache) {
    const int32_t max_kpts =
            options.top_k > 0 ? options.top_k
                              : (options.n_limit > 0 ? options.n_limit : 20000);
    if (!cache->EnsureVulkanDkdScratch(h, w, max_kpts, error)) {
        return false;
    }
    if (!GpuTensor::Allocate(backend, max_kpts * 2, 1, 1,
                             &result->keypoints_norm, error) ||
        !GpuTensor::Allocate(backend, max_kpts, 1, 1, &result->scores, error)) {
        return false;
    }

    AlikedVulkanDkdScratch *scratch = cache->vulkan_dkd_scratch();
    int32_t count = 0;
    if (!ggml_vulkan_aliked_run_dkd(
                backend->handle, score_map.tensor, h, w, options.radius,
                options.top_k, options.scores_th, options.n_limit,
                result->keypoints_norm.tensor, result->scores.tensor, &count,
                scratch->nms.tensor, scratch->tmp_a.tensor,
                scratch->tmp_b.tensor, scratch->tmp_c.tensor,
                scratch->block_keys.tensor, scratch->block_indices.tensor,
                scratch->indices_dev.tensor)) {
        if (error) {
            *error = "Vulkan DKD failed";
        }
        return false;
    }
    result->count = count;
    SyncGpuPipeline(backend);
    return true;
}

bool RunSddhVulkan(const GpuTensor &feature_map,
                   int32_t descriptor_dim,
                   int32_t fh,
                   int32_t fw,
                   const GpuTensor &keypoints_norm,
                   int32_t keypoint_count,
                   int32_t kernel_size,
                   int32_t n_pos,
                   GpuPipelineCache *cache,
                   internal::Backend *backend,
                   GpuTensor *descriptors,
                   std::string *error) {
    if (!cache->HasSddhWeights()) {
        if (error) {
            *error = "SDDH weights not loaded in cache";
        }
        return false;
    }
    if (!cache->EnsureVulkanSddhScratch(keypoint_count, descriptor_dim,
                                        kernel_size, fh, fw, error)) {
        return false;
    }
    if (!GpuTensor::Allocate(backend, keypoint_count * descriptor_dim, 1, 1,
                             descriptors, error)) {
        return false;
    }

    const GpuPipelineCache::CachedSddhWeights &w = cache->SddhWeightTensors();
    AlikedVulkanSddhScratch *scratch = cache->vulkan_sddh_scratch();
    SyncGpuPipeline(backend);
    if (!ggml_vulkan_aliked_whcn_to_nchw(backend->handle, feature_map.tensor,
                                         scratch->feature_nchw.tensor,
                                         descriptor_dim, fh, fw)) {
        if (error) {
            *error = "SDDH feature WHCN->NCHW failed";
        }
        return false;
    }
    SyncGpuPipeline(backend);
    const bool ok = ggml_vulkan_aliked_run_sddh(
            backend->handle, scratch->feature_nchw.tensor, descriptor_dim, fh,
            fw, keypoints_norm.tensor, keypoint_count, kernel_size, n_pos,
            w.offset_0_w.tensor, w.offset_0_b.tensor, w.offset_2_w.tensor,
            w.offset_2_b.tensor, w.sf_conv_w.tensor, w.agg_weights.tensor,
            scratch->workspace.tensor, descriptors->tensor);
    if (!ok && error) {
        *error = "Vulkan SDDH failed";
        return false;
    }
    SyncGpuPipeline(backend);
    return true;
}
#endif

}  // namespace

AlikedCustomOpBackend DetectCustomOpBackend(internal::Backend *backend) {
    if (backend == nullptr || !backend->IsGpu()) {
        return AlikedCustomOpBackend::kCpu;
    }
    if (UseCudaCustomKernels(backend)) {
        return AlikedCustomOpBackend::kCuda;
    }
#if defined(LIGHTGLUE_HAS_VULKAN)
    if (UseVulkanCompute(backend)) {
        return AlikedCustomOpBackend::kVulkanCompute;
    }
#endif
    if (backend->IsVulkan()) {
        return AlikedCustomOpBackend::kVulkanBridge;
    }
    return AlikedCustomOpBackend::kCpu;
}

bool DcnConvBnDispatch(GpuPipelineCache *cache,
                       const GpuTensor &input,
                       const std::vector<float> &offset_w,
                       const std::vector<float> &offset_b,
                       const std::vector<float> &regular_w,
                       int32_t oc,
                       const std::vector<float> &gamma,
                       const std::vector<float> &beta,
                       const std::vector<float> &mean,
                       const std::vector<float> &var,
                       const std::string &cache_prefix,
                       GpuTensor *output,
                       std::string *error) {
    const bool debug = std::getenv("LIGHTGLUE_ALIKED_DCN_DEBUG") != nullptr;
#if defined(LIGHTGLUE_HAS_VULKAN)
    if (DetectCustomOpBackend(cache->backend()) ==
                AlikedCustomOpBackend::kVulkanCompute &&
        debug) {
        GpuTensor cpu_out;
        if (!DcnConvBnCpuBridge(cache, input, offset_w, offset_b, regular_w, oc,
                                gamma, beta, mean, var,
                                cache_prefix + ".cpu_ref", &cpu_out, error)) {
            return false;
        }
        SyncGpuPipeline(cache->backend());
        if (!DcnConvBnVulkan(cache, input, offset_w, offset_b, regular_w, oc,
                             gamma, beta, mean, var, cache_prefix, output,
                             error)) {
            return false;
        }
        RecordDcnParity(cache->backend(), cache_prefix, cpu_out, *output,
                        error);
        return true;
    }
#endif
    switch (DetectCustomOpBackend(cache->backend())) {
#if defined(LIGHTGLUE_HAS_CUDA)
        case AlikedCustomOpBackend::kCuda:
            return DcnConvBnCuda(cache, input, offset_w, offset_b, regular_w,
                                 oc, gamma, beta, mean, var, cache_prefix,
                                 output, error);
#endif
#if defined(LIGHTGLUE_HAS_VULKAN)
        case AlikedCustomOpBackend::kVulkanCompute:
            return DcnConvBnVulkan(cache, input, offset_w, offset_b, regular_w,
                                   oc, gamma, beta, mean, var, cache_prefix,
                                   output, error);
#endif
        case AlikedCustomOpBackend::kVulkanBridge:
            return DcnConvBnCpuBridge(cache, input, offset_w, offset_b,
                                      regular_w, oc, gamma, beta, mean, var,
                                      cache_prefix, output, error);
        default:
            if (error) {
                *error = "DCN dispatch requires GPU backend";
            }
            return false;
    }
}

bool RunDkdDispatch(const GpuTensor &score_map,
                    int32_t h,
                    int32_t w,
                    const DkdOptions &options,
                    internal::Backend *backend,
                    GpuKeypointResult *result,
                    std::string *error,
                    GpuPipelineCache *cache) {
#if defined(LIGHTGLUE_HAS_CUDA)
    if (UseCudaCustomKernels(backend)) {
        return RunDkdGpu(score_map, h, w, options, backend, result, error,
                         cache != nullptr ? cache->dkd_scratch() : nullptr);
    }
#endif
#if defined(LIGHTGLUE_HAS_VULKAN)
    if (UseVulkanCompute(backend) && cache != nullptr) {
        return RunDkdVulkan(score_map, h, w, options, backend, result, error,
                            cache);
    }
#endif
#if !defined(LIGHTGLUE_HAS_CUDA) && !defined(LIGHTGLUE_HAS_VULKAN)
    (void)cache;
#endif

    std::vector<float> score_nchw;
    if (!score_map.DownloadNchw(backend, &score_nchw, 1, h, w, error)) {
        return false;
    }
    const DkdOutput cpu = RunDkd(score_nchw, h, w, options, w, h);
    const int32_t count = static_cast<int32_t>(cpu.scores.size());
    if (!GpuTensor::Allocate(backend, count * 2, 1, 1, &result->keypoints_norm,
                             error)) {
        return false;
    }
    if (!GpuTensor::Allocate(backend, count, 1, 1, &result->scores, error)) {
        return false;
    }
    ggml_backend_tensor_set(result->keypoints_norm.tensor,
                            cpu.keypoints_norm.data(), 0,
                            cpu.keypoints_norm.size() * sizeof(float));
    ggml_backend_tensor_set(result->scores.tensor, cpu.scores.data(), 0,
                            cpu.scores.size() * sizeof(float));
    result->count = count;
    return true;
}

bool RunSddhDispatch(const GpuTensor &feature_map,
                     int32_t descriptor_dim,
                     int32_t fh,
                     int32_t fw,
                     const std::vector<float> &keypoints_norm,
                     int32_t keypoint_count,
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
    GpuTensor kpts_gpu;
    if (!GpuTensor::Allocate(backend, keypoint_count * 2, 1, 1, &kpts_gpu,
                             error)) {
        return false;
    }
    ggml_backend_tensor_set(kpts_gpu.tensor, keypoints_norm.data(), 0,
                            keypoints_norm.size() * sizeof(float));

#if defined(LIGHTGLUE_HAS_CUDA)
    if (UseCudaCustomKernels(backend)) {
        return RunSddhGpu(feature_map, descriptor_dim, fh, fw, kpts_gpu,
                          keypoint_count, kernel_size, n_pos, offset_0_w,
                          offset_0_b, offset_2_w, offset_2_b, sf_conv_w,
                          agg_weights, backend, descriptors, error, cache);
    }
#endif
#if defined(LIGHTGLUE_HAS_VULKAN)
    if (UseVulkanCompute(backend) && cache != nullptr &&
        std::getenv("LIGHTGLUE_ALIKED_VULKAN_SDDH") != nullptr) {
        return RunSddhVulkan(feature_map, descriptor_dim, fh, fw, kpts_gpu,
                             keypoint_count, kernel_size, n_pos, cache, backend,
                             descriptors, error);
    }
#endif
#if !defined(LIGHTGLUE_HAS_CUDA) && !defined(LIGHTGLUE_HAS_VULKAN)
    (void)cache;
#endif

    std::vector<float> feature_nchw;
    if (!feature_map.DownloadNchw(backend, &feature_nchw, 128, fh, fw, error)) {
        return false;
    }
    const std::vector<float> desc =
            RunSddh(feature_nchw, descriptor_dim, fh, fw, keypoints_norm,
                    kernel_size, n_pos, offset_0_w, offset_0_b, offset_2_w,
                    offset_2_b, sf_conv_w, agg_weights);
    if (!GpuTensor::Allocate(backend, keypoint_count * descriptor_dim, 1, 1,
                             descriptors, error)) {
        return false;
    }
    ggml_backend_tensor_set(descriptors->tensor, desc.data(), 0,
                            desc.size() * sizeof(float));
    return true;
}

}  // namespace lightglue::aliked_internal
