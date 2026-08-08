// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aliked_gpu_ops.hpp"

#include "deform_conv.hpp"
#include "ggml_gpu_ops.hpp"
#include "gpu_sync.hpp"
#include "postprocess.hpp"
#include "score_debug.hpp"

#if defined(AICORE_CUDA_ALIKED)
#include <cuda_runtime.h>

#include "cuda/aliked_cuda.hpp"
#endif

#if defined(AICORE_VULKAN_ALIKED)
#include "gpu_tensor.hpp"
#include "vulkan/vulkan_aliked_dispatch.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace lightglue::aliked_internal {
namespace {

float VectorMaxAbsDiff(const std::vector<float> &a,
                       const std::vector<float> &b) {
    const size_t n = std::min(a.size(), b.size());
    float best = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        best = std::max(best, std::fabs(a[i] - b[i]));
    }
    return best;
}

void DumpSddhStageCompare(const char *label,
                          const std::vector<float> &cpu,
                          const std::vector<float> &gpu) {
    if (cpu.empty()) {
        std::fprintf(stderr, "[sddh-stage] %s cpu=EMPTY gpu=%zu max_abs=NA\n",
                     label, gpu.size());
        return;
    }
    std::fprintf(stderr,
                 "[sddh-stage] %s len=%zu max_abs=%.6g cpu0=%.6g gpu0=%.6g\n",
                 label, cpu.size(), VectorMaxAbsDiff(cpu, gpu), cpu[0],
                 gpu.empty() ? 0.0f : gpu[0]);
}

#if defined(AICORE_VULKAN_ALIKED)
void DebugCompareVulkanSddhStages(const GpuTensor &feature_map,
                                  int32_t dim,
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
                                  const GpuTensor &descriptors,
                                  GpuPipelineCache *cache) {
    if (std::getenv("LIGHTGLUE_ALIKED_SDDH_DEBUG") == nullptr ||
        keypoint_count <= 0) {
        return;
    }
    const int32_t k = 0;
    std::vector<float> feature_nchw;
    if (!feature_map.DownloadNchw(backend, &feature_nchw, dim, fh, fw,
                                  nullptr)) {
        std::fprintf(stderr, "[sddh-stage] feature download failed\n");
        return;
    }
    SddhStageDump cpu{};
    if (!RunSddhStages(feature_nchw, dim, fh, fw, keypoints_norm, k,
                       kernel_size, n_pos, offset_0_w, offset_0_b, offset_2_w,
                       offset_2_b, sf_conv_w, agg_weights, &cpu)) {
        std::fprintf(stderr, "[sddh-stage] cpu RunSddhStages failed\n");
        return;
    }

    std::vector<float> vk_desc(static_cast<size_t>(keypoint_count) * dim);
    ggml_backend_tensor_get(descriptors.tensor, vk_desc.data(), 0,
                            vk_desc.size() * sizeof(float));
    const float *vk0 = vk_desc.data();
    float vk_norm = 0.0f;
    for (int32_t c = 0; c < dim; ++c) {
        vk_norm += vk0[c] * vk0[c];
    }
    vk_norm = std::sqrt(vk_norm);
    float cpu_norm = 0.0f;
    for (float v : cpu.desc) {
        cpu_norm += v * v;
    }
    cpu_norm = std::sqrt(cpu_norm);
    float cos = 0.0f;
    for (int32_t c = 0; c < dim; ++c) {
        cos += cpu.desc[static_cast<size_t>(c)] * vk0[c];
    }
    std::fprintf(stderr,
                 "[sddh-stage] k=%d kpt_norm=(%.6g,%.6g) x_wh=%.4f y_wh=%.4f "
                 "cpu_desc_norm=%.6g vk_desc_norm=%.6g cos=%.6g\n",
                 k, keypoints_norm[0], keypoints_norm[1], cpu.x_wh, cpu.y_wh,
                 cpu_norm, vk_norm, cos);

    AlikedVulkanSddhScratch *scratch = cache->vulkan_sddh_scratch();
    if (scratch == nullptr || scratch->workspace.tensor == nullptr) {
        return;
    }
    const size_t patch_size =
            static_cast<size_t>(dim) * kernel_size * kernel_size;
    const size_t stride = patch_size + 64 + static_cast<size_t>(dim) * 3;
    std::vector<float> ws(stride);
    const size_t byte_off = static_cast<size_t>(k) * stride * sizeof(float);
    ggml_backend_tensor_get(scratch->workspace.tensor, ws.data(), byte_off,
                            ws.size() * sizeof(float));

    const size_t off_raw = patch_size;
    const size_t off_final = patch_size + 32;
    const size_t sampled = patch_size + 64;
    const size_t transformed = sampled + static_cast<size_t>(dim);
    const size_t desc_base = transformed + static_cast<size_t>(dim);

    DumpSddhStageCompare(
            "patch", cpu.patch,
            std::vector<float>(ws.begin(), ws.begin() + patch_size));
    DumpSddhStageCompare("offset_raw", cpu.offset_raw,
                         std::vector<float>(ws.begin() + off_raw,
                                            ws.begin() + off_raw + 32));
    DumpSddhStageCompare("offset_final", cpu.offset_final,
                         std::vector<float>(ws.begin() + off_final,
                                            ws.begin() + off_final + 32));
    DumpSddhStageCompare("sampled", cpu.sampled,
                         std::vector<float>(ws.begin() + sampled,
                                            ws.begin() + sampled +
                                                    static_cast<size_t>(dim)));
    DumpSddhStageCompare("transformed", cpu.transformed,
                         std::vector<float>(ws.begin() + transformed,
                                            ws.begin() + transformed +
                                                    static_cast<size_t>(dim)));
    DumpSddhStageCompare("desc_pre_norm", cpu.desc_pre_norm,
                         std::vector<float>(ws.begin() + desc_base,
                                            ws.begin() + desc_base +
                                                    static_cast<size_t>(dim)));

    const int32_t k_last = keypoint_count - 1;
    if (k_last <= 0) {
        return;
    }
    SddhStageDump cpu_last{};
    if (!RunSddhStages(feature_nchw, dim, fh, fw, keypoints_norm, k_last,
                       kernel_size, n_pos, offset_0_w, offset_0_b, offset_2_w,
                       offset_2_b, sf_conv_w, agg_weights, &cpu_last)) {
        return;
    }
    std::vector<float> ws_last(stride);
    const size_t off_last =
            static_cast<size_t>(k_last) * stride * sizeof(float);
    ggml_backend_tensor_get(scratch->workspace.tensor, ws_last.data(), off_last,
                            ws_last.size() * sizeof(float));
    const float *vk_last = vk_desc.data() + static_cast<size_t>(k_last) * dim;
    float vk_last_norm = 0.0f;
    float cos_last = 0.0f;
    for (int32_t c = 0; c < dim; ++c) {
        vk_last_norm += vk_last[c] * vk_last[c];
        cos_last += cpu_last.desc[static_cast<size_t>(c)] * vk_last[c];
    }
    vk_last_norm = std::sqrt(vk_last_norm);
    std::fprintf(
            stderr,
            "[sddh-stage] k=%d vk_desc_norm=%.6g cos=%.6g patch_max_abs=%.6g\n",
            k_last, vk_last_norm, cos_last,
            VectorMaxAbsDiff(cpu_last.patch,
                             std::vector<float>(ws_last.begin(),
                                                ws_last.begin() + patch_size)));

    std::vector<float> cos_all(static_cast<size_t>(keypoint_count));
    for (int32_t ki = 0; ki < keypoint_count; ++ki) {
        SddhStageDump cpu_ki{};
        if (!RunSddhStages(feature_nchw, dim, fh, fw, keypoints_norm, ki,
                           kernel_size, n_pos, offset_0_w, offset_0_b,
                           offset_2_w, offset_2_b, sf_conv_w, agg_weights,
                           &cpu_ki)) {
            cos_all[static_cast<size_t>(ki)] = 0.0f;
            continue;
        }
        const float *vk_i = vk_desc.data() + static_cast<size_t>(ki) * dim;
        float dot = 0.0f;
        for (int32_t c = 0; c < dim; ++c) {
            dot += cpu_ki.desc[static_cast<size_t>(c)] * vk_i[c];
        }
        cos_all[static_cast<size_t>(ki)] = dot;
    }
    std::nth_element(cos_all.begin(), cos_all.begin() + cos_all.size() / 2,
                     cos_all.end());
    std::fprintf(stderr, "[sddh-stage] all-k desc cos median=%.6g min=%.6g\n",
                 cos_all[cos_all.size() / 2],
                 *std::min_element(cos_all.begin(), cos_all.end()));
}
#endif

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
    float max_abs_diff = 0.0f;
    double cpu_sum = 0.0;
    double vk_sum = 0.0;
    for (size_t i = 0; i < entry.cpu.size(); ++i) {
        cpu_sum += static_cast<double>(entry.cpu[i]);
        vk_sum += static_cast<double>(entry.vulkan[i]);
        max_abs_diff = std::max(max_abs_diff,
                                std::fabs(entry.vulkan[i] - entry.cpu[i]));
    }
    const uint32_t wg_x = (static_cast<uint32_t>(entry.w) + 15u) / 16u;
    const uint32_t wg_y = (static_cast<uint32_t>(entry.h) + 15u) / 16u;
    const uint32_t wg_z = static_cast<uint32_t>(entry.c);
    std::fprintf(
            stderr,
            "[dcn-parity] name=%s c=%d h=%d w=%d dispatch_elems=(%d,%d,%d) "
            "wg=(%u,%u,%u) cpu_sum=%.3f vk_sum=%.3f max_abs_diff=%.6e\n",
            name.c_str(), entry.c, entry.h, entry.w, entry.w, entry.h, entry.c,
            wg_x, wg_y, wg_z, cpu_sum, vk_sum, max_abs_diff);
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

bool UseCudaCustomKernels(internal::Backend *backend) {
#if defined(AICORE_CUDA_ALIKED)
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

#if defined(AICORE_CUDA_ALIKED)
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
    SyncGpuPipeline(cache->backend());
    return true;
}
#endif

#if defined(AICORE_VULKAN_ALIKED)
bool LogDcnSubstageIfDebug(internal::Backend *backend,
                           const GpuTensor &tensor,
                           int32_t c,
                           int32_t h,
                           int32_t w,
                           const std::string &cache_prefix,
                           const char *suffix,
                           std::string *error) {
    if (backend == nullptr || !backend->IsVulkan() || !DkdDebugEnabled()) {
        return true;
    }
    BarrierGpuPipeline(backend);
    return LogBackboneStage(backend, tensor, c, h, w,
                            (cache_prefix + suffix).c_str(), error);
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
    GpuTensor offset;
    if (!ConvGpu(cache->ggml()->runner(), input, offset_w, 18, 3, 3, &offset_b,
                 1, 1, &offset, (cache_prefix + ".offset").c_str(), error)) {
        return false;
    }
    if (!EnsureDenseWhcn(cache->backend(), &offset, error)) {
        return false;
    }
    if (!LogDcnSubstageIfDebug(cache->backend(), offset, 18, input.h, input.w,
                               cache_prefix, ".offset", error)) {
        return false;
    }

    if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
        std::cerr << "dcn_vk offset conv done " << cache_prefix << "\n";
    }

    const float max_offset =
            static_cast<float>(std::max(input.h, input.w)) / 4.0f;
    if (!RunClampGpu(cache->backend(), &offset, -max_offset, max_offset,
                     error)) {
        if (error && error->empty()) {
            *error = "offset clamp failed";
        }
        return false;
    }
    if (!LogDcnSubstageIfDebug(cache->backend(), offset, 18, input.h, input.w,
                               cache_prefix, ".offset_clamp", error)) {
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
    internal::Backend *backend = cache->backend();
    const GpuPipelineCache::CachedDcnWeights *weights =
            cache->FindDcnWeight(weight_key);
    if (weights == nullptr) {
        if (error) {
            *error = "DCN weight cache miss";
        }
        return false;
    }
    if (!GpuTensor::Allocate(backend, input.w, input.h, oc, output, error)) {
        return false;
    }
    // gallocr / cached conv outputs may carry padded strides; vk deform reads
    // flat WHCN.
    GpuTensor input_contig;
    GpuTensor offset_contig;
    if (!GpuTensor::Allocate(backend, input.w, input.h, input.c, &input_contig,
                             error) ||
        !GpuTensor::Allocate(backend, offset.w, offset.h, offset.c,
                             &offset_contig, error)) {
        return false;
    }
    BackendTensorCopyCompat(cache->backend(), input.tensor,
                            input_contig.tensor);
    BackendTensorCopyCompat(cache->backend(), offset.tensor,
                            offset_contig.tensor);
    if (std::getenv("LIGHTGLUE_ALIKED_DCN_DEBUG") != nullptr ||
        DkdDebugEnabled()) {
        const uint32_t wg_x = (static_cast<uint32_t>(input.w) + 15u) / 16u;
        const uint32_t wg_y = (static_cast<uint32_t>(input.h) + 15u) / 16u;
        const uint32_t wg_z = static_cast<uint32_t>(oc);
        std::fprintf(
                stderr,
                "[dcn-dispatch] stage=%s ic=%d ih=%d iw=%d oc=%d kh=3 kw=3 "
                "pad=1 layout=WHCN elems=(%d,%d,%d) wg=(%u,%u,%u) "
                "global_inv=(%u,%u,%u)\n",
                cache_prefix.c_str(), input.c, input.h, input.w, oc, input.w,
                input.h, oc, wg_x, wg_y, wg_z, wg_x * 16u, wg_y * 16u, wg_z);
    }
    if (!VkAlikedDeformConv2d(backend->handle, input_contig.tensor,
                              offset_contig.tensor, weights->weight.tensor,
                              weights->bias.tensor, output->tensor, input.c,
                              input.h, input.w, oc, 3, 3, 1, 1)) {
        if (error) {
            *error = "deform conv Vulkan failed";
        }
        return false;
    }
    if (!LogDcnSubstageIfDebug(cache->backend(), *output, oc, input.h, input.w,
                               cache_prefix, ".deform_out", error)) {
        return false;
    }
    if (std::getenv("LIGHTGLUE_ALIKED_TRACE")) {
        std::cerr << "dcn_vk deform done " << cache_prefix << "\n";
    }
    FlushGpuPipeline(backend);
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
    SyncGpuTensorMeta(const_cast<GpuTensor *>(&score_map));
    BarrierGpuPipeline(backend);
    VkAlikedQueueIdle(backend->handle);

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
    FlushGpuPipeline(backend);
    if (!VkAlikedRunDkd(backend->handle, score_map.tensor, h, w, options.radius,
                        options.top_k, options.scores_th, options.n_limit,
                        result->keypoints_norm.tensor, result->scores.tensor,
                        &count, scratch->nms.tensor, scratch->tmp_a.tensor,
                        scratch->tmp_b.tensor, scratch->tmp_c.tensor,
                        scratch->block_keys.tensor,
                        scratch->block_indices.tensor,
                        scratch->indices_dev.tensor)) {
        if (error) {
            *error = "Vulkan DKD failed";
        }
        return false;
    }
    result->count = count;
    VkAlikedQueueIdle(backend->handle);
    FlushGpuPipeline(backend);
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
    FlushGpuPipeline(backend);

    const ggml_tensor *feat = feature_map.tensor;
    if (!IsContiguousWhcn(feature_map.tensor, fw, fh, descriptor_dim) &&
        scratch->feature_contig.tensor != nullptr) {
        if (VkAlikedDenseCopyWhcn(backend->handle, feature_map.tensor,
                                  scratch->feature_contig.tensor, fw, fh,
                                  descriptor_dim)) {
            feat = scratch->feature_contig.tensor;
        } else {
            if (error) {
                *error = "SDDH feature map is not contiguous WHCN";
            }
            return false;
        }
    }

    const bool ok = VkAlikedRunSddh(
            backend->handle, feat, descriptor_dim, fh, fw,
            keypoints_norm.tensor, keypoint_count, kernel_size, n_pos,
            w.offset_0_w.tensor, w.offset_0_b.tensor, w.offset_2_w.tensor,
            w.offset_2_b.tensor, w.sf_conv_w.tensor, w.agg_weights.tensor,
            scratch->workspace.tensor, descriptors->tensor);
    if (!ok && error) {
        *error = "Vulkan SDDH failed";
        return false;
    }
    FlushGpuPipeline(backend);
    ggml_backend_synchronize(backend->handle);
    return true;
}
#endif

}  // namespace

#if defined(AICORE_VULKAN_ALIKED)
bool UseVulkanCompute(internal::Backend *backend) {
    return backend != nullptr && backend->IsVulkan() &&
           backend->vulkan_config.compute && VkAlikedAvailable(backend->handle);
}

bool UseVulkanGpuUpsample(internal::Backend *backend) {
    if (backend == nullptr || !backend->IsVulkan() ||
        !VkAlikedAvailable(backend->handle)) {
        return false;
    }
    return backend->vulkan_config.gpu_upsample;
}

bool UseVulkanDcn(internal::Backend *backend) {
    if (backend == nullptr || !backend->IsVulkan() ||
        !VkAlikedAvailable(backend->handle)) {
        return false;
    }
    return backend->vulkan_config.dcn;
}

bool UseVulkanPostprocess(internal::Backend *backend) {
    if (backend == nullptr || !backend->IsVulkan() ||
        !VkAlikedAvailable(backend->handle)) {
        return false;
    }
    // VkAliked DKD is opt-in until parity gates cover all drivers.
    return backend->vulkan_config.postprocess;
}

bool UseVulkanSddh(internal::Backend *backend) {
    if (backend == nullptr || !backend->IsVulkan() ||
        !VkAlikedAvailable(backend->handle)) {
        return false;
    }
    // The scalar shader is retained for diagnostics; the exact CPU fallback is
    // faster and more reliable on current Vulkan drivers.
    return backend->vulkan_config.sddh;
}
#endif

AlikedCustomOpBackend DetectCustomOpBackend(internal::Backend *backend) {
    if (backend == nullptr || !backend->IsGpu()) {
        return AlikedCustomOpBackend::kCpu;
    }
    if (UseCudaCustomKernels(backend)) {
        return AlikedCustomOpBackend::kCuda;
    }
#if defined(AICORE_VULKAN_ALIKED)
    if (UseVulkanCompute(backend)) {
        return AlikedCustomOpBackend::kVulkanCompute;
    }
#endif
#if defined(AICORE_VULKAN_ALIKED)
    if (backend->IsVulkan()) {
        return AlikedCustomOpBackend::kVulkanBridge;
    }
#endif
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
#if defined(AICORE_VULKAN_ALIKED)
    if (debug && cache->backend() != nullptr && cache->backend()->IsVulkan()) {
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
#if defined(AICORE_CUDA_ALIKED)
        case AlikedCustomOpBackend::kCuda:
            return DcnConvBnCuda(cache, input, offset_w, offset_b, regular_w,
                                 oc, gamma, beta, mean, var, cache_prefix,
                                 output, error);
#endif
#if defined(AICORE_VULKAN_ALIKED)
        case AlikedCustomOpBackend::kVulkanCompute:
            // Compute availability alone does not qualify the custom DCN
            // implementation.  Respect the session capability gate here as
            // well as in the bridge path; otherwise `compute=true` silently
            // bypasses a disabled DCN and can corrupt ALIKED score maps.
            if (UseVulkanDcn(cache->backend())) {
                return DcnConvBnVulkan(cache, input, offset_w, offset_b,
                                       regular_w, oc, gamma, beta, mean, var,
                                       cache_prefix, output, error);
            }
            return DcnConvBnCpuBridge(cache, input, offset_w, offset_b,
                                      regular_w, oc, gamma, beta, mean, var,
                                      cache_prefix, output, error);
#endif
#if defined(AICORE_VULKAN_ALIKED)
        case AlikedCustomOpBackend::kVulkanBridge:
            if (UseVulkanDcn(cache->backend())) {
                return DcnConvBnVulkan(cache, input, offset_w, offset_b,
                                       regular_w, oc, gamma, beta, mean, var,
                                       cache_prefix, output, error);
            }
            return DcnConvBnCpuBridge(cache, input, offset_w, offset_b,
                                      regular_w, oc, gamma, beta, mean, var,
                                      cache_prefix, output, error);
#endif
        case AlikedCustomOpBackend::kCpu:
            // Metal / CPU / any GPU without native DCN → CPU bridge fallback.
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
#if defined(AICORE_CUDA_ALIKED)
    if (UseCudaCustomKernels(backend)) {
        return RunDkdGpu(score_map, h, w, options, backend, result, error,
                         cache != nullptr ? cache->dkd_scratch() : nullptr);
    }
#endif
#if defined(AICORE_VULKAN_ALIKED)
    if (UseVulkanPostprocess(backend) && cache != nullptr) {
        return RunDkdVulkan(score_map, h, w, options, backend, result, error,
                            cache);
    }
#endif
#if !defined(AICORE_CUDA_ALIKED) && !defined(AICORE_VULKAN_ALIKED)
    (void)cache;
#endif

    std::vector<float> score_nchw;
    SyncGpuTensorMeta(const_cast<GpuTensor *>(&score_map));
    const int32_t sh = score_map.h;
    const int32_t sw = score_map.w;
    if (sh <= 0 || sw <= 0 || score_map.c <= 0) {
        if (error) {
            *error = "invalid score map tensor shape for DKD";
        }
        return false;
    }
    SyncGpuPipeline(backend);
    FlushGpuPipeline(backend);
    BarrierGpuPipeline(backend);
    if (!score_map.DownloadNchw(backend, &score_nchw, score_map.c, sh, sw,
                                error)) {
        return false;
    }
    if (sh <= 0 || sw <= 0 ||
        score_nchw.size() !=
                static_cast<size_t>(sh) * static_cast<size_t>(sw)) {
        if (error) {
            *error = "score map host buffer size mismatch";
        }
        return false;
    }
    const DkdOutput cpu = RunDkd(score_nchw, sh, sw, options, sw, sh);
    const int32_t count = static_cast<int32_t>(cpu.scores.size());
    if (count == 0) {
        result->count = 0;
        return true;
    }
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
    FlushGpuPipeline(backend);
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

#if defined(AICORE_CUDA_ALIKED)
    if (UseCudaCustomKernels(backend)) {
        return RunSddhGpu(feature_map, descriptor_dim, fh, fw, kpts_gpu,
                          keypoint_count, kernel_size, n_pos, offset_0_w,
                          offset_0_b, offset_2_w, offset_2_b, sf_conv_w,
                          agg_weights, backend, descriptors, error, cache);
    }
#endif
#if defined(AICORE_VULKAN_ALIKED)
    if (UseVulkanSddh(backend) && cache != nullptr && cache->HasSddhWeights()) {
        if (std::getenv("LIGHTGLUE_ALIKED_VULKAN_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[vk-aliked] RunSddhDispatch: Vulkan SDDH path\n");
        }
        if (!RunSddhVulkan(feature_map, descriptor_dim, fh, fw, kpts_gpu,
                           keypoint_count, kernel_size, n_pos, cache, backend,
                           descriptors, error)) {
            return false;
        }
        DebugCompareVulkanSddhStages(feature_map, descriptor_dim, fh, fw,
                                     keypoints_norm, keypoint_count,
                                     kernel_size, n_pos, offset_0_w, offset_0_b,
                                     offset_2_w, offset_2_b, sf_conv_w,
                                     agg_weights, backend, *descriptors, cache);
        return true;
    }
    if (UseVulkanSddh(backend) &&
        std::getenv("LIGHTGLUE_ALIKED_VULKAN_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "[vk-aliked] RunSddhDispatch: CPU fallback (cache=%p "
                     "has_sddh=%d)\n",
                     static_cast<void *>(cache),
                     cache != nullptr && cache->HasSddhWeights());
    }
#endif
#if !defined(AICORE_CUDA_ALIKED) && !defined(AICORE_VULKAN_ALIKED)
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
    FlushGpuPipeline(backend);
    return true;
}

}  // namespace lightglue::aliked_internal
