// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/gpu_pipeline_cache.hpp"

#if defined(AICORE_CUDA_ALIKED)
#include "tasks/aliked/cuda/aliked_cuda.hpp"

#endif
#include "tasks/aliked/gpu_sync.hpp"
#include "tasks/aliked/gpu_tensor.hpp"
#include "tasks/aliked/model_weights.hpp"
#include "tasks/aliked/tensor_ops.hpp"

namespace lightglue::aliked_internal {
namespace {

float *DevPtr(const GpuTensor &tensor) {
    return reinterpret_cast<float *>(tensor.tensor->data);
}

bool UploadWeightTensor(internal::Backend *backend,
                        const std::vector<float> &weights,
                        GpuTensor *tensor,
                        std::string *error) {
    if (!GpuTensor::Allocate(backend, static_cast<int32_t>(weights.size()), 1,
                             1, tensor, error)) {
        return false;
    }
    ggml_backend_tensor_set(tensor->tensor, weights.data(), 0,
                            weights.size() * sizeof(float));
    return true;
}

void PreloadDcn(GpuPipelineCache *cache,
                const TensorMap &tensors,
                const std::string &prefix,
                int32_t ic,
                int32_t oc,
                std::string *error) {
    const FusedConv2dNchw fused1 = FuseConvBnNchw(
            RequireTensor(tensors, prefix + "_conv1_regular_conv_weight",
                          error),
            oc, ic, 3, 3, nullptr,
            RequireTensor(tensors, prefix + "_bn1_weight", error),
            RequireTensor(tensors, prefix + "_bn1_bias", error),
            RequireTensor(tensors, prefix + "_bn1_running_mean", error),
            RequireTensor(tensors, prefix + "_bn1_running_var", error));
    cache->EnsureDcnWeight(prefix + ".dcn1.deform", fused1, error);
    const FusedConv2dNchw fused2 = FuseConvBnNchw(
            RequireTensor(tensors, prefix + "_conv2_regular_conv_weight",
                          error),
            oc, oc, 3, 3, nullptr,
            RequireTensor(tensors, prefix + "_bn2_weight", error),
            RequireTensor(tensors, prefix + "_bn2_bias", error),
            RequireTensor(tensors, prefix + "_bn2_running_mean", error),
            RequireTensor(tensors, prefix + "_bn2_running_var", error));
    cache->EnsureDcnWeight(prefix + ".dcn2.deform", fused2, error);
}

}  // namespace

GpuPipelineCache::GpuPipelineCache(internal::Backend *backend)
    : backend_(backend), ggml_(backend), compute_ggml_(backend) {
#if defined(AICORE_CUDA_ALIKED)
    dkd_scratch_ = std::make_unique<AlikedDkdScratch>();
    sddh_scratch_ = std::make_unique<AlikedSddhScratch>();
#endif
}

GpuPipelineCache::~GpuPipelineCache() = default;

bool GpuPipelineCache::EnsureDcnWeight(const std::string &key,
                                       const FusedConv2dNchw &fused,
                                       std::string *error) {
    if (dcn_weights_.count(key) > 0) {
        return true;
    }
    CachedDcnWeights cached;
    if (!GpuTensor::Allocate(backend_,
                             static_cast<int32_t>(fused.kernel.size()), 1, 1,
                             &cached.weight, error)) {
        return false;
    }
    ggml_backend_tensor_set(cached.weight.tensor, fused.kernel.data(), 0,
                            fused.kernel.size() * sizeof(float));
    if (!GpuTensor::Allocate(backend_, static_cast<int32_t>(fused.bias.size()),
                             1, 1, &cached.bias, error)) {
        return false;
    }
    ggml_backend_tensor_set(cached.bias.tensor, fused.bias.data(), 0,
                            fused.bias.size() * sizeof(float));
    dcn_weights_.emplace(key, std::move(cached));
    return true;
}

const float *GpuPipelineCache::DcnWeightPtr(const std::string &key) const {
    return DevPtr(dcn_weights_.at(key).weight);
}

const float *GpuPipelineCache::DcnBiasPtr(const std::string &key) const {
    return DevPtr(dcn_weights_.at(key).bias);
}

bool GpuPipelineCache::EnsureDcnWorkspace(
        int32_t w, int32_t h, int32_t in_c, int32_t out_c, std::string *error) {
    if (w != dcn_ws_w_ || h != dcn_ws_h_) {
        dcn_ws_in_c_ = 0;
        dcn_ws_out_c_ = 0;
    }
    const int32_t alloc_in_c = std::max(in_c, dcn_ws_in_c_);
    const int32_t alloc_out_c = std::max(out_c, dcn_ws_out_c_);
    if (w == dcn_ws_w_ && h == dcn_ws_h_ && alloc_in_c == dcn_ws_in_c_ &&
        alloc_out_c == dcn_ws_out_c_) {
        return true;
    }
    if (!GpuTensor::Allocate(backend_, w, h, alloc_in_c, &dcn_nchw_in_,
                             error)) {
        return false;
    }
    if (!GpuTensor::Allocate(backend_, w, h, 18, &dcn_nchw_offset_, error)) {
        return false;
    }
    if (!GpuTensor::Allocate(backend_, w, h, alloc_out_c, &dcn_nchw_out_,
                             error)) {
        return false;
    }
    dcn_ws_w_ = w;
    dcn_ws_h_ = h;
    dcn_ws_in_c_ = alloc_in_c;
    dcn_ws_out_c_ = alloc_out_c;
    return true;
}

float *GpuPipelineCache::DcnNchwInPtr() { return DevPtr(dcn_nchw_in_); }

float *GpuPipelineCache::DcnNchwOffsetPtr() { return DevPtr(dcn_nchw_offset_); }

float *GpuPipelineCache::DcnNchwOutPtr() { return DevPtr(dcn_nchw_out_); }

const GpuPipelineCache::CachedDcnWeights *GpuPipelineCache::FindDcnWeight(
        const std::string &key) const {
    const auto it = dcn_weights_.find(key);
    if (it == dcn_weights_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool GpuPipelineCache::EnsureInput(int32_t w,
                                   int32_t h,
                                   int32_t c,
                                   std::string *error) {
    if (w == input_w_ && h == input_h_ && c == input_c_ &&
        input_buffer_.tensor != nullptr) {
        return true;
    }
    if (!GpuTensor::Allocate(backend_, w, h, c, &input_buffer_, error)) {
        return false;
    }
    input_w_ = w;
    input_h_ = h;
    input_c_ = c;
    return true;
}

bool GpuPipelineCache::Warmup(const TensorMap &tensors, std::string *error) {
    if (warmed_up_) {
        return true;
    }
    if (backend_ == nullptr || !backend_->IsGpu()) {
        if (error) {
            *error = "GpuPipelineCache requires a GPU backend";
        }
        return false;
    }

#if defined(AICORE_VULKAN_ALIKED)
    if (backend_->IsVulkan()) {
        ApplyVulkanAlikedPerfDefaults();
    }
#endif

    PreloadDcn(this, tensors, "block3", 32, 64, error);
    if (error != nullptr && !error->empty()) {
        return false;
    }
    PreloadDcn(this, tensors, "block4", 64, 128, error);
    if (error != nullptr && !error->empty()) {
        return false;
    }

    std::vector<float> ones8(8, 1.0f);
    std::vector<float> zeros8(8, 0.0f);
    std::vector<float> ones4(4, 1.0f);
    std::vector<float> zeros4(4, 0.0f);
    std::vector<float> ones1(1, 1.0f);
    std::vector<float> zeros1(1, 0.0f);

    score_head_layers_ = {
            {FuseConvBn(RequireTensor(tensors, "score_head_0_weight", error), 8,
                        128, 1, 1, nullptr, ones8, zeros8, zeros8, ones8),
             0, 1, "score_head_0", true},
            {FuseConvBn(RequireTensor(tensors, "score_head_2_weight", error), 4,
                        8, 3, 3, nullptr, ones4, zeros4, zeros4, ones4),
             1, 1, "score_head_2", true},
            {FuseConvBn(RequireTensor(tensors, "score_head_4_weight", error), 4,
                        4, 3, 3, nullptr, ones4, zeros4, zeros4, ones4),
             1, 1, "score_head_4", true},
    };
    score_head_final_ =
            FuseConvBn(RequireTensor(tensors, "score_head_6_weight", error), 1,
                       4, 3, 3, nullptr, ones1, zeros1, zeros1, ones1);
    if (error != nullptr && !error->empty()) {
        return false;
    }

    for (const GgmlGpuSession::ConvChainSpec &layer : score_head_layers_) {
        if (!ggml_.runner()->EnsureCachedPublic(layer.cache_key, layer.weights,
                                                error)) {
            return false;
        }
    }
    if (!ggml_.runner()->EnsureCachedPublic("score_head_6", score_head_final_,
                                            error)) {
        return false;
    }

    if (!EnsureSddhWeights(
                RequireTensor(tensors, "desc_head_offset_conv_0_weight", error),
                RequireTensor(tensors, "desc_head_offset_conv_0_bias", error),
                RequireTensor(tensors, "desc_head_offset_conv_2_weight", error),
                RequireTensor(tensors, "desc_head_offset_conv_2_bias", error),
                RequireTensor(tensors, "desc_head_sf_conv_weight", error),
                RequireTensor(tensors, "desc_head_agg_weights", error),
                error)) {
        return false;
    }

    warmed_up_ = true;
    return true;
}

bool GpuPipelineCache::EnsureComputeLinked(std::string *error) {
    if (compute_linked_) {
        return true;
    }
    if (!warmed_up_) {
        if (error) {
            *error = "GpuPipelineCache compute link requires Warmup";
        }
        return false;
    }
    compute_ggml_.runner()->ImportWeightEntriesFrom(*ggml_.runner());
    compute_linked_ = true;
    return true;
}

bool GpuPipelineCache::ShareWarmStateFrom(const GpuPipelineCache &source,
                                          std::string *error) {
    if (backend_ != source.backend_) {
        if (error) {
            *error = "GpuPipelineCache backend mismatch for weight sharing";
        }
        return false;
    }
    if (!source.warmed_up_) {
        if (error) {
            *error = "GpuPipelineCache weight source is not warmed up";
        }
        return false;
    }

    ggml_.runner()->ImportWeightEntriesFrom(*source.ggml()->runner());
    compute_ggml_.runner()->ImportWeightEntriesFrom(*source.ggml()->runner());
    score_head_layers_ = source.score_head_layers_;
    score_head_final_ = source.score_head_final_;
    warmed_up_ = true;
    compute_linked_ = true;
    return true;
}

bool GpuPipelineCache::EnsureSddhWeights(const std::vector<float> &offset_0_w,
                                         const std::vector<float> &offset_0_b,
                                         const std::vector<float> &offset_2_w,
                                         const std::vector<float> &offset_2_b,
                                         const std::vector<float> &sf_conv_w,
                                         const std::vector<float> &agg_weights,
                                         std::string *error) {
    if (sddh_loaded_) {
        return true;
    }
    if (!UploadWeightTensor(backend_, offset_0_w, &sddh_weights_.offset_0_w,
                            error) ||
        !UploadWeightTensor(backend_, offset_0_b, &sddh_weights_.offset_0_b,
                            error) ||
        !UploadWeightTensor(backend_, offset_2_w, &sddh_weights_.offset_2_w,
                            error) ||
        !UploadWeightTensor(backend_, offset_2_b, &sddh_weights_.offset_2_b,
                            error) ||
        !UploadWeightTensor(backend_, sf_conv_w, &sddh_weights_.sf_conv_w,
                            error) ||
        !UploadWeightTensor(backend_, agg_weights, &sddh_weights_.agg_weights,
                            error)) {
        return false;
    }
    sddh_loaded_ = true;
    return true;
}

const float *GpuPipelineCache::SddhOffset0WPtr() const {
    return DevPtr(sddh_weights_.offset_0_w);
}

const float *GpuPipelineCache::SddhOffset0BPtr() const {
    return DevPtr(sddh_weights_.offset_0_b);
}

const float *GpuPipelineCache::SddhOffset2WPtr() const {
    return DevPtr(sddh_weights_.offset_2_w);
}

const float *GpuPipelineCache::SddhOffset2BPtr() const {
    return DevPtr(sddh_weights_.offset_2_b);
}

const float *GpuPipelineCache::SddhSfConvWPtr() const {
    return DevPtr(sddh_weights_.sf_conv_w);
}

const float *GpuPipelineCache::SddhAggWeightsPtr() const {
    return DevPtr(sddh_weights_.agg_weights);
}

#if defined(AICORE_VULKAN_ALIKED)

bool GpuPipelineCache::EnsureVulkanDkdScratch(int32_t h,
                                              int32_t w,
                                              int32_t max_kpts,
                                              std::string *error) {
    const int32_t count = h * w;
    if (vulkan_dkd_scratch_.map_count >= count &&
        vulkan_dkd_scratch_.max_kpts >= max_kpts) {
        return true;
    }
    if (!GpuTensor::Allocate(backend_, count, 1, 1, &vulkan_dkd_scratch_.nms,
                             error) ||
        !GpuTensor::Allocate(backend_, count, 1, 1, &vulkan_dkd_scratch_.tmp_a,
                             error) ||
        !GpuTensor::Allocate(backend_, count, 1, 1, &vulkan_dkd_scratch_.tmp_b,
                             error) ||
        !GpuTensor::Allocate(backend_, count, 1, 1, &vulkan_dkd_scratch_.tmp_c,
                             error)) {
        return false;
    }
    constexpr int32_t kCandidates = 256 * 32;
    if (!GpuTensor::Allocate(backend_, kCandidates, 1, 1,
                             &vulkan_dkd_scratch_.block_keys, error) ||
        !GpuTensor::Allocate(backend_, kCandidates, 1, 1,
                             &vulkan_dkd_scratch_.block_indices, error)) {
        return false;
    }
    if (!GpuTensor::Allocate(backend_, max_kpts, 1, 1,
                             &vulkan_dkd_scratch_.indices_dev, error)) {
        return false;
    }
    vulkan_dkd_scratch_.map_count = count;
    vulkan_dkd_scratch_.max_kpts = max_kpts;
    return true;
}

bool GpuPipelineCache::EnsureVulkanSddhScratch(int32_t count,
                                               int32_t dim,
                                               int32_t kernel_size,
                                               int32_t feat_h,
                                               int32_t feat_w,
                                               std::string *error) {
    const int32_t capacity = count > 0 ? count : 1;
    const bool workspace_ok = count <= vulkan_sddh_scratch_.capacity_count &&
                              dim == vulkan_sddh_scratch_.dim &&
                              kernel_size == vulkan_sddh_scratch_.kernel_size;
    const bool feature_ok =
            feat_h == vulkan_sddh_scratch_.feat_h &&
            feat_w == vulkan_sddh_scratch_.feat_w &&
            vulkan_sddh_scratch_.feature_contig.tensor != nullptr;
    if (workspace_ok && feature_ok) {
        return true;
    }
    if (!workspace_ok) {
        const size_t per = static_cast<size_t>(dim) *
                                   static_cast<size_t>(kernel_size) *
                                   static_cast<size_t>(kernel_size) +
                           64 + static_cast<size_t>(dim) * 3;
        // Vulkan SSBO misalign can add up to 16 floats before tensor data;
        // shader indexes workspace_off + k*stride — keep tail slack to avoid
        // OOB writes.
        constexpr size_t kMisalignSlack = 16;
        const size_t floats =
                per * static_cast<size_t>(capacity) + kMisalignSlack;
        if (!GpuTensor::Allocate(backend_, static_cast<int32_t>(floats), 1, 1,
                                 &vulkan_sddh_scratch_.workspace, error)) {
            return false;
        }
        vulkan_sddh_scratch_.capacity_count = capacity;
        vulkan_sddh_scratch_.dim = dim;
        vulkan_sddh_scratch_.kernel_size = kernel_size;
    }
    if (!feature_ok) {
        if (!GpuTensor::Allocate(backend_, feat_w, feat_h, dim,
                                 &vulkan_sddh_scratch_.feature_contig, error)) {
            return false;
        }
        vulkan_sddh_scratch_.feat_h = feat_h;
        vulkan_sddh_scratch_.feat_w = feat_w;
    }
    return true;
}

#endif

}  // namespace lightglue::aliked_internal
