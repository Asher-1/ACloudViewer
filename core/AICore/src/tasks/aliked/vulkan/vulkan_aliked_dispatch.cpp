// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/aliked/vulkan/vulkan_aliked_dispatch.hpp"

#include <ggml-backend.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>

#include "tasks/aliked/aliked_common.hpp"

#if defined(AICORE_VULKAN_ALIKED) && (defined(__linux__) || defined(__APPLE__))
#include <dlfcn.h>
#endif

namespace lightglue::aliked_internal {
namespace {

using FnAvailable = bool (*)(ggml_backend_t);
using FnWhcnToNchw = bool (*)(ggml_backend_t,
                              const ggml_tensor *,
                              ggml_tensor *,
                              int32_t,
                              int32_t,
                              int32_t);
using FnNchwToWhcn = bool (*)(ggml_backend_t,
                              const ggml_tensor *,
                              ggml_tensor *,
                              int32_t,
                              int32_t,
                              int32_t);
using FnDenseCopy = bool (*)(ggml_backend_t,
                             const ggml_tensor *,
                             ggml_tensor *,
                             int32_t,
                             int32_t,
                             int32_t);
using FnClamp = bool (*)(ggml_backend_t, ggml_tensor *, size_t, float, float);
using FnL2Norm =
        bool (*)(ggml_backend_t, ggml_tensor *, int32_t, int32_t, int32_t);
using FnDeformConv = bool (*)(ggml_backend_t,
                              const ggml_tensor *,
                              const ggml_tensor *,
                              const ggml_tensor *,
                              const ggml_tensor *,
                              ggml_tensor *,
                              int32_t,
                              int32_t,
                              int32_t,
                              int32_t,
                              int32_t,
                              int32_t,
                              int32_t,
                              int32_t);
using FnRunDkd = bool (*)(ggml_backend_t,
                          const ggml_tensor *,
                          int32_t,
                          int32_t,
                          int32_t,
                          int32_t,
                          float,
                          int32_t,
                          ggml_tensor *,
                          ggml_tensor *,
                          int32_t *,
                          ggml_tensor *,
                          ggml_tensor *,
                          ggml_tensor *,
                          ggml_tensor *,
                          ggml_tensor *,
                          ggml_tensor *,
                          ggml_tensor *);
using FnRunSddh = bool (*)(ggml_backend_t,
                           const ggml_tensor *,
                           int32_t,
                           int32_t,
                           int32_t,
                           const ggml_tensor *,
                           int32_t,
                           int32_t,
                           int32_t,
                           const ggml_tensor *,
                           const ggml_tensor *,
                           const ggml_tensor *,
                           const ggml_tensor *,
                           const ggml_tensor *,
                           const ggml_tensor *,
                           ggml_tensor *,
                           ggml_tensor *);
using FnUpsample = bool (*)(ggml_backend_t,
                            const ggml_tensor *,
                            ggml_tensor *,
                            int32_t,
                            int32_t,
                            int32_t,
                            int32_t,
                            int32_t);
using FnQueueIdle = bool (*)(ggml_backend_t);

struct VulkanAlikedApi {
    FnAvailable available = nullptr;
    FnQueueIdle queue_idle = nullptr;
    FnWhcnToNchw whcn_to_nchw = nullptr;
    FnNchwToWhcn nchw_to_whcn = nullptr;
    FnDenseCopy dense_copy = nullptr;
    FnClamp clamp = nullptr;
    FnL2Norm l2norm = nullptr;
    FnDeformConv deform_conv = nullptr;
    FnRunDkd run_dkd = nullptr;
    FnRunSddh run_sddh = nullptr;
    FnUpsample upsample = nullptr;
    bool resolved = false;
};

std::mutex g_api_mutex;
VulkanAlikedApi g_api;

template <typename Call>
bool CallVulkanNoThrow(const char *operation, Call &&call) {
    try {
        return call();
    } catch (const std::exception &e) {
        ALIKED_LOG_ERR("%s failed: %s", operation, e.what());
    } catch (...) {
        ALIKED_LOG_ERR("%s failed with an unknown exception", operation);
    }
    return false;
}

template <typename Fn>
Fn ResolveFn(ggml_backend_reg_t reg, const char *name) {
    if (reg != nullptr) {
        if (void *p = ggml_backend_reg_get_proc_address(reg, name)) {
            return reinterpret_cast<Fn>(p);
        }
    }
#if defined(AICORE_VULKAN_ALIKED) && (defined(__linux__) || defined(__APPLE__))
    if (void *p = dlsym(RTLD_DEFAULT, name)) {
        return reinterpret_cast<Fn>(p);
    }
    static void *vk_module = []() -> void * {
        return dlopen("libggml-vulkan.so", RTLD_LAZY | RTLD_GLOBAL);
    }();
    if (vk_module != nullptr) {
        if (void *p = dlsym(vk_module, name)) {
            return reinterpret_cast<Fn>(p);
        }
    }
#endif
    return nullptr;
}

void EnsureResolved(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    if (g_api.resolved && g_api.available != nullptr) {
        return;
    }

    ggml_backend_reg_t reg = nullptr;
    if (backend != nullptr) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev != nullptr) {
            reg = ggml_backend_dev_backend_reg(dev);
        }
    }

    g_api.available =
            ResolveFn<FnAvailable>(reg, "ggml_vulkan_aliked_available");
    g_api.queue_idle =
            ResolveFn<FnQueueIdle>(reg, "ggml_vulkan_aliked_queue_idle");
    g_api.whcn_to_nchw =
            ResolveFn<FnWhcnToNchw>(reg, "ggml_vulkan_aliked_whcn_to_nchw");
    g_api.nchw_to_whcn =
            ResolveFn<FnNchwToWhcn>(reg, "ggml_vulkan_aliked_nchw_to_whcn");
    g_api.dense_copy =
            ResolveFn<FnDenseCopy>(reg, "ggml_vulkan_aliked_dense_copy_whcn");
    g_api.clamp = ResolveFn<FnClamp>(reg, "ggml_vulkan_aliked_clamp_inplace");
    g_api.l2norm =
            ResolveFn<FnL2Norm>(reg, "ggml_vulkan_aliked_l2norm_inplace");
    g_api.deform_conv =
            ResolveFn<FnDeformConv>(reg, "ggml_vulkan_aliked_deform_conv2d");
    g_api.run_dkd = ResolveFn<FnRunDkd>(reg, "ggml_vulkan_aliked_run_dkd");
    g_api.run_sddh = ResolveFn<FnRunSddh>(reg, "ggml_vulkan_aliked_run_sddh");
    g_api.upsample =
            ResolveFn<FnUpsample>(reg, "ggml_vulkan_aliked_upsample_bilinear");
    g_api.resolved = g_api.available != nullptr;
}

void LogVkAlikedProbe(ggml_backend_t backend) {
    // The historical LIGHTGLUE_ALIKED_VULKAN_TRACE gate was development
    // scaffolding and is removed; the probe dump is dormant.
    (void)backend;
}

}  // namespace

bool VkAlikedAvailable(ggml_backend_t backend) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    return false;
#else
    EnsureResolved(backend);
    LogVkAlikedProbe(backend);
    return g_api.available != nullptr && CallVulkanNoThrow("availability", [&] {
               return g_api.available(backend);
           });
#endif
}

bool VkAlikedWhcnToNchw(ggml_backend_t backend,
                        const ggml_tensor *whcn,
                        ggml_tensor *nchw,
                        int32_t c,
                        int32_t h,
                        int32_t w) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)whcn;
    (void)nchw;
    (void)c;
    (void)h;
    (void)w;
    return false;
#else
    EnsureResolved(backend);
    return g_api.whcn_to_nchw != nullptr &&
           CallVulkanNoThrow("whcn_to_nchw", [&] {
               return g_api.whcn_to_nchw(backend, whcn, nchw, c, h, w);
           });
#endif
}

bool VkAlikedNchwToWhcn(ggml_backend_t backend,
                        const ggml_tensor *nchw,
                        ggml_tensor *whcn,
                        int32_t c,
                        int32_t h,
                        int32_t w) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)nchw;
    (void)whcn;
    (void)c;
    (void)h;
    (void)w;
    return false;
#else
    EnsureResolved(backend);
    return g_api.nchw_to_whcn != nullptr &&
           CallVulkanNoThrow("nchw_to_whcn", [&] {
               return g_api.nchw_to_whcn(backend, nchw, whcn, c, h, w);
           });
#endif
}

bool VkAlikedDenseCopyWhcn(ggml_backend_t backend,
                           const ggml_tensor *src,
                           ggml_tensor *dst,
                           int32_t w,
                           int32_t h,
                           int32_t c) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)src;
    (void)dst;
    (void)w;
    (void)h;
    (void)c;
    return false;
#else
    EnsureResolved(backend);
    return g_api.dense_copy != nullptr && CallVulkanNoThrow("dense_copy", [&] {
               return g_api.dense_copy(backend, src, dst, w, h, c);
           });
#endif
}

bool VkAlikedClampInplace(ggml_backend_t backend,
                          ggml_tensor *data,
                          size_t count,
                          float min_value,
                          float max_value) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)data;
    (void)count;
    (void)min_value;
    (void)max_value;
    return false;
#else
    EnsureResolved(backend);
    return g_api.clamp != nullptr && CallVulkanNoThrow("clamp", [&] {
               return g_api.clamp(backend, data, count, min_value, max_value);
           });
#endif
}

bool VkAlikedL2NormInplace(ggml_backend_t backend,
                           ggml_tensor *data,
                           int32_t channels,
                           int32_t h,
                           int32_t w) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)data;
    (void)channels;
    (void)h;
    (void)w;
    return false;
#else
    EnsureResolved(backend);
    return g_api.l2norm != nullptr && CallVulkanNoThrow("l2norm", [&] {
               return g_api.l2norm(backend, data, channels, h, w);
           });
#endif
}

bool VkAlikedDeformConv2d(ggml_backend_t backend,
                          const ggml_tensor *input,
                          const ggml_tensor *offset,
                          const ggml_tensor *weight,
                          const ggml_tensor *bias,
                          ggml_tensor *output,
                          int32_t ic,
                          int32_t ih,
                          int32_t iw,
                          int32_t oc,
                          int32_t kh,
                          int32_t kw,
                          int32_t pad,
                          int32_t layout) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)input;
    (void)offset;
    (void)weight;
    (void)bias;
    (void)output;
    (void)ic;
    (void)ih;
    (void)iw;
    (void)oc;
    (void)kh;
    (void)kw;
    (void)pad;
    (void)layout;
    return false;
#else
    EnsureResolved(backend);
    return g_api.deform_conv != nullptr &&
           CallVulkanNoThrow("deform_conv", [&] {
               return g_api.deform_conv(backend, input, offset, weight, bias,
                                        output, ic, ih, iw, oc, kh, kw, pad,
                                        layout);
           });
#endif
}

bool VkAlikedRunDkd(ggml_backend_t backend,
                    const ggml_tensor *score_map,
                    int32_t h,
                    int32_t w,
                    int32_t radius,
                    int32_t top_k,
                    float scores_th,
                    int32_t n_limit,
                    ggml_tensor *keypoints_norm,
                    ggml_tensor *scores,
                    int32_t *out_count,
                    ggml_tensor *nms,
                    ggml_tensor *tmp_a,
                    ggml_tensor *tmp_b,
                    ggml_tensor *tmp_c,
                    ggml_tensor *block_keys,
                    ggml_tensor *block_indices,
                    ggml_tensor *indices_dev) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)score_map;
    (void)h;
    (void)w;
    (void)radius;
    (void)top_k;
    (void)scores_th;
    (void)n_limit;
    (void)keypoints_norm;
    (void)scores;
    (void)out_count;
    (void)nms;
    (void)tmp_a;
    (void)tmp_b;
    (void)tmp_c;
    (void)block_keys;
    (void)block_indices;
    (void)indices_dev;
    return false;
#else
    EnsureResolved(backend);
    return g_api.run_dkd != nullptr && CallVulkanNoThrow("dkd", [&] {
               return g_api.run_dkd(backend, score_map, h, w, radius, top_k,
                                    scores_th, n_limit, keypoints_norm, scores,
                                    out_count, nms, tmp_a, tmp_b, tmp_c,
                                    block_keys, block_indices, indices_dev);
           });
#endif
}

bool VkAlikedRunSddh(ggml_backend_t backend,
                     const ggml_tensor *feature_map,
                     int32_t dim,
                     int32_t h,
                     int32_t w,
                     const ggml_tensor *keypoints_norm,
                     int32_t count,
                     int32_t kernel_size,
                     int32_t n_pos,
                     const ggml_tensor *offset_0_w,
                     const ggml_tensor *offset_0_b,
                     const ggml_tensor *offset_2_w,
                     const ggml_tensor *offset_2_b,
                     const ggml_tensor *sf_conv_w,
                     const ggml_tensor *agg_weights,
                     ggml_tensor *workspace,
                     ggml_tensor *descriptors) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)feature_map;
    (void)dim;
    (void)h;
    (void)w;
    (void)keypoints_norm;
    (void)count;
    (void)kernel_size;
    (void)n_pos;
    (void)offset_0_w;
    (void)offset_0_b;
    (void)offset_2_w;
    (void)offset_2_b;
    (void)sf_conv_w;
    (void)agg_weights;
    (void)workspace;
    (void)descriptors;
    return false;
#else
    EnsureResolved(backend);
    return g_api.run_sddh != nullptr && CallVulkanNoThrow("sddh", [&] {
               return g_api.run_sddh(backend, feature_map, dim, h, w,
                                     keypoints_norm, count, kernel_size, n_pos,
                                     offset_0_w, offset_0_b, offset_2_w,
                                     offset_2_b, sf_conv_w, agg_weights,
                                     workspace, descriptors);
           });
#endif
}

bool VkAlikedUpsampleBilinear(ggml_backend_t backend,
                              const ggml_tensor *input,
                              ggml_tensor *output,
                              int32_t ic,
                              int32_t ih,
                              int32_t iw,
                              int32_t out_h,
                              int32_t out_w) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    (void)input;
    (void)output;
    (void)ic;
    (void)ih;
    (void)iw;
    (void)out_h;
    (void)out_w;
    return false;
#else
    EnsureResolved(backend);
    return g_api.upsample != nullptr && CallVulkanNoThrow("upsample", [&] {
               return g_api.upsample(backend, input, output, ic, ih, iw, out_h,
                                     out_w);
           });
#endif
}

bool VkAlikedQueueIdle(ggml_backend_t backend) {
#if !defined(AICORE_VULKAN_ALIKED)
    (void)backend;
    return false;
#else
    EnsureResolved(backend);
    if (g_api.queue_idle != nullptr) {
        return CallVulkanNoThrow("queue_idle",
                                 [&] { return g_api.queue_idle(backend); });
    }
    if (backend != nullptr) {
        return CallVulkanNoThrow("backend_synchronize", [&] {
            ggml_backend_synchronize(backend);
            return true;
        });
    }
    return false;
#endif
}

}  // namespace lightglue::aliked_internal
