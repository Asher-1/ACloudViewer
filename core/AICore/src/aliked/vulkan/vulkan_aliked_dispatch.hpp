// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>

namespace lightglue::aliked_internal {

// Runtime-resolved ALIKED Vulkan API (ggml dynamic backend: no link-time dep on
// libggml-vulkan.so). Symbols come from ggml_backend_reg_get_proc_address or dlsym.
bool VkAlikedAvailable(ggml_backend_t backend);
bool VkAlikedWhcnToNchw(ggml_backend_t backend, const ggml_tensor *whcn,
                       ggml_tensor *nchw, int32_t c, int32_t h, int32_t w);
bool VkAlikedNchwToWhcn(ggml_backend_t backend, const ggml_tensor *nchw,
                        ggml_tensor *whcn, int32_t c, int32_t h, int32_t w);
bool VkAlikedDenseCopyWhcn(ggml_backend_t backend, const ggml_tensor *src,
                           ggml_tensor *dst, int32_t w, int32_t h, int32_t c);
bool VkAlikedClampInplace(ggml_backend_t backend, ggml_tensor *data, size_t count,
                          float min_value, float max_value);
bool VkAlikedL2NormInplace(ggml_backend_t backend, ggml_tensor *data,
                           int32_t channels, int32_t h, int32_t w);
bool VkAlikedDeformConv2d(ggml_backend_t backend, const ggml_tensor *input,
                          const ggml_tensor *offset, const ggml_tensor *weight,
                          const ggml_tensor *bias, ggml_tensor *output, int32_t ic,
                          int32_t ih, int32_t iw, int32_t oc, int32_t kh, int32_t kw,
                          int32_t pad, int32_t layout);
bool VkAlikedRunDkd(ggml_backend_t backend, const ggml_tensor *score_map, int32_t h,
                    int32_t w, int32_t radius, int32_t top_k, float scores_th,
                    int32_t n_limit, ggml_tensor *keypoints_norm, ggml_tensor *scores,
                    int32_t *out_count, ggml_tensor *nms, ggml_tensor *tmp_a,
                    ggml_tensor *tmp_b, ggml_tensor *tmp_c, ggml_tensor *block_keys,
                    ggml_tensor *block_indices, ggml_tensor *indices_dev);
bool VkAlikedRunSddh(ggml_backend_t backend, const ggml_tensor *feature_map,
                     int32_t dim, int32_t h, int32_t w,
                     const ggml_tensor *keypoints_norm, int32_t count,
                     int32_t kernel_size, int32_t n_pos,
                     const ggml_tensor *offset_0_w, const ggml_tensor *offset_0_b,
                     const ggml_tensor *offset_2_w, const ggml_tensor *offset_2_b,
                     const ggml_tensor *sf_conv_w, const ggml_tensor *agg_weights,
                     ggml_tensor *workspace, ggml_tensor *descriptors);
bool VkAlikedUpsampleBilinear(ggml_backend_t backend, const ggml_tensor *input,
                              ggml_tensor *output, int32_t ic, int32_t ih, int32_t iw,
                              int32_t out_h, int32_t out_w);
// Drain custom VkAliked + ggml compute queues between extract passes.
bool VkAlikedQueueIdle(ggml_backend_t backend);

}  // namespace lightglue::aliked_internal
