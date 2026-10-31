// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// FocalSparseConv (pcdet.models.backbones_3d.focal_sparse_conv) host
// implementation. split_voxels / check_repeat / combine_out are inherently
// sequential gather/scatter over the active sparse voxel set, so they run on
// the host; the SubM convs reuse ext::sparse_convolution (CPU).
//
// Semantic parity target: pcdet focal_sparse_conv.py / focal_sparse_utils.py.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "sparse_ops.h"

namespace openpcdet::ext {

// Like build_sparse_indice_map, but caches the result per (coordinate set,
// spatial shape, convolution params). Focal layers run several SubM convs
// over the same active set (e.g. conv2.1/conv2.2 and conv_imp vs the previous
// SubM layer), so the cached path skips redundant rulebook generation. The
// cache is keyed by a content hash and verified against the full coordinates,
// so a collision cannot produce a wrong map.
bool build_sparse_indice_map_cached(const sparse_tensor& input,
                                    const sparse_conv_params& params,
                                    sparse_indice_map& map,
                                    std::string& error);

// Dense (compressed [Cout, K*Cin]) -> strided [K, Cin, Cout] sparse conv
// weight. `weights` is the GGML [K*Cin, Cout] layout actually stored in the
// GGUF.
std::vector<float> streamed_sparse_conv_weight(
        const std::vector<float>& weights,
        int kernel_volume,
        int in_channels,
        int out_channels);

// Fuse a folded BN (scale/bias) into a sparse conv bias vector.
// Returns bias of length out_channels (plus conv bias if provided).
std::vector<float> sparse_conv_bias(const std::vector<float>* conv_bias,
                                    const std::vector<float>& bn_scale,
                                    const std::vector<float>& bn_bias);

// kernel_offsets for a symmetric kernel: lexicographic z/y/x in
// [-half, +half]^3, matching pcdet focal_sparse_utils.get_offsets.
std::vector<std::array<int32_t, 3>> focal_kernel_offsets(int kernel_size);

// check_repeat: sort by the spconv linearized coordinate, then merge rows that
// land on the same voxel by summing their features (and, if provided, summing
// then averaging the mask). Mirrors focal_sparse_utils.check_repeat.
// Returns the deduplicated features [n, channels], coordinates [n, 4] and
// (if features_add != null) the averaged mask [n].
void focal_check_repeat(const std::vector<float>& features,
                        const std::vector<int32_t>& coordinates,
                        int channels,
                        bool flip_first,
                        const std::vector<float>* features_add,
                        std::vector<float>& out_features,
                        std::vector<int32_t>& out_coordinates,
                        std::vector<float>* out_add);

// split_voxels: select foreground/background voxels from the predicted 3-D
// importance map, expand foreground voxels to their kernel neighbors and merge
// duplicates. Mirrors focal_sparse_utils.split_voxels for one batch element.
//
//   features      [n, channels] active features of batch b
//   coordinates   [n, 4] active coords (batch,z,y,x) of batch b
//   imps          [n, kernel_volume] predicted importance (conv_imp output)
//   kernel_offsets[kernel_volume, 3] kernel neighbor offsets
//   spatial_shape [3] (z,y,x) used to clip expanded neighbors
//
// Outputs fore/back features+coords and the per-row foreground kernel mask.
bool focal_split_voxels(
        const std::vector<float>& features,
        const std::vector<int32_t>& coordinates,
        int channels,
        const std::vector<float>& imps,
        int kernel_volume,
        int kernel_size,
        const std::vector<std::array<int32_t, 3>>& kernel_offsets,
        const std::vector<int32_t>& spatial_shape,
        bool mask_multi,
        bool topk,
        float threshold,
        std::vector<float>& fore_features,
        std::vector<int32_t>& fore_coordinates,
        std::vector<float>& back_features,
        std::vector<int32_t>& back_coordinates,
        std::vector<float>& mask_kernel_fore,
        std::string& error);

// combine_out: concatenate fore/back and deduplicate per batch element.
bool focal_combine_out(const std::vector<float>& fore_features,
                       const std::vector<int32_t>& fore_coordinates,
                       const std::vector<float>& back_features,
                       const std::vector<int32_t>& back_coordinates,
                       int channels,
                       int batch_size,
                       std::vector<float>& features,
                       std::vector<int32_t>& coordinates,
                       std::string& error);

}  // namespace openpcdet::ext