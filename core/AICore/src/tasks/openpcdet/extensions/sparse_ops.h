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
#include <memory>
#include <string>
#include <vector>

namespace openpcdet::ext {

// Coordinate order matches spconv: [batch, z, y, x] for 3-D and
// [batch, y, x] for 2-D. Features are row-major [active, channels].
struct sparse_tensor {
    int ndim = 3;
    std::vector<int32_t> spatial_shape;
    std::vector<int32_t> coordinates;
    std::vector<float> features;
    int channels = 0;

    int64_t active_count() const;
    bool validate(std::string& error) const;
};

struct sparse_conv_params {
    std::vector<int32_t> kernel_size;
    std::vector<int32_t> stride;
    std::vector<int32_t> padding;
    std::vector<int32_t> dilation;
    bool submanifold = false;
    bool transposed = false;
};

struct sparse_indice_map {
    int ndim = 3;
    std::vector<int32_t> spatial_shape;
    std::vector<int32_t> coordinates;
    // SparseInverseConv restores this exact active set instead of generating
    // every coordinate reachable by a dense transposed convolution.
    std::vector<int32_t> input_spatial_shape;
    std::vector<int32_t> input_coordinates;
    // GGML layout [kernel_volume, output_active]. Missing neighbors are -1.
    std::vector<int32_t> indices;
    int32_t kernel_volume = 0;
    // Non-negative when this layer has identical SubM geometry to an earlier
    // layer and can reuse that layer's uploaded GGML index tensor.
    int32_t reuse_of = -1;
    // Compact (CSR-style) rulebook: only the valid neighbors, grouped by
    // kernel offset. The gathered features feed one GEMM per kernel block
    // (no zero rows in the K dimension) and are merged back with a scatter
    // sum, mirroring spconv's rulebook execution.
    std::vector<int32_t>
            compact_indices;  // [sum(valid_k)] input row per valid (k, j)
    std::vector<int32_t>
            compact_offsets;  // [kernel_volume + 1] start of each kernel group
    std::vector<int32_t>
            scatter_rows;  // row-major [output_active, kernel_volume] compact
                           // column per (j, k), -1 invalid

    // CUDA rulebooks (sparse_rulebook_cuda.cu, AICore port of the upstream
    // OpenPCDet-GGML GPU rulebook) keep their large arrays resident across
    // map generation, graph upload and HeightCompression. `device_owner`
    // carries the opaque CUDA allocation lifetime without leaking CUDA
    // headers into portable model code. Host arrays may be omitted in the
    // non-capture fast path.
    std::shared_ptr<void> device_owner;
    const int32_t* device_coordinates = nullptr;
    const int32_t* device_indices = nullptr;
    const int32_t* device_compact_offsets = nullptr;
    const int32_t* device_compact_indices = nullptr;
    const int32_t* device_scatter_rows = nullptr;
    int64_t device_output_active_count = 0;

    int64_t output_active_count() const;
    bool has_device_rulebook() const {
        return device_owner && device_coordinates && device_compact_offsets &&
               device_compact_indices && device_scatter_rows &&
               device_output_active_count > 0;
    }
};

// Coordinate/indice generation is backend-independent and contains no feature
// arithmetic. The returned map can be cached for repeated sparse blocks.
bool build_sparse_indice_map(const sparse_tensor& input,
                             const sparse_conv_params& params,
                             sparse_indice_map& map,
                             std::string& error);

bool build_sparse_inverse_indice_map(const sparse_indice_map& forward,
                                     sparse_indice_map& inverse,
                                     std::string& error);

// Build the feature-arithmetic part of sparse convolution in a GGML graph.
// Two lowering strategies share the same inputs and produce bit-compatible
// results:
//  - dense lowering (default, every backend): stock-op get_rows + reshape +
//    mul_mat over the [kernel_volume, output_active] dense rulebook with -1
//    entries remapped to an appended zero input row. The K reduction happens
//    inside the GEMM, so no scatter pass is needed.
//  - compact lowering (CUDA-ext path, use_compact=true): fused
//    ggml_sparse_gather of only the valid neighbors (compacted), one GEMM
//    per kernel group and a multi-block ggml_sparse_scatter_add_n merge,
//    mirroring spconv's rulebook execution. Typical LiDAR sparsity (~25%
//    valid neighbors) cuts the gather traffic and GEMM work ~4x; the many
//    small GEMMs would be inefficient on CPUs, so CPU keeps the dense form.
//
// features: [in_channels, input_active]  (dense lowering needs the zero row;
//            compact lowering consumes the compact_indices upload)
// dense_indices: I32 [kernel_volume * output_active] flat, -1 pre-remapped
// compact_indices: I32 [sum(valid_k)] valid-neighbor input rows (may be null
//                  on the dense lowering)
// scatter_rows: I32 row-major [output_active, kernel_volume] compact column
//               per (j, k), -1 invalid (may be null on the dense lowering)
// weights: [in_channels * kernel_volume, out_channels]
ggml_tensor* build_sparse_convolution(ggml_context* ctx,
                                      ggml_tensor* features,
                                      ggml_tensor* dense_indices,
                                      ggml_tensor* compact_indices,
                                      ggml_tensor* scatter_rows,
                                      const sparse_indice_map& map,
                                      ggml_tensor* weights,
                                      ggml_tensor* bias = nullptr,
                                      bool use_compact = false);

// Remap the map's rulebook (-1 entries) into get_rows-safe indices: -1
// becomes `zero_row` (the appended zero feature row). The output layout is
// byte-identical to map.indices otherwise.
void remap_dense_indices(const sparse_indice_map& map,
                         int32_t zero_row,
                         std::vector<int32_t>& out);

// Upload the sparse rulebook tensor to the backend.
inline void upload_dense_indices(ggml_tensor* dense,
                                 const std::vector<int32_t>& dense_data) {
    if (dense && dense->buffer)
        ggml_backend_tensor_set(dense, dense_data.data(), 0,
                                dense_data.size() * sizeof(int32_t));
}

// LSD radix sort over non-negative uint64 keys (16-bit digits, 4 passes).
// Stable: equal keys keep their relative order. Used by the rulebook output
// candidate sort and by focal_check_repeat's packed (key, row) sort.
void radix_sort_u64(std::vector<uint64_t>& values);

}  // namespace openpcdet::ext
