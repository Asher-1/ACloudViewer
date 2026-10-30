#pragma once

#include "sparse_ops.h"

#include <string>

namespace openpcdet::ext {

// Build the same sorted output coordinates and stable compact rulebook as the
// CPU reference. Intermediate candidate generation, sort/unique, hash lookup,
// and compaction execute on CUDA; the resulting metadata remains host-owned so
// the existing GGML graph builder can size its dynamic sparse tensors.
bool build_sparse_indice_map_cuda(const sparse_tensor & input,
                                  const sparse_conv_params & params,
                                  sparse_indice_map & map,
                                  std::string & error,
                                  const int32_t * input_coordinates_device = nullptr,
                                  int32_t input_active_count = 0,
                                  bool retain_host_arrays = true);

// Packs an exact device rulebook into tensors sized for a persistent
// high-water-capacity graph. Compact kernel groups are rebased to the
// capacity offsets and unused output rows remain -1 in the scatter table.
bool upload_sparse_indice_map_cuda(const sparse_indice_map & exact,
                                   const sparse_indice_map & capacity,
                                   int32_t * compact_device,
                                   int32_t * scatter_device,
                                   int32_t * dense_device,
                                   std::string & error);

// AICore addition for the stock-op sparse graph: packs the device rulebook
// into a get_rows-safe dense index tensor by remapping every -1 entry to
// `zero_row` (the appended zero feature row) entirely on device. Mirrors
// ext::remap_dense_indices on the host. `capacity_pair_count` is the full
// kv * output_capacity extent of the destination tensor; padding beyond the
// exact pairs is written to zero_row so freshly allocated capacity tensors
// never gather out-of-bounds rows.
bool remap_dense_indices_cuda(const sparse_indice_map & exact, int32_t zero_row,
                              int32_t * dense_device, int64_t capacity_pair_count,
                              std::string & error);

bool synchronize_sparse_rulebook_cuda(std::string & error);

} // namespace openpcdet::ext
