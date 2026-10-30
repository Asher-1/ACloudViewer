#pragma once

#include "../backend.h"
#include "../model.h"
#include "sparse_backbone.h"

#include <vector>

namespace openpcdet {

struct voxel_backbone_result {
    std::vector<ext::sparse_indice_map> maps;
    std::vector<std::vector<float>> layer_features;
    std::vector<float> spatial_features;
    int width = 0;
    int height = 0;
    // True when HeightCompression already wrote the dense BEV straight into
    // the caller's CUDA-resident tensor (spatial_hwc_device); spatial_features
    // stays empty in that case.
    bool spatial_device_ready = false;
};

// Sparse execution state. Capacity grows monotonically to the largest
// active/output counts observed by this model instance. Exact rulebooks are
// remapped and packed into the dense index tensors each frame, while graph
// topology and gallocr placement survive across forwards.
//
// AICore port note: the cached graph uses the stock-op dense lowering
// (get_rows -> reshape -> mul_mat -> bias) on CPU/Vulkan. With the CUDA
// extension compiled in and a CUDA backend, the cached graph switches to the
// upstream compact lowering (fused sparse_gather + per-kernel mul_mat +
// multi-block sparse_scatter_add_n), so the cache additionally carries the
// per-layer compact/scatter index tensors and the per-kernel compact-offset
// high-water marks.
struct sparse_graph_cache {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * input = nullptr;   // [point_dim, input_capacity + 1]; last row zero
    ggml_tensor * output = nullptr;  // [out_channels, output_capacity(last)]
    std::vector<ggml_tensor *> dense_tensors;
    std::vector<ggml_tensor *> compact_tensors;
    std::vector<ggml_tensor *> scatter_tensors;
    std::vector<ext::sparse_indice_map> capacity_maps; // shape metadata only
    std::vector<int64_t> output_capacity;              // per map, >= frame Nout
    std::vector<ggml_backend_buffer_t> side_buffers;
    int64_t input_capacity = 0;

    ~sparse_graph_cache();
    void clear();
    sparse_graph_cache() = default;
    sparse_graph_cache(const sparse_graph_cache &) = delete;
    sparse_graph_cache & operator=(const sparse_graph_cache &) = delete;
};

bool run_voxel_backbone(engine_backend & backend, const hparams & hp,
                        const sparse_backbone_weights & weights,
                        const float * points, int32_t point_count, int32_t point_dim,
                        bool capture_layers, voxel_backbone_result & result,
                        const tap_sink & sink, std::string & error,
                        float * spatial_hwc_device = nullptr,
                        sparse_graph_cache * cache = nullptr,
                        bool spatial_device_whc = false);

} // namespace openpcdet
