// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// PointPillar — full detector (pcdet/models/detectors/pointpillar.py).
//
// Forward orchestration:
//   1. Voxelize the point cloud (CPU)
//   2. PillarVFE (CPU) -> pillar_features [V, 64]
//   3. PointPillarScatter (CPU) -> BEV spatial_features [nx, ny, 64, 1]
//   4. Build ggml graph: BaseBEVBackbone -> AnchorHeadSingle
//   5. Allocate, upload BEV, compute, read back cls/box/dir
//   6. CPU: anchor decode -> direction classify -> sigmoid -> max-class
//   7. CPU: 3D NMS
//   8. Return detections
#pragma once

#include "../model.h"
#include "anchor_head.h"
#include "anchor_head_multi.h"
#include "bev_backbone.h"
#include "center_head.h"
#include "pillar_vfe.h"
#ifdef AICORE_OPENPCDET_CUDA_EXT
#include "../extensions/vfe_scatter_cuda.h"
#endif

#include <string>
#include <vector>

namespace openpcdet {

struct pointpillar : model {
    pillar_vfe_weights vfe_w;
    bev_backbone_weights bb_w;
    anchor_head_weights head_w;
    anchor_multi_weights multi_head_w;
    center_head_weights center_head_w;
#ifdef AICORE_OPENPCDET_CUDA_EXT
    // Flattened f32 copies of the PFN layers for the CUDA VFE/scatter kernel
    // (weights stay on the host file context; see weight_to_f32).
    std::vector<ext::gpu_pfn_layer> gpu_vfe_layers;
#endif

    // Persistent ggml graph, built once on the first forward() and reused on
    // subsequent frames. Only the spatial input data changes between frames, so
    // the (expensive) graph construction / allocation is amortized to a single
    // call. Weight tensors live in the stable device_ctx, so reusing the graph
    // across frames is numerically identical to rebuilding it.
    //
    // AICore port note: pillar scatter runs in-graph on stock ops for every
    // backend: a zero-filled canvas (ggml_fill) receives each pillar's features
    // through ggml_set_rows (destination rows are unique per pillar, so the
    // assignment semantics are exact). The canvas carries one trailing "trash"
    // row that absorbs the padded slots of frames with fewer pillars than the
    // graph capacity, so padded writes never race a real cell.
    ggml_context *graph_ctx = nullptr;
    ggml_cgraph *graph_gf = nullptr;
    ggml_tensor *graph_input = nullptr;
    ggml_tensor *graph_sparse_features = nullptr;  // [C_bev, capacity]
    ggml_tensor *graph_scatter_rows = nullptr;     // I64 [capacity], cell index
    // True when the persistent graph consumes the dense BEV [nx, ny, C, 1]
    // tensor directly (CUDA-ext VFE/scatter writes it D2D per frame). False
    // keeps the stock in-graph set_rows scatter topology (CPU/Vulkan and the
    // host path — byte-identical to the non-ext build).
    bool graph_dense_direct = false;
    bool graph_built = false;
    // Release the persistent graph (also the grow path: dynamic-pillar sweeps
    // can exceed max_voxels, so the graph is rebuilt with a larger canvas).
    void release_graph();
    head_outputs cache_head;
    std::vector<anchor_multi_single_outputs> cache_multi;
    std::vector<center_single_outputs> cache_center;
    std::vector<ggml_tensor *> cache_taps;
    // Frame-invariant anchor grid (depends only on h + feature-map size).
    anchor_grid cached_anchors;
    bool anchors_cached = false;

    bool load(const std::string &gguf_path,
              const std::string &device,
              int n_threads) override;
    bool forward(const float *points,
                 int32_t n_points,
                 int32_t point_dim,
                 std::vector<detection> &out_dets,
                 const tap_sink &sink = {}) override;
    ~pointpillar() override;
};

}  // namespace openpcdet
