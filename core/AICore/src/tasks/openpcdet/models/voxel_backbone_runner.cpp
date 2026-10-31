// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "voxel_backbone_runner.h"

#include "../ops/voxelization.h"

#ifdef AICORE_OPENPCDET_CUDA_EXT
#include "../extensions/height_compression_cuda.h"
#include "../extensions/sparse_rulebook_cuda.h"
#endif

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace openpcdet {

namespace {

// Debug-only stage profiler (stderr). Mirrors the upstream OpenPCDet-GGML
// OPCPD_PROFILE switch: diagnostics only, default off, never changes task
// behavior. The public timing contract stays aicore_pipeline_timings.
class stage_timer {
public:
    stage_timer() : active_(std::getenv("OPCPD_PROFILE") != nullptr) {}
    void lap(const char *name) {
        if (!active_) return;
        const auto now = std::chrono::high_resolution_clock::now();
        const double ms =
                std::chrono::duration<double, std::milli>(now - last_).count();
        std::fprintf(stderr, "[voxel_backbone] %-22s %8.2f ms\n", name, ms);
        last_ = now;
    }

private:
    bool active_;
    std::chrono::high_resolution_clock::time_point last_ =
            std::chrono::high_resolution_clock::now();
};

std::vector<float> readback_f32(ggml_tensor *tensor) {
    std::vector<float> output(static_cast<size_t>(ggml_nelements(tensor)));
    if (ggml_is_contiguous(tensor)) {
        ggml_backend_tensor_get(tensor, output.data(), 0,
                                output.size() * sizeof(float));
        return output;
    }
    const int64_t rows = tensor->ne[3] * tensor->ne[2] * tensor->ne[1];
    for (int64_t row = 0; row < rows; ++row) {
        ggml_backend_tensor_get(tensor, output.data() + row * tensor->ne[0],
                                row * tensor->nb[1],
                                tensor->ne[0] * sizeof(float));
    }
    return output;
}

// Grow the cached graph capacity so the largest observed input/output counts
// fit. Returns true when the cached graph must be rebuilt (first use, layer
// count change, or any capacity growth).
bool grow_sparse_capacity(const std::vector<ext::sparse_indice_map> &exact,
                          int64_t input_active,
                          sparse_graph_cache &cache) {
    bool grew = cache.ctx == nullptr ||
                cache.capacity_maps.size() != exact.size() ||
                input_active > cache.input_capacity;
    std::vector<ext::sparse_indice_map> capacity(exact.size());
    std::vector<int64_t> dense_capacity(exact.size(), 0);
    for (size_t i = 0; i < exact.size(); ++i) {
        const auto &current = exact[i];
        auto &target = capacity[i];
        if (current.reuse_of >= 0) {
            target = capacity[static_cast<size_t>(current.reuse_of)];
            target.reuse_of = current.reuse_of;
            dense_capacity[i] =
                    dense_capacity[static_cast<size_t>(current.reuse_of)];
            continue;
        }
        const ext::sparse_indice_map *previous =
                i < cache.capacity_maps.size() &&
                                cache.capacity_maps[i].reuse_of < 0
                        ? &cache.capacity_maps[i]
                        : nullptr;
        target.ndim = current.ndim;
        target.spatial_shape = current.spatial_shape;
        target.kernel_volume = current.kernel_volume;
        target.reuse_of = -1;
        const int64_t old_output =
                previous ? previous->output_active_count() : 0;
        target.device_output_active_count =
                std::max<int64_t>(old_output, current.output_active_count());
        grew = grew || target.device_output_active_count > old_output;
        // Per-kernel compact-offset high-water marks: consumed by the CUDA
        // rulebook packing (upload_sparse_indice_map_cuda) and kept at zero
        // capacity sizes for the stock dense graph, which only reads
        // device_output_active_count through output_active_count().
        target.compact_offsets.assign(
                static_cast<size_t>(target.kernel_volume) + 1, 0);
        for (int32_t k = 0; k < target.kernel_volume; ++k) {
            const int32_t count =
                    current.compact_offsets[k + 1] - current.compact_offsets[k];
            const int32_t old_count =
                    previous ? previous->compact_offsets[k + 1] -
                                       previous->compact_offsets[k]
                             : 0;
            const int32_t capacity_count = std::max(count, old_count);
            target.compact_offsets[k + 1] =
                    target.compact_offsets[k] + capacity_count;
            grew = grew || capacity_count > old_count;
        }
        dense_capacity[i] = target.device_output_active_count;
    }
    cache.input_capacity = std::max(cache.input_capacity, input_active);
    cache.capacity_maps = std::move(capacity);
    cache.output_capacity = std::move(dense_capacity);
    return grew;
}

}  // namespace

sparse_graph_cache::~sparse_graph_cache() { clear(); }

void sparse_graph_cache::clear() {
    for (auto &b : side_buffers) ggml_backend_buffer_free(b);
    side_buffers.clear();
    if (ctx) ggml_free(ctx);
    if (galloc) ggml_gallocr_free(galloc);
    ctx = nullptr;
    graph = nullptr;
    galloc = nullptr;
    input = nullptr;
    output = nullptr;
    dense_tensors.clear();
    compact_tensors.clear();
    scatter_tensors.clear();
    capacity_maps.clear();
    output_capacity.clear();
    input_capacity = 0;
}

bool run_voxel_backbone(engine_backend &backend,
                        const hparams &hp,
                        const sparse_backbone_weights &weights,
                        const float *points,
                        int32_t point_count,
                        int32_t point_dim,
                        bool capture_layers,
                        voxel_backbone_result &result,
                        const tap_sink &sink,
                        std::string &error,
                        float *spatial_hwc_device,
                        sparse_graph_cache *cache,
                        bool spatial_device_whc) {
    result = {};
    if (point_count < 0 || (point_count > 0 && !points) ||
        point_dim != hp.num_point_features) {
        error = "invalid sparse voxel point buffer or feature dimension";
        return false;
    }
    if (point_count == 0) return true;

    stage_timer timer;
    voxelization_params params;
    params.point_cloud_range = hp.point_cloud_range;
    params.voxel_size = hp.voxel_size;
    params.grid_size = hp.grid_size;
    params.max_points_per_voxel = hp.max_points_per_voxel;
    params.max_voxels = hp.max_voxels;
    mean_voxel_output voxels;
    if (!voxelize_mean(params, points, point_count, point_dim, voxels, error))
        return false;
    const int active = voxels.actual_voxels;
    if (active == 0) return true;

    std::vector<float> features = std::move(voxels.features);
    std::vector<int32_t> coordinates = std::move(voxels.coords);
    if (sink) sink("mean_vfe", features.data(), active, point_dim);
    timer.lap("voxelize+mean");

    const std::vector<int32_t> sparse_shape = {
            hp.grid_size[2] + 1, hp.grid_size[1], hp.grid_size[0]};
    // Host rulebook copies are only needed when the results leave the device:
    // taps/capture read them on the host, non-CUDA backends upload from them,
    // and the host HeightCompression path walks the host coordinates.
    const bool retain_host_rulebooks = capture_layers || sink ||
                                       !backend.is_cuda() ||
                                       spatial_hwc_device == nullptr;
    if (!build_sparse_backbone_maps(coordinates, sparse_shape, point_dim,
                                    weights, result.maps, error,
                                    backend.is_cuda(), retain_host_rulebooks))
        return false;
    timer.lap("indice maps");

    // The cached fast path (persistent graph + capacity-grown rulebook
    // tensors) applies to every non-tapped CUDA frame; other backends and
    // tapped runs rebuild the graph from the exact frame shapes.
    const bool use_cache =
            cache && backend.is_cuda() && !capture_layers && !sink;
#ifdef AICORE_OPENPCDET_CUDA_EXT
    // CUDA-ext compact lowering (upstream fused sparse gather + per-kernel
    // GEMM + multi-block scatter add). CPU/Vulkan keep the stock dense graph.
    // Quantized sparse weights keep the dense lowering (the per-kernel weight
    // views cannot slice Q8_0 row-blocks), so the whole model stays dense and
    // the upload uses the remapped dense indices.
    bool use_compact = backend.is_cuda();
    for (const auto &layer : weights.layers) {
        if (ggml_is_quantized(layer.weight->type)) {
            use_compact = false;
            break;
        }
    }
#else
    const bool use_compact = false;
#endif
    if (use_cache && grow_sparse_capacity(result.maps, active, *cache) &&
        cache->ctx) {
        auto capacity_maps = std::move(cache->capacity_maps);
        auto output_capacity = std::move(cache->output_capacity);
        const int64_t input_capacity = cache->input_capacity;
        cache->clear();
        cache->capacity_maps = std::move(capacity_maps);
        cache->output_capacity = std::move(output_capacity);
        cache->input_capacity = input_capacity;
    }
    const auto &graph_maps = use_cache ? cache->capacity_maps : result.maps;
    constexpr size_t graph_nodes = 8192;
    ggml_context *ctx = use_cache ? cache->ctx : nullptr;
    ggml_cgraph *graph = use_cache ? cache->graph : nullptr;
    ggml_tensor *input = use_cache ? cache->input : nullptr;
    ggml_tensor *output = use_cache ? cache->output : nullptr;
    std::vector<ggml_tensor *> dense_tensors;
    std::vector<ggml_tensor *> compact_tensors;
    std::vector<ggml_tensor *> scatter_tensors;
    std::vector<ggml_tensor *> layer_tensors;
    if (!ctx) {
        ggml_init_params graph_params = {
                ggml_tensor_overhead() * graph_nodes +
                        ggml_graph_overhead_custom(graph_nodes, false),
                nullptr,
                true,
        };
        ctx = ggml_init(graph_params);
        if (!ctx) {
            error = "sparse graph context init failed";
            return false;
        }
        const int64_t graph_input_active =
                use_cache ? cache->input_capacity : active;
        // The input carries one extra row at index graph_input_active: the
        // permanent zero row that every -1 rulebook entry is remapped to.
        // (Consumed by the dense lowering; the compact lowering never reads
        // it, so keeping the extra row is layout-neutral.)
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, point_dim,
                                   graph_input_active + 1);
        ggml_set_input(input);
        dense_tensors.reserve(graph_maps.size());
        if (use_compact) {
            compact_tensors.reserve(graph_maps.size());
            scatter_tensors.reserve(graph_maps.size());
        }
        for (size_t index = 0; index < graph_maps.size(); ++index) {
            const auto &map = graph_maps[index];
            if (map.reuse_of >= 0) {
                if (map.reuse_of >= static_cast<int32_t>(index)) {
                    error = "invalid sparse indice map reuse";
                    ggml_free(ctx);
                    return false;
                }
                dense_tensors.push_back(dense_tensors[map.reuse_of]);
                if (use_compact) {
                    compact_tensors.push_back(compact_tensors[map.reuse_of]);
                    scatter_tensors.push_back(scatter_tensors[map.reuse_of]);
                }
            } else {
                const int64_t out_capacity =
                        use_cache ? cache->output_capacity[index]
                                  : map.output_active_count();
                ggml_tensor *dense = ggml_new_tensor_1d(
                        ctx, GGML_TYPE_I32, map.kernel_volume * out_capacity);
                ggml_set_input(dense);
                dense_tensors.push_back(dense);
                if (use_compact) {
                    const int32_t kv = map.kernel_volume;
                    // Capacity compact offsets: per-kernel high-water marks
                    // (frame counts may grow across frames; the cached graph
                    // is rebuilt when they do).
                    const auto &capacity_map = graph_maps[index];
                    ggml_tensor *compact = ggml_new_tensor_1d(
                            ctx, GGML_TYPE_I32,
                            capacity_map.compact_offsets.empty()
                                    ? 0
                                    : static_cast<int64_t>(
                                              capacity_map
                                                      .compact_offsets[kv]));
                    ggml_set_input(compact);
                    ggml_tensor *scatter = ggml_new_tensor_2d(
                            ctx, GGML_TYPE_I32, out_capacity, kv);
                    ggml_set_input(scatter);
                    compact_tensors.push_back(compact);
                    scatter_tensors.push_back(scatter);
                }
            }
        }
        output = build_sparse_backbone_graph(
                ctx, input, dense_tensors, compact_tensors, scatter_tensors,
                graph_maps, weights,
                (capture_layers || sink) ? &layer_tensors : nullptr,
                use_compact);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(graph, output);
        for (ggml_tensor *tensor : layer_tensors)
            ggml_build_forward_expand(graph, tensor);
        ggml_gallocr_t graph_allocator = backend.galloc;
        if (use_cache) {
            cache->galloc = ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(backend.be));
            graph_allocator = cache->galloc;
        }
        if (!graph_allocator ||
            !ggml_gallocr_alloc_graph(graph_allocator, graph)) {
            error = "sparse graph allocation failed";
            ggml_free(ctx);
            return false;
        }
        if (use_cache) {
            cache->ctx = ctx;
            cache->graph = graph;
            cache->input = input;
            cache->output = output;
            cache->dense_tensors = dense_tensors;
            cache->compact_tensors = compact_tensors;
            cache->scatter_tensors = scatter_tensors;
        }
    } else {
        dense_tensors = cache->dense_tensors;
        if (use_compact) {
            compact_tensors = cache->compact_tensors;
            scatter_tensors = cache->scatter_tensors;
        }
    }
    // The quantized sparse graph does not reference every rulebook tensor, so
    // gallocr leaves those without buffers and the upload below would receive
    // NULL pointers. Give the missing ones a dedicated side buffer.
    std::vector<ggml_backend_buffer_t> side_buffers;
    {
        bool any_missing = false;
        for (size_t index = 0; index < graph_maps.size() && !any_missing;
             ++index) {
            for (ggml_tensor *t :
                 {dense_tensors[index],
                  use_compact ? compact_tensors[index] : nullptr,
                  use_compact ? scatter_tensors[index] : nullptr}) {
                if (t && !t->buffer) any_missing = true;
            }
        }
        if (any_missing) {
            size_t total = 0;
            for (size_t index = 0; index < graph_maps.size(); ++index) {
                for (ggml_tensor *t :
                     {dense_tensors[index],
                      use_compact ? compact_tensors[index] : nullptr,
                      use_compact ? scatter_tensors[index] : nullptr}) {
                    if (t && !t->buffer) total += GGML_PAD(ggml_nbytes(t), 256);
                }
            }
            ggml_backend_buffer_t side =
                    ggml_backend_alloc_buffer(backend.be, total);
            size_t off = 0;
            for (size_t index = 0; index < graph_maps.size(); ++index) {
                for (ggml_tensor *t :
                     {dense_tensors[index],
                      use_compact ? compact_tensors[index] : nullptr,
                      use_compact ? scatter_tensors[index] : nullptr}) {
                    if (t && !t->buffer) {
                        void *addr =
                                static_cast<char *>(
                                        ggml_backend_buffer_get_base(side)) +
                                off;
                        ggml_backend_tensor_alloc(side, t, addr);
                        off += GGML_PAD(ggml_nbytes(t), 256);
                    }
                }
            }
            ggml_backend_buffer_set_usage(side,
                                          GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            side_buffers.push_back(side);
            if (use_cache) cache->side_buffers.push_back(side);
        }
    }
    {
        // Input upload: active rows first, then the zero row that -1 entries
        // resolve to (written on every rebuild/reset; kept intact afterwards).
        ggml_backend_tensor_set(input, features.data(), 0,
                                features.size() * sizeof(float));
        const int64_t zero_row = use_cache ? cache->input_capacity : active;
        std::vector<float> zeros(static_cast<size_t>(point_dim), 0.0f);
        ggml_backend_tensor_set(
                input, zeros.data(),
                static_cast<size_t>(zero_row) * point_dim * sizeof(float),
                static_cast<size_t>(point_dim) * sizeof(float));
    }
    std::vector<int32_t> remapped;
    for (size_t index = 0; index < result.maps.size(); ++index) {
        const auto &map = result.maps[index];
        if (map.reuse_of >= 0) continue;
#ifdef AICORE_OPENPCDET_CUDA_EXT
        // Device-resident rulebook: pack compact/scatter/dense index tensors
        // entirely on device (D2D), so no host rulebook download or H2D
        // upload happens at all. The compact lowering consumes compact +
        // scatter; the dense tensor is still packed for the quantized fallback
        // and for a device-side dense remap should the graph need it.
        if (backend.is_cuda() && use_compact && map.has_device_rulebook()) {
            if (!ext::upload_sparse_indice_map_cuda(
                        map, graph_maps[index],
                        compact_tensors[index] && compact_tensors[index]->buffer
                                ? static_cast<int32_t *>(
                                          compact_tensors[index]->data)
                                : nullptr,
                        scatter_tensors[index] && scatter_tensors[index]->buffer
                                ? static_cast<int32_t *>(
                                          scatter_tensors[index]->data)
                                : nullptr,
                        dense_tensors[index] && dense_tensors[index]->buffer
                                ? static_cast<int32_t *>(
                                          dense_tensors[index]->data)
                                : nullptr,
                        error)) {
                if (!use_cache) ggml_free(ctx);
                return false;
            }
            continue;
        }
        // Device-resident rulebook with the dense (stock-op) graph: remap on
        // device (D2D), padding included, so no host rulebook download or H2D
        // upload happens at all.
        if (backend.is_cuda() && !use_compact && map.has_device_rulebook()) {
            ggml_tensor *t = dense_tensors[index];
            const int64_t capacity_pairs = t ? t->ne[0] : 0;
            int64_t zero_row;
            if (use_cache) {
                zero_row = index == 0 ? cache->input_capacity
                                      : cache->output_capacity[index - 1];
            } else {
                zero_row =
                        index == 0
                                ? static_cast<int64_t>(active)
                                : result.maps[index - 1].output_active_count();
            }
            if (t && t->buffer &&
                capacity_pairs >=
                        map.kernel_volume * map.output_active_count()) {
                if (!ext::remap_dense_indices_cuda(
                            map, static_cast<int32_t>(zero_row),
                            static_cast<int32_t *>(t->data), capacity_pairs,
                            error)) {
                    if (!use_cache) ggml_free(ctx);
                    return false;
                }
                continue;
            }
        }
#endif
        // Remap -1 entries to the zero row of THIS layer's gather source
        // (layer 0 gathers the graph input; later layers gather the previous
        // layer's output, which the dense graph terminates with one appended
        // zero column).
        int64_t zero_row;
        if (use_cache) {
            zero_row = index == 0 ? cache->input_capacity
                                  : cache->output_capacity[index - 1];
        } else {
            zero_row = index == 0
                               ? static_cast<int64_t>(active)
                               : result.maps[index - 1].output_active_count();
        }
        ext::remap_dense_indices(map, static_cast<int32_t>(zero_row), remapped);
        if (use_cache) {
            // Prefix upload into the capacity-sized tensor; the padding is
            // written by the device remap above when a rulebook is resident,
            // and otherwise carries stale-but-valid indices from earlier
            // frames (see grow_sparse_capacity: the first frame after any
            // (re)build size-matches the tensors, and shrinking layers keep
            // their previous in-bounds prefix).
            ggml_tensor *t = dense_tensors[index];
            if (t && t->buffer) {
                ggml_backend_tensor_set(t, remapped.data(), 0,
                                        remapped.size() * sizeof(int32_t));
            }
        } else {
            ext::upload_dense_indices(dense_tensors[index], remapped);
        }
    }
#ifdef AICORE_OPENPCDET_CUDA_EXT
    if (backend.is_cuda() && !ext::synchronize_sparse_rulebook_cuda(error)) {
        if (!use_cache) ggml_free(ctx);
        return false;
    }
#endif
    timer.lap("upload");
    if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
        error = "sparse graph compute failed";
        if (use_cache)
            cache->clear();
        else
            ggml_free(ctx);
        return false;
    }
    timer.lap("graph compute");

    if (capture_layers) result.layer_features.resize(layer_tensors.size());
    for (size_t index = 0; index < layer_tensors.size(); ++index) {
        std::vector<float> values = readback_f32(layer_tensors[index]);
        if (sink)
            sink(ggml_get_name(layer_tensors[index]), values.data(),
                 result.maps[index].output_active_count(),
                 weights.specs[index].out_channels);
        if (capture_layers) result.layer_features[index] = std::move(values);
    }
    const auto &final_map = result.maps.back();
    const int depth = final_map.spatial_shape[0];
    result.height = final_map.spatial_shape[1];
    result.width = final_map.spatial_shape[2];
    const int sparse_channels = weights.specs.back().out_channels;
    if (sparse_channels * depth != hp.num_bev_features) {
        error = "HeightCompression channel mismatch";
        if (!use_cache) ggml_free(ctx);
        return false;
    }

#ifdef AICORE_OPENPCDET_CUDA_EXT
    // Device-resident HeightCompression: scatter the sparse rows straight
    // into the caller's CUDA tensor (optionally in GGML [W,H,C] order) and
    // skip the host readback + H2D re-upload entirely.
    const bool cuda_device_output =
            spatial_hwc_device && !capture_layers && !sink && backend.is_cuda();
    if (cuda_device_output) {
        const bool device_coordinates = final_map.device_coordinates != nullptr;
        const int32_t *height_coordinates =
                device_coordinates ? final_map.device_coordinates
                                   : final_map.coordinates.data();
        if (!ext::gpu_height_compress(static_cast<const float *>(output->data),
                                      height_coordinates,
                                      final_map.output_active_count(),
                                      sparse_channels, depth, result.height,
                                      result.width, spatial_hwc_device, error,
                                      device_coordinates, spatial_device_whc)) {
            if (!use_cache) ggml_free(ctx);
            return false;
        }
        result.spatial_device_ready = true;
        timer.lap("height compress D2D");
        if (!use_cache) ggml_free(ctx);
        return true;
    }
#else
    (void)spatial_hwc_device;
    (void)spatial_device_whc;
#endif

    // Host HeightCompression: scatter each active sparse row into the dense
    // BEV [H*W, C*depth] WHC buffer by its (y, x) coordinate and z slot.
    std::vector<float> sparse_output =
            capture_layers && !result.layer_features.empty()
                    ? result.layer_features.back()
                    : readback_f32(output);
    result.spatial_features.assign(static_cast<size_t>(result.height) *
                                           result.width * hp.num_bev_features,
                                   0.0f);
    for (int64_t row = 0; row < final_map.output_active_count(); ++row) {
        const int32_t *coordinate = final_map.coordinates.data() + row * 4;
        float *target = result.spatial_features.data() +
                        (static_cast<size_t>(coordinate[2]) * result.width +
                         coordinate[3]) *
                                hp.num_bev_features;
        const float *source = sparse_output.data() + row * sparse_channels;
        for (int channel = 0; channel < sparse_channels; ++channel)
            target[channel * depth + coordinate[1]] = source[channel];
    }
    if (sink)
        sink("spatial_features", result.spatial_features.data(),
             static_cast<int64_t>(result.height) * result.width,
             hp.num_bev_features);
    timer.lap("readback+compress");
    if (!use_cache) ggml_free(ctx);
    return true;
}

}  // namespace openpcdet
