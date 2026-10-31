// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pointpillar.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "../extensions/center_head_ops.h"
#include "../ops/nms_3d.h"
#include "../ops/voxelization.h"

namespace openpcdet {

namespace {

// Parallel-for over [0,n): splits into contiguous chunks, one per HW thread.
template <class F>
void parallel_for(int64_t n, F &&body) {
    if (n <= 0) return;
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    nt = (unsigned)std::min<int64_t>(nt, n);
    if (nt <= 1) {
        body(int64_t{0}, n);
        return;
    }
    const int64_t chunk = (n + nt - 1) / nt;
    std::vector<std::thread> ths;
    ths.reserve(nt);
    for (int64_t a = 0; a < n; a += chunk)
        ths.emplace_back(
                [&body, a, b = std::min(n, a + chunk)] { body(a, b); });
    for (auto &t : ths) t.join();
}

// Read back a ggml tensor as f32 (handles non-contiguous via per-row copy).
std::vector<float> readback_f32(ggml_tensor *t) {
    std::vector<float> out((size_t)ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, out.data(), 0,
                                ggml_nelements(t) * sizeof(float));
        return out;
    }
    // 4-D fallback: copy row-by-row (matches free-splatter.cpp's tap reader).
    const int64_t n_outer = t->ne[3] * t->ne[2] * t->ne[1];
    for (int64_t r = 0; r < n_outer; r++) {
        ggml_backend_tensor_get(t, out.data() + r * t->ne[0], r * t->nb[1],
                                t->ne[0] * sizeof(float));
    }
    return out;
}

#ifdef AICORE_OPENPCDET_CUDA_EXT
// Host f32 read of a weight tensor (f32/f16/dequant). VFE weights stay on the
// CPU file context (host memory), so read t->data directly like to_f32().
std::vector<float> weight_to_f32(ggml_tensor *t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, out.data(), n);
    } else {
        const int64_t row_size = t->ne[0];
        const int64_t rows = row_size ? n / row_size : 0;
        const size_t src_row_bytes = ggml_row_size(t->type, row_size);
        const ggml_type_traits *tr = ggml_get_type_traits(t->type);
        for (int64_t r = 0; r < rows; r++)
            tr->to_float((const char *)t->data + r * src_row_bytes,
                         out.data() + r * row_size, row_size);
    }
    return out;
}
#endif

std::vector<float> whc_to_anchor_major(const std::vector<float> &source,
                                       int width,
                                       int height,
                                       int anchors,
                                       int channels) {
    std::vector<float> output((size_t)width * height * anchors * channels);
    for (int anchor = 0; anchor < anchors; anchor++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int channel = 0; channel < channels; channel++) {
                    const int source_channel = anchor * channels + channel;
                    const size_t source_index =
                            (size_t)x +
                            (size_t)width *
                                    (y + (size_t)height * source_channel);
                    const size_t target_row =
                            ((size_t)anchor * height + y) * width + x;
                    output[target_row * channels + channel] =
                            source[source_index];
                }
            }
        }
    }
    return output;
}

}  // namespace

pointpillar::~pointpillar() { release_graph(); }

void pointpillar::release_graph() {
    if (graph_ctx) {
        ggml_free(graph_ctx);
        graph_ctx = nullptr;
    }
    graph_gf = nullptr;
    graph_input = nullptr;
    graph_sparse_features = nullptr;
    graph_scatter_rows = nullptr;
    graph_dense_direct = false;
    graph_built = false;
    cache_taps.clear();
}

bool pointpillar::load(const std::string &gguf_path,
                       const std::string &device,
                       int n_threads) {
    if (!model::load(gguf_path, device, n_threads)) return false;
    std::string err;
    if (!load_pillar_vfe_weights(file, file.hp, vfe_w, err)) {
        error = err;
        return false;
    }
#ifdef AICORE_OPENPCDET_CUDA_EXT
    gpu_vfe_layers.clear();
    gpu_vfe_layers.reserve(vfe_w.layers.size());
    for (const auto &L : vfe_w.layers) {
        ext::gpu_pfn_layer g;
        g.linear_w = weight_to_f32(L.linear_w);
        g.linear_b =
                L.linear_b ? weight_to_f32(L.linear_b) : std::vector<float>{};
        g.bn_scale =
                L.bn_scale ? weight_to_f32(L.bn_scale) : std::vector<float>{};
        g.bn_bias = L.bn_bias ? weight_to_f32(L.bn_bias) : std::vector<float>{};
        g.in_channels = L.linear_w ? static_cast<int>(L.linear_w->ne[1]) : 0;
        g.out_channels = L.out_channels;
        g.last_layer = L.last_layer;
        gpu_vfe_layers.push_back(std::move(g));
    }
#endif
    if (!load_bev_backbone_weights(file, file.hp, bb_w, err)) {
        error = err;
        return false;
    }
    if (file.hp.head_kind == dense_head_kind::center) {
        if (!load_center_head_weights(file, file.hp, center_head_w, err)) {
            error = err;
            return false;
        }
    } else if (file.hp.head_kind == dense_head_kind::anchor_multi) {
        if (!load_anchor_multi_weights(file, file.hp, multi_head_w, err)) {
            error = err;
            return false;
        }
    } else if (!load_anchor_head_weights(file, file.hp, head_w, err)) {
        error = err;
        return false;
    }
    // Re-resolve weight pointers to the device mirror (realize_weights was
    // called by model::load before us; on CPU this is a no-op since file.ctx
    // tensors are valid directly).
    if (device_ctx) {
        for (auto &L : vfe_w.layers) {
            // VFE runs on CPU; tensors stay on file.ctx, no remap needed.
            (void)L;
        }
        auto remap = [&](ggml_tensor *&tref) {
            if (tref) tref = ggml_get_tensor(device_ctx, ggml_get_name(tref));
        };
        for (auto &lvl : bb_w.levels)
            for (auto &b : lvl) {
                remap(b.conv_w);
                remap(b.bn_scale);
                remap(b.bn_bias);
            }
        for (auto &d : bb_w.deblocks) {
            remap(d.conv_w);
            remap(d.bn_scale);
            remap(d.bn_bias);
        }
        if (file.hp.head_kind == dense_head_kind::center) {
            auto remap_block = [&](conv_bn_block_weights &block) {
                remap(block.conv_w);
                remap(block.bn_scale);
                remap(block.bn_bias);
            };
            auto remap_sequence = [&](center_sequence_weights &sequence) {
                for (auto &block : sequence.middle) remap_block(block);
                remap(sequence.output_weight);
                remap(sequence.output_bias);
            };
            remap_block(center_head_w.shared);
            for (auto &head : center_head_w.heads) {
                remap_sequence(head.heatmap);
                remap_sequence(head.center);
                remap_sequence(head.center_z);
                remap_sequence(head.dimensions);
                remap_sequence(head.rotation);
                remap_sequence(head.velocity);
            }
        } else if (file.hp.head_kind == dense_head_kind::anchor_multi) {
            auto remap_block = [&](conv_bn_block_weights &block) {
                remap(block.conv_w);
                remap(block.bn_scale);
                remap(block.bn_bias);
            };
            auto remap_sequence = [&](dense_head_sequence_weights &sequence) {
                for (auto &block : sequence.middle) remap_block(block);
                remap(sequence.output.weight);
                remap(sequence.output.bias);
            };
            if (multi_head_w.has_shared) remap_block(multi_head_w.shared);
            for (auto &head : multi_head_w.heads) {
                remap_sequence(head.classification);
                for (auto &branch : head.regression) remap_sequence(branch);
                remap_sequence(head.box);
                remap_sequence(head.direction);
            }
        } else {
            remap(head_w.conv_cls_w);
            remap(head_w.conv_cls_b);
            remap(head_w.conv_box_w);
            remap(head_w.conv_box_b);
            remap(head_w.conv_dir_cls_w);
            remap(head_w.conv_dir_cls_b);
        }
    }
    return true;
}

bool pointpillar::forward(const float *points,
                          int32_t n_points,
                          int32_t point_dim,
                          std::vector<detection> &out_dets,
                          const tap_sink &sink) {
    const hparams &h = file.hp;
    out_dets.clear();
    if (n_points < 0 || (n_points > 0 && !points)) {
        error = "invalid point buffer";
        return false;
    }
    if (point_dim != h.num_point_features) {
        error = "point_dim mismatch: model expects " +
                std::to_string(h.num_point_features) + ", got " +
                std::to_string(point_dim);
        return false;
    }
    if (n_points == 0) return true;

        // AICore port note: stage timings are reported through the common
        // aicore_pipeline_timings contract by the C API wrapper; the upstream
        // stderr lap profiler stays available under OPCPD_PROFILE_LAPS builds.
#ifdef OPCPD_PROFILE_LAPS
    const bool prof = true;
#else
    const bool prof = false;
#endif
    auto t_clk = std::chrono::steady_clock::now();
    auto lap = [&](const char *name) {
        if (!prof) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(
                stderr, "[profile] %-16s %7.2f ms\n", name,
                std::chrono::duration<double, std::milli>(now - t_clk).count());
        t_clk = now;
    };

    // --- 1-2. Pillar grouping and VFE ---
    // CUDA-ext path (upstream OpenPCDet-GGML vfe_scatter_cuda): the per-point
    // MLP, the per-pillar maxpool and the BEV scatter run on the GPU and, from
    // the second frame on, the BEV lands straight in the persistent graph's
    // dense input tensor (D2D). CPU/Vulkan keep the host VFE + in-graph
    // stock-op scatter unchanged.
    voxel_output vox;
    std::vector<int32_t> pillar_coords;
    std::vector<float> pillar_features;
    int V = 0;
    std::vector<float> spatial_whc;  // filled by GPU path or host scatter below
    bool gpu_vfe_scatter_done = false;
    bool gpu_spatial_direct =
            false;  // spatial written straight into the ggml device tensor
    if (h.vfe_dynamic) {
#ifdef AICORE_OPENPCDET_CUDA_EXT
        if (be.is_cuda()) {
            std::string gerr;
            float *spatial_dev =
                    (graph_built && graph_input && graph_dense_direct)
                            ? static_cast<float *>(graph_input->data)
                            : nullptr;
            if (!ext::gpu_pillar_vfe_scatter(
                        points, n_points, point_dim, h.grid_size[0],
                        h.grid_size[1], h.grid_size[2], h.voxel_size[0],
                        h.voxel_size[1], h.voxel_size[2],
                        h.point_cloud_range[0], h.point_cloud_range[1],
                        h.point_cloud_range[2], h.use_absolute_xyz,
                        h.with_distance, gpu_vfe_layers, h.num_bev_features,
                        pillar_coords, pillar_features, spatial_whc, gerr,
                        spatial_dev, static_cast<bool>(sink))) {
                error = "gpu vfe+scatter failed: " + gerr;
                return false;
            }
            V = static_cast<int>(pillar_coords.size() / 4);
            gpu_vfe_scatter_done = true;
            gpu_spatial_direct = (spatial_dev != nullptr);
            lap("dynamic_vfe_gpu");
        }
#endif
        if (!gpu_vfe_scatter_done) {
            if (!run_dynamic_pillar_vfe_cpu(vfe_w, h, points, n_points,
                                            point_dim, pillar_coords,
                                            pillar_features, error, sink))
                return false;
            V = static_cast<int>(pillar_coords.size() / 4);
            lap("dynamic_vfe");
        }
    } else {
        voxelization_params vp;
        vp.point_cloud_range = h.point_cloud_range;
        vp.voxel_size = h.voxel_size;
        vp.grid_size = h.grid_size;
        vp.max_points_per_voxel = h.max_points_per_voxel;
        vp.max_voxels = h.max_voxels;
        vp.min_points_per_voxel = 1;
        if (!voxelize(vp, points, n_points, point_dim, vox, error))
            return false;
        lap("voxelize");
        V = vox.actual_voxels;
        if (V > 0) {
#ifdef AICORE_OPENPCDET_CUDA_EXT
            if (be.is_cuda()) {
                std::string gerr;
                float *spatial_dev =
                        (graph_built && graph_input && graph_dense_direct)
                                ? static_cast<float *>(graph_input->data)
                                : nullptr;
                if (!ext::gpu_pillar_vfe_scatter_vox(
                            vox.voxels.data(), vox.num_points.data(),
                            vox.coords.data(), V, h.max_points_per_voxel,
                            point_dim, h.grid_size[0], h.grid_size[1],
                            h.voxel_size[0], h.voxel_size[1], h.voxel_size[2],
                            h.point_cloud_range[0], h.point_cloud_range[1],
                            h.point_cloud_range[2], h.use_absolute_xyz,
                            h.with_distance, gpu_vfe_layers, h.num_bev_features,
                            pillar_features, spatial_whc, gerr, spatial_dev,
                            static_cast<bool>(sink))) {
                    error = "gpu vox vfe+scatter failed: " + gerr;
                    return false;
                }
                pillar_coords = vox.coords;
                gpu_vfe_scatter_done = true;
                gpu_spatial_direct = (spatial_dev != nullptr);
                lap("vfe_gpu");
            }
#endif
            if (!gpu_vfe_scatter_done &&
                !run_pillar_vfe_cpu(vfe_w, h, vox.voxels.data(),
                                    vox.num_points.data(), vox.coords.data(), V,
                                    h.max_points_per_voxel, point_dim,
                                    pillar_features, error, sink))
                return false;
        }
        if (!gpu_vfe_scatter_done) {
            pillar_coords = vox.coords;
        }
        lap("vfe");
    }
    if (V == 0) return true;
    if (pillar_features.size() != (size_t)V * h.num_bev_features) {
        error = "PillarVFE output shape mismatch";
        return false;
    }
    if (sink)
        sink("pillar_features", pillar_features.data(), V,
             h.vfe_num_filters.back());

    // --- 3. PointPillarScatter -> BEV input for the ggml graph ---
    // AICore port note: for the common nz==1 pillar grid the scatter runs
    // in-graph on stock ops (ggml_fill zero canvas + ggml_set_rows), so no
    // host BEV buffer exists on the fast path. The host scatter below runs
    // only for the 3-D PointPillarScatter3d layout (nz>1) or when a tap sink
    // needs the host spatial_features buffer.
    const int C_bev = h.num_bev_features;  // 64
    const int nx = h.grid_size[0], ny = h.grid_size[1], nz = h.grid_size[2];
    const bool host_scatter = nz != 1 || sink || !gpu_vfe_scatter_done;
    if (host_scatter) {
        // Directly build the ggml [W,H,C,N] layout (W fastest), so the previous
        // separate HWC->WHC repack pass is folded away. Pillar scatter for the
        // common nz==1 (PointPillar) case writes C contiguous channels per cell
        // at a strided location.
        spatial_whc.assign((size_t)nx * ny * C_bev, 0.0f);
        if (nz == 1) {
            // Each pillar writes C independent channels to far-apart (strided)
            // locations, so the loop is memory-latency bound; parallelize over
            // V to hide that latency across threads.
            parallel_for(V, [&](int64_t v_begin, int64_t v_end) {
                for (int64_t v = v_begin; v < v_end; v++) {
                    const int32_t gx = pillar_coords[v * 4 + 3];
                    const int32_t gy = pillar_coords[v * 4 + 2];
                    const float *src =
                            pillar_features.data() + (size_t)v * C_bev;
                    float *dst =
                            spatial_whc.data() + (size_t)gx + (size_t)nx * gy;
                    for (int c = 0; c < C_bev; c++)
                        dst[(size_t)nx * ny * c] = src[c];
                }
            });
        } else {
            // 3D scatter (PointPillarScatter3d): build HWC then repack to WHC.
            std::vector<float> bev((size_t)nx * ny * C_bev, 0.0f);
            for (int v = 0; v < V; v++) {
                const int32_t gx = pillar_coords[v * 4 + 3];
                const int32_t gy = pillar_coords[v * 4 + 2];
                const int32_t gz = pillar_coords[v * 4 + 1];
                const float *src = pillar_features.data() + (size_t)v * C_bev;
                const size_t idx = (size_t)(gz * ny * nx + gy * nx + gx);
                float *dst = bev.data() + idx * C_bev;
                for (int c = 0; c < C_bev; c++) dst[c] = src[c];
            }
            parallel_for(C_bev, [&](int64_t c_begin, int64_t c_end) {
                for (int64_t c = c_begin; c < c_end; c++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            spatial_whc[(size_t)x +
                                        (size_t)nx * (y + (size_t)ny * c)] =
                                    bev[((size_t)y * nx + x) * C_bev + c];
                        }
                    }
                }
            });
        }
    }
    lap("scatter");
    if (sink) {
        // Emit the spatial features in the OpenPCDet HWC layout ([cell][C]).
        std::vector<float> bev_hwc((size_t)nx * ny * C_bev, 0.0f);
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
                for (int c = 0; c < C_bev; c++)
                    bev_hwc[((size_t)y * nx + x) * C_bev + c] =
                            spatial_whc[(size_t)x +
                                        (size_t)nx * (y + (size_t)ny * c)];
        sink("spatial_features", bev_hwc.data(), (int64_t)nx * ny, C_bev);
    }

    // --- 4. Build the ggml graph: backbone + head ---
    // Output spatial_2d tensor shape depends on the backbone's upsample
    // strides. For the standard PointPillar config (upsample_strides=[1,2,4]),
    // the final resolution is [nx, ny] (concat of all upsampled levels at
    // their respective strides — but OpenPCDet's PointPillar config is set up
    // so all upsamples produce [nx, ny]).
    //
    // The graph is built and allocated only once (on the first frame) and then
    // reused: weights live in the stable device_ctx and only the spatial input
    // data changes between frames, so rebuilding every frame is pure overhead.
    // DynamicPillarVFE keeps every pillar (no max_voxels clamp, matching
    // upstream), so a sweep can exceed max_voxels — the canvas starts at
    // max(max_voxels, first-frame V) and the graph is rebuilt (grown) if a
    // later frame produces more pillars.
    if (graph_built && graph_sparse_features &&
        V > graph_sparse_features->ne[1]) {
        release_graph();
    }
    if (!graph_built) {
        const size_t graph_nodes = 4096;
        ggml_init_params gp = {
                ggml_tensor_overhead() * graph_nodes +
                        ggml_graph_overhead_custom(graph_nodes, false),
                nullptr,
                /*no_alloc =*/true,
        };
        ggml_context *ctx = ggml_init(gp);
        if (!ctx) {
            error = "graph ctx init failed";
            return false;
        }
        graph_ctx = ctx;

#ifdef AICORE_OPENPCDET_CUDA_EXT
        // CUDA-ext graph topology: the GPU VFE/scatter writes the dense BEV
        // straight into this tensor (D2D from the second frame on; the host
        // buffer feeds the first frame). Decided once per graph build — the
        // backend is fixed for the model's lifetime.
        graph_dense_direct = be.is_cuda() && gpu_vfe_scatter_done;
#else
        graph_dense_direct = false;
#endif
        if (nz == 1 && !host_scatter && !graph_dense_direct) {
            // AICore port note — in-graph stock-op pillar scatter: a
            // zero-filled canvas (ggml_fill) with one trailing trash row
            // receives each pillar's feature row through ggml_set_rows (unique
            // destination rows, so the assignment semantics are exact). The
            // canvas capacity starts at max(max_voxels, first-frame V) and
            // keeps the persistent graph shape stable across frames (growing
            // via release_graph when a later dynamic-pillar frame exceeds it);
            // padded slots write stale feature rows to the trash row, which is
            // cropped away before the backbone consumes the canvas.
            const int64_t cells = static_cast<int64_t>(nx) * ny;
            const int64_t capacity =
                    std::max<int64_t>(1, std::max<int64_t>(h.max_voxels, V));
            graph_sparse_features =
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C_bev, capacity);
            graph_scatter_rows =
                    ggml_new_tensor_2d(ctx, GGML_TYPE_I64, capacity, 1);
            ggml_set_input(graph_sparse_features);
            ggml_set_input(graph_scatter_rows);
            ggml_tensor *canvas = ggml_fill(
                    ctx,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C_bev, cells + 1),
                    0.0f);
            ggml_tensor *scattered = ggml_set_rows(
                    ctx, canvas, graph_sparse_features, graph_scatter_rows);
            ggml_tensor *cell_major = ggml_view_2d(ctx, scattered, C_bev, cells,
                                                   scattered->nb[1], 0);
            ggml_tensor *whc = ggml_cont_2d(
                    ctx, ggml_transpose(ctx, cell_major), cells, C_bev);
            graph_input = ggml_reshape_4d(ctx, whc, nx, ny, C_bev, 1);
        } else {
            // BEV input tensor: [W=nx, H=ny, C=C_bev, N=1]
            graph_input =
                    ggml_new_tensor_4d(ctx, GGML_TYPE_F32, nx, ny, C_bev, 1);
            ggml_set_input(graph_input);
        }

        cache_taps.clear();
        ggml_tensor *spatial_2d = build_bev_backbone_graph(
                ctx, bb_w, h, graph_input, sink ? &cache_taps : nullptr);
        cache_head = head_outputs{};
        cache_multi.clear();
        cache_center.clear();
        if (h.head_kind == dense_head_kind::center) {
            cache_center =
                    build_center_head_graph(ctx, center_head_w, h, spatial_2d);
            for (auto &output : cache_center) {
                ggml_set_output(output.heatmap);
                ggml_set_output(output.center);
                ggml_set_output(output.center_z);
                ggml_set_output(output.dimensions);
                ggml_set_output(output.rotation);
                if (output.velocity) ggml_set_output(output.velocity);
            }
        } else if (h.head_kind == dense_head_kind::anchor_multi) {
            cache_multi =
                    build_anchor_multi_graph(ctx, multi_head_w, h, spatial_2d);
            for (auto &output : cache_multi) {
                ggml_set_output(output.classification);
                for (ggml_tensor *branch : output.regression)
                    ggml_set_output(branch);
                if (output.box) ggml_set_output(output.box);
                if (output.direction) ggml_set_output(output.direction);
            }
        } else {
            cache_head = build_anchor_head_graph(ctx, head_w, h, spatial_2d);
            ggml_set_output(cache_head.cls);
            ggml_set_output(cache_head.box);
            if (cache_head.dir) ggml_set_output(cache_head.dir);
        }

        if (sink) {
            // The build_*_graph helpers marked intermediate taps via
            // ggml_set_output; walk the graph to collect them. For simplicity,
            // only the head outputs and spatial_2d are emitted here —
            // component-level taps are emitted by the CPU helpers (vfe,
            // scatter).
            ggml_set_name(spatial_2d, "spatial_features_2d");
            ggml_set_output(spatial_2d);
            if (cache_taps.empty() || cache_taps.back() != spatial_2d)
                cache_taps.push_back(spatial_2d);
        }

        graph_gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        if (h.head_kind == dense_head_kind::center) {
            for (auto &output : cache_center) {
                ggml_build_forward_expand(graph_gf, output.heatmap);
                ggml_build_forward_expand(graph_gf, output.center);
                ggml_build_forward_expand(graph_gf, output.center_z);
                ggml_build_forward_expand(graph_gf, output.dimensions);
                ggml_build_forward_expand(graph_gf, output.rotation);
                if (output.velocity)
                    ggml_build_forward_expand(graph_gf, output.velocity);
            }
        } else if (h.head_kind == dense_head_kind::anchor_multi) {
            for (auto &output : cache_multi) {
                ggml_build_forward_expand(graph_gf, output.classification);
                for (ggml_tensor *branch : output.regression)
                    ggml_build_forward_expand(graph_gf, branch);
                if (output.box) ggml_build_forward_expand(graph_gf, output.box);
                if (output.direction)
                    ggml_build_forward_expand(graph_gf, output.direction);
            }
        } else {
            ggml_build_forward_expand(graph_gf, cache_head.cls);
            ggml_build_forward_expand(graph_gf, cache_head.box);
            if (cache_head.dir)
                ggml_build_forward_expand(graph_gf, cache_head.dir);
        }
        for (auto *t : cache_taps) ggml_build_forward_expand(graph_gf, t);
        lap("graph_build");

        if (!ggml_gallocr_alloc_graph(be.galloc, graph_gf)) {
            error = "graph alloc failed";
            return false;
        }
        lap("alloc");
        graph_built = true;
    }

    // Local aliases so the (shared) decode/readback code below can keep using
    // the same names regardless of whether the graph was just built or reused.
    head_outputs &head = cache_head;
    auto &multi_outputs = cache_multi;
    auto &center_outputs = cache_center;
    auto &tap_list = cache_taps;
    ggml_tensor *spatial = graph_input;
    ggml_context *ctx = graph_ctx;
    ggml_cgraph *gf = graph_gf;
    (void)ctx;
    (void)gf;

    if (graph_sparse_features) {
        // In-graph set_rows scatter upload: real pillar rows take slots
        // [0, V), every padded slot points at the trailing trash row so the
        // (stale) feature data it uploads can never corrupt a real cell.
        const int64_t cells = static_cast<int64_t>(nx) * ny;
        const int64_t capacity = graph_sparse_features->ne[1];
        if (V > capacity) {
            error = "pillar count exceeds graph scatter capacity";
            return false;
        }
        std::vector<int64_t> scatter_rows_host(static_cast<size_t>(capacity),
                                               cells);
        for (int v = 0; v < V; ++v) {
            const int32_t gx = pillar_coords[static_cast<size_t>(v) * 4 + 3];
            const int32_t gy = pillar_coords[static_cast<size_t>(v) * 4 + 2];
            if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) {
                error = "pillar coordinate outside scatter grid";
                return false;
            }
            scatter_rows_host[static_cast<size_t>(v)] = (int64_t)gy * nx + gx;
        }
        ggml_backend_tensor_set(graph_sparse_features, pillar_features.data(),
                                0,
                                static_cast<size_t>(V) * C_bev * sizeof(float));
        ggml_backend_tensor_set(
                graph_scatter_rows, scatter_rows_host.data(), 0,
                static_cast<size_t>(capacity) * sizeof(int64_t));
    } else if (!(graph_dense_direct && gpu_spatial_direct)) {
        // Dense BEV upload: the first frame after the graph build (and any
        // frame whose VFE ran on the host) uploads the host buffer; frames
        // whose VFE scattered D2D already have the data in place.
        ggml_backend_tensor_set(spatial, spatial_whc.data(), 0,
                                spatial_whc.size() * sizeof(float));
    }
    lap("upload");

    if (ggml_backend_graph_compute(be.be, gf) != GGML_STATUS_SUCCESS) {
        error = "graph compute failed";
        return false;
    }
    lap("compute");

    if (h.head_kind == dense_head_kind::center) {
        size_t label_offset = 0;
        ext::center_decode_params params;
        params.point_cloud_range = h.point_cloud_range;
        params.voxel_size = h.voxel_size;
        params.feature_map_stride = h.center_feature_map_stride;
        params.max_objects = h.center_max_objects;
        params.score_threshold = h.score_thresh;
        params.post_center_limit_range = h.center_post_limit_range;
        params.nms_threshold = h.nms_thresh;
        params.nms_pre_maxsize = h.nms_pre_maxsize;
        params.nms_post_maxsize = h.max_boxes;
        for (size_t head_index = 0; head_index < center_outputs.size();
             head_index++) {
            const center_single_outputs &output = center_outputs[head_index];
            const int width = static_cast<int>(output.heatmap->ne[0]);
            const int height = static_cast<int>(output.heatmap->ne[1]);
            const int classes = h.center_class_counts[head_index];
            if (label_offset + classes > h.center_label_indices.size()) {
                error = "CenterHead flattened class mapping overflow";
                return false;
            }
            auto read_hwc = [&](ggml_tensor *tensor) {
                const std::vector<float> whc = readback_f32(tensor);
                return whc_to_hwc(whc.data(), width, height,
                                  static_cast<int>(tensor->ne[2]));
            };
            std::vector<float> heatmap = read_hwc(output.heatmap);
            std::vector<float> center = read_hwc(output.center);
            std::vector<float> center_z = read_hwc(output.center_z);
            std::vector<float> dimensions = read_hwc(output.dimensions);
            std::vector<float> rotation = read_hwc(output.rotation);
            std::vector<float> velocity = output.velocity
                                                  ? read_hwc(output.velocity)
                                                  : std::vector<float>();
            ext::center_head_view view;
            view.heatmap = heatmap.data();
            view.center = center.data();
            view.center_z = center_z.data();
            view.dimensions = dimensions.data();
            view.rotation = rotation.data();
            view.velocity = velocity.empty() ? nullptr : velocity.data();
            view.height = height;
            view.width = width;
            view.class_count = classes;
            view.class_mapping.assign(
                    h.center_label_indices.begin() + label_offset,
                    h.center_label_indices.begin() + label_offset + classes);
            std::vector<detection> head_detections;
            if (!ext::decode_center_head(view, params, head_detections,
                                         error)) {
                return false;
            }
            out_dets.insert(out_dets.end(), head_detections.begin(),
                            head_detections.end());
            if (sink) {
                const std::string prefix = "head." + std::to_string(head_index);
                sink(prefix + ".hm", heatmap.data(),
                     static_cast<int64_t>(width) * height, classes);
                sink(prefix + ".center", center.data(),
                     static_cast<int64_t>(width) * height, 2);
                sink(prefix + ".center_z", center_z.data(),
                     static_cast<int64_t>(width) * height, 1);
                sink(prefix + ".dim", dimensions.data(),
                     static_cast<int64_t>(width) * height, 3);
                sink(prefix + ".rot", rotation.data(),
                     static_cast<int64_t>(width) * height, 2);
                if (!velocity.empty())
                    sink(prefix + ".vel", velocity.data(),
                         static_cast<int64_t>(width) * height, 2);
            }
            label_offset += classes;
        }
        if (sink) {
            for (ggml_tensor *tensor : tap_list) {
                const std::vector<float> whc = readback_f32(tensor);
                std::vector<float> hwc =
                        whc_to_hwc(whc.data(), static_cast<int>(tensor->ne[0]),
                                   static_cast<int>(tensor->ne[1]),
                                   static_cast<int>(tensor->ne[2]));
                sink(ggml_get_name(tensor), hwc.data(),
                     tensor->ne[0] * tensor->ne[1], tensor->ne[2]);
            }
        }
        lap("center_decode_nms");

        return true;
    }

    if (h.head_kind == dense_head_kind::anchor_multi) {
        size_t label_offset = 0;
        size_t set_offset = 0;
        for (size_t head_index = 0; head_index < multi_outputs.size();
             head_index++) {
            const auto &output = multi_outputs[head_index];
            const int classes = h.multi_head_class_counts[head_index];
            const int anchors_per_location =
                    h.multi_head_anchor_counts[head_index];
            const int width = (int)output.classification->ne[0];
            const int height = (int)output.classification->ne[1];
            const int rows = width * height * anchors_per_location;
            if (label_offset + classes > h.multi_label_indices.size() ||
                set_offset + classes > h.multi_anchor_set_indices.size()) {
                error = "AnchorHeadMulti flattened metadata overflow";
                return false;
            }
            std::vector<int32_t> anchor_sets(
                    h.multi_anchor_set_indices.begin() + set_offset,
                    h.multi_anchor_set_indices.begin() + set_offset + classes);
            anchor_grid anchors =
                    generate_multihead_anchors(h, anchor_sets, width, height);
            if (anchors.total != rows) {
                error = "AnchorHeadMulti anchor count mismatch in head " +
                        std::to_string(head_index);
                return false;
            }

            std::vector<float> cls = whc_to_anchor_major(
                    readback_f32(output.classification), width, height,
                    anchors_per_location, classes);
            // Fused sigmoid + candidate collection: single contiguous pass over
            // rows computing every class's score per row (better cache locality
            // than the strided per-class access), pre-building each class's
            // candidate list. Numerically identical to the per-class passes
            // (same exp/sigmoid/threshold, same row-ascending push_back order).
            std::vector<float> all_scores((size_t)rows * classes);
            std::vector<std::vector<int32_t>> cand_lists((size_t)classes);
            for (int row = 0; row < rows; row++) {
                const float *cls_row = cls.data() + (size_t)row * classes;
                float *sc_row = all_scores.data() + (size_t)row * classes;
                for (int c = 0; c < classes; c++) {
                    const float score = 1.0f / (1.0f + std::exp(-cls_row[c]));
                    sc_row[c] = score;
                    if (score >= h.score_thresh) cand_lists[c].push_back(row);
                }
            }

            // Gather box/direction ONLY for the rows we will decode (the union
            // of per-class candidates) instead of re-packing the full anchor
            // grid (whc_to_anchor_major) for every regression branch. The
            // per-row values read are identical, so results are bit-identical.
            // With sink (validation taps) we fall back to gathering the full
            // grid.
            std::vector<int32_t> gather_rows;
            if (sink) {
                gather_rows.resize((size_t)rows);
                for (int r = 0; r < rows; r++) gather_rows[(size_t)r] = r;
            } else {
                size_t total = 0;
                for (int c = 0; c < classes; c++) total += cand_lists[c].size();
                gather_rows.reserve(total);
                for (int c = 0; c < classes; c++)
                    gather_rows.insert(gather_rows.end(), cand_lists[c].begin(),
                                       cand_lists[c].end());
                std::sort(gather_rows.begin(), gather_rows.end());
                gather_rows.erase(
                        std::unique(gather_rows.begin(), gather_rows.end()),
                        gather_rows.end());
            }
            const int n_gather = (int)gather_rows.size();
            std::vector<int32_t> row_to_compact;
            if (sink) {
                row_to_compact.resize((size_t)rows);
                for (int r = 0; r < rows; r++) row_to_compact[(size_t)r] = r;
            } else {
                row_to_compact.assign((size_t)rows, -1);
                for (int i = 0; i < n_gather; i++)
                    row_to_compact[(size_t)gather_rows[i]] = i;
            }

            auto gather_row_whc = [&](const std::vector<float> &source,
                                      int channels, std::vector<float> &out) {
                // out[i*channels..] = source at (x,y,anchor) for
                // gather_rows[i]. Same index math as whc_to_anchor_major (row
                // -> x,y,anchor); the channel stride in the [W,H,C,N] source is
                // W*H, not 1.
                for (int i = 0; i < n_gather; i++) {
                    const int row = gather_rows[i];
                    const int x = row % width;
                    const int t = row / width;
                    const int y = t % height;
                    const int anchor = t / height;
                    const size_t base =
                            (size_t)x +
                            (size_t)width * ((size_t)y +
                                             (size_t)height * ((size_t)anchor *
                                                               channels));
                    const size_t ch_stride = (size_t)width * height;
                    float *dst = out.data() + (size_t)i * channels;
                    for (int ch = 0; ch < channels; ch++)
                        dst[ch] = source[base + ch_stride * ch];
                }
            };

            std::vector<float> cand_box((size_t)n_gather * h.code_size);
            if (h.multi_separate_reg) {
                int code_offset = 0;
                if (output.regression.size() != h.multi_reg_channels.size()) {
                    error = "AnchorHeadMulti regression branch count mismatch";
                    return false;
                }
                for (size_t branch = 0; branch < output.regression.size();
                     branch++) {
                    const int channels = h.multi_reg_channels[branch];
                    const std::vector<float> branch_whc =
                            readback_f32(output.regression[branch]);
                    for (int i = 0; i < n_gather; i++) {
                        const int row = gather_rows[i];
                        const int x = row % width;
                        const int t = row / width;
                        const int y = t % height;
                        const int anchor = t / height;
                        const size_t base =
                                (size_t)x +
                                (size_t)width *
                                        ((size_t)y +
                                         (size_t)height *
                                                 ((size_t)anchor * channels));
                        const size_t ch_stride = (size_t)width * height;
                        float *dst = cand_box.data() + (size_t)i * h.code_size +
                                     code_offset;
                        for (int ch = 0; ch < channels; ch++)
                            dst[ch] = branch_whc[base + ch_stride * ch];
                    }
                    code_offset += channels;
                }
                if (code_offset != h.code_size) {
                    error = "AnchorHeadMulti regression code size mismatch";
                    return false;
                }
            } else {
                const std::vector<float> box_whc = readback_f32(output.box);
                gather_row_whc(box_whc, h.code_size, cand_box);
            }
            std::vector<float> cand_dir;
            if (output.direction) {
                cand_dir.resize((size_t)n_gather * h.num_dir_bins);
                const std::vector<float> dir_whc =
                        readback_f32(output.direction);
                gather_row_whc(dir_whc, h.num_dir_bins, cand_dir);
            }
            if (sink) {
                const std::string prefix = "head." + std::to_string(head_index);
                sink(prefix + ".cls", cls.data(), rows, classes);
                sink(prefix + ".box", cand_box.data(), rows, h.code_size);
                if (!cand_dir.empty())
                    sink(prefix + ".dir", cand_dir.data(), rows,
                         h.num_dir_bins);
            }

            for (int class_index = 0; class_index < classes; class_index++) {
                // Pass 1 (precomputed): candidate rows above threshold.
                std::vector<int32_t> candidates =
                        std::move(cand_lists[class_index]);
                const float *scores =
                        all_scores.data() + class_index;  // stride = classes
                if (candidates.empty()) continue;
                if ((int)candidates.size() > h.nms_pre_maxsize) {
                    std::partial_sort(candidates.begin(),
                                      candidates.begin() + h.nms_pre_maxsize,
                                      candidates.end(),
                                      [&](int32_t a, int32_t b) {
                                          return scores[(size_t)a * classes] >
                                                 scores[(size_t)b * classes];
                                      });
                    candidates.resize(h.nms_pre_maxsize);
                }
                // Threshold-first decode: build compact per-candidate inputs
                // and decode only candidate rows instead of the whole anchor
                // grid. decode_boxes / apply_direction_classifier are
                // deterministic per row, so with the same threshold +
                // nms_pre_maxsize the detections are bit-identical to decoding
                // the full grid first.
                const int n_cand = (int)candidates.size();
                std::vector<float> cand_box_sel((size_t)n_cand * h.code_size);
                std::vector<float> cand_anchor((size_t)n_cand * 7);
                std::vector<int32_t> cand_comp((size_t)n_cand);
                for (int i = 0; i < n_cand; i++) {
                    const int row = candidates[i];
                    const int comp = row_to_compact[(size_t)row];
                    cand_comp[(size_t)i] = comp;
                    std::copy_n(cand_box.data() + (size_t)comp * h.code_size,
                                h.code_size,
                                cand_box_sel.data() + (size_t)i * h.code_size);
                    std::copy_n(anchors.anchors.data() + (size_t)row * 7, 7,
                                cand_anchor.data() + (size_t)i * 7);
                }
                std::vector<float> candidate_boxes = decode_boxes(
                        h, cand_box_sel.data(), 1, n_cand, cand_anchor.data());
                if (!cand_dir.empty()) {
                    std::vector<float> cand_dir_sel((size_t)n_cand *
                                                    h.num_dir_bins);
                    for (int i = 0; i < n_cand; i++) {
                        std::copy_n(
                                cand_dir.data() + (size_t)cand_comp[(size_t)i] *
                                                          h.num_dir_bins,
                                h.num_dir_bins,
                                cand_dir_sel.data() +
                                        (size_t)i * h.num_dir_bins);
                    }
                    apply_direction_classifier(h, cand_dir_sel.data(), 1,
                                               n_cand, candidate_boxes);
                }
                std::vector<float> candidate_scores(n_cand);
                for (int i = 0; i < n_cand; i++)
                    candidate_scores[i] =
                            scores[(size_t)candidates[i] * classes];
                const std::vector<int32_t> kept =
                        nms_3d(candidate_boxes, candidate_scores, h.nms_thresh,
                               h.max_boxes);
                for (int32_t kept_index : kept) {
                    const float *box_data =
                            candidate_boxes.data() + (size_t)kept_index * 7;
                    detection detection_value;
                    detection_value.x = box_data[0];
                    detection_value.y = box_data[1];
                    detection_value.z = box_data[2];
                    detection_value.dx = box_data[3];
                    detection_value.dy = box_data[4];
                    detection_value.dz = box_data[5];
                    detection_value.heading = box_data[6];
                    detection_value.score = candidate_scores[kept_index];
                    detection_value.label =
                            h.multi_label_indices[label_offset + class_index];
                    out_dets.push_back(detection_value);
                }
            }
            label_offset += classes;
            set_offset += classes;
        }
        if (sink) {
            for (ggml_tensor *tensor : tap_list) {
                const std::vector<float> whc = readback_f32(tensor);
                std::vector<float> hwc =
                        whc_to_hwc(whc.data(), (int)tensor->ne[0],
                                   (int)tensor->ne[1], (int)tensor->ne[2]);
                sink(ggml_get_name(tensor), hwc.data(),
                     tensor->ne[0] * tensor->ne[1], tensor->ne[2]);
            }
        }
        lap("multi_decode_nms");

        return true;
    }

    // --- 5. Read back head outputs ---
    // Layout: head.cls is [W, H, A*num_class, N], head.box is [W, H,
    // A*code_size, N]. After PyTorch's permute(0,2,3,1) the layout would be [N,
    // H, W, A*C'] — the engine reads in [W, H, A*C', N] order and indexes
    // accordingly.
    const int A_per_loc = h.num_anchors_per_location();
    const int num_class = h.num_class;
    const int code_size = h.code_size;
    const int num_dir = h.num_dir_bins;
    const int fm_w = (int)head.cls->ne[0];
    const int fm_h = (int)head.cls->ne[1];
    const int A_total = fm_w * fm_h * A_per_loc;

    const std::vector<float> cls_whc = readback_f32(head.cls);
    const std::vector<float> box_whc = readback_f32(head.box);
    const std::vector<float> dir_whc =
            head.dir ? readback_f32(head.dir) : std::vector<float>();
    std::vector<float> cls_raw =
            whc_to_hwc(cls_whc.data(), fm_w, fm_h, A_per_loc * num_class);
    std::vector<float> box_raw =
            whc_to_hwc(box_whc.data(), fm_w, fm_h, A_per_loc * code_size);
    std::vector<float> dir_raw = head.dir
                                         ? whc_to_hwc(dir_whc.data(), fm_w,
                                                      fm_h, A_per_loc * num_dir)
                                         : std::vector<float>();
    lap("readback");

    if (sink) {
        sink("head.cls", cls_raw.data(), (int64_t)fm_w * fm_h * A_per_loc,
             num_class);
        sink("head.box", box_raw.data(), (int64_t)fm_w * fm_h * A_per_loc,
             code_size);
        if (head.dir)
            sink("head.dir", dir_raw.data(), (int64_t)fm_w * fm_h * A_per_loc,
                 num_dir);
        for (ggml_tensor *t : tap_list) {
            const std::vector<float> whc = readback_f32(t);
            std::vector<float> hwc = whc_to_hwc(whc.data(), (int)t->ne[0],
                                                (int)t->ne[1], (int)t->ne[2]);
            sink(ggml_get_name(t), hwc.data(), t->ne[0] * t->ne[1], t->ne[2]);
        }
    }

    // --- 6. Generate anchors (CPU) — frame-invariant, cached after first frame
    // ---
    if (!anchors_cached || cached_anchors.feature_map_w != fm_w ||
        cached_anchors.feature_map_h != fm_h) {
        cached_anchors = generate_anchors(h, fm_w, fm_h);
        anchors_cached = true;
    }
    const anchor_grid &ag = cached_anchors;
    if (ag.total != A_total) {
        error = "anchor count mismatch: engine=" + std::to_string(A_total) +
                " vs generator=" + std::to_string(ag.total);
        return false;
    }

    // --- 7. Decode boxes (CPU) ---
    // The head output is [W, H, A, C'] in row-major with W fastest — the
    // anchor grid from generate_anchors is in the same [y, x, A_per_loc] order
    // (y outer, x middle, A_per_loc inner). decode_boxes takes [N=1, A,
    // code_size].
    std::vector<float> boxes = decode_boxes(h, box_raw.data(), /*N=*/1, A_total,
                                            ag.anchors.data());
    lap("decode");

    // Apply direction classifier
    if (head.dir) {
        apply_direction_classifier(h, dir_raw.data(), /*N=*/1, A_total, boxes);
        lap("dir_cls");
    }

    // --- 8. Sigmoid cls, take max-class, threshold ---
    // OpenPCDet post_processing: max over classes (or class-agnostic max),
    // then class_agnostic_nms with score_thresh.
    std::vector<float> scores(A_total);
    std::vector<int32_t> labels(A_total);
    for (int a = 0; a < A_total; a++) {
        const float *cls = cls_raw.data() + (size_t)a * num_class;
        int best_c = 0;
        float best_s = cls[0];
        for (int c = 1; c < num_class; c++) {
            if (cls[c] > best_s) {
                best_s = cls[c];
                best_c = c;
            }
        }
        // sigmoid
        scores[a] = 1.0f / (1.0f + std::exp(-best_s));
        labels[a] = best_c + 1;  // OpenPCDet uses 1-indexed labels
    }
    lap("sigmoid");

    // Score threshold
    std::vector<int32_t> cand;
    cand.reserve(A_total);
    for (int a = 0; a < A_total; a++) {
        if (scores[a] >= h.score_thresh) cand.push_back(a);
    }
    if (cand.empty()) {
        return true;
    }
    if ((int)cand.size() > h.nms_pre_maxsize) {
        std::partial_sort(
                cand.begin(), cand.begin() + h.nms_pre_maxsize, cand.end(),
                [&](int32_t a, int32_t b) { return scores[a] > scores[b]; });
        cand.resize(h.nms_pre_maxsize);
    }

    std::vector<float> cand_boxes(cand.size() * 7);
    std::vector<float> cand_scores(cand.size());
    std::vector<int32_t> cand_labels(cand.size());
    for (size_t i = 0; i < cand.size(); i++) {
        const int a = cand[i];
        std::memcpy(&cand_boxes[i * 7], &boxes[(size_t)a * 7],
                    7 * sizeof(float));
        cand_scores[i] = scores[a];
        cand_labels[i] = labels[a];
    }

    // --- 9. 3D NMS (CPU) ---
    std::vector<int32_t> kept =
            nms_3d(cand_boxes, cand_scores, h.nms_thresh, h.max_boxes);
    lap("nms");

    out_dets.reserve(kept.size());
    for (int32_t idx : kept) {
        detection d;
        d.x = cand_boxes[(size_t)idx * 7 + 0];
        d.y = cand_boxes[(size_t)idx * 7 + 1];
        d.z = cand_boxes[(size_t)idx * 7 + 2];
        d.dx = cand_boxes[(size_t)idx * 7 + 3];
        d.dy = cand_boxes[(size_t)idx * 7 + 4];
        d.dz = cand_boxes[(size_t)idx * 7 + 5];
        d.heading = cand_boxes[(size_t)idx * 7 + 6];
        d.score = cand_scores[idx];
        d.label = cand_labels[idx];
        out_dets.push_back(d);
    }

    lap("total");
    return true;
}

}  // namespace openpcdet
