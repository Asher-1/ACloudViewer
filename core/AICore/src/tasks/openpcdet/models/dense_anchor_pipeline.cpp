#include "dense_anchor_pipeline.h"

#include "../ops/nms_3d.h"
#include "../extensions/center_head_ops.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace openpcdet {

namespace {

// Batched readback: enqueue all head branches with async D2H copies and pay
// exactly one stream sync for the whole batch. The per-branch synchronous
// tensor_get measured as ~36 stream syncs per frame on the multihead models,
// each serializing against the graph replay.
struct readback_batch {
    ggml_backend_t backend = nullptr;
    struct item {
        ggml_tensor * tensor = nullptr;
        std::vector<float> data;
    };
    std::vector<item> items;

    item & enqueue(ggml_backend_t be, ggml_tensor * tensor) {
        backend = backend ? backend : be;
        items.push_back({tensor, std::vector<float>(static_cast<size_t>(ggml_nelements(tensor)))});
        item & it = items.back();
        if (ggml_is_contiguous(tensor)) {
            ggml_backend_tensor_get_async(be, tensor, it.data.data(), 0,
                                          it.data.size() * sizeof(float));
        } else {
            // Rare padded/strided branch: fall back to the synchronous bulk
            // read (one sync) inside finish(); the async path needs a pinned
            // staging buffer to be worthwhile and these are not hot.
            ggml_backend_tensor_get(tensor, it.data.data(), 0,
                                    it.data.size() * sizeof(float));
        }
        return it;
    }
    void finish() {
        if (backend && !items.empty()) ggml_backend_synchronize(backend);
    }
};

std::vector<float> readback_f32(ggml_tensor * tensor) {
    std::vector<float> output(static_cast<size_t>(ggml_nelements(tensor)));
    if (ggml_is_contiguous(tensor)) {
        ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
        return output;
    }
    // Non-contiguous (padded/strided) head output: one bulk D2H of the whole
    // memory span including the padding gaps, then a host-side repack. The
    // previous per-row get cost one stream sync + one memcpy per row, which
    // measured as ~36 syncs per frame on the multihead models.
    const int64_t rows = tensor->ne[3] * tensor->ne[2] * tensor->ne[1];
    const size_t   esz = ggml_type_size(tensor->type);
    const size_t   span = static_cast<size_t>(tensor->nb[1]) * (rows - 1) +
                          static_cast<size_t>(tensor->ne[0]) * esz;
    std::vector<char> raw(span);
    ggml_backend_tensor_get(tensor, raw.data(), 0, span);
    for (int64_t row = 0; row < rows; ++row) {
        std::memcpy(output.data() + static_cast<size_t>(row) * tensor->ne[0],
                    raw.data() + static_cast<size_t>(row) * tensor->nb[1],
                    static_cast<size_t>(tensor->ne[0]) * esz);
    }
    return output;
}

std::vector<float> whc_to_anchor_major(const std::vector<float> & source,
                                       int width, int height,
                                       int anchors, int channels) {
    std::vector<float> output(static_cast<size_t>(width) * height * anchors * channels);
    for (int anchor = 0; anchor < anchors; ++anchor)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int channel = 0; channel < channels; ++channel) {
                    const int source_channel = anchor * channels + channel;
                    const size_t source_index = static_cast<size_t>(x) + static_cast<size_t>(width) *
                        (y + static_cast<size_t>(height) * source_channel);
                    const size_t target_row = (static_cast<size_t>(anchor) * height + y) * width + x;
                    output[target_row * channels + channel] = source[source_index];
                }
    return output;
}

} // namespace

bool run_dense_anchor_raw(engine_backend & backend,
                          const hparams & hp,
                          const bev_backbone_weights & backbone,
                          const anchor_head_weights & head_weights,
                          const std::vector<float> & spatial_hwc,
                          int width, int height, bool keep_features,
                          dense_anchor_result & result,
                          const tap_sink & sink,
                          std::string & error) {
    result = {};
    if (hp.head_kind != dense_head_kind::anchor_single || width <= 0 || height <= 0 ||
        spatial_hwc.size() != static_cast<size_t>(width) * height * hp.num_bev_features) {
        error = "invalid dense AnchorHeadSingle input";
        return false;
    }
    constexpr size_t graph_nodes = 4096;
    ggml_init_params params = {
        ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
        nullptr, true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        error = "dense graph context init failed";
        return false;
    }
    // HWC row-major input (channel fastest) is uploaded as-is into [C, W, H];
    // the GPU permute+cont below performs the HWC->WHC transpose (see
    // run_dense_center_pipeline for the permute axis rationale).
    ggml_tensor * spatial = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, hp.num_bev_features, width, height, 1);
    ggml_set_input(spatial);
    ggml_tensor * spatial_whc = ggml_cont(ctx, ggml_permute(ctx, spatial, 2, 0, 1, 3));
    std::vector<ggml_tensor *> taps;
    ggml_tensor * spatial_2d = build_bev_backbone_graph(
        ctx, backbone, hp, spatial_whc, sink ? &taps : nullptr);
    head_outputs head = build_anchor_head_graph(ctx, head_weights, hp, spatial_2d);
    ggml_set_output(head.cls);
    ggml_set_output(head.box);
    if (head.dir) ggml_set_output(head.dir);
    if (sink || keep_features) {
        ggml_set_name(spatial_2d, "spatial_features_2d");
        ggml_set_output(spatial_2d);
        if (sink && (taps.empty() || taps.back() != spatial_2d)) taps.push_back(spatial_2d);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    ggml_build_forward_expand(graph, head.cls);
    ggml_build_forward_expand(graph, head.box);
    if (head.dir) ggml_build_forward_expand(graph, head.dir);
    for (ggml_tensor * tap : taps) ggml_build_forward_expand(graph, tap);
    if (!ggml_gallocr_alloc_graph(backend.galloc, graph)) {
        error = "dense graph allocation failed";
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(spatial, spatial_hwc.data(), 0, spatial_hwc.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
        error = "dense graph compute failed";
        ggml_free(ctx);
        return false;
    }

    const int anchors_per_location = hp.num_anchors_per_location();
    const int feature_width = static_cast<int>(head.cls->ne[0]);
    const int feature_height = static_cast<int>(head.cls->ne[1]);
    const int anchor_count = feature_width * feature_height * anchors_per_location;
    const std::vector<float> cls_whc = readback_f32(head.cls);
    const std::vector<float> box_whc = readback_f32(head.box);
    const std::vector<float> dir_whc = head.dir ? readback_f32(head.dir) : std::vector<float>();
    std::vector<float> cls = whc_to_hwc(cls_whc.data(), feature_width, feature_height,
                                        anchors_per_location * hp.num_class);
    std::vector<float> box = whc_to_hwc(box_whc.data(), feature_width, feature_height,
                                        anchors_per_location * hp.code_size);
    std::vector<float> direction = head.dir
        ? whc_to_hwc(dir_whc.data(), feature_width, feature_height,
                     anchors_per_location * hp.num_dir_bins)
        : std::vector<float>();
    if (keep_features) {
        const std::vector<float> feature_whc = readback_f32(spatial_2d);
        result.feature_width = static_cast<int>(spatial_2d->ne[0]);
        result.feature_height = static_cast<int>(spatial_2d->ne[1]);
        result.feature_channels = static_cast<int>(spatial_2d->ne[2]);
        result.spatial_features_2d = whc_to_hwc(
            feature_whc.data(), result.feature_width, result.feature_height,
            result.feature_channels);
    }
    if (sink) {
        sink("head.cls", cls.data(), anchor_count, hp.num_class);
        sink("head.box", box.data(), anchor_count, hp.code_size);
        if (!direction.empty()) sink("head.dir", direction.data(), anchor_count, hp.num_dir_bins);
        for (ggml_tensor * tap : taps) {
            const std::vector<float> whc = readback_f32(tap);
            std::vector<float> hwc = whc_to_hwc(
                whc.data(), static_cast<int>(tap->ne[0]), static_cast<int>(tap->ne[1]),
                static_cast<int>(tap->ne[2]));
            sink(ggml_get_name(tap), hwc.data(), tap->ne[0] * tap->ne[1], tap->ne[2]);
        }
    }

    anchor_grid anchors = generate_anchors(hp, feature_width, feature_height);
    if (anchors.total != anchor_count) {
        error = "dense head anchor count mismatch";
        ggml_free(ctx);
        return false;
    }
    std::vector<float> decoded = decode_boxes(hp, box.data(), 1, anchor_count, anchors.anchors.data());
    if (!direction.empty()) apply_direction_classifier(hp, direction.data(), 1, anchor_count, decoded);

    result.boxes = std::move(decoded);
    result.scores.resize(anchor_count);
    result.labels.resize(anchor_count);
    for (int row = 0; row < anchor_count; ++row) {
        const float * logits = cls.data() + static_cast<size_t>(row) * hp.num_class;
        int label = 0;
        for (int c = 1; c < hp.num_class; ++c) if (logits[c] > logits[label]) label = c;
        result.scores[row] = 1.0f / (1.0f + std::exp(-logits[label]));
        result.labels[row] = label + 1;
    }
    ggml_free(ctx);
    return true;
}

bool postprocess_dense_anchors(const dense_anchor_result & input,
                               float score_threshold, int pre_max, int post_max,
                               float nms_threshold,
                               std::vector<detection> & detections,
                               std::string & error) {
    detections.clear();
    const size_t count = input.scores.size();
    if (input.boxes.size() != count * 7 || input.labels.size() != count ||
        pre_max <= 0 || post_max <= 0 || nms_threshold < 0.0f) {
        error = "invalid dense anchor postprocess input";
        return false;
    }
    std::vector<int32_t> candidates;
    candidates.reserve(count);
    for (size_t row = 0; row < count; ++row)
        if (input.scores[row] >= score_threshold)
            candidates.push_back(static_cast<int32_t>(row));
    if (static_cast<int>(candidates.size()) > pre_max) {
        std::partial_sort(candidates.begin(), candidates.begin() + pre_max,
                          candidates.end(), [&](int32_t a, int32_t b) {
                              return input.scores[a] > input.scores[b];
                          });
        candidates.resize(pre_max);
    }
    std::vector<float> candidate_boxes(candidates.size() * 7);
    std::vector<float> candidate_scores(candidates.size());
    std::vector<int32_t> candidate_labels(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        const int row = candidates[i];
        std::memcpy(candidate_boxes.data() + i * 7,
                    input.boxes.data() + static_cast<size_t>(row) * 7,
                    7 * sizeof(float));
        candidate_scores[i] = input.scores[row];
        candidate_labels[i] = input.labels[row];
    }
    const std::vector<int32_t> kept = nms_3d(
        candidate_boxes, candidate_scores, nms_threshold, post_max);
    detections.reserve(kept.size());
    for (int32_t index : kept) {
        const float * box_data = candidate_boxes.data() + static_cast<size_t>(index) * 7;
        detection value;
        value.x = box_data[0]; value.y = box_data[1]; value.z = box_data[2];
        value.dx = box_data[3]; value.dy = box_data[4]; value.dz = box_data[5];
        value.heading = box_data[6]; value.score = candidate_scores[index];
        value.label = candidate_labels[index];
        detections.push_back(value);
    }
    return true;
}

bool run_dense_anchor_pipeline(engine_backend & backend,
                               const hparams & hp,
                               const bev_backbone_weights & backbone,
                               const anchor_head_weights & head_weights,
                               const std::vector<float> & spatial_hwc,
                               int width, int height,
                               std::vector<detection> & detections,
                               const tap_sink & sink,
                               std::string & error) {
    dense_anchor_result raw;
    return run_dense_anchor_raw(backend, hp, backbone, head_weights,
                                spatial_hwc, width, height, false,
                                raw, sink, error) &&
           postprocess_dense_anchors(raw, hp.score_thresh, hp.nms_pre_maxsize,
                                     hp.max_boxes, hp.nms_thresh,
                                     detections, error);
}

bool run_dense_center_pipeline(engine_backend & backend,
                               const hparams & hp,
                               const bev_backbone_weights & backbone,
                               const center_head_weights & head_weights,
                               const std::vector<float> & spatial_hwc,
                               int width, int height,
                               std::vector<detection> & detections,
                               const tap_sink & sink,
                               std::string & error,
                               dense_center_cache * cache,
                               bool spatial_device_ready,
                               bool direct_whc_input) {
    // Debug-only stage profiler (stderr, OPCPD_PROFILE): diagnostics only,
    // default off, never changes task behavior.
    const bool profile = std::getenv("OPCPD_PROFILE") != nullptr;
    auto lap = [&](const char * name, const std::chrono::high_resolution_clock::time_point & from) {
        if (!profile) return;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - from).count();
        std::fprintf(stderr, "[dense] %-24s %8.2f ms\n", name, ms);
    };
    auto t_enter = std::chrono::high_resolution_clock::now();
    (void) t_enter;
    const size_t expected_elements =
        static_cast<size_t>(width) * height * hp.num_bev_features;
    if (hp.head_kind != dense_head_kind::center || width <= 0 || height <= 0 ||
        (!spatial_device_ready && spatial_hwc.size() != expected_elements)) {
        error = "invalid dense CenterHead input";
        return false;
    }

    // Frame-to-frame the graph topology is identical (fixed spatial grid), so
    // with a cache we build + gallocr-allocate once and then only re-upload the
    // input. The tap sink changes the topology, so sink mode never caches.
    const bool reuse = cache != nullptr && !sink && cache->ctx != nullptr;
    if (reuse && cache->direct_whc_input != direct_whc_input) {
        error = "CenterHead cache input layout mismatch";
        return false;
    }
    constexpr size_t graph_nodes = 4096;
    ggml_context * ctx = nullptr;
    std::vector<ggml_tensor *> taps;
    std::vector<center_single_outputs> outputs;
    ggml_tensor * spatial = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<std::vector<float>> readback_buffers;

    if (!reuse) {
        ggml_init_params params = {
            ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
            nullptr, true,
        };
        ctx = ggml_init(params);
        if (!ctx) { error = "CenterHead graph context init failed"; return false; }
    } else {
        ctx     = cache->ctx;
        graph   = cache->graph;
        spatial = cache->spatial;
        outputs = cache->outputs;
        readback_buffers.swap(cache->readback_buffers);
    }
    auto t_graph_start = std::chrono::high_resolution_clock::now();
    // CenterPoint's device pipeline writes GGML-native WHC directly into the
    // persistent input tensor, avoiding a full 33 MB HWC->WHC transpose every
    // frame. Other callers retain the HWC input contract and graph transpose.
    if (!reuse) {
        spatial = direct_whc_input
            ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height,
                                 hp.num_bev_features, 1)
            : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hp.num_bev_features,
                                 width, height, 1);
        ggml_set_input(spatial);
        ggml_tensor * spatial_whc = direct_whc_input
            ? spatial
            : ggml_cont(ctx, ggml_permute(ctx, spatial, 2, 0, 1, 3));
        ggml_tensor * spatial_2d = build_bev_backbone_graph(
            ctx, backbone, hp, spatial_whc, sink ? &taps : nullptr);
        outputs = build_center_head_graph(ctx, head_weights, hp, spatial_2d);
        for (auto & output : outputs) {
            ggml_set_output(output.heatmap); ggml_set_output(output.center);
            ggml_set_output(output.center_z); ggml_set_output(output.dimensions);
            ggml_set_output(output.rotation);
            if (output.velocity) ggml_set_output(output.velocity);
        }
        if (sink) {
            ggml_set_name(spatial_2d, "spatial_features_2d");
            ggml_set_output(spatial_2d);
            if (taps.empty() || taps.back() != spatial_2d) taps.push_back(spatial_2d);
        }
        graph = ggml_new_graph_custom(ctx, graph_nodes, false);
        for (auto & output : outputs) {
            ggml_build_forward_expand(graph, output.heatmap);
            ggml_build_forward_expand(graph, output.center);
            ggml_build_forward_expand(graph, output.center_z);
            ggml_build_forward_expand(graph, output.dimensions);
            ggml_build_forward_expand(graph, output.rotation);
            if (output.velocity) ggml_build_forward_expand(graph, output.velocity);
        }
        for (ggml_tensor * tap : taps) ggml_build_forward_expand(graph, tap);
        if (!ggml_gallocr_alloc_graph(backend.galloc, graph)) {
            error = "CenterHead graph allocation failed";
            if (!cache) ggml_free(ctx);
            return false;
        }
        if (cache) {
            cache->ctx = ctx; cache->graph = graph; cache->spatial = spatial;
            cache->direct_whc_input = direct_whc_input;
            cache->outputs = outputs;
        }
    }
    if (spatial_device_ready) {
        if (!reuse) {
            error = "device-resident CenterHead input requires a populated graph cache";
            if (!cache) ggml_free(ctx);
            return false;
        }
    } else {
        if (direct_whc_input) {
            std::vector<float> spatial_whc(expected_elements);
            for (int channel = 0; channel < hp.num_bev_features; ++channel)
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x)
                        spatial_whc[static_cast<size_t>(x) + static_cast<size_t>(width) *
                            (y + static_cast<size_t>(height) * channel)] =
                            spatial_hwc[(static_cast<size_t>(y) * width + x) *
                                        hp.num_bev_features + channel];
            ggml_backend_tensor_set(spatial, spatial_whc.data(), 0,
                                    spatial_whc.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(
                spatial, spatial_hwc.data(), 0, spatial_hwc.size() * sizeof(float));
        }
    }
    lap("graph alloc+upload", t_graph_start);
    if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
        error = "CenterHead graph compute failed";
        if (cache) cache->invalidate();
        ggml_free(ctx); return false;
    }
    lap("graph compute", t_graph_start);

    ext::center_decode_params decode;
    decode.point_cloud_range = hp.point_cloud_range;
    decode.voxel_size = hp.voxel_size;
    decode.feature_map_stride = hp.center_feature_map_stride;
    decode.max_objects = hp.center_max_objects;
    decode.score_threshold = hp.score_thresh;
    decode.post_center_limit_range = hp.center_post_limit_range;
    decode.nms_threshold = hp.nms_thresh;
    decode.nms_pre_maxsize = hp.nms_pre_maxsize;
    decode.nms_post_maxsize = hp.max_boxes;
    detections.clear();
    auto t_decode = std::chrono::high_resolution_clock::now();
    size_t label_offset = 0;
    // Batch the device->host copies: issue every tensor readback asynchronously
    // on the backend stream, then synchronize once. 36 separate synchronous
    // cudaMemcpy calls cost ~150us each (fixed sync overhead) on this model;
    // a single sync after the batch drops readback from ~5.5ms to ~1.5ms.
    std::vector<ggml_tensor *> readback_tensors;
    auto queue_readback = [&](ggml_tensor * tensor) {
        const size_t slot = readback_tensors.size();
        readback_tensors.push_back(tensor);
        const size_t elements = static_cast<size_t>(ggml_nelements(tensor));
        if (slot == readback_buffers.size()) readback_buffers.emplace_back(elements);
        else readback_buffers[slot].resize(elements);
    };
    for (const auto & output : outputs) {
        queue_readback(output.heatmap); queue_readback(output.center);
        queue_readback(output.center_z); queue_readback(output.dimensions);
        queue_readback(output.rotation);
        if (output.velocity) queue_readback(output.velocity);
    }
    readback_buffers.resize(readback_tensors.size());
    for (size_t i = 0; i < readback_tensors.size(); ++i) {
        ggml_backend_tensor_get_async(backend.be, readback_tensors[i],
                                      readback_buffers[i].data(), 0,
                                      readback_buffers[i].size() * sizeof(float));
    }
    ggml_backend_synchronize(backend.be);
    const auto t_readback_done = std::chrono::high_resolution_clock::now();
    lap("readback", t_decode);
    for (size_t head_index = 0; head_index < outputs.size(); ++head_index) {
        const auto & output = outputs[head_index];
        const int output_width = static_cast<int>(output.heatmap->ne[0]);
        const int output_height = static_cast<int>(output.heatmap->ne[1]);
        const int classes = hp.center_class_counts[head_index];
        if (label_offset + classes > hp.center_label_indices.size()) {
            error = "CenterHead flattened class mapping overflow";
            if (cache) cache->invalidate();
            ggml_free(ctx); return false;
        }
        // Decode reads the GGML WHC layout directly (contiguous location walk);
        // sink (tap dump) transposes to HWC below if enabled.
        const size_t buffer_index = head_index * (hp.center_has_velocity ? 6 : 5);
        auto buffer = [&](int slot) -> const std::vector<float> & {
            return readback_buffers[buffer_index + slot];
        };
        const std::vector<float> & heatmap = buffer(0);
        const std::vector<float> & center = buffer(1);
        const std::vector<float> & center_z = buffer(2);
        const std::vector<float> & dimensions = buffer(3);
        const std::vector<float> & rotation = buffer(4);
        const std::vector<float> & velocity = hp.center_has_velocity ? buffer(5) : std::vector<float>();
        ext::center_head_view view;
        view.heatmap = heatmap.data(); view.center = center.data();
        view.center_z = center_z.data(); view.dimensions = dimensions.data();
        view.rotation = rotation.data();
        view.velocity = velocity.empty() ? nullptr : velocity.data();
        view.height = output_height; view.width = output_width; view.class_count = classes;
        view.whc_layout = true;
        view.class_mapping.assign(hp.center_label_indices.begin() + label_offset,
                                  hp.center_label_indices.begin() + label_offset + classes);
        std::vector<detection> head_detections;
        if (!ext::decode_center_head(view, decode, head_detections, error)) {
            if (cache) cache->invalidate();
            ggml_free(ctx); return false;
        }
        detections.insert(detections.end(), head_detections.begin(), head_detections.end());
        if (sink) {
            const std::string prefix = "head." + std::to_string(head_index);
            const int64_t rows = static_cast<int64_t>(output_width) * output_height;
            // sink expects HWC; transpose the WHC readback on the fly.
            const std::vector<float> hm_hwc = whc_to_hwc(
                heatmap.data(), output_width, output_height, classes);
            const std::vector<float> center_hwc = whc_to_hwc(
                center.data(), output_width, output_height, 2);
            const std::vector<float> center_z_hwc = whc_to_hwc(
                center_z.data(), output_width, output_height, 1);
            const std::vector<float> dim_hwc = whc_to_hwc(
                dimensions.data(), output_width, output_height, 3);
            const std::vector<float> rot_hwc = whc_to_hwc(
                rotation.data(), output_width, output_height, 2);
            sink(prefix + ".hm", hm_hwc.data(), rows, classes);
            sink(prefix + ".center", center_hwc.data(), rows, 2);
            sink(prefix + ".center_z", center_z_hwc.data(), rows, 1);
            sink(prefix + ".dim", dim_hwc.data(), rows, 3);
            sink(prefix + ".rot", rot_hwc.data(), rows, 2);
            if (!velocity.empty()) {
                const std::vector<float> vel_hwc = whc_to_hwc(
                    velocity.data(), output_width, output_height, 2);
                sink(prefix + ".vel", vel_hwc.data(), rows, 2);
            }
        }
        label_offset += classes;
    }
    lap("decode", t_readback_done);
    lap("readback+decode", t_decode);
    if (sink) {
        for (ggml_tensor * tap : taps) {
            const std::vector<float> whc = readback_f32(tap);
            std::vector<float> hwc = whc_to_hwc(
                whc.data(), static_cast<int>(tap->ne[0]), static_cast<int>(tap->ne[1]),
                static_cast<int>(tap->ne[2]));
            sink(ggml_get_name(tap), hwc.data(), tap->ne[0] * tap->ne[1], tap->ne[2]);
        }
    }
    if (cache) {
        readback_buffers.swap(cache->readback_buffers);  // keep scratch for next frame
    } else {
        ggml_free(ctx);
    }
    return true;
}

dense_center_cache::~dense_center_cache() {
    if (ctx) ggml_free(ctx);
}

void dense_center_cache::invalidate() {
    ctx = nullptr;
    graph = nullptr;
    spatial = nullptr;
    direct_whc_input = false;
    outputs.clear();
    readback_buffers.clear();
}

bool run_dense_anchor_multi_pipeline(engine_backend & backend,
                                     const hparams & hp,
                                     const bev_backbone_weights & backbone,
                                     const anchor_multi_weights & head_weights,
                                     const std::vector<float> & spatial_hwc,
                                     int width, int height,
                                     std::vector<detection> & detections,
                                     const tap_sink & sink,
                                     std::string & error) {
    if (hp.head_kind != dense_head_kind::anchor_multi || width <= 0 || height <= 0 ||
        spatial_hwc.size() != static_cast<size_t>(width) * height * hp.num_bev_features) {
        error = "invalid dense AnchorHeadMulti input";
        return false;
    }
    std::vector<float> spatial_whc(spatial_hwc.size());
    for (int channel = 0; channel < hp.num_bev_features; ++channel)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                spatial_whc[static_cast<size_t>(x) + static_cast<size_t>(width) *
                    (y + static_cast<size_t>(height) * channel)] =
                    spatial_hwc[(static_cast<size_t>(y) * width + x) * hp.num_bev_features + channel];

    constexpr size_t graph_nodes = 4096;
    ggml_init_params params = {
        ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
        nullptr, true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { error = "AnchorHeadMulti graph context init failed"; return false; }
    ggml_tensor * spatial = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, width, height, hp.num_bev_features, 1);
    ggml_set_input(spatial);
    std::vector<ggml_tensor *> taps;
    ggml_tensor * spatial_2d = build_bev_backbone_graph(
        ctx, backbone, hp, spatial, sink ? &taps : nullptr);
    std::vector<anchor_multi_single_outputs> outputs = build_anchor_multi_graph(
        ctx, head_weights, hp, spatial_2d);
    for (auto & output : outputs) {
        ggml_set_output(output.classification);
        for (ggml_tensor * branch : output.regression) ggml_set_output(branch);
        if (output.box) ggml_set_output(output.box);
        if (output.direction) ggml_set_output(output.direction);
    }
    if (sink) {
        ggml_set_name(spatial_2d, "spatial_features_2d");
        ggml_set_output(spatial_2d);
        if (taps.empty() || taps.back() != spatial_2d) taps.push_back(spatial_2d);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    for (auto & output : outputs) {
        ggml_build_forward_expand(graph, output.classification);
        for (ggml_tensor * branch : output.regression) ggml_build_forward_expand(graph, branch);
        if (output.box) ggml_build_forward_expand(graph, output.box);
        if (output.direction) ggml_build_forward_expand(graph, output.direction);
    }
    for (ggml_tensor * tap : taps) ggml_build_forward_expand(graph, tap);
    if (!ggml_gallocr_alloc_graph(backend.galloc, graph)) {
        error = "AnchorHeadMulti graph allocation failed";
        ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(spatial, spatial_whc.data(), 0, spatial_whc.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
        error = "AnchorHeadMulti graph compute failed";
        ggml_free(ctx); return false;
    }

    detections.clear();
    size_t label_offset = 0;
    size_t set_offset = 0;
    // Phase 1 - enqueue every branch of every head as one async D2H batch so
    // the whole decode pays a single stream sync (phase 2 is the finish()).
    readback_batch rb;
    struct head_slots { size_t cls; std::vector<size_t> reg; size_t box; size_t dir; };
    std::vector<head_slots> slots;
    slots.reserve(outputs.size());
    for (const auto & output : outputs) {
        head_slots sl{};
        sl.cls = rb.items.size();
        rb.enqueue(backend.be, output.classification);
        sl.reg.reserve(output.regression.size());
        for (const auto & branch : output.regression) {
            sl.reg.push_back(rb.items.size());
            rb.enqueue(backend.be, branch);
        }
        sl.box = rb.items.size();
        if (output.box) rb.enqueue(backend.be, output.box);
        sl.dir = rb.items.size();
        if (output.direction) rb.enqueue(backend.be, output.direction);
        slots.push_back(std::move(sl));
    }
    rb.finish();
    for (size_t head_index = 0; head_index < outputs.size(); ++head_index) {
        const auto & output = outputs[head_index];
        const int classes = hp.multi_head_class_counts[head_index];
        const int anchors_per_location = hp.multi_head_anchor_counts[head_index];
        const int output_width = static_cast<int>(output.classification->ne[0]);
        const int output_height = static_cast<int>(output.classification->ne[1]);
        const int rows = output_width * output_height * anchors_per_location;
        if (label_offset + classes > hp.multi_label_indices.size() ||
            set_offset + classes > hp.multi_anchor_set_indices.size()) {
            error = "AnchorHeadMulti flattened metadata overflow";
            ggml_free(ctx); return false;
        }
        std::vector<int32_t> anchor_sets(
            hp.multi_anchor_set_indices.begin() + set_offset,
            hp.multi_anchor_set_indices.begin() + set_offset + classes);
        anchor_grid anchors = generate_multihead_anchors(
            hp, anchor_sets, output_width, output_height);
        if (anchors.total != rows) {
            error = "AnchorHeadMulti anchor count mismatch";
            ggml_free(ctx); return false;
        }
        std::vector<float> classification = whc_to_anchor_major(
            std::move(rb.items[slots[head_index].cls].data), output_width, output_height,
            anchors_per_location, classes);
        std::vector<float> encoded(static_cast<size_t>(rows) * hp.code_size);
        if (hp.multi_separate_reg) {
            int code_offset = 0;
            if (output.regression.size() != hp.multi_reg_channels.size()) {
                error = "AnchorHeadMulti regression branch count mismatch";
                ggml_free(ctx); return false;
            }
            for (size_t branch = 0; branch < output.regression.size(); ++branch) {
                const int channels = hp.multi_reg_channels[branch];
                std::vector<float> values = whc_to_anchor_major(
                    std::move(rb.items[slots[head_index].reg[branch]].data), output_width,
                    output_height, anchors_per_location, channels);
                for (int row = 0; row < rows; ++row)
                    std::copy_n(values.data() + static_cast<size_t>(row) * channels, channels,
                                encoded.data() + static_cast<size_t>(row) * hp.code_size + code_offset);
                code_offset += channels;
            }
            if (code_offset != hp.code_size) {
                error = "AnchorHeadMulti regression code size mismatch";
                ggml_free(ctx); return false;
            }
        } else {
            encoded = whc_to_anchor_major(std::move(rb.items[slots[head_index].box].data),
                                          output_width, output_height, anchors_per_location,
                                          hp.code_size);
        }
        std::vector<float> direction = output.direction
            ? whc_to_anchor_major(std::move(rb.items[slots[head_index].dir].data), output_width,
                                  output_height, anchors_per_location, hp.num_dir_bins)
            : std::vector<float>();
        if (sink) {
            const std::string prefix = "head." + std::to_string(head_index);
            sink(prefix + ".cls", classification.data(), rows, classes);
            sink(prefix + ".box", encoded.data(), rows, hp.code_size);
            if (!direction.empty()) sink(prefix + ".dir", direction.data(), rows, hp.num_dir_bins);
        }
        std::vector<float> decoded = decode_boxes(
            hp, encoded.data(), 1, rows, anchors.anchors.data());
        if (!direction.empty()) apply_direction_classifier(hp, direction.data(), 1, rows, decoded);
        for (int class_index = 0; class_index < classes; ++class_index) {
            std::vector<int32_t> candidates;
            std::vector<float> scores(rows);
            for (int row = 0; row < rows; ++row) {
                const float score = 1.0f / (1.0f + std::exp(
                    -classification[static_cast<size_t>(row) * classes + class_index]));
                scores[row] = score;
                if (score >= hp.score_thresh) candidates.push_back(row);
            }
            if (static_cast<int>(candidates.size()) > hp.nms_pre_maxsize) {
                std::partial_sort(candidates.begin(), candidates.begin() + hp.nms_pre_maxsize,
                                  candidates.end(), [&](int32_t a, int32_t b) {
                                      return scores[a] > scores[b];
                                  });
                candidates.resize(hp.nms_pre_maxsize);
            }
            std::vector<float> boxes(candidates.size() * 7);
            std::vector<float> candidate_scores(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i) {
                std::copy_n(decoded.data() + static_cast<size_t>(candidates[i]) * 7, 7,
                            boxes.data() + i * 7);
                candidate_scores[i] = scores[candidates[i]];
            }
            const std::vector<int32_t> kept = nms_3d(
                boxes, candidate_scores, hp.nms_thresh, hp.max_boxes);
            for (int32_t index : kept) {
                const float * box = boxes.data() + static_cast<size_t>(index) * 7;
                detection value;
                value.x = box[0]; value.y = box[1]; value.z = box[2];
                value.dx = box[3]; value.dy = box[4]; value.dz = box[5];
                value.heading = box[6]; value.score = candidate_scores[index];
                value.label = hp.multi_label_indices[label_offset + class_index];
                detections.push_back(value);
            }
        }
        label_offset += classes;
        set_offset += classes;
    }
    if (sink) {
        for (ggml_tensor * tap : taps) {
            const std::vector<float> whc = readback_f32(tap);
            std::vector<float> hwc = whc_to_hwc(
                whc.data(), static_cast<int>(tap->ne[0]), static_cast<int>(tap->ne[1]),
                static_cast<int>(tap->ne[2]));
            sink(ggml_get_name(tap), hwc.data(), tap->ne[0] * tap->ne[1], tap->ne[2]);
        }
    }
    ggml_free(ctx);
    return true;
}

} // namespace openpcdet
