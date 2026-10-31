// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "parta2.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../extensions/roi_ops.h"
#include "../ops/nms_3d.h"
#include "dense_anchor_pipeline.h"

namespace openpcdet {

namespace {

std::vector<point_mlp_layer_spec> hidden_specs(const std::string& prefix,
                                               size_t count,
                                               bool shared) {
    std::vector<point_mlp_layer_spec> result;
    for (size_t index = 0; index < count; ++index) {
        const int module = static_cast<int>(index) * 4;
        result.push_back({prefix + "." + std::to_string(module) + ".weight", "",
                          prefix + "." + std::to_string(module + 1), true});
    }
    (void)shared;
    return result;
}

bool load_sparse_layer(model_file& file,
                       const std::string& conv,
                       const std::string& bn,
                       sparse_unet_layer& layer,
                       std::string& error) {
    layer.weight = file.require((conv + ".weight").c_str());
    layer.scale = file.require((bn + ".eval_scale").c_str());
    layer.bias = file.require((bn + ".eval_bias").c_str());
    if (!file.error.empty()) {
        error = file.error;
        return false;
    }
    return true;
}

void remap_sparse_layer(sparse_unet_layer& layer, ggml_context* context) {
    if (!context) return;
    layer.weight = ggml_get_tensor(context, ggml_get_name(layer.weight));
    layer.scale = ggml_get_tensor(context, ggml_get_name(layer.scale));
    layer.bias = ggml_get_tensor(context, ggml_get_name(layer.bias));
}

bool run_roi_sparse(
        engine_backend& backend,
        const std::unordered_map<std::string, sparse_unet_layer>& weights,
        const std::vector<int32_t>& coordinates,
        const std::vector<float>& part_features,
        const std::vector<float>& rpn_features,
        int rpn_channels,
        std::vector<float>& output,
        std::string& error) {
    output.clear();
    const int64_t active = static_cast<int64_t>(coordinates.size()) / 4;
    if (active <= 0 ||
        part_features.size() != static_cast<size_t>(active) * 4 ||
        rpn_features.size() != static_cast<size_t>(active) * rpn_channels) {
        error = "bad PartA2 sparse RoI input";
        return false;
    }
    ext::sparse_tensor sparse;
    sparse.ndim = 3;
    sparse.spatial_shape = {12, 12, 12};
    sparse.coordinates = coordinates;
    sparse.channels = 4;
    ext::sparse_indice_map map;
    const ext::sparse_conv_params params{{3, 3, 3}, {1, 1, 1}, {1, 1, 1},
                                         {1, 1, 1}, true,      false};
    if (!ext::build_sparse_indice_map(sparse, params, map, error)) return false;
    constexpr size_t nodes = 512;
    ggml_init_params graph_params{
            ggml_tensor_overhead() * nodes +
                    ggml_graph_overhead_custom(nodes, false),
            nullptr, true};
    ggml_context* ctx = ggml_init(graph_params);
    if (!ctx) {
        error = "PartA2 RoI sparse context init failed";
        return false;
    }
    ggml_tensor* part = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, active);
    ggml_tensor* rpn =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rpn_channels, active);
    ggml_tensor* input_part = part;
    ggml_tensor* input_rpn = rpn;
    // AICore port note: the stock-op lowering gathers with get_rows over the
    // dense [kernel_volume, output_active] rulebook, whose -1 entries resolve
    // to one appended zero feature row; the upstream compact/scatter rulebooks
    // do not exist here. Both chains share the (channel-agnostic) rulebook.
    ggml_tensor* dense = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, map.kernel_volume * map.output_active_count());
    ggml_set_input(part);
    ggml_set_input(rpn);
    ggml_set_input(dense);
    auto apply = [&](const char* name, ggml_tensor* input) {
        const auto& layer = weights.at(name);
        ggml_tensor* zero_col = ggml_fill(
                ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input->ne[0], 1),
                0.0f);
        ggml_tensor* conv_in = ggml_concat(ctx, input, zero_col, 1);
        ggml_tensor* value = ext::build_sparse_convolution(
                ctx, conv_in, dense, nullptr, nullptr, map, layer.weight,
                nullptr);
        return ggml_relu(ctx, ggml_add(ctx, ggml_mul(ctx, value, layer.scale),
                                       layer.bias));
    };
    part = apply("part.0", part);
    part = apply("part.1", part);
    rpn = apply("rpn.0", rpn);
    rpn = apply("rpn.1", rpn);
    ggml_tensor* merged = ggml_concat(ctx, rpn, part, 0);
    ggml_set_output(merged);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(graph, merged);
    if (!ggml_gallocr_alloc_graph(backend.galloc, graph)) {
        error = "PartA2 RoI sparse graph allocation failed";
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(input_part, part_features.data(), 0,
                            part_features.size() * sizeof(float));
    ggml_backend_tensor_set(input_rpn, rpn_features.data(), 0,
                            rpn_features.size() * sizeof(float));
    {
        // AICore port note: -1 rulebook entries are remapped to the appended
        // zero row (index == active) before upload.
        std::vector<int32_t> remapped;
        ext::remap_dense_indices(map, static_cast<int32_t>(active), remapped);
        ext::upload_dense_indices(dense, remapped);
    }
    if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
        error = "PartA2 RoI sparse graph compute failed";
        ggml_free(ctx);
        return false;
    }
    output.resize(static_cast<size_t>(ggml_nelements(merged)));
    ggml_backend_tensor_get(merged, output.data(), 0,
                            output.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

}  // namespace

bool parta2::load(const std::string& path,
                  const std::string& device,
                  int threads) {
    if (!model::load(path, device, threads)) return false;
    const hparams& hp = file.hp;
    if (hp.arch != arch_kind::parta2 || !hp.parta2_anchor_proposals) {
        error = "only the available PartA2-Anchor checkpoint is executable";
        return false;
    }
    std::string load_error;
    if (!load_sparse_unet_weights(file, true, unet, load_error) ||
        !load_bev_backbone_weights(file, hp, backbone, load_error) ||
        !load_anchor_head_weights(file, hp, anchor_head, load_error) ||
        !load_point_mlp(file,
                        {{"point_head.cls_layers.0.weight",
                          "point_head.cls_layers.0.bias", "", false}},
                        point_cls, load_error) ||
        !load_point_mlp(file,
                        {{"point_head.part_reg_layers.0.weight",
                          "point_head.part_reg_layers.0.bias", "", false}},
                        point_part, load_error)) {
        error = load_error;
        return false;
    }
    const struct {
        const char* name;
        const char* conv;
        const char* bn;
    } sparse_sources[] = {
            {"part.0", "roi_head.conv_part.0.0", "roi_head.conv_part.0.1"},
            {"part.1", "roi_head.conv_part.1.0", "roi_head.conv_part.1.1"},
            {"rpn.0", "roi_head.conv_rpn.0.0", "roi_head.conv_rpn.0.1"},
            {"rpn.1", "roi_head.conv_rpn.1.0", "roi_head.conv_rpn.1.1"},
    };
    for (const auto& source : sparse_sources) {
        sparse_unet_layer layer;
        if (!load_sparse_layer(file, source.conv, source.bn, layer,
                               load_error)) {
            error = load_error;
            return false;
        }
        roi_sparse.emplace(source.name, layer);
    }
    const std::vector<point_mlp_layer_spec> cls_output = {
            {"roi_head.cls_layers.7.weight", "roi_head.cls_layers.7.bias", "",
             false}};
    const std::vector<point_mlp_layer_spec> reg_output = {
            {"roi_head.reg_layers.7.weight", "roi_head.reg_layers.7.bias", "",
             false}};
    if (!load_point_mlp(file,
                        hidden_specs("roi_head.shared_fc_layer",
                                     hp.roi_shared_fc_channels.size(), true),
                        roi_shared, load_error) ||
        !load_point_mlp(file,
                        hidden_specs("roi_head.cls_layers",
                                     hp.roi_cls_fc_channels.size(), false),
                        roi_cls_hidden, load_error) ||
        !load_point_mlp(file, cls_output, roi_cls_output, load_error) ||
        !load_point_mlp(file,
                        hidden_specs("roi_head.reg_layers",
                                     hp.roi_reg_fc_channels.size(), false),
                        roi_reg_hidden, load_error) ||
        !load_point_mlp(file, reg_output, roi_reg_output, load_error)) {
        error = load_error;
        return false;
    }
    remap_sparse_unet_weights(unet, device_ctx);
    if (device_ctx) {
        auto remap = [&](ggml_tensor*& tensor) {
            if (tensor)
                tensor = ggml_get_tensor(device_ctx, ggml_get_name(tensor));
        };
        for (auto& level : backbone.levels)
            for (auto& block : level) {
                remap(block.conv_w);
                remap(block.bn_scale);
                remap(block.bn_bias);
            }
        for (auto& block : backbone.deblocks) {
            remap(block.conv_w);
            remap(block.bn_scale);
            remap(block.bn_bias);
        }
        remap(anchor_head.conv_cls_w);
        remap(anchor_head.conv_cls_b);
        remap(anchor_head.conv_box_w);
        remap(anchor_head.conv_box_b);
        remap(anchor_head.conv_dir_cls_w);
        remap(anchor_head.conv_dir_cls_b);
        remap_point_mlp(point_cls, device_ctx);
        remap_point_mlp(point_part, device_ctx);
        for (auto& entry : roi_sparse)
            remap_sparse_layer(entry.second, device_ctx);
        remap_point_mlp(roi_shared, device_ctx);
        remap_point_mlp(roi_cls_hidden, device_ctx);
        remap_point_mlp(roi_cls_output, device_ctx);
        remap_point_mlp(roi_reg_hidden, device_ctx);
        remap_point_mlp(roi_reg_output, device_ctx);
    }
    return true;
}

bool parta2::forward(const float* points,
                     int32_t count,
                     int32_t dimension,
                     std::vector<detection>& detections,
                     const tap_sink& sink) {
    detections.clear();
    const hparams& hp = file.hp;
    sparse_unet_result unet_result;
    if (!run_sparse_unet(be, hp, unet, points, count, dimension, unet_result,
                         sink, error))
        return false;
    if (unet_result.point_features.empty() ||
        unet_result.spatial_features.empty())
        return true;
    const int64_t point_count =
            static_cast<int64_t>(unet_result.point_coords.size()) / 3;
    std::vector<float> point_logits, part_logits;
    if (!run_point_mlp(be, point_cls, unet_result.point_features, point_count,
                       point_logits, error) ||
        !run_point_mlp(be, point_part, unet_result.point_features, point_count,
                       part_logits, error))
        return false;
    std::vector<float> point_scores(point_count),
            part_offsets(part_logits.size());
    for (int64_t row = 0; row < point_count; ++row) {
        point_scores[row] = 1.0f / (1.0f + std::exp(-point_logits[row]));
        for (int axis = 0; axis < 3; ++axis)
            part_offsets[row * 3 + axis] =
                    1.0f / (1.0f + std::exp(-part_logits[row * 3 + axis]));
    }
    hparams proposal_hp = hp;
    proposal_hp.score_thresh = 0.0f;
    proposal_hp.nms_thresh = hp.roi_proposal_nms_thresh;
    proposal_hp.nms_pre_maxsize = hp.roi_proposal_pre_max;
    proposal_hp.max_boxes = hp.roi_proposal_post_max;
    std::vector<detection> proposals;
    if (!run_dense_anchor_pipeline(be, proposal_hp, backbone, anchor_head,
                                   unet_result.spatial_features,
                                   unet_result.width, unet_result.height,
                                   proposals, sink, error))
        return false;
    if (proposals.empty()) return true;
    std::vector<float> boxes(proposals.size() * 7);
    for (size_t index = 0; index < proposals.size(); ++index) {
        const auto& box = proposals[index];
        const float values[] = {box.x,  box.y,  box.z,      box.dx,
                                box.dy, box.dz, box.heading};
        std::copy_n(values, 7, boxes.data() + index * 7);
    }
    std::vector<float> part_features(static_cast<size_t>(point_count) * 4);
    for (int64_t row = 0; row < point_count; ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            float value = hp.roi_aware_disable_part
                                  ? unet_result.point_coords[row * 3 + axis]
                                  : part_offsets[row * 3 + axis];
            if (point_scores[row] < hp.roi_aware_seg_threshold) value = 0.0f;
            part_features[row * 4 + axis] = value;
        }
        part_features[row * 4 + 3] = point_scores[row];
    }
    std::vector<float> pooled_part, pooled_rpn;
    if (!ext::roi_aware_pool3d(
                boxes.data(), proposals.size(), unet_result.point_coords.data(),
                point_count, part_features.data(), 4, hp.roi_aware_pool_size,
                hp.roi_aware_pool_size, hp.roi_aware_pool_size,
                hp.roi_aware_max_points, ext::roi_pool_method::average,
                pooled_part, error) ||
        !ext::roi_aware_pool3d(boxes.data(), proposals.size(),
                               unet_result.point_coords.data(), point_count,
                               unet_result.point_features.data(), 16,
                               hp.roi_aware_pool_size, hp.roi_aware_pool_size,
                               hp.roi_aware_pool_size, hp.roi_aware_max_points,
                               ext::roi_pool_method::max, pooled_rpn, error))
        return false;
    const int grid = hp.roi_aware_pool_size;
    std::vector<int32_t> sparse_coords;
    std::vector<float> sparse_part, sparse_rpn;
    for (size_t roi = 0; roi < proposals.size(); ++roi)
        for (int x = 0; x < grid; ++x)
            for (int y = 0; y < grid; ++y)
                for (int z = 0; z < grid; ++z) {
                    const size_t cell =
                            ((roi * grid + x) * grid + y) * grid + z;
                    const float* part = pooled_part.data() + cell * 4;
                    if (part[0] == 0.0f && part[1] == 0.0f && part[2] == 0.0f &&
                        part[3] == 0.0f)
                        continue;
                    sparse_coords.insert(sparse_coords.end(),
                                         {static_cast<int32_t>(roi), x, y, z});
                    sparse_part.insert(sparse_part.end(), part, part + 4);
                    const float* rpn = pooled_rpn.data() + cell * 16;
                    sparse_rpn.insert(sparse_rpn.end(), rpn, rpn + 16);
                }
    if (sparse_coords.empty()) return true;
    std::vector<float> sparse_output;
    if (!run_roi_sparse(be, roi_sparse, sparse_coords, sparse_part, sparse_rpn,
                        16, sparse_output, error))
        return false;
    const int sparse_channels = 128;
    const int cells = grid * grid * grid;
    std::vector<float> dense(
            static_cast<size_t>(proposals.size()) * sparse_channels * cells,
            0.0f);
    const int64_t sparse_count = static_cast<int64_t>(sparse_coords.size()) / 4;
    for (int64_t row = 0; row < sparse_count; ++row) {
        const int32_t* coordinate = sparse_coords.data() + row * 4;
        const int cell =
                (coordinate[1] * grid + coordinate[2]) * grid + coordinate[3];
        for (int channel = 0; channel < sparse_channels; ++channel)
            dense[(static_cast<size_t>(coordinate[0]) * sparse_channels +
                   channel) *
                          cells +
                  cell] = sparse_output[row * sparse_channels + channel];
    }
    std::vector<float> shared, cls_hidden, cls, reg_hidden, residuals;
    if (!run_point_mlp(be, roi_shared, dense, proposals.size(), shared,
                       error) ||
        !run_point_mlp(be, roi_cls_hidden, shared, proposals.size(), cls_hidden,
                       error) ||
        !run_point_mlp(be, roi_cls_output, cls_hidden, proposals.size(), cls,
                       error) ||
        !run_point_mlp(be, roi_reg_hidden, shared, proposals.size(), reg_hidden,
                       error) ||
        !run_point_mlp(be, roi_reg_output, reg_hidden, proposals.size(),
                       residuals, error))
        return false;
    std::vector<float> refined;
    if (!ext::decode_roi_residuals(boxes.data(), residuals.data(),
                                   proposals.size(), refined, error))
        return false;
    std::vector<int32_t> candidates;
    std::vector<float> scores(proposals.size());
    for (size_t roi = 0; roi < proposals.size(); ++roi) {
        scores[roi] = 1.0f / (1.0f + std::exp(-cls[roi]));
        if (scores[roi] >= hp.score_thresh)
            candidates.push_back(static_cast<int32_t>(roi));
    }
    std::vector<float> candidate_boxes(candidates.size() * 7),
            candidate_scores(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        std::copy_n(refined.data() + candidates[index] * 7, 7,
                    candidate_boxes.data() + index * 7);
        candidate_scores[index] = scores[candidates[index]];
    }
    const std::vector<int32_t> kept = nms_3d(candidate_boxes, candidate_scores,
                                             hp.nms_thresh, hp.max_boxes);
    for (int local : kept) {
        const int roi = candidates[local];
        const float* box = refined.data() + roi * 7;
        detections.push_back({box[0], box[1], box[2], box[3], box[4], box[5],
                              box[6], scores[roi], proposals[roi].label});
    }
    return true;
}

}  // namespace openpcdet
