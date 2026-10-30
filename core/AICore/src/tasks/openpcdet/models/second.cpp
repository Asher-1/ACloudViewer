#include "second.h"

#include "../extensions/roi_ops.h"
#include "dense_anchor_pipeline.h"
#include "voxel_backbone_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace openpcdet {

namespace {

std::vector<point_mlp_layer_spec> second_shared_specs(const hparams & hp) {
    std::vector<point_mlp_layer_spec> result;
    int index = 0;
    for (size_t layer = 0; layer < hp.roi_shared_fc_channels.size(); ++layer) {
        const std::string prefix = "roi_head.shared_fc_layer." + std::to_string(index);
        result.push_back({prefix + ".weight", "",
                          "roi_head.shared_fc_layer." + std::to_string(index + 1), true});
        index += layer + 1 < hp.roi_shared_fc_channels.size() ? 4 : 3;
    }
    return result;
}

std::vector<point_mlp_layer_spec> second_iou_specs(const hparams & hp) {
    std::vector<point_mlp_layer_spec> result;
    int index = 0;
    for (size_t layer = 0; layer < hp.roi_iou_fc_channels.size(); ++layer) {
        const std::string prefix = "roi_head.iou_layers." + std::to_string(index);
        result.push_back({prefix + ".weight", "",
                          "roi_head.iou_layers." + std::to_string(index + 1), true});
        index += layer == 0 ? 4 : 3;
    }
    const std::string output = "roi_head.iou_layers." + std::to_string(index);
    result.push_back({output + ".weight", output + ".bias", "", false});
    return result;
}

} // namespace

bool second::load(const std::string & gguf_path, const std::string & device, int n_threads) {
    if (!model::load(gguf_path, device, n_threads)) return false;
    if ((file.hp.arch != arch_kind::second && file.hp.arch != arch_kind::voxel_detector) ||
        (file.hp.head_kind != dense_head_kind::anchor_single &&
         file.hp.head_kind != dense_head_kind::anchor_multi &&
         file.hp.head_kind != dense_head_kind::center)) {
        error = "voxel runtime has unsupported dense head";
        return false;
    }
    if (!file.hp.roi_kind.empty() && file.hp.roi_kind != "SECONDHead") {
        error = "SECOND runtime has unsupported RoI head";
        return false;
    }
    std::string load_error;
    if (!load_sparse_backbone_weights(file, file.hp, sparse_w, load_error) ||
        !load_bev_backbone_weights(file, file.hp, backbone_w, load_error) ||
        (file.hp.head_kind == dense_head_kind::center
            ? !load_center_head_weights(file, file.hp, center_head_w, load_error)
            : file.hp.head_kind == dense_head_kind::anchor_multi
                ? !load_anchor_multi_weights(file, file.hp, multi_head_w, load_error)
                : !load_anchor_head_weights(file, file.hp, head_w, load_error)) ||
        (file.hp.roi_kind == "SECONDHead" &&
         (!load_point_mlp(file, second_shared_specs(file.hp), roi_shared_w, load_error) ||
          !load_point_mlp(file, second_iou_specs(file.hp), roi_iou_w, load_error)))) {
        error = load_error;
        return false;
    }
    if (device_ctx) {
        auto remap = [&](ggml_tensor *& tensor) {
            if (tensor) tensor = ggml_get_tensor(device_ctx, ggml_get_name(tensor));
        };
        for (auto & layer : sparse_w.layers) {
            remap(layer.weight);
            remap(layer.bias);
        }
        for (auto & level : backbone_w.levels) {
            for (auto & block : level) {
                remap(block.conv_w); remap(block.bn_scale); remap(block.bn_bias);
            }
        }
        for (auto & block : backbone_w.deblocks) {
            remap(block.conv_w); remap(block.bn_scale); remap(block.bn_bias);
        }
        if (file.hp.head_kind == dense_head_kind::center) {
            auto remap_block = [&](conv_bn_block_weights & block) {
                remap(block.conv_w); remap(block.bn_scale); remap(block.bn_bias);
            };
            auto remap_sequence = [&](center_sequence_weights & sequence) {
                for (auto & block : sequence.middle) remap_block(block);
                remap(sequence.output_weight); remap(sequence.output_bias);
            };
            remap_block(center_head_w.shared);
            for (auto & head : center_head_w.heads) {
                remap_sequence(head.heatmap); remap_sequence(head.center);
                remap_sequence(head.center_z); remap_sequence(head.dimensions);
                remap_sequence(head.rotation); remap_sequence(head.velocity);
            }
        } else if (file.hp.head_kind == dense_head_kind::anchor_multi) {
            auto remap_block = [&](conv_bn_block_weights & block) {
                remap(block.conv_w); remap(block.bn_scale); remap(block.bn_bias);
            };
            auto remap_sequence = [&](dense_head_sequence_weights & sequence) {
                for (auto & block : sequence.middle) remap_block(block);
                remap(sequence.output.weight); remap(sequence.output.bias);
            };
            if (multi_head_w.has_shared) remap_block(multi_head_w.shared);
            for (auto & head : multi_head_w.heads) {
                remap_sequence(head.classification);
                for (auto & branch : head.regression) remap_sequence(branch);
                remap_sequence(head.box); remap_sequence(head.direction);
            }
        } else {
            remap(head_w.conv_cls_w); remap(head_w.conv_cls_b);
            remap(head_w.conv_box_w); remap(head_w.conv_box_b);
            remap(head_w.conv_dir_cls_w); remap(head_w.conv_dir_cls_b);
        }
        if (file.hp.roi_kind == "SECONDHead") {
            remap_point_mlp(roi_shared_w, device_ctx);
            remap_point_mlp(roi_iou_w, device_ctx);
        }
    }
    return true;
}

bool second::forward(const float * points, int32_t n_points, int32_t point_dim,
                     std::vector<detection> & detections, const tap_sink & sink) {
    const hparams & hp = file.hp;
    detections.clear();
    // Debug-only wall-clock print (OPCPD_PROFILE): diagnostics only.
    const bool profile = std::getenv("OPCPD_PROFILE") != nullptr;
    const auto t0 = std::chrono::high_resolution_clock::now();
    // CUDA-ext CenterHead chain: once the dense pipeline cache exists, the
    // sparse backbone HeightCompression writes the BEV straight into the
    // persistent device input tensor and the dense graph consumes it without
    // any host round trip. First frame (cache not built yet) stays on the
    // host path; the graph is then built with direct WHC input.
    float * center_device_input = nullptr;
    if (hp.head_kind == dense_head_kind::center && !sink && center_cache.spatial &&
        !be.is_cpu() && (be.device.find("cuda") != std::string::npos ||
                         be.device.find("CUDA") != std::string::npos)) {
        center_device_input = static_cast<float *>(center_cache.spatial->data);
    }
    voxel_backbone_result backbone;
    if (!run_voxel_backbone(be, hp, sparse_w, points, n_points, point_dim,
                            false, backbone, sink, error, center_device_input,
                            sink ? nullptr : &sparse_cache,
                            center_device_input != nullptr)) return false;
    if (backbone.spatial_features.empty() && !backbone.spatial_device_ready) return true;

    bool ok = false;
    if (hp.head_kind == dense_head_kind::center) {
        ok = run_dense_center_pipeline(be, hp, backbone_w, center_head_w,
                                       backbone.spatial_features, backbone.width,
                                       backbone.height, detections, sink, error,
                                       &center_cache, backbone.spatial_device_ready,
                                       be.is_cuda() && !sink);
    } else if (hp.head_kind == dense_head_kind::anchor_multi) {
        ok = run_dense_anchor_multi_pipeline(be, hp, backbone_w, multi_head_w,
                                             backbone.spatial_features, backbone.width,
                                             backbone.height, detections, sink, error);
    } else {
        if (hp.roi_kind != "SECONDHead") {
            ok = run_dense_anchor_pipeline(be, hp, backbone_w, head_w,
                                           backbone.spatial_features, backbone.width,
                                           backbone.height, detections, sink, error);
        } else {
            dense_anchor_result raw;
            if (!run_dense_anchor_raw(be, hp, backbone_w, head_w,
                                      backbone.spatial_features, backbone.width,
                                      backbone.height, true, raw, sink, error)) return false;
            std::vector<detection> proposals;
            if (!postprocess_dense_anchors(
                    raw, -std::numeric_limits<float>::infinity(),
                    hp.roi_proposal_pre_max, hp.roi_proposal_post_max,
                    hp.roi_proposal_nms_thresh, proposals, error)) return false;
            if (proposals.empty()) return true;
            std::vector<float> proposal_boxes(proposals.size() * 7);
            for (size_t index = 0; index < proposals.size(); ++index) {
                const auto & value = proposals[index];
                const float box[] = {value.x, value.y, value.z, value.dx,
                                     value.dy, value.dz, value.heading};
                std::copy_n(box, 7, proposal_boxes.data() + index * 7);
            }
            std::vector<float> pooled;
            if (!ext::rotated_bev_roi_align(
                    raw.spatial_features_2d.data(), raw.feature_height,
                    raw.feature_width, raw.feature_channels,
                    proposal_boxes.data(), proposals.size(), hp.roi_grid_size,
                    hp.point_cloud_range[0], hp.point_cloud_range[1],
                    hp.voxel_size[0], hp.voxel_size[1],
                    hp.roi_grid_downsample_ratio, pooled, error)) return false;
            std::vector<float> shared, iou_logits;
            if (!run_point_mlp(be, roi_shared_w, pooled, proposals.size(), shared, error) ||
                !run_point_mlp(be, roi_iou_w, shared, proposals.size(), iou_logits, error)) return false;
            if (sink) {
                sink("roi_head.grid_pool", pooled.data(), proposals.size(),
                     static_cast<int64_t>(pooled.size() / proposals.size()));
                sink("roi_head.shared_fc", shared.data(), proposals.size(),
                     static_cast<int64_t>(shared.size() / proposals.size()));
                sink("roi_head.iou", iou_logits.data(), proposals.size(), 1);
            }
            dense_anchor_result refined;
            refined.boxes = std::move(proposal_boxes);
            refined.scores.resize(proposals.size());
            refined.labels.resize(proposals.size());
            for (size_t index = 0; index < proposals.size(); ++index) {
                refined.scores[index] = 1.0f / (1.0f + std::exp(-iou_logits[index]));
                refined.labels[index] = proposals[index].label;
            }
            ok = postprocess_dense_anchors(refined, hp.score_thresh,
                                           hp.nms_pre_maxsize, hp.max_boxes,
                                           hp.nms_thresh, detections, error);
        }
    }
    if (profile) {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::fprintf(stderr, "[second] dense pipeline + decode  %8.2f ms\n", ms);
    }
    return ok;
}

} // namespace openpcdet
