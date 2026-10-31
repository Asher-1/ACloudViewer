// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pvrcnn.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../extensions/pointnet2_ops.h"
#include "../extensions/roi_ops.h"
#include "../ops/nms_3d.h"
#include "dense_anchor_pipeline.h"
#include "voxel_backbone_runner.h"

namespace openpcdet {

namespace {

std::vector<point_mlp_layer_spec> sequential_specs(const std::string& prefix,
                                                   int depth,
                                                   int stride = 3) {
    std::vector<point_mlp_layer_spec> result;
    for (int layer = 0; layer < depth; ++layer) {
        const int index = layer * stride;
        result.push_back({prefix + "." + std::to_string(index) + ".weight", "",
                          prefix + "." + std::to_string(index + 1), true});
    }
    return result;
}

std::vector<point_mlp_layer_spec> two_hidden_specs(const std::string& prefix) {
    return {
            {prefix + ".0.weight", "", prefix + ".1", true},
            {prefix + ".4.weight", "", prefix + ".5", true},
    };
}

std::vector<point_mlp_layer_spec> output_specs(const std::string& prefix) {
    auto result = two_hidden_specs(prefix);
    result.push_back({prefix + ".7.weight", prefix + ".7.bias", "", false});
    return result;
}

bool max_pool_groups(const std::vector<float>& input,
                     int64_t groups,
                     int samples,
                     std::vector<float>& output,
                     std::string& error) {
    if (groups < 0 || samples <= 0 ||
        input.size() % (static_cast<size_t>(groups) * samples) != 0) {
        error = "bad PVRCNN max-pool shape";
        return false;
    }
    const int channels = static_cast<int>(
            input.size() / (static_cast<size_t>(groups) * samples));
    output.assign(static_cast<size_t>(groups) * channels, 0.0f);
    for (int64_t group = 0; group < groups; ++group)
        for (int channel = 0; channel < channels; ++channel) {
            float maximum = input[(group * samples) * channels + channel];
            for (int sample = 1; sample < samples; ++sample)
                maximum = std::max(
                        maximum,
                        input[(group * samples + sample) * channels + channel]);
            output[group * channels + channel] = maximum;
        }
    return true;
}

bool aggregate_centers(engine_backend& backend,
                       const float* xyz,
                       int64_t point_count,
                       const float* features,
                       int channels,
                       const float* centers,
                       int64_t center_count,
                       const std::vector<float>& radii,
                       const std::vector<int32_t>& nsamples,
                       const std::vector<point_mlp_weights>& branches,
                       std::vector<float>& output,
                       std::string& error) {
    if (!xyz || !centers || point_count <= 0 || center_count <= 0 ||
        channels < 0 || (channels && !features) ||
        radii.size() != nsamples.size() || radii.size() != branches.size()) {
        error = "bad PVRCNN local aggregation input";
        return false;
    }
    output.clear();
    ext::ball_query_index query_index;
    const float maximum_radius = *std::max_element(radii.begin(), radii.end());
    if (!query_index.build(xyz, point_count, maximum_radius, error))
        return false;
    for (size_t branch = 0; branch < branches.size(); ++branch) {
        std::vector<int32_t> indices;
        if (!query_index.query(centers, center_count, radii[branch],
                               nsamples[branch], indices, error))
            return false;
        const int input_channels = channels + 3;
        std::vector<float> grouped(static_cast<size_t>(center_count) *
                                           nsamples[branch] * input_channels,
                                   0.0f);
        for (int64_t center = 0; center < center_count; ++center)
            for (int sample = 0; sample < nsamples[branch]; ++sample) {
                const int32_t source =
                        indices[center * nsamples[branch] + sample];
                if (source < 0) continue;
                float* destination =
                        grouped.data() +
                        (center * nsamples[branch] + sample) * input_channels;
                for (int axis = 0; axis < 3; ++axis)
                    destination[axis] =
                            xyz[static_cast<int64_t>(source) * 3 + axis] -
                            centers[center * 3 + axis];
                if (channels)
                    std::copy_n(
                            features + static_cast<int64_t>(source) * channels,
                            channels, destination + 3);
            }
        std::vector<float> transformed, pooled;
        if (!run_point_mlp(backend, branches[branch], grouped,
                           center_count * nsamples[branch], transformed,
                           error) ||
            !max_pool_groups(transformed, center_count, nsamples[branch],
                             pooled, error))
            return false;
        const int branch_channels =
                static_cast<int>(pooled.size() / center_count);
        const int old_channels =
                output.empty() ? 0
                               : static_cast<int>(output.size() / center_count);
        std::vector<float> concatenated(static_cast<size_t>(center_count) *
                                        (old_channels + branch_channels));
        for (int64_t row = 0; row < center_count; ++row) {
            if (old_channels)
                std::copy_n(output.data() + row * old_channels, old_channels,
                            concatenated.data() +
                                    row * (old_channels + branch_channels));
            std::copy_n(pooled.data() + row * branch_channels, branch_channels,
                        concatenated.data() +
                                row * (old_channels + branch_channels) +
                                old_channels);
        }
        output = std::move(concatenated);
    }
    return true;
}

int layer_index(const sparse_backbone_weights& weights,
                const std::string& name) {
    for (size_t index = 0; index < weights.specs.size(); ++index)
        if (weights.specs[index].name == name) return static_cast<int>(index);
    return -1;
}

void sparse_centers(const hparams& hp,
                    const ext::sparse_indice_map& map,
                    int downsample,
                    std::vector<float>& xyz) {
    xyz.resize(static_cast<size_t>(map.output_active_count()) * 3);
    for (int64_t row = 0; row < map.output_active_count(); ++row) {
        const int32_t* coordinate = map.coordinates.data() + row * 4;
        xyz[row * 3] = (coordinate[3] + 0.5f) * hp.voxel_size[0] * downsample +
                       hp.point_cloud_range[0];
        xyz[row * 3 + 1] =
                (coordinate[2] + 0.5f) * hp.voxel_size[1] * downsample +
                hp.point_cloud_range[1];
        xyz[row * 3 + 2] =
                (coordinate[1] + 0.5f) * hp.voxel_size[2] * downsample +
                hp.point_cloud_range[2];
    }
}

std::vector<float> bilinear_bev(const hparams& hp,
                                const std::vector<float>& bev,
                                int width,
                                int height,
                                int channels,
                                const std::vector<float>& keypoints) {
    const int64_t count = static_cast<int64_t>(keypoints.size()) / 3;
    std::vector<float> result(static_cast<size_t>(count) * channels, 0.0f);
    const float stride =
            hp.feature_map_strides.empty() ? 8.0f : hp.feature_map_strides[0];
    for (int64_t row = 0; row < count; ++row) {
        const float x = (keypoints[row * 3] - hp.point_cloud_range[0]) /
                        hp.voxel_size[0] / stride;
        const float y = (keypoints[row * 3 + 1] - hp.point_cloud_range[1]) /
                        hp.voxel_size[1] / stride;
        int x0 = static_cast<int>(std::floor(x)), x1 = x0 + 1;
        int y0 = static_cast<int>(std::floor(y)), y1 = y0 + 1;
        x0 = std::clamp(x0, 0, width - 1);
        x1 = std::clamp(x1, 0, width - 1);
        y0 = std::clamp(y0, 0, height - 1);
        y1 = std::clamp(y1, 0, height - 1);
        const float wa = (x1 - x) * (y1 - y), wb = (x1 - x) * (y - y0);
        const float wc = (x - x0) * (y1 - y), wd = (x - x0) * (y - y0);
        for (int channel = 0; channel < channels; ++channel) {
            auto at = [&](int px, int py) {
                return bev[(static_cast<size_t>(py) * width + px) * channels +
                           channel];
            };
            result[row * channels + channel] =
                    at(x0, y0) * wa + at(x0, y1) * wb + at(x1, y0) * wc +
                    at(x1, y1) * wd;
        }
    }
    return result;
}

std::vector<float> roi_grid_points(const std::vector<detection>& proposals,
                                   int grid) {
    const int grids = grid * grid * grid;
    std::vector<float> result(static_cast<size_t>(proposals.size()) * grids *
                              3);
    for (size_t roi = 0; roi < proposals.size(); ++roi) {
        const auto& box = proposals[roi];
        const float cosine = std::cos(box.heading),
                    sine = std::sin(box.heading);
        int index = 0;
        for (int x = 0; x < grid; ++x)
            for (int y = 0; y < grid; ++y)
                for (int z = 0; z < grid; ++z, ++index) {
                    const float local_x =
                            (x + 0.5f) / grid * box.dx - box.dx * 0.5f;
                    const float local_y =
                            (y + 0.5f) / grid * box.dy - box.dy * 0.5f;
                    float* target = result.data() + (roi * grids + index) * 3;
                    target[0] = local_x * cosine - local_y * sine + box.x;
                    target[1] = local_x * sine + local_y * cosine + box.y;
                    target[2] =
                            (z + 0.5f) / grid * box.dz - box.dz * 0.5f + box.z;
                }
    }
    return result;
}

}  // namespace

bool pvrcnn::load(const std::string& path,
                  const std::string& device,
                  int threads) {
    if (!model::load(path, device, threads)) return false;
    const hparams& hp = file.hp;
    if (hp.arch != arch_kind::pvrcnn ||
        hp.head_kind != dense_head_kind::anchor_single ||
        hp.pfe_sa_sources.size() != 5 || hp.roi_grid_radii.size() != 2) {
        error = "unsupported PVRCNN topology";
        return false;
    }
    std::string load_error;
    if (!load_sparse_backbone_weights(file, hp, sparse_w, load_error) ||
        !load_bev_backbone_weights(file, hp, backbone_w, load_error) ||
        !load_anchor_head_weights(file, hp, head_w, load_error)) {
        error = load_error;
        return false;
    }
    pfe_sa.resize(hp.pfe_sa_sources.size());
    size_t depth_offset = 0;
    for (size_t source = 0; source < hp.pfe_sa_sources.size(); ++source) {
        const int branches = hp.pfe_sa_radius_counts[source];
        pfe_sa[source].resize(branches);
        for (int branch = 0; branch < branches; ++branch) {
            const std::string prefix =
                    source == 0
                            ? "pfe.SA_rawpoints.mlps." + std::to_string(branch)
                            : "pfe.SA_layers." + std::to_string(source - 1) +
                                      ".mlps." + std::to_string(branch);
            if (depth_offset >= hp.pfe_sa_mlp_depths.size() ||
                !load_point_mlp(
                        file,
                        sequential_specs(prefix,
                                         hp.pfe_sa_mlp_depths[depth_offset++]),
                        pfe_sa[source][branch], load_error)) {
                error = load_error.empty()
                                ? "PVRCNN PFE depth metadata overflow"
                                : load_error;
                return false;
            }
        }
    }
    const std::vector<point_mlp_layer_spec> fusion = {
            {"pfe.vsa_point_feature_fusion.0.weight", "",
             "pfe.vsa_point_feature_fusion.1", true},
    };
    const std::vector<point_mlp_layer_spec> point = {
            {"point_head.cls_layers.0.weight", "", "point_head.cls_layers.1",
             true},
            {"point_head.cls_layers.3.weight", "", "point_head.cls_layers.4",
             true},
            {"point_head.cls_layers.6.weight", "point_head.cls_layers.6.bias",
             "", false},
    };
    if (!load_point_mlp(file, fusion, pfe_fusion, load_error) ||
        !load_point_mlp(file, point, point_cls, load_error)) {
        error = load_error;
        return false;
    }
    roi_pool.resize(hp.roi_grid_radii.size());
    for (size_t branch = 0; branch < roi_pool.size(); ++branch) {
        const std::string prefix =
                "roi_head.roi_grid_pool_layer.mlps." + std::to_string(branch);
        if (!load_point_mlp(
                    file,
                    sequential_specs(prefix, hp.roi_grid_mlp_depths[branch]),
                    roi_pool[branch], load_error)) {
            error = load_error;
            return false;
        }
    }
    if (!load_point_mlp(file, two_hidden_specs("roi_head.shared_fc_layer"),
                        roi_shared, load_error) ||
        !load_point_mlp(file, output_specs("roi_head.cls_layers"), roi_cls,
                        load_error) ||
        !load_point_mlp(file, output_specs("roi_head.reg_layers"), roi_reg,
                        load_error)) {
        error = load_error;
        return false;
    }
    if (device_ctx) {
        auto remap = [&](ggml_tensor*& tensor) {
            if (tensor)
                tensor = ggml_get_tensor(device_ctx, ggml_get_name(tensor));
        };
        for (auto& layer : sparse_w.layers) {
            remap(layer.weight);
            remap(layer.bias);
        }
        for (auto& level : backbone_w.levels)
            for (auto& block : level) {
                remap(block.conv_w);
                remap(block.bn_scale);
                remap(block.bn_bias);
            }
        for (auto& block : backbone_w.deblocks) {
            remap(block.conv_w);
            remap(block.bn_scale);
            remap(block.bn_bias);
        }
        remap(head_w.conv_cls_w);
        remap(head_w.conv_cls_b);
        remap(head_w.conv_box_w);
        remap(head_w.conv_box_b);
        remap(head_w.conv_dir_cls_w);
        remap(head_w.conv_dir_cls_b);
        for (auto& source : pfe_sa)
            for (auto& branch : source) remap_point_mlp(branch, device_ctx);
        remap_point_mlp(pfe_fusion, device_ctx);
        remap_point_mlp(point_cls, device_ctx);
        for (auto& branch : roi_pool) remap_point_mlp(branch, device_ctx);
        remap_point_mlp(roi_shared, device_ctx);
        remap_point_mlp(roi_cls, device_ctx);
        remap_point_mlp(roi_reg, device_ctx);
    }
    return true;
}

bool pvrcnn::forward(const float* points,
                     int32_t count,
                     int32_t dimension,
                     std::vector<detection>& detections,
                     const tap_sink& sink) {
    detections.clear();
    const hparams& hp = file.hp;
    voxel_backbone_result voxel;
    if (!run_voxel_backbone(be, hp, sparse_w, points, count, dimension, true,
                            voxel, sink, error))
        return false;
    if (voxel.spatial_features.empty()) return true;

    hparams proposal_hp = hp;
    proposal_hp.score_thresh = 0.0f;
    proposal_hp.nms_thresh = hp.roi_proposal_nms_thresh;
    proposal_hp.nms_pre_maxsize = hp.roi_proposal_pre_max;
    proposal_hp.max_boxes = hp.roi_proposal_post_max;
    tap_sink dense_sink = [&](const std::string& name, const float* data,
                              int64_t rows, int64_t columns) {
        if (sink) sink(name, data, rows, columns);
    };
    std::vector<detection> proposals;
    if (!run_dense_anchor_pipeline(be, proposal_hp, backbone_w, head_w,
                                   voxel.spatial_features, voxel.width,
                                   voxel.height, proposals, dense_sink, error))
        return false;
    if (proposals.empty()) return true;

    std::vector<float> raw_xyz(static_cast<size_t>(count) * 3);
    for (int row = 0; row < count; ++row)
        std::copy_n(points + static_cast<int64_t>(row) * dimension, 3,
                    raw_xyz.data() + row * 3);
    std::vector<int32_t> key_indices;
    // AICore port note: the upstream task-private CUDA FPS workspace is
    // dropped; the host FPS runs identically on every backend.
    if (!ext::farthest_point_sample(raw_xyz.data(), count, hp.pfe_num_keypoints,
                                    key_indices, error))
        return false;
    std::vector<float> keypoints(static_cast<size_t>(hp.pfe_num_keypoints) * 3);
    for (int row = 0; row < hp.pfe_num_keypoints; ++row)
        std::copy_n(raw_xyz.data() + static_cast<int64_t>(key_indices[row]) * 3,
                    3, keypoints.data() + row * 3);

    std::vector<float> before_fusion =
            bilinear_bev(hp, voxel.spatial_features, voxel.width, voxel.height,
                         hp.num_bev_features, keypoints);
    size_t radius_offset = 0;
    const std::vector<std::string> stage_names = {"", "conv1.0", "conv2.2",
                                                  "conv3.2", "conv4.2"};
    const std::vector<int> stage_strides = {1, 1, 2, 4, 8};
    for (size_t source = 0; source < pfe_sa.size(); ++source) {
        const int branches = hp.pfe_sa_radius_counts[source];
        std::vector<float> radii(
                hp.pfe_sa_radii.begin() + radius_offset,
                hp.pfe_sa_radii.begin() + radius_offset + branches);
        std::vector<int32_t> samples(
                hp.pfe_sa_nsamples.begin() + radius_offset,
                hp.pfe_sa_nsamples.begin() + radius_offset + branches);
        radius_offset += branches;
        const float* source_xyz = nullptr;
        const float* source_features = nullptr;
        int64_t source_count = 0;
        int source_channels = 0;
        std::vector<float> xyz_storage, feature_storage;
        if (source == 0) {
            source_xyz = raw_xyz.data();
            source_count = count;
            source_channels = dimension - 3;
            feature_storage.resize(static_cast<size_t>(count) *
                                   source_channels);
            for (int row = 0; row < count; ++row)
                std::copy_n(points + static_cast<int64_t>(row) * dimension + 3,
                            source_channels,
                            feature_storage.data() + row * source_channels);
            source_features = feature_storage.data();
        } else {
            const int index = layer_index(sparse_w, stage_names[source]);
            if (index < 0) {
                error = "PVRCNN sparse feature stage missing";
                return false;
            }
            sparse_centers(hp, voxel.maps[index], stage_strides[source],
                           xyz_storage);
            source_xyz = xyz_storage.data();
            source_count = voxel.maps[index].output_active_count();
            source_channels = sparse_w.specs[index].out_channels;
            source_features = voxel.layer_features[index].data();
        }
        std::vector<float> pooled;
        if (!aggregate_centers(be, source_xyz, source_count, source_features,
                               source_channels, keypoints.data(),
                               hp.pfe_num_keypoints, radii, samples,
                               pfe_sa[source], pooled, error))
            return false;
        const int old_channels =
                static_cast<int>(before_fusion.size() / hp.pfe_num_keypoints);
        const int added_channels =
                static_cast<int>(pooled.size() / hp.pfe_num_keypoints);
        std::vector<float> concatenated(
                static_cast<size_t>(hp.pfe_num_keypoints) *
                (old_channels + added_channels));
        for (int row = 0; row < hp.pfe_num_keypoints; ++row) {
            std::copy_n(before_fusion.data() + row * old_channels, old_channels,
                        concatenated.data() +
                                row * (old_channels + added_channels));
            std::copy_n(pooled.data() + row * added_channels, added_channels,
                        concatenated.data() +
                                row * (old_channels + added_channels) +
                                old_channels);
        }
        before_fusion = std::move(concatenated);
    }
    std::vector<float> fused, point_logits;
    if (!run_point_mlp(be, pfe_fusion, before_fusion, hp.pfe_num_keypoints,
                       fused, error) ||
        !run_point_mlp(be, point_cls, before_fusion, hp.pfe_num_keypoints,
                       point_logits, error))
        return false;
    for (int row = 0; row < hp.pfe_num_keypoints; ++row) {
        const float scale = 1.0f / (1.0f + std::exp(-point_logits[row]));
        for (int channel = 0; channel < hp.pfe_num_output_features; ++channel)
            fused[row * hp.pfe_num_output_features + channel] *= scale;
    }
    if (sink) {
        sink("pfe.before_fusion", before_fusion.data(), hp.pfe_num_keypoints,
             before_fusion.size() / hp.pfe_num_keypoints);
        sink("pfe.fused", fused.data(), hp.pfe_num_keypoints,
             hp.pfe_num_output_features);
        sink("point_head.cls", point_logits.data(), hp.pfe_num_keypoints, 1);
    }

    const int grids_per_roi =
            hp.roi_grid_size * hp.roi_grid_size * hp.roi_grid_size;
    std::vector<float> grid_xyz = roi_grid_points(proposals, hp.roi_grid_size);
    std::vector<float> pooled;
    if (!aggregate_centers(
                be, keypoints.data(), hp.pfe_num_keypoints, fused.data(),
                hp.pfe_num_output_features, grid_xyz.data(),
                static_cast<int64_t>(proposals.size()) * grids_per_roi,
                hp.roi_grid_radii, hp.roi_grid_nsamples, roi_pool, pooled,
                error))
        return false;
    const int pooled_channels = static_cast<int>(
            pooled.size() / (proposals.size() * grids_per_roi));
    std::vector<float> flattened(proposals.size() * grids_per_roi *
                                 pooled_channels);
    for (size_t roi = 0; roi < proposals.size(); ++roi)
        for (int channel = 0; channel < pooled_channels; ++channel)
            for (int grid = 0; grid < grids_per_roi; ++grid)
                flattened[(roi * pooled_channels + channel) * grids_per_roi +
                          grid] =
                        pooled[(roi * grids_per_roi + grid) * pooled_channels +
                               channel];
    std::vector<float> shared, refined_cls, residuals;
    if (!run_point_mlp(be, roi_shared, flattened, proposals.size(), shared,
                       error) ||
        !run_point_mlp(be, roi_cls, shared, proposals.size(), refined_cls,
                       error) ||
        !run_point_mlp(be, roi_reg, shared, proposals.size(), residuals, error))
        return false;

    std::vector<float> rois(proposals.size() * 7);
    for (size_t roi = 0; roi < proposals.size(); ++roi) {
        const auto& box = proposals[roi];
        const float values[7] = {box.x,  box.y,  box.z,      box.dx,
                                 box.dy, box.dz, box.heading};
        std::copy_n(values, 7, rois.data() + roi * 7);
    }
    std::vector<float> refined;
    if (!ext::decode_roi_residuals(rois.data(), residuals.data(),
                                   proposals.size(), refined, error))
        return false;
    std::vector<int32_t> candidates;
    std::vector<float> scores(proposals.size());
    for (size_t roi = 0; roi < proposals.size(); ++roi) {
        scores[roi] = 1.0f / (1.0f + std::exp(-refined_cls[roi]));
        if (scores[roi] >= hp.score_thresh)
            candidates.push_back(static_cast<int32_t>(roi));
    }
    if (static_cast<int>(candidates.size()) > hp.nms_pre_maxsize) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + hp.nms_pre_maxsize,
                          candidates.end(),
                          [&](int a, int b) { return scores[a] > scores[b]; });
        candidates.resize(hp.nms_pre_maxsize);
    }
    std::vector<float> boxes(candidates.size() * 7),
            candidate_scores(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        std::copy_n(refined.data() + candidates[index] * 7, 7,
                    boxes.data() + index * 7);
        candidate_scores[index] = scores[candidates[index]];
    }
    const std::vector<int32_t> kept =
            nms_3d(boxes, candidate_scores, hp.nms_thresh, hp.max_boxes);
    for (int local : kept) {
        const int roi = candidates[local];
        const float* box = refined.data() + roi * 7;
        detections.push_back({box[0], box[1], box[2], box[3], box[4], box[5],
                              box[6], scores[roi], proposals[roi].label});
    }
    return true;
}

}  // namespace openpcdet
