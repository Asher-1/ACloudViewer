// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pointrcnn.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../extensions/pointnet2_ops.h"
#include "../extensions/roi_ops.h"
#include "../ops/nms_3d.h"

namespace openpcdet {

namespace {

std::vector<point_mlp_layer_spec> sequential_specs(const std::string& prefix,
                                                   int depth,
                                                   int stride = 3) {
    std::vector<point_mlp_layer_spec> specs;
    for (int layer = 0; layer < depth; ++layer) {
        const int index = layer * stride;
        specs.push_back({prefix + "." + std::to_string(index) + ".weight", "",
                         prefix + "." + std::to_string(index + 1), true});
    }
    return specs;
}

std::vector<point_mlp_layer_spec> point_head_specs(const std::string& prefix) {
    return {
            {prefix + ".0.weight", "", prefix + ".1", true},
            {prefix + ".3.weight", "", prefix + ".4", true},
            {prefix + ".6.weight", prefix + ".6.bias", "", false},
    };
}

std::vector<point_mlp_layer_spec> roi_fc_specs(const std::string& prefix) {
    return {
            {prefix + ".0.weight", "", prefix + ".1", true},
            {prefix + ".4.weight", "", prefix + ".5", true},
            {prefix + ".7.weight", prefix + ".7.bias", "", false},
    };
}

bool max_pool_groups(const std::vector<float>& values,
                     int64_t groups,
                     int samples,
                     std::vector<float>& output,
                     std::string& error) {
    if (groups < 0 || samples <= 0 ||
        values.size() % (static_cast<size_t>(groups) * samples) != 0) {
        error = "bad PointNet max-pool shape";
        return false;
    }
    const int channels = static_cast<int>(
            values.size() / (static_cast<size_t>(groups) * samples));
    output.assign(static_cast<size_t>(groups) * channels, 0.0f);
    for (int64_t group = 0; group < groups; ++group) {
        for (int channel = 0; channel < channels; ++channel) {
            float maximum = values[(group * samples) * channels + channel];
            for (int sample = 1; sample < samples; ++sample)
                maximum = std::max(
                        maximum, values[(group * samples + sample) * channels +
                                        channel]);
            output[group * channels + channel] = maximum;
        }
    }
    return true;
}

bool set_abstraction(engine_backend& backend,
                     const float* xyz,
                     int64_t point_count,
                     const std::vector<float>& features,
                     const std::vector<point_mlp_weights>& branches,
                     const std::vector<float>& radii,
                     const std::vector<int32_t>& nsamples,
                     int sample_count,
                     std::vector<float>& sampled_xyz,
                     std::vector<float>& sampled_features,
                     std::string& error) {
    const int channels =
            point_count ? static_cast<int>(features.size() / point_count) : 0;
    std::vector<int32_t> centers;
    // AICore port note: the upstream task-private CUDA FPS extension is
    // dropped; the host FPS runs identically on every backend.
    if (!ext::farthest_point_sample(xyz, point_count, sample_count, centers,
                                    error))
        return false;
    ext::ball_query_index query_index;
    const float maximum_radius = *std::max_element(radii.begin(), radii.end());
    if (!query_index.build(xyz, point_count, maximum_radius, error))
        return false;
    sampled_xyz.clear();
    sampled_features.clear();
    for (size_t branch = 0; branch < branches.size(); ++branch) {
        std::vector<float> center_xyz, grouped, transformed, pooled;
        std::vector<int32_t> indices;
        if (!ext::sample_and_group(xyz, point_count, features.data(), channels,
                                   centers.data(), sample_count, radii[branch],
                                   nsamples[branch], true, center_xyz, indices,
                                   grouped, error, &query_index) ||
            !run_point_mlp(
                    backend, branches[branch], grouped,
                    static_cast<int64_t>(sample_count) * nsamples[branch],
                    transformed, error) ||
            !max_pool_groups(transformed, sample_count, nsamples[branch],
                             pooled, error))
            return false;
        if (branch == 0) sampled_xyz = std::move(center_xyz);
        const int output_channels =
                static_cast<int>(pooled.size() / sample_count);
        const int old_channels =
                branch == 0 ? 0
                            : static_cast<int>(sampled_features.size() /
                                               sample_count);
        std::vector<float> concatenated(static_cast<size_t>(sample_count) *
                                        (old_channels + output_channels));
        for (int row = 0; row < sample_count; ++row) {
            if (old_channels)
                std::copy_n(sampled_features.data() + row * old_channels,
                            old_channels,
                            concatenated.data() +
                                    row * (old_channels + output_channels));
            std::copy_n(pooled.data() + row * output_channels, output_channels,
                        concatenated.data() +
                                row * (old_channels + output_channels) +
                                old_channels);
        }
        sampled_features = std::move(concatenated);
    }
    return true;
}

std::vector<float> decode_point_boxes(const hparams& hp,
                                      const float* xyz,
                                      const std::vector<float>& cls,
                                      const std::vector<float>& encoded) {
    const int64_t count = static_cast<int64_t>(cls.size()) / hp.num_class;
    std::vector<float> boxes(static_cast<size_t>(count) * 7);
    for (int64_t row = 0; row < count; ++row) {
        int label = 0;
        for (int cls_index = 1; cls_index < hp.num_class; ++cls_index)
            if (cls[row * hp.num_class + cls_index] >
                cls[row * hp.num_class + label])
                label = cls_index;
        const float* mean = hp.point_mean_sizes.data() + label * 3;
        const float diagonal = std::sqrt(mean[0] * mean[0] + mean[1] * mean[1]);
        const float* code = encoded.data() + row * 8;
        float* box = boxes.data() + row * 7;
        box[0] = code[0] * diagonal + xyz[row * 3];
        box[1] = code[1] * diagonal + xyz[row * 3 + 1];
        box[2] = code[2] * mean[2] + xyz[row * 3 + 2];
        box[3] = std::exp(code[3]) * mean[0];
        box[4] = std::exp(code[4]) * mean[1];
        box[5] = std::exp(code[5]) * mean[2];
        box[6] = std::atan2(code[7], code[6]);
    }
    return boxes;
}

}  // namespace

bool pointrcnn::load(const std::string& path,
                     const std::string& device,
                     int threads) {
    if (!model::load(path, device, threads)) return false;
    if (file.hp.arch != arch_kind::pointrcnn ||
        file.hp.pointnet_sa_npoints.size() != 4 ||
        file.hp.pointnet_sa_radius_counts !=
                std::vector<int32_t>({2, 2, 2, 2}) ||
        file.hp.pointnet_fp_depths.size() != 4 ||
        file.hp.roi_sa_npoints.size() != 3) {
        error = "unsupported PointRCNN topology";
        return false;
    }
    std::string load_error;
    backbone_sa.resize(4);
    size_t depth_index = 0;
    for (int stage = 0; stage < 4; ++stage) {
        backbone_sa[stage].resize(2);
        for (int branch = 0; branch < 2; ++branch) {
            const int depth = file.hp.pointnet_sa_mlp_depths[depth_index++];
            const std::string prefix = "backbone_3d.SA_modules." +
                                       std::to_string(stage) + ".mlps." +
                                       std::to_string(branch);
            if (!load_point_mlp(file, sequential_specs(prefix, depth),
                                backbone_sa[stage][branch], load_error)) {
                error = load_error;
                return false;
            }
        }
    }
    backbone_fp.resize(4);
    for (int module = 0; module < 4; ++module) {
        const std::string prefix =
                "backbone_3d.FP_modules." + std::to_string(module) + ".mlp";
        if (!load_point_mlp(file,
                            sequential_specs(
                                    prefix, file.hp.pointnet_fp_depths[module]),
                            backbone_fp[module], load_error)) {
            error = load_error;
            return false;
        }
    }
    if (!load_point_mlp(file, point_head_specs("point_head.cls_layers"),
                        point_cls, load_error) ||
        !load_point_mlp(file, point_head_specs("point_head.box_layers"),
                        point_box, load_error)) {
        error = load_error;
        return false;
    }
    const std::vector<point_mlp_layer_spec> xyz_specs = {
            {"roi_head.xyz_up_layer.0.weight", "roi_head.xyz_up_layer.0.bias",
             "", true},
            {"roi_head.xyz_up_layer.2.weight", "roi_head.xyz_up_layer.2.bias",
             "", true},
    };
    const std::vector<point_mlp_layer_spec> merge_specs = {
            {"roi_head.merge_down_layer.0.weight",
             "roi_head.merge_down_layer.0.bias", "", true},
    };
    if (!load_point_mlp(file, xyz_specs, roi_xyz_up, load_error) ||
        !load_point_mlp(file, merge_specs, roi_merge, load_error)) {
        error = load_error;
        return false;
    }
    roi_sa.resize(3);
    for (int stage = 0; stage < 3; ++stage) {
        const std::string prefix =
                "roi_head.SA_modules." + std::to_string(stage) + ".mlps.0";
        if (!load_point_mlp(
                    file,
                    sequential_specs(prefix, file.hp.roi_sa_mlp_depths[stage]),
                    roi_sa[stage], load_error)) {
            error = load_error;
            return false;
        }
    }
    if (!load_point_mlp(file, roi_fc_specs("roi_head.cls_layers"), roi_cls,
                        load_error) ||
        !load_point_mlp(file, roi_fc_specs("roi_head.reg_layers"), roi_reg,
                        load_error)) {
        error = load_error;
        return false;
    }
    if (device_ctx) {
        for (auto& stage : backbone_sa)
            for (auto& branch : stage) remap_point_mlp(branch, device_ctx);
        for (auto& value : backbone_fp) remap_point_mlp(value, device_ctx);
        remap_point_mlp(point_cls, device_ctx);
        remap_point_mlp(point_box, device_ctx);
        remap_point_mlp(roi_xyz_up, device_ctx);
        remap_point_mlp(roi_merge, device_ctx);
        for (auto& value : roi_sa) remap_point_mlp(value, device_ctx);
        remap_point_mlp(roi_cls, device_ctx);
        remap_point_mlp(roi_reg, device_ctx);
    }
    return true;
}

bool pointrcnn::forward(const float* points,
                        int32_t count,
                        int32_t dimension,
                        std::vector<detection>& detections,
                        const tap_sink& sink) {
    detections.clear();
    const hparams& hp = file.hp;
    if (!points || count != hp.sample_points ||
        dimension != hp.num_point_features) {
        error = "PointRCNN requires the pinned, pre-sampled point count and "
                "feature dimension";
        return false;
    }
    // AICore port note: the upstream task-private CUDA FPS workspace is
    // dropped; the host FPS runs identically on every backend.
    std::vector<std::vector<float>> xyz(5), features(5);
    xyz[0].resize(static_cast<size_t>(count) * 3);
    features[0].resize(count);
    for (int row = 0; row < count; ++row) {
        std::copy_n(points + static_cast<int64_t>(row) * dimension, 3,
                    xyz[0].data() + row * 3);
        features[0][row] = points[static_cast<int64_t>(row) * dimension + 3];
    }
    size_t radius_offset = 0;
    for (int stage = 0; stage < 4; ++stage) {
        std::vector<float> radii(
                hp.pointnet_sa_radii.begin() + radius_offset,
                hp.pointnet_sa_radii.begin() + radius_offset + 2);
        std::vector<int32_t> samples(
                hp.pointnet_sa_nsamples.begin() + radius_offset,
                hp.pointnet_sa_nsamples.begin() + radius_offset + 2);
        if (!set_abstraction(be, xyz[stage].data(), xyz[stage].size() / 3,
                             features[stage], backbone_sa[stage], radii,
                             samples, hp.pointnet_sa_npoints[stage],
                             xyz[stage + 1], features[stage + 1], error))
            return false;
        radius_offset += 2;
        if (sink)
            sink("backbone.sa." + std::to_string(stage),
                 features[stage + 1].data(), xyz[stage + 1].size() / 3,
                 features[stage + 1].size() / (xyz[stage + 1].size() / 3));
    }
    for (int level = 3; level >= 0; --level) {
        std::vector<float> propagated, transformed;
        const int known_channels = static_cast<int>(
                features[level + 1].size() / (xyz[level + 1].size() / 3));
        const int skip_channels = static_cast<int>(features[level].size() /
                                                   (xyz[level].size() / 3));
        if (!ext::feature_propagate(
                    xyz[level].data(), xyz[level].size() / 3,
                    xyz[level + 1].data(), xyz[level + 1].size() / 3,
                    features[level + 1].data(), known_channels,
                    features[level].data(), skip_channels, propagated, error) ||
            !run_point_mlp(be, backbone_fp[level], propagated,
                           xyz[level].size() / 3, transformed, error))
            return false;
        features[level] = std::move(transformed);
    }
    if (sink) sink("backbone.fp.out", features[0].data(), count, 128);
    std::vector<float> cls, encoded;
    if (!run_point_mlp(be, point_cls, features[0], count, cls, error) ||
        !run_point_mlp(be, point_box, features[0], count, encoded, error))
        return false;
    std::vector<float> point_boxes =
            decode_point_boxes(hp, xyz[0].data(), cls, encoded);
    ext::proposal_result proposals;
    if (!ext::select_proposals(point_boxes.data(), count, cls.data(),
                               hp.num_class, false, hp.roi_proposal_pre_max,
                               hp.roi_proposal_post_max,
                               hp.roi_proposal_nms_thresh, proposals, error))
        return false;
    const int roi_count = static_cast<int>(proposals.indices.size());
    if (roi_count == 0) return true;
    std::vector<float> rois(static_cast<size_t>(roi_count) * 7);
    for (int roi = 0; roi < roi_count; ++roi)
        std::copy_n(point_boxes.data() +
                            static_cast<int64_t>(proposals.indices[roi]) * 7,
                    7, rois.data() + static_cast<int64_t>(roi) * 7);

    std::vector<float> pool_features(static_cast<size_t>(count) * 130);
    for (int row = 0; row < count; ++row) {
        float maximum = cls[row * hp.num_class];
        for (int label = 1; label < hp.num_class; ++label)
            maximum = std::max(maximum, cls[row * hp.num_class + label]);
        pool_features[row * 130] = 1.0f / (1.0f + std::exp(-maximum));
        const float x = xyz[0][row * 3], y = xyz[0][row * 3 + 1],
                    z = xyz[0][row * 3 + 2];
        pool_features[row * 130 + 1] =
                std::sqrt(x * x + y * y + z * z) / hp.roi_depth_normalizer -
                0.5f;
        std::copy_n(features[0].data() + row * 128, 128,
                    pool_features.data() + row * 130 + 2);
    }
    std::vector<float> pooled;
    std::vector<uint8_t> empty;
    if (!ext::roi_point_pool3d(
                rois.data(), roi_count, xyz[0].data(), count,
                pool_features.data(), 130, hp.roi_point_extra_width.data(),
                hp.roi_point_sample_count, pooled, empty, error) ||
        !ext::canonicalize_roi_points(rois.data(), roi_count,
                                      hp.roi_point_sample_count, 133, pooled,
                                      error))
        return false;
    const int64_t pooled_rows =
            static_cast<int64_t>(roi_count) * hp.roi_point_sample_count;
    std::vector<float> xyz_prefix(static_cast<size_t>(pooled_rows) * 5);
    std::vector<float> merged_input(static_cast<size_t>(pooled_rows) * 256);
    for (int64_t row = 0; row < pooled_rows; ++row)
        std::copy_n(pooled.data() + row * 133, 5, xyz_prefix.data() + row * 5);
    std::vector<float> xyz_features;
    if (!run_point_mlp(be, roi_xyz_up, xyz_prefix, pooled_rows, xyz_features,
                       error))
        return false;
    for (int64_t row = 0; row < pooled_rows; ++row) {
        std::copy_n(xyz_features.data() + row * 128, 128,
                    merged_input.data() + row * 256);
        std::copy_n(pooled.data() + row * 133 + 5, 128,
                    merged_input.data() + row * 256 + 128);
    }
    std::vector<float> roi_features;
    if (!run_point_mlp(be, roi_merge, merged_input, pooled_rows, roi_features,
                       error))
        return false;

    int current_points = hp.roi_point_sample_count;
    std::vector<float> current_xyz(static_cast<size_t>(roi_count) *
                                   current_points * 3);
    for (int64_t row = 0; row < pooled_rows; ++row)
        std::copy_n(pooled.data() + row * 133, 3, current_xyz.data() + row * 3);
    for (int stage = 0; stage < 3; ++stage) {
        const int next_points =
                hp.roi_sa_npoints[stage] < 0 ? 1 : hp.roi_sa_npoints[stage];
        const int nsample = hp.roi_sa_npoints[stage] < 0
                                    ? current_points
                                    : hp.roi_sa_nsamples[stage];
        const int feature_channels = static_cast<int>(
                roi_features.size() /
                (static_cast<size_t>(roi_count) * current_points));
        std::vector<float> grouped;
        std::vector<float> next_xyz(static_cast<size_t>(roi_count) *
                                    next_points * 3);
        for (int roi = 0; roi < roi_count; ++roi) {
            const float* roi_xyz =
                    current_xyz.data() +
                    static_cast<int64_t>(roi) * current_points * 3;
            const float* roi_values = roi_features.data() +
                                      static_cast<int64_t>(roi) *
                                              current_points * feature_channels;
            if (hp.roi_sa_npoints[stage] < 0) {
                for (int point = 0; point < current_points; ++point) {
                    grouped.insert(grouped.end(), roi_xyz + point * 3,
                                   roi_xyz + point * 3 + 3);
                    grouped.insert(grouped.end(),
                                   roi_values + point * feature_channels,
                                   roi_values + point * feature_channels +
                                           feature_channels);
                }
            } else {
                std::vector<int32_t> centers, indices;
                std::vector<float> center_xyz, roi_grouped;
                if (!ext::farthest_point_sample(roi_xyz, current_points,
                                                next_points, centers, error))
                    return false;
                if (!ext::sample_and_group(roi_xyz, current_points, roi_values,
                                           feature_channels, centers.data(),
                                           next_points, hp.roi_sa_radii[stage],
                                           nsample, true, center_xyz, indices,
                                           roi_grouped, error))
                    return false;
                std::copy(center_xyz.begin(), center_xyz.end(),
                          next_xyz.begin() +
                                  static_cast<int64_t>(roi) * next_points * 3);
                grouped.insert(grouped.end(), roi_grouped.begin(),
                               roi_grouped.end());
            }
        }
        std::vector<float> transformed, pooled_stage;
        if (!run_point_mlp(
                    be, roi_sa[stage], grouped,
                    static_cast<int64_t>(roi_count) * next_points * nsample,
                    transformed, error) ||
            !max_pool_groups(transformed,
                             static_cast<int64_t>(roi_count) * next_points,
                             nsample, pooled_stage, error))
            return false;
        current_xyz = std::move(next_xyz);
        roi_features = std::move(pooled_stage);
        current_points = next_points;
    }
    std::vector<float> refined_cls, residuals;
    if (!run_point_mlp(be, roi_cls, roi_features, roi_count, refined_cls,
                       error) ||
        !run_point_mlp(be, roi_reg, roi_features, roi_count, residuals, error))
        return false;
    std::vector<float> refined_boxes;
    if (!ext::decode_roi_residuals(rois.data(), residuals.data(), roi_count,
                                   refined_boxes, error))
        return false;
    std::vector<int32_t> candidates;
    std::vector<float> scores(roi_count);
    for (int roi = 0; roi < roi_count; ++roi) {
        scores[roi] = 1.0f / (1.0f + std::exp(-refined_cls[roi]));
        if (scores[roi] >= hp.score_thresh) candidates.push_back(roi);
    }
    if (static_cast<int>(candidates.size()) > hp.nms_pre_maxsize) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + hp.nms_pre_maxsize,
                          candidates.end(),
                          [&](int a, int b) { return scores[a] > scores[b]; });
        candidates.resize(hp.nms_pre_maxsize);
    }
    std::vector<float> candidate_boxes(candidates.size() * 7),
            candidate_scores(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        std::copy_n(refined_boxes.data() + candidates[index] * 7, 7,
                    candidate_boxes.data() + index * 7);
        candidate_scores[index] = scores[candidates[index]];
    }
    const std::vector<int32_t> kept = nms_3d(candidate_boxes, candidate_scores,
                                             hp.nms_thresh, hp.max_boxes);
    for (int32_t local : kept) {
        const int roi = candidates[local];
        const float* box = refined_boxes.data() + roi * 7;
        detections.push_back({box[0], box[1], box[2], box[3], box[4], box[5],
                              box[6], scores[roi], proposals.labels[roi]});
    }
    return true;
}

}  // namespace openpcdet
