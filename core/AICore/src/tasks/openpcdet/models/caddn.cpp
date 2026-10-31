// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "caddn.h"

#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>

#include "../extensions/image_geometry_ops.h"
#include "dense_anchor_pipeline.h"

namespace openpcdet {
namespace {

ggml_tensor *require_resolved(caddn &model, const std::string &name) {
    ggml_tensor *tensor = model.weight(name);
    if (!tensor && model.error.empty()) model.error = "missing tensor: " + name;
    return tensor;
}

conv_bn_block_weights load_block(caddn &model,
                                 const std::string &conv,
                                 const std::string &bn) {
    return {require_resolved(model, conv),
            require_resolved(model, bn + ".scale"),
            require_resolved(model, bn + ".bias")};
}

ggml_tensor *conv_bn(caddn &model,
                     ggml_context *context,
                     const conv_bn_block_weights &weights,
                     ggml_tensor *input,
                     int stride,
                     int padding,
                     int dilation = 1) {
    ggml_tensor *output = build_conv_bias_dilated(
            context, weights.conv_w, nullptr, input, stride, padding, dilation);
    ggml_tensor *scale =
            ggml_reshape_4d(context, weights.bn_scale, 1, 1, output->ne[2], 1);
    ggml_tensor *bias =
            ggml_reshape_4d(context, weights.bn_bias, 1, 1, output->ne[2], 1);
    (void)model;
    return ggml_add(context, ggml_mul(context, output, scale), bias);
}

ggml_tensor *bottleneck(caddn &model,
                        ggml_context *context,
                        const caddn_bottleneck_weights &weights,
                        ggml_tensor *input,
                        int stride,
                        int dilation) {
    ggml_tensor *output = ggml_relu(
            context, conv_bn(model, context, weights.conv1, input, 1, 0));
    output = ggml_relu(context, conv_bn(model, context, weights.conv2, output,
                                        stride, dilation, dilation));
    output = conv_bn(model, context, weights.conv3, output, 1, 0);
    ggml_tensor *identity =
            weights.has_downsample ? conv_bn(model, context, weights.downsample,
                                             input, stride, 0)
                                   : input;
    return ggml_relu(context, ggml_add(context, output, identity));
}

std::vector<float> readback(ggml_tensor *tensor) {
    std::vector<float> output(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, output.data(), 0,
                            output.size() * sizeof(float));
    return output;
}

bool invert_4x4(const float *source, float *inverse) {
    double augmented[4][8] = {};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column)
            augmented[row][column] = source[row * 4 + column];
        augmented[row][row + 4] = 1.0;
    }
    for (int column = 0; column < 4; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 4; ++row)
            if (std::abs(augmented[row][column]) >
                std::abs(augmented[pivot][column]))
                pivot = row;
        if (std::abs(augmented[pivot][column]) < 1e-12) return false;
        if (pivot != column)
            for (int k = 0; k < 8; ++k)
                std::swap(augmented[pivot][k], augmented[column][k]);
        const double divisor = augmented[column][column];
        for (int k = 0; k < 8; ++k) augmented[column][k] /= divisor;
        for (int row = 0; row < 4; ++row)
            if (row != column) {
                const double factor = augmented[row][column];
                for (int k = 0; k < 8; ++k)
                    augmented[row][k] -= factor * augmented[column][k];
            }
    }
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            inverse[row * 4 + column] =
                    static_cast<float>(augmented[row][column + 4]);
    return true;
}

void multiply_4x4(const float *left, const float *right, float *output) {
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column) {
            output[row * 4 + column] = 0.0f;
            for (int k = 0; k < 4; ++k)
                output[row * 4 + column] +=
                        left[row * 4 + k] * right[k * 4 + column];
        }
}

bool sample_frustum(const hparams &hp,
                    const std::vector<float> &features,
                    const std::vector<float> &logits,
                    int width,
                    int height,
                    const float lidar_to_camera[16],
                    const float camera_to_image[12],
                    std::vector<float> &voxel,
                    std::string &error) {
    const int channels = 64, bins = hp.caddn_depth_bins;
    if (features.size() != static_cast<size_t>(width) * height * channels ||
        logits.size() != static_cast<size_t>(width) * height * (bins + 1)) {
        error = "CaDDN image graph output shape mismatch";
        return false;
    }
    std::vector<float> probabilities(static_cast<size_t>(width) * height *
                                     bins);
#pragma omp parallel for schedule(static)
    for (int pixel = 0; pixel < width * height; ++pixel) {
        float maximum = -INFINITY;
        for (int depth = 0; depth <= bins; ++depth)
            maximum = std::max(maximum,
                               logits[pixel + static_cast<size_t>(width) *
                                                      height * depth]);
        float sum = 0.0f;
        for (int depth = 0; depth <= bins; ++depth)
            sum += std::exp(logits[pixel + static_cast<size_t>(width) * height *
                                                   depth] -
                            maximum);
        for (int depth = 0; depth < bins; ++depth)
            probabilities[pixel + static_cast<size_t>(width) * height * depth] =
                    std::exp(logits[pixel + static_cast<size_t>(width) *
                                                    height * depth] -
                             maximum) /
                    sum;
    }
    ext::frustum_grid_params params;
    params.grid_size = hp.grid_size;
    params.point_cloud_range = hp.point_cloud_range;
    params.image_height = hp.caddn_image_height;
    params.image_width = hp.caddn_image_width;
    params.depth_bins = bins;
    params.depth_min = hp.caddn_depth_min;
    params.depth_max = hp.caddn_depth_max;
    params.bin_mode = ext::depth_bin_mode::linear_increasing;
    std::vector<float> grid;
    if (!ext::generate_frustum_grid(lidar_to_camera, camera_to_image, params,
                                    grid, error))
        return false;
    const int nx = hp.grid_size[0], ny = hp.grid_size[1], nz = hp.grid_size[2];
    voxel.assign(static_cast<size_t>(nx) * ny * nz * channels, 0.0f);
#pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                const float *coordinate =
                        grid.data() +
                        ((static_cast<size_t>(x) * ny + y) * nz + z) * 3;
                const float fx = (coordinate[0] + 1.0f) * 0.5f * (width - 1);
                const float fy = (coordinate[1] + 1.0f) * 0.5f * (height - 1);
                const float fd = (coordinate[2] + 1.0f) * 0.5f * (bins - 1);
                const int x0 = static_cast<int>(std::floor(fx));
                const int y0 = static_cast<int>(std::floor(fy));
                const int d0 = static_cast<int>(std::floor(fd));
                const float tx = fx - x0, ty = fy - y0, td = fd - d0;
                for (int dz = 0; dz < 2; ++dz)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            const int ix = x0 + dx, iy = y0 + dy, id = d0 + dz;
                            if (ix < 0 || ix >= width || iy < 0 ||
                                iy >= height || id < 0 || id >= bins)
                                continue;
                            const float weight = (dx ? tx : 1.0f - tx) *
                                                 (dy ? ty : 1.0f - ty) *
                                                 (dz ? td : 1.0f - td);
                            const size_t pixel =
                                    static_cast<size_t>(ix) +
                                    static_cast<size_t>(width) * iy;
                            const float depth_probability =
                                    probabilities[pixel +
                                                  static_cast<size_t>(width) *
                                                          height * id];
                            for (int channel = 0; channel < channels;
                                 ++channel) {
                                const size_t destination =
                                        static_cast<size_t>(x) +
                                        static_cast<size_t>(nx) *
                                                (y +
                                                 static_cast<size_t>(ny) *
                                                         (z + nz * channel));
                                voxel[destination] +=
                                        weight * depth_probability *
                                        features[pixel +
                                                 static_cast<size_t>(width) *
                                                         height * channel];
                            }
                        }
            }
        }
    return true;
}

}  // namespace

bool caddn::load(const std::string &path,
                 const std::string &device,
                 int threads) {
    if (!model::load(path, device, threads)) return false;
    if (file.hp.arch != arch_kind::caddn ||
        file.hp.head_kind != dense_head_kind::anchor_single) {
        error = "unsupported CaDDN topology";
        return false;
    }
    image_stem = load_block(*this, "vfe.ffn.ddn.model.backbone.conv1.weight",
                            "vfe.ffn.ddn.model.backbone.bn1.eval");
    const int counts[4] = {3, 4, 23, 3};
    for (int layer = 0; layer < 4; ++layer)
        for (int block = 0; block < counts[layer]; ++block) {
            const std::string prefix = "vfe.ffn.ddn.model.backbone.layer" +
                                       std::to_string(layer + 1) + "." +
                                       std::to_string(block);
            auto &target = image_layers[layer].emplace_back();
            target.conv1 = load_block(*this, prefix + ".conv1.weight",
                                      prefix + ".bn1.eval");
            target.conv2 = load_block(*this, prefix + ".conv2.weight",
                                      prefix + ".bn2.eval");
            target.conv3 = load_block(*this, prefix + ".conv3.weight",
                                      prefix + ".bn3.eval");
            target.has_downsample = block == 0;
            if (target.has_downsample)
                target.downsample =
                        load_block(*this, prefix + ".downsample.0.weight",
                                   prefix + ".downsample.1.eval");
        }
    for (int branch = 0; branch < 4; ++branch) {
        const std::string prefix = "vfe.ffn.ddn.model.classifier.0.convs." +
                                   std::to_string(branch);
        aspp[branch] =
                load_block(*this, prefix + ".0.weight", prefix + ".1.eval");
    }
    aspp[4] =
            load_block(*this, "vfe.ffn.ddn.model.classifier.0.convs.4.1.weight",
                       "vfe.ffn.ddn.model.classifier.0.convs.4.2.eval");
    aspp_project =
            load_block(*this, "vfe.ffn.ddn.model.classifier.0.project.0.weight",
                       "vfe.ffn.ddn.model.classifier.0.project.1.eval");
    classifier_hidden =
            load_block(*this, "vfe.ffn.ddn.model.classifier.1.weight",
                       "vfe.ffn.ddn.model.classifier.2.eval");
    classifier_output_weight =
            require_resolved(*this, "vfe.ffn.ddn.model.classifier.4.weight");
    classifier_output_bias =
            require_resolved(*this, "vfe.ffn.ddn.model.classifier.4.bias");
    channel_reduce = load_block(*this, "vfe.ffn.channel_reduce.conv.weight",
                                "vfe.ffn.channel_reduce.bn.eval");
    collapse = load_block(*this, "caddn.collapse.weight",
                          "caddn.collapse.bn_fold");
    if (!error.empty()) return false;
    if (!load_bev_backbone_weights(file, file.hp, backbone, error) ||
        !load_anchor_head_weights(file, file.hp, head, error))
        return false;
    if (device_ctx) {
        auto remap = [&](ggml_tensor *&tensor) {
            if (tensor)
                tensor = ggml_get_tensor(device_ctx, ggml_get_name(tensor));
        };
        for (auto &level : backbone.levels)
            for (auto &block : level) {
                remap(block.conv_w);
                remap(block.bn_scale);
                remap(block.bn_bias);
            }
        for (auto &block : backbone.deblocks) {
            remap(block.conv_w);
            remap(block.bn_scale);
            remap(block.bn_bias);
        }
        remap(head.conv_cls_w);
        remap(head.conv_cls_b);
        remap(head.conv_box_w);
        remap(head.conv_box_b);
        remap(head.conv_dir_cls_w);
        remap(head.conv_dir_cls_b);
    }
    return true;
}

bool caddn::forward(const float *,
                    int32_t,
                    int32_t,
                    std::vector<detection> &detections,
                    const tap_sink &) {
    detections.clear();
    error = "CaDDN requires openpcdet_run_scene with image and calibration "
            "matrices";
    return false;
}

bool caddn::forward_scene(const scene_input &scene,
                          std::vector<detection> &detections,
                          const tap_sink &sink) {
    detections.clear();
    const hparams &hp = file.hp;
    if (!scene.images || scene.num_cameras != 1 ||
        scene.image_height != hp.caddn_image_height ||
        scene.image_width != hp.caddn_image_width ||
        scene.image_channels != 3 || !scene.lidar_to_image ||
        !scene.camera_intrinsics) {
        error = "CaDDN requires one RGB image and 4x4 "
                "lidar_to_image/camera_intrinsics matrices";
        return false;
    }
    const int input_width = scene.image_width,
              input_height = scene.image_height;
    std::vector<float> input(static_cast<size_t>(input_width) * input_height *
                             3);
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float deviation[3] = {0.229f, 0.224f, 0.225f};
    for (int channel = 0; channel < 3; ++channel)
        for (int y = 0; y < input_height; ++y)
            for (int x = 0; x < input_width; ++x) {
                const float value =
                        scene.images[(static_cast<size_t>(y) * input_width +
                                      x) * 3 +
                                     channel];
                input[static_cast<size_t>(x) +
                      static_cast<size_t>(input_width) *
                              (y + input_height * channel)] =
                        value == 0.0f
                                ? 0.0f
                                : (value - mean[channel]) / deviation[channel];
            }

    constexpr size_t graph_nodes = 16384;
    ggml_init_params params{
            ggml_tensor_overhead() * graph_nodes +
                    ggml_graph_overhead_custom(graph_nodes, false),
            nullptr, true};
    ggml_context *context = ggml_init(params);
    if (!context) {
        error = "CaDDN image graph context init failed";
        return false;
    }
    ggml_tensor *image = ggml_new_tensor_4d(context, GGML_TYPE_F32, input_width,
                                            input_height, 3, 1);
    ggml_set_input(image);
    ggml_tensor *value = ggml_relu(
            context, conv_bn(*this, context, image_stem, image, 2, 3));
    value = ggml_pool_2d(context, value, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1);
    ggml_tensor *image_features = nullptr;
    const int strides[4] = {1, 2, 1, 1};
    const int base_dilations[4] = {1, 1, 2, 4};
    for (int layer = 0; layer < 4; ++layer)
        for (size_t block = 0; block < image_layers[layer].size(); ++block) {
            const int stride = block == 0 ? strides[layer] : 1;
            int dilation = base_dilations[layer];
            if (block == 0 && layer >= 2) dilation /= 2;
            value = bottleneck(*this, context, image_layers[layer][block],
                               value, stride, dilation);
            if (layer == 0 && block + 1 == image_layers[layer].size())
                image_features = value;
        }
    std::array<ggml_tensor *, 5> branches{};
    branches[0] = build_conv_bn_relu(context, aspp[0], value, 1, 0);
    const int rates[3] = {12, 24, 36};
    for (int branch = 1; branch < 4; ++branch)
        branches[branch] = build_conv_bn_relu_dilated(
                context, aspp[branch], value, 1, rates[branch - 1],
                rates[branch - 1]);
    ggml_tensor *global =
            ggml_pool_2d(context, value, GGML_OP_POOL_AVG, value->ne[0],
                         value->ne[1], value->ne[0], value->ne[1], 0, 0);
    global = build_conv_bn_relu(context, aspp[4], global, 1, 0);
    branches[4] = ggml_repeat(context, global, branches[0]);
    ggml_tensor *classifier = branches[0];
    for (int branch = 1; branch < 5; ++branch)
        classifier = ggml_concat(context, classifier, branches[branch], 2);
    classifier = build_conv_bn_relu(context, aspp_project, classifier, 1, 0);
    classifier =
            build_conv_bn_relu(context, classifier_hidden, classifier, 1, 1);
    ggml_tensor *depth_logits =
            build_conv_bias(context, classifier_output_weight,
                            classifier_output_bias, classifier, 1, 0);
    depth_logits = ggml_interpolate(
            context, depth_logits, image_features->ne[0], image_features->ne[1],
            depth_logits->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
    image_features =
            build_conv_bn_relu(context, channel_reduce, image_features, 1, 0);
    ggml_set_output(depth_logits);
    ggml_set_output(image_features);
    ggml_cgraph *graph = ggml_new_graph_custom(context, graph_nodes, false);
    ggml_build_forward_expand(graph, depth_logits);
    ggml_build_forward_expand(graph, image_features);
    if (!ggml_gallocr_alloc_graph(be.galloc, graph)) {
        error = "CaDDN image graph allocation failed";
        ggml_free(context);
        return false;
    }
    ggml_backend_tensor_set(image, input.data(), 0,
                            input.size() * sizeof(float));
    if (ggml_backend_graph_compute(be.be, graph) != GGML_STATUS_SUCCESS) {
        error = "CaDDN image graph compute failed";
        ggml_free(context);
        return false;
    }
    const int feature_width = image_features->ne[0],
              feature_height = image_features->ne[1];
    // AICore port note: the upstream CUDA-resident frustum sampling fast path
    // (caddn_cuda extension) is intentionally absent — every backend runs the
    // host frustum sampler below, so image features and depth logits are always
    // read back to host memory.
    std::vector<float> feature_values = readback(image_features);
    std::vector<float> logit_values = readback(depth_logits);
    std::vector<float> voxel;
    if (sink) {
        sink("caddn.image_features", feature_values.data(),
             static_cast<int64_t>(feature_width) * feature_height, 64);
        sink("caddn.depth_logits", logit_values.data(),
             static_cast<int64_t>(feature_width) * feature_height,
             hp.caddn_depth_bins + 1);
    }
    float inverse_intrinsics[16], lidar_to_camera[16];
    if (!invert_4x4(scene.camera_intrinsics, inverse_intrinsics)) {
        error = "CaDDN camera intrinsics matrix is singular";
        return false;
    }
    multiply_4x4(inverse_intrinsics, scene.lidar_to_image, lidar_to_camera);
    float camera_to_image[12];
    for (int row = 0; row < 3; ++row)
        std::copy_n(scene.camera_intrinsics + row * 4, 4,
                    camera_to_image + row * 4);
    ggml_free(context);
    if (!sample_frustum(hp, feature_values, logit_values, feature_width,
                        feature_height, lidar_to_camera, camera_to_image, voxel,
                        error))
        return false;
    if (sink)
        sink("caddn.voxel_features", voxel.data(),
             static_cast<int64_t>(hp.grid_size[0]) * hp.grid_size[1],
             static_cast<int64_t>(hp.grid_size[2]) * 64);

    context = ggml_init(params);
    if (!context) {
        error = "CaDDN collapse context init failed";
        return false;
    }
    ggml_tensor *voxel_input =
            ggml_new_tensor_4d(context, GGML_TYPE_F32, hp.grid_size[0],
                               hp.grid_size[1], hp.grid_size[2] * 64, 1);
    ggml_set_input(voxel_input);
    ggml_tensor *spatial =
            build_conv_bn_relu(context, collapse, voxel_input, 1, 0);
    ggml_set_output(spatial);
    graph = ggml_new_graph_custom(context, 128, false);
    ggml_build_forward_expand(graph, spatial);
    if (!ggml_gallocr_alloc_graph(be.galloc, graph)) {
        error = "CaDDN collapse graph allocation failed";
        ggml_free(context);
        return false;
    }
    ggml_backend_tensor_set(voxel_input, voxel.data(), 0,
                            voxel.size() * sizeof(float));
    if (ggml_backend_graph_compute(be.be, graph) != GGML_STATUS_SUCCESS) {
        error = "CaDDN collapse graph compute failed";
        ggml_free(context);
        return false;
    }
    std::vector<float> spatial_whc = readback(spatial);
    std::vector<float> spatial_hwc =
            whc_to_hwc(spatial_whc.data(), hp.grid_size[0], hp.grid_size[1],
                       hp.num_bev_features);
    ggml_free(context);
    voxel.clear();
    voxel.shrink_to_fit();
    if (sink)
        sink("spatial_features", spatial_hwc.data(),
             static_cast<int64_t>(hp.grid_size[0]) * hp.grid_size[1],
             hp.num_bev_features);
    const bool ok = run_dense_anchor_pipeline(
            be, hp, backbone, head, spatial_hwc, hp.grid_size[0],
            hp.grid_size[1], detections, sink, error);
    return ok;
}

}  // namespace openpcdet
