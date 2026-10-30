#include "voxelrcnn.h"

#include "../extensions/deterministic_sparse_conv.h"
#include "../extensions/focal_sparse_conv.h"
#include "../extensions/roi_ops.h"
#include "../ops/nms_3d.h"
#include "../ops/voxelization.h"
#include "dense_anchor_pipeline.h"
#include "voxel_backbone_runner.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace openpcdet {

namespace {

std::vector<point_mlp_layer_spec> hidden_specs(const std::string & prefix) {
    return {
        {prefix + ".0.weight", "", prefix + ".1", true},
        {prefix + ".4.weight", "", prefix + ".5", true},
    };
}

int layer_index(const sparse_backbone_weights & weights, const std::string & name) {
    for (size_t index = 0; index < weights.specs.size(); ++index)
        if (weights.specs[index].name == name) return static_cast<int>(index);
    return -1;
}

std::vector<float> roi_grid_points(const std::vector<detection> & proposals, int grid) {
    const int grids = grid * grid * grid;
    std::vector<float> result(static_cast<size_t>(proposals.size()) * grids * 3);
    for (size_t roi = 0; roi < proposals.size(); ++roi) {
        const auto & box = proposals[roi];
        const float cosine = std::cos(box.heading), sine = std::sin(box.heading);
        int index = 0;
        for (int x = 0; x < grid; ++x)
            for (int y = 0; y < grid; ++y)
                for (int z = 0; z < grid; ++z, ++index) {
                    const float local_x = (x + 0.5f) / grid * box.dx - box.dx * 0.5f;
                    const float local_y = (y + 0.5f) / grid * box.dy - box.dy * 0.5f;
                    float * target = result.data() + (roi * grids + index) * 3;
                    target[0] = local_x * cosine - local_y * sine + box.x;
                    target[1] = local_x * sine + local_y * cosine + box.y;
                    target[2] = (z + 0.5f) / grid * box.dz - box.dz * 0.5f + box.z;
                }
    }
    return result;
}

bool voxel_pool_source(engine_backend & backend, const hparams & hp,
                       const ext::sparse_indice_map & map,
                       const std::vector<float> & features, int channels, int stride,
                       const std::vector<float> & grid_xyz, float radius, int nsample,
                       const int32_t * query_range, const voxelrcnn_pool_weights & weights,
                       std::vector<float> & output, std::vector<float> * neighbor_xyz,
                       std::string & error) {
    const int64_t source_count = map.output_active_count();
    const int64_t center_count = static_cast<int64_t>(grid_xyz.size()) / 3;
    if (source_count <= 0 || center_count <= 0 || channels <= 0 || nsample <= 0 ||
        features.size() != static_cast<size_t>(source_count) * channels) {
        error = "bad VoxelRCNN voxel-pool input"; return false;
    }
    std::vector<float> transformed_features;
    if (!run_point_mlp(backend, weights.feature_in, features, source_count,
                       transformed_features, error)) return false;
    const int hidden = static_cast<int>(transformed_features.size() / source_count);
    const int depth = map.spatial_shape[0], height = map.spatial_shape[1], width = map.spatial_shape[2];
    std::vector<int32_t> voxel_to_row(static_cast<size_t>(depth) * height * width, -1);
    for (int64_t row = 0; row < source_count; ++row) {
        const int32_t * coordinate = map.coordinates.data() + row * 4;
        voxel_to_row[(static_cast<size_t>(coordinate[1]) * height + coordinate[2]) * width +
                     coordinate[3]] = static_cast<int32_t>(row);
    }
    std::vector<int32_t> indices(static_cast<size_t>(center_count) * nsample, -1);
    const float radius2 = radius * radius;
    for (int64_t center = 0; center < center_count; ++center) {
        const float * point = grid_xyz.data() + center * 3;
        const int cx = static_cast<int>(std::floor((point[0] - hp.point_cloud_range[0]) /
                                                   hp.voxel_size[0])) / stride;
        const int cy = static_cast<int>(std::floor((point[1] - hp.point_cloud_range[1]) /
                                                   hp.voxel_size[1])) / stride;
        const int cz = static_cast<int>(std::floor((point[2] - hp.point_cloud_range[2]) /
                                                   hp.voxel_size[2])) / stride;
        int found = 0;
        for (int dz = -query_range[0]; dz <= query_range[0] && found < nsample; ++dz) {
            const int z = cz + dz; if (z < 0 || z >= depth) continue;
            for (int dy = -query_range[1]; dy <= query_range[1] && found < nsample; ++dy) {
                const int y = cy + dy; if (y < 0 || y >= height) continue;
                for (int dx = -query_range[2]; dx <= query_range[2] && found < nsample; ++dx) {
                    const int x = cx + dx; if (x < 0 || x >= width) continue;
                    const int32_t row = voxel_to_row[(static_cast<size_t>(z) * height + y) * width + x];
                    if (row < 0) continue;
                    const int32_t * coordinate = map.coordinates.data() + static_cast<int64_t>(row) * 4;
                    const float sx = (coordinate[3] + 0.5f) * hp.voxel_size[0] * stride + hp.point_cloud_range[0];
                    const float sy = (coordinate[2] + 0.5f) * hp.voxel_size[1] * stride + hp.point_cloud_range[1];
                    const float sz = (coordinate[1] + 0.5f) * hp.voxel_size[2] * stride + hp.point_cloud_range[2];
                    const float ddx = sx-point[0], ddy = sy-point[1], ddz = sz-point[2];
                    if (ddx*ddx + ddy*ddy + ddz*ddz > radius2) continue;
                    if (found == 0) std::fill_n(indices.data() + center * nsample, nsample, row);
                    indices[center * nsample + found++] = row;
                }
            }
        }
    }
    std::vector<float> relative(static_cast<size_t>(center_count) * nsample * 3, 0.0f);
    if (neighbor_xyz) neighbor_xyz->assign(relative.size(), 0.0f);
    for (int64_t center = 0; center < center_count; ++center)
        for (int sample = 0; sample < nsample; ++sample) {
            const int32_t row = indices[center * nsample + sample];
            if (row < 0) continue;
            const int32_t * coordinate = map.coordinates.data() + static_cast<int64_t>(row) * 4;
            float * target = relative.data() + (center * nsample + sample) * 3;
            target[0] = (coordinate[3] + 0.5f) * hp.voxel_size[0] * stride +
                        hp.point_cloud_range[0] - grid_xyz[center * 3];
            target[1] = (coordinate[2] + 0.5f) * hp.voxel_size[1] * stride +
                        hp.point_cloud_range[1] - grid_xyz[center * 3 + 1];
            target[2] = (coordinate[1] + 0.5f) * hp.voxel_size[2] * stride +
                        hp.point_cloud_range[2] - grid_xyz[center * 3 + 2];
            if (neighbor_xyz) {
                float * absolute = neighbor_xyz->data() + (center * nsample + sample) * 3;
                absolute[0] = target[0] + grid_xyz[center * 3];
                absolute[1] = target[1] + grid_xyz[center * 3 + 1];
                absolute[2] = target[2] + grid_xyz[center * 3 + 2];
            }
        }
    std::vector<float> position;
    if (!run_point_mlp(backend, weights.position, relative,
                       center_count * nsample, position, error)) return false;
    std::vector<float> pooled(static_cast<size_t>(center_count) * hidden, 0.0f);
    for (int64_t center = 0; center < center_count; ++center) {
        const bool empty = indices[center * nsample] < 0;
        for (int channel = 0; channel < hidden; ++channel) {
            float maximum = 0.0f;
            if (!empty) {
                maximum = 0.0f;
                for (int sample = 0; sample < nsample; ++sample) {
                    const int32_t row = indices[center * nsample + sample];
                    const float value = transformed_features[static_cast<int64_t>(row) * hidden + channel] +
                        position[(center * nsample + sample) * hidden + channel];
                    maximum = std::max(maximum, value);
                }
            }
            pooled[center * hidden + channel] = maximum;
        }
    }
    return run_point_mlp(backend, weights.output, pooled, center_count, output, error);
}

ggml_tensor * require_semseg(voxelrcnn & model, const std::string & name) {
    ggml_tensor * tensor = model.weight(name);
    if (!tensor && model.error.empty()) model.error = "missing tensor: " + name;
    return tensor;
}

// Like CaDDN's load_block, but reads the focal semseg BN-fold names
// (`<bn>.eval_scale` / `<bn>.eval_bias`) written by convert_focal_voxelrcnn.py.
conv_bn_block_weights load_semseg_block(voxelrcnn & model, const std::string & conv,
                                        const std::string & bn) {
    return {require_semseg(model, conv),
            require_semseg(model, bn + ".eval_scale"),
            require_semseg(model, bn + ".eval_bias")};
}

// 2-D conv + BN-fold affine. Reuses the same conv_bn contract as CaDDN: the
// converter writes the raw conv weight and folds BN into `.scale`/`.bias`.
ggml_tensor * conv_bn(voxelrcnn & model, ggml_context * context,
                      const conv_bn_block_weights & weights, ggml_tensor * input,
                      int stride, int padding, int dilation = 1) {
    ggml_tensor * output = build_conv_bias_dilated(context, weights.conv_w, nullptr,
                                                   input, stride, padding, dilation);
    ggml_tensor * scale = ggml_reshape_4d(context, weights.bn_scale, 1, 1, output->ne[2], 1);
    ggml_tensor * bias = ggml_reshape_4d(context, weights.bn_bias, 1, 1, output->ne[2], 1);
    (void) model;
    return ggml_add(context, ggml_mul(context, output, scale), bias);
}

// Standard ResNet Bottleneck (conv1 1x1, conv2 3x3, conv3 1x1 + optional
// downsample). ResNet50 layer1 runs with stride=1, dilation=1.
ggml_tensor * bottleneck(voxelrcnn & model, ggml_context * context,
                         const caddn_bottleneck_weights & weights,
                         ggml_tensor * input, int stride, int dilation) {
    ggml_tensor * output = ggml_relu(context, conv_bn(model, context, weights.conv1, input, 1, 0));
    output = ggml_relu(context, conv_bn(model, context, weights.conv2, output, stride, dilation, dilation));
    output = conv_bn(model, context, weights.conv3, output, 1, 0);
    ggml_tensor * identity = weights.has_downsample
        ? conv_bn(model, context, weights.downsample, input, stride, 0) : input;
    return ggml_relu(context, ggml_add(context, output, identity));
}

// ---------------------------------------------------------------------------
// Focal VoxelBackBone8x (upstream models/focal_backbone.cpp). AICore port
// note: stage timings are reported through the common aicore_pipeline_timings
// contract by the C API wrapper, so the upstream OPCPD_PROFILE timer is gone.
// ---------------------------------------------------------------------------

ext::sparse_conv_params conv3_params(bool submanifold, int stride = 1,
                                     std::vector<int32_t> padding = {1, 1, 1}) {
    ext::sparse_conv_params params;
    params.kernel_size = {3, 3, 3};
    params.stride = {stride, stride, stride};
    params.padding = std::move(padding);
    params.dilation = {1, 1, 1};
    params.submanifold = submanifold;
    return params;
}

// Run one sparse conv as a small GGML graph and read the output back to host.
// The active voxel set (indice map) is rebuilt from `in_coords` each call.
//
// AICore port note: stock-op lowering only. The input carries one appended
// zero row; `ext::remap_dense_indices` rewrites every -1 rulebook entry to
// that row, reproducing the upstream gather's "-1 writes zeros" semantics,
// and the fused [K*Cin, Cout] weight feeds a single get_rows + mul_mat
// (ext::build_sparse_convolution) on every backend.
//
// 方案 C (DETERMINISTIC_FOCAL_DESIGN.md): `deterministic = true` switches this
// conv to the fixed-order shared numeric core (extensions/
// deterministic_sparse_conv.h) instead of the ggml mul_mat graph — same
// reduction order (k outer / c inner) and single-rounding fma on CPU and CUDA,
// so the three backends agree bit-for-bit. Only the focal chain passes true;
// non-f32 weights fall back to the stock graph (determinism requires exact
// f32 dequantized-weight reads).
bool run_sparse_conv(engine_backend & backend, const ext::sparse_conv_params & params,
                     const std::vector<float> & in_features, int in_channels,
                     const std::vector<int32_t> & in_coords,
                     const std::vector<int32_t> & in_spatial_shape,
                     const sparse_layer_weights & weights,
                     std::vector<float> & out_features,
                     ext::sparse_indice_map & map,
                     std::vector<int32_t> & out_coords,
                     std::string & error,
                     bool apply_relu = true,
                     bool deterministic = false) {
    ext::sparse_tensor input;
    input.ndim = 3;
    input.spatial_shape = in_spatial_shape;
    input.coordinates = in_coords;
    input.channels = in_channels;
    if (!ext::build_sparse_indice_map_cached(input, params, map, error)) return false;
    const int64_t in_rows = static_cast<int64_t>(in_features.size()) / in_channels;
    const int64_t out_rows = map.output_active_count();
    const int64_t kernel_volume = map.kernel_volume;

    if (deterministic) {
        if (weights.weight == nullptr || weights.weight->type != GGML_TYPE_F32) {
            // Deterministic core consumes exact f32 weights; f16/q8_0 focal
            // variants keep the stock mul_mat graph (their outputs were never
            // bit-identical across backends by construction).
            error = "deterministic sparse conv requires f32 weights";
            return false;
        }
        if (weights.weight->ne[0] != kernel_volume * in_channels) {
            error = "deterministic sparse conv weight shape mismatch";
            return false;
        }
        const int out_channels = static_cast<int>(weights.weight->ne[1]);
        if (in_rows <= 0 || out_rows <= 0 ||
            in_rows > std::numeric_limits<int32_t>::max() / 2 ||
            out_rows > std::numeric_limits<int32_t>::max() / 2) {
            error = "deterministic sparse conv row count out of range";
            return false;
        }
        std::vector<int32_t> neighbor_rows;
        if (!det::det_expand_neighbor_rows(map, neighbor_rows, error)) return false;
        std::vector<float> weight_host(static_cast<size_t>(ggml_nelements(weights.weight)));
        ggml_backend_tensor_get(weights.weight, weight_host.data(), 0,
                                weight_host.size() * sizeof(float));
        std::vector<float> bias_host;
        if (weights.bias) {
            bias_host.resize(static_cast<size_t>(ggml_nelements(weights.bias)));
            ggml_backend_tensor_get(weights.bias, bias_host.data(), 0,
                                    bias_host.size() * sizeof(float));
        } else {
            // conv_imp is bias-free: +0.0f is the exact fma identity element.
            bias_host.assign(static_cast<size_t>(out_channels), 0.0f);
        }
        std::vector<float> det_out(static_cast<size_t>(out_rows) * out_channels);
        if (backend.is_cuda()) {
#ifdef AICORE_OPENPCDET_CUDA_EXT
            if (!det::det_sparse_conv_cuda(in_features.data(),
                                           static_cast<int>(in_rows), in_channels,
                                           neighbor_rows.data(), weight_host.data(),
                                           bias_host.data(), det_out.data(),
                                           static_cast<int>(out_rows), out_channels,
                                           static_cast<int>(kernel_volume), error))
                return false;
#else
            error = "deterministic CUDA path requires the OpenPCDet CUDA extension";
            return false;
#endif
        } else {
            det::det_sparse_conv_cpu(in_features.data(), in_channels,
                                     neighbor_rows.data(), weight_host.data(),
                                     bias_host.data(), det_out.data(),
                                     static_cast<int>(out_rows), out_channels,
                                     static_cast<int>(kernel_volume));
        }
        // Same relu the graph path applies on the backend: max(0, x) is exact
        // and backend-independent.
        if (apply_relu)
            for (float & value : det_out) value = det::det_relu(value);
        out_features = std::move(det_out);
        out_coords = map.coordinates;
        return true;
    }

    ggml_init_params graph_params = {
        ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false),
        nullptr, true,
    };
    ggml_context * ctx = ggml_init(graph_params);
    if (!ctx) { error = "focal sparse graph context init failed"; return false; }
    // [in_channels, in_rows + 1]: the trailing zero row absorbs -1 rulebook
    // entries exactly like the upstream sparse gather.
    ggml_tensor * features = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                in_channels, in_rows + 1);
    ggml_set_input(features);
    ggml_tensor * dense = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, kernel_volume * out_rows);
    ggml_set_input(dense);
    ggml_tensor * output = ext::build_sparse_convolution(
        ctx, features, dense, nullptr, nullptr, map, weights.weight, weights.bias);
    if (apply_relu) output = ggml_relu(ctx, output);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_gallocr_alloc_graph(backend.galloc, graph)) {
        error = "focal sparse graph allocation failed"; ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(features, in_features.data(), 0,
                            in_features.size() * sizeof(float));
    const std::vector<float> zeros(static_cast<size_t>(in_channels), 0.0f);
    ggml_backend_tensor_set(features, zeros.data(),
                            static_cast<size_t>(in_rows) * in_channels * sizeof(float),
                            zeros.size() * sizeof(float));
    std::vector<int32_t> remapped;
    ext::remap_dense_indices(map, static_cast<int32_t>(in_rows), remapped);
    ext::upload_dense_indices(dense, remapped);
    if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
        error = "focal sparse graph compute failed"; ggml_free(ctx); return false;
    }
    out_features.resize(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, out_features.data(), 0,
                            out_features.size() * sizeof(float));
    out_coords = map.coordinates;
    ggml_free(ctx);
    return true;
}

// FocalSparseConv core: conv_imp on `imp_features` -> split on `split_features`
// -> combine -> main conv on `split_channels`. split_features and imp_features
// differ only for the multimodal fusion point (imp feeds the concatenated
// image+voxel features while split feeds the LiDAR features).
//
// 方案 C: with hp.focal_deterministic both sparse convs of EVERY FocalSparseConv
// (conv_imp and the main conv — the main conv's output is the next layer's
// imp_features input, so determinizing conv_imp alone cannot stop the drift)
// run through the shared fixed-order core; false keeps the stock ggml mul_mat
// path as the A/B control.
bool run_focal_conv_core(engine_backend & backend, const hparams & hp,
                         const focal_conv_weights & weights,
                         const std::vector<float> & split_features, int split_channels,
                         const std::vector<float> & imp_features, int imp_channels,
                         const std::vector<int32_t> & coords,
                         const std::vector<int32_t> & spatial_shape,
                         std::vector<float> & out_features,
                         std::vector<int32_t> & out_coords,
                         ext::sparse_indice_map & out_map,
                         const tap_sink & sink, std::string & error) {
    const int kernel_volume = hp.focal_kernel_size * hp.focal_kernel_size * hp.focal_kernel_size;
    // 1. importance map
    std::vector<float> imps;
    ext::sparse_indice_map imp_map;
    std::vector<int32_t> imp_coords;
    if (!run_sparse_conv(backend, conv3_params(true), imp_features, imp_channels,
                         coords, spatial_shape, weights.conv_imp,
                         imps, imp_map, imp_coords, error, /*apply_relu=*/false,
                         /*deterministic=*/hp.focal_deterministic))
        return false;
    if (sink) {
        const char * tag = (imp_channels != split_channels) ? "focal_conv_imp_mm" :
                           (imp_channels == 16 ? "focal_conv_imp_c1" : imp_channels == 32 ? "focal_conv_imp_c2" : "focal_conv_imp_c3");
        sink(tag, imps.data(), static_cast<int64_t>(imps.size() / kernel_volume), kernel_volume);
    }
    // 2. split foreground/background (on the LiDAR features)
    std::vector<float> fore_feat, back_feat, mask_kernel;
    std::vector<int32_t> fore_coord, back_coord;
    const auto offsets = ext::focal_kernel_offsets(hp.focal_kernel_size);
    std::vector<int32_t> shape = {spatial_shape[0], spatial_shape[1], spatial_shape[2]};
    if (!ext::focal_split_voxels(split_features, coords, split_channels, imps,
                                 kernel_volume, hp.focal_kernel_size, offsets,
                                 shape, hp.focal_mask_multi, hp.focal_topk,
                                 hp.focal_threshold, fore_feat, fore_coord,
                                 back_feat, back_coord, mask_kernel, error))
        return false;
    // 3. multiply foreground mask kernel
    if (!hp.focal_skip_mask_kernel) {
        for (size_t r = 0; r < fore_feat.size() / static_cast<size_t>(split_channels); ++r)
            for (int c = 0; c < split_channels; ++c)
                fore_feat[r * static_cast<size_t>(split_channels) + c] *= mask_kernel[r];
    }
    // 4. combine foreground + background (remove_repeat)
    std::vector<float> combined_feat;
    std::vector<int32_t> combined_coords;
    if (!ext::focal_combine_out(fore_feat, fore_coord, back_feat, back_coord,
                                split_channels, 1, combined_feat, combined_coords, error))
        return false;
    if (sink && split_channels == 16) {
        const int64_t nci = static_cast<int64_t>(combined_coords.size());
        std::vector<float> cic(nci);
        for (int64_t i = 0; i < nci; ++i) cic[i] = static_cast<float>(combined_coords[i]);
        sink("focal_mainconv_in_coords", cic.data(), nci / 4, 4);
        sink("focal_mainconv_in", combined_feat.data(),
             static_cast<int64_t>(combined_coords.size()) / 4, split_channels);
    }
    // 5. main conv on the combined active set (always submanifold, stride 1)
    return run_sparse_conv(backend, conv3_params(true), combined_feat, split_channels,
                           combined_coords, spatial_shape, weights.conv,
                           out_features, out_map, out_coords, error,
                           /*apply_relu=*/true,
                           /*deterministic=*/hp.focal_deterministic);
}

// Project voxel centres (lidar xyz) onto the image plane and sample the
// semantic image features. Mirrors FocalSparseConv.construct_multimodal_features.
//   image_feat : [image_ch, image_height*image_width] row-major (c, y*W+x)
std::vector<float> sample_image_features(const std::vector<float> & voxel_xyz,
                                         const std::vector<float> & image_feat,
                                         int image_width, int image_height, int image_ch,
                                         const float * lidar_to_camera,
                                         const float * camera_to_image) {
    const int64_t n = static_cast<int64_t>(voxel_xyz.size()) / 3;
    std::vector<float> sampled(static_cast<size_t>(n) * image_ch, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        const float x = voxel_xyz[i * 3], y = voxel_xyz[i * 3 + 1], z = voxel_xyz[i * 3 + 2];
        const float cx = lidar_to_camera[0] * x + lidar_to_camera[1] * y +
                         lidar_to_camera[2] * z + lidar_to_camera[3];
        const float cy = lidar_to_camera[4] * x + lidar_to_camera[5] * y +
                         lidar_to_camera[6] * z + lidar_to_camera[7];
        const float cz = lidar_to_camera[8] * x + lidar_to_camera[9] * y +
                         lidar_to_camera[10] * z + lidar_to_camera[11];
        const float u = camera_to_image[0] * cx + camera_to_image[1] * cy +
                        camera_to_image[2] * cz + camera_to_image[3];
        const float v = camera_to_image[4] * cx + camera_to_image[5] * cy +
                        camera_to_image[6] * cz + camera_to_image[7];
        const float w = camera_to_image[8] * cx + camera_to_image[9] * cy +
                        camera_to_image[10] * cz + camera_to_image[11];
        if (w <= 0.0f) continue;
        const int px = static_cast<int>(u / w);
        const int py = static_cast<int>(v / w);
        if (px < 0 || px >= image_width || py < 0 || py >= image_height) continue;
        for (int c = 0; c < image_ch; ++c)
            sampled[i * image_ch + c] =
                image_feat[c * (image_height * image_width) + py * image_width + px];
    }
    return sampled;
}

// 4x4 -> 3x4 camera intrinsics top rows (P2).
void intrinsics_to_camera_to_image(const float * intrinsics, float * camera_to_image) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            camera_to_image[r * 4 + c] = intrinsics[r * 4 + c];
}

// 4x4 matrix multiply / inverse (same semantics as CaDDN's caddn.cpp).
void multiply_4x4(const float * left, const float * right, float * output) {
    float result[16];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += left[r * 4 + k] * right[k * 4 + c];
            result[r * 4 + c] = sum;
        }
    std::copy_n(result, 16, output);
}

bool invert_4x4(const float * source, float * inverse) {
    const float * m = source;
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det == 0.0f) return false;
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) inverse[i] = inv[i] * det;
    return true;
}

} // namespace

bool load_focal_backbone_weights(model_file & file, const hparams & hp,
                                 focal_backbone_weights & weights,
                                 std::string & error) {
    auto require = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * tensor = file.require(name);
        if (!tensor && file.error.empty()) file.error = "missing tensor: " + std::string(name);
        return tensor;
    };
    auto load_conv = [&](const std::string & name, sparse_layer_weights & out) -> bool {
        out.weight = require((name + ".weight").c_str());
        out.bias = require((name + ".bias").c_str());
        if (!file.error.empty()) { error = file.error; return false; }
        return true;
    };
    auto load_fconv = [&](const std::string & name, focal_conv_weights & out) -> bool {
        if (!load_conv(name + ".conv", out.conv)) return false;
        out.conv_imp.weight = require((name + ".conv_imp.weight").c_str());
        out.conv_bn_scale = file.tensor((name + ".conv_bn_scale").c_str());  // optional
        if (!file.error.empty()) { error = file.error; return false; }
        return true;
    };
    auto load_list = [&](const std::string & name, std::vector<sparse_layer_weights> & out,
                         int count) -> bool {
        out.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            if (!load_conv(name + "." + std::to_string(i), out[static_cast<size_t>(i)])) return false;
        return true;
    };
    if (!load_conv("sparse.conv_input", weights.conv_input) ||
        !load_conv("sparse.conv1.0", weights.conv1_0) ||
        !load_fconv("sparse.conv1.1", weights.conv1_1) ||
        !load_list("sparse.conv2", weights.conv2_pre, 3) ||
        !load_fconv("sparse.conv2.3", weights.conv2_3) ||
        !load_list("sparse.conv3", weights.conv3_pre, 3) ||
        !load_fconv("sparse.conv3.3", weights.conv3_3) ||
        !load_list("sparse.conv4", weights.conv4, 3) ||
        !load_conv("sparse.conv_out", weights.conv_out))
        return false;
    if (hp.focal_multimodal)
        if (!load_fconv("sparse.conv_focal_multimodal", weights.conv_focal_multimodal))
            return false;
    return true;
}

bool run_focal_backbone(engine_backend & backend, const hparams & hp,
                        const focal_backbone_weights & weights,
                        const float * points, int32_t point_count, int32_t point_dim,
                        const std::vector<float> * image_features,
                        int image_width, int image_height,
                        const std::vector<float> & lidar_to_image,
                        const std::vector<float> & camera_intrinsics,
                        focal_backbone_result & result,
                        const tap_sink & sink, std::string & error) {
    result = {};
    if (point_count < 0 || (point_count > 0 && !points) ||
        point_dim != hp.num_point_features) {
        error = "invalid focal sparse voxel point buffer or feature dimension";
        return false;
    }
    if (point_count == 0) return true;

    voxelization_params params;
    params.point_cloud_range = hp.point_cloud_range;
    params.voxel_size = hp.voxel_size;
    params.grid_size = hp.grid_size;
    params.max_points_per_voxel = hp.max_points_per_voxel;
    params.max_voxels = hp.max_voxels;
    voxel_output voxels;
    if (!voxelize(params, points, point_count, point_dim, voxels, error)) return false;
    const int active = voxels.actual_voxels;
    if (active == 0) return true;

    std::vector<float> mean_features(static_cast<size_t>(active) * point_dim, 0.0f);
    for (int row = 0; row < active; ++row) {
        const int count = voxels.num_points[row];
        for (int point = 0; point < count; ++point) {
            const float * source = voxels.voxels.data() +
                (static_cast<size_t>(row) * hp.max_points_per_voxel + point) * point_dim;
            for (int channel = 0; channel < point_dim; ++channel)
                mean_features[static_cast<size_t>(row) * point_dim + channel] += source[channel];
        }
        for (int channel = 0; channel < point_dim; ++channel)
            mean_features[static_cast<size_t>(row) * point_dim + channel] /= count;
    }
    std::vector<int32_t> coords(voxels.coords.begin(),
        voxels.coords.begin() + static_cast<size_t>(active) * 4);
    if (sink) sink("mean_vfe", mean_features.data(), active, point_dim);

    const std::vector<int32_t> in_shape = {hp.grid_size[2] + 1, hp.grid_size[1], hp.grid_size[0]};
    std::vector<float> out_feat; std::vector<int32_t> out_coords;
    ext::sparse_indice_map map;

    // conv_input -> conv1.0
    std::vector<float> x;
    if (!run_sparse_conv(backend, conv3_params(true), mean_features, point_dim, coords,
                         in_shape, weights.conv_input, x, map, out_coords, error))
        return false;
    coords = std::move(out_coords);
    if (sink) sink("focal_conv_input_out", x.data(), static_cast<int64_t>(x.size() / 16), 16);
    std::vector<float> x1;
    if (!run_sparse_conv(backend, conv3_params(true), x, 16, coords, in_shape,
                         weights.conv1_0, x1, map, out_coords, error))
        return false;
    coords = std::move(out_coords);
    if (sink) sink("focal_conv1_0_out", x1.data(), static_cast<int64_t>(x1.size() / 16), 16);

    // conv1.1 (focal) -> x_conv1
    std::vector<float> xconv1_feat; std::vector<int32_t> xconv1_coords;
    ext::sparse_indice_map xconv1_map;
    if (!run_focal_conv_core(backend, hp, weights.conv1_1, x1, 16, x1, 16,
                             coords, in_shape, xconv1_feat, xconv1_coords,
                             xconv1_map, sink, error))
        return false;
    if (sink) sink("focal_conv1_1_out", xconv1_feat.data(), static_cast<int64_t>(xconv1_feat.size() / 16), 16);

    // conv_focal_multimodal (multimodal fusion at conv1)
    std::vector<float> fused_feat = xconv1_feat;
    std::vector<int32_t> fused_coords = xconv1_coords;
    ext::sparse_indice_map fused_map = xconv1_map;
    if (hp.focal_multimodal && image_features) {
        if (image_width <= 0 || image_height <= 0 ||
            lidar_to_image.size() < 16 || camera_intrinsics.size() < 16) {
            error = "Focal multimodal requires image size and calibration matrices";
            return false;
        }
        // lidar_to_camera = inverse(intrinsics) * lidar_to_image (CaDDN semantics).
        float inv[16];
        invert_4x4(camera_intrinsics.data(), inv);
        float lidar_to_camera[16];
        multiply_4x4(inv, lidar_to_image.data(), lidar_to_camera);
        float camera_to_image[12];
        intrinsics_to_camera_to_image(camera_intrinsics.data(), camera_to_image);
        // voxel centres in lidar xyz: coords [b,z,y,x] -> xyz via voxel_size + range.
        const int64_t nvox = static_cast<int64_t>(fused_coords.size()) / 4;
        std::vector<float> voxel_xyz(static_cast<size_t>(nvox) * 3);
        for (int64_t r = 0; r < nvox; ++r) {
            const int32_t * c = fused_coords.data() + r * 4;
            voxel_xyz[r * 3]     = c[3] * hp.voxel_size[0] + hp.point_cloud_range[0];
            voxel_xyz[r * 3 + 1] = c[2] * hp.voxel_size[1] + hp.point_cloud_range[1];
            voxel_xyz[r * 3 + 2] = c[1] * hp.voxel_size[2] + hp.point_cloud_range[2];
        }
        const int image_ch = hp.focal_image_channel;
        const std::vector<float> sampled =
            sample_image_features(voxel_xyz, *image_features, image_width, image_height,
                                  image_ch, lidar_to_camera, camera_to_image);
        // x_predict = concat(image, voxel) for conv_imp (in_channels = img + voxel).
        std::vector<float> imp_input(static_cast<size_t>(nvox) * (image_ch + 16));
        for (int64_t r = 0; r < nvox; ++r) {
            std::copy_n(&sampled[r * image_ch], image_ch, &imp_input[r * (image_ch + 16)]);
            std::copy_n(&fused_feat[r * 16], 16, &imp_input[r * (image_ch + 16) + image_ch]);
        }
        if (sink) sink("focal_mm_in_voxel", fused_feat.data(), nvox, 16);
        if (sink) sink("focal_mm_in_sampled", sampled.data(), nvox, image_ch);
        // conv_focal_multimodal: conv_imp on fused input, split on LiDAR features,
        // main conv, then sum-fuse image features into the output.
        std::vector<float> mf_feat; std::vector<int32_t> mf_coords;
        ext::sparse_indice_map mf_map;
        if (!run_focal_conv_core(backend, hp, weights.conv_focal_multimodal,
                                 fused_feat, 16, imp_input, image_ch + 16,
                                 fused_coords, in_shape, mf_feat, mf_coords, mf_map,
                                 sink, error))
            return false;
        // Taps must read weight/bias through ggml_backend_tensor_get: on GPU
        // backends the tensors live in device memory and a raw ->data deref
        // segfaults the sink's fwrite.
        if (sink && weights.conv_focal_multimodal.conv.bias) {
            const int64_t nb = ggml_nelements(weights.conv_focal_multimodal.conv.bias);
            std::vector<float> bias_tap(static_cast<size_t>(nb));
            ggml_backend_tensor_get(weights.conv_focal_multimodal.conv.bias,
                                    bias_tap.data(), 0, bias_tap.size() * sizeof(float));
            sink("focal_mm_mainconv_bias", bias_tap.data(), nb, 1);
        }
        if (sink && weights.conv_focal_multimodal.conv.weight) {
            const int64_t nw = ggml_nelements(weights.conv_focal_multimodal.conv.weight);
            std::vector<float> weight_tap(static_cast<size_t>(nw));
            ggml_backend_tensor_get(weights.conv_focal_multimodal.conv.weight,
                                    weight_tap.data(), 0, weight_tap.size() * sizeof(float));
            sink("focal_mm_mainconv_weight", weight_tap.data(), nw, 1);
        }
        // sum-fuse image features sampled at the output voxels.
        const int64_t nout = static_cast<int64_t>(mf_coords.size()) / 4;
        if (sink) sink("focal_mm_conv_out", mf_feat.data(), nout, 16);
        std::vector<float> ovoxel(static_cast<size_t>(nout) * 3);
        for (int64_t r = 0; r < nout; ++r) {
            const int32_t * c = mf_coords.data() + r * 4;
            ovoxel[r * 3]     = c[3] * hp.voxel_size[0] + hp.point_cloud_range[0];
            ovoxel[r * 3 + 1] = c[2] * hp.voxel_size[1] + hp.point_cloud_range[1];
            ovoxel[r * 3 + 2] = c[1] * hp.voxel_size[2] + hp.point_cloud_range[2];
        }
        const std::vector<float> osampled =
            sample_image_features(ovoxel, *image_features, image_width, image_height,
                                  image_ch, lidar_to_camera, camera_to_image);
        if (sink) sink("focal_mm_sumfuse_sampled", osampled.data(), nout, image_ch);
        // Py applies BN1 AFTER the image sum-fuse (bn1(conv(x)+image)). The BN
        // gain is already folded into the main conv, so pre-scale the sampled
        // image by that gain before summing to reproduce Py exactly.
        std::vector<float> mm_bn_scale;
        if (weights.conv_focal_multimodal.conv_bn_scale) {
            const int64_t nsc = ggml_nelements(weights.conv_focal_multimodal.conv_bn_scale);
            mm_bn_scale.resize(static_cast<size_t>(nsc));
            ggml_backend_tensor_get(weights.conv_focal_multimodal.conv_bn_scale,
                                    mm_bn_scale.data(), 0, nsc * sizeof(float));
        }
        for (int64_t r = 0; r < nout; ++r)
            for (int c = 0; c < image_ch; ++c)
                mf_feat[r * 16 + c] +=
                    (mm_bn_scale.empty() ? 1.0f : mm_bn_scale[static_cast<size_t>(c)]) *
                    osampled[r * image_ch + c];
        fused_feat = std::move(mf_feat);
        fused_coords = std::move(mf_coords);
        fused_map = std::move(mf_map);
        if (sink) {
            const int64_t nc = static_cast<int64_t>(fused_coords.size());
            std::vector<float> cpf(nc);
            for (int64_t i = 0; i < nc; ++i) cpf[i] = static_cast<float>(fused_coords[i]);
            sink("focal_mm_out_coords", cpf.data(), nc / 4, 4);
        }
    }
    if (sink) sink("focal_conv_mm_out", fused_feat.data(), static_cast<int64_t>(fused_feat.size() / 16), 16);

    // conv2 -> x_conv2 (spconv stride2 + 2 subm + focal conv2.3)
    std::vector<int32_t> cur_shape = in_shape;
    std::vector<float> cur = fused_feat; std::vector<int32_t> curc = fused_coords;
    for (int i = 0; i < 3; ++i) {
        std::vector<float> next; std::vector<int32_t> nextc;
        const int stride = (i == 0) ? 2 : 1;
        if (!run_sparse_conv(backend, conv3_params(i != 0, stride),
                             cur, i == 0 ? 16 : 32, curc, cur_shape,
                             weights.conv2_pre[static_cast<size_t>(i)],
                             next, map, nextc, error))
            return false;
        cur = std::move(next); curc = std::move(nextc);
        cur_shape = map.spatial_shape;
        if (sink) sink(("focal_conv2_" + std::to_string(i) + "_out").c_str(),
                       cur.data(), static_cast<int64_t>(cur.size() / (i == 0 ? 16 : 32)), (i == 0 ? 16 : 32));
    }
    std::vector<float> x2; std::vector<int32_t> c2; ext::sparse_indice_map m2;
    if (!run_focal_conv_core(backend, hp, weights.conv2_3, cur, 32, cur, 32,
                             curc, cur_shape, x2, c2, m2, sink, error))
        return false;
    if (sink) sink("focal_conv2_3_out", x2.data(), static_cast<int64_t>(x2.size() / 32), 32);

    // conv3 -> x_conv3 (spconv stride2 + 2 subm + focal conv3.3)
    cur = x2; curc = c2;
    for (int i = 0; i < 3; ++i) {
        std::vector<float> next; std::vector<int32_t> nextc;
        const int stride = (i == 0) ? 2 : 1;
        if (!run_sparse_conv(backend, conv3_params(i != 0, stride),
                             cur, i == 0 ? 32 : 64, curc, cur_shape,
                             weights.conv3_pre[static_cast<size_t>(i)],
                             next, map, nextc, error))
            return false;
        cur = std::move(next); curc = std::move(nextc);
        cur_shape = map.spatial_shape;
    }
    std::vector<float> x3; std::vector<int32_t> c3; ext::sparse_indice_map m3;
    if (!run_focal_conv_core(backend, hp, weights.conv3_3, cur, 64, cur, 64,
                             curc, cur_shape, x3, c3, m3, sink, error))
        return false;

    // conv4 -> x_conv4 (spconv stride2 padding {0,1,1} + 2 subm)
    cur = x3; curc = c3;
    for (int i = 0; i < 3; ++i) {
        std::vector<float> next; std::vector<int32_t> nextc;
        const int stride = (i == 0) ? 2 : 1;
        const std::vector<int32_t> pad = (i == 0) ? std::vector<int32_t>{0, 1, 1}
                                                  : std::vector<int32_t>{1, 1, 1};
        if (!run_sparse_conv(backend, conv3_params(i != 0, stride, pad),
                             cur, 64, curc, cur_shape,
                             weights.conv4[static_cast<size_t>(i)],
                             next, map, nextc, error))
            return false;
        cur = std::move(next); curc = std::move(nextc);
        cur_shape = map.spatial_shape;
    }
    std::vector<float> x4 = cur; std::vector<int32_t> c4 = curc; ext::sparse_indice_map m4 = map;

    // conv_out (3x1x1 stride 2x1x1) -> dense-head spatial features
    std::vector<float> out8; std::vector<int32_t> c8; ext::sparse_indice_map m8;
    ext::sparse_conv_params out_params;
    out_params.kernel_size = {3, 1, 1};
    out_params.stride = {2, 1, 1};
    out_params.padding = {0, 0, 0};
    out_params.dilation = {1, 1, 1};
    out_params.submanifold = false;
    if (!run_sparse_conv(backend, out_params, x4, 64, c4, cur_shape,
                         weights.conv_out, out8, m8, c8, error))
        return false;

    // multi-scale RoI sources: x_conv2 / x_conv3 / x_conv4
    result.maps = {m2, m3, m4};
    result.layer_features = {x2, x3, x4};

    // HeightCompression for the dense head (conv_out channel=128).
    const int depth = m8.spatial_shape[0];
    result.height = m8.spatial_shape[1];
    result.width = m8.spatial_shape[2];
    const int sparse_channels = 128;
    if (sparse_channels * depth != hp.num_bev_features) {
        error = "Focal HeightCompression channel mismatch"; return false;
    }
    result.spatial_features.assign(static_cast<size_t>(result.height) * result.width *
                                   hp.num_bev_features, 0.0f);
    for (int64_t row = 0; row < m8.output_active_count(); ++row) {
        const int32_t * coordinate = m8.coordinates.data() + row * 4;
        float * target = result.spatial_features.data() +
            (static_cast<size_t>(coordinate[2]) * result.width + coordinate[3]) * hp.num_bev_features;
        const float * source = out8.data() + row * sparse_channels;
        for (int channel = 0; channel < sparse_channels; ++channel)
            target[channel * depth + coordinate[1]] = source[channel];
    }
    if (sink) sink("spatial_features", result.spatial_features.data(),
                   static_cast<int64_t>(result.height) * result.width, hp.num_bev_features);
    return true;
}

bool voxelrcnn::load(const std::string & path, const std::string & device, int threads) {
    if (!model::load(path, device, threads)) return false;
    const hparams & hp = file.hp;
    if (hp.arch != arch_kind::voxelrcnn || hp.head_kind != dense_head_kind::anchor_single ||
        hp.roi_voxel_source_counts.size() != 3 || hp.roi_voxel_radii.size() != 3) {
        error = "unsupported VoxelRCNN topology"; return false;
    }
    std::string load_error;
    // The multimodal focal model stores the sparse 3-D backbone in the focal
    // layout (conv1.1/conv2.3/conv3.3 are FocalSparseConv with a conv_imp) and
    // does not carry the plain VoxelBackBone8x weights, so the focal backbone is
    // loaded instead of the plain sparse chain.
    const bool focal = hp.focal_multimodal;
    if (focal) {
        if (!load_focal_backbone_weights(file, hp, focal_w, load_error)) {
            error = load_error; return false;
        }
    } else if (!load_sparse_backbone_weights(file, hp, sparse_w, load_error)) {
        error = load_error; return false;
    }
    if (!load_bev_backbone_weights(file, hp, backbone_w, load_error) ||
        !load_anchor_head_weights(file, hp, head_w, load_error)) {
        error = load_error; return false;
    }
    // Semantic-image branch — present only on the multimodal focal variant.
    // Detect it by the presence of the ResNet50 conv1 weight.
    if (resolved("semseg.model.backbone.conv1.weight")) {
        semseg.enabled = true;
        semseg.image_stem = load_semseg_block(*this, "semseg.model.backbone.conv1.weight",
                                              "semseg.model.backbone.bn1");
        const int layer1_blocks = 3;  // ResNet50 layer1
        for (int block = 0; block < layer1_blocks; ++block) {
            const std::string prefix = "semseg.model.backbone.layer1." + std::to_string(block);
            auto & target = semseg.image_layers.emplace_back();
            target.conv1 = load_semseg_block(*this, prefix + ".conv1.weight", prefix + ".bn1");
            target.conv2 = load_semseg_block(*this, prefix + ".conv2.weight", prefix + ".bn2");
            target.conv3 = load_semseg_block(*this, prefix + ".conv3.weight", prefix + ".bn3");
            target.has_downsample = block == 0;
            if (target.has_downsample)
                target.downsample = load_semseg_block(*this, prefix + ".downsample.0.weight",
                                                      prefix + ".downsample.1");
        }
        semseg.reduce = load_semseg_block(*this, "semseg.reduce_blocks.0.conv.weight",
                                          "semseg.reduce_blocks.0.bn");
        if (!error.empty()) return false;
    }
    roi_pool.resize(3);
    for (int source = 0; source < 3; ++source) {
        const std::string base = "roi_head.roi_grid_pool_layers." + std::to_string(source);
        const std::vector<point_mlp_layer_spec> in = {
            {base + ".mlps_in.0.0.weight", "", base + ".mlps_in.0.1", false},
        };
        const std::vector<point_mlp_layer_spec> position = {
            {base + ".mlps_pos.0.0.weight", "", base + ".mlps_pos.0.1", false},
        };
        const std::vector<point_mlp_layer_spec> out = {
            {base + ".mlps_out.0.0.weight", "", base + ".mlps_out.0.1", true},
        };
        if (!load_point_mlp(file, in, roi_pool[source].feature_in, load_error) ||
            !load_point_mlp(file, position, roi_pool[source].position, load_error) ||
            !load_point_mlp(file, out, roi_pool[source].output, load_error)) {
            error = load_error; return false;
        }
    }
    const std::vector<point_mlp_layer_spec> cls_output = {
        {"roi_head.cls_pred_layer.weight", "roi_head.cls_pred_layer.bias", "", false},
    };
    const std::vector<point_mlp_layer_spec> reg_output = {
        {"roi_head.reg_pred_layer.weight", "roi_head.reg_pred_layer.bias", "", false},
    };
    if (!load_point_mlp(file, hidden_specs("roi_head.shared_fc_layer"), roi_shared, load_error) ||
        !load_point_mlp(file, hidden_specs("roi_head.cls_fc_layers"), roi_cls_hidden, load_error) ||
        !load_point_mlp(file, cls_output, roi_cls_output, load_error) ||
        !load_point_mlp(file, hidden_specs("roi_head.reg_fc_layers"), roi_reg_hidden, load_error) ||
        !load_point_mlp(file, reg_output, roi_reg_output, load_error)) {
        error = load_error; return false;
    }
    if (device_ctx) {
        auto remap = [&](ggml_tensor *& tensor) {
            if (tensor) tensor = ggml_get_tensor(device_ctx, ggml_get_name(tensor));
        };
        for (auto & layer : sparse_w.layers) { remap(layer.weight); remap(layer.bias); }
        if (focal) {
            auto remap_sparse = [&](sparse_layer_weights & layer) {
                remap(layer.weight); remap(layer.bias);
            };
            auto remap_focal = [&](focal_conv_weights & layer) {
                remap_sparse(layer.conv); remap_sparse(layer.conv_imp);
                remap(layer.conv_bn_scale);
            };
            remap_sparse(focal_w.conv_input);
            remap_sparse(focal_w.conv1_0);
            remap_focal(focal_w.conv1_1);
            for (auto & layer : focal_w.conv2_pre) remap_sparse(layer);
            remap_focal(focal_w.conv2_3);
            for (auto & layer : focal_w.conv3_pre) remap_sparse(layer);
            remap_focal(focal_w.conv3_3);
            for (auto & layer : focal_w.conv4) remap_sparse(layer);
            remap_sparse(focal_w.conv_out);
            if (hp.focal_multimodal) remap_focal(focal_w.conv_focal_multimodal);
        }
        for (auto & level : backbone_w.levels) for (auto & block : level) {
            remap(block.conv_w); remap(block.bn_scale); remap(block.bn_bias);
        }
        for (auto & block : backbone_w.deblocks) {
            remap(block.conv_w); remap(block.bn_scale); remap(block.bn_bias);
        }
        remap(head_w.conv_cls_w); remap(head_w.conv_cls_b);
        remap(head_w.conv_box_w); remap(head_w.conv_box_b);
        remap(head_w.conv_dir_cls_w); remap(head_w.conv_dir_cls_b);
        for (auto & pool : roi_pool) {
            remap_point_mlp(pool.feature_in, device_ctx);
            remap_point_mlp(pool.position, device_ctx);
            remap_point_mlp(pool.output, device_ctx);
        }
        remap_point_mlp(roi_shared, device_ctx); remap_point_mlp(roi_cls_hidden, device_ctx);
        remap_point_mlp(roi_cls_output, device_ctx); remap_point_mlp(roi_reg_hidden, device_ctx);
        remap_point_mlp(roi_reg_output, device_ctx);
        if (semseg.enabled) {
            auto remap_block = [&](conv_bn_block_weights & block) {
                remap(block.conv_w); remap(block.bn_scale); remap(block.bn_bias);
            };
            remap_block(semseg.image_stem);
            for (auto & block : semseg.image_layers) {
                remap_block(block.conv1); remap_block(block.conv2); remap_block(block.conv3);
                if (block.has_downsample) remap_block(block.downsample);
            }
            remap_block(semseg.reduce);
        }
    }
    return true;
}

bool voxelrcnn::forward(const float * points, int32_t count, int32_t dimension,
                        std::vector<detection> & detections, const tap_sink & sink) {
    detections.clear();
    const hparams & hp = file.hp;
    // AICore port note: the multimodal focal variant stores its backbone in the
    // focal layout (sparse_w is empty), so the points-only entry dispatches on
    // hparams to a LiDAR-only focal chain. The full multimodal path (semseg
    // image branch + fusion) stays in forward_scene, as upstream.
    if (hp.focal_multimodal) {
        focal_backbone_result focal;
        const std::vector<float> no_image;
        const std::vector<float> no_calib;
        if (!run_focal_backbone(be, hp, focal_w, points, count, dimension,
                                nullptr, 0, 0, no_calib, no_calib,
                                focal, sink, error)) return false;
        if (focal.spatial_features.empty()) return true;
        voxel_backbone_result voxel;
        voxel.maps = std::move(focal.maps);
        voxel.layer_features = std::move(focal.layer_features);
        voxel.spatial_features = std::move(focal.spatial_features);
        voxel.width = focal.width;
        voxel.height = focal.height;
        return run_roi_heads(voxel, true, detections, sink);
    }
    voxel_backbone_result voxel;
    if (!run_voxel_backbone(be, hp, sparse_w, points, count, dimension,
                            true, voxel, sink, error)) return false;
    // The plain VoxelBackBone8x produces a full layer list; the RoI sources are
    // selected by name inside run_roi_heads (sources_prepared == false).
    return run_roi_heads(voxel, false, detections, sink);
}

bool voxelrcnn::run_roi_heads(const voxel_backbone_result & voxel, bool sources_prepared,
                              std::vector<detection> & detections, const tap_sink & sink) {
    const hparams & hp = file.hp;
    if (voxel.spatial_features.empty()) return true;
    const std::vector<std::string> stage_names = {"conv2.2", "conv3.2", "conv4.2"};
    const std::vector<int> strides = {2, 4, 8};
    auto source_index = [&](int source) -> int {
        return sources_prepared ? source : layer_index(sparse_w, stage_names[source]);
    };
    auto source_channels = [&](int index) -> int {
        if (sources_prepared) return static_cast<int>(
            voxel.layer_features[index].size() / voxel.maps[index].output_active_count());
        return sparse_w.specs[index].out_channels;
    };
    if (sink) {
        for (int source = 0; source < 3; ++source) {
            const int index = source_index(source);
            if (index < 0) continue;
            std::vector<float> coordinates(voxel.maps[index].coordinates.begin(),
                                           voxel.maps[index].coordinates.end());
            sink("sparse." + stage_names[source] + ".coords", coordinates.data(),
                 voxel.maps[index].output_active_count(), 4);
        }
    }
    hparams proposal_hp = hp;
    proposal_hp.score_thresh = 0.0f;
    proposal_hp.nms_thresh = hp.roi_proposal_nms_thresh;
    proposal_hp.nms_pre_maxsize = hp.roi_proposal_pre_max;
    proposal_hp.max_boxes = hp.roi_proposal_post_max;
    std::vector<detection> proposals;
    if (!run_dense_anchor_pipeline(be, proposal_hp, backbone_w, head_w,
            voxel.spatial_features, voxel.width, voxel.height,
            proposals, sink, error)) return false;
    if (proposals.empty()) return true;

    const int grids = hp.roi_grid_size * hp.roi_grid_size * hp.roi_grid_size;
    std::vector<float> grid_xyz = roi_grid_points(proposals, hp.roi_grid_size);
    if (sink) {
        sink("roi.grid_xyz", grid_xyz.data(),
             static_cast<int64_t>(proposals.size()) * grids, 3);
        std::vector<float> grid_coords(static_cast<size_t>(proposals.size()) * grids * 4);
        for (int64_t row = 0; row < static_cast<int64_t>(proposals.size()) * grids; ++row) {
            grid_coords[row * 4] = 0.0f;
            grid_coords[row * 4 + 1] = static_cast<float>(
                static_cast<int>(std::floor((grid_xyz[row * 3 + 2] - hp.point_cloud_range[2]) /
                                            hp.voxel_size[2])) / 2);
            grid_coords[row * 4 + 2] = static_cast<float>(
                static_cast<int>(std::floor((grid_xyz[row * 3 + 1] - hp.point_cloud_range[1]) /
                                            hp.voxel_size[1])) / 2);
            grid_coords[row * 4 + 3] = static_cast<float>(
                static_cast<int>(std::floor((grid_xyz[row * 3] - hp.point_cloud_range[0]) /
                                            hp.voxel_size[0])) / 2);
        }
        sink("roi.new_coords", grid_coords.data(),
             static_cast<int64_t>(proposals.size()) * grids, 4);
    }
    std::vector<float> multi_scale;
    for (int source = 0; source < 3; ++source) {
        const int index = source_index(source);
        const int channels = source_channels(index);
        if (index < 0) { error = "VoxelRCNN sparse feature stage missing"; return false; }
        std::vector<float> pooled;
        std::vector<float> neighbor_xyz;
        if (!voxel_pool_source(be, hp, voxel.maps[index], voxel.layer_features[index],
                channels, strides[source], grid_xyz,
                hp.roi_voxel_radii[source], hp.roi_voxel_nsamples[source],
                hp.roi_voxel_query_ranges.data() + source * 3,
                roi_pool[source], pooled, sink ? &neighbor_xyz : nullptr, error)) return false;
        if (sink) sink("roi.neighbor_xyz." + std::to_string(source), neighbor_xyz.data(),
                       static_cast<int64_t>(proposals.size()) * grids * hp.roi_voxel_nsamples[source], 3);
        if (sink) sink("roi.pool." + std::to_string(source), pooled.data(),
                       static_cast<int64_t>(proposals.size()) * grids,
                       pooled.size() / (proposals.size() * grids));
        const int64_t rows = static_cast<int64_t>(proposals.size()) * grids;
        const int old_channels = multi_scale.empty() ? 0 : static_cast<int>(multi_scale.size() / rows);
        const int new_channels = static_cast<int>(pooled.size() / rows);
        std::vector<float> joined(static_cast<size_t>(rows) * (old_channels + new_channels));
        for (int64_t row = 0; row < rows; ++row) {
            if (old_channels) std::copy_n(multi_scale.data() + row * old_channels, old_channels,
                joined.data() + row * (old_channels + new_channels));
            std::copy_n(pooled.data() + row * new_channels, new_channels,
                joined.data() + row * (old_channels + new_channels) + old_channels);
        }
        multi_scale = std::move(joined);
    }
    std::vector<float> shared, cls_hidden, reg_hidden, refined_cls, residuals;
    if (!run_point_mlp(be, roi_shared, multi_scale, proposals.size(), shared, error) ||
        !run_point_mlp(be, roi_cls_hidden, shared, proposals.size(), cls_hidden, error) ||
        !run_point_mlp(be, roi_cls_output, cls_hidden, proposals.size(), refined_cls, error) ||
        !run_point_mlp(be, roi_reg_hidden, shared, proposals.size(), reg_hidden, error) ||
        !run_point_mlp(be, roi_reg_output, reg_hidden, proposals.size(), residuals, error)) return false;
    if (sink) {
        sink("roi.pool.concat", multi_scale.data(),
             static_cast<int64_t>(proposals.size()) * grids,
             multi_scale.size() / (proposals.size() * grids));
        sink("roi.shared", shared.data(), proposals.size(), shared.size() / proposals.size());
        sink("roi.cls", refined_cls.data(), proposals.size(), 1);
        sink("roi.reg", residuals.data(), proposals.size(), 7);
    }
    std::vector<float> rois(proposals.size() * 7);
    for (size_t roi = 0; roi < proposals.size(); ++roi) {
        const auto & box = proposals[roi];
        const float values[7] = {box.x,box.y,box.z,box.dx,box.dy,box.dz,box.heading};
        std::copy_n(values, 7, rois.data() + roi * 7);
    }
    std::vector<float> refined;
    if (!ext::decode_roi_residuals(rois.data(), residuals.data(), proposals.size(),
                                   refined, error)) return false;
    std::vector<int32_t> candidates;
    std::vector<float> scores(proposals.size());
    for (size_t roi = 0; roi < proposals.size(); ++roi) {
        scores[roi] = 1.0f / (1.0f + std::exp(-refined_cls[roi]));
        if (scores[roi] >= hp.score_thresh) candidates.push_back(static_cast<int32_t>(roi));
    }
    if (static_cast<int>(candidates.size()) > hp.nms_pre_maxsize) {
        std::partial_sort(candidates.begin(), candidates.begin() + hp.nms_pre_maxsize,
            candidates.end(), [&](int a, int b) { return scores[a] > scores[b]; });
        candidates.resize(hp.nms_pre_maxsize);
    }
    std::vector<float> boxes(candidates.size() * 7), candidate_scores(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        std::copy_n(refined.data() + candidates[index] * 7, 7, boxes.data() + index * 7);
        candidate_scores[index] = scores[candidates[index]];
    }
    const std::vector<int32_t> kept = nms_3d(boxes, candidate_scores, hp.nms_thresh, hp.max_boxes);
    for (int local : kept) {
        const int roi = candidates[local];
        const float * box = refined.data() + roi * 7;
        detections.push_back({box[0],box[1],box[2],box[3],box[4],box[5],box[6],
                              scores[roi],proposals[roi].label});
    }
    return true;
}

bool voxelrcnn::forward_scene(const scene_input & scene, std::vector<detection> & detections,
                              const tap_sink & sink) {
    detections.clear();
    // Plain (LiDAR-only) VoxelRCNN ignores the image entirely.
    if (!semseg.enabled)
        return forward(scene.points, scene.n_points, scene.point_dim, detections, sink);
    const hparams & hp = file.hp;
    if (!scene.images || scene.num_cameras != 1 || scene.image_channels != 3 ||
        scene.image_height <= 0 || scene.image_width <= 0 ||
        !scene.lidar_to_image || !scene.camera_intrinsics) {
        error = "Focal VoxelRCNN (multimodal) requires one RGB image and 4x4 calibration matrices";
        return false;
    }
    const int input_width = scene.image_width, input_height = scene.image_height;
    std::vector<float> input(static_cast<size_t>(input_width) * input_height * 3);
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float deviation[3] = {0.229f, 0.224f, 0.225f};
    for (int channel = 0; channel < 3; ++channel) for (int y = 0; y < input_height; ++y)
        for (int x = 0; x < input_width; ++x) {
            const float value = scene.images[(static_cast<size_t>(y) * input_width + x) * 3 + channel];
            input[static_cast<size_t>(x) + static_cast<size_t>(input_width) *
                  (y + input_height * channel)] =
                value == 0.0f ? 0.0f : (value - mean[channel]) / deviation[channel];
        }

    constexpr size_t graph_nodes = 8192;
    ggml_init_params params{ggml_tensor_overhead() * graph_nodes +
        ggml_graph_overhead_custom(graph_nodes, false), nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) { error = "Focal semseg graph context init failed"; return false; }
    ggml_tensor * image = ggml_new_tensor_4d(context, GGML_TYPE_F32,
                                              input_width, input_height, 3, 1);
    ggml_set_input(image);
    // ResNet50 stem: conv1 (7x7 s2 p3) + BN + ReLU + maxpool.
    ggml_tensor * value = ggml_relu(context, conv_bn(*this, context, semseg.image_stem, image, 2, 3));
    value = ggml_pool_2d(context, value, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1);
    // ResNet50 layer1 (3 Bottleneck blocks, stride=1, dilation=1).
    for (const auto & block : semseg.image_layers)
        value = bottleneck(*this, context, block, value, 1, 1);
    // Channel reduce: 1x1 conv 256 -> 16 + BN + ReLU.
    ggml_tensor * image_features = ggml_relu(context, conv_bn(*this, context, semseg.reduce, value, 1, 0));
    // PyTorch FocalSparseConv.construct_multimodal_features bilinearly upsamples the
    // semseg features to the full input image size and samples with raw pixel coords.
    if (image_features->ne[0] != input_width || image_features->ne[1] != input_height) {
        image_features = ggml_interpolate(context, image_features, input_width, input_height,
                                          image_features->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
    }
    ggml_set_output(image_features);
    ggml_cgraph * graph = ggml_new_graph_custom(context, graph_nodes, false);
    ggml_build_forward_expand(graph, image_features);
    if (!ggml_gallocr_alloc_graph(be.galloc, graph)) {
        error = "Focal semseg graph allocation failed"; ggml_free(context); return false;
    }
    ggml_backend_tensor_set(image, input.data(), 0, input.size() * sizeof(float));
    if (ggml_backend_graph_compute(be.be, graph) != GGML_STATUS_SUCCESS) {
        error = "Focal semseg graph compute failed"; ggml_free(context); return false;
    }
    const int feature_width = image_features->ne[0], feature_height = image_features->ne[1];
    std::vector<float> feature_values(static_cast<size_t>(ggml_nelements(image_features)));
    ggml_backend_tensor_get(image_features, feature_values.data(),
                            0, feature_values.size() * sizeof(float));
    if (sink) sink("voxelrcnn.image_features", feature_values.data(),
                   static_cast<int64_t>(feature_width) * feature_height, 16);
    ggml_free(context);
    // Run the Focal VoxelBackBone8xFocal: sparse focal chain + multimodal fusion
    // of the 16-ch image features onto the BEV voxels via camera calibration.
    std::vector<float> lidar_to_image(scene.lidar_to_image,
                                      scene.lidar_to_image + 16);
    std::vector<float> camera_intrinsics(scene.camera_intrinsics,
                                         scene.camera_intrinsics + 16);
    focal_backbone_result focal;
    if (!run_focal_backbone(be, hp, focal_w, scene.points, scene.n_points,
                            scene.point_dim, &feature_values,
                            feature_width, feature_height,
                            lidar_to_image, camera_intrinsics,
                            focal, sink, error))
        return false;
    if (focal.spatial_features.empty()) return true;
    // Convert to the shared voxel_backbone_result consumed by run_roi_heads.
    voxel_backbone_result voxel;
    voxel.maps = std::move(focal.maps);
    voxel.layer_features = std::move(focal.layer_features);
    voxel.spatial_features = std::move(focal.spatial_features);
    voxel.width = focal.width;
    voxel.height = focal.height;
    return run_roi_heads(voxel, true, detections, sink);
}

} // namespace openpcdet
