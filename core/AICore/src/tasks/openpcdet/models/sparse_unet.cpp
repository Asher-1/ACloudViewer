#include "sparse_unet.h"

#include "../ops/voxelization.h"

#include <ggml.h>

#include <algorithm>
#include <array>

namespace openpcdet {

namespace {

struct layer_source { const char * name; const char * conv; const char * bn; };

std::vector<layer_source> sources(bool detection_output) {
    std::vector<layer_source> result = {
        {"conv_input", "backbone_3d.conv_input.0", "backbone_3d.conv_input.1"},
        {"conv1", "backbone_3d.conv1.0.0", "backbone_3d.conv1.0.1"},
        {"conv2.0", "backbone_3d.conv2.0.0", "backbone_3d.conv2.0.1"},
        {"conv2.1", "backbone_3d.conv2.1.0", "backbone_3d.conv2.1.1"},
        {"conv2.2", "backbone_3d.conv2.2.0", "backbone_3d.conv2.2.1"},
        {"conv3.0", "backbone_3d.conv3.0.0", "backbone_3d.conv3.0.1"},
        {"conv3.1", "backbone_3d.conv3.1.0", "backbone_3d.conv3.1.1"},
        {"conv3.2", "backbone_3d.conv3.2.0", "backbone_3d.conv3.2.1"},
        {"conv4.0", "backbone_3d.conv4.0.0", "backbone_3d.conv4.0.1"},
        {"conv4.1", "backbone_3d.conv4.1.0", "backbone_3d.conv4.1.1"},
        {"conv4.2", "backbone_3d.conv4.2.0", "backbone_3d.conv4.2.1"},
        {"up_t4.1", "backbone_3d.conv_up_t4.conv1", "backbone_3d.conv_up_t4.bn1"},
        {"up_t4.2", "backbone_3d.conv_up_t4.conv2", "backbone_3d.conv_up_t4.bn2"},
        {"up_m4", "backbone_3d.conv_up_m4.0", "backbone_3d.conv_up_m4.1"},
        {"inv4", "backbone_3d.inv_conv4.0", "backbone_3d.inv_conv4.1"},
        {"up_t3.1", "backbone_3d.conv_up_t3.conv1", "backbone_3d.conv_up_t3.bn1"},
        {"up_t3.2", "backbone_3d.conv_up_t3.conv2", "backbone_3d.conv_up_t3.bn2"},
        {"up_m3", "backbone_3d.conv_up_m3.0", "backbone_3d.conv_up_m3.1"},
        {"inv3", "backbone_3d.inv_conv3.0", "backbone_3d.inv_conv3.1"},
        {"up_t2.1", "backbone_3d.conv_up_t2.conv1", "backbone_3d.conv_up_t2.bn1"},
        {"up_t2.2", "backbone_3d.conv_up_t2.conv2", "backbone_3d.conv_up_t2.bn2"},
        {"up_m2", "backbone_3d.conv_up_m2.0", "backbone_3d.conv_up_m2.1"},
        {"inv2", "backbone_3d.inv_conv2.0", "backbone_3d.inv_conv2.1"},
        {"up_t1.1", "backbone_3d.conv_up_t1.conv1", "backbone_3d.conv_up_t1.bn1"},
        {"up_t1.2", "backbone_3d.conv_up_t1.conv2", "backbone_3d.conv_up_t1.bn2"},
        {"up_m1", "backbone_3d.conv_up_m1.0", "backbone_3d.conv_up_m1.1"},
        {"conv5", "backbone_3d.conv5.0.0", "backbone_3d.conv5.0.1"},
    };
    if (detection_output)
        result.push_back({"conv_out", "backbone_3d.conv_out.0", "backbone_3d.conv_out.1"});
    return result;
}

ext::sparse_conv_params subm3() {
    return {{3,3,3}, {1,1,1}, {1,1,1}, {1,1,1}, true, false};
}

ext::sparse_conv_params down3(std::array<int32_t,3> padding) {
    return {{3,3,3}, {2,2,2}, {padding[0],padding[1],padding[2]},
            {1,1,1}, false, false};
}

std::vector<float> readback(ggml_tensor * tensor) {
    std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    return result;
}

} // namespace

bool load_sparse_unet_weights(model_file & file, bool detection_output,
                              sparse_unet_weights & weights,
                              std::string & error) {
    weights = {};
    weights.has_detection_output = detection_output;
    for (const auto & source : sources(detection_output)) {
        sparse_unet_layer layer;
        layer.weight = file.require((std::string(source.conv) + ".weight").c_str());
        layer.scale = file.require((std::string(source.bn) + ".eval_scale").c_str());
        layer.bias = file.require((std::string(source.bn) + ".eval_bias").c_str());
        if (!file.error.empty()) { error = file.error; weights = {}; return false; }
        weights.layers.emplace(source.name, layer);
    }
    return true;
}

void remap_sparse_unet_weights(sparse_unet_weights & weights,
                               ggml_context * device_context) {
    if (!device_context) return;
    for (auto & entry : weights.layers) {
        auto remap = [&](ggml_tensor *& tensor) {
            tensor = ggml_get_tensor(device_context, ggml_get_name(tensor));
        };
        remap(entry.second.weight); remap(entry.second.scale); remap(entry.second.bias);
    }
}

bool run_sparse_unet(engine_backend & backend, const hparams & hp,
                     const sparse_unet_weights & weights,
                     const float * points, int32_t point_count, int32_t point_dim,
                     sparse_unet_result & result, const tap_sink & sink,
                     std::string & error) {
    result = {};
    if (!points || point_count <= 0 || point_dim != hp.num_point_features ||
        hp.grid_size.size() != 3 || hp.voxel_size.size() != 3) {
        error = "invalid sparse U-Net input"; return false;
    }
    voxelization_params voxel_params;
    voxel_params.point_cloud_range = hp.point_cloud_range;
    voxel_params.voxel_size = hp.voxel_size;
    voxel_params.grid_size = hp.grid_size;
    voxel_params.max_points_per_voxel = hp.max_points_per_voxel;
    voxel_params.max_voxels = hp.max_voxels;
    voxel_output voxels;
    if (!voxelize(voxel_params, points, point_count, point_dim, voxels, error)) return false;
    const int active = voxels.actual_voxels;
    if (active == 0) return true;
    std::vector<float> input_features(static_cast<size_t>(active) * point_dim, 0.0f);
    for (int row = 0; row < active; ++row) {
        for (int point = 0; point < voxels.num_points[row]; ++point) {
            const float * source = voxels.voxels.data() +
                (static_cast<size_t>(row) * hp.max_points_per_voxel + point) * point_dim;
            for (int channel = 0; channel < point_dim; ++channel)
                input_features[row * point_dim + channel] += source[channel];
        }
        for (int channel = 0; channel < point_dim; ++channel)
            input_features[row * point_dim + channel] /= voxels.num_points[row];
    }
    std::vector<int32_t> coordinates(voxels.coords.begin(),
        voxels.coords.begin() + static_cast<size_t>(active) * 4);
    ext::sparse_tensor current;
    current.ndim = 3;
    current.spatial_shape = {hp.grid_size[2] + 1, hp.grid_size[1], hp.grid_size[0]};
    current.coordinates = coordinates;
    current.channels = point_dim;
    std::unordered_map<std::string, ext::sparse_indice_map> maps;
    auto build = [&](const std::string & name, const ext::sparse_conv_params & params,
                     int channels) {
        ext::sparse_indice_map map;
        if (!ext::build_sparse_indice_map(current, params, map, error)) return false;
        current.spatial_shape = map.spatial_shape;
        current.coordinates = map.coordinates;
        current.channels = channels;
        maps.emplace(name, std::move(map));
        return true;
    };
    if (!build("subm1", subm3(), 16) || !build("sp2", down3({1,1,1}), 32) ||
        !build("subm2", subm3(), 32) || !build("sp3", down3({1,1,1}), 64) ||
        !build("subm3", subm3(), 64) || !build("sp4", down3({0,1,1}), 64) ||
        !build("subm4", subm3(), 64)) return false;
    if (weights.has_detection_output) {
        ext::sparse_tensor output_input = current;
        ext::sparse_conv_params out{{3,1,1},{2,1,1},{0,0,0},{1,1,1},false,false};
        ext::sparse_indice_map map;
        if (!ext::build_sparse_indice_map(output_input, out, map, error)) return false;
        maps.emplace("out", std::move(map));
    }
    for (const auto & pair : {std::pair<const char *, const char *>{"sp4","inv4"},
                              {"sp3","inv3"}, {"sp2","inv2"}}) {
        ext::sparse_indice_map inverse;
        if (!ext::build_sparse_inverse_indice_map(maps.at(pair.first), inverse, error)) return false;
        maps.emplace(pair.second, std::move(inverse));
    }

    constexpr size_t graph_nodes = 4096;
    ggml_init_params graph_params{
        ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
        nullptr, true};
    ggml_context * ctx = ggml_init(graph_params);
    if (!ctx) { error = "sparse U-Net graph context init failed"; return false; }
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, point_dim, active);
    ggml_set_input(input);
    // AICore port note: the stock-op lowering gathers with get_rows over a
    // dense [kernel_volume, output_active] rulebook whose -1 entries resolve
    // to one appended zero feature row, so each map needs exactly one I32
    // tensor and one upload (the compact/scatter rulebooks of the upstream
    // patch do not exist here). Every application of a map gathers a tensor
    // with the same active count (SubM/identity topologies), so the remapped
    // zero row is stable per map and one tensor is shared by all its layers.
    std::unordered_map<std::string, ggml_tensor *> dense_tensors;
    for (const auto & entry : maps) {
        ggml_tensor * dense = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32,
            entry.second.kernel_volume * entry.second.output_active_count());
        ggml_set_input(dense);
        dense_tensors.emplace(entry.first, dense);
    }
    auto apply = [&](const char * layer_name, ggml_tensor * features,
                     const char * map_name, bool relu) {
        const auto & layer = weights.layers.at(layer_name);
        const auto & map = maps.at(map_name);
        ggml_tensor * zero_col = ggml_fill(
            ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, features->ne[0], 1), 0.0f);
        ggml_tensor * conv_in = ggml_concat(ctx, features, zero_col, 1);
        ggml_tensor * value = ext::build_sparse_convolution(
            ctx, conv_in, dense_tensors.at(map_name), nullptr, nullptr, map,
            layer.weight, nullptr);
        value = ggml_add(ctx, ggml_mul(ctx, value, layer.scale), layer.bias);
        return relu ? ggml_relu(ctx, value) : value;
    };
    ggml_tensor * x = apply("conv_input", input, "subm1", true);
    ggml_tensor * x1 = apply("conv1", x, "subm1", true);
    ggml_tensor * x2 = apply("conv2.0", x1, "sp2", true);
    x2 = apply("conv2.1", x2, "subm2", true);
    x2 = apply("conv2.2", x2, "subm2", true);
    ggml_tensor * x3 = apply("conv3.0", x2, "sp3", true);
    x3 = apply("conv3.1", x3, "subm3", true);
    x3 = apply("conv3.2", x3, "subm3", true);
    ggml_tensor * x4 = apply("conv4.0", x3, "sp4", true);
    x4 = apply("conv4.1", x4, "subm4", true);
    x4 = apply("conv4.2", x4, "subm4", true);
    ggml_tensor * encoded = weights.has_detection_output
        ? apply("conv_out", x4, "out", true) : nullptr;
    auto reduce_channels = [&](ggml_tensor * value, int output_channels) {
        const int64_t ratio = value->ne[0] / output_channels;
        ggml_tensor * shaped = ggml_reshape_3d(ctx, value, ratio, output_channels, value->ne[1]);
        return ggml_reshape_2d(ctx, ggml_sum_rows(ctx, shaped), output_channels, value->ne[1]);
    };
    auto ur = [&](int stage, ggml_tensor * lateral, ggml_tensor * bottom,
                  const char * subm, const char * inverse, int out_channels) {
        const std::string base = "up_t" + std::to_string(stage);
        ggml_tensor * transformed = apply((base + ".1").c_str(), lateral, subm, true);
        transformed = apply((base + ".2").c_str(), transformed, subm, false);
        transformed = ggml_relu(ctx, ggml_add(ctx, transformed, lateral));
        ggml_tensor * joined = ggml_concat(ctx, bottom, transformed, 0);
        ggml_tensor * mixed = apply(("up_m" + std::to_string(stage)).c_str(),
                                    joined, subm, true);
        mixed = ggml_add(ctx, mixed, reduce_channels(joined, out_channels));
        return apply((inverse ? "inv" + std::to_string(stage) : "conv5").c_str(),
                     mixed, inverse ? inverse : subm, true);
    };
    ggml_tensor * up4 = ur(4, x4, x4, "subm4", "inv4", 64);
    ggml_tensor * up3 = ur(3, x3, up4, "subm3", "inv3", 64);
    ggml_tensor * up2 = ur(2, x2, up3, "subm2", "inv2", 32);
    ggml_tensor * up1 = ur(1, x1, up2, "subm1", nullptr, 16);
    ggml_set_name(up1, "parta2.point_features");
    ggml_set_output(up1);
    if (encoded) { ggml_set_name(encoded, "parta2.encoded"); ggml_set_output(encoded); }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    ggml_build_forward_expand(graph, up1);
    if (encoded) ggml_build_forward_expand(graph, encoded);
    if (!ggml_gallocr_alloc_graph(backend.galloc, graph)) {
        error = "sparse U-Net graph allocation failed"; ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(input, input_features.data(), 0,
                            input_features.size() * sizeof(float));
    {
        // AICore port note: -1 rulebook entries are remapped to the appended
        // zero row (index == the map's input active count) before upload.
        std::vector<int32_t> remapped;
        for (const auto & entry : maps) {
            const int32_t zero_row = static_cast<int32_t>(
                entry.second.input_coordinates.size() / (entry.second.ndim + 1));
            ext::remap_dense_indices(entry.second, zero_row, remapped);
            ext::upload_dense_indices(dense_tensors.at(entry.first), remapped);
        }
    }
    if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
        error = "sparse U-Net graph compute failed"; ggml_free(ctx); return false;
    }
    result.point_features = readback(up1);
    const auto & point_map = maps.at("subm1");
    result.point_coords.resize(static_cast<size_t>(point_map.output_active_count()) * 3);
    for (int64_t row = 0; row < point_map.output_active_count(); ++row) {
        const int32_t * coordinate = point_map.coordinates.data() + row * 4;
        result.point_coords[row * 3] = (coordinate[3] + 0.5f) * hp.voxel_size[0] + hp.point_cloud_range[0];
        result.point_coords[row * 3 + 1] = (coordinate[2] + 0.5f) * hp.voxel_size[1] + hp.point_cloud_range[1];
        result.point_coords[row * 3 + 2] = (coordinate[1] + 0.5f) * hp.voxel_size[2] + hp.point_cloud_range[2];
    }
    if (sink) {
        sink("parta2.point_features", result.point_features.data(),
             point_map.output_active_count(), 16);
        sink("parta2.point_coords", result.point_coords.data(),
             point_map.output_active_count(), 3);
    }
    if (encoded) {
        const auto & encoded_map = maps.at("out");
        const std::vector<float> encoded_features = readback(encoded);
        const int depth = encoded_map.spatial_shape[0];
        result.height = encoded_map.spatial_shape[1];
        result.width = encoded_map.spatial_shape[2];
        const int channels = static_cast<int>(encoded->ne[0]);
        result.spatial_features.assign(
            static_cast<size_t>(result.height) * result.width * depth * channels, 0.0f);
        for (int64_t row = 0; row < encoded_map.output_active_count(); ++row) {
            const int32_t * coordinate = encoded_map.coordinates.data() + row * 4;
            float * target = result.spatial_features.data() +
                (static_cast<size_t>(coordinate[2]) * result.width + coordinate[3]) * depth * channels;
            for (int channel = 0; channel < channels; ++channel)
                target[channel * depth + coordinate[1]] = encoded_features[row * channels + channel];
        }
    }
    ggml_free(ctx);
    return true;
}

} // namespace openpcdet
