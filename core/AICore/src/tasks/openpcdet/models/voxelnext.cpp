#include "voxelnext.h"

#include "../extensions/center_head_ops.h"
#include "../ops/voxelization.h"

#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>

namespace openpcdet {

namespace {

ext::sparse_conv_params conv3d(int kernel, bool submanifold, int stride = 1) {
    ext::sparse_conv_params result;
    result.kernel_size = {kernel, kernel, kernel};
    result.stride = {stride, stride, stride};
    result.padding = {kernel / 2, kernel / 2, kernel / 2};
    result.dilation = {1, 1, 1};
    result.submanifold = submanifold;
    return result;
}

ext::sparse_conv_params conv2d(int kernel, bool submanifold) {
    ext::sparse_conv_params result;
    result.kernel_size = {kernel, kernel};
    result.stride = {1, 1};
    result.padding = {kernel / 2, kernel / 2};
    result.dilation = {1, 1};
    result.submanifold = submanifold;
    return result;
}

int64_t product(const std::vector<int32_t> & values) {
    return std::accumulate(values.begin(), values.end(), int64_t{1},
                           [](int64_t a, int32_t b) { return a * b; });
}

bool check_sparse_weight(ggml_tensor * weight, ggml_tensor * bias,
                         int kernel_volume, int input_channels,
                         int output_channels, const std::string & name,
                         std::string & error) {
    if (!weight || !bias ||
        weight->ne[0] != static_cast<int64_t>(kernel_volume) * input_channels ||
        weight->ne[1] != output_channels || ggml_nelements(bias) != output_channels) {
        error = "VoxelNeXt sparse weight shape mismatch: " + name;
        return false;
    }
    return true;
}

bool load_layer(model_file & file, const std::string & prefix,
                int kernel_volume, int input_channels, int output_channels,
                sparse_layer_weights & layer, std::string & error) {
    layer.weight = file.require((prefix + ".weight").c_str());
    layer.bias = file.require((prefix + ".bias").c_str());
    if (!file.error.empty()) { error = file.error; return false; }
    return check_sparse_weight(layer.weight, layer.bias, kernel_volume,
                               input_channels, output_channels, prefix, error);
}

bool load_sequence(model_file & file, const hparams & hp,
                   const std::string & prefix, int output_channels,
                   voxelnext_sequence_weights & sequence, std::string & error) {
    sequence.middle.resize(hp.center_middle_convs);
    const int channels = hp.center_shared_channels;
    const int kernel_volume = hp.center_head_kernel_size * hp.center_head_kernel_size;
    for (int i = 0; i < hp.center_middle_convs; ++i) {
        if (!load_layer(file, prefix + ".middle." + std::to_string(i),
                        kernel_volume, channels, channels,
                        sequence.middle[i], error)) return false;
    }
    return load_layer(file, prefix + ".out", 1, channels, output_channels,
                      sequence.output, error);
}

std::vector<float> readback_f32(ggml_tensor * tensor) {
    std::vector<float> output(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
    return output;
}

// AICore port note: the stock-op lowering gathers with get_rows over a dense
// [kernel_volume, output_active] rulebook whose -1 entries resolve to one
// appended zero feature row, so every layer call first appends that zero
// column (exact replacement of the upstream gather's "-1 writes zeros"
// semantics); the upstream compact/scatter rulebooks do not exist here.
ggml_tensor * apply_sparse_layer(ggml_context * ctx, ggml_tensor * features,
                                 ggml_tensor * dense_indices,
                                 const ext::sparse_indice_map & map,
                                 const sparse_layer_weights & weights,
                                 bool relu) {
    ggml_tensor * zero_col = ggml_fill(
        ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, features->ne[0], 1), 0.0f);
    ggml_tensor * conv_in = ggml_concat(ctx, features, zero_col, 1);
    ggml_tensor * result = ext::build_sparse_convolution(
        ctx, conv_in, dense_indices, nullptr, nullptr, map, weights.weight, weights.bias);
    return relu ? ggml_relu(ctx, result) : result;
}

// Maps rows from one or more sparse 3-D tensors into unique sparse BEV rows.
// Host-side port of the upstream sparse_bev_reduce_map (extensions/sparse_ops
// CUDA-device fields dropped): `indices` is [output_active_count,
// max_contributors] in row-major storage; rows that have fewer contributors
// are padded with -1. Source row indices refer to the concatenation of all
// input feature matrices.
struct bev_reduce_map {
    std::vector<int32_t> coordinates; // [N, 3] = batch, y, x
    std::vector<int32_t> indices;
    std::vector<int32_t> spatial_shape; // [height, width]
    int32_t max_contributors = 0;

    int64_t output_active_count() const { return coordinates.size() / 3; }
};

bool build_bev_reduce_map(
        const std::vector<const ext::sparse_indice_map *> & inputs,
        const std::vector<int32_t> & coordinate_scales,
        const std::vector<int32_t> & output_spatial_shape,
        bev_reduce_map & map,
        std::string & error) {
    if (inputs.empty() || inputs.size() != coordinate_scales.size() ||
        output_spatial_shape.size() != 2 || output_spatial_shape[0] <= 0 ||
        output_spatial_shape[1] <= 0) {
        error = "bad sparse BEV reduce metadata";
        return false;
    }

    struct bev_coord {
        int32_t b, y, x;
        bool operator<(const bev_coord & other) const {
            return std::array<int32_t, 3>{b, y, x} <
                   std::array<int32_t, 3>{other.b, other.y, other.x};
        }
        bool operator==(const bev_coord & other) const {
            return b == other.b && y == other.y && x == other.x;
        }
    };
    struct contribution { bev_coord coordinate; int32_t source_row; };

    std::vector<contribution> contributions;
    int64_t source_offset = 0;
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const ext::sparse_indice_map * input = inputs[input_index];
        const int32_t scale = coordinate_scales[input_index];
        if (!input || input->ndim != 3 || scale <= 0 ||
            input->coordinates.size() % 4 != 0) {
            error = "sparse BEV reduce requires valid 3-D inputs and positive scales";
            return false;
        }
        const int64_t rows = input->output_active_count();
        contributions.reserve(contributions.size() + static_cast<size_t>(rows));
        for (int64_t row = 0; row < rows; ++row) {
            const int32_t * c = input->coordinates.data() + row * 4;
            const int64_t y = static_cast<int64_t>(c[2]) * scale;
            const int64_t x = static_cast<int64_t>(c[3]) * scale;
            if (c[0] < 0 || y < 0 || x < 0 || y >= output_spatial_shape[0] ||
                x >= output_spatial_shape[1] || source_offset + row >
                    std::numeric_limits<int32_t>::max()) {
                error = "sparse BEV reduce coordinate or source index out of range";
                return false;
            }
            contributions.push_back({{c[0], static_cast<int32_t>(y),
                                      static_cast<int32_t>(x)},
                                     static_cast<int32_t>(source_offset + row)});
        }
        source_offset += rows;
    }
    std::sort(contributions.begin(), contributions.end(),
              [](const contribution & a, const contribution & b) {
                  if (a.coordinate == b.coordinate) return a.source_row < b.source_row;
                  return a.coordinate < b.coordinate;
              });

    map = {};
    map.spatial_shape = output_spatial_shape;
    size_t begin = 0;
    while (begin < contributions.size()) {
        size_t end = begin + 1;
        while (end < contributions.size() &&
               contributions[end].coordinate == contributions[begin].coordinate) ++end;
        map.max_contributors = std::max<int32_t>(
            map.max_contributors, static_cast<int32_t>(end - begin));
        const auto & c = contributions[begin].coordinate;
        map.coordinates.insert(map.coordinates.end(), {c.b, c.y, c.x});
        begin = end;
    }
    if (contributions.empty()) return true;

    map.indices.assign(static_cast<size_t>(map.output_active_count()) *
                           map.max_contributors, -1);
    begin = 0;
    size_t output_row = 0;
    while (begin < contributions.size()) {
        size_t end = begin + 1;
        while (end < contributions.size() &&
               contributions[end].coordinate == contributions[begin].coordinate) ++end;
        for (size_t slot = 0; begin + slot < end; ++slot) {
            map.indices[output_row * map.max_contributors + slot] =
                contributions[begin + slot].source_row;
        }
        ++output_row;
        begin = end;
    }
    return true;
}

// Stock-op replacement of the upstream ggml_sparse_gather-based
// build_sparse_reduce_sum: `flat_indices` is the map's [max_contributors,
// output_active] table exposed as one I32 vector with every -1 entry
// pre-remapped to the appended zero row of `features`. The per-contributor
// gather, channel-major permute and row sum reproduce the upstream
// reduction order exactly.
ggml_tensor * build_bev_reduce_sum(ggml_context * ctx, ggml_tensor * features,
                                   ggml_tensor * flat_indices,
                                   int64_t max_contributors) {
    const int64_t channels = features->ne[0];
    const int64_t n_output = ggml_nelements(flat_indices) / max_contributors;
    ggml_tensor * gathered = ggml_get_rows(ctx, features, flat_indices);
    // Row-major [max_contributors, output_active] indices gather to the same
    // [channels, max_contributors, output_active] layout as the upstream
    // ggml_sparse_gather output; the permute then puts the contributor slot
    // first so sum_rows reduces exactly over the contributors.
    ggml_tensor * shaped = ggml_reshape_3d(ctx, gathered,
                                           channels, max_contributors, n_output);
    ggml_tensor * contributor_major = ggml_cont(
        ctx, ggml_permute(ctx, shaped, 1, 0, 2, 3));
    ggml_tensor * reduced = ggml_sum_rows(ctx, contributor_major);
    return ggml_reshape_2d(ctx, reduced, channels, n_output);
}

struct head_outputs {
    ggml_tensor * heatmap = nullptr;
    ggml_tensor * center = nullptr;
    ggml_tensor * center_z = nullptr;
    ggml_tensor * dimensions = nullptr;
    ggml_tensor * rotation = nullptr;
    ggml_tensor * velocity = nullptr;
    ggml_tensor * iou = nullptr;
};

ggml_tensor * build_sequence(ggml_context * ctx,
                             const voxelnext_sequence_weights & sequence,
                             ggml_tensor * input, ggml_tensor * middle_dense,
                             const ext::sparse_indice_map & middle_map,
                             ggml_tensor * identity_dense,
                             const ext::sparse_indice_map & identity_map) {
    ggml_tensor * value = input;
    for (const auto & layer : sequence.middle)
        value = apply_sparse_layer(ctx, value, middle_dense, middle_map, layer, true);
    return apply_sparse_layer(ctx, value, identity_dense, identity_map,
                              sequence.output, false);
}

} // namespace

std::vector<sparse_layer_spec> make_voxelnext_backbone_specs(
        int input_channels, const std::vector<int32_t> & channels,
        const std::vector<int32_t> & kernel_sizes) {
    if (channels.size() != 5 || kernel_sizes.size() != 4) return {};
    std::vector<sparse_layer_spec> result;
    result.push_back({"conv_input", input_channels, channels[0], conv3d(3, true)});
    auto add_residual = [&](int stage, int block, int channel) {
        const std::string base = "conv" + std::to_string(stage) + "." +
                                 std::to_string(block);
        sparse_layer_spec first{base + ".conv1", channel, channel, conv3d(3, true)};
        sparse_layer_spec second{base + ".conv2", channel, channel, conv3d(3, true)};
        first.residual_begin = true;
        second.relu = false;
        second.residual_end = true;
        result.push_back(std::move(first));
        result.push_back(std::move(second));
    };
    add_residual(1, 0, channels[0]);
    add_residual(1, 1, channels[0]);
    for (int stage = 2; stage <= 6; ++stage) {
        const int input = stage == 2 ? channels[0] : channels[std::min(stage - 2, 4)];
        const int output = channels[std::min(stage - 1, 4)];
        const int kernel = kernel_sizes[std::min(stage - 2, 3)];
        result.push_back({"conv" + std::to_string(stage) + ".0",
                          input, output, conv3d(kernel, false, 2)});
        add_residual(stage, 1, output);
        add_residual(stage, 2, output);
    }
    return result;
}

bool voxelnext::load(const std::string & gguf_path, const std::string & device,
                     int n_threads) {
    if (!model::load(gguf_path, device, n_threads)) return false;
    const hparams & hp = file.hp;
    if (hp.arch != arch_kind::voxelnext || hp.head_kind != dense_head_kind::voxelnext ||
        (hp.center_double_flip && hp.center_has_iou)) {
        error = hp.center_double_flip && hp.center_has_iou
            ? "VoxelNeXt DOUBLE_FLIP with IoU branch is unsupported"
            : "invalid VoxelNeXt architecture metadata";
        return false;
    }
    backbone.specs = make_voxelnext_backbone_specs(
        hp.num_point_features, hp.sparse_channels, hp.sparse_kernel_sizes);
    backbone.layers.resize(backbone.specs.size());
    for (size_t i = 0; i < backbone.specs.size(); ++i) {
        const auto & spec = backbone.specs[i];
        if (!load_layer(file, "voxelnext.backbone." + spec.name,
                        static_cast<int>(product(spec.params.kernel_size)),
                        spec.in_channels, spec.out_channels,
                        backbone.layers[i], error)) return false;
    }
    const int channels = hp.sparse_out_channels;
    if (!load_layer(file, "voxelnext.bev_conv", 9, hp.sparse_channels[3],
                    channels, bev_conv, error) ||
        !load_layer(file, "voxelnext.shared_conv", 9, channels,
                    hp.center_shared_channels, shared_conv, error)) return false;

    heads.resize(hp.center_num_heads);
    for (int i = 0; i < hp.center_num_heads; ++i) {
        const std::string prefix = "voxelnext.head." + std::to_string(i) + ".";
        auto & head = heads[i];
        if (!load_sequence(file, hp, prefix + "hm", hp.center_class_counts[i],
                           head.heatmap, error) ||
            !load_sequence(file, hp, prefix + "center", 2, head.center, error) ||
            !load_sequence(file, hp, prefix + "center_z", 1, head.center_z, error) ||
            !load_sequence(file, hp, prefix + "dim", 3, head.dimensions, error) ||
            !load_sequence(file, hp, prefix + "rot", 2, head.rotation, error) ||
            (hp.center_has_velocity &&
             !load_sequence(file, hp, prefix + "vel", 2, head.velocity, error)) ||
            (hp.center_has_iou &&
             !load_sequence(file, hp, prefix + "iou", 1, head.iou, error))) return false;
    }

    if (device_ctx) {
        auto remap = [&](ggml_tensor *& tensor) {
            if (tensor) tensor = ggml_get_tensor(device_ctx, ggml_get_name(tensor));
        };
        auto remap_layer = [&](sparse_layer_weights & layer) {
            remap(layer.weight); remap(layer.bias);
        };
        auto remap_sequence = [&](voxelnext_sequence_weights & sequence) {
            for (auto & layer : sequence.middle) remap_layer(layer);
            remap_layer(sequence.output);
        };
        for (auto & layer : backbone.layers) remap_layer(layer);
        remap_layer(bev_conv); remap_layer(shared_conv);
        for (auto & head : heads) {
            remap_sequence(head.heatmap); remap_sequence(head.center);
            remap_sequence(head.center_z); remap_sequence(head.dimensions);
            remap_sequence(head.rotation); remap_sequence(head.velocity);
            remap_sequence(head.iou);
        }
    }
    return true;
}

bool voxelnext::forward_single(const float * points, int32_t n_points, int32_t point_dim,
                               raw_result & result, const tap_sink & sink) {
    const hparams & hp = file.hp;
    result = {};
    if (n_points < 0 || (n_points > 0 && !points) || point_dim != hp.num_point_features) {
        error = "invalid VoxelNeXt point buffer or feature dimension";
        return false;
    }
    if (n_points == 0) return true;
    // AICore port note: stage timings are reported through the common
    // aicore_pipeline_timings contract by the C API wrapper; the upstream
    // stderr lap profiler stays available under OPCPD_PROFILE_LAPS builds.
#ifdef OPCPD_PROFILE_LAPS
    const bool profile = true;
#else
    const bool profile = false;
#endif
    auto lap_start = std::chrono::steady_clock::now();
    auto lap = [&](const char * name) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[profile] %-24s %8.2f ms\n", name,
                     std::chrono::duration<double, std::milli>(now - lap_start).count());
        lap_start = now;
    };

    voxelization_params vp;
    vp.point_cloud_range = hp.point_cloud_range;
    vp.voxel_size = hp.voxel_size;
    vp.grid_size = hp.grid_size;
    vp.max_points_per_voxel = hp.max_points_per_voxel;
    vp.max_voxels = hp.max_voxels;
    voxel_output voxels;
    if (!voxelize(vp, points, n_points, point_dim, voxels, error)) return false;
    const int active = voxels.actual_voxels;
    if (active == 0) return true;
    std::vector<float> features(static_cast<size_t>(active) * point_dim, 0.0f);
    for (int row = 0; row < active; ++row) {
        const int count = voxels.num_points[row];
        for (int p = 0; p < count; ++p) {
            const float * source = voxels.voxels.data() +
                (static_cast<size_t>(row) * hp.max_points_per_voxel + p) * point_dim;
            for (int c = 0; c < point_dim; ++c)
                features[static_cast<size_t>(row) * point_dim + c] += source[c];
        }
        for (int c = 0; c < point_dim; ++c)
            features[static_cast<size_t>(row) * point_dim + c] /= count;
    }
    std::vector<int32_t> coordinates(voxels.coords.begin(),
        voxels.coords.begin() + static_cast<size_t>(active) * 4);
    if (sink) sink("mean_vfe", features.data(), active, point_dim);
    lap("voxelize_mean_vfe");

    std::vector<ext::sparse_indice_map> maps;
    const std::vector<int32_t> sparse_shape = {
        hp.grid_size[2] + 1, hp.grid_size[1], hp.grid_size[0]};
    if (!build_sparse_backbone_maps(coordinates, sparse_shape, point_dim,
                                    backbone, maps, error)) return false;
    if (maps.size() != 30) { error = "unexpected VoxelNeXt backbone topology"; return false; }
    bev_reduce_map reduce_map;
    if (!build_bev_reduce_map(
            {&maps[19], &maps[24], &maps[29]}, {1, 2, 4},
            {maps[19].spatial_shape[1], maps[19].spatial_shape[2]},
            reduce_map, error)) return false;
    if (sink) {
        for (size_t i = 0; i < maps.size(); ++i) {
            std::vector<float> map_coordinates(maps[i].coordinates.begin(),
                                               maps[i].coordinates.end());
            sink("voxelnext.backbone." + backbone.specs[i].name + ".coords",
                 map_coordinates.data(), maps[i].output_active_count(), 4);
        }
        std::vector<float> bev_coordinates(reduce_map.coordinates.begin(),
                                           reduce_map.coordinates.end());
        sink("voxelnext.bev_reduce.coords", bev_coordinates.data(),
             reduce_map.output_active_count(), 3);
    }

    ext::sparse_tensor bev_input;
    bev_input.ndim = 2;
    bev_input.channels = hp.sparse_channels[3];
    bev_input.spatial_shape = reduce_map.spatial_shape;
    bev_input.coordinates = reduce_map.coordinates;
    ext::sparse_indice_map bev_map, shared_map, head_middle_map, identity_map;
    if (!ext::build_sparse_indice_map(bev_input, conv2d(3, false), bev_map, error)) return false;
    ext::sparse_tensor shared_input = bev_input;
    shared_input.channels = hp.sparse_out_channels;
    shared_input.spatial_shape = bev_map.spatial_shape;
    shared_input.coordinates = bev_map.coordinates;
    if (!ext::build_sparse_indice_map(shared_input, conv2d(3, true), shared_map, error)) return false;
    ext::sparse_tensor head_input = shared_input;
    head_input.channels = hp.center_shared_channels;
    if (!ext::build_sparse_indice_map(head_input,
            conv2d(hp.center_head_kernel_size, true), head_middle_map, error) ||
        !ext::build_sparse_indice_map(head_input, conv2d(1, true), identity_map, error)) return false;
    if (sink) {
        std::vector<float> shared_coordinates(shared_map.coordinates.begin(),
                                              shared_map.coordinates.end());
        sink("voxelnext.shared.coords", shared_coordinates.data(),
             shared_map.output_active_count(), 3);
    }
    lap("sparse_indice_maps");

    constexpr size_t graph_nodes = 8192;
    ggml_init_params graph_params = {
        ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
        nullptr, true,
    };
    ggml_context * ctx = ggml_init(graph_params);
    if (!ctx) { error = "VoxelNeXt graph context init failed"; return false; }
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, point_dim, active);
    ggml_set_input(input);
    // AICore port note: one dense I32 rulebook tensor per non-reuse map; the
    // -1 entries are remapped to the gather source's appended zero row before
    // upload (identical SubM geometry shares one tensor, see reuse_of).
    std::vector<ggml_tensor *> dense_tensors(maps.size(), nullptr);
    for (size_t i = 0; i < maps.size(); ++i) {
        if (maps[i].reuse_of >= 0) {
            dense_tensors[i] = dense_tensors[maps[i].reuse_of];
        } else {
            dense_tensors[i] = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32,
                maps[i].kernel_volume * maps[i].output_active_count());
            ggml_set_input(dense_tensors[i]);
        }
    }
    ggml_tensor * value = input;
    ggml_tensor * residual = nullptr;
    ggml_tensor * x4 = nullptr, * x5 = nullptr, * x6 = nullptr;
    std::vector<ggml_tensor *> taps;
    for (size_t i = 0; i < backbone.specs.size(); ++i) {
        const auto & spec = backbone.specs[i];
        if (spec.residual_begin) residual = value;
        value = apply_sparse_layer(ctx, value, dense_tensors[i], maps[i],
                                   backbone.layers[i], false);
        if (spec.residual_end) { value = ggml_add(ctx, value, residual); residual = nullptr; }
        if (spec.relu || spec.residual_end) value = ggml_relu(ctx, value);
        ggml_set_name(value, ("voxelnext.backbone." + spec.name).c_str());
        if (i == 19) x4 = value;
        if (i == 24) x5 = value;
        if (i == 29) x6 = value;
        if (sink) { ggml_set_output(value); taps.push_back(value); }
    }
    ggml_tensor * pyramid = ggml_concat(ctx, ggml_concat(ctx, x4, x5, 1), x6, 1);
    ggml_tensor * pyramid_zero = ggml_fill(
        ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pyramid->ne[0], 1), 0.0f);
    ggml_tensor * pyramid_padded = ggml_concat(ctx, pyramid, pyramid_zero, 1);
    const int32_t pyramid_rows = static_cast<int32_t>(pyramid->ne[1]);
    ggml_tensor * reduce_indices = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32,
        reduce_map.max_contributors * reduce_map.output_active_count());
    ggml_set_input(reduce_indices);
    value = build_bev_reduce_sum(ctx, pyramid_padded, reduce_indices,
                                 reduce_map.max_contributors);
    ggml_tensor * bev_reduced = value;
    ggml_set_name(bev_reduced, "voxelnext.bev_reduce");
    if (sink) ggml_set_output(bev_reduced);
    // AICore port note: the bev/shared/head rulebook tensors are created right
    // before their first use; apply_sparse_layer appends the zero column that
    // the layer's remapped -1 rulebook entries resolve to.
    ggml_tensor * bev_dense = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, bev_map.kernel_volume * bev_map.output_active_count());
    ggml_tensor * shared_dense = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, shared_map.kernel_volume * shared_map.output_active_count());
    ggml_tensor * middle_dense = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32,
        head_middle_map.kernel_volume * head_middle_map.output_active_count());
    ggml_tensor * identity_dense = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32,
        identity_map.kernel_volume * identity_map.output_active_count());
    for (ggml_tensor * tensor : {bev_dense, shared_dense, middle_dense, identity_dense})
        ggml_set_input(tensor);
    value = apply_sparse_layer(ctx, value, bev_dense, bev_map, bev_conv, true);
    ggml_tensor * bev_convolved = value;
    ggml_set_name(bev_convolved, "voxelnext.bev_conv");
    if (sink) ggml_set_output(bev_convolved);
    value = apply_sparse_layer(ctx, value, shared_dense, shared_map,
                               shared_conv, true);
    ggml_set_name(value, "voxelnext.shared");
    if (sink) { ggml_set_output(value); taps.push_back(value); }

    std::vector<head_outputs> outputs(heads.size());
    for (size_t i = 0; i < heads.size(); ++i) {
        const auto & source = heads[i];
        auto & out = outputs[i];
        out.heatmap = build_sequence(ctx, source.heatmap, value,
                                     middle_dense, head_middle_map,
                                     identity_dense, identity_map);
        out.center = build_sequence(ctx, source.center, value,
                                    middle_dense, head_middle_map,
                                    identity_dense, identity_map);
        out.center_z = build_sequence(ctx, source.center_z, value,
                                      middle_dense, head_middle_map,
                                      identity_dense, identity_map);
        out.dimensions = build_sequence(ctx, source.dimensions, value,
                                        middle_dense, head_middle_map,
                                        identity_dense, identity_map);
        out.rotation = build_sequence(ctx, source.rotation, value,
                                      middle_dense, head_middle_map,
                                      identity_dense, identity_map);
        if (hp.center_has_velocity)
            out.velocity = build_sequence(ctx, source.velocity, value,
                                          middle_dense, head_middle_map,
                                          identity_dense, identity_map);
        if (hp.center_has_iou)
            out.iou = build_sequence(ctx, source.iou, value,
                                     middle_dense, head_middle_map,
                                     identity_dense, identity_map);
        for (ggml_tensor * tensor : {out.heatmap, out.center, out.center_z,
                                     out.dimensions, out.rotation, out.velocity, out.iou})
            if (tensor) ggml_set_output(tensor);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
    for (const auto & out : outputs)
        for (ggml_tensor * tensor : {out.heatmap, out.center, out.center_z,
                                     out.dimensions, out.rotation, out.velocity, out.iou})
            if (tensor) ggml_build_forward_expand(graph, tensor);
    for (ggml_tensor * tap : taps) ggml_build_forward_expand(graph, tap);
    if (sink) {
        ggml_build_forward_expand(graph, bev_reduced);
        ggml_build_forward_expand(graph, bev_convolved);
    }
    if (!ggml_gallocr_alloc_graph(be.galloc, graph)) {
        error = "VoxelNeXt graph allocation failed"; ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(input, features.data(), 0, features.size() * sizeof(float));
    {
        // AICore port note: -1 rulebook entries resolve to the appended zero
        // row of each layer's gather source: the graph input for layer 0, the
        // previous layer's output row count afterwards (SubM reuse chains
        // keep identical counts, so the shared tensor stays valid).
        std::vector<int32_t> remapped;
        for (size_t i = 0; i < maps.size(); ++i) {
            if (maps[i].reuse_of >= 0) continue;
            const int32_t zero_row = i == 0
                ? static_cast<int32_t>(active)
                : static_cast<int32_t>(maps[i - 1].output_active_count());
            ext::remap_dense_indices(maps[i], zero_row, remapped);
            ext::upload_dense_indices(dense_tensors[i], remapped);
        }
        ext::remap_dense_indices(bev_map,
                                 static_cast<int32_t>(reduce_map.output_active_count()),
                                 remapped);
        ext::upload_dense_indices(bev_dense, remapped);
        ext::remap_dense_indices(shared_map,
                                 static_cast<int32_t>(bev_map.output_active_count()),
                                 remapped);
        ext::upload_dense_indices(shared_dense, remapped);
        ext::remap_dense_indices(head_middle_map,
                                 static_cast<int32_t>(shared_map.output_active_count()),
                                 remapped);
        ext::upload_dense_indices(middle_dense, remapped);
        ext::remap_dense_indices(identity_map,
                                 static_cast<int32_t>(shared_map.output_active_count()),
                                 remapped);
        ext::upload_dense_indices(identity_dense, remapped);
        remapped.resize(reduce_map.indices.size());
        for (size_t k = 0; k < reduce_map.indices.size(); ++k)
            remapped[k] = reduce_map.indices[k] < 0 ? pyramid_rows
                                                    : reduce_map.indices[k];
        ext::upload_dense_indices(reduce_indices, remapped);
    }
    if (ggml_backend_graph_compute(be.be, graph) != GGML_STATUS_SUCCESS) {
        error = "VoxelNeXt graph compute failed"; ggml_free(ctx); return false;
    }
    lap("sparse_compute");
    if (sink) {
        for (size_t i = 0; i < backbone.specs.size(); ++i) {
            const std::vector<float> values = readback_f32(taps[i]);
            sink(ggml_get_name(taps[i]), values.data(), maps[i].output_active_count(),
                 backbone.specs[i].out_channels);
        }
        const std::vector<float> shared_values = readback_f32(taps.back());
        sink(ggml_get_name(taps.back()), shared_values.data(),
             shared_map.output_active_count(), hp.center_shared_channels);
        const std::vector<float> reduced_values = readback_f32(bev_reduced);
        sink(ggml_get_name(bev_reduced), reduced_values.data(),
             reduce_map.output_active_count(), hp.sparse_channels[3]);
        const std::vector<float> convolved_values = readback_f32(bev_convolved);
        sink(ggml_get_name(bev_convolved), convolved_values.data(),
             bev_map.output_active_count(), hp.sparse_out_channels);
    }

    result.coordinates = shared_map.coordinates;
    result.spatial_shape = shared_map.spatial_shape;
    result.heads.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto & destination = result.heads[i];
        destination.heatmap = readback_f32(outputs[i].heatmap);
        destination.center = readback_f32(outputs[i].center);
        destination.center_z = readback_f32(outputs[i].center_z);
        destination.dimensions = readback_f32(outputs[i].dimensions);
        destination.rotation = readback_f32(outputs[i].rotation);
        if (outputs[i].velocity) destination.velocity = readback_f32(outputs[i].velocity);
        if (outputs[i].iou) destination.iou = readback_f32(outputs[i].iou);
        if (sink) {
            const std::string prefix = "voxelnext.head." + std::to_string(i) + ".";
            const int active_count = static_cast<int>(shared_map.output_active_count());
            sink(prefix + "hm", destination.heatmap.data(), active_count,
                 hp.center_class_counts[i]);
            sink(prefix + "center", destination.center.data(), active_count, 2);
            sink(prefix + "center_z", destination.center_z.data(), active_count, 1);
            sink(prefix + "dim", destination.dimensions.data(), active_count, 3);
            sink(prefix + "rot", destination.rotation.data(), active_count, 2);
        }
    }
    ggml_free(ctx);
    lap("readback_heads");
    return true;
}

bool voxelnext::merge_double_flip(const std::array<raw_result, 4> & inputs,
                                  raw_result & output) {
    const hparams & hp = file.hp;
    if (inputs[0].spatial_shape.size() != 2 || inputs[0].heads.size() != heads.size()) {
        error = "bad VoxelNeXt double-flip source shape";
        return false;
    }
    using coordinate = std::pair<int32_t, int32_t>;
    std::map<coordinate, int32_t> row_by_coordinate;
    std::array<std::vector<coordinate>, 4> transformed_coordinates;
    for (int flip = 0; flip < 4; ++flip) {
        const auto & source = inputs[flip];
        if (source.spatial_shape != inputs[0].spatial_shape ||
            source.heads.size() != inputs[0].heads.size() ||
            source.coordinates.size() % 3 != 0) {
            error = "incompatible VoxelNeXt double-flip outputs";
            return false;
        }
        const int64_t rows = static_cast<int64_t>(source.coordinates.size()) / 3;
        transformed_coordinates[flip].resize(rows);
        for (int64_t row = 0; row < rows; ++row) {
            int32_t y = source.coordinates[row * 3 + 1];
            int32_t x = source.coordinates[row * 3 + 2];
            if (flip == 1 || flip == 3) y = source.spatial_shape[0] - y;
            if (flip == 2 || flip == 3) x = source.spatial_shape[1] - x;
            transformed_coordinates[flip][row] = {y, x};
            row_by_coordinate.emplace(coordinate{y, x}, 0);
        }
    }

    int32_t merged_rows = 0;
    output = {};
    output.spatial_shape = inputs[0].spatial_shape;
    output.heatmap_is_probability = true;
    output.dimensions_are_linear = true;
    output.coordinates.resize(row_by_coordinate.size() * 3);
    for (auto & entry : row_by_coordinate) {
        entry.second = merged_rows;
        output.coordinates[merged_rows * 3] = 0;
        output.coordinates[merged_rows * 3 + 1] = entry.first.first;
        output.coordinates[merged_rows * 3 + 2] = entry.first.second;
        ++merged_rows;
    }
    std::vector<int32_t> counts(merged_rows, 0);
    output.heads.resize(inputs[0].heads.size());
    for (size_t head = 0; head < output.heads.size(); ++head) {
        auto & destination = output.heads[head];
        const int classes = hp.center_class_counts[head];
        destination.heatmap.assign(static_cast<size_t>(merged_rows) * classes, 0.0f);
        destination.center.assign(static_cast<size_t>(merged_rows) * 2, 0.0f);
        destination.center_z.assign(merged_rows, 0.0f);
        destination.dimensions.assign(static_cast<size_t>(merged_rows) * 3, 0.0f);
        destination.rotation.assign(static_cast<size_t>(merged_rows) * 2, 0.0f);
        if (hp.center_has_velocity)
            destination.velocity.assign(static_cast<size_t>(merged_rows) * 2, 0.0f);
    }

    for (int flip = 0; flip < 4; ++flip) {
        const auto & source_result = inputs[flip];
        const int64_t source_rows = static_cast<int64_t>(source_result.coordinates.size()) / 3;
        for (int64_t source_row = 0; source_row < source_rows; ++source_row) {
            const int32_t destination_row = row_by_coordinate.at(
                transformed_coordinates[flip][source_row]);
            ++counts[destination_row];
            for (size_t head = 0; head < output.heads.size(); ++head) {
                const auto & source = source_result.heads[head];
                auto & destination = output.heads[head];
                const int classes = hp.center_class_counts[head];
                for (int channel = 0; channel < classes; ++channel) {
                    const float logit = source.heatmap[source_row * classes + channel];
                    const float exponential = std::exp(logit >= 0.0f ? -logit : logit);
                    const float probability = logit >= 0.0f
                        ? 1.0f / (1.0f + exponential)
                        : exponential / (1.0f + exponential);
                    destination.heatmap[destination_row * classes + channel] += probability;
                }
                const float x_sign = (flip == 2 || flip == 3) ? -1.0f : 1.0f;
                const float y_sign = (flip == 1 || flip == 3) ? -1.0f : 1.0f;
                destination.center[destination_row * 2] +=
                    x_sign * source.center[source_row * 2];
                destination.center[destination_row * 2 + 1] +=
                    y_sign * source.center[source_row * 2 + 1];
                destination.center_z[destination_row] += source.center_z[source_row];
                for (int channel = 0; channel < 3; ++channel)
                    destination.dimensions[destination_row * 3 + channel] +=
                        std::exp(source.dimensions[source_row * 3 + channel]);
                destination.rotation[destination_row * 2] +=
                    x_sign * source.rotation[source_row * 2];
                destination.rotation[destination_row * 2 + 1] +=
                    y_sign * source.rotation[source_row * 2 + 1];
                if (hp.center_has_velocity) {
                    destination.velocity[destination_row * 2] +=
                        x_sign * source.velocity[source_row * 2];
                    destination.velocity[destination_row * 2 + 1] +=
                        y_sign * source.velocity[source_row * 2 + 1];
                }
            }
        }
    }
    for (int32_t row = 0; row < merged_rows; ++row) {
        const float scale = 1.0f / counts[row];
        for (size_t head = 0; head < output.heads.size(); ++head) {
            auto & value = output.heads[head];
            const int classes = hp.center_class_counts[head];
            for (int channel = 0; channel < classes; ++channel)
                value.heatmap[row * classes + channel] *= scale;
            for (int channel = 0; channel < 2; ++channel) {
                value.center[row * 2 + channel] *= scale;
                value.rotation[row * 2 + channel] *= scale;
                if (hp.center_has_velocity) value.velocity[row * 2 + channel] *= scale;
            }
            value.center_z[row] *= scale;
            for (int channel = 0; channel < 3; ++channel)
                value.dimensions[row * 3 + channel] *= scale;
        }
    }
    return true;
}

bool voxelnext::decode(const raw_result & result,
                       std::vector<detection> & detections,
                       const tap_sink & sink) {
    const hparams & hp = file.hp;
    detections.clear();
    if (result.coordinates.empty()) return true;
    ext::center_decode_params params;
    params.point_cloud_range = hp.point_cloud_range;
    params.voxel_size = hp.voxel_size;
    params.feature_map_stride = hp.center_feature_map_stride;
    params.max_objects = hp.center_max_objects;
    params.score_threshold = hp.score_thresh;
    params.post_center_limit_range = hp.center_post_limit_range;
    params.nms_threshold = hp.nms_thresh;
    params.nms_pre_maxsize = hp.nms_pre_maxsize;
    params.nms_post_maxsize = hp.max_boxes;
    params.iou_rectifier = hp.center_rectifier;
    params.class_nms_thresholds = hp.center_nms_thresholds;
    params.class_nms_pre_maxsizes = hp.center_nms_pre_maxsizes;
    params.class_nms_post_maxsizes = hp.center_nms_post_maxsizes;
    size_t label_offset = 0;
    const int active_count = static_cast<int>(result.coordinates.size() / 3);
    for (size_t i = 0; i < result.heads.size(); ++i) {
        const auto & source = result.heads[i];
        ext::center_head_view view;
        view.heatmap = source.heatmap.data(); view.center = source.center.data();
        view.center_z = source.center_z.data(); view.dimensions = source.dimensions.data();
        view.rotation = source.rotation.data();
        view.velocity = source.velocity.empty() ? nullptr : source.velocity.data();
        view.iou = source.iou.empty() ? nullptr : source.iou.data();
        view.coordinates = result.coordinates.data();
        view.active_count = active_count;
        view.class_count = hp.center_class_counts[i];
        view.heatmap_is_probability = result.heatmap_is_probability;
        view.dimensions_are_linear = result.dimensions_are_linear;
        view.class_mapping.assign(hp.center_label_indices.begin() + label_offset,
                                  hp.center_label_indices.begin() + label_offset +
                                      hp.center_class_counts[i]);
        label_offset += hp.center_class_counts[i];
        std::vector<detection> current;
        if (!ext::decode_center_head(view, params, current, error)) return false;
        detections.insert(detections.end(), current.begin(), current.end());
        if (sink && hp.center_double_flip) {
            const std::string prefix = "voxelnext.double_flip.head." + std::to_string(i) + ".";
            sink(prefix + "hm", source.heatmap.data(), active_count, view.class_count);
            sink(prefix + "center", source.center.data(), active_count, 2);
            sink(prefix + "center_z", source.center_z.data(), active_count, 1);
            sink(prefix + "dim", source.dimensions.data(), active_count, 3);
            sink(prefix + "rot", source.rotation.data(), active_count, 2);
        }
    }
    std::sort(detections.begin(), detections.end(),
              [](const detection & a, const detection & b) { return a.score > b.score; });
    return true;
}

bool voxelnext::forward(const float * points, int32_t n_points, int32_t point_dim,
                        std::vector<detection> & detections, const tap_sink & sink) {
    detections.clear();
    if (!file.hp.center_double_flip) {
        raw_result result;
        return forward_single(points, n_points, point_dim, result, sink) &&
               decode(result, detections, sink);
    }
    if (n_points < 0 || (n_points > 0 && !points) || point_dim != file.hp.num_point_features) {
        error = "invalid VoxelNeXt point buffer or feature dimension";
        return false;
    }
    if (n_points == 0) return true;
    std::array<raw_result, 4> results;
    if (!forward_single(points, n_points, point_dim, results[0], sink)) return false;
    std::vector<float> flipped(static_cast<size_t>(n_points) * point_dim);
    for (int flip = 1; flip < 4; ++flip) {
        std::copy_n(points, flipped.size(), flipped.data());
        for (int32_t point = 0; point < n_points; ++point) {
            if (flip == 2 || flip == 3)
                flipped[static_cast<size_t>(point) * point_dim] *= -1.0f;
            if (flip == 1 || flip == 3)
                flipped[static_cast<size_t>(point) * point_dim + 1] *= -1.0f;
        }
        if (!forward_single(flipped.data(), n_points, point_dim, results[flip], {})) return false;
    }
    raw_result merged;
    return merge_double_flip(results, merged) && decode(merged, detections, sink);
}

} // namespace openpcdet
