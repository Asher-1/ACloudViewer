#include "sparse_backbone.h"

#ifdef AICORE_OPENPCDET_CUDA_EXT
#include "../extensions/sparse_rulebook_cuda.h"
#endif

#include <numeric>

namespace openpcdet {

namespace {

ext::sparse_conv_params conv3(bool submanifold, int stride = 1,
                              std::vector<int32_t> padding = {1, 1, 1}) {
    ext::sparse_conv_params params;
    params.kernel_size = {3, 3, 3};
    params.stride = {stride, stride, stride};
    params.padding = std::move(padding);
    params.dilation = {1, 1, 1};
    params.submanifold = submanifold;
    return params;
}

int64_t volume(const std::vector<int32_t> & values) {
    return std::accumulate(values.begin(), values.end(), int64_t{1},
                           [](int64_t a, int32_t b) { return a * b; });
}

bool same_geometry(const ext::sparse_conv_params & a, const ext::sparse_conv_params & b) {
    return a.kernel_size == b.kernel_size && a.stride == b.stride &&
           a.padding == b.padding && a.dilation == b.dilation &&
           a.submanifold == b.submanifold && a.transposed == b.transposed;
}

} // namespace

std::vector<sparse_layer_spec> make_voxel_backbone_8x_specs(
        int input_channels, bool residual) {
    if (!residual) {
        std::vector<sparse_layer_spec> result = {
            {"conv_input", input_channels, 16, conv3(true)},
            {"conv1.0", 16, 16, conv3(true)},
            {"conv2.0", 16, 32, conv3(false, 2)},
            {"conv2.1", 32, 32, conv3(true)},
            {"conv2.2", 32, 32, conv3(true)},
            {"conv3.0", 32, 64, conv3(false, 2)},
            {"conv3.1", 64, 64, conv3(true)},
            {"conv3.2", 64, 64, conv3(true)},
            {"conv4.0", 64, 64, conv3(false, 2, {0, 1, 1})},
            {"conv4.1", 64, 64, conv3(true)},
            {"conv4.2", 64, 64, conv3(true)},
        };
        ext::sparse_conv_params output;
        output.kernel_size = {3, 1, 1};
        output.stride = {2, 1, 1};
        output.padding = {0, 0, 0};
        output.dilation = {1, 1, 1};
        result.push_back({"conv_out", 64, 128, std::move(output)});
        return result;
    }

    std::vector<sparse_layer_spec> result;
    result.push_back({"conv_input", input_channels, 16, conv3(true)});
    auto add_residual = [&](const std::string & name, int channels) {
        sparse_layer_spec first{name + ".conv1", channels, channels, conv3(true)};
        sparse_layer_spec second{name + ".conv2", channels, channels, conv3(true)};
        first.residual_begin = true;
        second.relu = false;
        second.residual_end = true;
        result.push_back(std::move(first));
        result.push_back(std::move(second));
    };
    add_residual("conv1.0", 16); add_residual("conv1.1", 16);
    result.push_back({"conv2.0", 16, 32, conv3(false, 2)});
    add_residual("conv2.1", 32); add_residual("conv2.2", 32);
    result.push_back({"conv3.0", 32, 64, conv3(false, 2)});
    add_residual("conv3.1", 64); add_residual("conv3.2", 64);
    result.push_back({"conv4.0", 64, 128, conv3(false, 2, {0, 1, 1})});
    add_residual("conv4.1", 128); add_residual("conv4.2", 128);
    ext::sparse_conv_params output;
    output.kernel_size = {3, 1, 1}; output.stride = {2, 1, 1};
    output.padding = {0, 0, 0}; output.dilation = {1, 1, 1};
    result.push_back({"conv_out", 128, 128, std::move(output)});
    return result;
}

bool load_sparse_backbone_weights(model_file & file,
                                  const hparams & hp,
                                  sparse_backbone_weights & weights,
                                  std::string & error) {
    weights.specs = make_voxel_backbone_8x_specs(hp.num_point_features, hp.sparse_residual);
    const auto & specs = weights.specs;
    weights.layers.resize(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        const std::string prefix = "sparse." + specs[i].name;
        auto & layer = weights.layers[i];
        layer.weight = file.require((prefix + ".weight").c_str());
        layer.bias = file.require((prefix + ".bias").c_str());
        if (!file.error.empty()) {
            error = file.error;
            return false;
        }
        const int64_t expected_input = volume(specs[i].params.kernel_size) * specs[i].in_channels;
        if (ggml_n_dims(layer.weight) != 2 || layer.weight->ne[0] != expected_input ||
            layer.weight->ne[1] != specs[i].out_channels ||
            ggml_nelements(layer.bias) != specs[i].out_channels) {
            error = "sparse weight shape mismatch: " + prefix;
            return false;
        }
    }
    return true;
}

bool build_sparse_backbone_maps(const std::vector<int32_t> & coordinates,
                                const std::vector<int32_t> & spatial_shape,
                                int input_channels,
                                const sparse_backbone_weights & weights,
                                std::vector<ext::sparse_indice_map> & maps,
                                std::string & error,
                                bool use_cuda_rulebook,
                                bool retain_host_rulebooks) {
    const auto & specs = weights.specs;
    ext::sparse_tensor current;
    current.ndim = 3;
    current.spatial_shape = spatial_shape;
    current.coordinates = coordinates;
    current.channels = input_channels;
    const int32_t * current_device_coordinates = nullptr;
    int32_t current_active = static_cast<int32_t>(coordinates.size() / 4);
    maps.clear();
    maps.reserve(specs.size());
    for (size_t layer_index = 0; layer_index < specs.size(); ++layer_index) {
        const auto & spec = specs[layer_index];
        if (current.channels != spec.in_channels) {
            error = "sparse backbone channel mismatch before " + spec.name;
            return false;
        }
        ext::sparse_indice_map map;
        const bool reuse_previous = layer_index > 0 && spec.params.submanifold &&
            same_geometry(spec.params, specs[layer_index - 1].params);
        if (reuse_previous) {
            const auto & previous = maps.back();
            map.ndim = previous.ndim;
            map.kernel_volume = previous.kernel_volume;
            map.spatial_shape = previous.spatial_shape;
            map.coordinates = current.coordinates;
            map.compact_offsets = previous.compact_offsets;
            map.reuse_of = static_cast<int32_t>(layer_index - 1);
#ifdef AICORE_OPENPCDET_CUDA_EXT
            // SubM reuse shares the previous layer's device-resident rulebook
            // (identical geometry, identical input active set).
            map.device_owner = previous.device_owner;
            map.device_coordinates = previous.device_coordinates;
            map.device_indices = previous.device_indices;
            map.device_compact_offsets = previous.device_compact_offsets;
            map.device_compact_indices = previous.device_compact_indices;
            map.device_scatter_rows = previous.device_scatter_rows;
            map.device_output_active_count = previous.output_active_count();
#endif
        } else {
            bool built = false;
#ifdef AICORE_OPENPCDET_CUDA_EXT
            // GPU rulebook generation (upstream OpenPCDet-GGML
            // sparse_rulebook_cuda). Transposed/2-D geometries fail inside and
            // fall back to the host generator below, matching upstream.
            if (use_cuda_rulebook) {
                if (!ext::build_sparse_indice_map_cuda(
                        current, spec.params, map, error, current_device_coordinates,
                        current_active, retain_host_rulebooks)) {
                    error = spec.name + ": " + error;
                    return false;
                }
                built = true;
            }
#else
            (void) use_cuda_rulebook;
#endif
            if (!built) {
                if (!ext::build_sparse_indice_map(current, spec.params, map, error)) {
                    error = spec.name + ": " + error;
                    return false;
                }
            }
        }
        current.spatial_shape = map.spatial_shape;
        current.coordinates = map.coordinates;
        current.channels = spec.out_channels;
#ifdef AICORE_OPENPCDET_CUDA_EXT
        current_device_coordinates = map.device_coordinates;
        current_active = static_cast<int32_t>(map.output_active_count());
#endif
        maps.push_back(std::move(map));
    }
    return true;
}

ggml_tensor * build_sparse_backbone_graph(
    ggml_context * ctx,
    ggml_tensor * input_features,
    const std::vector<ggml_tensor *> & dense_indices,
    const std::vector<ggml_tensor *> & compact_indices,
    const std::vector<ggml_tensor *> & scatter_rows,
    const std::vector<ext::sparse_indice_map> & maps,
    const sparse_backbone_weights & weights,
    std::vector<ggml_tensor *> * taps,
    bool use_compact) {
    const auto & specs = weights.specs;
    GGML_ASSERT(dense_indices.size() == specs.size() &&
                maps.size() == specs.size() &&
                weights.layers.size() == specs.size());
    if (use_compact) {
        GGML_ASSERT(compact_indices.size() == specs.size() &&
                    scatter_rows.size() == specs.size());
    }
    ggml_tensor * features = input_features;
    ggml_tensor * residual = nullptr;
    for (size_t i = 0; i < specs.size(); ++i) {
        if (specs[i].residual_begin) residual = features;
        if (use_compact) {
            // Compact lowering (CUDA-ext fused ops): the gather consumes the
            // compact rulebook directly and -1 scatter entries write zeros,
            // so no appended zero column and no zero input row are needed.
            // Residual blocks read the clean [C, N] output before the scatter
            // result feeds the next layer.
            features = ext::build_sparse_convolution(
                ctx, features, dense_indices[i], compact_indices[i], scatter_rows[i],
                maps[i], weights.layers[i].weight, weights.layers[i].bias, true);
        } else {
            ggml_tensor * conv_in = features;
            if (i > 0) {
                // AICore port note: the next layer's rulebook remaps -1 entries
                // to an appended zero feature row (stock-op replacement of the
                // upstream gather's "-1 writes zeros" semantics). Elementwise
                // ops above keep a zero row zero, so appending one zero column
                // per layer preserves exact numerics. The concat feeds the conv
                // only — residual keeps pointing at the clean [C, N] output so
                // the residual add below stays shape-exact (residual blocks
                // captured the padded tensor before, which broke nuScenes
                // SECOND).
                ggml_tensor * zero_col = ggml_fill(
                    ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, features->ne[0], 1), 0.0f);
                conv_in = ggml_concat(ctx, features, zero_col, 1);
            }
            features = ext::build_sparse_convolution(
                ctx, conv_in, dense_indices[i], nullptr, nullptr, maps[i],
                weights.layers[i].weight, weights.layers[i].bias);
        }
        if (specs[i].residual_end) {
            GGML_ASSERT(residual && residual->ne[0] == features->ne[0] && residual->ne[1] == features->ne[1]);
            features = ggml_add(ctx, features, residual);
            residual = nullptr;
        }
        if (specs[i].relu || specs[i].residual_end) features = ggml_relu(ctx, features);
        ggml_set_name(features, ("sparse." + specs[i].name).c_str());
        if (taps) {
            ggml_set_output(features);
            taps->push_back(features);
        }
    }
    return features;
}

} // namespace openpcdet
