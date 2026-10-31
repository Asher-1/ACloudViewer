// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "point_mlp.h"

#include <ggml.h>

#include <sstream>
#include <utility>

namespace openpcdet {

namespace {

std::pair<int64_t, int64_t> matrix_shape(const point_mlp_layer_weights& layer) {
    const ggml_tensor* weight = layer.weight;
    const ggml_tensor* affine = layer.scale  ? layer.scale
                                : layer.bias ? layer.bias
                                             : layer.shift;
    if (affine && ggml_nelements(affine) > 0) {
        const int64_t output_channels = ggml_nelements(affine);
        return {ggml_nelements(weight) / output_channels, output_channels};
    }
    int output_axis = -1;
    for (int axis = GGML_MAX_DIMS - 1; axis >= 1; --axis) {
        if (weight->ne[axis] > 1) {
            output_axis = axis;
            break;
        }
    }
    if (output_axis < 1) return {0, 0};
    int64_t input_channels = 1;
    for (int axis = 0; axis < output_axis; ++axis)
        input_channels *= weight->ne[axis];
    return {input_channels, weight->ne[output_axis]};
}

}  // namespace

bool load_point_mlp(model_file& file,
                    const std::vector<point_mlp_layer_spec>& specs,
                    point_mlp_weights& weights,
                    std::string& error) {
    weights.layers.clear();
    for (const point_mlp_layer_spec& spec : specs) {
        point_mlp_layer_weights layer;
        layer.weight = file.require(spec.weight.c_str());
        layer.relu = spec.relu;
        if (!spec.batch_norm.empty()) {
            layer.scale =
                    file.require((spec.batch_norm + ".eval_scale").c_str());
            layer.shift =
                    file.require((spec.batch_norm + ".eval_bias").c_str());
        } else if (!spec.bias.empty()) {
            layer.bias = file.require(spec.bias.c_str());
        }
        if (!file.error.empty()) {
            error = file.error;
            weights.layers.clear();
            return false;
        }
        weights.layers.push_back(layer);
    }
    if (weights.layers.empty()) {
        error = "point MLP has no layers";
        return false;
    }
    return true;
}

void remap_point_mlp(point_mlp_weights& weights, ggml_context* device_context) {
    if (!device_context) return;
    auto remap = [&](ggml_tensor*& tensor) {
        if (tensor)
            tensor = ggml_get_tensor(device_context, ggml_get_name(tensor));
    };
    for (point_mlp_layer_weights& layer : weights.layers) {
        remap(layer.weight);
        remap(layer.bias);
        remap(layer.scale);
        remap(layer.shift);
    }
}

bool run_point_mlp(engine_backend& backend,
                   const point_mlp_weights& weights,
                   const std::vector<float>& input,
                   int64_t row_count,
                   std::vector<float>& output,
                   std::string& error) {
    output.clear();
    if (row_count < 0 || weights.layers.empty()) {
        error = "bad point MLP input";
        return false;
    }
    const ggml_tensor* first_weight = weights.layers.front().weight;
    const auto first_shape = first_weight ? matrix_shape(weights.layers.front())
                                          : std::pair<int64_t, int64_t>{0, 0};
    if (!first_weight || first_shape.first <= 0 ||
        input.size() != static_cast<size_t>(row_count * first_shape.first)) {
        std::ostringstream message;
        message << "point MLP input shape mismatch at "
                << (first_weight ? ggml_get_name(first_weight) : "<null>")
                << ": expected " << first_shape.first << " channels x "
                << row_count << " rows, got "
                << (row_count > 0 ? input.size() / row_count : input.size())
                << " channels (" << input.size() << " values)";
        error = message.str();
        return false;
    }
    constexpr size_t graph_nodes = 256;
    ggml_init_params params = {
            ggml_tensor_overhead() * graph_nodes +
                    ggml_graph_overhead_custom(graph_nodes, false),
            nullptr,
            true,
    };
    ggml_context* context = ggml_init(params);
    if (!context) {
        error = "point MLP context allocation failed";
        return false;
    }
    ggml_tensor* current = ggml_new_tensor_2d(context, GGML_TYPE_F32,
                                              first_shape.first, row_count);
    ggml_set_input(current);
    ggml_tensor* input_tensor = current;
    for (const point_mlp_layer_weights& layer : weights.layers) {
        const auto shape = layer.weight ? matrix_shape(layer)
                                        : std::pair<int64_t, int64_t>{0, 0};
        if (!layer.weight || shape.first != current->ne[0]) {
            std::ostringstream message;
            message << "point MLP weight shape mismatch at "
                    << (layer.weight ? ggml_get_name(layer.weight) : "<null>")
                    << ": expected input " << shape.first << ", got "
                    << current->ne[0];
            error = message.str();
            ggml_free(context);
            return false;
        }
        ggml_tensor* matrix = ggml_reshape_2d(context, layer.weight,
                                              shape.first, shape.second);
        current = ggml_mul_mat(context, matrix, current);
        if (layer.scale) current = ggml_mul(context, current, layer.scale);
        if (layer.shift) current = ggml_add(context, current, layer.shift);
        if (layer.bias) current = ggml_add(context, current, layer.bias);
        if (layer.relu) current = ggml_relu(context, current);
    }
    ggml_set_output(current);
    ggml_cgraph* graph = ggml_new_graph_custom(context, graph_nodes, false);
    ggml_build_forward_expand(graph, current);
    if (!ggml_gallocr_alloc_graph(backend.galloc, graph)) {
        error = "point MLP graph allocation failed";
        ggml_free(context);
        return false;
    }
    ggml_backend_tensor_set(input_tensor, input.data(), 0,
                            input.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.be, graph) != GGML_STATUS_SUCCESS) {
        error = "point MLP graph compute failed";
        ggml_free(context);
        return false;
    }
    output.resize(static_cast<size_t>(ggml_nelements(current)));
    ggml_backend_tensor_get(current, output.data(), 0,
                            output.size() * sizeof(float));
    ggml_free(context);
    return true;
}

}  // namespace openpcdet
