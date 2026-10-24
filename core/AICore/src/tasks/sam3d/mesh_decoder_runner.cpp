// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mesh_decoder_runner.hpp"

#include <algorithm>
#include <cmath>

#include "backend.hpp"
#include "common.hpp"
#include "dtype_contract.hpp"
#include "gguf_loader.hpp"
#include "graph_builder.hpp"
#include "mesh_decoder_graph.hpp"
#include "sparse_ops.hpp"

namespace sam3d {

bool run_mesh_decoder(const float* input_features,
                      const int32_t* input_coordinates,
                      int64_t input_token_count,
                      const MeshDecoderRunOptions& options,
                      MeshDecoderRunResult& result,
                      std::string& error) {
    result = {};
    if (input_features == nullptr || input_coordinates == nullptr ||
        input_token_count <= 0) {
        error = "mesh decoder requires non-empty [8,N] features and [4,N] "
                "coordinates";
        return false;
    }
    if (options.model_path.empty()) {
        error = "mesh decoder model path is empty";
        return false;
    }

    auto backend = Backend::create(options.backend, options.threads);
    if (!backend) {
        error = "failed to create mesh decoder backend";
        return false;
    }
    GGUFModel model;
    if (!model.load(options.model_path, backend->weights_buffer_type())) {
        error = "failed to load mesh decoder GGUF: " + options.model_path;
        return false;
    }
    if (model.str("sam3d.model") != "slat_decoder_mesh") {
        error = "mesh decoder requires slat_decoder_mesh GGUF, got " +
                model.str("sam3d.model");
        return false;
    }

    const int base_resolution = model.i32("meshdec.resolution", 64);
    MeshTables tables;
    if (!tables.build(input_coordinates, input_token_count, base_resolution)) {
        error = "failed to construct mesh decoder sparse "
                "subdivision/convolution tables";
        return false;
    }
    const int64_t expected_final_tokens = input_token_count * 64;
    if (tables.levels[2].n != expected_final_tokens) {
        error = "mesh decoder two-stage subdivision has an unexpected final "
                "support";
        return false;
    }

    GraphContext context;
    MeshDecoderGraph graph_builder;
    graph_builder.g = &context;
    graph_builder.m = &model;
    graph_builder.tb = &tables;
    graph_builder.debug_stage = options.debug_stage;
#ifdef SAM3D_USE_CUDA
    graph_builder.use_pytorch_cuda_attention =
            options.native_attention && backend->has_gpu();
#endif
    ggml_tensor* input =
            context.input_f32("mesh_decoder_input", {8, input_token_count});
    graph_builder.x = input;
    graph_builder.inputs.push_back(input);
    graph_builder.table_data.push_back(nullptr);
    const std::vector<ggml_tensor*> outputs = graph_builder.build();
    if (outputs.size() != 1) {
        error = "mesh decoder graph returned an unexpected output count";
        return false;
    }
    const int64_t output_channels = outputs.front()->ne[0];
    const int64_t output_tokens = outputs.front()->ne[1];
    if (options.debug_stage.empty() &&
        (output_channels != 101 || output_tokens != expected_final_tokens)) {
        error = "mesh decoder final graph is not [101, N*64]";
        return false;
    }
    const MeshConvLevel* output_level = &tables.levels[0];
    if (output_tokens == tables.levels[1].n) output_level = &tables.levels[1];
    if (output_tokens == tables.levels[2].n) output_level = &tables.levels[2];
    if (output_tokens != output_level->n) {
        error = "mesh decoder stage does not have a matching sparse support";
        return false;
    }

    ggml_cgraph* graph = ggml_new_graph_custom(context.ctx(), 65536, false);
    ggml_set_output(outputs.front());
    ggml_build_forward_expand(graph, outputs.front());
    record_dtype_contract_graph("mesh_decoder", graph);
    if (!backend->alloc(graph) ||
        !backend->set_input_f32(input, input_features,
                                static_cast<size_t>(input_token_count) * 8)) {
        error = "failed to allocate or upload mesh decoder graph";
        return false;
    }
    for (size_t index = 0; index < graph_builder.inputs.size(); ++index) {
        ggml_tensor* graph_input = graph_builder.inputs[index];
        const auto& data = graph_builder.table_data[index];
        if (!graph_input->buffer || !data) continue;
        const bool uploaded =
                graph_input->type == GGML_TYPE_F32
                        ? backend->set_input_f32(
                                  graph_input,
                                  reinterpret_cast<const float*>(data->data()),
                                  data->size())
                        : backend->set_input_i32(graph_input, data->data(),
                                                 data->size());
        if (!uploaded) {
            error = "failed to upload mesh decoder sparse table " +
                    std::string(graph_input->name);
            return false;
        }
    }
    if (!backend->run(graph) ||
        !backend->get_tensor_f32(outputs.front(), result.features) ||
        result.features.size() !=
                static_cast<size_t>(output_tokens) * output_channels ||
        !std::all_of(result.features.begin(), result.features.end(),
                     [](float value) { return std::isfinite(value); })) {
        error = "mesh decoder graph produced invalid feature values";
        return false;
    }

    result.coordinates = output_level->coords;
    result.channels = output_channels;
    result.token_count = output_tokens;
    result.resolution = output_level->resolution;
    result.backend_name = backend->backend_name();
    return true;
}

}  // namespace sam3d
