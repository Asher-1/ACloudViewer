#pragma once

#include "bev_backbone.h"
#include "../gguf_loader.h"

#include <string>
#include <vector>

namespace openpcdet {

struct center_sequence_weights {
    std::vector<conv_bn_block_weights> middle;
    ggml_tensor * output_weight = nullptr;
    ggml_tensor * output_bias = nullptr;
};

struct center_single_weights {
    center_sequence_weights heatmap;
    center_sequence_weights center;
    center_sequence_weights center_z;
    center_sequence_weights dimensions;
    center_sequence_weights rotation;
    center_sequence_weights velocity;
};

struct center_head_weights {
    conv_bn_block_weights shared;
    std::vector<center_single_weights> heads;
};

struct center_single_outputs {
    ggml_tensor * heatmap = nullptr;
    ggml_tensor * center = nullptr;
    ggml_tensor * center_z = nullptr;
    ggml_tensor * dimensions = nullptr;
    ggml_tensor * rotation = nullptr;
    ggml_tensor * velocity = nullptr;
};

bool load_center_head_weights(model_file & file, const hparams & hp,
                              center_head_weights & weights, std::string & error);

std::vector<center_single_outputs> build_center_head_graph(
    ggml_context * ctx, const center_head_weights & weights,
    const hparams & hp, ggml_tensor * spatial_2d);

} // namespace openpcdet
