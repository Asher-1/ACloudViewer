#pragma once

#include "bev_backbone.h"
#include "../gguf_loader.h"

#include <string>
#include <vector>

namespace openpcdet {

struct conv_bias_weights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
};

struct dense_head_sequence_weights {
    std::vector<conv_bn_block_weights> middle;
    conv_bias_weights output;
};

struct anchor_multi_single_weights {
    dense_head_sequence_weights classification;
    std::vector<dense_head_sequence_weights> regression;
    dense_head_sequence_weights box;
    dense_head_sequence_weights direction;
};

struct anchor_multi_weights {
    conv_bn_block_weights shared;
    bool has_shared = false;
    std::vector<anchor_multi_single_weights> heads;
};

struct anchor_multi_single_outputs {
    ggml_tensor * classification = nullptr;
    std::vector<ggml_tensor *> regression;
    ggml_tensor * box = nullptr;
    ggml_tensor * direction = nullptr;
};

bool load_anchor_multi_weights(model_file & file, const hparams & hp,
                               anchor_multi_weights & weights, std::string & error);

std::vector<anchor_multi_single_outputs> build_anchor_multi_graph(
    ggml_context * ctx, const anchor_multi_weights & weights,
    const hparams & hp, ggml_tensor * spatial_2d);

} // namespace openpcdet
