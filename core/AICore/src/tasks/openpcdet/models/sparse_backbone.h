#pragma once

#include "../extensions/sparse_ops.h"
#include "../gguf_loader.h"

#include <ggml.h>

#include <string>
#include <vector>

namespace openpcdet {

struct sparse_layer_spec {
    std::string name;
    int in_channels;
    int out_channels;
    ext::sparse_conv_params params;
    bool relu = true;
    bool residual_begin = false;
    bool residual_end = false;
};

struct sparse_layer_weights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
};

struct sparse_backbone_weights {
    std::vector<sparse_layer_spec> specs;
    std::vector<sparse_layer_weights> layers;
};

std::vector<sparse_layer_spec> make_voxel_backbone_8x_specs(
    int input_channels, bool residual);

bool load_sparse_backbone_weights(model_file & file,
                                  const hparams & hp,
                                  sparse_backbone_weights & weights,
                                  std::string & error);

bool build_sparse_backbone_maps(const std::vector<int32_t> & coordinates,
                                const std::vector<int32_t> & spatial_shape,
                                int input_channels,
                                const sparse_backbone_weights & weights,
                                std::vector<ext::sparse_indice_map> & maps,
                                std::string & error,
                                bool use_cuda_rulebook = false,
                                bool retain_host_rulebooks = true);

ggml_tensor * build_sparse_backbone_graph(
    ggml_context * ctx,
    ggml_tensor * input_features,
    const std::vector<ggml_tensor *> & dense_indices,
    const std::vector<ggml_tensor *> & compact_indices,
    const std::vector<ggml_tensor *> & scatter_rows,
    const std::vector<ext::sparse_indice_map> & maps,
    const sparse_backbone_weights & weights,
    std::vector<ggml_tensor *> * taps = nullptr,
    bool use_compact = false);

} // namespace openpcdet
