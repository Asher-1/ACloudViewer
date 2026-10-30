#pragma once

#include "../backend.h"
#include "../gguf_loader.h"

#include <string>
#include <vector>

namespace openpcdet {

struct point_mlp_layer_spec {
    std::string weight;
    std::string bias;
    std::string batch_norm;
    bool relu = true;
};

struct point_mlp_layer_weights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * shift = nullptr;
    bool relu = true;
};

struct point_mlp_weights {
    std::vector<point_mlp_layer_weights> layers;
};

bool load_point_mlp(model_file & file, const std::vector<point_mlp_layer_spec> & specs,
                    point_mlp_weights & weights, std::string & error);

void remap_point_mlp(point_mlp_weights & weights, ggml_context * device_context);

// Input/output are row-major [row_count, channels]. All arithmetic is lowered
// to GGML MUL_MAT, broadcast affine, and RELU so every enabled backend shares
// the same graph.
bool run_point_mlp(engine_backend & backend, const point_mlp_weights & weights,
                   const std::vector<float> & input, int64_t row_count,
                   std::vector<float> & output, std::string & error);

} // namespace openpcdet
