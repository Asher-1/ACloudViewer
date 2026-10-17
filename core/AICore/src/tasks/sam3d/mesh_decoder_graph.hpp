// Native SLat mesh decoder graph.  The tensor output is one 101-channel
// FlexiCubes feature vector for every 256^3 sparse cell; mesh extraction is a
// separate host stage so it can be compared independently with the official
// SparseFeatures2Mesh implementation.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "graph_builder.hpp"
#include "gguf_loader.hpp"
#include "sparse_ops.hpp"

namespace sam3d {

struct MeshDecoderGraph {
    GraphContext* g = nullptr;
    const GGUFModel* m = nullptr;
    const MeshTables* tb = nullptr;
    std::string debug_stage;
    std::string prefix = "meshdec";
    // CUDA-only, F16-only fast path that reproduces the PyTorch memory-
    // efficient attention dispatch. Other backends retain the portable graph.
    bool use_pytorch_cuda_attention = false;

    ggml_tensor* x = nullptr;  // (8, N64), token-major input payload
    std::vector<ggml_tensor*> inputs;
    std::vector<std::shared_ptr<std::vector<int32_t>>> table_data;

    // Returns {raw_cube_features (101, N256)} in SparseSubdivide order.
    std::vector<ggml_tensor*> build();
};

}  // namespace sam3d
