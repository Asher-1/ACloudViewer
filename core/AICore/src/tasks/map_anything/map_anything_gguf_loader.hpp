// GGUF loader for VGGT-Omega weights + metadata (map_anything).
// Loads every tensor onto the host (mmapped by gguf), exposes scalar/array
// metadata needed to drive the graph, and hands tensors to the model.
#pragma once

#include "map_anything_common.hpp"

#include <memory>
#include <vector>

namespace map_anything {

struct VGGTMeta {
    std::string name = "vggt-omega";
    std::string dtype = "?";
    // GGUF general.architecture, normalized to the builder-registry key
    // ("vggt" in older files maps to "vggt_omega"). New architectures add
    // their own registry entry — see src/graph_builder.hpp.
    std::string architecture = "vggt_omega";

    int patch_size = 16;
    int embed_dim = 1024;
    int depth = 24;              // backbone blocks (DINOv3-style ViT-L)
    int aa_depth = 24;           // frame & inter-frame block pairs
    int num_heads = 16;
    int num_register_tokens = 16;
    int trunk_depth = 4;
    int dpt_features = 256;
    // vggt-omega text-alignment variant: build the TextAlignmentHead branch
    // (weights travel in the GGUF as f32/f16 extras alongside the flag)
    bool enable_text_alignment = false;
    int head_depth = 5;          // pi3 family TransformerDecoder blocks
    // DUSt3R pair-wise branch: the decoder dims differ from the encoder's
    // (ViT-L/16 encoder 1024/16, BaseDecoder 768/12). Default 0 so a GGUF
    // without the dust3r.dec_* keys hard-fails in Dust3rImpl::build_graph
    // instead of silently running someone else's dims.
    int dec_embed_dim = 0;
    int dec_num_heads = 0;
    int cached_layer_idx[4] = {4, 11, 17, 23};
    int register_attn_block_idx[5] = {2, 6, 9, 14, 20};

    float img_mean[3] = {0.485f, 0.456f, 0.406f};
    float img_std[3] = {0.229f, 0.224f, 0.225f};
};

struct GGUFModel {
    VGGTMeta meta;
    std::map<std::string, HostTensor> tensors;

    const HostTensor* tensor(const std::string& name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }
};

// Loads the whole GGUF (tensors staged on host). Returns nullptr on failure.
std::unique_ptr<GGUFModel> load_gguf(const std::string& path);

}  // namespace map_anything
