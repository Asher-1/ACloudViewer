// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#include "model_loader.hpp"
int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");
    if (!gguf) {
        std::fprintf(stderr, "DA_TEST_GGUF unset; skipping\n");
        return 77;
    }
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }
    const auto& c = ml.config();
    bool ok = c.embed_dim == 768 && c.depth == 12 && c.num_heads == 12 &&
              c.head_dim == 64 && c.alt_start == 4 && c.rope_start == 4 &&
              c.qknorm_start == 4 && c.out_layers.size() == 4;
    ok = ok && ml.tensor("vit.patch_embed.weight") != nullptr;
    ok = ok && ml.tensor("vit.blk.0.attn_qkv.weight") != nullptr;
    ok = ok && ml.tensor("vit.blk.11.mlp_fc2.weight") != nullptr;
    std::fprintf(stderr, "embed=%u depth=%u heads=%u -> %s\n", c.embed_dim,
                 c.depth, c.num_heads, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
