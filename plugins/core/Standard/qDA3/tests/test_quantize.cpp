// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <string>

#include "ggml.h"
#include "model_loader.hpp"
#include "quantize.hpp"
int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");
    if (!gguf) return 77;
    std::string out = std::string(gguf) + ".q8test.gguf";
    if (!aicore::depth::quantize_gguf(gguf, out, "q8_0")) {
        std::fprintf(stderr, "quantize failed\n");
        return 1;
    }
    aicore::depth::ModelLoader ml;
    if (!ml.load(out)) {
        std::fprintf(stderr, "load quantized failed\n");
        std::remove(out.c_str());
        return 1;
    }
    const auto& c = ml.config();
    bool ok = c.embed_dim == 768 && c.depth == 12;
    ggml_tensor* qkv =
            ml.tensor("vit.blk.0.attn_qkv.weight");  // should be quantized
    ggml_tensor* conv = ml.tensor("vit.patch_embed.weight");  // should stay f32
    ok = ok && qkv && conv && qkv->type == GGML_TYPE_Q8_0 &&
         conv->type == GGML_TYPE_F32;
    std::fprintf(stderr, "quantize: qkv=%s conv=%s -> %s\n",
                 qkv ? ggml_type_name(qkv->type) : "null",
                 conv ? ggml_type_name(conv->type) : "null",
                 ok ? "OK" : "FAIL");
    std::remove(out.c_str());
    return ok ? 0 : 1;
}
