// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Asset-free: synthesize a KV-only "free-splatter" GGUF in memory, write it,
// then load it back through model_file and assert every hyperparameter round-
// trips. Proves the converter's metadata contract and the loader agree.
#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <filesystem>
#include <string>

#include "gguf_loader.h"

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                         \
            failures++;                                                  \
        }                                                                \
    } while (0)

static std::string write_synth_gguf() {
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "free-splatter");
    gguf_set_val_u32(g, "free-splatter.block_count", 24);
    gguf_set_val_u32(g, "free-splatter.embedding_length", 1024);
    gguf_set_val_u32(g, "free-splatter.feed_forward_length", 4096);
    gguf_set_val_u32(g, "free-splatter.attention.head_count", 16);
    gguf_set_val_u32(g, "free-splatter.attention.key_length", 64);
    gguf_set_val_u32(g, "free-splatter.vision.patch_size", 8);
    gguf_set_val_u32(g, "free-splatter.vision.image_size", 512);
    gguf_set_val_u32(g, "free-splatter.vision.in_channels", 3);
    gguf_set_val_u32(g, "free-splatter.vision.gaussian_channels", 23);
    gguf_set_val_u32(g, "free-splatter.sh_degree", 1);
    gguf_set_val_bool(g, "free-splatter.sh_residual", true);
    gguf_set_val_bool(g, "free-splatter.use_2dgs", false);
    gguf_set_val_f32(g, "free-splatter.attention.layer_norm_epsilon", 1e-5f);
    gguf_set_val_f32(g, "free-splatter.scale_min_act", 1e-4f);
    gguf_set_val_f32(g, "free-splatter.scale_max_act", 0.02f);

    const std::string path = (std::filesystem::temp_directory_path() /
                              "free_splatter_test_loader.gguf")
                                     .string();
    gguf_write_to_file(g, path.c_str(), /*only_meta =*/true);
    gguf_free(g);
    return path;
}

int main() {
    const std::string path = write_synth_gguf();

    aicore::gaussian::model_file mf;
    CHECK(mf.open(path, /*with_data =*/false));
    if (!mf.error.empty())
        std::fprintf(stderr, "open error: %s\n", mf.error.c_str());

    const aicore::gaussian::hparams& hp = mf.hp;
    CHECK(hp.n_layer == 24);
    CHECK(hp.n_embd == 1024);
    CHECK(hp.n_ff == 4096);
    CHECK(hp.n_head == 16);
    CHECK(hp.head_dim == 64);
    CHECK(hp.patch_size == 8);
    CHECK(hp.image_size == 512);
    CHECK(hp.in_channels == 3);
    CHECK(hp.gaussian_channels == 23);
    CHECK(hp.sh_degree == 1);
    CHECK(hp.sh_residual == true);
    CHECK(hp.use_2dgs == false);
    CHECK(hp.ln_eps == 1e-5f);
    CHECK(hp.tokens_per_view() == 4096);  // (512/8)^2

    // A wrong architecture string must be rejected.
    {
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "not-free-splatter");
        const std::string bad = (std::filesystem::temp_directory_path() /
                                 "free_splatter_bad.gguf")
                                        .string();
        gguf_write_to_file(g, bad.c_str(), true);
        gguf_free(g);
        aicore::gaussian::model_file m2;
        CHECK(!m2.open(bad, false));  // must fail
        std::filesystem::remove(bad);
    }

    std::filesystem::remove(path);
    std::printf(failures ? "test_loader: %d FAILURES\n" : "test_loader: ok\n",
                failures);
    return failures ? 1 : 0;
}
