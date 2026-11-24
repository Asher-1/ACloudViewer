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
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>

#include "tasks/gaussian/gguf_loader.h"

class GaussianLoader : public ::testing::Test {
protected:
    void SetUp() override {
        const auto stamp =
                std::chrono::steady_clock::now().time_since_epoch().count();
        path = (std::filesystem::temp_directory_path() /
                (std::string("free_splatter_") +
                 ::testing::UnitTest::GetInstance()
                         ->current_test_info()
                         ->name() +
                 "_" + std::to_string(stamp) + ".gguf"))
                       .string();
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove(path, error);
        EXPECT_FALSE(error)
                << "Cannot remove synthetic GGUF: " << error.message();
    }

    std::string path;
};

TEST_F(GaussianLoader, RoundTripsSyntheticHyperparameters) {
    gguf_context* g = gguf_init_empty();
    ASSERT_NE(g, nullptr);
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

    gguf_write_to_file(g, path.c_str(), /*only_meta =*/true);
    gguf_free(g);
    ASSERT_TRUE(std::filesystem::is_regular_file(path)) << path;

    aicore::gaussian::model_file mf;
    ASSERT_TRUE(mf.open(path, /*with_data =*/false)) << mf.error;

    const aicore::gaussian::hparams& hp = mf.hp;
    EXPECT_EQ(hp.n_layer, 24);
    EXPECT_EQ(hp.n_embd, 1024);
    EXPECT_EQ(hp.n_ff, 4096);
    EXPECT_EQ(hp.n_head, 16);
    EXPECT_EQ(hp.head_dim, 64);
    EXPECT_EQ(hp.patch_size, 8);
    EXPECT_EQ(hp.image_size, 512);
    EXPECT_EQ(hp.in_channels, 3);
    EXPECT_EQ(hp.gaussian_channels, 23);
    EXPECT_EQ(hp.sh_degree, 1);
    EXPECT_TRUE(hp.sh_residual);
    EXPECT_FALSE(hp.use_2dgs);
    EXPECT_EQ(hp.ln_eps, 1e-5f);
    EXPECT_EQ(hp.tokens_per_view(), 4096);  // (512/8)^2
}

TEST_F(GaussianLoader, RejectsWrongArchitecture) {
    gguf_context* g = gguf_init_empty();
    ASSERT_NE(g, nullptr);
    gguf_set_val_str(g, "general.architecture", "not-free-splatter");
    gguf_write_to_file(g, path.c_str(), /*only_meta =*/true);
    gguf_free(g);
    ASSERT_TRUE(std::filesystem::is_regular_file(path)) << path;

    aicore::gaussian::model_file model;
    EXPECT_FALSE(model.open(path, /*with_data =*/false));
}
