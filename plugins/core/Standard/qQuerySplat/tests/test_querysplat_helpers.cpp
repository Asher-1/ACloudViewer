// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// qQuerySplat helper unit tests: the catalog surface (rows/defaults/shared
// encoder delegation) and the C-API catalog/path contract. Asset-gated
// e2e inference lives in the AICore validation matrix, not here.
#include <aicore/asset_digests.h>
#include <aicore/querysplat_capi.h>
#include <gtest/gtest.h>

#include <QDir>
#include <QFileInfo>

#include "QuerySplatModelCatalog.h"

TEST(QuerySplatCatalog, RowsArePublished) {
    const auto models = QuerySplatHelpers::catalogModels();
    ASSERT_EQ(models.size(), 4);  // decoder f32/f16/q8_0 + lpips
    int decoders = 0;
    int lpips = 0;
    for (const auto& m : models) {
        EXPECT_FALSE(m.filename.isEmpty());
        EXPECT_TRUE(m.downloadUrl.startsWith(QStringLiteral(
                "https://huggingface.co/Asher-1/querysplat_vggto")))
                << m.downloadUrl.toStdString();
        if (m.role == QStringLiteral("decoder")) ++decoders;
        if (m.role == QStringLiteral("lpips")) ++lpips;
        // every published row carries a pinned digest
        EXPECT_NE(aicore::AssetDigestForFile(m.filename.toUtf8().constData()),
                  nullptr)
                << m.filename.toStdString();
    }
    EXPECT_EQ(decoders, 3);
    EXPECT_EQ(lpips, 1);
}

TEST(QuerySplatCatalog, DecoderOrderPutsQ8First) {
    const auto decoders = QuerySplatHelpers::decoderModels();
    ASSERT_EQ(decoders.size(), 3);
    EXPECT_EQ(decoders[0].filename,
              QStringLiteral("querysplat-paper-q8_0.gguf"));
    EXPECT_EQ(QuerySplatHelpers::defaultDecoderIndex(), 0);
}

TEST(QuerySplatCatalog, DefaultEncoderDelegatesToMapAnything) {
    const QString enc = QuerySplatHelpers::encoderDefaultPath();
    EXPECT_TRUE(enc.contains(QStringLiteral("vggt-omega-1b-512")))
            << enc.toStdString();
    // the shared encoder resolves through the map_anything cache root
    EXPECT_TRUE(enc.contains(QStringLiteral("map_anything_models")))
            << enc.toStdString();
}

TEST(QuerySplatCatalog, ResolveCachePath) {
    if (char* p = aicore_querysplat_model_resolve_cache_path(
                "querysplat-paper-q8_0.gguf")) {
        const QString path = QString::fromUtf8(p);
        aicore_querysplat_free_buffer(p);
        EXPECT_TRUE(path.contains(QStringLiteral("querysplat_models")));
    } else {
        GTEST_SKIP() << "cache path resolution unavailable in this env";
    }
}

TEST(QuerySplatCapiContract, AbiVersionAndCatalog) {
    EXPECT_EQ(aicore_querysplat_abi_version(), 1);
    EXPECT_EQ(aicore_querysplat_model_count(), 4);
    ASSERT_NE(aicore_querysplat_model_at(0), nullptr);
    EXPECT_EQ(aicore_querysplat_model_at(4), nullptr);
    // role defaults: decoder/encoder/lpips resolve, unknown role does not
    EXPECT_NE(aicore_querysplat_model_default("decoder"), nullptr);
    EXPECT_NE(aicore_querysplat_model_default("encoder"), nullptr);
    EXPECT_NE(aicore_querysplat_model_default("lpips"), nullptr);
    EXPECT_EQ(aicore_querysplat_model_default("nope"), nullptr);
}

TEST(QuerySplatCapiContract, OptionsRejectInvalidValues) {
    aicore_querysplat_options* opts = aicore_querysplat_options_new();
    ASSERT_NE(opts, nullptr);
    // invalid values keep the previous state (documented contract)
    aicore_querysplat_options_set_tto_steps(opts, -3);
    aicore_querysplat_options_set_opacity_threshold(opts, 42.0f);
    aicore_querysplat_options_set_log_level(opts, 9);
    aicore_querysplat_options_set_image_size(opts, -1, -1);
    aicore_querysplat_options_free(opts);
}

TEST(QuerySplatCapiContract, NotReadyContextCarriesError) {
    // a context whose paths do not exist loads as not-ready with last_error
    // set (never a crash); the C contract keeps the pointer non-NULL.
    aicore_querysplat_options* opts = aicore_querysplat_options_new();
    ASSERT_NE(opts, nullptr);
    aicore_querysplat_options_set_device(opts, "cpu");
    aicore_querysplat_ctx* ctx = aicore_querysplat_load_opts(
            "/nonexistent/encoder.gguf", "/nonexistent/decoder.gguf", opts);
    aicore_querysplat_options_free(opts);
    if (ctx) {
        EXPECT_EQ(aicore_querysplat_is_ready(ctx), 0);
        // every result accessor returns 0/NULL on a not-ready context
        EXPECT_EQ(aicore_querysplat_result_view_count(ctx), 0);
        EXPECT_EQ(aicore_querysplat_result_gaussians(ctx), nullptr);
        EXPECT_EQ(aicore_querysplat_default_image_size(ctx), 0);
        aicore_querysplat_free(ctx);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
