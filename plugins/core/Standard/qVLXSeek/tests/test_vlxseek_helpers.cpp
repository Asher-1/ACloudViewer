// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Pure-logic unit tests for the qVLXSeek helpers (catalog bridge, task
// template coverage, batch schema option table). No GGUF model required.

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>

#include "VLXSeekModelCatalog.h"

namespace {

constexpr int kRoleWedetect = 0;
constexpr int kRoleVision = 1;
constexpr int kRoleLlm = 2;

}  // namespace

TEST(VLXSeekCatalog, RowCountAndRoles) {
    const QVector<VLXSeekModelEntry> rows = VLXSeekHelpers::catalogModels();
    ASSERT_EQ(rows.size(), 7);
    int wedetect = 0, vision = 0, llm = 0;
    for (const VLXSeekModelEntry& row : rows) {
        if (row.role == kRoleWedetect) ++wedetect;
        if (row.role == kRoleVision) ++vision;
        if (row.role == kRoleLlm) ++llm;
        EXPECT_FALSE(row.filename.isEmpty());
        EXPECT_TRUE(row.downloadUrl.startsWith(QStringLiteral("https://")));
        EXPECT_EQ(row.sha256.size(), 64);
        EXPECT_GT(row.sizeBytes, 0);
    }
    EXPECT_EQ(wedetect, 3);
    EXPECT_EQ(vision, 2);
    EXPECT_EQ(llm, 2);
}

TEST(VLXSeekCatalog, WedetectAccuracyRowIsF32) {
    // The detector accuracy rows are always f32 (the official .pth dtype);
    // quantized detector rows are smoke-only.
    for (const VLXSeekModelEntry& row : VLXSeekHelpers::catalogModels()) {
        if (row.role != kRoleWedetect) continue;
        if (row.quantization == QStringLiteral("f32")) {
            EXPECT_TRUE(row.displayName.contains(
                    QStringLiteral("required")));
        } else {
            EXPECT_TRUE(row.displayName.contains(
                    QStringLiteral("smoke only")));
        }
    }
}

TEST(VLXSeekCatalog, DefaultIndexIsRecommendedVisionRow) {
    const int def = VLXSeekHelpers::catalogDefaultIndex();
    ASSERT_GE(def, 0);
    const QVector<VLXSeekModelEntry> rows = VLXSeekHelpers::catalogModels();
    ASSERT_LT(def, rows.size());
    EXPECT_EQ(rows[def].role, kRoleVision);
    EXPECT_TRUE(rows[def].displayName.contains(QStringLiteral("recommended")));
}

TEST(VLXSeekCatalog, FindByFilenameRoundTrip) {
    VLXSeekModelEntry out;
    ASSERT_TRUE(VLXSeekHelpers::findModelByFilename(
            QStringLiteral("vlx-10b-llm-q8_0.gguf"), &out));
    EXPECT_EQ(out.role, kRoleLlm);
    EXPECT_EQ(out.quantization, QStringLiteral("q8_0"));
    EXPECT_FALSE(VLXSeekHelpers::findModelByFilename(
            QStringLiteral("no-such-file.gguf"), &out));
}

TEST(VLXSeekCatalog, CatalogJsonIsComplete) {
    const QJsonArray rows = VLXSeekHelpers::catalogJson();
    ASSERT_EQ(rows.size(), 7);
    const QJsonDocument doc(rows);
    EXPECT_TRUE(doc.toJson(QJsonDocument::Compact).contains("vlx-seek-1.5-10b"));
}

TEST(VLXSeekCatalog, CacheDirUnderDataRoot) {
    const QString cache = VLXSeekHelpers::modelCacheDir();
    if (cache.isEmpty()) {
        GTEST_SKIP() << "AICore data root unavailable";
    }
    EXPECT_TRUE(cache.contains(QStringLiteral("vlx_seek_models")));
}

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
