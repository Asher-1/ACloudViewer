// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// qTrellis catalog helper tests — pure logic, no GGUF assets required.

#include <gtest/gtest.h>

#include <QDir>
#include <QTemporaryDir>

#include "TrellisModelCatalog.h"

namespace {

TEST(TrellisCatalog, PresetShapes) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    ASSERT_EQ(presets.size(), 3);
    EXPECT_EQ(presets[0].files.size(), 3);   // coarse
    EXPECT_EQ(presets[1].files.size(), 8);   // 512 + PBR
    EXPECT_EQ(presets[2].files.size(), 10);  // 1024 + PBR
}

TEST(TrellisCatalog, ResolvePresetFiles) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = TrellisHelpers::resolvePresetFiles(
            presets[1], tmp.path(), QStringLiteral("dino_q8"),
            QStringLiteral("ss_dec_f16"));
    ASSERT_EQ(paths.size(), 8);
    EXPECT_TRUE(paths[0].endsWith(QStringLiteral("dino_q8.gguf")));
    EXPECT_TRUE(paths[1].endsWith(QStringLiteral("ss_flow_q8.gguf")));
    EXPECT_TRUE(paths[2].endsWith(QStringLiteral("ss_dec_f16.gguf")));
    // Every resolved path lives under the cache dir.
    for (const QString& p : paths) {
        EXPECT_TRUE(p.startsWith(tmp.path()));
    }
}

TEST(TrellisCatalog, VariantSelection) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = TrellisHelpers::resolvePresetFiles(
            presets[0], tmp.path(), QStringLiteral("dino_f16"),
            QStringLiteral("ss_dec_q8"));
    ASSERT_EQ(paths.size(), 3);
    EXPECT_TRUE(paths[0].endsWith(QStringLiteral("dino_f16.gguf")));
    EXPECT_TRUE(paths[2].endsWith(QStringLiteral("ss_dec_q8.gguf")));
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
