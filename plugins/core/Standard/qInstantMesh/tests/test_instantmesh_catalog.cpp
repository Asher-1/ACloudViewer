// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// qInstantMesh catalog helper tests — pure logic, no GGUF assets required.

#include <gtest/gtest.h>

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

#include "InstantMeshModelCatalog.h"
#include "ecvAssetIntegrity.h"

namespace {

// Minimal GGUF-shaped stand-in for the cheap-check path (magic + 64 KiB
// minimum body); no real model bytes are needed for presence validation.
void writeGgufShapedFile(const QString& path) {
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.write("GGUF", 4);
    f.write(QByteArray(64 * 1024, '\0'));
    f.close();
}

TEST(InstantMeshCatalog, RmbgValidityUsesPinnedDigestNotCatalogLookup) {
    // RMBG rows are deliberately absent from the InstantMesh catalog (the
    // RMBG task owns them). The validity check must fall back to the global
    // pinned digest table; treating the catalog lookup miss as "invalid"
    // made every cached RMBG model look missing and re-download on each
    // dialog open (regression: ACloudViewer.20261104-113626 log).
    QTemporaryDir tmp;
    const QString name = QStringLiteral("rmbg_f16.gguf");
    const QString path = tmp.filePath(name);
    writeGgufShapedFile(path);
    ASSERT_FALSE(ecvAssetIntegrity::PinnedDigest(name).isEmpty());
    // No ledger yet: presence + GGUF magic acceptance (RMBG publishes no
    // size, so there is no exact-size fallback).
    EXPECT_TRUE(InstantMeshHelpers::isValidModelFile(path, name));

    // A recorded verification (size + mtime trust) stays accepted.
    ASSERT_TRUE(ecvAssetIntegrity::markVerified(
            path, {QCryptographicHash::Sha256,
                   ecvAssetIntegrity::PinnedDigest(name)}));
    EXPECT_TRUE(InstantMeshHelpers::isValidModelFile(path, name));

    // Non-GGUF payloads are rejected before any trust decision.
    const QString junk = tmp.filePath(QStringLiteral("rmbg_q8.gguf"));
    QFile j(junk);
    ASSERT_TRUE(j.open(QIODevice::WriteOnly));
    j.write("JUNKJUNK", 8);
    j.write(QByteArray(64 * 1024, '\0'));
    j.close();
    EXPECT_FALSE(InstantMeshHelpers::isValidModelFile(
            junk, QStringLiteral("rmbg_q8.gguf")));

    // Unknown instantmesh-catalog names still fail.
    EXPECT_FALSE(InstantMeshHelpers::isValidModelFile(
            path, QStringLiteral("not_in_catalog_f16.gguf")));
}

TEST(InstantMeshCatalog, PresetShapes) {
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    ASSERT_EQ(presets.size(), 2);
    EXPECT_EQ(presets[0].files.size(), 6);  // full: cond/unet/vae/dino/tr/syn
    EXPECT_EQ(presets[1].files.size(), 6);  // geometry: 3 empty + 3 models
    // Geometry preset leaves the Zero123++ slots empty in position.
    EXPECT_TRUE(presets[1].files[0].isEmpty());
    EXPECT_TRUE(presets[1].files[1].isEmpty());
    EXPECT_TRUE(presets[1].files[2].isEmpty());
}

TEST(InstantMeshCatalog, PresetsUseCanonicalF16Names) {
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    for (const InstantMeshPreset& p : presets) {
        for (const QString& f : p.files) {
            if (f.isEmpty()) continue;
            // The Zero123++ cond constants only publish f32; everything else
            // in the presets names the canonical f16 GGUFs.
            if (f.startsWith(QStringLiteral("zero123pp_cond_"))) {
                EXPECT_EQ(f, QStringLiteral("zero123pp_cond_f32.gguf"));
            } else {
                EXPECT_TRUE(f.contains(QStringLiteral("_f16")))
                        << "preset file not f16: " << f.toStdString();
            }
        }
    }
}

TEST(InstantMeshCatalog, ResolveFullPresetF16Chain) {
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = InstantMeshHelpers::resolvePresetFiles(
            presets[0], tmp.path(), QStringLiteral("f16"));
    ASSERT_EQ(paths.size(), 6);
    // f16 chain: cond stays f32 (no f16 published), unet/vae/dino/tr/syn f16.
    EXPECT_TRUE(paths[0].endsWith(QStringLiteral("zero123pp_cond_f32.gguf")));
    EXPECT_TRUE(paths[1].endsWith(QStringLiteral("zero123pp_unet_f16.gguf")));
    EXPECT_TRUE(paths[2].endsWith(QStringLiteral("zero123pp_vae_f16.gguf")));
    EXPECT_TRUE(paths[3].endsWith(QStringLiteral("dino_f16.gguf")));
    EXPECT_TRUE(paths[4].endsWith(QStringLiteral("lrm_transformer_f16.gguf")));
    EXPECT_TRUE(paths[5].endsWith(QStringLiteral("synthesizer_f16.gguf")));
}

TEST(InstantMeshCatalog, ResolveQ8ChainKeepsZero123ppAt16) {
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = InstantMeshHelpers::resolvePresetFiles(
            presets[0], tmp.path(), QStringLiteral("q8"));
    ASSERT_EQ(paths.size(), 6);
    // The Zero123++ trio has no q8 release: cond stays f32, unet/vae stay
    // f16; the InstantMesh trio takes q8.
    EXPECT_TRUE(paths[0].endsWith(QStringLiteral("zero123pp_cond_f32.gguf")));
    EXPECT_TRUE(paths[1].endsWith(QStringLiteral("zero123pp_unet_f16.gguf")));
    EXPECT_TRUE(paths[2].endsWith(QStringLiteral("zero123pp_vae_f16.gguf")));
    EXPECT_TRUE(paths[3].endsWith(QStringLiteral("dino_q8.gguf")));
    EXPECT_TRUE(paths[4].endsWith(QStringLiteral("lrm_transformer_q8.gguf")));
    EXPECT_TRUE(paths[5].endsWith(QStringLiteral("synthesizer_q8.gguf")));
}

TEST(InstantMeshCatalog, ResolveGeometryPresetKeepsPlaceholders) {
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = InstantMeshHelpers::resolvePresetFiles(
            presets[1], tmp.path(), QStringLiteral("f16"));
    ASSERT_EQ(paths.size(), 6);
    // The first three slots (cond/unet/vae) must stay empty so the caller's
    // index map onto aicore_instantmesh_model_paths does not shift.
    EXPECT_TRUE(paths[0].isEmpty());
    EXPECT_TRUE(paths[1].isEmpty());
    EXPECT_TRUE(paths[2].isEmpty());
    EXPECT_TRUE(paths[3].endsWith(QStringLiteral("dino_f16.gguf")));
}

TEST(InstantMeshCatalog, CatalogCoversAllPresetFiles) {
#ifdef AICore_ENABLED
    const QVector<InstantMeshModelEntry> catalog =
            InstantMeshHelpers::catalogModels();
    ASSERT_FALSE(catalog.isEmpty());
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    for (const InstantMeshPreset& p : presets) {
        for (const QString& f : p.files) {
            if (f.isEmpty()) continue;
            InstantMeshModelEntry e;
            EXPECT_TRUE(InstantMeshHelpers::findModelByFilename(f, &e))
                    << "catalog misses preset file: " << f.toStdString();
        }
    }
    // 14 published files: 3 x (dino, transformer, synthesizer) + 5 Zero123++.
    EXPECT_EQ(catalog.size(), 14);
    // RMBG is intentionally absent (single owner: the in-tree RMBG catalog).
    for (const InstantMeshModelEntry& e : catalog) {
        EXPECT_FALSE(e.filename.startsWith(QStringLiteral("rmbg")))
                << "RMBG must stay in the RMBG task catalog";
    }
#else
    GTEST_SKIP() << "built without AICore";
#endif
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
