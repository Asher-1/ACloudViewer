// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// qTrellis catalog helper tests — pure logic, no GGUF assets required.

#include <gtest/gtest.h>

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include "TrellisModelCatalog.h"
#include "ecvModelDownloader.h"

namespace {

TEST(TrellisCatalog, PresetShapes) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    ASSERT_EQ(presets.size(), 3);
    EXPECT_EQ(presets[0].files.size(), 3);   // coarse (dino/ss_flow/ss_dec)
    EXPECT_EQ(presets[1].files.size(), 10);  // 512 + PBR (slat_hr/tex_hr = "")
    EXPECT_EQ(presets[2].files.size(), 10);  // 1024 + PBR
}

TEST(TrellisCatalog, PresetsDefaultToF16) {
    // Upstream recommends f16 precision: every default preset file must be
    // an f16 variant (q8 stays selectable via the dialog's variant combos).
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    for (const TrellisPreset& p : presets) {
        for (const QString& f : p.files) {
            if (f.isEmpty()) continue;
            EXPECT_TRUE(f.contains(QStringLiteral("_f16")))
                    << "preset file not f16: " << f.toStdString();
        }
    }
}

TEST(TrellisCatalog, ResolvePresetFiles) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = TrellisHelpers::resolvePresetFiles(
            presets[1], tmp.path(), QStringLiteral("dino_q8"),
            QStringLiteral("ss_dec_f16"));
    // File lists keep aicore_trellis_model_paths field order: the 512 preset
    // leaves the 1024 slots (index 4 = slat_hr_flow, 9 = tex_flow_hr) as
    // empty placeholders so later fields do not shift.
    ASSERT_EQ(paths.size(), 10);
    EXPECT_TRUE(paths[0].endsWith(QStringLiteral("dino_q8.gguf")));
    EXPECT_TRUE(paths[1].endsWith(QStringLiteral("ss_flow_f16.gguf")));
    EXPECT_TRUE(paths[2].endsWith(QStringLiteral("ss_dec_f16.gguf")));
    EXPECT_TRUE(paths[3].endsWith(QStringLiteral("slat_flow_f16.gguf")));
    EXPECT_TRUE(paths[4].isEmpty());
    EXPECT_TRUE(paths[5].endsWith(QStringLiteral("shape_dec_f16.gguf")));
    EXPECT_TRUE(
            paths[8].endsWith(QStringLiteral("tex_slat_flow_512_f16.gguf")));
    EXPECT_TRUE(paths[9].isEmpty());
    // Every non-placeholder path lives under the cache dir.
    for (const QString& p : paths) {
        if (!p.isEmpty()) {
            EXPECT_TRUE(p.startsWith(tmp.path()));
        }
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

TEST(TrellisCatalog, HfMirror) {
    // The f16 flow DiTs exceed GitHub's 2 GB release limit; they must be
    // published on the HF mirror with their exact LFS size and SHA-256.
    HfModelInfo info;
    ASSERT_TRUE(TrellisHelpers::hfModelInfo("ss_flow_f16.gguf", &info));
    EXPECT_EQ(info.sizeBytes, 2615168864LL);
    // HF LFS content fingerprint (oid) — the value the downloader stream-
    // checks against.
    EXPECT_EQ(info.sha256,
              QStringLiteral("1dded5b74237d24e6876a642a26f90b43742e355441857"
                             "3860f810e3bbe61e8c"));
    EXPECT_EQ(TrellisHelpers::hfDownloadUrl("ss_flow_f16.gguf"),
              QStringLiteral("https://huggingface.co/Asher-1/Trellis2-models/"
                             "resolve/main/ss_flow_f16.gguf"));
    // Unknown files are not on the mirror.
    EXPECT_FALSE(TrellisHelpers::hfModelInfo("nope.gguf", nullptr));
    EXPECT_TRUE(TrellisHelpers::hfDownloadUrl("nope.gguf").isEmpty());
    // Every preset file must be published on the mirror, each with a
    // well-formed SHA-256 (64 hex chars).
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    for (const TrellisPreset& p : presets) {
        for (const QString& f : p.files) {
            if (f.isEmpty()) continue;
            HfModelInfo m;
            ASSERT_TRUE(TrellisHelpers::hfModelInfo(f, &m))
                    << "missing on HF mirror: " << f.toStdString();
            EXPECT_EQ(m.sha256.size(), 64)
                    << "bad SHA-256 for " << f.toStdString();
            for (const QChar& c : m.sha256) {
                EXPECT_TRUE(c.isDigit() ||
                            (c >= QLatin1Char('a') && c <= QLatin1Char('f')))
                        << "non-hex char in SHA-256 for " << f.toStdString();
            }
        }
    }
    // q8 alternatives and the RMBG set stay available too.
    EXPECT_TRUE(TrellisHelpers::hfModelInfo("dino_q8.gguf", nullptr));
    EXPECT_TRUE(TrellisHelpers::hfModelInfo("ss_dec_q8.gguf", nullptr));
    EXPECT_TRUE(TrellisHelpers::hfModelInfo("rmbg_f16.gguf", nullptr));
    EXPECT_TRUE(TrellisHelpers::hfModelInfo("rmbg_q8.gguf", nullptr));
}

TEST(TrellisCatalog, DeployedFileValidation) {
    QTemporaryDir tmp;
    // A GGUF that exists but is the wrong size must fail validation: this is
    // how truncated downloads / stale manual deployments are caught.
    const QString path = tmp.filePath(QStringLiteral("dino_f16.gguf"));
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("GGUF", 4);  // correct magic, wrong size
        f.close();
    }
    EXPECT_FALSE(TrellisHelpers::isValidModelFile(
            path, QStringLiteral("dino_f16.gguf")));
    // An unknown filename falls back to the generic GGUF check (still fails
    // here because the file is far below the 64 KiB floor).
    EXPECT_FALSE(TrellisHelpers::isValidModelFile(
            path, QStringLiteral("not_on_mirror.gguf")));
    // Deep SHA-256 verification also rejects wrong-size files (and unknown
    // filenames outright).
    EXPECT_FALSE(TrellisHelpers::verifyModelFileSha256(
            path, QStringLiteral("dino_f16.gguf")));
    EXPECT_FALSE(TrellisHelpers::verifyModelFileSha256(
            path, QStringLiteral("not_on_mirror.gguf")));
}

TEST(TrellisCatalog, ExpectedHashValidation) {
    // ecvModelDownloader content-level checks: expectedSize + expectedSha256.
    QTemporaryDir tmp;
    const QString path = tmp.filePath(QStringLiteral("model.gguf"));
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("GGUF", 4);
        f.write(QByteArray(4096, 'x'));
        f.close();
    }
    const QByteArray good =
            QCryptographicHash::hash(QByteArray("GGUF") + QByteArray(4096, 'x'),
                                     QCryptographicHash::Sha256)
                    .toHex();
    const QByteArray bad =
            QByteArray("00").repeated(32);  // valid hex, wrong value
    // Size + hash both match -> valid; wrong hash -> invalid; empty hash
    // skips the content check (backwards compatible).
    EXPECT_TRUE(
            ecvModelDownloader::isValidCachedFile(path, 1, true, 4100, good));
    EXPECT_FALSE(
            ecvModelDownloader::isValidCachedFile(path, 1, true, 4100, bad));
    EXPECT_TRUE(ecvModelDownloader::isValidCachedFile(path, 1, true, 4100));
    EXPECT_TRUE(ecvModelDownloader::isValidCachedFile(path, 1, true, 0));
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
