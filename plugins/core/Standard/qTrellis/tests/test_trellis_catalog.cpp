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
#include <QFileInfo>
#include <QTemporaryDir>

#include "TrellisModelCatalog.h"
#include "ecvAssetIntegrity.h"
#include "ecvModelDownloader.h"

namespace {

TEST(TrellisCatalog, PresetShapes) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    ASSERT_EQ(presets.size(), 3);
    EXPECT_EQ(presets[0].files.size(), 3);   // coarse (dino/ss_flow/ss_dec)
    EXPECT_EQ(presets[1].files.size(), 10);  // 512 + PBR (slat_hr/tex_hr = "")
    EXPECT_EQ(presets[2].files.size(), 10);  // 1024 + PBR
}

TEST(TrellisCatalog, PresetsUseCanonicalF16Names) {
    // The preset file lists name the canonical f16 GGUFs;
    // resolvePresetFiles() substitutes the q8 variants per the selected
    // quantization chain. The f16 names are therefore the stable contract
    // between the presets and the HF-mirror table.
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    for (const TrellisPreset& p : presets) {
        for (const QString& f : p.files) {
            if (f.isEmpty()) continue;
            EXPECT_TRUE(f.contains(QStringLiteral("_f16")))
                    << "preset file not f16: " << f.toStdString();
        }
    }
}

TEST(TrellisCatalog, ResolvePresetFilesQ8Chain) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = TrellisHelpers::resolvePresetFiles(
            presets[1], tmp.path(), QStringLiteral("q8"));
    // File lists keep aicore_trellis_model_paths field order: the 512 preset
    // leaves the 1024 slots (index 4 = slat_hr_flow, 9 = tex_flow_hr) as
    // empty placeholders so later fields do not shift.
    //
    // q8 chain (upstream 2026-08-30 CUDA+Vulkan e2e experiment): ONLY
    // ss_dec takes the q8 substitution — the chaotic samplers (dino cond /
    // ss_flow / slat_flow) and the three VAE stages all stay f16 (q8 cond
    // error is amplified into different voxel sets; shape_dec q8 flips the
    // level-0 subdivision logits, tex_dec q8 collapses the PBR material).
    ASSERT_EQ(paths.size(), 10);
    EXPECT_TRUE(paths[0].endsWith(QStringLiteral("dino_f16.gguf")));
    EXPECT_TRUE(paths[1].endsWith(QStringLiteral("ss_flow_f16.gguf")));
    EXPECT_TRUE(paths[2].endsWith(QStringLiteral("ss_dec_q8.gguf")));
    EXPECT_TRUE(paths[3].endsWith(QStringLiteral("slat_flow_f16.gguf")));
    EXPECT_TRUE(paths[4].isEmpty());
    EXPECT_TRUE(paths[5].endsWith(QStringLiteral("shape_dec_f16.gguf")));
    EXPECT_TRUE(paths[6].endsWith(QStringLiteral("shape_enc_f16.gguf")));
    EXPECT_TRUE(paths[7].endsWith(QStringLiteral("tex_dec_f16.gguf")));
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

TEST(TrellisCatalog, ResolvePresetFilesF16Chain) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = TrellisHelpers::resolvePresetFiles(
            presets[0], tmp.path(), QStringLiteral("f16"));
    ASSERT_EQ(paths.size(), 3);
    EXPECT_TRUE(paths[0].endsWith(QStringLiteral("dino_f16.gguf")));
    EXPECT_TRUE(paths[1].endsWith(QStringLiteral("ss_flow_f16.gguf")));
    EXPECT_TRUE(paths[2].endsWith(QStringLiteral("ss_dec_f16.gguf")));
}

TEST(TrellisCatalog, ResolvePresetFilesF32Chain) {
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    QTemporaryDir tmp;
    const QStringList paths = TrellisHelpers::resolvePresetFiles(
            presets[1], tmp.path(), QStringLiteral("f32"));
    // f32 (exact) chain: the chaotic chain upgrades to full-f32 weights;
    // shape_enc / tex_dec / the texture flows stay f16 (no f32 texture
    // GGUFs — texture noise affects appearance, not the voxel set).
    ASSERT_EQ(paths.size(), 10);
    EXPECT_TRUE(paths[0].endsWith(QStringLiteral("dino_f32.gguf")));
    EXPECT_TRUE(paths[1].endsWith(QStringLiteral("ss_flow_f32.gguf")));
    EXPECT_TRUE(paths[2].endsWith(QStringLiteral("ss_dec_f32.gguf")));
    EXPECT_TRUE(paths[3].endsWith(QStringLiteral("slat_flow_f32.gguf")));
    EXPECT_TRUE(paths[5].endsWith(QStringLiteral("shape_dec_f32.gguf")));
    EXPECT_TRUE(paths[6].endsWith(QStringLiteral("shape_enc_f16.gguf")));
    EXPECT_TRUE(paths[7].endsWith(QStringLiteral("tex_dec_f16.gguf")));
    EXPECT_TRUE(
            paths[8].endsWith(QStringLiteral("tex_slat_flow_512_f16.gguf")));
}

TEST(TrellisCatalog, PrecisionSensitiveDecodersStayF16) {
    EXPECT_TRUE(TrellisHelpers::isPrecisionSensitiveDecoder(
            QStringLiteral("shape_dec_f16.gguf")));
    EXPECT_TRUE(TrellisHelpers::isPrecisionSensitiveDecoder(
            QStringLiteral("tex_dec_f16.gguf")));
    EXPECT_FALSE(TrellisHelpers::isPrecisionSensitiveDecoder(
            QStringLiteral("ss_flow_f16.gguf")));
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

TEST(TrellisCatalog, IntegrityAnchorVerification) {
    // ecvAssetIntegrity one-shot content verification (verifyNow).
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
    const ecvAssetIntegrity::Anchor goodAnchor{QCryptographicHash::Sha256,
                                               good};
    const ecvAssetIntegrity::Anchor badAnchor{QCryptographicHash::Sha256, bad};
    // Digest match -> verified; wrong digest -> rejected; an anchor without
    // a digest has nothing to compare against.
    EXPECT_TRUE(ecvAssetIntegrity::verifyNow(path, goodAnchor));
    EXPECT_FALSE(ecvAssetIntegrity::verifyNow(path, badAnchor));
    EXPECT_FALSE(ecvAssetIntegrity::verifyNow(path, {}));
}

TEST(TrellisCatalog, IntegrityLedgerStatTrust) {
    // Verify-once-at-ingestion + stat-trusted access: after a DeepVerify
    // pass records the ledger, the access-path check answers from stat
    // metadata alone.
    QTemporaryDir tmp;
    const QString path = tmp.filePath(QStringLiteral("model.gguf"));
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("GGUF", 4);
        f.write(QByteArray(4096, 'x'));
        f.close();
    }
    const ecvAssetIntegrity::Anchor anchor{
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(QByteArray("GGUF") +
                                             QByteArray(4096, 'x'),
                                     QCryptographicHash::Sha256)
                    .toHex()};
    // Ledger miss + DeepVerify self-heals and records the state.
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(
            path, anchor, 0, false, ecvAssetIntegrity::OnMiss::DeepVerify));
    // Second access is stat-trusted (same answer, no hashing).
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, anchor));
    // Tampering with the content breaks the stat match; a DeepVerify
    // re-check must reject the file.
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::Append));
        f.write("tamper", 6);
        f.close();
    }
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(
            path, anchor, 0, false, ecvAssetIntegrity::OnMiss::DeepVerify));
}

TEST(TrellisCatalog, IntegrityLedgerPinBump) {
    // Changing the pinned digest in source must bust the ledger: a file
    // verified against the old pin may not be trusted for the new pin
    // (this is what forces a re-download when a release is republished).
    QTemporaryDir tmp;
    const QString path = tmp.filePath(QStringLiteral("model.gguf"));
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("GGUF", 4);
        f.write(QByteArray(4096, 'x'));
        f.close();
    }
    const ecvAssetIntegrity::Anchor oldPin{
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(QByteArray("GGUF") +
                                             QByteArray(4096, 'x'),
                                     QCryptographicHash::Sha256)
                    .toHex()};
    ASSERT_TRUE(ecvAssetIntegrity::markVerified(path, oldPin));
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, oldPin));

    const QByteArray newDigest =
            QByteArray("11").repeated(32);  // valid hex, different pin
    const ecvAssetIntegrity::Anchor newPin{QCryptographicHash::Sha256,
                                           newDigest};
    // The current file generation was verified against the old pin, so the
    // bytes cannot match the new pin: rejected immediately, without any
    // hashing, and the old evidence stays on disk.
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(path, newPin));
    EXPECT_TRUE(QFileInfo::exists(path + QStringLiteral(".cvintegrity")));
}

TEST(TrellisCatalog, IntegrityLedgerEmptyPinPreservesPinnedEntry) {
    // An empty-pin query (presence checks that only do junk/size checks)
    // must trust — not destroy — a ledger entry recorded against a pinned
    // digest; otherwise the byte-level evidence would be lost on the first
    // dialog refresh and every later check would fall back to cheap checks.
    QTemporaryDir tmp;
    const QString path = tmp.filePath(QStringLiteral("model.gguf"));
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("GGUF", 4);
        f.write(QByteArray(4096, 'x'));
        f.close();
    }
    const ecvAssetIntegrity::Anchor pin{
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(QByteArray("GGUF") +
                                             QByteArray(4096, 'x'),
                                     QCryptographicHash::Sha256)
                    .toHex()};
    ASSERT_TRUE(ecvAssetIntegrity::markVerified(path, pin));
    // Empty-pin presence query: trusted from the pinned evidence.
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, {}));
    // The pinned evidence survives the query.
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, pin));
    EXPECT_TRUE(QFileInfo::exists(path + QStringLiteral(".cvintegrity")));
}

TEST(TrellisCatalog, IntegrityLedgerSizeOnlyEntries) {
    // Anchor-less assets (no content pin, e.g. GitHub-release GGUFs) get a
    // size-only ledger entry recorded at ingestion; access checks trust the
    // stat until the file changes, and the catalog's exact size guards the
    // no-ledger path.
    QTemporaryDir tmp;
    const QString path = tmp.filePath(QStringLiteral("model.gguf"));
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("GGUF", 4);
        f.write(QByteArray(4096, 'x'));
        f.close();
    }
    const ecvAssetIntegrity::Anchor noPin;  // empty digest
    ASSERT_TRUE(ecvAssetIntegrity::markVerified(path, noPin));
    EXPECT_TRUE(ecvAssetIntegrity::isVerified(path, noPin));
    // Same path, bigger content: stat no longer matches the entry; the
    // catalog-style exact-size guard rejects it too.
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::Append));
        f.write("extra", 5);
        f.close();
    }
    EXPECT_FALSE(ecvAssetIntegrity::isVerified(
            path, noPin, 0, false,
            ecvAssetIntegrity::OnMiss::CheapChecksOnly, 4100));
}

TEST(TrellisCatalog, RemoveIfNotVerified) {
    QTemporaryDir tmp;
    const QString path = tmp.filePath(QStringLiteral("junk.gguf"));
    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("GGUF", 4);
        f.write(QByteArray(4096, 'x'));
        f.close();
    }
    // Right magic but below the size floor: not verified -> removed.
    ecvAssetIntegrity::removeIfNotVerified(path, {}, 8192, true);
    EXPECT_FALSE(QFileInfo::exists(path));
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
