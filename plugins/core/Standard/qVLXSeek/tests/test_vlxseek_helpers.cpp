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
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryFile>

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
            EXPECT_TRUE(row.displayName.contains(QStringLiteral("required")));
        } else {
            EXPECT_TRUE(row.displayName.contains(QStringLiteral("smoke only")));
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
    EXPECT_TRUE(
            doc.toJson(QJsonDocument::Compact).contains("vlx-seek-1.5-10b"));
}

TEST(VLXSeekCatalog, CacheDirUnderDataRoot) {
    const QString cache = VLXSeekHelpers::modelCacheDir();
    if (cache.isEmpty()) {
        GTEST_SKIP() << "AICore data root unavailable";
    }
    EXPECT_TRUE(cache.contains(QStringLiteral("vlx_seek_models")));
}

TEST(VLXSeekTargetIndexes, ParseValidLists) {
    QString error;
    const QVector<int32_t> targets =
            VLXSeekHelpers::parseTargetIndexes(QStringLiteral("2, 3"), &error);
    EXPECT_TRUE(error.isEmpty());
    ASSERT_EQ(targets.size(), 2);
    EXPECT_EQ(targets[0], 2);
    EXPECT_EQ(targets[1], 3);

    const QVector<int32_t> single =
            VLXSeekHelpers::parseTargetIndexes(QStringLiteral("7"), &error);
    EXPECT_TRUE(error.isEmpty());
    ASSERT_EQ(single.size(), 1);
    EXPECT_EQ(single[0], 7);
}

TEST(VLXSeekTargetIndexes, EmptyTextYieldsEmptyVector) {
    QString error;
    EXPECT_TRUE(
            VLXSeekHelpers::parseTargetIndexes(QString(), &error).isEmpty());
    EXPECT_TRUE(
            VLXSeekHelpers::parseTargetIndexes(QStringLiteral(" , "), &error)
                    .isEmpty());
    EXPECT_TRUE(error.isEmpty());
}

TEST(VLXSeekTargetIndexes, RejectsMalformedAndNegative) {
    QString error;
    EXPECT_TRUE(
            VLXSeekHelpers::parseTargetIndexes(QStringLiteral("2, x"), &error)
                    .isEmpty());
    EXPECT_FALSE(error.isEmpty());

    error.clear();
    EXPECT_TRUE(VLXSeekHelpers::parseTargetIndexes(QStringLiteral("-1"), &error)
                        .isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(VLXSeekExternalBoxes, ParsesUpstreamBboxListFormat) {
    QTemporaryFile file;
    ASSERT_TRUE(file.open());
    file.write(R"([[10.5, 20, 110, 130],[5, 6, 7, 8]])");
    file.close();

    QString error;
    const QVector<float> boxes =
            VLXSeekHelpers::parseExternalBoxes(file.fileName(), &error);
    EXPECT_TRUE(error.isEmpty());
    ASSERT_EQ(boxes.size(), 8);
    EXPECT_FLOAT_EQ(boxes[0], 10.5f);
    EXPECT_FLOAT_EQ(boxes[1], 20.0f);
    EXPECT_FLOAT_EQ(boxes[4], 5.0f);
}

TEST(VLXSeekExternalBoxes, RejectsWrongCoordCountAndNonNumeric) {
    QTemporaryFile bad(QDir::tempPath() + "/vlxbad-XXXXXX.json");
    ASSERT_TRUE(bad.open());
    bad.write(R"([[1, 2, 3]])");
    bad.close();
    QString error;
    EXPECT_TRUE(VLXSeekHelpers::parseExternalBoxes(bad.fileName(), &error)
                        .isEmpty());
    EXPECT_FALSE(error.isEmpty());

    QTemporaryFile nonNum(QDir::tempPath() + "/vlxbad-XXXXXX.json");
    ASSERT_TRUE(nonNum.open());
    nonNum.write(R"([["a", 2, 3, 4]])");
    nonNum.close();
    error.clear();
    EXPECT_TRUE(VLXSeekHelpers::parseExternalBoxes(nonNum.fileName(), &error)
                        .isEmpty());
    EXPECT_FALSE(error.isEmpty());

    error.clear();
    EXPECT_TRUE(VLXSeekHelpers::parseExternalBoxes(
                        QStringLiteral("/nonexistent/boxes.json"), &error)
                        .isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(VLXSeekExternalBoxes, RejectsOverCapLists) {
    QJsonArray rows;
    for (int i = 0; i < 101; ++i)
        rows.append(QJsonArray{i * 1.0, 0.0, i * 1.0 + 5.0, 5.0});
    QTemporaryFile file(QDir::tempPath() + "/vlxcap-XXXXXX.json");
    ASSERT_TRUE(file.open());
    file.write(QJsonDocument(rows).toJson());
    file.close();

    QString error;
    EXPECT_TRUE(VLXSeekHelpers::parseExternalBoxes(file.fileName(), &error)
                        .isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(VLXSeekOcrProposer, PresetResolvesPublishedCatalogAssets) {
    const VLXSeekHelpers::OcrProposerModels models =
            VLXSeekHelpers::ocrProposerModels();
    if (models.cacheDir.isEmpty()) {
        GTEST_SKIP() << "AICore data root unavailable";
    }
    // The region_ocr proposer pins the default PP-OCRv4 Chinese recipe at
    // the q8_0 tier; the pairing must keep coming from the public OCR
    // preset API, and the assets must stay published catalog rows.
    EXPECT_EQ(models.detFileName, QStringLiteral("ppocr-det-ch-q8_0.gguf"));
    EXPECT_EQ(models.recFileName, QStringLiteral("ppocr-rec-ch-q8_0.gguf"));
    EXPECT_TRUE(models.cacheDir.contains(QStringLiteral("ocr_models")));
    EXPECT_TRUE(VLXSeekHelpers::ocrModelDownloadUrl(models.detFileName)
                        .startsWith(QStringLiteral("https://")));
    EXPECT_TRUE(VLXSeekHelpers::ocrModelDownloadUrl(models.recFileName)
                        .startsWith(QStringLiteral("https://")));
    EXPECT_TRUE(
            VLXSeekHelpers::ocrModelDownloadUrl(QStringLiteral("no-such.gguf"))
                    .isEmpty());
}

TEST(VLXSeekOcrProposer, QuadsConvertToAxisAlignedBoxes) {
    // One usable quad (corners deliberately unordered) + one degenerate
    // (zero-area) quad that must be skipped.
    const float quads[] = {110.0f, 20.0f,  10.0f, 20.0f, 10.0f, 130.0f,
                           110.5f, 130.0f, 5.0f,  5.0f,  5.0f,  5.0f,
                           5.0f,   5.0f,   5.0f,  5.0f};
    QVector<float> boxes;
    ASSERT_TRUE(VLXSeekHelpers::ocrQuadsToBoxes(quads, 2, &boxes));
    ASSERT_EQ(boxes.size(), 4);
    EXPECT_FLOAT_EQ(boxes[0], 10.0f);
    EXPECT_FLOAT_EQ(boxes[1], 20.0f);
    EXPECT_FLOAT_EQ(boxes[2], 110.5f);
    EXPECT_FLOAT_EQ(boxes[3], 130.0f);
}

TEST(VLXSeekOcrProposer, QuadsCapAtProposalContract) {
    QVector<float> raw;
    raw.reserve(101 * 8);
    for (int i = 0; i < 101; ++i)
        raw << float(i) << 0.0f << float(i) + 5.0f << 5.0f << float(i) << 5.0f
            << float(i) + 5.0f << 0.0f;
    QVector<float> boxes;
    ASSERT_TRUE(VLXSeekHelpers::ocrQuadsToBoxes(raw.constData(), 101, &boxes));
    EXPECT_EQ(boxes.size(), 400);

    QVector<float> none;
    const float degenerate[] = {5, 5, 5, 5, 5, 5, 5, 5};
    EXPECT_FALSE(VLXSeekHelpers::ocrQuadsToBoxes(degenerate, 1, &none));
    EXPECT_TRUE(none.isEmpty());
    EXPECT_FALSE(VLXSeekHelpers::ocrQuadsToBoxes(nullptr, 0, &none));
}

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
