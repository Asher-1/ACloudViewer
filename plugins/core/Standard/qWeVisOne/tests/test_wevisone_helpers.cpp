// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Pure-logic unit tests for the qWeVisOne helpers (catalog bridge, task
// family coverage, upstream --ref-boxes parser). No GGUF model required.

#include <gtest/gtest.h>

#include <QColor>
#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>

#include "WeVisOneModelCatalog.h"

namespace {

constexpr int kRoleVlm = 0;
constexpr int kRoleMmproj = 1;
constexpr int kRoleDecoder = 2;

}  // namespace

TEST(WeVisOneCatalog, RowCountAndRoles) {
    const QVector<WeVisOneModelEntry> rows = WeVisOneHelpers::catalogModels();
    ASSERT_EQ(rows.size(), 12);
    int vlm = 0, mmproj = 0, decoder = 0;
    for (const WeVisOneModelEntry& row : rows) {
        if (row.role == kRoleVlm) ++vlm;
        if (row.role == kRoleMmproj) ++mmproj;
        if (row.role == kRoleDecoder) ++decoder;
        EXPECT_FALSE(row.filename.isEmpty());
        EXPECT_TRUE(row.downloadUrl.startsWith(QStringLiteral("https://")));
        EXPECT_EQ(row.sha256.size(), 64);
        EXPECT_GT(row.sizeBytes, 0);
        EXPECT_TRUE(row.family == QStringLiteral("2b") ||
                    row.family == QStringLiteral("4b"));
    }
    // Two families x (3 VLM tiers + 1 f16 mmproj + 2 decoder tiers).
    EXPECT_EQ(vlm, 6);
    EXPECT_EQ(mmproj, 2);
    EXPECT_EQ(decoder, 4);
}

TEST(WeVisOneCatalog, FamiliesInFirstSeenOrder) {
    const QStringList families = WeVisOneHelpers::catalogFamilies();
    ASSERT_EQ(families.size(), 2);
    EXPECT_EQ(families[0], QStringLiteral("2b"));
    EXPECT_EQ(families[1], QStringLiteral("4b"));
}

TEST(WeVisOneCatalog, MmprojAlwaysF16DecoderHasBothTiers) {
    // The only published projector tier is f16; the decoder ships the
    // parity-gated f32 checkpoint dtype plus an f16 smoke tier per family.
    for (const QString& family : WeVisOneHelpers::catalogFamilies()) {
        QStringList decoderTiers;
        for (const WeVisOneModelEntry& row : WeVisOneHelpers::catalogModels()) {
            if (row.family != family) continue;
            if (row.role == kRoleMmproj)
                EXPECT_EQ(row.quantization, QStringLiteral("f16"));
            if (row.role == kRoleDecoder)
                decoderTiers.push_back(row.quantization);
        }
        decoderTiers.sort();
        EXPECT_EQ(decoderTiers,
                  QStringList({QStringLiteral("f16"), QStringLiteral("f32")}));
    }
}

TEST(WeVisOneCatalog, DefaultIndexIsRecommended2bQ8VlmRow) {
    const int def = WeVisOneHelpers::catalogDefaultIndex();
    ASSERT_GE(def, 0);
    const QVector<WeVisOneModelEntry> rows = WeVisOneHelpers::catalogModels();
    ASSERT_LT(def, rows.size());
    EXPECT_EQ(rows[def].role, kRoleVlm);
    EXPECT_EQ(rows[def].family, QStringLiteral("2b"));
    EXPECT_EQ(rows[def].quantization, QStringLiteral("q8_0"));
    EXPECT_TRUE(rows[def].displayName.contains(QStringLiteral("recommended")));
}

TEST(WeVisOneCatalog, FindByFilenameRoundTrip) {
    WeVisOneModelEntry out;
    ASSERT_TRUE(WeVisOneHelpers::findModelByFilename(
            QStringLiteral("wevisone-vlm-4b-q4_K_M.gguf"), &out));
    EXPECT_EQ(out.role, kRoleVlm);
    EXPECT_EQ(out.family, QStringLiteral("4b"));
    EXPECT_EQ(out.quantization, QStringLiteral("q4_K_M"));
    EXPECT_FALSE(WeVisOneHelpers::findModelByFilename(
            QStringLiteral("no-such-file.gguf"), &out));
}

TEST(WeVisOneCatalog, CatalogJsonIsComplete) {
    const QJsonArray rows = WeVisOneHelpers::catalogJson();
    ASSERT_EQ(rows.size(), 12);
    const QJsonDocument doc(rows);
    EXPECT_TRUE(doc.toJson(QJsonDocument::Compact).contains("wevisone-vlm-2b"));
    EXPECT_TRUE(doc.toJson(QJsonDocument::Compact)
                        .contains("wevisone-mask-decoder-4b"));
}

TEST(WeVisOneCatalog, CacheDirUnderDataRoot) {
    const QString cache = WeVisOneHelpers::modelCacheDir();
    if (cache.isEmpty()) {
        GTEST_SKIP() << "AICore data root unavailable";
    }
    EXPECT_TRUE(cache.contains(QStringLiteral("wevisone_models")));
}

TEST(WeVisOneRefBoxes, ParsesUpstreamSpecFormat) {
    QString error;
    const QVector<float> single = WeVisOneHelpers::parseRefBoxes(
            QStringLiteral("880,120,1500,1070"), &error);
    EXPECT_TRUE(error.isEmpty());
    ASSERT_EQ(single.size(), 4);
    EXPECT_FLOAT_EQ(single[0], 880.0f);
    EXPECT_FLOAT_EQ(single[3], 1070.0f);

    // Semicolon-separated boxes with whitespace (the upstream CLI spec).
    const QVector<float> multi = WeVisOneHelpers::parseRefBoxes(
            QStringLiteral("1, 2, 3, 4; 5.5, 6, 7, 8"), &error);
    EXPECT_TRUE(error.isEmpty());
    ASSERT_EQ(multi.size(), 8);
    EXPECT_FLOAT_EQ(multi[4], 5.5f);
    EXPECT_FLOAT_EQ(multi[7], 8.0f);
}

TEST(WeVisOneRefBoxes, EmptyTextYieldsEmptyVector) {
    QString error;
    EXPECT_TRUE(WeVisOneHelpers::parseRefBoxes(QString(), &error).isEmpty());
    EXPECT_TRUE(WeVisOneHelpers::parseRefBoxes(QStringLiteral(" ; "), &error)
                        .isEmpty());
    EXPECT_TRUE(error.isEmpty());
}

TEST(WeVisOneRefBoxes, RejectsMalformedQuadruples) {
    QString error;
    EXPECT_TRUE(WeVisOneHelpers::parseRefBoxes(QStringLiteral("1,2,3"), &error)
                        .isEmpty());
    EXPECT_FALSE(error.isEmpty());

    error.clear();
    EXPECT_TRUE(
            WeVisOneHelpers::parseRefBoxes(QStringLiteral("1,2,3,4,5"), &error)
                    .isEmpty());
    EXPECT_FALSE(error.isEmpty());

    error.clear();
    EXPECT_TRUE(
            WeVisOneHelpers::parseRefBoxes(QStringLiteral("a,b,c,d"), &error)
                    .isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(WeVisOneAnswerLog, DetectionAnswerBecomesReadableLabelBoxes) {
    // The verbatim detection answer from a real run (console log): one
    // single-box ref, one multi-box ref, one single-box ref.
    const QString raw = QStringLiteral(
            "<ref>sofa</ref><box><30><454><964><876></box>, "
            "<ref>pillow</ref><box><100><459><301><589>,<108><545><333><688>,"
            "<629><528><809><666>,<724><464><924><664></box>, "
            "<ref>lamp</ref><box><29><133><122><571></box>");
    const QString out = WeVisOneHelpers::formatAnswerForLog(raw);
    EXPECT_EQ(out,
              QStringLiteral("sofa  : [30,454,964,876]\n"
                             "pillow: [100,459,301,589], [108,545,333,688], "
                             "[629,528,809,666], [724,464,924,664]\n"
                             "lamp  : [29,133,122,571]"));
    // No raw structure tokens may leak into the readable rendering.
    EXPECT_FALSE(out.contains(QStringLiteral("<ref>")));
    EXPECT_FALSE(out.contains(QStringLiteral("<box>")));
}

TEST(WeVisOneAnswerLog, CountingTailKeptAndStatedTotalParsed) {
    // Verbatim upstream CLI answer for the cups case study (2b q8_0 CUDA):
    // three localization boxes but the model states the true total 4 — the
    // stated total is the count, the tail must never be dropped.
    const QString raw = QStringLiteral(
            "<ref>cups on the "
            "left</ref><box><0><57><326><189>,<0><188><326><281>,"
            "<0><224><320><831></box>Total number: 4.");
    const QString out = WeVisOneHelpers::formatAnswerForLog(raw);
    EXPECT_EQ(out, QStringLiteral("cups on the left: [0,57,326,189], "
                                  "[0,188,326,281], [0,224,320,831]\n"
                                  "Total number: 4."));
    EXPECT_EQ(WeVisOneHelpers::statedTotalCount(raw), 4);
    EXPECT_EQ(WeVisOneHelpers::statedTotalCount(
                      QStringLiteral("<ref>sofa</ref><box><1><2><3><4></box>")),
              -1);
}

TEST(WeVisOneAnswerLog, ConvertsToPixelsWhenImageSizeKnown) {
    // 0-1000 tokens scale to the image frame: x*2, y*1 for a 2000x1000 image.
    const QString out = WeVisOneHelpers::formatAnswerForLog(
            QStringLiteral("<ref>sofa</ref><box><30><454><964><876></box>"),
            QSize(2000, 1000));
    EXPECT_EQ(out, QStringLiteral("sofa: [60,454,1928,876]"));
}

TEST(WeVisOneAnswerLog, PunctuationOnlyTailRidesOnLastLine) {
    // Freeform answers end with a sentence-final "." after the last
    // grounding group; it must not become a lonely orphan line.
    const QString raw = QStringLiteral(
            "The room features a <ref>sofa</ref>"
            "<box><30><454><964><876></box>.");
    const QString out = WeVisOneHelpers::formatAnswerForLog(raw);
    EXPECT_EQ(out, QStringLiteral("The room features a\n"
                                  "sofa: [30,454,964,876]."));
}

TEST(WeVisOneAnswerLog, PointAndPlainTextAnswers) {
    const QString point = WeVisOneHelpers::formatAnswerForLog(
            QStringLiteral("<ref>umbrella</ref><point><512><300></point>"));
    EXPECT_EQ(point, QStringLiteral("umbrella: (512,300)"));
    // ocr / freeform / count answers carry no structure: pass through.
    EXPECT_EQ(WeVisOneHelpers::formatAnswerForLog(QStringLiteral("  COFFEE  ")),
              QStringLiteral("COFFEE"));
    EXPECT_EQ(WeVisOneHelpers::formatAnswerForLog(QStringLiteral("3")),
              QStringLiteral("3"));
}

TEST(WeVisOneDemoSample, PrefersCaseStudyOverOfficialAndFormatsRefBoxes) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString path = dir.filePath(QStringLiteral("manifest.json"));
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.write(QStringLiteral("{\"samples\":["
                           "{\"task\":\"matting\",\"image\":\"seabreeze.jpg\","
                           "\"query\":\"person\"},"
                           "{\"task\":\"matting\",\"image\":\"chicken.jpg\","
                           "\"query\":\"chicken\"},"
                           "{\"task\":\"detection\",\"image\":\"livingroom."
                           "jpg\",\"query\":\"sofa\"},"
                           "{\"task\":\"visual_prompt\",\"image\":\"seabreeze."
                           "jpg\",\"query\":\"the person\","
                           "\"ref_boxes\":[[880,120,1500,1070]]}"
                           "]}")
                    .toUtf8());
    f.close();
    WeVisOneHelpers::WeVisOneDemoSample s;
    // matting: skip the official seabreeze, prefer the case-study chicken.
    ASSERT_TRUE(WeVisOneHelpers::readDemoSample(path, QStringLiteral("matting"),
                                                &s));
    EXPECT_EQ(s.image, QStringLiteral("chicken.jpg"));
    EXPECT_EQ(s.query, QStringLiteral("chicken"));
    EXPECT_TRUE(s.refBoxes.isEmpty());
    // detection: only official images -> first sample.
    ASSERT_TRUE(WeVisOneHelpers::readDemoSample(
            path, QStringLiteral("detection"), &s));
    EXPECT_EQ(s.image, QStringLiteral("livingroom.jpg"));
    // visual_prompt: ref_boxes rendered as the upstream --ref-boxes spec.
    ASSERT_TRUE(WeVisOneHelpers::readDemoSample(
            path, QStringLiteral("visual_prompt"), &s));
    EXPECT_EQ(s.refBoxes, QStringLiteral("880,120,1500,1070"));
    // Unknown task / absent manifest -> false (caller falls back).
    EXPECT_FALSE(
            WeVisOneHelpers::readDemoSample(path, QStringLiteral("nope"), &s));
    EXPECT_FALSE(WeVisOneHelpers::readDemoSample(
            dir.filePath(QStringLiteral("absent.json")),
            QStringLiteral("matting"), &s));
}

TEST(WeVisOneDemoSample, ChessCasePreferredWhenManifestOffersIt) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString path = dir.filePath(QStringLiteral("manifest.json"));
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    // Mirrors the released manifest: chess.jpg is offered for referring,
    // count_box and count_point; count_box also has a case-study original
    // (cups) which must NOT outrank the classic chessboard case.
    f.write(QStringLiteral(
                    "{\"samples\":["
                    "{\"task\":\"referring\",\"image\":\"dogs.jpg\","
                    "\"query\":\"the dog closest to the camera\"},"
                    "{\"task\":\"referring\",\"image\":\"chess.jpg\","
                    "\"query\":\"the white king\"},"
                    "{\"task\":\"count_box\",\"image\":\"boys.jpg\","
                    "\"query\":\"people\"},"
                    "{\"task\":\"count_box\",\"image\":\"chess.jpg\","
                    "\"query\":\"chess pieces\"},"
                    "{\"task\":\"count_box\",\"image\":\"cups.jpg\","
                    "\"query\":\"cups on the left\"},"
                    "{\"task\":\"count_point\",\"image\":\"chess.jpg\","
                    "\"query\":\"white chess pieces\"},"
                    "{\"task\":\"count_point\",\"image\":\"colored_balls."
                    "jpg\",\"query\":\"green balls\"}"
                    "]}")
                    .toUtf8());
    f.close();
    WeVisOneHelpers::WeVisOneDemoSample s;
    ASSERT_TRUE(WeVisOneHelpers::readDemoSample(
            path, QStringLiteral("referring"), &s));
    EXPECT_EQ(s.image, QStringLiteral("chess.jpg"));
    EXPECT_EQ(s.query, QStringLiteral("the white king"));
    ASSERT_TRUE(WeVisOneHelpers::readDemoSample(
            path, QStringLiteral("count_box"), &s));
    EXPECT_EQ(s.image, QStringLiteral("chess.jpg"));
    EXPECT_EQ(s.query, QStringLiteral("chess pieces"));
    ASSERT_TRUE(WeVisOneHelpers::readDemoSample(
            path, QStringLiteral("count_point"), &s));
    EXPECT_EQ(s.image, QStringLiteral("chess.jpg"));
    EXPECT_EQ(s.query, QStringLiteral("white chess pieces"));
}

TEST(WeVisOneTasks, ElevenReleasedFamiliesInUpstreamOrder) {
    const QStringList tasks = WeVisOneHelpers::taskNames();
    ASSERT_EQ(tasks.size(), 11);
    EXPECT_EQ(tasks[0], QStringLiteral("detection"));
    EXPECT_EQ(tasks[5], QStringLiteral("ocr"));
    EXPECT_EQ(tasks[9], QStringLiteral("visual_prompt"));
    EXPECT_EQ(tasks[10], QStringLiteral("freeform"));
    EXPECT_TRUE(tasks.contains(QStringLiteral("count_box")));
    EXPECT_TRUE(tasks.contains(QStringLiteral("count_point")));
    EXPECT_TRUE(tasks.contains(QStringLiteral("ocr_grounding")));
    EXPECT_TRUE(tasks.contains(QStringLiteral("segmentation")));
    EXPECT_TRUE(tasks.contains(QStringLiteral("matting")));
    EXPECT_TRUE(tasks.contains(QStringLiteral("referring")));
    EXPECT_TRUE(tasks.contains(QStringLiteral("point")));
}

// ---- overlay rendering: visible point markers (count_point payload) ----

TEST(WeVisOneOverlay, PointMarkerGeometryMirrorsUpstreamOverlay) {
    // visualization.py render_overlay: radius = max(6, round(w/85)),
    // outline_thickness = max(2, round(radius/3)).
    EXPECT_EQ(WeVisOneHelpers::pointMarkerRadius(1875), 22);  // chess.jpg
    EXPECT_EQ(WeVisOneHelpers::pointMarkerOutline(22), 7);
    EXPECT_EQ(WeVisOneHelpers::pointMarkerRadius(389), 6);  // cups.jpg
    EXPECT_EQ(WeVisOneHelpers::pointMarkerOutline(6), 2);
    EXPECT_EQ(WeVisOneHelpers::pointMarkerRadius(100), 6);  // floor
    EXPECT_EQ(WeVisOneHelpers::pointMarkerRadius(850), 10);
    EXPECT_EQ(WeVisOneHelpers::pointMarkerOutline(10), 3);
}

TEST(WeVisOneOverlay, DegeneratePointBoxBecomesVisibleDot) {
    QImage input(850, 600, QImage::Format_RGB888);
    input.fill(Qt::black);
    QVector<QRectF> boxes;
    boxes.push_back(QRectF(400, 300, 0, 0));  // [x,y,x,y] point payload
    const QImage out = WeVisOneHelpers::annotateResult(input, boxes,
                                                       QVector<float>(), 0, 0);
    const int radius = WeVisOneHelpers::pointMarkerRadius(850);       // 10
    const int outline = WeVisOneHelpers::pointMarkerOutline(radius);  // 3
    const auto px = [&out](int x, int y) { return out.pixelColor(x, y); };
    // Filled core, white ring, untouched background: the old width-3
    // rectangle left a 3x3 speck here, invisible on large images.
    EXPECT_EQ(px(400, 300), QColor(255, 0, 0));
    EXPECT_EQ(px(400 + radius - outline, 300), QColor(255, 0, 0));
    EXPECT_EQ(px(400 + radius, 300), QColor(255, 255, 255));
    EXPECT_EQ(px(400 + radius + outline, 300), QColor(0, 0, 0));
    int painted = 0;
    for (int y = 300 - radius - outline; y <= 300 + radius + outline; ++y) {
        for (int x = 400 - radius - outline; x <= 400 + radius + outline; ++x) {
            if (px(x, y) != QColor(0, 0, 0)) ++painted;
        }
    }
    EXPECT_GT(painted, 100);
}

TEST(WeVisOneOverlay, RealBoxKeepsWidthThreeRedStroke) {
    QImage input(200, 100, QImage::Format_RGB888);
    input.fill(Qt::black);
    QVector<QRectF> boxes;
    boxes.push_back(QRectF(20, 10, 60, 40));
    const QImage out = WeVisOneHelpers::annotateResult(input, boxes,
                                                       QVector<float>(), 0, 0);
    EXPECT_EQ(out.pixelColor(20, 10), QColor(255, 0, 0));
    EXPECT_EQ(out.pixelColor(80, 50), QColor(255, 0, 0));
    EXPECT_EQ(out.pixelColor(19, 10), QColor(255, 0, 0));  // center - 1
    EXPECT_EQ(out.pixelColor(18, 10), QColor(0, 0, 0));
    EXPECT_EQ(out.pixelColor(50, 30), QColor(0, 0, 0));  // hollow inside
}

TEST(WeVisOneOverlay, MatteCompositeMatchesExportHelperFormula) {
    QImage input(4, 4, QImage::Format_RGB888);
    input.fill(QColor(100, 100, 100));
    QVector<float> matte(4, 1.0f);  // 2x2 fully opaque matte
    const QImage out = WeVisOneHelpers::annotateResult(input, QVector<QRectF>(),
                                                       matte, 2, 2);
    // round(100*(1-0.55)) = 45; round(100*0.45 + 220*0.55) = 166.
    EXPECT_EQ(out.pixelColor(0, 0), QColor(45, 166, 45));
    EXPECT_EQ(out.pixelColor(3, 3), QColor(45, 166, 45));
}

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}