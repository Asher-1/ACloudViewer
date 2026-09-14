// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Pure-logic qGKD helper tests (no AICore inference, no GPU).

#include <gtest/gtest.h>

#include <QGuiApplication>
#include <QImage>
#include <QPointF>
#include <QSet>
#include <QStringList>

#include "GKDModelCatalog.h"

namespace {

GKDKeypoint makeKeypoint(const QString& prompt, float x, float y) {
    GKDKeypoint kp;
    kp.prompt = prompt;
    kp.x = x;
    kp.y = y;
    kp.score = 0.9f;
    return kp;
}

}  // namespace

TEST(GkdSplitPrompts, EmptyAndWhitespace) {
    EXPECT_TRUE(GKDHelpers::splitPrompts(QString()).isEmpty());
    EXPECT_TRUE(GKDHelpers::splitPrompts(QStringLiteral("  ,, ")).isEmpty());
}

TEST(GkdSplitPrompts, BasicSplit) {
    const QStringList out = GKDHelpers::splitPrompts(
            QStringLiteral("nose, left eye,  right eye ,"));
    ASSERT_EQ(out.size(), 3);
    EXPECT_EQ(out[0], QStringLiteral("nose"));
    EXPECT_EQ(out[1], QStringLiteral("left eye"));
    EXPECT_EQ(out[2], QStringLiteral("right eye"));
}

TEST(GkdSplitPrompts, QuotedCommaSurvives) {
    const QStringList out =
            GKDHelpers::splitPrompts(QStringLiteral("\"left, right\", nose"));
    ASSERT_EQ(out.size(), 2);
    EXPECT_EQ(out[0], QStringLiteral("left, right"));
    EXPECT_EQ(out[1], QStringLiteral("nose"));
}

TEST(GkdParseCoordinatePairs, EmptyOk) {
    QVector<QPointF> out;
    EXPECT_TRUE(GKDHelpers::parseCoordinatePairs(QString(), &out));
    EXPECT_TRUE(out.isEmpty());
}

TEST(GkdParseCoordinatePairs, Pairs) {
    QVector<QPointF> out;
    ASSERT_TRUE(GKDHelpers::parseCoordinatePairs(
            QStringLiteral("10,20 30,40 -5,-6"), &out));
    ASSERT_EQ(out.size(), 3);
    EXPECT_EQ(out[0], QPointF(10, 20));
    EXPECT_EQ(out[1], QPointF(30, 40));
    EXPECT_EQ(out[2], QPointF(-5, -6));
}

TEST(GkdParseCoordinatePairs, RejectsOddAndGarbage) {
    QVector<QPointF> out;
    EXPECT_FALSE(
            GKDHelpers::parseCoordinatePairs(QStringLiteral("1,2 3"), &out));
    EXPECT_FALSE(
            GKDHelpers::parseCoordinatePairs(QStringLiteral("1,2 abc"), &out));
}

TEST(GkdRenderResult, RendersKeypointsWithThreshold) {
    QImage source(64, 64, QImage::Format_RGB32);
    source.fill(Qt::black);

    GKDKeypointSet set;
    set.label = QStringLiteral("human");
    set.hasBox = true;
    set.x1 = 4;
    set.y1 = 4;
    set.x2 = 60;
    set.y2 = 60;
    GKDKeypoint strong;
    strong.x = 20;
    strong.y = 20;
    strong.score = 0.9f;
    strong.prompt = QStringLiteral("nose");
    GKDKeypoint weak;
    weak.x = 40;
    weak.y = 40;
    weak.score = 0.1f;
    weak.prompt = QStringLiteral("tail");
    set.keypoints = {strong, weak};

    const QImage rendered = GKDHelpers::renderResult(source, {set}, 0.5f);
    EXPECT_FALSE(rendered.isNull());
    EXPECT_NE(rendered, source);

    // The weak keypoint is skipped, so its neighborhood stays black.
    EXPECT_EQ(rendered.pixelColor(40, 40), QColor(Qt::black));
}

TEST(GkdRenderResult, DrawsSkeletonBonesUnderKeypoints) {
    QImage source(64, 64, QImage::Format_RGB32);
    source.fill(Qt::black);

    GKDKeypointSet withBones;
    withBones.hasBox = false;
    withBones.keypoints = {makeKeypoint(QStringLiteral("nose"), 16, 32),
                           makeKeypoint(QStringLiteral("right eye"), 48, 32)};
    withBones.bones = {QPointF(16, 32), QPointF(48, 32)};

    GKDKeypointSet withoutBones = withBones;
    withoutBones.bones.clear();

    const QImage rendered = GKDHelpers::renderResult(source, {withBones}, 0.5f);
    const QImage plain = GKDHelpers::renderResult(source, {withoutBones}, 0.5f);
    EXPECT_NE(rendered, plain);
    // The bone midpoint is tinted only when bones render; a row well
    // above the 2px segment stays identical.
    EXPECT_NE(rendered.pixelColor(32, 32), plain.pixelColor(32, 32));
    EXPECT_EQ(rendered.pixelColor(32, 26), plain.pixelColor(32, 26));
}

TEST(GkdRenderResult, PointLabelsOptInOnly) {
    QImage source(96, 64, QImage::Format_RGB32);
    source.fill(Qt::white);

    auto onePointSet = [](float x, float y) {
        GKDKeypointSet set;
        set.keypoints = {makeKeypoint(QStringLiteral("nose"), x, y)};
        return set;
    };

    // Default (labels OFF): the pixel right of the dot keeps the source
    // color — no label background, in any mode.
    const QImage silent =
            GKDHelpers::renderResult(source, {onePointSet(16, 32)}, 0.5f);
    EXPECT_EQ(silent.pixelColor(28, 32), QColor(Qt::white));

    // Opt-in: the label background hugs the dot to its right (dark
    // translucent box over the white source).
    const QImage labeled =
            GKDHelpers::renderResult(source, {onePointSet(16, 32)}, 0.5f, true);
    EXPECT_NE(labeled.pixelColor(28, 32), QColor(Qt::white));

    // Opt-in also applies to multi-object runs (the user explicitly
    // asked for the labels there).
    const QImage multi = GKDHelpers::renderResult(
            source, {onePointSet(16, 12), onePointSet(16, 52)}, 0.5f, true);
    EXPECT_NE(multi.pixelColor(28, 12), QColor(Qt::white));
}

TEST(GkdRenderResult, RightEdgeLabelsFlipLeftOfDot) {
    // A keypoint near the right edge used to have its label run off the
    // image and get cropped (hand-X-ray / chair scenes). The label must
    // flip to the LEFT of the dot and stay fully on the canvas.
    QImage source(96, 64, QImage::Format_RGB32);
    source.fill(Qt::white);
    GKDKeypointSet set;
    set.keypoints = {makeKeypoint(QStringLiteral("nose"), 88, 32)};
    const QImage img = GKDHelpers::renderResult(source, {set}, 0.5f, true);
    // Left of the dot: the label background (dark box over white).
    EXPECT_NE(img.pixelColor(70, 32), QColor(Qt::white));
    // Right of the dot: nothing but the source (the dot itself ends at
    // x≈91; the old pass put the label background here).
    EXPECT_EQ(img.pixelColor(93, 32), QColor(Qt::white));
}

TEST(GkdRenderResult, DenseSceneFitsCapacityBudget) {
    // Six keypoints on a 96x64 canvas: six "nose 0.50" labels at the
    // base font exceed the 30% coverage budget, so the pass shrinks the
    // font and then drops the lowest-scoring labels instead of painting
    // a wall of text (measured fish-swarm / person-17 failure mode).
    // The label background (black @ 160 alpha over white) mixes to gray
    // ≈95; its share must stay near the budget (the ungoverned pass
    // covered ~73% of this canvas).
    QImage source(96, 64, QImage::Format_RGB32);
    source.fill(Qt::white);
    GKDKeypointSet set;
    for (int i = 0; i < 6; ++i)
        set.keypoints.append(
                makeKeypoint(QStringLiteral("nose"), 8.0f + i * 16.0f, 32.0f));
    const QImage img = GKDHelpers::renderResult(source, {set}, 0.5f, true);
    long grayish = 0;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) {
            const QColor c = img.pixelColor(x, y);
            if (c.red() == c.green() && c.green() == c.blue() &&
                c.red() >= 70 && c.red() <= 120)
                ++grayish;
        }
    EXPECT_GT(grayish, 0);
    EXPECT_LT(grayish * 100.0 / (96.0 * 64.0), 35.0);
}

TEST(GkdPlaceLabels, OverlappingRectsStackDownward) {
    // Two labels dropped on the same spot: the greedy pass keeps the
    // first at its preferred position and pushes the second below it
    // with the 2 px gutter; no pair intersects after placement.
    const QSize canvas(200, 200);
    QVector<QRect> preferred = {QRect(20, 10, 50, 16), QRect(20, 10, 50, 16)};
    const QVector<QRect> placed = GKDHelpers::placeLabels(preferred, canvas);
    ASSERT_EQ(placed.size(), 2);
    EXPECT_EQ(placed[0], preferred[0]);
    EXPECT_FALSE(placed[0].intersects(placed[1].adjusted(-2, -2, 2, 2)));
    EXPECT_GT(placed[1].top(), placed[0].bottom());
}

TEST(GkdPlaceLabels, DisjointRectsStayPut) {
    // Already-disjoint labels must not move (no leader arrows would be
    // drawn for them).
    const QSize canvas(200, 200);
    QVector<QRect> preferred = {QRect(4, 4, 40, 16), QRect(4, 40, 40, 16)};
    const QVector<QRect> placed = GKDHelpers::placeLabels(preferred, canvas);
    EXPECT_EQ(placed, preferred);
}

TEST(GkdPlaceLabels, CascadePinnedToCanvas) {
    // Five labels dropped on the same spot on a 100 px tall canvas: the
    // push-down cascade must pin the last ones to the bottom edge
    // instead of running them out of the image (measured failure:
    // cropped labels on dense scenes).
    const QSize canvas(200, 100);
    QVector<QRect> preferred(5, QRect(20, 10, 50, 16));
    const QVector<QRect> placed = GKDHelpers::placeLabels(preferred, canvas);
    for (const QRect& r : placed) {
        EXPECT_GE(r.top(), 2);
        EXPECT_LE(r.bottom(), canvas.height() - 2);
    }
    // At least one label actually rode the pin (the cascade reached the
    // bottom edge rather than fitting all five with gutters).
    int pinned = 0;
    for (const QRect& r : placed)
        if (r.bottom() >= canvas.height() - 3) ++pinned;
    EXPECT_GT(pinned, 0);
}

TEST(GkdRenderResult, EmptyInputGivesEmptyImage) {
    EXPECT_TRUE(GKDHelpers::renderResult(QImage(), {}, 0.3f).isNull());
}

TEST(GkdGroupColor, Deterministic) {
    EXPECT_EQ(GKDHelpers::groupColor(3), GKDHelpers::groupColor(3));
    EXPECT_NE(GKDHelpers::groupColor(0), GKDHelpers::groupColor(1));
}

// ---- Official-demo skeleton helpers ------------------------------------

TEST(GkdSkeleton, ParseSkeletonPairsAndGarbage) {
    EXPECT_TRUE(GKDHelpers::parseSkeleton(QString()).isEmpty());
    EXPECT_TRUE(GKDHelpers::parseSkeleton(QStringLiteral("  ")).isEmpty());
    const auto bones =
            GKDHelpers::parseSkeleton(QStringLiteral("1-2 3-4 junk 16 - 14"));
    ASSERT_EQ(bones.size(), 3);
    EXPECT_EQ(bones[0].first, 1);
    EXPECT_EQ(bones[0].second, 2);
    EXPECT_EQ(bones[1].first, 3);
    EXPECT_EQ(bones[1].second, 4);
    EXPECT_EQ(bones[2].first, 16);
    EXPECT_EQ(bones[2].second, 14);
}

TEST(GkdSkeleton, BuildBonesByPromptSkipsHiddenEndpoints) {
    // Face-5 demo texts; the left eye fell below the display cut.
    const QStringList prompts{QStringLiteral("nose"),
                              QStringLiteral("left eye"),
                              QStringLiteral("right eye")};
    // Full result list (official COCO semantics): the hidden left eye
    // stays in the data and is filtered by the minScore cut.
    QVector<GKDKeypoint> keypoints{
            makeKeypoint(QStringLiteral("nose"), 10, 10),
            makeKeypoint(QStringLiteral("left eye"), 20, 20),
            makeKeypoint(QStringLiteral("right eye"), 30, 10)};
    keypoints[1].score = 0.05f;
    const auto skeleton =
            GKDHelpers::parseSkeleton(QStringLiteral("1-2 1-3 2-3"));
    // 1-2 (nose/left eye) and 2-3 (left eye/right eye) lose their
    // below-cut left-eye endpoint; only 1-3 links the two shown points.
    const auto bones =
            GKDHelpers::buildBones(prompts, keypoints, skeleton, 0.10f);
    ASSERT_EQ(bones.size(), 2);
    EXPECT_EQ(bones[0], QPointF(10, 10));
    EXPECT_EQ(bones[1], QPointF(30, 10));
}

TEST(GkdSkeleton, BuildBonesByOriginalIndexForVisualMode) {
    // Visual-prompt mode: no texts, order is the support-point order.
    QVector<GKDKeypoint> keypoints{makeKeypoint(QString(), 1, 1),
                                   makeKeypoint(QString(), 2, 2),
                                   makeKeypoint(QString(), 3, 3)};
    keypoints[1].score = 0.05f;  // below the display cut
    const auto skeleton =
            GKDHelpers::parseSkeleton(QStringLiteral("1-2 2-3 1-3"));
    const auto bones =
            GKDHelpers::buildBones(QStringList(), keypoints, skeleton, 0.10f);
    // 1-2 and 2-3 touch the hidden second support point; only 1-3 links
    // the two shown points.
    ASSERT_EQ(bones.size(), 2);
    EXPECT_EQ(bones[0], QPointF(1, 1));
    EXPECT_EQ(bones[1], QPointF(3, 3));
}

// ---- Mode presets (the "Use test data" one-click fills) ----------------

TEST(GkdModePresets, ModeTableIsComplete) {
    const QStringList modes = GKDHelpers::promptModes();
    ASSERT_EQ(modes.size(), 4);
    EXPECT_EQ(modes[0], QStringLiteral("text"));
    EXPECT_EQ(modes[1], QStringLiteral("visual"));
    EXPECT_EQ(modes[2], QStringLiteral("multimodal"));
    EXPECT_EQ(modes[3], QStringLiteral("multi"));
    // Every mode resolves to a non-empty preset with a query image.
    for (const QString& mode : modes) {
        const GKDModePreset preset = GKDHelpers::modePreset(mode);
        EXPECT_FALSE(preset.queryImage.isEmpty()) << mode.toStdString();
    }
    // Only the multi-object composition uses the YOLO-World detector.
    for (const QString& mode : modes) {
        EXPECT_EQ(GKDHelpers::modeUsesYolo(mode),
                  mode == QStringLiteral("multi"));
    }
}

TEST(GkdModePresets, TextDemoMatchesUpstream) {
    const GKDModePreset preset = GKDHelpers::modePreset(QStringLiteral("text"));
    EXPECT_EQ(preset.queryImage, QStringLiteral("2007_007524.jpg"));
    // The official text demo detects five face keypoints.
    EXPECT_EQ(GKDHelpers::splitPrompts(preset.kpsTexts).size(), 5);
    EXPECT_TRUE(preset.supportImage.isEmpty());
    EXPECT_TRUE(preset.supportKps.isEmpty());
    EXPECT_TRUE(preset.objectClasses.isEmpty());
}

TEST(GkdModePresets, VisualDemoSupportKpsParse) {
    const GKDModePreset preset =
            GKDHelpers::modePreset(QStringLiteral("visual"));
    // Cross-instance few-shot: the cat-face support points drive the
    // detection on a different species (front-facing pug).
    EXPECT_EQ(preset.queryImage, QStringLiteral("2008_000808.jpg"));
    EXPECT_EQ(preset.supportImage, QStringLiteral("2007_003778.jpg"));
    QVector<QPointF> kps;
    ASSERT_TRUE(GKDHelpers::parseCoordinatePairs(preset.supportKps, &kps));
    // Official 1-shot demo: left eye (343,166), right eye (281,158),
    // nose (311,197) on the support image.
    ASSERT_EQ(kps.size(), 3);
    EXPECT_EQ(kps[0], QPointF(343, 166));
    EXPECT_EQ(kps[1], QPointF(281, 158));
    EXPECT_EQ(kps[2], QPointF(311, 197));
    EXPECT_TRUE(preset.kpsTexts.isEmpty());
    EXPECT_TRUE(preset.objectClasses.isEmpty());
}

TEST(GkdModePresets, MultimodalCountsMatchBackendContract) {
    const GKDModePreset preset =
            GKDHelpers::modePreset(QStringLiteral("multimodal"));
    // The backend rejects multimodal prompts unless
    // n_kps_texts == n_support_kps — the preset must satisfy that.
    EXPECT_EQ(GKDHelpers::splitPrompts(preset.kpsTexts).size(), 3);
    QVector<QPointF> kps;
    ASSERT_TRUE(GKDHelpers::parseCoordinatePairs(preset.supportKps, &kps));
    EXPECT_EQ(kps.size(), 3);
}

TEST(GkdModePresets, MultiObjectDemoDefinesDetectorInputs) {
    const GKDModePreset preset =
            GKDHelpers::modePreset(QStringLiteral("multi"));
    EXPECT_FALSE(preset.queryImage.isEmpty());
    EXPECT_FALSE(preset.objectClasses.isEmpty());
    // GKD per box needs keypoint texts (worker-side hard requirement).
    EXPECT_FALSE(GKDHelpers::splitPrompts(preset.kpsTexts).isEmpty());
    EXPECT_TRUE(preset.supportImage.isEmpty());
    EXPECT_TRUE(preset.supportKps.isEmpty());
}

TEST(GkdModePresets, RotationCoversBundledScenes) {
    // Every scenario list starts with the recommended first-frame image
    // and then walks the rest of the bundled dataset. Single-object
    // modes rotate single-subject queries only: multi-target lineup
    // images stay in the multi-object rotation.
    EXPECT_EQ(GKDHelpers::modePresets(QStringLiteral("text")).size(), 5);
    EXPECT_EQ(GKDHelpers::modePresets(QStringLiteral("visual")).size(), 4);
    EXPECT_EQ(GKDHelpers::modePresets(QStringLiteral("multimodal")).size(), 2);
    const QVector<GKDModePreset> rotation =
            GKDHelpers::modePresets(QStringLiteral("multi"));
    ASSERT_EQ(rotation.size(), 7);

    // Every scenario is runnable as-is: query + classes + keypoint texts
    // (the worker's hard requirements), no support fields.
    QSet<QString> seenQueries;
    for (const GKDModePreset& preset : rotation) {
        EXPECT_FALSE(preset.queryImage.isEmpty());
        EXPECT_FALSE(preset.objectClasses.isEmpty());
        EXPECT_FALSE(GKDHelpers::splitPrompts(preset.kpsTexts).isEmpty());
        EXPECT_TRUE(preset.supportImage.isEmpty());
        seenQueries.insert(preset.queryImage);
    }
    // Successive clicks walk DIFFERENT images (no duplicate scenarios).
    EXPECT_EQ(seenQueries.size(), rotation.size());
    // The first scenario stays the official 20-point alpaca demo.
    EXPECT_EQ(rotation.first().queryImage, QStringLiteral("alpaca_150.jpg"));
    EXPECT_EQ(GKDHelpers::modePreset(QStringLiteral("multi")).queryImage,
              rotation.first().queryImage);

    // Skeletons ride along with every scenario that has one upstream
    // (schemas or README --skeleton flags). awa_pose, hand_xray and
    // nabird leave 'skeleton' empty upstream, so the xray/tiger/pigs/
    // birds presets carry self-defined topologies over the official
    // point sets; fish has no official schema and renders points only.
    EXPECT_FALSE(
            GKDHelpers::modePreset(QStringLiteral("text")).skeleton.isEmpty());
    const QVector<GKDModePreset> textRotation =
            GKDHelpers::modePresets(QStringLiteral("text"));
    // hand_xray: self-defined per-finger chains (19 edges over 24 pts).
    EXPECT_EQ(GKDHelpers::parseSkeleton(textRotation.at(1).skeleton).size(),
              19);
    // tiger: self-defined awa_pose topology, antler rows trimmed
    // (26 edges over 35 points).
    EXPECT_EQ(GKDHelpers::parseSkeleton(textRotation.at(4).skeleton).size(),
              26);
    EXPECT_FALSE(GKDHelpers::modePreset(QStringLiteral("multi"))
                         .skeleton.isEmpty());  // alpaca animal_pose
    // Pigs scene: scene conf 0.15 (measured all-real at that cut) and
    // the same 26-edge awa topology as the tiger.
    const QVector<GKDModePreset> multiRotation =
            GKDHelpers::modePresets(QStringLiteral("multi"));
    EXPECT_FLOAT_EQ(multiRotation.at(2).yoloConf, 0.15f);
    EXPECT_EQ(GKDHelpers::parseSkeleton(multiRotation.at(2).skeleton).size(),
              26);
    // Birds: self-defined nabird topology (10 edges over 11 points).
    EXPECT_EQ(GKDHelpers::parseSkeleton(multiRotation.at(6).skeleton).size(),
              10);

    // Every bundled dataset image is reachable through some mode's
    // rotation (14 images: the support image serves visual/multimodal
    // only; the other 13 appear as query images somewhere — the
    // dish-washing scene was dropped from the rotation).
    QSet<QString> allQueries;
    for (const QString& mode :
         {QStringLiteral("text"), QStringLiteral("visual"),
          QStringLiteral("multimodal"), QStringLiteral("multi")}) {
        for (const GKDModePreset& preset : GKDHelpers::modePresets(mode)) {
            allQueries.insert(preset.queryImage);
        }
    }
    EXPECT_EQ(allQueries.size(), 13);
}

int main(int argc, char** argv) {
    // QPainter text rendering requires a QGuiApplication instance.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QGuiApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
