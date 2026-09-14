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
#include <QStringList>

#include "GKDModelCatalog.h"

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

TEST(GkdRenderResult, EmptyInputGivesEmptyImage) {
    EXPECT_TRUE(GKDHelpers::renderResult(QImage(), {}, 0.3f).isNull());
}

TEST(GkdGroupColor, Deterministic) {
    EXPECT_EQ(GKDHelpers::groupColor(3), GKDHelpers::groupColor(3));
    EXPECT_NE(GKDHelpers::groupColor(0), GKDHelpers::groupColor(1));
}

int main(int argc, char** argv) {
    // QPainter text rendering requires a QGuiApplication instance.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QGuiApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
