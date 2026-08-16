// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Pure-logic unit tests for the qRFDetr helpers (JSON parsing, palette,
// segmentation-name detection, catalog mirror). No GGUF model required.

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QImage>

#include "RFDetrModelCatalog.h"

TEST(RFDetrHelpers, ParseDetectionsJsonBasic) {
    const QByteArray json =
            "{\"model\":\"base\",\"segmentation\":0,\"image_size\":640,"
            "\"num_classes\":80,\"num_queries\":300,"
            "\"image\":{\"width\":1280,\"height\":720},"
            "\"detections\":["
            "{\"class_id\":0,\"class_name\":\"person\",\"score\":0.93,"
            "\"box\":[10.5,20.0,110.5,220.0]},"
            "{\"class_id\":2,\"class_name\":\"car\",\"score\":0.81,"
            "\"box\":[300,400,500,600]}]}";
    RFDetrRunResult out;
    ASSERT_TRUE(RFDetrHelpers::parseDetectionsJson(json, &out));
    EXPECT_EQ(out.modelVariant, QStringLiteral("base"));
    EXPECT_EQ(out.imageSize, 640);
    EXPECT_EQ(out.numClasses, 80);
    EXPECT_FALSE(out.segmentation);
    EXPECT_EQ(out.detections.size(), 2);
    EXPECT_EQ(out.totalDetected, 2);

    const RFDetrDetection& person = out.detections.at(0);
    EXPECT_EQ(person.classId, 0u);
    EXPECT_EQ(person.className, QStringLiteral("person"));
    EXPECT_NEAR(person.score, 0.93f, 1e-4f);
    EXPECT_NEAR(person.x1, 10.5f, 1e-3f);
    EXPECT_NEAR(person.y2, 220.0f, 1e-3f);

    const RFDetrDetection& car = out.detections.at(1);
    EXPECT_EQ(car.classId, 2u);
    EXPECT_EQ(car.className, QStringLiteral("car"));
}

TEST(RFDetrHelpers, ParseDetectionsJsonEmptyArray) {
    const QByteArray json =
            "{\"model\":\"nano\",\"segmentation\":1,\"image_size\":640,"
            "\"num_classes\":80,\"num_queries\":300,"
            "\"image\":{\"width\":100,\"height\":100},\"detections\":[]}";
    RFDetrRunResult out;
    ASSERT_TRUE(RFDetrHelpers::parseDetectionsJson(json, &out));
    EXPECT_TRUE(out.segmentation);
    EXPECT_EQ(out.detections.size(), 0);
}

TEST(RFDetrHelpers, ParseDetectionsJsonInvalid) {
    RFDetrRunResult out;
    EXPECT_FALSE(RFDetrHelpers::parseDetectionsJson("not json", &out));
    EXPECT_FALSE(RFDetrHelpers::parseDetectionsJson("[]", &out));
}

TEST(RFDetrHelpers, ClassColorDeterministic) {
    EXPECT_EQ(RFDetrHelpers::classColor(0), RFDetrHelpers::classColor(0));
    EXPECT_EQ(RFDetrHelpers::classColor(20), RFDetrHelpers::classColor(0));
    EXPECT_EQ(RFDetrHelpers::classColor(21), RFDetrHelpers::classColor(1));
    EXPECT_NE(RFDetrHelpers::classColor(3), RFDetrHelpers::classColor(4));
}

TEST(RFDetrHelpers, FilenameIsSegmentationDetection) {
    EXPECT_FALSE(RFDetrHelpers::filenameIsSegmentation(
            QStringLiteral("rfdetr-base-f16.gguf")));
    EXPECT_FALSE(RFDetrHelpers::filenameIsSegmentation(
            QStringLiteral("rfdetr-large-f16.gguf")));
}

TEST(RFDetrHelpers, FilenameIsSegmentationSeg) {
    EXPECT_TRUE(RFDetrHelpers::filenameIsSegmentation(
            QStringLiteral("rfdetr-seg-nano-f16.gguf")));
    EXPECT_TRUE(RFDetrHelpers::filenameIsSegmentation(
            QStringLiteral("rfdetr-seg-medium-f16.gguf")));
}

TEST(RFDetrHelpers, CatalogMirror) {
    const QVector<RFDetrModelEntry> all = RFDetrHelpers::catalogModels();
    ASSERT_GT(all.size(), 0);
    EXPECT_EQ(all.size(),
              RFDetrHelpers::detectionModels().size() +
                      RFDetrHelpers::segmentationModels().size());
    bool foundBase = false;
    for (const RFDetrModelEntry& e : all) {
        EXPECT_FALSE(e.filename.isEmpty());
        EXPECT_TRUE(e.downloadUrl.startsWith(QStringLiteral("https://")));
        if (e.filename == QStringLiteral("rfdetr-base-f16.gguf")) {
            foundBase = true;
        }
    }
    EXPECT_TRUE(foundBase);

    RFDetrModelEntry entry;
    EXPECT_TRUE(RFDetrHelpers::findModelByFilename(
            QStringLiteral("rfdetr-base-f16.gguf"), &entry));
    EXPECT_FALSE(RFDetrHelpers::findModelByFilename(
            QStringLiteral("does-not-exist.gguf"), nullptr));
}

TEST(RFDetrHelpers, DrawDetectionsSmoke) {
    QImage img(320, 240, QImage::Format_RGB888);
    img.fill(Qt::black);

    RFDetrDetection d;
    d.classId = 0;
    d.className = QStringLiteral("person");
    d.score = 0.9f;
    d.x1 = 10;
    d.y1 = 20;
    d.x2 = 100;
    d.y2 = 200;

    QVector<RFDetrDetection> dets{d};
    RFDetrHelpers::drawDetections(&img, dets);

    // The box edge pixel at (10, 20) must no longer be pure black.
    const QRgb edge = img.pixel(10, 20);
    EXPECT_TRUE(qRed(edge) != 0 || qGreen(edge) != 0 || qBlue(edge) != 0);

    // Out-of-image boxes must be clipped, not crash.
    RFDetrDetection bad = d;
    bad.x1 = -50;
    bad.y1 = -50;
    bad.x2 = 500;
    bad.y2 = 500;
    QVector<RFDetrDetection> badDets{bad};
    RFDetrHelpers::drawDetections(&img, badDets);

    // Null image is a no-op.
    RFDetrHelpers::drawDetections(nullptr, dets);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
