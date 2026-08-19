// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QGuiApplication>
#include <QImage>

#include "YOLOModelCatalog.h"

TEST(YOLOHelpers, ParseDetectionsJsonBasic) {
    const QByteArray json =
            "{\"model\":\"yolov8n\",\"task\":\"detect\",\"image_size\":640,"
            "\"num_classes\":80,\"end2end\":0,"
            "\"image\":{\"width\":1280,\"height\":720},"
            "\"detections\":["
            "{\"class_id\":0,\"class_name\":\"person\",\"score\":0.93,"
            "\"box\":[10.5,20.0,110.5,220.0]},"
            "{\"class_id\":2,\"class_name\":\"car\",\"score\":0.81,"
            "\"box\":[300,400,500,600]}]}";
    YOLORunResult out;
    ASSERT_TRUE(YOLOHelpers::parseDetectionsJson(json, &out));
    EXPECT_TRUE(out.modelVariant == QStringLiteral("yolov8n"));
    EXPECT_TRUE(out.imageSize == 640);
    EXPECT_TRUE(out.numClasses == 80);
    EXPECT_FALSE(out.end2end);
    EXPECT_TRUE(out.detections.size() == 2);
    EXPECT_TRUE(out.totalDetected == 2);

    const YOLODetection& person = out.detections.at(0);
    EXPECT_TRUE(person.classId == 0u);
    EXPECT_TRUE(person.className == QStringLiteral("person"));
    EXPECT_NEAR(person.score, 0.93f, 1e-4f);
    EXPECT_NEAR(person.x1, 10.5f, 1e-3f);
    EXPECT_NEAR(person.y2, 220.0f, 1e-3f);

    const YOLODetection& car = out.detections.at(1);
    EXPECT_TRUE(car.classId == 2u);
    EXPECT_TRUE(car.className == QStringLiteral("car"));
}

TEST(YOLOHelpers, ParseDetectionsJsonEmptyArray) {
    // yolo26 models report end2end = 1 (NMS is baked into the head).
    const QByteArray json =
            "{\"model\":\"yolo26n\",\"task\":\"detect\",\"image_size\":640,"
            "\"num_classes\":80,\"end2end\":1,"
            "\"image\":{\"width\":100,\"height\":100},\"detections\":[]}";
    YOLORunResult out;
    ASSERT_TRUE(YOLOHelpers::parseDetectionsJson(json, &out));
    EXPECT_TRUE(out.end2end);
    EXPECT_TRUE(out.detections.size() == 0);
}

TEST(YOLOHelpers, ParseDetectionsJsonInvalid) {
    YOLORunResult out;
    EXPECT_FALSE(YOLOHelpers::parseDetectionsJson("not json", &out));
    EXPECT_FALSE(YOLOHelpers::parseDetectionsJson("[]", &out));
}

TEST(YOLOHelpers, ParseDepthStatsJsonBasic) {
    const QByteArray json =
            "{\"model\":\"yolo26n-depth\",\"task\":\"depth\",\"image_size\":"
            "640,"
            "\"image\":{\"width\":640,\"height\":480},"
            "\"depth_width\":640,\"depth_height\":480,"
            "\"min_depth\":0.5,\"max_depth\":80.0,\"mean_depth\":5.2,"
            "\"p95_depth\":12.3,\"valid_pixels\":307200}";
    YOLODepthStats stats;
    ASSERT_TRUE(YOLOHelpers::parseDepthStatsJson(json, &stats));
    EXPECT_EQ(stats.width, 640);
    EXPECT_EQ(stats.height, 480);
    EXPECT_NEAR(stats.minDepth, 0.5, 1e-9);
    EXPECT_NEAR(stats.maxDepth, 80.0, 1e-9);
    EXPECT_NEAR(stats.meanDepth, 5.2, 1e-9);
    EXPECT_NEAR(stats.p95Depth, 12.3, 1e-9);
    EXPECT_EQ(stats.validPixels, 307200LL);
}

TEST(YOLOHelpers, ParseDepthStatsJsonInvalid) {
    YOLODepthStats stats;
    EXPECT_FALSE(YOLOHelpers::parseDepthStatsJson("not json", &stats));
    EXPECT_FALSE(YOLOHelpers::parseDepthStatsJson("[]", &stats));
    // A zero-sized depth map is rejected even with valid statistics.
    EXPECT_FALSE(YOLOHelpers::parseDepthStatsJson(
            "{\"depth_width\":0,\"depth_height\":480,\"min_depth\":1.0,"
            "\"max_depth\":2.0,\"mean_depth\":1.5,\"p95_depth\":1.8,"
            "\"valid_pixels\":100}",
            &stats));
}

TEST(YOLOHelpers, ClassColorDeterministic) {
    EXPECT_TRUE(YOLOHelpers::classColor(0) == YOLOHelpers::classColor(0));
    EXPECT_TRUE(YOLOHelpers::classColor(20) == YOLOHelpers::classColor(0));
    EXPECT_TRUE(YOLOHelpers::classColor(21) == YOLOHelpers::classColor(1));
    EXPECT_TRUE(YOLOHelpers::classColor(3) != YOLOHelpers::classColor(4));
}

TEST(YOLOHelpers, FilenameIsDepthDetection) {
    EXPECT_FALSE(
            YOLOHelpers::filenameIsDepth(QStringLiteral("yolov8n-f16.gguf")));
    EXPECT_FALSE(
            YOLOHelpers::filenameIsDepth(QStringLiteral("yolo26x-q8_0.gguf")));
}

TEST(YOLOHelpers, FilenameIsDepthDepth) {
    EXPECT_TRUE(YOLOHelpers::filenameIsDepth(
            QStringLiteral("yolo26n-depth-f16.gguf")));
    // Case-insensitive so catalog names and user-picked files match alike.
    EXPECT_TRUE(YOLOHelpers::filenameIsDepth(
            QStringLiteral("YOLO26N-DEPTH-F32.GGUF")));
}

TEST(YOLOHelpers, CatalogMirror) {
    // 33 models = 11 variants (10 detection + 1 depth) x 3 quants
    // (f32, f16, q8_0), mirroring the AICore catalog.
    const QVector<YOLOModelEntry> all = YOLOHelpers::catalogModels();
    ASSERT_EQ(all.size(), 33);
    EXPECT_EQ(YOLOHelpers::detectionModels().size(), 30);
    EXPECT_EQ(YOLOHelpers::depthModels().size(), 3);
    EXPECT_TRUE(all.size() == YOLOHelpers::detectionModels().size() +
                                      YOLOHelpers::depthModels().size());
    for (const YOLOModelEntry& e : YOLOHelpers::depthModels()) {
        EXPECT_TRUE(e.depthCapable);
    }

    bool foundV8Nano = false;
    bool found26Nano = false;
    bool foundDepth = false;
    for (const YOLOModelEntry& e : all) {
        EXPECT_FALSE(e.filename.isEmpty());
        EXPECT_TRUE(e.downloadUrl.startsWith(QStringLiteral("https://")));
        EXPECT_TRUE(
                e.downloadUrl.contains(QStringLiteral("/yolo_gguf_models/")));
        if (e.filename == QStringLiteral("yolov8n-f16.gguf")) {
            foundV8Nano = true;
            EXPECT_FALSE(e.end2end);  // YOLOv8: classic NMS head
            EXPECT_FALSE(e.depthCapable);
        }
        if (e.filename == QStringLiteral("yolo26n-f16.gguf")) {
            found26Nano = true;
            EXPECT_TRUE(e.end2end);  // YOLO26: end-to-end head
            EXPECT_FALSE(e.depthCapable);
        }
        if (e.filename == QStringLiteral("yolo26n-depth-f16.gguf")) {
            foundDepth = true;
            EXPECT_TRUE(e.end2end);
            EXPECT_TRUE(e.depthCapable);
        }
    }
    EXPECT_TRUE(foundV8Nano);
    EXPECT_TRUE(found26Nano);
    EXPECT_TRUE(foundDepth);

    YOLOModelEntry entry;
    EXPECT_TRUE(YOLOHelpers::findModelByFilename(
            QStringLiteral("yolov8n-f16.gguf"), &entry));
    EXPECT_TRUE(entry.displayName.contains(QStringLiteral("YOLOv8 Nano")));
    EXPECT_FALSE(YOLOHelpers::findModelByFilename(
            QStringLiteral("does-not-exist.gguf"), nullptr));
}

TEST(YOLOHelpers, ModelDisplayLabelDoesNotDuplicateQuantNote) {
    YOLOModelEntry entry;
    entry.displayName =
            QStringLiteral("YOLOv8 Nano — F16 — half precision (recommended)");
    entry.quantNote = QStringLiteral("F16 — half precision (recommended)");
    const QString label = YOLOHelpers::modelDisplayLabel(entry);
    EXPECT_EQ(label.count(entry.quantNote), 1);

    entry.displayName = QStringLiteral("YOLOv8 Nano");
    EXPECT_EQ(YOLOHelpers::modelDisplayLabel(entry).count(entry.quantNote), 1);
}

TEST(YOLOHelpers, PackedRgb888RemovesRowPadding) {
    QImage image(3, 2, QImage::Format_RGB888);
    ASSERT_GT(image.bytesPerLine(), image.width() * 3);
    for (int y = 0; y < image.height(); ++y) {
        uchar* row = image.scanLine(y);
        for (int x = 0; x < image.width() * 3; ++x) {
            row[x] = static_cast<uchar>(y * 32 + x);
        }
    }

    QByteArray scratch;
    const uchar* packed = YOLOHelpers::packedRgb888Data(image, &scratch);
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(scratch.size(), image.width() * image.height() * 3);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width() * 3; ++x) {
            EXPECT_EQ(packed[y * image.width() * 3 + x],
                      static_cast<uchar>(y * 32 + x));
        }
    }
}

TEST(YOLOHelpers, DrawDetectionsSmoke) {
    QImage img(320, 240, QImage::Format_ARGB32);
    img.fill(Qt::black);

    YOLODetection d;
    d.classId = 0;
    d.className = QStringLiteral("person");
    d.score = 0.9f;
    d.x1 = 10;
    d.y1 = 20;
    d.x2 = 100;
    d.y2 = 200;

    QVector<YOLODetection> dets{d};
    YOLOHelpers::drawDetections(&img, dets);

    // The box edge pixel at (10, 20) must no longer be pure black.
    const QRgb edge = img.pixel(10, 20);
    EXPECT_TRUE(qRed(edge) != 0 || qGreen(edge) != 0 || qBlue(edge) != 0);

    // Out-of-image boxes must be clipped, not crash.
    YOLODetection bad = d;
    bad.x1 = -50;
    bad.y1 = -50;
    bad.x2 = 500;
    bad.y2 = 500;
    QVector<YOLODetection> badDets{bad};
    YOLOHelpers::drawDetections(&img, badDets);

    // Null image is a no-op.
    YOLOHelpers::drawDetections(nullptr, dets);
}

TEST(YOLOHelpers, DepthColorImageTurboRamp) {
    // 4x1 map: near (t=0), mid (t=0.5), far (t=1), invalid.
    const float depth[4] = {1.0f, 2.0f, 3.0f, -1.0f};
    const QImage img = YOLOHelpers::depthColorImage(depth, 4, 1, 1.0, 3.0);
    ASSERT_FALSE(img.isNull());
    EXPECT_EQ(img.format(), QImage::Format_RGB888);

    const QRgb nearPx = img.pixel(0, 0);  // t=0 -> blue side of the ramp
    EXPECT_GT(qBlue(nearPx), qRed(nearPx));
    const QRgb midPx = img.pixel(1, 0);  // t=0.5 -> green dominant
    EXPECT_GT(qGreen(midPx), qRed(midPx));
    EXPECT_GT(qGreen(midPx), qBlue(midPx));
    const QRgb farPx = img.pixel(2, 0);  // t=1 -> red side of the ramp
    EXPECT_GT(qRed(farPx), qBlue(farPx));
    const QRgb badPx = img.pixel(3, 0);  // invalid depth renders black
    EXPECT_EQ(qRed(badPx) | qGreen(badPx) | qBlue(badPx), 0);
}

TEST(YOLOHelpers, DepthColorImageRangeAndInvalid) {
    // All-invalid map with auto range -> null image (no valid pixels).
    const float invalid[2] = {-1.0f, 0.0f};
    EXPECT_TRUE(YOLOHelpers::depthColorImage(invalid, 2, 1).isNull());

    // Explicit range keeps invalid pixels as black instead.
    const QImage explicitRange =
            YOLOHelpers::depthColorImage(invalid, 2, 1, 1.0, 2.0);
    ASSERT_FALSE(explicitRange.isNull());
    EXPECT_EQ(qRed(explicitRange.pixel(0, 0)) |
                      qGreen(explicitRange.pixel(0, 0)) |
                      qBlue(explicitRange.pixel(0, 0)),
              0);

    // Single valid pixel: the lo == hi guard keeps the map renderable.
    const float single[1] = {5.0f};
    const QImage singleImg = YOLOHelpers::depthColorImage(single, 1, 1);
    ASSERT_FALSE(singleImg.isNull());
    EXPECT_NE(qRed(singleImg.pixel(0, 0)) | qGreen(singleImg.pixel(0, 0)) |
                      qBlue(singleImg.pixel(0, 0)),
              0);

    // Auto range (minDepth >= maxDepth) falls back to min..p95 over valid
    // pixels: {1,2,3,3} -> lo=1, hi=3, same orientation as the ramp test.
    const float depth[4] = {1.0f, 2.0f, 3.0f, 3.0f};
    const QImage autoImg = YOLOHelpers::depthColorImage(depth, 4, 1);
    ASSERT_FALSE(autoImg.isNull());
    EXPECT_GT(qBlue(autoImg.pixel(0, 0)), qRed(autoImg.pixel(0, 0)));
    EXPECT_GT(qRed(autoImg.pixel(3, 0)), qBlue(autoImg.pixel(3, 0)));
}

TEST(YOLOHelpers, DrawDepthLegendSmoke) {
    QImage img(320, 240, QImage::Format_RGB888);
    img.fill(Qt::black);

    YOLOHelpers::drawDepthLegend(&img, 1.0, 10.0);
    EXPECT_EQ(img.format(), QImage::Format_ARGB32);  // converted in place

    // The color bar at the top-right corner must no longer be pure black.
    // (barW = max(8, w/60) = 8, x0 = w - barW - 8 = 304, y0 = 8.)
    bool colored = false;
    for (int y = 8; y < 40 && !colored; ++y) {
        for (int x = 300; x < 312; ++x) {
            const QRgb px = img.pixel(x, y);
            if (qRed(px) | qGreen(px) | qBlue(px)) {
                colored = true;
                break;
            }
        }
    }
    EXPECT_TRUE(colored);

    // Degenerate range and null image are no-ops (no crash).
    YOLOHelpers::drawDepthLegend(&img, 5.0, 5.0);
    YOLOHelpers::drawDepthLegend(nullptr, 1.0, 2.0);
}

int main(int argc, char** argv) {
    // QPainter text rendering requires a QGuiApplication instance.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QGuiApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
