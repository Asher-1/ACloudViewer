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
    // 63 models = 21 variants (10 detection + 10 segmentation + 1 depth)
    // x 3 quants (f32, f16, q8_0), mirroring the AICore catalog.
    const QVector<YOLOModelEntry> all = YOLOHelpers::catalogModels();
    ASSERT_EQ(all.size(), 63);
    // Each task tab filters on the GGUF task: pure detect / segment / depth.
    EXPECT_EQ(YOLOHelpers::detectionModels().size(), 30);
    EXPECT_EQ(YOLOHelpers::segmentModels().size(), 30);
    EXPECT_EQ(YOLOHelpers::depthModels().size(), 3);
    EXPECT_EQ(YOLOHelpers::taskModels(QStringLiteral("detect")).size(), 30);
    EXPECT_EQ(YOLOHelpers::taskModels(QStringLiteral("segment")).size(), 30);
    EXPECT_EQ(YOLOHelpers::taskModels(QStringLiteral("depth")).size(), 3);
    EXPECT_TRUE(all.size() == YOLOHelpers::detectionModels().size() +
                                      YOLOHelpers::segmentModels().size() +
                                      YOLOHelpers::depthModels().size());
    for (const YOLOModelEntry& e : YOLOHelpers::depthModels()) {
        EXPECT_TRUE(e.depthCapable);
        EXPECT_EQ(e.task, QStringLiteral("depth"));
    }
    for (const YOLOModelEntry& e : YOLOHelpers::detectionModels()) {
        EXPECT_EQ(e.task, QStringLiteral("detect"));
    }
    for (const YOLOModelEntry& e : YOLOHelpers::segmentModels()) {
        EXPECT_EQ(e.task, QStringLiteral("segment"));
        EXPECT_FALSE(e.depthCapable);
    }

    bool foundV8Nano = false;
    bool found26Nano = false;
    bool foundDepth = false;
    bool foundSeg = false;
    for (const YOLOModelEntry& e : all) {
        EXPECT_FALSE(e.filename.isEmpty());
        EXPECT_TRUE(e.downloadUrl.startsWith(QStringLiteral("https://")));
        EXPECT_TRUE(
                e.downloadUrl.contains(QStringLiteral("/yolo_gguf_models/")));
        if (e.filename == QStringLiteral("yolov8n-f16.gguf")) {
            foundV8Nano = true;
            EXPECT_FALSE(e.end2end);  // YOLOv8: classic NMS head
            EXPECT_FALSE(e.depthCapable);
            EXPECT_EQ(e.task, QStringLiteral("detect"));
        }
        if (e.filename == QStringLiteral("yolo26n-f16.gguf")) {
            found26Nano = true;
            EXPECT_TRUE(e.end2end);  // YOLO26: end-to-end head
            EXPECT_FALSE(e.depthCapable);
            EXPECT_EQ(e.task, QStringLiteral("detect"));
        }
        if (e.filename == QStringLiteral("yolo26n-depth-f16.gguf")) {
            foundDepth = true;
            EXPECT_TRUE(e.end2end);
            EXPECT_TRUE(e.depthCapable);
            EXPECT_EQ(e.task, QStringLiteral("depth"));
        }
        if (e.filename == QStringLiteral("yolov8n-seg-f16.gguf")) {
            foundSeg = true;
            EXPECT_EQ(e.task, QStringLiteral("segment"));
            EXPECT_FALSE(e.depthCapable);
        }
    }
    EXPECT_TRUE(foundV8Nano);
    EXPECT_TRUE(found26Nano);
    EXPECT_TRUE(foundDepth);
    EXPECT_TRUE(foundSeg);

    YOLOModelEntry entry;
    EXPECT_TRUE(YOLOHelpers::findModelByFilename(
            QStringLiteral("yolov8n-f16.gguf"), &entry));
    EXPECT_TRUE(entry.displayName.contains(QStringLiteral("YOLOv8 Nano")));
    EXPECT_EQ(entry.task, QStringLiteral("detect"));
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

TEST(YOLOHelpers, DrawDetectionsLabelStaysInsideImage) {
    QImage img(100, 100, QImage::Format_RGB888);
    img.fill(Qt::black);

    // Box hugging the top edge: the banner anchor (above the box) would
    // land at a negative y, i.e. fully off-canvas. The fixed logic flips
    // the banner below the box top.
    YOLODetection top;
    top.classId = 1;
    top.className = QStringLiteral("top");
    top.score = 0.9f;
    top.x1 = 40;
    top.y1 = 2;
    top.x2 = 60;
    top.y2 = 30;

    // Box hugging the left edge: the banner anchor x = 0 would render the
    // text half-off-canvas; it must be clamped inside the image.
    YOLODetection left;
    left.classId = 2;
    left.className = QStringLiteral("left");
    left.score = 0.8f;
    left.x1 = 0;
    left.y1 = 40;
    left.x2 = 20;
    left.y2 = 60;

    QVector<YOLODetection> dets{top, left};
    YOLOHelpers::drawDetections(&img, dets);

    const auto nearColor = [](QRgb px, QRgb ref) {
        return qAbs(qRed(px) - qRed(ref)) < 40 &&
               qAbs(qGreen(px) - qGreen(ref)) < 40 &&
               qAbs(qBlue(px) - qBlue(ref)) < 40;
    };
    const QRgb c1 = YOLOHelpers::classColor(1);
    const QRgb c2 = YOLOHelpers::classColor(2);

    // The flipped banner of `top` renders below the box top (y >= 4); its
    // right-hand blank region (beyond the white text) must be on-canvas.
    bool bannerInsideTop = false;
    for (int y = 6; y < 16 && !bannerInsideTop; ++y) {
        for (int x = 62; x < 90; ++x) {
            if (nearColor(img.pixel(x, y), c1)) {
                bannerInsideTop = true;
                break;
            }
        }
    }
    EXPECT_TRUE(bannerInsideTop);

    // The banner of `left` is clamped to x >= 2; its blank region right of
    // the text must be visible at the box's y range.
    bool bannerInsideLeft = false;
    for (int y = 22; y < 32 && !bannerInsideLeft; ++y) {
        for (int x = 68; x < 90; ++x) {
            if (nearColor(img.pixel(x, y), c2)) {
                bannerInsideLeft = true;
                break;
            }
        }
    }
    EXPECT_TRUE(bannerInsideLeft);
}

TEST(YOLOHelpers, DrawSegmentationTintAndClipping) {
    QImage img(120, 90, QImage::Format_RGB888);
    img.fill(Qt::black);

    // One detection with a full-canvas mask (2x2, every pixel foreground)
    // scaled to the 120x90 image, plus a second with an empty mask.
    YOLODetection d;
    d.classId = 0;
    d.className = QStringLiteral("person");
    d.score = 0.9f;
    d.x1 = 0;
    d.y1 = 0;
    d.x2 = 119;
    d.y2 = 89;

    YOLOSegMask full;
    full.w = 2;
    full.h = 2;
    full.bits = QByteArray("\x01\x01\x01\x01", 4);
    YOLOSegMask empty;
    empty.w = 2;
    empty.h = 2;
    empty.bits = QByteArray("\x00\x00\x00\x00", 4);

    QVector<YOLOSegMask> masks{full, empty};
    QVector<YOLODetection> dets{d, d};
    YOLOHelpers::drawSegmentation(&img, masks, dets);
    EXPECT_EQ(img.format(), QImage::Format_ARGB32);  // converted in place

    // Foreground mask pixels are tinted away from pure black.
    const QRgb fg = img.pixel(10, 10);
    EXPECT_TRUE(qRed(fg) != 0 || qGreen(fg) != 0 || qBlue(fg) != 0);

    // Undersized mask data is skipped (no crash, image untouched there).
    YOLOSegMask bad;
    bad.w = 100;
    bad.h = 100;
    bad.bits = QByteArray("\x01", 1);  // 1 byte < 100*100
    QVector<YOLOSegMask> badMasks{bad};
    YOLOHelpers::drawSegmentation(&img, badMasks, dets);

    // Null image / empty masks are no-ops.
    YOLOHelpers::drawSegmentation(nullptr, masks, dets);
    QImage untouched(4, 4, QImage::Format_RGB888);
    untouched.fill(Qt::black);
    YOLOHelpers::drawSegmentation(&untouched, {}, {});
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
