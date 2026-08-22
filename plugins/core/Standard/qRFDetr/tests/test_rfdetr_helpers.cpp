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
#include <QGuiApplication>
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
    EXPECT_TRUE(out.modelVariant == QStringLiteral("base"));
    EXPECT_TRUE(out.imageSize == 640);
    EXPECT_TRUE(out.numClasses == 80);
    EXPECT_FALSE(out.segmentation);
    EXPECT_TRUE(out.detections.size() == 2);
    EXPECT_TRUE(out.totalDetected == 2);

    const RFDetrDetection& person = out.detections.at(0);
    EXPECT_TRUE(person.classId == 0u);
    EXPECT_TRUE(person.className == QStringLiteral("person"));
    EXPECT_NEAR(person.score, 0.93f, 1e-4f);
    EXPECT_NEAR(person.x1, 10.5f, 1e-3f);
    EXPECT_NEAR(person.y2, 220.0f, 1e-3f);

    const RFDetrDetection& car = out.detections.at(1);
    EXPECT_TRUE(car.classId == 2u);
    EXPECT_TRUE(car.className == QStringLiteral("car"));
}

TEST(RFDetrHelpers, ParseDetectionsJsonEmptyArray) {
    const QByteArray json =
            "{\"model\":\"nano\",\"segmentation\":1,\"image_size\":640,"
            "\"num_classes\":80,\"num_queries\":300,"
            "\"image\":{\"width\":100,\"height\":100},\"detections\":[]}";
    RFDetrRunResult out;
    ASSERT_TRUE(RFDetrHelpers::parseDetectionsJson(json, &out));
    EXPECT_TRUE(out.segmentation);
    EXPECT_TRUE(out.detections.size() == 0);
}

TEST(RFDetrHelpers, ParseDetectionsJsonInvalid) {
    RFDetrRunResult out;
    EXPECT_FALSE(RFDetrHelpers::parseDetectionsJson("not json", &out));
    EXPECT_FALSE(RFDetrHelpers::parseDetectionsJson("[]", &out));
}

TEST(RFDetrHelpers, ParseModelInfoJsonBasic) {
    const QByteArray json =
            "{\"variant\":\"base\",\"num_classes\":3,"
            "\"class_names\":[\"person\",\"bicycle\",\"car\"]}";
    QStringList names;
    ASSERT_TRUE(RFDetrHelpers::parseModelInfoJson(json, &names));
    ASSERT_EQ(names.size(), 3);
    EXPECT_EQ(names.at(0), QStringLiteral("person"));
    EXPECT_EQ(names.at(1), QStringLiteral("bicycle"));
    EXPECT_EQ(names.at(2), QStringLiteral("car"));
}

TEST(RFDetrHelpers, ParseModelInfoJsonEscapes) {
    // Class names with quotes / backslashes must round-trip unescaped.
    const QByteArray json =
            "{\"variant\":\"x\",\"num_classes\":2,"
            "\"class_names\":[\"a\\\"b\",\"c\\\\d\"]}";
    QStringList names;
    ASSERT_TRUE(RFDetrHelpers::parseModelInfoJson(json, &names));
    ASSERT_EQ(names.size(), 2);
    EXPECT_EQ(names.at(0), QStringLiteral("a\"b"));
    EXPECT_EQ(names.at(1), QStringLiteral("c\\d"));
}

TEST(RFDetrHelpers, ParseModelInfoJsonPreservesEmptySlots) {
    // COCO 91-class layout: empty-string slots (id 0 = background, plus
    // unused COCO ids) must be preserved so the dialog's list-row index
    // stays equal to class_id — dropping them would mis-map the allowlist.
    const QByteArray json =
            "{\"variant\":\"base\",\"num_classes\":4,"
            "\"class_names\":[\"\",\"person\",\"\",\"car\"]}";
    QStringList names;
    ASSERT_TRUE(RFDetrHelpers::parseModelInfoJson(json, &names));
    ASSERT_EQ(names.size(), 4);
    EXPECT_TRUE(names.at(0).isEmpty());   // slot 0 preserved
    EXPECT_EQ(names.at(1), QStringLiteral("person"));
    EXPECT_TRUE(names.at(2).isEmpty());   // slot 2 preserved
    EXPECT_EQ(names.at(3), QStringLiteral("car"));
}

TEST(RFDetrHelpers, ParseModelInfoJsonMissingNames) {
    QStringList names;
    EXPECT_FALSE(RFDetrHelpers::parseModelInfoJson("not json", &names));
    EXPECT_FALSE(RFDetrHelpers::parseModelInfoJson("[]", &names));
    EXPECT_FALSE(RFDetrHelpers::parseModelInfoJson(
            "{\"variant\":\"base\",\"num_classes\":80}", &names));
    EXPECT_FALSE(RFDetrHelpers::parseModelInfoJson(
            "{\"variant\":\"base\",\"num_classes\":0,"
            "\"class_names\":[]}",
            &names));
    // NULL output is rejected without crashing.
    EXPECT_FALSE(RFDetrHelpers::parseModelInfoJson("{\"a\":1}", nullptr));
}

TEST(RFDetrHelpers, ClassColorDeterministic) {
    EXPECT_TRUE(RFDetrHelpers::classColor(0) == RFDetrHelpers::classColor(0));
    EXPECT_TRUE(RFDetrHelpers::classColor(20) == RFDetrHelpers::classColor(0));
    EXPECT_TRUE(RFDetrHelpers::classColor(21) == RFDetrHelpers::classColor(1));
    EXPECT_TRUE(RFDetrHelpers::classColor(3) != RFDetrHelpers::classColor(4));
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
    // 44 models = 11 variants (5 detection + 6 segmentation) x 4 quants
    // (f32, f16, q8_0, q4_K), mirroring the AICore catalog.
    const QVector<RFDetrModelEntry> all = RFDetrHelpers::catalogModels();
    ASSERT_EQ(all.size(), 44);
    EXPECT_EQ(RFDetrHelpers::detectionModels().size(), 20);
    EXPECT_EQ(RFDetrHelpers::segmentationModels().size(), 24);
    EXPECT_TRUE(all.size() ==
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

TEST(RFDetrHelpers, ModelDisplayLabelDoesNotDuplicateQuantNote) {
    RFDetrModelEntry entry;
    entry.displayName = QStringLiteral("RF-DETR Small - F16 - half precision");
    entry.quantNote = QStringLiteral("half precision");
    const QString label = RFDetrHelpers::modelDisplayLabel(entry);
    EXPECT_EQ(label.count(entry.quantNote), 1);

    entry.displayName = QStringLiteral("RF-DETR Small");
    EXPECT_EQ(RFDetrHelpers::modelDisplayLabel(entry).count(entry.quantNote),
              1);
}

TEST(RFDetrHelpers, PackedRgb888RemovesRowPadding) {
    QImage image(3, 2, QImage::Format_RGB888);
    ASSERT_GT(image.bytesPerLine(), image.width() * 3);
    for (int y = 0; y < image.height(); ++y) {
        uchar* row = image.scanLine(y);
        for (int x = 0; x < image.width() * 3; ++x) {
            row[x] = static_cast<uchar>(y * 32 + x);
        }
    }

    QByteArray scratch;
    const uchar* packed = RFDetrHelpers::packedRgb888Data(image, &scratch);
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(scratch.size(), image.width() * image.height() * 3);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width() * 3; ++x) {
            EXPECT_EQ(packed[y * image.width() * 3 + x],
                      static_cast<uchar>(y * 32 + x));
        }
    }
}

TEST(RFDetrHelpers, DrawDetectionsSmoke) {
    QImage img(320, 240, QImage::Format_ARGB32);
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

TEST(RFDetrHelpers, DrawDetectionsLabelStaysInsideImage) {
    QImage img(100, 100, QImage::Format_RGB888);
    img.fill(Qt::black);

    // Box hugging the top edge: the banner anchor (above the box) would
    // land at a negative y, i.e. fully off-canvas. The fixed logic flips
    // the banner below the box top.
    RFDetrDetection top;
    top.classId = 1;
    top.className = QStringLiteral("top");
    top.score = 0.9f;
    top.x1 = 40;
    top.y1 = 2;
    top.x2 = 60;
    top.y2 = 30;

    // Box hugging the left edge: the banner anchor x = 0 would render the
    // text half-off-canvas; it must be clamped inside the image.
    RFDetrDetection left;
    left.classId = 2;
    left.className = QStringLiteral("left");
    left.score = 0.8f;
    left.x1 = 0;
    left.y1 = 40;
    left.x2 = 20;
    left.y2 = 60;

    QVector<RFDetrDetection> dets{top, left};
    RFDetrHelpers::drawDetections(&img, dets);

    const auto nearColor = [](QRgb px, QRgb ref) {
        return qAbs(qRed(px) - qRed(ref)) < 40 &&
               qAbs(qGreen(px) - qGreen(ref)) < 40 &&
               qAbs(qBlue(px) - qBlue(ref)) < 40;
    };
    const QRgb c1 = RFDetrHelpers::classColor(1);
    const QRgb c2 = RFDetrHelpers::classColor(2);

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

TEST(RFDetrHelpers, DrawMaskTintFromRawBytes) {
    // A tiny raw mask (0/255, row-major, model resolution) must be tinted
    // over the frame without a PNG round-trip (the AICore video path hands
    // raw bytes; drawDetections wraps them as a zero-copy Grayscale8 view).
    QImage img(64, 64, QImage::Format_ARGB32);
    img.fill(Qt::black);

    RFDetrDetection d;
    d.classId = 1;
    // Box placed clear of the mask blob (which scales to the top-left
    // 32x32) so its label banner does not overlap the background probe.
    d.x1 = 32;
    d.y1 = 32;
    d.x2 = 55;
    d.y2 = 55;
    d.maskWidth = 4;
    d.maskHeight = 4;
    d.maskRaw = QByteArray(16, char(0));
    // foreground blob in the top-left quarter of the mask
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            d.maskRaw[y * 4 + x] = char(255);
        }
    }

    QVector<RFDetrDetection> dets{d};
    RFDetrHelpers::drawDetections(&img, dets, 1.0f /*opaque tint*/);

    // Mask pixel (0,0) maps to frame pixel (0,0) — must carry the class
    // color (classId 1 -> palette[1]).
    const QRgb tinted = img.pixel(0, 0);
    EXPECT_NE(qRed(tinted) | qGreen(tinted) | qBlue(tinted), 0);
    const QRgb expected = RFDetrHelpers::classColor(1);
    EXPECT_EQ(qRed(tinted), qRed(expected));
    EXPECT_EQ(qGreen(tinted), qGreen(expected));
    EXPECT_EQ(qBlue(tinted), qBlue(expected));

    // Outside the mask blob and clear of the box border + label banner
    // (frame pixel 60,60: bottom-right corner, away from all overlays).
    const QRgb bg = img.pixel(60, 60);
    EXPECT_EQ(qRed(bg), 0);
    EXPECT_EQ(qGreen(bg), 0);
    EXPECT_EQ(qBlue(bg), 0);

    // A detection without mask bytes is a no-op (no crash).
    RFDetrDetection plain;
    QVector<RFDetrDetection> plainDets{plain};
    RFDetrHelpers::drawDetections(&img, plainDets);
}

int main(int argc, char** argv) {
    // QPainter text rendering requires a QGuiApplication instance.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QGuiApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
