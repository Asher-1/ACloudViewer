// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Pure-logic unit tests for the qRMBG helpers (catalog mirror, info JSON,
// alpha statistics, checkerboard preview). No GGUF model required.

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QImage>

#include "RMBGModelCatalog.h"

TEST(RMBGHelpers, ParseInfoJsonBasic) {
    const QByteArray json =
            "{\"model\":\"RMBG-2.0 (BiRefNet-Swin-L)\",\"input_size\":1024,"
            "\"backend\":\"cpu\",\"math_profile\":\"default\","
            "\"device\":\"cpu\",\"threads\":4}";
    RMBGRunResult out;
    ASSERT_TRUE(RMBGHelpers::parseInfoJson(json, &out));
    EXPECT_TRUE(out.modelVariant.contains(QStringLiteral("RMBG-2.0")));
    EXPECT_TRUE(out.inputSize == 1024);
    EXPECT_TRUE(out.backend == QStringLiteral("cpu"));
    EXPECT_TRUE(out.mathProfile == QStringLiteral("default"));
    EXPECT_TRUE(out.resolvedDevice == QStringLiteral("cpu"));
    EXPECT_TRUE(out.infoJson == json);
}

TEST(RMBGHelpers, ParseInfoJsonInvalid) {
    RMBGRunResult out;
    EXPECT_FALSE(RMBGHelpers::parseInfoJson("not json", &out));
    EXPECT_FALSE(RMBGHelpers::parseInfoJson("[]", &out));
}

TEST(RMBGHelpers, CatalogMirror) {
    // 3 quantized variants (f32, f16, q8) mirror the AICore catalog.
    const QVector<RMBGModelEntry> all = RMBGHelpers::catalogModels();
    ASSERT_EQ(all.size(), 3);
    bool foundRmbg = false;
    for (const RMBGModelEntry& e : all) {
        EXPECT_FALSE(e.filename.isEmpty());
        EXPECT_TRUE(e.downloadUrl.startsWith(QStringLiteral("https://")));
        if (e.filename == QStringLiteral("rmbg_f16.gguf")) {
            foundRmbg = true;
        }
    }
    EXPECT_TRUE(foundRmbg);

    RMBGModelEntry entry;
    EXPECT_TRUE(RMBGHelpers::findModelByFilename(
            QStringLiteral("rmbg_f16.gguf"), &entry));
    EXPECT_FALSE(RMBGHelpers::findModelByFilename(
            QStringLiteral("does-not-exist.gguf"), nullptr));
}

TEST(RMBGHelpers, ModelDisplayLabelDoesNotDuplicateQuantNote) {
    RMBGModelEntry entry;
    entry.displayName = QStringLiteral("RMBG-2.0 - F16 - half precision");
    entry.quantNote = QStringLiteral("half precision");
    const QString label = RMBGHelpers::modelDisplayLabel(entry);
    EXPECT_EQ(label.count(entry.quantNote), 1);

    entry.displayName = QStringLiteral("RMBG-2.0");
    EXPECT_EQ(RMBGHelpers::modelDisplayLabel(entry).count(entry.quantNote), 1);
}

TEST(RMBGHelpers, PackedRgb888RemovesRowPadding) {
    QImage image(3, 2, QImage::Format_RGB888);
    ASSERT_GT(image.bytesPerLine(), image.width() * 3);
    for (int y = 0; y < image.height(); ++y) {
        uchar* row = image.scanLine(y);
        for (int x = 0; x < image.width() * 3; ++x) {
            row[x] = static_cast<uchar>(y * 32 + x);
        }
    }

    QByteArray scratch;
    const uchar* packed = RMBGHelpers::packedRgb888Data(image, &scratch);
    ASSERT_NE(packed, nullptr);
    ASSERT_EQ(scratch.size(), image.width() * image.height() * 3);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width() * 3; ++x) {
            EXPECT_EQ(packed[y * image.width() * 3 + x],
                      static_cast<uchar>(y * 32 + x));
        }
    }
}

TEST(RMBGHelpers, AlphaStatsOpaque) {
    QImage img(16, 16, QImage::Format_ARGB32);
    img.fill(qRgba(255, 0, 0, 255));
    double mean = 0.0, fg = 0.0;
    RMBGHelpers::computeAlphaStats(img, &mean, &fg);
    EXPECT_NEAR(mean, 1.0, 1e-9);
    EXPECT_NEAR(fg, 1.0, 1e-9);
}

TEST(RMBGHelpers, AlphaStatsTransparent) {
    QImage img(16, 16, QImage::Format_ARGB32);
    img.fill(qRgba(255, 0, 0, 255));
    // left half fully transparent
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 8; ++x) {
            img.setPixel(x, y, qRgba(0, 0, 0, 0));
        }
    }
    double mean = 0.0, fg = 0.0;
    RMBGHelpers::computeAlphaStats(img, &mean, &fg);
    EXPECT_NEAR(mean, 0.5, 1e-9);
    EXPECT_NEAR(fg, 0.5, 1e-9);
}

TEST(RMBGHelpers, AlphaStatsNullImage) {
    double mean = 1.0, fg = 1.0;
    RMBGHelpers::computeAlphaStats(QImage(), &mean, &fg);
    EXPECT_DOUBLE_EQ(mean, 0.0);
    EXPECT_DOUBLE_EQ(fg, 0.0);
}

TEST(RMBGHelpers, CheckerboardDimensions) {
    const QImage cb = RMBGHelpers::makeCheckerboard(QSize(20, 12), 4);
    EXPECT_TRUE(cb.size() == QSize(20, 12));
    // (0,0) is white, (4,0) is the dark cell
    EXPECT_TRUE(cb.pixel(0, 0) != cb.pixel(4, 0));
    EXPECT_TRUE(cb.pixel(0, 0) == cb.pixel(8, 0));  // same parity cell
}

TEST(RMBGHelpers, CompositeTransparentPixels) {
    QImage rgba(8, 8, QImage::Format_ARGB32);
    rgba.fill(qRgba(10, 200, 30, 255));
    // one fully transparent pixel
    rgba.setPixel(3, 3, qRgba(0, 0, 0, 0));
    const QImage out = RMBGHelpers::compositeOnCheckerboard(rgba, 4);
    EXPECT_TRUE(out.size() == rgba.size());
    EXPECT_TRUE(qAlpha(out.pixel(0, 0)) == 255u);
    // Transparent pixel must reveal the checkerboard (not black).
    const QRgb t = out.pixel(3, 3);
    EXPECT_TRUE(qRed(t) != 0 || qGreen(t) != 0 || qBlue(t) != 0);
}

TEST(RMBGHelpers, ApplyAlphaThreshold) {
    QImage rgba(4, 4, QImage::Format_ARGB32);
    rgba.setPixel(0, 0, qRgba(255, 0, 0, 100));  // alpha 100 < 0.5*255
    rgba.setPixel(0, 1, qRgba(0, 255, 0, 200));  // alpha 200 >= 0.5*255
    rgba.setPixel(1, 0, qRgba(10, 20, 30, 0));   // already transparent

    const QImage out = RMBGHelpers::applyAlphaThreshold(rgba, 0.5f);
    EXPECT_EQ(out.size(), rgba.size());
    EXPECT_EQ(qAlpha(out.pixel(0, 0)), 0u);    // below cut -> transparent
    EXPECT_EQ(qAlpha(out.pixel(0, 1)), 200u);  // above cut -> kept
    EXPECT_EQ(qAlpha(out.pixel(1, 0)), 0u);    // untouched
    // RGB of the kept pixel is preserved.
    EXPECT_EQ(qRed(out.pixel(0, 1)), 0);
    EXPECT_EQ(qGreen(out.pixel(0, 1)), 255);

    // Threshold <= 0 disables the pass entirely (no-op).
    const QImage noop = RMBGHelpers::applyAlphaThreshold(rgba, 0.0f);
    EXPECT_TRUE(noop == rgba);
    // Threshold >= 1 keeps everything (cut clamps to 255).
    const QImage kept = RMBGHelpers::applyAlphaThreshold(rgba, 1.5f);
    EXPECT_TRUE(kept == rgba);
    // Null image is a safe no-op.
    EXPECT_TRUE(RMBGHelpers::applyAlphaThreshold(QImage(), 0.5f).isNull());
}

TEST(RMBGHelpers, FormatStats) {
    const QString s = RMBGHelpers::formatAlphaStats(0.823, 0.456);
    EXPECT_TRUE(s.contains(QStringLiteral("82.3")));
    EXPECT_TRUE(s.contains(QStringLiteral("45.6")));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
