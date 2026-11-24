// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Focused regression tests for the shared AICore UI helper header
// (libs/CVPluginAPI/include/ecvAICoreUiHelper.h), which every AICore plugin
// dialog (qDA3, qDeepLSD, qFaceDetect, qLightGlue, qFreeSplatter, qRFDetr,
// qRMBG, qYOLO) uses for DPI-aware sizing and compact layout.
//
// The invariants below protect the "compact layout" contract established by
// the multi-round layout fixes:
//   - spacing/margins stay tight (a regression back to 8px spacing or wide
//     margins fails here),
//   - the two-column form grid keeps its 0/1/0/1 column-stretch pattern,
//   - group boxes never expand vertically (that created blank space),
//   - DPI scaling never shrinks a nominal size (qMax floor).
//
// Runs headless: QT_QPA_PLATFORM=offscreen is forced before QApplication.

#include <gtest/gtest.h>

#include <QApplication>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScreen>
#include <QSizePolicy>
#include <QTemporaryDir>
#include <QUuid>
#include <QVBoxLayout>
#include <QWidget>
#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "QtCompat.h"
#include "ecvAICoreUiHelper.h"

// ---------------------------------------------------------------------------
//  DPI scaling invariants
// ---------------------------------------------------------------------------

TEST(UiHelpers, DpiScalingInvariants) {
    // The qMax(px, round(px*dpi/96)) floor must never shrink a nominal size:
    // on a 75% scaled display a naive multiply would make 24 -> 18 and break
    // every minimum-height contract in the plugin dialogs.
    for (int px : {1, 4, 8, 24, 72, 96, 140}) {
        EXPECT_TRUE(ecvAICoreUi::dpiScaled(px) >= px);
    }
    // Monotonic: a larger nominal size never yields a smaller result.
    for (int a = 1; a < 200; a += 7) {
        EXPECT_TRUE(ecvAICoreUi::dpiScaled(a) <= ecvAICoreUi::dpiScaled(a + 1));
    }
    // At 96 logical DPI (the offscreen-platform default) scaling is identity.
    const QScreen* screen = QGuiApplication::primaryScreen();
    if (screen && std::abs(screen->logicalDotsPerInch() - 96.0) < 0.5) {
        EXPECT_TRUE(ecvAICoreUi::dpiScaled(4) == 4);
        EXPECT_TRUE(ecvAICoreUi::dpiScaled(24) == 24);
        EXPECT_TRUE(ecvAICoreUi::dpiScaled(72) == 72);
    }
    // Derived helpers stay on the same floor.
    EXPECT_TRUE(ecvAICoreUi::controlHeight() == ecvAICoreUi::dpiScaled(24));
    EXPECT_TRUE(ecvAICoreUi::compactSpinWidth() == ecvAICoreUi::dpiScaled(72));
    EXPECT_TRUE(ecvAICoreUi::previewSize() == ecvAICoreUi::dpiScaled(96));
}

// ---------------------------------------------------------------------------
//  Compact layout invariants
// ---------------------------------------------------------------------------

TEST(UiHelpers, CompactLayoutInvariants) {
    // Tab pages use tight 4px margins and 4px vertical spacing.  A regression
    // to the old 8px spacing (the "not compact" complaint) fails here.
    QWidget host;
    auto* tabLayout = new QVBoxLayout(&host);
    ecvAICoreUi::setupTabLayout(tabLayout);
    const QMargins m = tabLayout->contentsMargins();
    EXPECT_TRUE(m.left() == 4 && m.top() == 4 && m.right() == 4 &&
                m.bottom() == 4);
    EXPECT_TRUE(tabLayout->spacing() == 4);

    // The two-column form grid: 2px row spacing, 6px column spacing and the
    // 0/1/0/1 stretch pattern that keeps labels fixed and fields expanding.
    QGridLayout grid;
    ecvAICoreUi::setupFormGrid(&grid);
    EXPECT_TRUE(grid.horizontalSpacing() == 6);
    EXPECT_TRUE(grid.verticalSpacing() == 2);
    EXPECT_TRUE(grid.columnStretch(0) == 0);
    EXPECT_TRUE(grid.columnStretch(1) == 1);
    EXPECT_TRUE(grid.columnStretch(2) == 0);
    EXPECT_TRUE(grid.columnStretch(3) == 1);

    // labelColWidth must be DPI-scaled into both label columns.
    QGridLayout grid2;
    ecvAICoreUi::setupFormGrid(&grid2, 90);
    EXPECT_TRUE(grid2.columnMinimumWidth(0) == ecvAICoreUi::dpiScaled(90));
    EXPECT_TRUE(grid2.columnMinimumWidth(2) == ecvAICoreUi::dpiScaled(90));

    // Group boxes must never claim extra vertical space (Maximum policy) and
    // keep the tightened 6/4/6/4 margins — the "blank strip" regression.
    QGroupBox box;
    auto* boxGrid = new QGridLayout(&box);
    ecvAICoreUi::tightenGroupBox(&box);
    EXPECT_TRUE(box.sizePolicy().verticalPolicy() == QSizePolicy::Maximum);
    const QMargins bg = boxGrid->contentsMargins();
    EXPECT_TRUE(bg.left() == 6 && bg.top() == 4 && bg.right() == 6 &&
                bg.bottom() == 4);
    EXPECT_TRUE(boxGrid->verticalSpacing() == 2);
}

// ---------------------------------------------------------------------------
//  Widget factory invariants
// ---------------------------------------------------------------------------

TEST(UiHelpers, WidgetFactoryInvariants) {
    // Browse buttons carry the DPI-scaled fixed width (setFixedWidth stores
    // the value in both minimum and maximum width).
    auto* browse = ecvAICoreUi::makeBrowseBtn(QStringLiteral("Browse…"));
    EXPECT_TRUE(browse->minimumWidth() == ecvAICoreUi::dpiScaled(72));
    EXPECT_TRUE(browse->maximumWidth() == ecvAICoreUi::dpiScaled(72));
    EXPECT_TRUE(browse->sizePolicy().horizontalPolicy() == QSizePolicy::Fixed);
    delete browse;

    // Fixed-width form labels scale the same way.
    auto* label = ecvAICoreUi::makeLabel(QStringLiteral("Label"), 100);
    EXPECT_TRUE(label->minimumWidth() == ecvAICoreUi::dpiScaled(100));
    EXPECT_TRUE(label->maximumWidth() == ecvAICoreUi::dpiScaled(100));
    delete label;

    // Sample-data button is teal-accented (styles the shared brand color).
    auto* sample = ecvAICoreUi::makeSampleDataBtn();
    EXPECT_TRUE(sample->styleSheet().contains(QStringLiteral("#00897b")));
    delete sample;
}

TEST(UiHelpers, ExclusiveFileCreation) {
    QTemporaryDir directory;
    EXPECT_TRUE(directory.isValid());
    const QString path =
            directory.filePath(QString::fromUtf8("archive \xc3\xa9.zip"));
    QFile first(path);
    QString error;
    EXPECT_TRUE(qtCompatOpenNewFile(first, &error));
    EXPECT_TRUE(first.write("original") == 8);
    EXPECT_TRUE(first.flush());
    QFile second(path);
    EXPECT_TRUE(!qtCompatOpenNewFile(second, &error));
    EXPECT_TRUE(!error.isEmpty() && !second.isOpen());
    first.close();
    QFile read(path);
    EXPECT_TRUE(read.open(QIODevice::ReadOnly));
    EXPECT_TRUE(read.readAll() == QByteArray("original"));
    QFile missing(directory.filePath(QStringLiteral("missing/child.zip")));
    EXPECT_TRUE(!qtCompatOpenNewFile(missing, &error) && !error.isEmpty());
}

TEST(UiHelpers, QtCompatibilityTextAndPalette) {
    // Import/export must use UTF-8 regardless of the process locale.
    QByteArray bytes = QByteArray::fromHex("c3a90ae282ac0a");
    QTextStream input(&bytes, QIODevice::ReadOnly);
    qtCompatSetTextStreamUtf8(input);
    EXPECT_TRUE(input.readLine() == QString::fromUtf8("\xc3\xa9"));
    EXPECT_TRUE(input.readLine() == QString::fromUtf8("\xe2\x82\xac"));
    QByteArray outputBytes;
    QTextStream output(&outputBytes, QIODevice::WriteOnly);
    qtCompatSetTextStreamUtf8(output);
    output << QString::fromUtf8("\xc3\xa9\n\xe2\x82\xac\n");
    output.flush();
    EXPECT_TRUE(outputBytes == bytes);

    const QUuid uuid(QStringLiteral("{01234567-89ab-cdef-0123-456789abcdef}"));
    EXPECT_TRUE(qtCompatUuidWithoutBraces(uuid) ==
                QStringLiteral("01234567-89ab-cdef-0123-456789abcdef"));
    EXPECT_TRUE(qtCompatUuidId128(uuid) ==
                QStringLiteral("0123456789abcdef0123456789abcdef"));
    EXPECT_TRUE(qtCompatUuidId128(QUuid()) == QString(32, QLatin1Char('0')));

    QPalette palette;
    const QColor text = palette.color(QPalette::Active, QPalette::Text);
    const QColor placeholder("#8e8e93");
    qtCompatSetPlaceholderTextColor(palette, QPalette::Active, placeholder);
    EXPECT_TRUE(palette.color(QPalette::Active, QPalette::Text) == text);
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    EXPECT_TRUE(palette.color(QPalette::Active, QPalette::PlaceholderText) ==
                placeholder);
#endif
#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
    QImage image(2, 1, qtCompatQImageFormatGrayscale16());
    EXPECT_TRUE(!image.isNull() && image.depth() == 16);
#else
    EXPECT_TRUE(qtCompatQImageFormatGrayscale16() == QImage::Format_Invalid);
#endif
}

TEST(UiHelpers, FractionalEventPositions) {
    // Rounding before canvas scaling makes Qt5 and Qt6 choose different pixels.
    QMouseEvent mouse(QEvent::MouseMove, QPointF(10.25, 20.75),
                      QPointF(100.25, 200.75), Qt::NoButton, Qt::NoButton,
                      Qt::NoModifier);
    EXPECT_TRUE(qtCompatMouseEventPos(&mouse) == QPointF(10.25, 20.75));
    EXPECT_TRUE(qtCompatMouseEventGlobalPos(&mouse) == QPointF(100.25, 200.75));
    EXPECT_TRUE(qtCompatMouseEventPosInt(&mouse) == QPoint(10, 21));
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    QWheelEvent wheel(QPointF(10.25, 20.75), QPointF(100.25, 200.75), QPoint(),
                      QPoint(0, 120), Qt::NoButton, Qt::NoModifier,
                      Qt::NoScrollPhase, false);
#else
    QWheelEvent wheel(QPointF(10.25, 20.75), QPointF(100.25, 200.75), QPoint(),
                      QPoint(0, 120), 120, Qt::Vertical, Qt::NoButton,
                      Qt::NoModifier, Qt::NoScrollPhase,
                      Qt::MouseEventNotSynthesized, false);
#endif
    EXPECT_TRUE(qtCompatWheelEventPos(&wheel) == QPointF(10.25, 20.75));
    EXPECT_TRUE(qtCompatWheelEventDelta(&wheel) == 120);
}

int main(int argc, char** argv) {
    // Never require a display: all assertions run offscreen on every CI
    // runner (Linux/macOS/Windows).  An externally set platform (e.g. a
    // local xcb session or xvfb-run) is respected so the test can also be
    // exercised under a real display.
    if (!qEnvironmentVariableIsSet("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QApplication app(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
