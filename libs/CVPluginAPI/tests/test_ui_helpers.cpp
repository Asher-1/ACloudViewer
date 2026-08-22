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

#include <QApplication>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScreen>
#include <QSizePolicy>
#include <QVBoxLayout>
#include <QWidget>

#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "ecvAICoreUiHelper.h"

int g_test_failures = 0;

#define UI_CHECK(cond)                                                       \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,       \
                         #cond);                                             \
            ++g_test_failures;                                               \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
//  DPI scaling invariants
// ---------------------------------------------------------------------------

static void test_dpi_scaling_invariants() {
    // The qMax(px, round(px*dpi/96)) floor must never shrink a nominal size:
    // on a 75% scaled display a naive multiply would make 24 -> 18 and break
    // every minimum-height contract in the plugin dialogs.
    for (int px : {1, 4, 8, 24, 72, 96, 140}) {
        UI_CHECK(ecvAICoreUi::dpiScaled(px) >= px);
    }
    // Monotonic: a larger nominal size never yields a smaller result.
    for (int a = 1; a < 200; a += 7) {
        UI_CHECK(ecvAICoreUi::dpiScaled(a) <= ecvAICoreUi::dpiScaled(a + 1));
    }
    // At 96 logical DPI (the offscreen-platform default) scaling is identity.
    const QScreen* screen = QGuiApplication::primaryScreen();
    if (screen &&
        std::abs(screen->logicalDotsPerInch() - 96.0) < 0.5) {
        UI_CHECK(ecvAICoreUi::dpiScaled(4) == 4);
        UI_CHECK(ecvAICoreUi::dpiScaled(24) == 24);
        UI_CHECK(ecvAICoreUi::dpiScaled(72) == 72);
    }
    // Derived helpers stay on the same floor.
    UI_CHECK(ecvAICoreUi::controlHeight() == ecvAICoreUi::dpiScaled(24));
    UI_CHECK(ecvAICoreUi::compactSpinWidth() == ecvAICoreUi::dpiScaled(72));
    UI_CHECK(ecvAICoreUi::previewSize() == ecvAICoreUi::dpiScaled(96));
}

// ---------------------------------------------------------------------------
//  Compact layout invariants
// ---------------------------------------------------------------------------

static void test_compact_layout_invariants() {
    // Tab pages use tight 4px margins and 4px vertical spacing.  A regression
    // to the old 8px spacing (the "not compact" complaint) fails here.
    QWidget host;
    auto* tabLayout = new QVBoxLayout(&host);
    ecvAICoreUi::setupTabLayout(tabLayout);
    const QMargins m = tabLayout->contentsMargins();
    UI_CHECK(m.left() == 4 && m.top() == 4 && m.right() == 4 && m.bottom() == 4);
    UI_CHECK(tabLayout->spacing() == 4);

    // The two-column form grid: 2px row spacing, 6px column spacing and the
    // 0/1/0/1 stretch pattern that keeps labels fixed and fields expanding.
    QGridLayout grid;
    ecvAICoreUi::setupFormGrid(&grid);
    UI_CHECK(grid.horizontalSpacing() == 6);
    UI_CHECK(grid.verticalSpacing() == 2);
    UI_CHECK(grid.columnStretch(0) == 0);
    UI_CHECK(grid.columnStretch(1) == 1);
    UI_CHECK(grid.columnStretch(2) == 0);
    UI_CHECK(grid.columnStretch(3) == 1);

    // labelColWidth must be DPI-scaled into both label columns.
    QGridLayout grid2;
    ecvAICoreUi::setupFormGrid(&grid2, 90);
    UI_CHECK(grid2.columnMinimumWidth(0) == ecvAICoreUi::dpiScaled(90));
    UI_CHECK(grid2.columnMinimumWidth(2) == ecvAICoreUi::dpiScaled(90));

    // Group boxes must never claim extra vertical space (Maximum policy) and
    // keep the tightened 6/4/6/4 margins — the "blank strip" regression.
    QGroupBox box;
    auto* boxGrid = new QGridLayout(&box);
    ecvAICoreUi::tightenGroupBox(&box);
    UI_CHECK(box.sizePolicy().verticalPolicy() == QSizePolicy::Maximum);
    const QMargins bg = boxGrid->contentsMargins();
    UI_CHECK(bg.left() == 6 && bg.top() == 4 && bg.right() == 6 &&
             bg.bottom() == 4);
    UI_CHECK(boxGrid->verticalSpacing() == 2);
}

// ---------------------------------------------------------------------------
//  Widget factory invariants
// ---------------------------------------------------------------------------

static void test_widget_factory_invariants() {
    // Browse buttons carry the DPI-scaled fixed width (setFixedWidth stores
    // the value in both minimum and maximum width).
    auto* browse = ecvAICoreUi::makeBrowseBtn(QStringLiteral("Browse…"));
    UI_CHECK(browse->minimumWidth() == ecvAICoreUi::dpiScaled(72));
    UI_CHECK(browse->maximumWidth() == ecvAICoreUi::dpiScaled(72));
    UI_CHECK(browse->sizePolicy().horizontalPolicy() == QSizePolicy::Fixed);
    delete browse;

    // Fixed-width form labels scale the same way.
    auto* label = ecvAICoreUi::makeLabel(QStringLiteral("Label"), 100);
    UI_CHECK(label->minimumWidth() == ecvAICoreUi::dpiScaled(100));
    UI_CHECK(label->maximumWidth() == ecvAICoreUi::dpiScaled(100));
    delete label;

    // Sample-data button is teal-accented (styles the shared brand color).
    auto* sample = ecvAICoreUi::makeSampleDataBtn();
    UI_CHECK(sample->styleSheet().contains(QStringLiteral("#00897b")));
    delete sample;
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

    test_dpi_scaling_invariants();
    test_compact_layout_invariants();
    test_widget_factory_invariants();

    if (g_test_failures == 0) {
        fprintf(stderr, "test_cvpluginapi_ui_helpers ok\n");
        return 0;
    }
    fprintf(stderr, "test_cvpluginapi_ui_helpers: %d failure(s)\n",
                 g_test_failures);
    return 1;
}
