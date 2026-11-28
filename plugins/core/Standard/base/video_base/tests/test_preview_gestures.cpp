// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QApplication>
#include <QMouseEvent>

#include "VideoPlaybackWidget.h"

namespace {

class ProbeWidget : public VideoPlaybackWidget {
public:
    using VideoPlaybackWidget::VideoPlaybackWidget;

    QWidget* label() { return previewLabel(); }

    int presses = 0;
    int releases = 0;
    bool lastPressCtrl = false;

protected:
    bool onPreviewMousePress(QMouseEvent* event) override {
        ++presses;
        lastPressCtrl = event->modifiers().testFlag(Qt::ControlModifier);
        return true;  // consume, like the real gesture hooks
    }
    bool onPreviewMouseRelease(QMouseEvent* event) override {
        ++releases;
        return true;
    }
};

}  // namespace

TEST(VideoBasePreviewGestures, ForwardsMousePressReleaseAndModifiers) {
    // Never require a display: the gesture chain is pure event delivery, so
    // all assertions run offscreen on every CI runner.  An externally set
    // platform (e.g. a local xcb session or xvfb-run) is respected so the
    // test can also be exercised under a real display.
    if (!qEnvironmentVariableIsSet("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    ProbeWidget w;
    w.resize(640, 480);
    w.show();
    QCoreApplication::processEvents();

    auto* label = w.label();
    if (!label) {
        FAIL() << "no preview label";
    }

    const QPointF pos(100, 100);
    QMouseEvent press(QEvent::MouseButtonPress, pos, Qt::LeftButton,
                      Qt::LeftButton, Qt::ControlModifier);
    QCoreApplication::sendEvent(label, &press);
    QCoreApplication::processEvents();

    QMouseEvent release(QEvent::MouseButtonRelease, pos, Qt::LeftButton,
                        Qt::LeftButton, Qt::ControlModifier);
    QCoreApplication::sendEvent(label, &release);
    QCoreApplication::processEvents();

    printf("presses=%d releases=%d lastPressCtrl=%d\n", w.presses, w.releases,
           int(w.lastPressCtrl));

    EXPECT_EQ(w.presses, 1);
    EXPECT_EQ(w.releases, 1);
    EXPECT_TRUE(w.lastPressCtrl);
}

int main(int argc, char** argv) {
    if (!qEnvironmentVariableIsSet("QT_QPA_PLATFORM"))
        qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
