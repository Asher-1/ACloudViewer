// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    ProbeWidget w;
    w.resize(640, 480);
    w.show();
    QCoreApplication::processEvents();

    auto* label = w.label();
    if (!label) {
        printf("FAIL: no preview label\n");
        return 1;
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

    if (w.presses != 1 || w.releases != 1) {
        printf("FAIL: synthetic mouse events never reached the "
               "onPreviewMouse* hooks (presses=%d releases=%d)\n",
               w.presses, w.releases);
        return 2;
    }
    if (!w.lastPressCtrl) {
        printf("FAIL: Ctrl modifier lost in forwarding\n");
        return 3;
    }
    printf("PASS: gesture chain intact\n");
    return 0;
}
