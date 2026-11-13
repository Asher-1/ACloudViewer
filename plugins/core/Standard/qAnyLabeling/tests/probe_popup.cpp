// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Diagnostic probe: load the qAnyLabeling plugin, trigger its action, then
// report whether the AnnotationReviewPopup toast becomes visible when the
// annotation window opens, and with which geometry. Runs fully offscreen.

#include <ecvStdPluginInterface.h>

#include <QApplication>
#include <QEvent>
#include <QEventLoop>
#include <QMainWindow>
#include <QObject>
#include <QPluginLoader>
#include <QTimer>
#include <QWidget>
#include <cstdio>
#include <cstdlib>
// backtrace() is glibc/BSD-specific; the Windows CI has no <execinfo.h>.
#ifndef _WIN32
#include <execinfo.h>
#define PROBE_HAS_BACKTRACE 1
#endif

namespace {

class ShowSpy : public QObject {
public:
    bool eventFilter(QObject* obj, QEvent* ev) override {
        if (obj->objectName() == QStringLiteral("AnnotationReviewPopup") &&
            (ev->type() == QEvent::Show ||
             ev->type() == QEvent::ShowToParent)) {
#ifdef PROBE_HAS_BACKTRACE
            void* frames[32];
            const int count = backtrace(frames, 32);
            char** symbols = backtrace_symbols(frames, count);
            for (int i = 0; i < count && i < 24; ++i)
                std::printf("FRAME %02d: %s\n", i, symbols[i]);
            std::free(symbols);
#else
            std::printf(
                    "Show event on AnnotationReviewPopup "
                    "(no backtrace on this platform)\n");
#endif
        }
        return QObject::eventFilter(obj, ev);
    }
};

}  // namespace

int main(int argc, char** argv) {
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    if (argc < 2 || argc > 3) return 2;

    // QLibrary instead of QPluginLoader: the plugin binary itself loads fine
    // (verified with dlopen); skip Qt's metadata validation layer.
    QLibrary library(QString::fromLocal8Bit(argv[1]));
    if (!library.load()) {
        std::fprintf(stderr, "library load failed: %s\n",
                     library.errorString().toLocal8Bit().constData());
        return 1;
    }
    auto instanceFn = reinterpret_cast<QObject* (*)(void)>(
            library.resolve("qt_plugin_instance"));
    if (!instanceFn) {
        std::fprintf(stderr, "qt_plugin_instance missing\n");
        return 1;
    }
    QObject* pluginObject = instanceFn();
    auto* plugin = qobject_cast<ccStdPluginInterface*>(pluginObject);
    if (!plugin) {
        std::fprintf(stderr, "not a ccStdPluginInterface\n");
        return 1;
    }
    const QList<QAction*> actions = plugin->getActions();
    std::printf("plugin loaded, actions=%d\n", actions.size());

    ShowSpy spy;
    app.installEventFilter(&spy);

    actions.first()->trigger();
    QApplication::processEvents();
    for (int i = 0; i < 5; ++i) QApplication::processEvents();

    // Walk every top-level widget looking for the annotation window.
    QWidget* annotationWindow = nullptr;
    const QWidgetList tops = QApplication::topLevelWidgets();
    for (QWidget* top : tops) {
        std::printf("top-level: %s visible=%d size=%dx%d\n",
                    top->objectName().isEmpty()
                            ? top->metaObject()->className()
                            : top->objectName().toLocal8Bit().constData(),
                    int(top->isVisible()), top->width(), top->height());
        if (top->objectName() == QStringLiteral("qAnyLabelingWorkspace"))
            annotationWindow = top;
    }
    if (!annotationWindow) {
        std::fprintf(stderr, "annotation window not found\n");
        return 1;
    }
    QWidget* popup = annotationWindow->findChild<QWidget*>(
            QStringLiteral("AnnotationReviewPopup"));
    if (!popup) {
        std::fprintf(stderr, "review popup not found\n");
        return 1;
    }
    std::printf(
            "popup: visible=%d pos=(%d,%d) size=%dx%d isWindow=%d "
            "windowType=0x%llx\n",
            int(popup->isVisible()), popup->x(), popup->y(), popup->width(),
            popup->height(), int(popup->isWindow()),
            unsigned(long(popup->windowFlags() & Qt::WindowType_Mask)));
    if (popup->isVisible()) {
        std::fprintf(stderr, "REGRESSION: popup visible on window open\n");
        return 1;
    }

    // End-to-end toast behavior: load an image, move the workspace away
    // from the screen origin (the real-world case), fire the review loop,
    // and verify the upstream Popup "default" placement (centered, 100
    // client pixels below the client top) plus the timed auto-close.
    if (argc == 3) {
        annotationWindow->move(421, 187);
        QApplication::processEvents();
        const QString image = QString::fromLocal8Bit(argv[2]);
        if (!QMetaObject::invokeMethod(annotationWindow, "openImage",
                                       Qt::DirectConnection,
                                       Q_ARG(QString, image))) {
            std::fprintf(stderr, "openImage not invocable\n");
            return 1;
        }
        for (int i = 0; i < 8; ++i) QApplication::processEvents();
        QAction* loopThrough = nullptr;
        for (QAction* action : annotationWindow->findChildren<QAction*>()) {
            if (action->text() == QStringLiteral("Loop Through Labels"))
                loopThrough = action;
        }
        if (!loopThrough || !loopThrough->isEnabled()) {
            std::fprintf(stderr, "loop action missing or still disabled\n");
            return 1;
        }
        loopThrough->trigger();
        QApplication::processEvents();
        const QPoint hostTopLeft = annotationWindow->mapToGlobal(QPoint(0, 0));
        const QPoint toastTopLeft = popup->mapToGlobal(QPoint(0, 0));
        const int centeredX = hostTopLeft.x() +
                              (annotationWindow->width() - popup->width()) / 2;
        const int expectedY = hostTopLeft.y() + 100;
        std::printf("toast: visible=%d global=(%d,%d) expected=(%d,%d)\n",
                    int(popup->isVisible()), toastTopLeft.x(), toastTopLeft.y(),
                    centeredX, expectedY);
        if (!popup->isVisible() ||
            toastTopLeft != QPoint(centeredX, expectedY)) {
            std::fprintf(stderr, "REGRESSION: toast misplaced or hidden\n");
            return 1;
        }
        QEventLoop settle;
        QTimer::singleShot(2200, &settle, &QEventLoop::quit);
        settle.exec();
        if (popup->isVisible()) {
            std::fprintf(stderr, "REGRESSION: toast did not auto-close\n");
            return 1;
        }
        std::printf("PROBE OK: toast placed correctly and auto-closed\n");
        return 0;
    }
    std::printf("PROBE OK: popup hidden on window open\n");
    return 0;
}
