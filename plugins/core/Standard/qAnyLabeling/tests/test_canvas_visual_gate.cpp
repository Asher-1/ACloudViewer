// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Gap #6 canvas visual gate: deterministic rendering-difference assertions on
// the shared annotation canvas, driven entirely through the loaded plugin
// (QPluginLoader + QObject layer, the test_plugin pattern). Instead of pixel
// baselines (fragile across font stacks), each gate proves that one
// visualization switch actually changes the rendered canvas — display modes,
// the overlay grid, and the fill-only opacity semantics. Skips honestly when
// the shared demo archive is absent.

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QEventLoop>
#include <QFileInfo>
#include <QImage>
#include <QMainWindow>
#include <QPluginLoader>
#include <QSlider>
#include <QTimer>
#include <QToolBar>
#include <QtGlobal>

#include "ecvStdPluginInterface.h"
#include "ecvTestDataRepository.h"

namespace {
QString g_pluginPath;
QMainWindow* openDemoWindow() {
    auto& repo = ecvTestDataRepository::instance();
    if (!repo.isDatasetAvailable(ecvTestDataRepository::Dataset::AnyLabeling))
        return nullptr;
    if (g_pluginPath.isEmpty()) return nullptr;
    QPluginLoader loader(g_pluginPath);
    QObject* pluginObject = loader.instance();
    if (!pluginObject) return nullptr;
    auto* plugin = qobject_cast<ccStdPluginInterface*>(pluginObject);
    if (!plugin) return nullptr;
    const QList<QAction*> actions = plugin->getActions();
    if (actions.size() != 1) return nullptr;
    actions.first()->trigger();
    QApplication::processEvents();
    QMainWindow* window = nullptr;
    for (QWidget* widget : QApplication::topLevelWidgets()) {
        if (widget->objectName() == QStringLiteral("qAnyLabelingWorkspace"))
            window = qobject_cast<QMainWindow*>(widget);
    }
    if (!window) return nullptr;
    // Warm fill through the Use Test Data image-modes entry (the demo
    // archive is available by the caller's contract).
    QAction* fill = window->findChild<QAction*>(
            QStringLiteral("AnnotationTestImageModes"));
    if (!fill) return nullptr;
    fill->trigger();
    QApplication::processEvents();
    return window;
}

quint64 hashImage(const QImage& image) {
    // Deterministic 64-bit FNV-1a over the rendered pixels: sufficient to
    // prove that a visualization switch changed the output.
    quint64 hash = 1469598103934665603ULL;
    for (int y = 0; y < image.height(); ++y) {
        const uchar* line = image.constScanLine(y);
        for (int i = 0; i < image.width() * 4; ++i) {
            hash ^= quint64(line[i]);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

QImage grabCanvas(QMainWindow* window) {
    QWidget* canvas =
            window->findChild<QWidget*>(QStringLiteral("AnnotationCanvas"));
    EXPECT_NE(canvas, nullptr);
    QApplication::processEvents();
    return canvas->grab().toImage();
}

void pump(int ms) {
    QEventLoop loop;
    QTimer::singleShot(ms, &loop, &QEventLoop::quit);
    loop.exec();
}

QAction* displayModeAction(QMainWindow* window, int mode) {
    for (QToolBar* bar : window->findChildren<QToolBar*>())
        for (QAction* action : bar->actions())
            if (action->data().toInt() == mode) return action;
    return nullptr;
}
}  // namespace

TEST(CanvasVisualGate, DisplayModesChangeTheRenderedOutput) {
    QMainWindow* window = openDemoWindow();
    if (!window) GTEST_SKIP() << "plugin or shared anylabeling demo absent";
    const quint64 label = hashImage(grabCanvas(window));
    // 0/1/3 mirror AnnotationCanvas::DisplayMode Label/Instance/Bitmap.
    QAction* instanceMode = displayModeAction(window, 1);
    QAction* maskMode = displayModeAction(window, 3);
    ASSERT_NE(instanceMode, nullptr);
    ASSERT_NE(maskMode, nullptr);
    instanceMode->trigger();
    pump(30);
    const quint64 instance = hashImage(grabCanvas(window));
    maskMode->trigger();
    pump(30);
    const quint64 mask = hashImage(grabCanvas(window));
    EXPECT_NE(label, instance);
    EXPECT_NE(instance, mask);
    EXPECT_NE(label, mask);
}

TEST(CanvasVisualGate, OverlayGridChangesTheRenderedOutput) {
    QMainWindow* window = openDemoWindow();
    if (!window) GTEST_SKIP() << "plugin or shared anylabeling demo absent";
    auto* grid = window->findChild<QCheckBox*>(
            QStringLiteral("AnnotationGridCheck"));
    ASSERT_NE(grid, nullptr);
    const quint64 plain = hashImage(grabCanvas(window));
    grid->setChecked(true);
    pump(30);
    const quint64 gridded = hashImage(grabCanvas(window));
    EXPECT_NE(plain, gridded);
    grid->setChecked(false);
}

TEST(CanvasVisualGate, OpacityAdjustsAnnotationsOnly) {
    // Fill-only opacity (upstream faab50b2): the adjustment must change the
    // annotated frame but leave the annotation-free frame identical — a
    // global-opacity regression would change both.
    QMainWindow* window = openDemoWindow();
    if (!window) GTEST_SKIP() << "plugin or shared anylabeling demo absent";
    auto* opacity =
            window->findChild<QSlider*>(QStringLiteral("AnnotationOpacity"));
    ASSERT_NE(opacity, nullptr);
    pump(30);
    const quint64 annotated = hashImage(grabCanvas(window));
    opacity->setValue(40);
    pump(30);
    const quint64 faded = hashImage(grabCanvas(window));
    EXPECT_NE(annotated, faded);
    opacity->setValue(100);
    pump(30);
    EXPECT_EQ(hashImage(grabCanvas(window)), annotated);
}

int main(int argc, char** argv) {
    QApplication application(argc, argv);
    if (argc > 1) g_pluginPath = QString::fromLocal8Bit(argv[1]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
