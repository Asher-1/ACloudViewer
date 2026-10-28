// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QAbstractButton>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEventLoop>
#include <QFile>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QHash>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QLineF>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QPainter>
#include <QProgressDialog>
#include <QPushButton>
#include <QScreen>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QTemporaryDir>
#include <QTimer>
#include <QToolButton>
#include <QVariantAnimation>
#include <QWheelEvent>
#include <QWindow>
#include <QtEndian>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "PointCloudCameraPanel.h"
#include "PointCloudCuboidView.h"
#include "PointCloudDatasetExporter.h"
#include "PointCloudWindow.h"
#include "QtCompat.h"
#define REQUIRE(x)                                               \
    do {                                                         \
        if (!(x)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
            return 1;                                            \
        }                                                        \
    } while (false)

static QWheelEvent wheelEvent(const QPointF& position,
                              const QPointF& globalPosition,
                              int delta) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    return QWheelEvent(position, globalPosition, QPoint(), QPoint(0, delta),
                       Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
#else
    return QWheelEvent(position, globalPosition, QPoint(), QPoint(0, delta),
                       delta, Qt::Vertical, Qt::NoButton, Qt::NoModifier,
                       Qt::NoScrollPhase, Qt::MouseEventNotSynthesized, false);
#endif
}

// Fixed-metric cases also run in ctest without Python or the upstream checkout.
static int testOrthographicEditEvents() {
    for (const auto axes : {std::pair<int, int>{0, 1}, {0, 2}, {1, 2}}) {
        PointCloudCuboid seed;
        seed.size = {{4, 4, 4}};
        PointCloudFrame frame;
        frame.path = "gesture.bin";
        frame.points.append(QVector4D(0, 0, 0, 1));
        frame.cuboids.append(seed);
        PointCloudDocument document;
        document.setFrame(frame);
        PointCloudCuboidView view(axes.first, axes.second, "gesture");
        view.resize(600, 400);
        view.setDocument(&document, seed.id);
        int commits = 0;
        bool valid = true;
        PointCloudCuboid edited;
        view.onEdited = [&](const PointCloudCuboid& box) {
            ++commits;
            edited = box;
            valid &= document.setCuboid(box);
            view.setDocument(&document, seed.id);
        };
        auto mouse = [&](QEvent::Type type, QPointF position,
                         Qt::MouseButton button, Qt::MouseButtons buttons,
                         Qt::KeyboardModifiers modifiers = Qt::NoModifier) {
            QMouseEvent event(type, position, button, buttons, modifiers);
            QApplication::sendEvent(&view, &event);
        };
        // 4 metres span 300 pixels after the source 50-pixel focus padding.
        // The final release is the only event carrying this displacement.
        mouse(QEvent::MouseButtonPress, {300, 200}, Qt::LeftButton,
              Qt::LeftButton);
        mouse(QEvent::MouseButtonRelease, {310, 205}, Qt::RightButton,
              Qt::LeftButton);
        REQUIRE(commits == 0);
        mouse(QEvent::MouseButtonRelease, {330, 215}, Qt::LeftButton,
              Qt::NoButton);
        REQUIRE(valid && commits == 1);
        REQUIRE(std::abs(edited.center[axes.first] - .4) < 1e-10);
        REQUIRE(std::abs(edited.center[axes.second] + .2) < 1e-10);

        document.setFrame(frame);
        view.setDocument(&document, seed.id);
        view.focusCuboid();
        mouse(QEvent::MouseButtonPress, {300, 200}, Qt::RightButton,
              Qt::RightButton);
        mouse(QEvent::MouseMove, {315, 200}, Qt::NoButton, Qt::RightButton);
        QEvent leave(QEvent::Leave);
        QApplication::sendEvent(&view, &leave);
        REQUIRE(valid && commits == 2);
        REQUIRE(std::abs(edited.center[axes.first] - .2) < 1e-10);
        mouse(QEvent::MouseButtonRelease, {360, 200}, Qt::RightButton,
              Qt::NoButton);
        REQUIRE(commits == 2);

        document.setFrame(frame);
        view.setDocument(&document, seed.id);
        view.focusCuboid();
        mouse(QEvent::MouseMove, {300, 22}, Qt::NoButton, Qt::NoButton);
        REQUIRE(view.cursor().shape() == Qt::OpenHandCursor);
        mouse(QEvent::MouseButtonPress, {300, 22}, Qt::LeftButton,
              Qt::LeftButton);
        REQUIRE(view.cursor().shape() == Qt::ClosedHandCursor);
        const double angle = 22 * std::acos(-1.0) / 180;
        mouse(QEvent::MouseButtonRelease,
              {300 + 178 * std::sin(angle), 200 - 178 * std::cos(angle)},
              Qt::LeftButton, Qt::NoButton, Qt::ShiftModifier);
        const int axis = 3 - axes.first - axes.second;
        const double direction = axes.first == 0 && axes.second == 2 ? -1 : 1;
        REQUIRE(valid && commits == 3);
        // Top -> upper right is clockwise, hence negative in world XY.
        REQUIRE(std::abs(edited.rotation[axis] +
                         direction * std::acos(-1.0) / 12) < 1e-10);
        REQUIRE(view.cursor().shape() == Qt::ArrowCursor);
    }
    return 0;
}

static int testOrthographicDepthCrop() {
    PointCloudCuboid box;
    box.id = 1;
    box.size = {{4, 1, 4}};
    PointCloudFrame frame;
    frame.path = QStringLiteral("crop.bin");
    frame.cuboids.append(box);
    frame.points = {QVector4D(-0.5f, 0.59f, 0.25f, 1),
                    QVector4D(0.5f, 0.61f, 0.25f, 1)};
    PointCloudDocument document;
    document.setFrame(frame);
    auto visiblePointPixels = [](PointCloudCuboidView& view) {
        QImage image(view.size(), QImage::Format_ARGB32);
        image.fill(Qt::transparent);
        view.render(&image);
        int count = 0;
        const QColor pointColor(QStringLiteral("#809bb0"));
        for (int y = 0; y < image.height(); ++y)
            for (int x = 0; x < image.width(); ++x)
                count += image.pixelColor(x, y) == pointColor;
        return count;
    };

    PointCloudCuboidView side(0, 2, QStringLiteral("Side"));
    side.resize(600, 400);
    side.setDocument(&document, box.id);
    side.show();
    QApplication::processEvents();
    REQUIRE(side.depthCropEnabled());
    REQUIRE(visiblePointPixels(side) == 1);
    side.setDepthCropEnabled(false);
    REQUIRE(visiblePointPixels(side) == 2);
    side.setDepthCropEnabled(true);
    side.setCreating(true);
    REQUIRE(visiblePointPixels(side) == 2);
    side.setCreating(false);
    QVector<int> visibleIndices{1};
    side.setVisiblePointIndices(&visibleIndices);
    side.setDepthCropEnabled(false);
    REQUIRE(visiblePointPixels(side) == 1);
    visibleIndices[0] = 0;
    side.update();
    REQUIRE(visiblePointPixels(side) == 1);

    auto colorPixels = [](PointCloudCuboidView& view, const QColor& color) {
        QImage image(view.size(), QImage::Format_ARGB32);
        image.fill(Qt::transparent);
        view.render(&image);
        int count = 0;
        for (int y = 0; y < image.height(); ++y)
            for (int x = 0; x < image.width(); ++x)
                count += image.pixelColor(x, y) == color;
        return count;
    };
    side.setVisiblePointIndices(nullptr);
    side.setDepthCropEnabled(false);
    const QColor firstColor(QStringLiteral("#d92d20"));
    const QColor secondColor(QStringLiteral("#168f6a"));
    side.setPointColorProvider([firstColor, secondColor](int index) {
        return index == 0 ? firstColor : secondColor;
    });
    REQUIRE(side.pointSize() == 2);
    const int smallPointPixels = colorPixels(side, firstColor);
    side.setPointSize(10);
    const int largePointPixels = colorPixels(side, firstColor);
    REQUIRE(smallPointPixels > 0 && largePointPixels > smallPointPixels);
    REQUIRE(colorPixels(side, secondColor) > 0);
    side.setPointSize(2);

    PointCloudCuboid frontBox = box;
    frontBox.size = {{1, 4, 4}};
    PointCloudFrame frontFrame = frame;
    frontFrame.cuboids = {frontBox};
    frontFrame.points = {QVector4D(0.59f, -0.5f, 0.25f, 1),
                         QVector4D(0.61f, 0.5f, 0.25f, 1)};
    PointCloudDocument frontDocument;
    frontDocument.setFrame(frontFrame);
    PointCloudCuboidView front(1, 2, QStringLiteral("Front"));
    front.resize(600, 400);
    front.setDocument(&frontDocument, frontBox.id);
    front.show();
    QApplication::processEvents();
    REQUIRE(front.depthCropEnabled());
    REQUIRE(visiblePointPixels(front) == 1);

    PointCloudFrame topFrame = frame;
    topFrame.points = {QVector4D(-0.5f, 0.1f, 0.75f, 1),
                       QVector4D(0.5f, 0.2f, -0.75f, 1)};
    PointCloudDocument topDocument;
    topDocument.setFrame(topFrame);
    PointCloudCuboidView top(0, 1, QStringLiteral("Top"));
    top.resize(600, 400);
    top.setDocument(&topDocument, box.id);
    top.show();
    QApplication::processEvents();
    REQUIRE(top.depthCropEnabled());
    REQUIRE(visiblePointPixels(top) == 2);
    return 0;
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    REQUIRE(testOrthographicDepthCrop() == 0);
    if (argc == 3 &&
        QString::fromLocal8Bit(argv[1]) == "--cuboid-gesture-probe") {
        QFile input(QString::fromLocal8Bit(argv[2]));
        REQUIRE(input.open(QIODevice::ReadOnly));
        const auto cases = QJsonDocument::fromJson(input.readAll()).array();
        QJsonArray output;
        for (const auto& value : cases) {
            const auto item = value.toObject();
            PointCloudCuboid box;
            REQUIRE(PointCloudCuboid::fromJson(item["box"].toObject(), &box,
                                               nullptr));
            PointCloudFrame frame;
            frame.path = "gesture.bin";
            frame.cuboids.append(box);
            if (!item["empty"].toBool())
                frame.points.append(QVector4D(0, 0, 0, 1));
            PointCloudDocument document;
            document.setFrame(frame);
            PointCloudCuboidView view(item["h"].toInt(), item["v"].toInt(),
                                      "probe");
            view.resize(600, 400);
            view.setDocument(&document, box.id);
            QJsonArray edits, cursors;
            bool validEdits = true;
            view.onEdited = [&](const PointCloudCuboid& changed) {
                edits.append(changed.toJson());
                validEdits &= document.setCuboid(changed);
                view.setDocument(&document, box.id);
            };
            for (const auto& eventValue : item["events"].toArray()) {
                const auto event = eventValue.toObject();
                const QString kind = event["kind"].toString();
                if (kind == "leave") {
                    QEvent leave(QEvent::Leave);
                    QApplication::sendEvent(&view, &leave);
                } else {
                    const auto point = event["point"].toArray();
                    const auto type = kind == "press" ? QEvent::MouseButtonPress
                                      : kind == "release"
                                              ? QEvent::MouseButtonRelease
                                              : QEvent::MouseMove;
                    QMouseEvent mouse(
                            type,
                            QPointF(point[0].toDouble(), point[1].toDouble()),
                            Qt::MouseButton(event["button"].toInt()),
                            Qt::MouseButtons(event["buttons"].toInt()),
                            event["shift"].toBool() ? Qt::ShiftModifier
                                                    : Qt::NoModifier);
                    QApplication::sendEvent(&view, &mouse);
                }
                cursors.append(int(view.cursor().shape()));
            }
            REQUIRE(validEdits);
            output.append(QJsonObject{{"case", item["case"]},
                                      {"edits", edits},
                                      {"cursors", cursors}});
        }
        const auto data = QJsonDocument(output).toJson(QJsonDocument::Compact);
        std::fwrite(data.constData(), 1, data.size(), stdout);
        return 0;
    }
    if (argc == 3 &&
        QString::fromLocal8Bit(argv[1]) == "--camera-render-probe") {
        QFile input(QString::fromLocal8Bit(argv[2]));
        REQUIRE(input.open(QIODevice::ReadOnly));
        const auto cases = QJsonDocument::fromJson(input.readAll()).array();
        QString probeError;
        for (const auto& value : cases) {
            const auto item = value.toObject();
            PointCloudFrame frame;
            frame.path = "frame.bin";
            for (const auto& point : item["points"].toArray()) {
                const auto p = point.toArray();
                frame.points.append(QVector4D(p[0].toDouble(), p[1].toDouble(),
                                              p[2].toDouble(),
                                              p[3].toDouble()));
            }
            for (const auto& label : item["labels"].toArray())
                frame.labels.append(quint32(label.toDouble()));
            for (const auto& color : item["rgb"].toArray()) {
                const auto c = color.toArray();
                frame.rgb.append(
                        QColor(c[0].toInt(), c[1].toInt(), c[2].toInt()));
            }
            for (const auto& box : item["cuboids"].toArray()) {
                PointCloudCuboid cuboid;
                REQUIRE(PointCloudCuboid::fromJson(box.toObject(), &cuboid,
                                                   nullptr));
                frame.cuboids.append(cuboid);
            }
            PointCloudCameraSource source;
            source.imagePaths.insert("frame", item["image"].toString());
            REQUIRE(loadPointCloudCalibration(item["calibration"].toString(),
                                              &source.calibration,
                                              &probeError));
            source.hasCalibration = true;
            PointCloudCameraPanel panel;
            panel.setSource(source);
            panel.setFrame(&frame);
            QHash<quint16, QColor> pointColors;
            QHash<int, QColor> cuboidColors;
            const auto palette = item["palette"].toObject();
            for (auto it = palette.begin(); it != palette.end(); ++it) {
                pointColors.insert(it.key().toInt(),
                                   QColor(it.value().toString()));
                cuboidColors.insert(it.key().toInt(),
                                    QColor(it.value().toString()));
            }
            panel.setPointColorStyle(PointCloudColorMode(item["mode"].toInt()),
                                     pointColors);
            panel.setCuboidStyle(cuboidColors, item["selected"].toInt(-1),
                                 true);
            if (item.contains("preview")) {
                PointCloudCuboid preview;
                REQUIRE(PointCloudCuboid::fromJson(item["preview"].toObject(),
                                                   &preview, nullptr));
                panel.setCreationPreview(&preview);
            }
            if (item.contains("edit_preview")) {
                PointCloudCuboid preview;
                REQUIRE(PointCloudCuboid::fromJson(
                        item["edit_preview"].toObject(), &preview, nullptr));
                panel.setEditPreview(&preview);
                if (item["clear_edit"].toBool()) panel.setEditPreview(nullptr);
            }
            QSet<quint16> hidden;
            for (const auto& instance : item["hidden_instances"].toArray())
                hidden.insert(instance.toInt());
            panel.setVisibility({}, hidden, {});
            panel.findChild<QButtonGroup*>("PointCloudCameraColor")
                    ->button(item["depth"].toBool() ? 0 : 1)
                    ->setChecked(true);
            panel.findChild<QSlider*>("PointCloudCameraPointSize")
                    ->setValue(item["size"].toInt(1));
            panel.findChild<QSlider*>("PointCloudCameraOpacity")
                    ->setValue(item["opacity"].toInt(65));
            auto* view =
                    panel.findChild<QGraphicsView*>("PointCloudCameraView");
            REQUIRE(view && panel.projectionEnabled());
            const QSize size =
                    source.calibration.imageSize * item["scale"].toDouble(1);
            QImage rendered(size, QImage::Format_ARGB32);
            rendered.fill(Qt::black);
            QPainter painter(&rendered);
            view->scene()->render(
                    &painter, QRectF(QPointF(), size),
                    QRectF(QPointF(), source.calibration.imageSize));
            painter.end();
            REQUIRE(rendered.save(item["output"].toString()));
        }
        return 0;
    }
    if (argc == 3 && QString::fromLocal8Bit(argv[1]) == "--render-probe") {
        QFile input(QString::fromLocal8Bit(argv[2]));
        REQUIRE(input.open(QIODevice::ReadOnly));
        const auto cases = QJsonDocument::fromJson(input.readAll()).array();
        for (const auto& value : cases) {
            const auto item = value.toObject();
            PointCloudWindow window;
            window.openFile(item["cloud"].toString());
            REQUIRE(window.loadClassDefinitions(item["classes"].toString()));
            window.findChild<QComboBox*>("PointCloudTask")->setCurrentIndex(1);
            if (item.contains("color_mode"))
                window.findChild<QComboBox*>("PointCloudColorMode")
                        ->setCurrentIndex(item["color_mode"].toInt());
            auto* objects = window.findChild<QListWidget*>("PointCloudCuboids");
            objects->setCurrentRow(item["selected"].toInt(-1));
            auto* view = window.findChild<QOpenGLWidget*>();
            REQUIRE(view);
            // Keep the real document/control wiring, isolate only the GL
            // surface so platform-specific text metrics do not mask geometry.
            view->setProperty("captureGeometryOnly", true);
            view->setParent(nullptr);
            view->resize(640, 480);
            view->show();
            for (auto* action : window.findChildren<QAction*>())
                if (action->text() == "Fit") {
                    action->trigger();
                    break;
                }
            if (item.contains("view"))
                window.findChild<QAction*>("PointCloudView" +
                                           item["view"].toString())
                        ->trigger();
            window.findChild<QSpinBox*>("PointCloudPointSize")->setValue(8);
            QEventLoop loop;
            QTimer::singleShot(150, &loop, &QEventLoop::quit);
            loop.exec();
            const QImage image = view->grabFramebuffer();
            REQUIRE(!image.isNull() && image.save(item["output"].toString()));
            delete view;
        }
        return 0;
    }
    QTemporaryDir temporary;
    if (!temporary.isValid()) return 1;
    QCoreApplication::setOrganizationName("qAnyLabelingTests");
    QCoreApplication::setApplicationName("PointCloudCamera");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       temporary.path());
    REQUIRE(testOrthographicEditEvents() == 0);
    QByteArray data;
    for (int i = 0; i < 100; ++i) {
        for (float value : {float(i % 10) - 5, float(i / 10) - 5, 0.0f, 0.5f}) {
            quint32 bits;
            std::memcpy(&bits, &value, 4);
            char raw[4];
            qToLittleEndian<quint32>(bits, reinterpret_cast<uchar*>(raw));
            data.append(raw, 4);
        }
    }
    const QString path = temporary.path() + QStringLiteral("/sample.bin");
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(data) != data.size())
        return 1;
    file.close();
    PointCloudCuboid seed;
    seed.size = {{4, 4, 1}};
    QString error;
    const QString boxesPath = temporary.filePath("sample.cuboids.json");
    REQUIRE(savePointCloudCuboids(boxesPath, path, {seed}, &error));
    PointCloudWindow window;
    window.show();
    window.openFile(path);
    QJsonArray classEntries;
    classEntries.append(
            QJsonObject{{QStringLiteral("id"), 0},
                        {QStringLiteral("name"), QStringLiteral("Unlabeled")},
                        {QStringLiteral("color"), QStringLiteral("#D92D20")}});
    const QString classesPath = temporary.filePath("classes.json");
    QFile classesFile(classesPath);
    REQUIRE(classesFile.open(QIODevice::WriteOnly));
    REQUIRE(classesFile.write(
                    QJsonDocument(QJsonObject{{QStringLiteral("version"), 1},
                                              {QStringLiteral("classes"),
                                               classEntries}})
                            .toJson()) > 0);
    classesFile.close();
    REQUIRE(window.loadClassDefinitions(classesPath));
    QEventLoop initialRender;
    QTimer::singleShot(250, &initialRender, &QEventLoop::quit);
    initialRender.exec();
    auto* view = window.findChild<QOpenGLWidget*>();
    if (!view) return 1;
    const QImage image = view->grabFramebuffer();
    if (image.isNull()) {
        std::fprintf(stderr, "OpenGL framebuffer is empty\n");
        return 1;
    }
    int changed = 0;
    const QRgb background = image.pixel(0, 0);
    for (int y = 0; y < image.height(); ++y)
        for (int x = 0; x < image.width(); ++x)
            changed += image.pixel(x, y) != background;
    if (changed < 30) {
        std::fprintf(stderr, "Point cloud did not render: %d changed pixels\n",
                     changed);
        return 1;
    }
    auto* task = window.findChild<QComboBox*>("PointCloudTask");
    auto* objects = window.findChild<QListWidget*>("PointCloudCuboids");
    REQUIRE(task && objects && objects->count() == 1);
    // The whole-frame loader runs off the GUI thread, as upstream does.
    // Its nested event loop must accept cancellation without replacing the
    // active document, even when the worker finishes in the same iteration.
    const QString cancelledPath = temporary.filePath("cancelled.bin");
    QFile cancelledFile(cancelledPath);
    REQUIRE(cancelledFile.open(QIODevice::WriteOnly));
    REQUIRE(cancelledFile.write(data) == data.size());
    cancelledFile.close();
    bool responsiveDuringLoad = false;
    QTimer::singleShot(0, &window, [&]() {
        auto* progress =
                window.findChild<QProgressDialog*>("PointCloudLoadProgress");
        responsiveDuringLoad = progress && !window.centralWidget()->isEnabled();
        if (progress) progress->cancel();
    });
    window.openFile(cancelledPath);
    REQUIRE(responsiveDuringLoad && window.centralWidget()->isEnabled() &&
            objects->count() == 1);
    REQUIRE(QFile::remove(cancelledPath));
    task->setCurrentIndex(1);
    window.findChild<QComboBox*>("PointCloudColorMode")->setCurrentIndex(0);
    objects->setCurrentRow(0);
    app.processEvents();
    auto* sideView = static_cast<PointCloudCuboidView*>(
            window.findChild<QWidget*>("CuboidView_0_2"));
    REQUIRE(sideView);
    QImage sideColors(sideView->size(), QImage::Format_ARGB32);
    sideColors.fill(Qt::transparent);
    sideView->render(&sideColors);
    int semanticColorPixels = 0;
    for (int y = 0; y < sideColors.height(); ++y)
        for (int x = 0; x < sideColors.width(); ++x)
            semanticColorPixels +=
                    sideColors.pixelColor(x, y) == QColor("#D92D20");
    if (!semanticColorPixels) {
        std::fprintf(stderr, "Semantic render: %dx%d, dpr %.3f, visible %d\n",
                     sideView->width(), sideView->height(),
                     sideView->devicePixelRatioF(), int(sideView->isVisible()));
        const QString artifacts = qEnvironmentVariable("QAL_UI_ARTIFACT_DIR");
        if (!artifacts.isEmpty())
            sideColors.save(QDir(artifacts).filePath("semantic-side.png"));
        QHash<QRgb, int> colors;
        for (int y = 0; y < sideColors.height(); ++y)
            for (int x = 0; x < sideColors.width(); ++x)
                ++colors[sideColors.pixel(x, y)];
        for (auto it = colors.cbegin(); it != colors.cend(); ++it)
            if (it.value() > 10)
                std::fprintf(stderr, "Color %08x: %d pixels\n", unsigned(it.key()), it.value());
    }
    REQUIRE(semanticColorPixels > 0);
    auto* cropAction = window.findChild<QAction*>("PointCloudCropDepth");
    auto* orthographic =
            window.findChild<QSplitter*>("PointCloudOrthographicViews");
    REQUIRE(cropAction && cropAction->isCheckable() && cropAction->isChecked());
    REQUIRE(objects->contextMenuPolicy() == Qt::ActionsContextMenu &&
            objects->actions().contains(cropAction));
    REQUIRE(orthographic && orthographic->count() == 3);
    auto* pointSize = window.findChild<QSpinBox*>("PointCloudPointSize");
    REQUIRE(pointSize);
    pointSize->setValue(8);
    for (const auto& name :
         {"CuboidView_0_1", "CuboidView_0_2", "CuboidView_1_2"}) {
        auto* ortho = static_cast<PointCloudCuboidView*>(
                window.findChild<QWidget*>(name));
        REQUIRE(ortho && ortho->pointSize() == 8);
    }
    pointSize->setValue(2);
    cropAction->setChecked(false);
    for (const auto& name :
         {"CuboidView_0_1", "CuboidView_0_2", "CuboidView_1_2"}) {
        auto* ortho = static_cast<PointCloudCuboidView*>(
                window.findChild<QWidget*>(name));
        REQUIRE(ortho && !ortho->depthCropEnabled());
    }
    cropAction->setChecked(true);
    REQUIRE(window.findChild<QLabel*>("PointCloudClassHeading")->text() ==
            QStringLiteral("Object class"));
    REQUIRE(window.findChild<QComboBox*>("PointCloudOperation")->isHidden());
    REQUIRE(window.findChild<QSpinBox*>("PointCloudBrushSize")->isHidden());
    view->repaint();
    const QImage detectionImage = view->grabFramebuffer();
    int coloredEdges = 0;
    for (int y = 0; y < detectionImage.height(); ++y)
        for (int x = 0; x < detectionImage.width(); ++x) {
            const QColor c = detectionImage.pixelColor(x, y);
            coloredEdges +=
                    c.green() > 70 && std::abs(c.green() - c.blue()) > 25;
        }
    REQUIRE(coloredEdges > 20);
    auto trigger = [&window](const QString& name) {
        auto* action = window.findChild<QAction*>(name);
        if (!action || !action->isEnabled()) return false;
        action->trigger();
        return true;
    };
    REQUIRE(trigger("CuboidDuplicate") && objects->count() == 2);
    REQUIRE(trigger("CuboidCopy") && trigger("CuboidPaste") &&
            objects->count() == 3);
    REQUIRE(trigger("CuboidLock"));
    REQUIRE(trigger("CuboidDelete") && objects->count() == 3);
    REQUIRE(trigger("CuboidLock"));
    REQUIRE(trigger("CuboidDelete") && objects->count() == 2);
    objects->setCurrentRow(0);
    bool edited = false;
    QTimer::singleShot(0, &window, [&]() {
        auto* dialog = window.findChild<QDialog*>("CuboidPropertiesDialog");
        if (!dialog) return;
        auto* field = dialog->findChild<QDoubleSpinBox*>("Cuboid_rotation_2");
        if (field) {
            field->setValue(30);
            QMetaObject::invokeMethod(field, "editingFinished");
            edited = true;
        }
        dialog->accept();
    });
    REQUIRE(trigger("CuboidProperties") && edited);
    // Actual orthographic body drag commits one document edit.
    auto* top = static_cast<PointCloudCuboidView*>(
            window.findChild<QWidget*>("CuboidView_0_1"));
    REQUIRE(top && top->isVisible());
    const auto originalOrthoSizes = orthographic->sizes();
    REQUIRE(originalOrthoSizes.size() == 3 &&
            originalOrthoSizes[0] + originalOrthoSizes[1] +
                            originalOrthoSizes[2] >
                    0);
    auto* expandButton =
            top->findChild<QToolButton*>("PointCloudCuboidExpandButton");
    REQUIRE(expandButton);
    QEvent enterTop(QEvent::Enter);
    QApplication::sendEvent(top, &enterTop);
    REQUIRE(expandButton->isVisible());
    QEvent leaveTop(QEvent::Leave);
    QApplication::sendEvent(top, &leaveTop);
    REQUIRE(!expandButton->isVisible());
    expandButton->click();
    app.processEvents();
    REQUIRE(top->isExpanded() && !top->isHidden());
    REQUIRE(view->isHidden());
    REQUIRE(window.findChild<QWidget*>("CuboidView_0_2")->isHidden());
    REQUIRE(window.findChild<QWidget*>("CuboidView_1_2")->isHidden());
    expandButton->click();
    app.processEvents();
    REQUIRE(!top->isExpanded());
    REQUIRE(!view->isHidden());
    REQUIRE(!window.findChild<QWidget*>("CuboidView_0_2")->isHidden());
    REQUIRE(!window.findChild<QWidget*>("CuboidView_1_2")->isHidden());
    REQUIRE(orthographic->sizes() == originalOrthoSizes);
    expandButton->click();
    app.processEvents();
    REQUIRE(top->isExpanded());
    task->setCurrentIndex(0);
    app.processEvents();
    REQUIRE(!top->isExpanded());
    REQUIRE(!view->isHidden());
    REQUIRE(!window.findChild<QWidget*>("CuboidView_0_2")->isHidden());
    REQUIRE(!window.findChild<QWidget*>("CuboidView_1_2")->isHidden());
    task->setCurrentIndex(1);
    app.processEvents();
    REQUIRE(trigger("CuboidFocus"));
    auto* perspectiveAnimation = view->findChild<QVariantAnimation*>(
            "CuboidPerspectiveFocusAnimation");
    REQUIRE(perspectiveAnimation && perspectiveAnimation->duration() > 0 &&
            perspectiveAnimation->easingCurve().type() == QEasingCurve::Linear);
    QKeyEvent cameraMove(QEvent::KeyPress, Qt::Key_I, Qt::AltModifier);
    QApplication::sendEvent(view, &cameraMove);
    REQUIRE(perspectiveAnimation->state() == QAbstractAnimation::Stopped);
    auto* focusAnimation = top->findChild<QVariantAnimation*>(
            "CuboidOrthographicFocusAnimation");
    REQUIRE(focusAnimation && focusAnimation->duration() == 250 &&
            focusAnimation->state() == QAbstractAnimation::Running);
    focusAnimation->setCurrentTime(125);
    REQUIRE(std::abs(focusAnimation->currentValue().toDouble() - 0.5) < 1e-6);
    top->focusCuboid();
    REQUIRE(focusAnimation->state() == QAbstractAnimation::Stopped);
    app.processEvents();
    auto mouse = [&](QEvent::Type type, const QPointF& point,
                     Qt::MouseButton button, Qt::MouseButtons buttons) {
        QMouseEvent event(type, point, button, buttons, Qt::NoModifier);
        QApplication::sendEvent(top, &event);
    };
    const QPointF center(top->width() / 2.0, top->height() / 2.0);
    auto* cameraPanel = window.findChild<PointCloudCameraPanel*>();
    REQUIRE(cameraPanel);
    view->setProperty("captureGeometryOnly", true);
    const QImage beforePreview = view->grabFramebuffer();
    mouse(QEvent::MouseButtonPress, center, Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseMove, center + QPointF(25, 0), Qt::NoButton,
          Qt::LeftButton);
    REQUIRE(top->editPreview() && cameraPanel->editPreview());
    const auto preview = *top->editPreview();
    for (const auto& name : {"CuboidView_0_2", "CuboidView_1_2"}) {
        auto* other = static_cast<PointCloudCuboidView*>(
                window.findChild<QWidget*>(name));
        REQUIRE(other && other->editPreview() &&
                *other->editPreview() == preview);
    }
    REQUIRE(*cameraPanel->editPreview() == preview);
    REQUIRE(beforePreview != view->grabFramebuffer());
    QKeyEvent cancelPreview(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
    QApplication::sendEvent(top, &cancelPreview);
    REQUIRE(!top->editPreview() && !cameraPanel->editPreview());
    REQUIRE(beforePreview == view->grabFramebuffer());
    view->setProperty("captureGeometryOnly", false);
    mouse(QEvent::MouseButtonPress, center, Qt::LeftButton, Qt::LeftButton);
    mouse(QEvent::MouseMove, center + QPointF(18, 0), Qt::NoButton,
          Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, center + QPointF(18, 0), Qt::LeftButton,
          Qt::NoButton);
    REQUIRE(!top->editPreview() && !cameraPanel->editPreview());
    for (auto* action : window.findChildren<QAction*>())
        if (action->text() == "Save") {
            action->trigger();
            break;
        }
    QVector<PointCloudCuboid> saved;
    REQUIRE(loadPointCloudCuboids(boxesPath, path, &saved, &error));
    REQUIRE(saved.size() == 2 &&
            std::abs(saved[0].rotation[2] - 3.141592653589793 / 6) < 1e-10);
    REQUIRE(saved[0].center[0] > 0.01 && saved[0].center[1] > 0.01);
    // Main-view arrows move in the current camera basis, create an undoable
    // document edit, and never move a locked object or consume modified arrows.
    const auto beforeNudge = saved;
    auto key = [&](int code, Qt::KeyboardModifiers modifiers = Qt::NoModifier) {
        QKeyEvent event(QEvent::KeyPress, code, modifiers);
        QApplication::sendEvent(view, &event);
    };
    auto saveBoxes = [&] {
        for (auto* action : window.findChildren<QAction*>())
            if (action->text() == "Save") {
                action->trigger();
                break;
            }
        return loadPointCloudCuboids(boxesPath, path, &saved, &error);
    };
    key(Qt::Key_Right);
    REQUIRE(saveBoxes());
    const double movement = .1 / std::sqrt(2.0);
    REQUIRE(std::abs(saved[0].center[0] - beforeNudge[0].center[0] - movement) <
            1e-10);
    REQUIRE(std::abs(saved[0].center[1] - beforeNudge[0].center[1] + movement) <
            1e-10);
    for (auto* action : window.findChildren<QAction*>())
        if (action->text() == "Undo") {
            action->trigger();
            break;
        }
    REQUIRE(saveBoxes() && saved == beforeNudge);
    key(Qt::Key_Right, Qt::ShiftModifier);
    REQUIRE(saveBoxes() && saved == beforeNudge);
    REQUIRE(trigger("CuboidLock"));
    REQUIRE(saveBoxes());
    const auto lockedBoxes = saved;
    key(Qt::Key_Up);
    REQUIRE(saveBoxes() && saved == lockedBoxes);
    REQUIRE(trigger("CuboidLock") && saveBoxes() && saved == beforeNudge);
    // Create from an orthographic rectangle and derive hidden depth from
    // points.
    auto* classes = window.findChild<QComboBox*>("PointCloudClass");
    REQUIRE(classes);
    classes->addItem("Object", 1);
    classes->setCurrentIndex(classes->count() - 1);
    REQUIRE(trigger("CuboidDraw"));
    app.processEvents();
    const QPointF origin(top->width() / 2.0, top->height() / 2.0);
    mouse(QEvent::MouseButtonPress, origin - QPointF(35, 35), Qt::LeftButton,
          Qt::LeftButton);
    mouse(QEvent::MouseMove, origin + QPointF(35, 35), Qt::NoButton,
          Qt::LeftButton);
    mouse(QEvent::MouseButtonRelease, origin + QPointF(35, 35), Qt::LeftButton,
          Qt::NoButton);
    REQUIRE(objects->count() == 3);
    for (auto* action : window.findChildren<QAction*>())
        if (action->text() == "Save") {
            action->trigger();
            break;
        }
    REQUIRE(loadPointCloudCuboids(boxesPath, path, &saved, &error) &&
            saved.size() == 3);
    REQUIRE(saved.last().size[2] == 0.01);

    // Loading runs off the UI thread; cancelled and invalid candidates retain
    // the displayed document and file list. A reentrant open is rejected.
    {
        QTemporaryDir other;
        REQUIRE(other.isValid());
        const QString candidate = other.filePath("candidate.bin");
        QFile large(candidate);
        REQUIRE(large.open(QIODevice::WriteOnly));
        const QByteArray block = data.repeated(64);
        for (int i = 0; i < 256; ++i)
            REQUIRE(large.write(block) == block.size());
        large.close();
        auto* frames = window.findChild<QListWidget*>("PointCloudFrames");
        const int previousCount = frames->count(),
                  previousRow = frames->currentRow();
        const auto beforeBoxes = saved;
        bool responsive = false, reentrant = false;
        QTimer::singleShot(0, &window, [&] {
            auto* progress = window.findChild<QProgressDialog*>(
                    "PointCloudLoadProgress");
            if (!progress) return;
            responsive = true;
            window.openFile(path);
            reentrant = frames->count() == previousCount;
            progress->cancel();
        });
        window.openFile(candidate);
        REQUIRE(responsive && reentrant && frames->count() == previousCount &&
                frames->currentRow() == previousRow &&
                objects->count() == beforeBoxes.size());
        REQUIRE(loadPointCloudCuboids(boxesPath, path, &saved, &error) &&
                saved == beforeBoxes);
        const QString broken = other.filePath("broken.bin");
        QFile invalid(broken);
        REQUIRE(invalid.open(QIODevice::WriteOnly));
        REQUIRE(invalid.write("bad", 3) == 3);
        invalid.close();
        bool warned = false;
        QTimer dismiss;
        dismiss.setInterval(10);
        QObject::connect(&dismiss, &QTimer::timeout, &window, [&] {
            if (auto* box = window.findChild<QMessageBox*>()) {
                warned = true;
                box->accept();
            }
        });
        dismiss.start();
        window.openFile(broken);
        dismiss.stop();
        REQUIRE(warned && frames->count() == previousCount &&
                frames->currentRow() == previousRow &&
                objects->count() == beforeBoxes.size());
    }

    // Separate task class tables survive switch/save/load and dataset import.
    PointCloudFrame labeledFrame;
    REQUIRE(loadPointCloudFrame(path, &labeledFrame, &error));
    labeledFrame.labels[0] = 9u | (12u << 16);
    REQUIRE(savePointCloudLabels(labeledFrame, &error));
    const QString unmatchedPath = temporary.filePath("unmatched.bin");
    QFile unmatched(unmatchedPath);
    REQUIRE(unmatched.open(QIODevice::WriteOnly) &&
            unmatched.write(data) == data.size());
    unmatched.close();
    PointCloudCuboid reserved = seed;
    reserved.classId = 50;
    const QString unmatchedBoxes = temporary.filePath("unmatched.cuboids.json");
    REQUIRE(savePointCloudCuboids(unmatchedBoxes, unmatchedPath, {reserved},
                                  &error));
    window.openFile(path);
    const QJsonArray detectionClasses{
            QJsonObject{{"id", 1}, {"name", "Vehicle"}, {"color", "#123456"}}};
    const QJsonArray segmentationClasses{
            QJsonObject{
                    {"id", 0}, {"name", "Background"}, {"color", "#808080"}},
            QJsonObject{{"id", 9}, {"name", "Road"}, {"color", "#345678"}}};
    const QString classPath = temporary.filePath("classes.json");
    QFile configFile(classPath);
    const QByteArray config =
            QJsonDocument(
                    QJsonObject{
                            {"version", 1},
                            {"detection",
                             QJsonObject{{"classes", detectionClasses}}},
                            {"segmentation",
                             QJsonObject{{"classes", segmentationClasses}}}})
                    .toJson();
    REQUIRE(configFile.open(QIODevice::WriteOnly) &&
            configFile.write(config) == config.size());
    configFile.close();
    REQUIRE(window.loadClassDefinitions(classPath, &error));
    REQUIRE(classes->findData(0) < 0 && classes->findData(9) < 0 &&
            classes->itemText(classes->findData(1)).contains("Vehicle"));
    task->setCurrentIndex(0);
    REQUIRE(classes->findData(0) >= 0 &&
            classes->itemText(classes->findData(9)).contains("Road") &&
            classes->findData(1) < 0);
    task->setCurrentIndex(1);
    REQUIRE(window.saveClassDefinitions(classPath, &error));
    REQUIRE(!window.saveClassDefinitions(unmatchedBoxes, &error));
    QFile groupedFile(classPath);
    REQUIRE(groupedFile.open(QIODevice::ReadOnly));
    auto grouped = QJsonDocument::fromJson(groupedFile.readAll()).object();
    groupedFile.close();
    REQUIRE(grouped["detection"].toObject()["classes"].toArray() ==
                    detectionClasses &&
            grouped["segmentation"].toObject()["classes"].toArray() ==
                    segmentationClasses);

    // Exercise the real export menu/dialog for each format.
    for (const QString& format :
         {QStringLiteral("datumaro"), QStringLiteral("kitti_raw"),
          QStringLiteral("sly_pointcloud")}) {
        const QString archive = temporary.filePath(format + "-ui.zip");
        bool dialogFilled = false;
        QTimer::singleShot(0, &window, [&]() {
            auto* dialog =
                    window.findChild<QDialog*>("PointCloudDatasetDialog");
            if (!dialog) return;
            auto* field =
                    dialog->findChild<QLineEdit*>("PointCloudDatasetPath");
            auto* selector =
                    dialog->findChild<QComboBox*>("PointCloudDatasetFormat");
            if (!field || !selector) return;
            field->setText(archive);
            selector->setCurrentIndex(selector->findData(format));
            dialogFilled = true;
            dialog->accept();
        });
        REQUIRE(trigger("PointCloudDatasetExport") && dialogFilled &&
                QFile::exists(archive));
        PointCloudDatasetImportResult exported;
        REQUIRE(importPointCloudDataset(archive, format, &exported, {},
                                        &error) &&
                exported.frames.size() == 2);
    }
    PointCloudDatasetExportOptions external;
    external.outputPath = temporary.filePath("import-ui.zip");
    external.format = "datumaro";
    PointCloudFrame importedFrame = labeledFrame;
    PointCloudCuboid importedBox = seed;
    importedBox.id = 13;
    importedBox.center = {{2, 3, 4}};
    importedFrame.cuboids = {importedBox};
    external.frames = {importedFrame};
    external.classNames.insert(1, "Truck");
    REQUIRE(exportPointCloudDataset(external, {}, &error));
    PointCloudDatasetImportPlan plan;
    REQUIRE(window.prepareDatasetImport(external.outputPath, "datumaro",
                                        classPath, &plan, {}, &error));
    REQUIRE(plan.targets.size() == 1 &&
            plan.targets[0].cuboids[0].classId == 51);
    REQUIRE(trigger("CuboidDuplicate") && objects->count() == 4);
    REQUIRE(!window.applyDatasetImport(plan, {}, &error) &&
            error.contains("changed"));
    for (auto* action : window.findChildren<QAction*>())
        if (action->text() == "Undo") {
            action->trigger();
            break;
        }
    REQUIRE(objects->count() == 3);
    REQUIRE(window.prepareDatasetImport(external.outputPath, "datumaro",
                                        classPath, &plan, {}, &error));
    REQUIRE(!window.applyDatasetImport(
            plan, [](int current, int) { return current < 1; }, &error));
    REQUIRE(objects->count() == 3);
    QVector<PointCloudCuboid> afterCancel;
    REQUIRE(loadPointCloudCuboids(boxesPath, path, &afterCancel, &error) &&
            afterCancel == saved);
    objects->setCurrentRow(0);
    bool changedDuringImport = false;
    REQUIRE(!window.applyDatasetImport(
            plan,
            [&](int, int) {
                if (!changedDuringImport) {
                    changedDuringImport = true;
                    trigger("CuboidDuplicate");
                }
                return true;
            },
            &error));
    REQUIRE(changedDuringImport && error.contains("changed during") &&
            objects->count() == 4);
    REQUIRE(loadPointCloudCuboids(boxesPath, path, &afterCancel, &error) &&
            afterCancel == saved);
    for (auto* action : window.findChildren<QAction*>())
        if (action->text() == "Undo") {
            action->trigger();
            break;
        }
    REQUIRE(objects->count() == 3);

    // The actual import dialog previews the replacement before applying it.
    bool importFilled = false, importConfirmed = false;
    QTimer confirmation;
    confirmation.setInterval(10);
    QObject::connect(&confirmation, &QTimer::timeout, &window, [&]() {
        auto* box =
                qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
        if (box && box->text().startsWith("Import ")) {
            importConfirmed = true;
            box->button(QMessageBox::Yes)->click();
        }
    });
    confirmation.start();
    QTimer::singleShot(0, &window, [&]() {
        auto* dialog = window.findChild<QDialog*>("PointCloudDatasetDialog");
        if (!dialog) return;
        auto* field = dialog->findChild<QLineEdit*>("PointCloudDatasetPath");
        auto* config =
                dialog->findChild<QLineEdit*>("PointCloudDatasetClasses");
        if (!field || !config) return;
        field->setText(external.outputPath);
        config->setText(classPath);
        importFilled = true;
        dialog->accept();
    });
    REQUIRE(trigger("PointCloudDatasetImport") && importFilled &&
            importConfirmed);
    confirmation.stop();
    REQUIRE(objects->count() == 1 &&
            classes->itemText(classes->findData(51)).contains("Truck"));
    REQUIRE(loadPointCloudFrame(path, &labeledFrame, &error) &&
            labeledFrame.labels[0] == (9u | (12u << 16)) &&
            labeledFrame.cuboids[0].classId == 51);
    QVector<PointCloudCuboid> untouched;
    REQUIRE(loadPointCloudCuboids(unmatchedBoxes, unmatchedPath, &untouched,
                                  &error) &&
            untouched == QVector<PointCloudCuboid>{reserved});
    task->setCurrentIndex(0);
    REQUIRE(classes->findData(51) < 0 &&
            classes->itemText(classes->findData(9)).contains("Road"));
    task->setCurrentIndex(1);
    REQUIRE(groupedFile.open(QIODevice::ReadOnly));
    grouped = QJsonDocument::fromJson(groupedFile.readAll()).object();
    groupedFile.close();
    REQUIRE(grouped["segmentation"].toObject()["classes"].toArray() ==
            segmentationClasses);
    REQUIRE(window.loadClassDefinitions(classPath, &error) &&
            classes->findData(51) >= 0);

    // The camera panel follows the upstream basename contract and overlays
    // projected point/cuboid geometry on the matched image.
    const QString cameraImagePath = temporary.filePath("sample.png");
    QImage cameraImage(64, 64, QImage::Format_RGB32);
    cameraImage.fill(Qt::black);
    REQUIRE(cameraImage.save(cameraImagePath));
    PointCloudCameraSource cameraSource;
    cameraSource.directory = temporary.path();
    cameraSource.imagePaths.insert(QStringLiteral("sample"), cameraImagePath);
    cameraSource.hasCalibration = true;
    cameraSource.calibration.imageSize = QSize(64, 64);
    cameraSource.calibration.cameraMatrix = {{32, 0, 32, 0, 32, 32, 0, 0, 1}};
    cameraSource.calibration.pointCloudToCamera = {
            {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
    PointCloudFrame cameraFrame;
    cameraFrame.path = path;
    cameraFrame.points = {QVector4D(0, 0, 5, 1), QVector4D(1, 0, 5, 1)};
    PointCloudCuboid cameraCuboid;
    cameraCuboid.center = {{0, 0, 5}};
    cameraCuboid.size = {{1, 1, 1}};
    cameraFrame.cuboids = {cameraCuboid};
    REQUIRE(cameraPanel &&
            cameraPanel->parentWidget() == window.findChild<QOpenGLWidget*>());
    cameraPanel->setSource(cameraSource);
    cameraPanel->setFrame(&cameraFrame);
    cameraPanel->show();
    cameraPanel->resizeImageRegion(QSize(320, 240));
    app.processEvents();
    REQUIRE(cameraPanel->hasImage());
    REQUIRE(cameraPanel->imagePath() == cameraImagePath);
    REQUIRE(cameraPanel->projectionEnabled());
    REQUIRE(cameraPanel->statusMessage().isEmpty());
    {
        // Source panel formulas on a square image: minimum size, retained
        // user request, small viewport clamp, and lower-left handle drag.
        QWidget* originalHost = cameraPanel->parentWidget();
        QWidget host;
        host.resize(640, 480);
        cameraPanel->attachToViewport(&host);
        host.show();
        cameraPanel->show();
        cameraPanel->resizeImageRegion(QSize());
        app.processEvents();
        REQUIRE(cameraPanel->geometry() == QRect(364, 8, 268, 240));
        cameraPanel->resizeImageRegion(QSize(400, 300));
        REQUIRE(cameraPanel->geometry() == QRect(232, 8, 400, 300));
        host.resize(200, 120);
        app.processEvents();
        REQUIRE(cameraPanel->geometry() == QRect(8, 8, 184, 104));
        host.resize(640, 480);
        app.processEvents();
        REQUIRE(cameraPanel->geometry() == QRect(232, 8, 400, 300));
        auto* handle = cameraPanel->findChild<QWidget*>(
                "PointCloudCameraResizeHandle");
        REQUIRE(handle && handle->isVisible() &&
                handle->geometry() == QRect(0, 284, 16, 16));
        const QPointF screen = handle->mapToGlobal(QPoint(8, 8));
        QMouseEvent press(QEvent::MouseButtonPress, QPointF(8, 8), screen,
                          Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        const auto pressScreen = qtCompatMouseEventGlobalPos(&press);
        QApplication::sendEvent(handle, &press);
        QMouseEvent move(QEvent::MouseMove, QPointF(-16, 24),
                         screen + QPointF(-24, 16), Qt::NoButton,
                         Qt::LeftButton, Qt::NoModifier);
        const auto moveScreen = qtCompatMouseEventGlobalPos(&move);
        QApplication::sendEvent(handle, &move);
        QMouseEvent release(QEvent::MouseButtonRelease, QPointF(8, 8),
                            screen + QPointF(-24, 16), Qt::LeftButton,
                            Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(handle, &release);
        if (cameraPanel->geometry() != QRect(208, 8, 424, 316)) {
            const auto geometry = cameraPanel->geometry();
            std::fprintf(stderr, "Camera drag: %d,%d %dx%d; screen delta %.3f,%.3f\n",
                         geometry.x(), geometry.y(), geometry.width(), geometry.height(),
                         moveScreen.x() - pressScreen.x(), moveScreen.y() - pressScreen.y());
        }
        REQUIRE(cameraPanel->geometry() == QRect(208, 8, 424, 316));
        REQUIRE(handle->pos() == QPoint(0, 300));
        cameraPanel->attachToViewport(originalHost);
        cameraPanel->show();
        cameraPanel->resizeImageRegion(QSize(320, 240));
        app.processEvents();
    }
    const QImage cameraOverlay = cameraPanel->grab().toImage();
    int overlayPixels = 0;
    for (int y = 0; y < cameraOverlay.height() - 28; ++y)
        for (int x = 0; x < cameraOverlay.width(); ++x) {
            const QColor color = cameraOverlay.pixelColor(x, y);
            overlayPixels +=
                    std::max({color.red(), color.green(), color.blue()}) > 50;
        }
    REQUIRE(overlayPixels > 10);
    cameraPanel->setOverlayEnabled(false);
    app.processEvents();
    const QImage cameraWithoutOverlay = cameraPanel->grab().toImage();
    int residualPixels = 0;
    for (int y = 0; y < cameraWithoutOverlay.height() - 28; ++y)
        for (int x = 0; x < cameraWithoutOverlay.width(); ++x) {
            const QColor color = cameraWithoutOverlay.pixelColor(x, y);
            residualPixels +=
                    std::max({color.red(), color.green(), color.blue()}) > 50;
        }
    REQUIRE(residualPixels < overlayPixels / 2);
    cameraPanel->setOverlayEnabled(true);
    cameraSource.calibration.imageSize = QSize(32, 32);
    cameraPanel->setSource(cameraSource);
    cameraPanel->setFrame(&cameraFrame);
    REQUIRE(cameraPanel->hasImage());
    REQUIRE(!cameraPanel->projectionEnabled());
    REQUIRE(cameraPanel->statusMessage().contains(
            QStringLiteral("does not match")));
    cameraFrame.path = temporary.filePath("missing.bin");
    cameraPanel->setFrame(&cameraFrame);
    REQUIRE(!cameraPanel->hasImage());

    // Each camera has its own images/calibration; navigation wraps and stale
    // projections are discarded when switching to an unmatched source/frame.
    const QString secondDirectory = temporary.filePath("camera_right/data");
    REQUIRE(QDir().mkpath(secondDirectory));
    const QString secondImagePath =
            QDir(secondDirectory).filePath("sample.png");
    cameraImage.fill(QColor(20, 30, 40));
    REQUIRE(cameraImage.save(secondImagePath));
    PointCloudCameraSource secondSource;
    REQUIRE(loadPointCloudCameraSource(secondDirectory, QString(),
                                       &secondSource, &error));
    cameraSource.calibration.imageSize = QSize(64, 64);
    cameraFrame.path = path;
    cameraPanel->setFrame(&cameraFrame);
    cameraPanel->setSources({cameraSource, secondSource});
    REQUIRE(cameraPanel->sources().size() == 2 &&
            cameraPanel->activeCameraIndex() == 0 &&
            cameraPanel->projectionEnabled());
    auto* cameraName = cameraPanel->findChild<QLabel*>("PointCloudCameraName");
    auto* navigationView =
            cameraPanel->findChild<QGraphicsView*>("PointCloudCameraView");
    auto* hoverNext = cameraPanel->findChild<QToolButton*>(
            "PointCloudCameraNavigateNext");
    auto* hoverPrevious = cameraPanel->findChild<QToolButton*>(
            "PointCloudCameraNavigatePrevious");
    auto* cameraNext = hoverNext;
    auto* cameraPrevious = hoverPrevious;
    REQUIRE(cameraName && !cameraName->text().isEmpty());
    REQUIRE(navigationView && hoverNext && hoverPrevious);
    QEvent enterCamera(QEvent::Enter);
    QEvent leaveCamera(QEvent::Leave);
    QApplication::sendEvent(navigationView, &enterCamera);
    REQUIRE(hoverNext->isVisible() && hoverPrevious->isVisible() &&
            hoverNext->size() == QSize(32, 32));
    hoverNext->click();
    REQUIRE(cameraPanel->activeCameraIndex() == 1 &&
            cameraPanel->imagePath() == secondImagePath);
    hoverPrevious->click();
    REQUIRE(cameraPanel->activeCameraIndex() == 0 &&
            cameraPanel->projectionEnabled());
    QApplication::sendEvent(navigationView, &leaveCamera);
    REQUIRE(!hoverNext->isVisible() && !hoverPrevious->isVisible());
    cameraPanel->setSources({cameraSource});
    QApplication::sendEvent(navigationView, &enterCamera);
    REQUIRE(!hoverNext->isVisible() && !hoverPrevious->isVisible());
    cameraPanel->setSources({cameraSource, secondSource});
    REQUIRE(hoverNext->isVisible() && hoverPrevious->isVisible());
    QApplication::sendEvent(navigationView, &leaveCamera);
    cameraNext->click();
    REQUIRE(cameraPanel->activeCameraIndex() == 1 &&
            cameraPanel->imagePath() == secondImagePath &&
            !cameraPanel->projectionEnabled());
    cameraNext->click();
    REQUIRE(cameraPanel->activeCameraIndex() == 0 &&
            cameraPanel->projectionEnabled());
    cameraPrevious->click();
    REQUIRE(cameraPanel->activeCameraIndex() == 1);
    cameraPanel->setActiveCameraIndex(9);
    REQUIRE(cameraPanel->activeCameraIndex() == 1);
    cameraPanel->setSources({secondSource, cameraSource});
    REQUIRE(cameraPanel->activeCameraIndex() == 0 &&
            cameraPanel->imagePath() == secondImagePath);
    cameraPanel->setActiveCameraIndex(1);
    REQUIRE(cameraName->text() == QFileInfo(temporary.path()).fileName());
    REQUIRE(cameraPanel->projectionEnabled());
    cameraFrame.path = temporary.filePath("missing.bin");
    cameraPanel->setFrame(&cameraFrame);
    REQUIRE(!cameraPanel->hasImage() && !cameraPanel->projectionEnabled());
    cameraPrevious->click();
    REQUIRE(!cameraPanel->hasImage());
    cameraFrame.path = path;
    cameraPanel->setFrame(&cameraFrame);
    REQUIRE(cameraPanel->imagePath() == secondImagePath);
    cameraPanel->setActiveCameraIndex(1);
    cameraFrame.rgb = {QColor(240, 10, 20), QColor(20, 230, 40)};
    cameraFrame.labels = {1, 2};
    cameraPanel->setPointColorStyle(PointCloudColorMode::Rgb, {});
    cameraPanel->setFrame(&cameraFrame);
    auto* pointsToggle =
            cameraPanel->findChild<QToolButton*>("PointCloudCameraPoints");
    auto* boxesToggle =
            cameraPanel->findChild<QToolButton*>("PointCloudCameraCuboids");
    auto* overlayColor =
            cameraPanel->findChild<QButtonGroup*>("PointCloudCameraColor");
    auto* overlaySize =
            cameraPanel->findChild<QSlider*>("PointCloudCameraPointSize");
    auto* overlayOpacity =
            cameraPanel->findChild<QSlider*>("PointCloudCameraOpacity");
    REQUIRE(pointsToggle && boxesToggle && overlayColor && overlaySize &&
            overlayOpacity);
    REQUIRE(overlaySize->value() == 1 && overlayOpacity->value() == 65 &&
            overlayColor->checkedId() == 0);
    auto* depthLegend =
            cameraPanel->findChild<QLabel*>("PointCloudCameraDepthLegend");
    auto* sizeValue =
            cameraPanel->findChild<QLabel*>("PointCloudCameraPointSizeValue");
    auto* opacityValue =
            cameraPanel->findChild<QLabel*>("PointCloudCameraOpacityValue");
    auto* sizeReset = cameraPanel->findChild<QToolButton*>(
            "PointCloudCameraPointSizeReset");
    auto* opacityReset = cameraPanel->findChild<QToolButton*>(
            "PointCloudCameraOpacityReset");
    REQUIRE(depthLegend && sizeValue && opacityValue && sizeReset &&
            opacityReset);
    REQUIRE(depthLegend->isEnabled() &&
            depthLegend->text().startsWith("Near ") &&
            depthLegend->text().contains(" → Far ") &&
            !depthLegend->toolTip().isEmpty());
    REQUIRE(sizeValue->text() == "1 px" && opacityValue->text() == "65%");
    boxesToggle->setChecked(false);
    overlayOpacity->setValue(100);
    overlayColor->button(1)->setChecked(true);
    REQUIRE(!depthLegend->isEnabled() && opacityValue->text() == "100%");
    auto* cameraView =
            cameraPanel->findChild<QGraphicsView*>("PointCloudCameraView");
    REQUIRE(cameraView);
    const auto projectedPixel = [&](int imageX, int imageY) {
        QApplication::processEvents();
        const auto image = cameraPanel->grab().toImage();
        const QPoint pixel = cameraView->viewport()->mapTo(
                cameraPanel,
                cameraView->mapFromScene(imageX + .5, imageY + .5));
        return image.pixelColor(pixel);
    };
    REQUIRE(projectedPixel(32, 32) == cameraFrame.rgb[0] &&
            projectedPixel(38, 32) == cameraFrame.rgb[1]);
    REQUIRE(!cameraView->renderHints().testFlag(
            QPainter::SmoothPixmapTransform));
    // The source takes the viewport's current colors, not the file's RGB.
    cameraPanel->setPointColorStyle(PointCloudColorMode::Semantic,
                                    {{1, Qt::red}, {2, Qt::green}});
    REQUIRE(projectedPixel(32, 32) == QColor(Qt::red) &&
            projectedPixel(38, 32) == QColor(Qt::green));
    cameraFrame.points[0].setW(0);
    cameraFrame.points[1].setW(1);
    cameraPanel->setFrame(&cameraFrame);
    cameraPanel->setPointColorStyle(PointCloudColorMode::Intensity, {});
    REQUIRE(projectedPixel(32, 32) == QColor(0, 63, 255) &&
            projectedPixel(38, 32) == QColor(255, 229, 50));
    cameraFrame.labels[0] = (9u << 16) | 1;
    cameraPanel->setFrame(&cameraFrame);
    cameraPanel->setPointColorStyle(PointCloudColorMode::Instance,
                                    {{1, Qt::red}, {2, Qt::green}});
    const quint32 instanceCode = cameraFrame.labels[0];
    const QColor instanceColor((instanceCode * 37ull + 83) % 191 + 64,
                               (instanceCode * 73ull + 41) % 191 + 64,
                               (instanceCode * 109ull + 19) % 191 + 64);
    REQUIRE(projectedPixel(32, 32) == instanceColor &&
            projectedPixel(38, 32) == QColor(0, 102, 0));
    cameraPanel->setVisibility({}, {9}, {});
    REQUIRE(projectedPixel(32, 32) == QColor(Qt::black) &&
            projectedPixel(38, 32) == QColor(0, 102, 0));
    cameraPanel->setVisibility({}, {}, {});
    cameraFrame.labels[0] = 1;
    cameraPanel->setFrame(&cameraFrame);
    cameraPanel->setPointColorStyle(PointCloudColorMode::Rgb, {});
    const double fittedScale = cameraView->transform().m11();
    const QPointF cameraMouse(40, 40);
    QWheelEvent zoom = wheelEvent(
            cameraMouse, cameraView->viewport()->mapToGlobal(cameraMouse.toPoint()), 120);
    QApplication::sendEvent(cameraView->viewport(), &zoom);
    REQUIRE(std::abs(cameraView->transform().m11() / fittedScale - 1.2) < 1e-8);
    REQUIRE(projectedPixel(32, 32) == cameraFrame.rgb[0]);
    QMouseEvent fitImage(QEvent::MouseButtonDblClick, cameraMouse,
                         Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(cameraView->viewport(), &fitImage);
    REQUIRE(std::abs(cameraView->transform().m11() - fittedScale) < 1e-8);
    const auto wheelCamera = [&](int delta) {
        QWheelEvent event = wheelEvent(
                cameraMouse, cameraView->viewport()->mapToGlobal(cameraMouse.toPoint()), delta);
        QApplication::sendEvent(cameraView->viewport(), &event);
    };
    for (int i = 0; i < 60; ++i) wheelCamera(120);
    REQUIRE(cameraView->transform().m11() <= 32.0 &&
            cameraView->transform().m11() > 32.0 / 1.2);
    const QPointF beforePan = cameraView->mapToScene(QPoint(50, 50));
    QMouseEvent panPress(QEvent::MouseButtonPress, QPointF(50, 50),
                         Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QMouseEvent panMove(QEvent::MouseMove, QPointF(90, 80), Qt::NoButton,
                        Qt::LeftButton, Qt::NoModifier);
    QMouseEvent panRelease(QEvent::MouseButtonRelease, QPointF(90, 80),
                           Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(cameraView->viewport(), &panPress);
    QApplication::sendEvent(cameraView->viewport(), &panMove);
    QApplication::sendEvent(cameraView->viewport(), &panRelease);
    REQUIRE(QLineF(beforePan, cameraView->mapToScene(QPoint(50, 50))).length() >
            .1);
    const double manualScale = cameraView->transform().m11();
    const QPointF manualCenter = cameraView->mapToScene(QPoint(50, 50));
    cameraFrame.labels[0] = 7;
    cameraFrame.cuboids[0].center[0] += .25;
    cameraPanel->setFrame(&cameraFrame);
    REQUIRE(cameraView->transform().m11() == manualScale &&
            cameraView->mapToScene(QPoint(50, 50)) == manualCenter);
    cameraFrame.labels[0] = 1;
    cameraFrame.cuboids[0].center[0] -= .25;
    cameraPanel->setFrame(&cameraFrame);
    REQUIRE(cameraView->transform().m11() == manualScale);
    cameraView->resize(cameraView->width() + 8, cameraView->height() + 8);
    REQUIRE(cameraView->transform().m11() == manualScale);
    for (int i = 0; i < 100; ++i) wheelCamera(-120);
    REQUIRE(cameraView->transform().m11() >= .02 &&
            cameraView->transform().m11() < .02 * 1.2);
    auto* cameraFit =
            cameraPanel->findChild<QToolButton*>("PointCloudCameraFit");
    REQUIRE(cameraFit);
    cameraFit->click();
    REQUIRE(cameraView->transform().m11() > .02 * 1.2);
    cameraPanel->setVisibility({1}, {}, {});
    REQUIRE(projectedPixel(32, 32) == QColor(Qt::black) &&
            projectedPixel(38, 32) == cameraFrame.rgb[1]);
    cameraPanel->setVisibility({}, {0}, {});
    REQUIRE(projectedPixel(38, 32) == cameraFrame.rgb[1]);
    cameraPanel->setVisibility({}, {}, {});
    overlayOpacity->setValue(0);
    REQUIRE(projectedPixel(38, 32) == QColor(Qt::black));
    overlayOpacity->setValue(100);
    pointsToggle->setChecked(false);
    REQUIRE(projectedPixel(38, 32) == QColor(Qt::black));
    pointsToggle->setChecked(true);
    overlaySize->setValue(3);
    REQUIRE(projectedPixel(39, 33) == cameraFrame.rgb[1]);
    REQUIRE(sizeValue->text() == "3 px");
    {
        PointCloudCameraPanel restored;
        REQUIRE(restored.findChild<QButtonGroup*>("PointCloudCameraColor")
                        ->checkedId() == 1);
        REQUIRE(restored.findChild<QSlider*>("PointCloudCameraPointSize")
                        ->value() == 3);
        REQUIRE(restored.findChild<QSlider*>("PointCloudCameraOpacity")
                        ->value() == 100);
    }
    overlayColor->button(0)->setChecked(true);
    sizeReset->click();
    REQUIRE(overlaySize->value() == 1 && sizeValue->text() == "1 px" &&
            depthLegend->isEnabled());
    REQUIRE(projectedPixel(32, 32) == QColor(255, 100, 65));
    opacityReset->click();
    REQUIRE(overlayOpacity->value() == 65 && opacityValue->text() == "65%");
    auto* settingsButton =
            cameraPanel->findChild<QToolButton*>("PointCloudCameraSettings");
    auto* settingsMenu =
            cameraPanel->findChild<QMenu*>("PointCloudCameraSettingsMenu");
    REQUIRE(settingsButton && settingsMenu);
    settingsButton->click();
    QApplication::processEvents();
    REQUIRE(settingsMenu->isVisible());
    cameraPanel->hide();
    REQUIRE(!settingsMenu->isVisible());
    cameraPanel->show();
    boxesToggle->setChecked(true);
    cameraPanel->clearSource();
    REQUIRE(cameraPanel->sources().isEmpty() &&
            cameraPanel->activeCameraIndex() == -1 && !cameraPanel->hasImage());
    REQUIRE(depthLegend->text() == "Near → Far" &&
            depthLegend->toolTip().isEmpty());
    QApplication::processEvents();
    auto* cameraMessage =
            cameraPanel->findChild<QLabel*>("PointCloudCameraMessage");
    REQUIRE(cameraMessage && cameraMessage->isVisible() &&
            !cameraMessage->text().isEmpty());
    {
        // The source retains four decoded camera images with LRU eviction.
        PointCloudCameraPanel cachedPanel;
        PointCloudCameraSource cachedSource;
        cachedSource.directory = temporary.path();
        QStringList images;
        for (int index = 0; index < 5; ++index) {
            const QString stem = QStringLiteral("camera-cache-%1").arg(index);
            const QString image = temporary.filePath(stem + ".png");
            REQUIRE(cameraImage.save(image));
            cachedSource.imagePaths.insert(stem, image);
            images.append(image);
        }
        cachedPanel.setSource(cachedSource);
        PointCloudFrame cachedFrame;
        const auto selectCachedFrame = [&](int index) {
            cachedFrame.path =
                    QFileInfo(images[index]).completeBaseName() + ".bin";
            cachedPanel.setFrame(&cachedFrame);
        };
        for (int index = 0; index < 4; ++index) {
            selectCachedFrame(index);
            REQUIRE(cachedPanel.hasImage());
        }
        REQUIRE(QFile::remove(images[0]));
        selectCachedFrame(0);
        REQUIRE(cachedPanel.hasImage());
        selectCachedFrame(4);
        REQUIRE(cachedPanel.hasImage());
        REQUIRE(QFile::remove(images[1]));
        selectCachedFrame(1);
        REQUIRE(!cachedPanel.hasImage());
        selectCachedFrame(0);
        REQUIRE(cachedPanel.hasImage());
        cachedPanel.setSources({cachedSource});
        REQUIRE(!cachedPanel.hasImage());
    }

    // Configure/add/remove sources through the real dialog. An invalid entry
    // keeps the dialog open and leaves the live source list unchanged.
    bool cameraConfigurationAccepted = false;
    QTimer::singleShot(0, &window, [&] {
        auto* dialog = window.findChild<QDialog*>(
                "PointCloudCameraConfigurationDialog");
        if (!dialog) return;
        auto* first =
                dialog->findChild<QLineEdit*>("PointCloudCameraDirectory");
        auto* buttons = dialog->findChild<QDialogButtonBox*>();
        auto* add = dialog->findChild<QPushButton*>("PointCloudCameraAdd");
        auto* preview = dialog->findChild<QLabel*>("PointCloudCameraPreview");
        auto* selector = dialog->findChild<QComboBox*>(
                "PointCloudCameraPreviewSelector");
        auto* next =
                dialog->findChild<QToolButton*>("PointCloudCameraPreviewNext");
        if (!first || !buttons || !add || !preview || !selector || !next) {
            dialog->reject();
            return;
        }
        const auto waitForPreview = [&] {
            QEventLoop loop;
            QTimer::singleShot(200, &loop, &QEventLoop::quit);
            loop.exec();
        };
        first->setText(temporary.filePath("missing-camera"));
        buttons->button(QDialogButtonBox::Ok)->click();
        waitForPreview();
        auto* validation = dialog->findChild<QLabel*>(
                "PointCloudCameraConfigurationError");
        if (!dialog->isVisible() || !validation ||
            validation->text().isEmpty() || !cameraPanel->sources().isEmpty() ||
            preview->property("loaded").toBool()) {
            dialog->reject();
            return;
        }
        first->setText(temporary.path());
        add->click();
        const auto fields =
                dialog->findChildren<QLineEdit*>("PointCloudCameraDirectory");
        if (fields.size() != 2) {
            dialog->reject();
            return;
        }
        fields[1]->setText(secondDirectory);
        add->click();
        // Preview follows source directory order and switches independently
        // of the live workspace, even with an invalid draft calibration.
        auto* calibration =
                dialog->findChild<QLineEdit*>("PointCloudCameraCalibration");
        calibration->setText(temporary.filePath("missing-calibration.json"));
        waitForPreview();
        if (selector->count() != 2 || !preview->property("loaded").toBool() ||
            preview->property("directory").toString() != temporary.path() ||
            !cameraPanel->sources().isEmpty()) {
            dialog->reject();
            return;
        }
        next->click();
        if (preview->property("directory").toString() != secondDirectory) {
            dialog->reject();
            return;
        }
        calibration->clear();
        const auto remove =
                dialog->findChildren<QPushButton*>("PointCloudCameraRemove");
        if (remove.size() != 3) {
            dialog->reject();
            return;
        }
        remove.last()->click();
        buttons->button(QDialogButtonBox::Ok)->click();
        cameraConfigurationAccepted = !dialog->isVisible();
    });
    REQUIRE(trigger("PointCloudCameraConfigure") &&
            cameraConfigurationAccepted);
    REQUIRE(cameraPanel->sources().size() == 2 &&
            cameraPanel->imagePath() == cameraImagePath);
    cameraNext->click();
    REQUIRE(cameraPanel->imagePath() == secondImagePath);
    bool previewCancelled = false;
    QTimer::singleShot(0, &window, [&] {
        auto* dialog = window.findChild<QDialog*>(
                "PointCloudCameraConfigurationDialog");
        if (!dialog) return;
        const auto fields =
                dialog->findChildren<QLineEdit*>("PointCloudCameraDirectory");
        if (fields.size() != 2) {
            dialog->reject();
            return;
        }
        fields[0]->setText(temporary.filePath("missing-draft"));
        fields[1]->setText(temporary.path());
        QEventLoop loop;
        QTimer::singleShot(200, &loop, &QEventLoop::quit);
        loop.exec();
        auto* preview = dialog->findChild<QLabel*>("PointCloudCameraPreview");
        previewCancelled =
                preview && preview->property("loaded").toBool() &&
                preview->property("directory").toString() == temporary.path();
        dialog->reject();
    });
    REQUIRE(trigger("PointCloudCameraConfigure") && previewCancelled);
    REQUIRE(cameraPanel->sources().size() == 2 &&
            cameraPanel->activeCameraIndex() == 1 &&
            cameraPanel->imagePath() == secondImagePath);
    cameraPanel->hide();
    // Actual OpenGL occlusion, equal-depth ties, point footprints and through
    // selection must agree with what the user sees, including after repaint.
    {
        const QString cloud = temporary.filePath("occlusion.bin");
        QFile raw(cloud);
        REQUIRE(raw.open(QIODevice::WriteOnly));
        const QVector<QVector4D> points{
                {0, 0, 2, 0}, {0, 0, -2, 0}, {0, 0, 2, 0}, {-4, -4, 0, 0},
                {4, 4, 0, 0}, {-4, 4, 0, 0}, {4, -4, 0, 0}};
        for (const auto& point : points)
            for (int axis = 0; axis < 4; ++axis) {
                const float value = point[axis];
                quint32 bits;
                std::memcpy(&bits, &value, 4);
                char bytes[4];
                qToLittleEndian<quint32>(bits, reinterpret_cast<uchar*>(bytes));
                REQUIRE(raw.write(bytes, 4) == 4);
            }
        raw.close();
        const QString configPath = temporary.filePath("occlusion-classes.json");
        QFile configFile(configPath);
        REQUIRE(configFile.open(QIODevice::WriteOnly));
        configFile.write(
                R"({"version":1,"classes":[{"id":0,"name":"Background","color":"#808080"},{"id":5,"name":"Selected","color":"#ff0000"}]})");
        configFile.close();
        PointCloudWindow picking;
        picking.show();
        picking.openFile(cloud);
        REQUIRE(picking.loadClassDefinitions(configPath, &error));
        auto* viewport = picking.findChild<QOpenGLWidget*>();
        REQUIRE(viewport);
        auto* perspective =
                picking.findChild<QAction*>("PointCloudPerspective");
        auto* through = picking.findChild<QAction*>("PointCloudThrough");
        REQUIRE(perspective && through);
        perspective->setChecked(false);
        picking.findChild<QAction*>("PointCloudViewTop")->trigger();
        picking.findChild<QAction*>("PointCloudToolBrush")->trigger();
        picking.findChild<QSpinBox*>("PointCloudPointSize")->setValue(10);
        picking.findChild<QSpinBox*>("PointCloudBrushSize")->setValue(2);
        auto* classes = picking.findChild<QComboBox*>("PointCloudClass");
        classes->setCurrentIndex(classes->findData(5));
        app.processEvents();
        viewport->repaint();
        auto stroke = [&](QPointF start, QPointF end) {
            QMouseEvent down(QEvent::MouseButtonPress, start, Qt::LeftButton,
                             Qt::LeftButton, Qt::NoModifier);
            QApplication::sendEvent(viewport, &down);
            if (start != end) {
                QMouseEvent move(QEvent::MouseMove, end, Qt::NoButton,
                                 Qt::LeftButton, Qt::NoModifier);
                QApplication::sendEvent(viewport, &move);
            }
            QMouseEvent up(QEvent::MouseButtonRelease, end, Qt::LeftButton,
                           Qt::NoButton, Qt::NoModifier);
            QApplication::sendEvent(viewport, &up);
            app.processEvents();
        };
        auto action = [&](const QKeySequence& shortcut) {
            for (auto* a : picking.findChildren<QAction*>())
                if (a->shortcut() == shortcut) {
                    a->trigger();
                    return true;
                }
            return false;
        };
        auto labels = [&]() {
            PointCloudFrame frame;
            QString error;
            if (!loadPointCloudFrame(cloud, &frame, &error))
                return QVector<quint32>();
            return frame.labels;
        };
        const QPointF center(viewport->width() / 2., viewport->height() / 2.);
        // The first brush event builds the projection/tile cache. A move in
        // the same physical tile must reuse it while still selecting the
        // same visible footprint.
        const int buildsBefore =
                viewport->property("selectionProjectionBuilds").toInt();
        QMouseEvent cachedDown(QEvent::MouseButtonPress, center, Qt::LeftButton,
                               Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(viewport, &cachedDown);
        const int buildsAfterDown =
                viewport->property("selectionProjectionBuilds").toInt();
        QMouseEvent cachedMove(QEvent::MouseMove, center + QPointF(1, 0),
                               Qt::NoButton, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(viewport, &cachedMove);
        const int buildsAfterMove =
                viewport->property("selectionProjectionBuilds").toInt();
        QMouseEvent cachedUp(QEvent::MouseButtonRelease, center + QPointF(1, 0),
                             Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(viewport, &cachedUp);
        app.processEvents();
        REQUIRE(buildsAfterDown == buildsBefore + 1 &&
                buildsAfterMove == buildsAfterDown);
        REQUIRE(viewport->property("selectionCandidateCount").toInt() <=
                points.size());
        REQUIRE(viewport->property("surfacePickingAvailable").toBool());
        REQUIRE(action(QKeySequence::Save));
        REQUIRE(labels() == QVector<quint32>({5, 0, 5, 0, 0, 0, 0}));
        REQUIRE(action(QKeySequence::Undo));
        through->setChecked(true);
        stroke(center, center);
        REQUIRE(action(QKeySequence::Save));
        REQUIRE(labels() == QVector<quint32>({5, 5, 5, 0, 0, 0, 0}));
        REQUIRE(action(QKeySequence::Undo));
        through->setChecked(false);
        // The brush intersects a point's outer footprint without its center.
        stroke(center + QPointF(6, 0), center + QPointF(6, 0));
        REQUIRE(action(QKeySequence::Save));
        REQUIRE(labels() == QVector<quint32>({5, 0, 5, 0, 0, 0, 0}));
        REQUIRE(action(QKeySequence::Undo));
        // Fast movement must include the entire segment between events.
        stroke(center - QPointF(20, 0), center + QPointF(20, 0));
        REQUIRE(action(QKeySequence::Save));
        REQUIRE(labels() == QVector<quint32>({5, 0, 5, 0, 0, 0, 0}));
        // Main-view creation follows the nearest visible point along the ray;
        // cancelling must leave the document unchanged and exit every view.
        auto* mode = picking.findChild<QComboBox*>("PointCloudTask");
        auto* boxes = picking.findChild<QListWidget*>("PointCloudCuboids");
        REQUIRE(mode && boxes);
        mode->setCurrentIndex(1);
        classes->setCurrentIndex(classes->findData(5));
        if (classes->currentData().toInt() == 0 ||
            classes->currentIndex() < 0) {
            classes->addItem("Object", 1);
            classes->setCurrentIndex(classes->count() - 1);
        }
        // Exercise the real window wiring: rendering mode and creation
        // preview must reach the floating camera without reopening its image.
        const QString pickingImage = temporary.filePath("occlusion.png");
        cameraImage.fill(Qt::black);
        REQUIRE(cameraImage.save(pickingImage));
        PointCloudCameraSource pickingSource = cameraSource;
        pickingSource.imagePaths.clear();
        pickingSource.imagePaths.insert("occlusion", pickingImage);
        auto* pickingCamera = picking.findChild<PointCloudCameraPanel*>();
        REQUIRE(pickingCamera);
        pickingCamera->setSource(pickingSource);
        pickingCamera->show();
        pickingCamera->findChild<QButtonGroup*>("PointCloudCameraColor")
                ->button(1)
                ->setChecked(true);
        pickingCamera->findChild<QSlider*>("PointCloudCameraOpacity")
                ->setValue(100);
        auto* pickingMode =
                picking.findChild<QComboBox*>("PointCloudColorMode");
        auto* pickingCameraView = pickingCamera->findChild<QGraphicsView*>(
                "PointCloudCameraView");
        REQUIRE(pickingMode && pickingCameraView);
        const auto pickingColor = [&] {
            QApplication::processEvents();
            return pickingCameraView->viewport()->grab().toImage().pixelColor(
                    pickingCameraView->mapFromScene(32.5, 32.5));
        };
        pickingMode->setCurrentIndex(0);
        REQUIRE(pickingColor() == QColor(Qt::red));
        const auto pickingScale = pickingCameraView->transform();
        pickingMode->setCurrentIndex(2);
        REQUIRE(pickingColor() == QColor(Qt::white) &&
                pickingCameraView->transform() == pickingScale);
        pickingCamera->findChild<QToolButton*>("PointCloudCameraPoints")
                ->setChecked(false);
        const auto goldPixels = [&] {
            QApplication::processEvents();
            const QImage image =
                    pickingCameraView->viewport()->grab().toImage();
            int count = 0;
            for (int y = 0; y < image.height(); ++y)
                for (int x = 0; x < image.width(); ++x) {
                    const QColor c = image.pixelColor(x, y);
                    count += c.red() > 150 && c.green() > 100 && c.blue() < 100;
                }
            return count;
        };
        REQUIRE(pickingCamera->hasImage() && goldPixels() == 0);
        picking.findChild<QAction*>("CuboidDraw")->trigger();
        app.processEvents();
        const QPointF mainCenter(viewport->width() / 2.,
                                 viewport->height() / 2.);
        QMouseEvent preview(QEvent::MouseMove, mainCenter, Qt::NoButton,
                            Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(viewport, &preview);
        REQUIRE(goldPixels() > 50);
        QEvent leavePreview(QEvent::Leave);
        QApplication::sendEvent(viewport, &leavePreview);
        REQUIRE(goldPixels() == 0);
        QWheelEvent previewZoom = wheelEvent(
                mainCenter, viewport->mapToGlobal(mainCenter.toPoint()), 120);
        QApplication::sendEvent(viewport, &previewZoom);
        REQUIRE(goldPixels() > 50);
        QApplication::sendEvent(viewport, &preview);
        REQUIRE(goldPixels() > 50);
        QMouseEvent outsidePreview(QEvent::MouseMove, QPointF(-1, -1),
                                   Qt::NoButton, Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(viewport, &outsidePreview);
        REQUIRE(goldPixels() == 0);
        QApplication::sendEvent(viewport, &preview);
        REQUIRE(goldPixels() > 50);
        QKeyEvent cancel(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
        QApplication::sendEvent(viewport, &cancel);
        REQUIRE(goldPixels() == 0);
        QMouseEvent doubleClick(QEvent::MouseButtonDblClick, mainCenter,
                                Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(viewport, &doubleClick);
        REQUIRE(boxes->count() == 0);
        // Empty-space double click fits the cloud; restore the top direction.
        picking.findChild<QAction*>("PointCloudViewTop")->trigger();
        picking.findChild<QAction*>("CuboidDraw")->trigger();
        QApplication::sendEvent(viewport, &doubleClick);
        app.processEvents();
        REQUIRE(boxes->count() == 1 && action(QKeySequence::Save));
        QVector<PointCloudCuboid> created;
        REQUIRE(loadPointCloudCuboids(
                temporary.filePath("occlusion.cuboids.json"), cloud, &created,
                &error));
        REQUIRE(created.size() == 1 && std::abs(created[0].center[0]) < 1e-10 &&
                std::abs(created[0].center[1]) < 1e-10 &&
                std::abs(created[0].center[2] - 2) < 1e-10);
        REQUIRE(created[0].size == CuboidVector({1, 1, 1}));
        REQUIRE(action(QKeySequence::Undo) && boxes->count() == 0);
        REQUIRE(action(QKeySequence::Save));
        picking.close();
    }
    if (app.arguments().size() > 2) {
        const QDir fixture(app.arguments().at(2));
        window.openFile(fixture.filePath("velodyne/000008.bin"));
        REQUIRE(window.loadClassDefinitions(
                fixture.filePath("../../pointcloud_classes.json"), &error));
        objects->setCurrentRow(0);
        REQUIRE(trigger("CuboidFocus"));
        PointCloudCameraSource fixtureCamera;
        REQUIRE(loadPointCloudCameraSource(fixture.filePath("image_02"),
                                           fixture.filePath("calibration.json"),
                                           &fixtureCamera, &error));
        cameraPanel->setSource(fixtureCamera);
        cameraPanel->show();
        REQUIRE(cameraPanel->hasImage() && cameraPanel->projectionEnabled());
    }
    view->update();
    QEventLoop repaintLoop;
    QTimer::singleShot(100, &repaintLoop, &QEventLoop::quit);
    repaintLoop.exec();
    if (app.arguments().size() > 2) {
        const auto rendered = view->grabFramebuffer();
        int blue = 0;
        for (int y = 0; y < rendered.height(); ++y)
            for (int x = 0; x < rendered.width(); ++x) {
                const auto c = rendered.pixelColor(x, y);
                blue += c.blue() > 120 && c.blue() > c.red() + 40 &&
                        c.green() > 50;
            }
        REQUIRE(blue > 50);
    }
    const auto screenshot = app.arguments().value(1);
    if (!screenshot.isEmpty()) {
        app.processEvents();
        // QWidget::grab re-renders the parent and omits QPainter content in
        // the QOpenGLWidget on Qt 5. Capture the composited window instead.
        const auto captured =
                window.windowHandle()->screen()->grabWindow(window.winId());
        REQUIRE(!captured.isNull() && captured.save(screenshot));
        if (app.arguments().size() > 2) {
            const auto origin = view->mapTo(&window, QPoint());
            const double ratio = captured.devicePixelRatioF();
            const auto mainView = captured.toImage().copy(QRect(
                    qRound(origin.x() * ratio), qRound(origin.y() * ratio),
                    qRound(view->width() * ratio),
                    qRound(view->height() * ratio)));
            int blue = 0;
            for (int y = 0; y < mainView.height(); ++y)
                for (int x = 0; x < mainView.width(); ++x) {
                    const auto c = mainView.pixelColor(x, y);
                    blue += c.blue() > 120 && c.blue() > c.red() + 40 &&
                            c.green() > 50;
                }
            REQUIRE(blue > 50);
        }
    }
    std::puts(
            "PASS actual 3D cuboid window: edit, multi-camera/configuration, "
            "separate task classes, dataset dialogs, preview/cancel/import, "
            "segmentation preservation");
    return 0;
}
