// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QCoreApplication>
#include <QJsonArray>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "AnnotationCanvas.h"
#include "AnnotationNavigator.h"

namespace {
void mouse(AnnotationCanvas& canvas,
           QEvent::Type type,
           const QPointF& point,
           Qt::MouseButton button = Qt::LeftButton,
           Qt::KeyboardModifiers modifiers = Qt::NoModifier) {
    const Qt::MouseButtons buttons =
            type == QEvent::MouseButtonRelease ? Qt::NoButton : Qt::LeftButton;
    QMouseEvent event(type, point, button, buttons, modifiers);
    QCoreApplication::sendEvent(&canvas, &event);
}

void drag(AnnotationCanvas& canvas,
          const QPointF& from,
          const QPointF& to,
          Qt::KeyboardModifiers modifiers = Qt::NoModifier) {
    mouse(canvas, QEvent::MouseButtonPress, from, Qt::LeftButton, modifiers);
    mouse(canvas, QEvent::MouseMove, to, Qt::LeftButton, modifiers);
    mouse(canvas, QEvent::MouseButtonRelease, to, Qt::LeftButton, modifiers);
}

void finish(AnnotationCanvas& canvas) {
    QKeyEvent event(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
    QCoreApplication::sendEvent(&canvas, &event);
}

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    // A fractional click just outside the image must not navigate, even when
    // rounding it to an integer would land on the image border.
    AnnotationNavigator navigator;
    navigator.resize(200, 200);
    navigator.setImage(QImage(100, 100, QImage::Format_RGB32));
    int navigationCount = 0;
    QPointF navigation;
    QObject::connect(&navigator, &AnnotationNavigator::navigationRequested,
                     [&](const QPointF& point) {
                         ++navigationCount;
                         navigation = point;
                     });
    QMouseEvent outside(QEvent::MouseButtonPress, QPointF(9.75, 50.25),
                        Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(&navigator, &outside);
    if (!check(navigationCount == 0,
               "Navigator must reject fractional clicks outside the image"))
        return 1;
    QMouseEvent inside(QEvent::MouseButtonPress, QPointF(10.25, 50.25),
                       Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(&navigator, &inside);
    if (!check(navigationCount == 1 &&
                       std::abs(navigation.x() - 0.25 / 180.0) < 1e-9 &&
                       std::abs(navigation.y() - 40.25 / 180.0) < 1e-9,
               "Navigator must normalize the original fractional coordinates"))
        return 1;
    AnnotationCanvas canvas;
    canvas.resize(400, 400);
    canvas.setImage(QImage(100, 100, QImage::Format_RGB32));
    canvas.show();
    QCoreApplication::processEvents();
    canvas.setCurrentLabel(QStringLiteral("object"));

    // Display adjustments are cache-only. They follow the upstream 0..150
    // slider mapping (50 is neutral), including the source 16-bit formula,
    // while the source image remains untouched for inference and export.
    AnnotationCanvas display;
    // Format_Grayscale16 requires Qt 5.13+; images can never carry that
    // format on Qt 5.12 (Ubuntu focal), so skip the 16-bit cases there.
#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
    QImage gray16(2, 1, QImage::Format_Grayscale16);
    auto* grayRow = reinterpret_cast<quint16*>(gray16.scanLine(0));
    grayRow[0] = 1000;
    grayRow[1] = 3000;
    display.setImage(gray16);
    display.setBrightnessContrast(100, 75);  // brightness 2.0, contrast 1.5
    const auto* adjusted = reinterpret_cast<const quint16*>(
            display.displayImage().constScanLine(0));
    const double expectedMean = 2000.0;
    const auto expected = [expectedMean](double value) {
        return static_cast<quint16>(std::max(
                0.0, std::min(65535.0,
                              value * 2.0 * 1.5 + expectedMean * 2.0 * -0.5)));
    };
    if (!check(adjusted[0] == expected(1000.0) &&
                       adjusted[1] == expected(3000.0) &&
                       reinterpret_cast<const quint16*>(
                               gray16.constScanLine(0))[0] == 1000,
               "Display adjustment must preserve source pixels and apply the "
               "16-bit brightness/contrast formula"))
        return 1;
    display.setBrightnessContrast(50, 50);
    if (!check(display.displayImage().constBits() == gray16.constBits(),
               "Neutral display adjustment must reuse the source image"))
        return 1;
#endif
    QImage rgb(1, 1, QImage::Format_RGB32);
    rgb.setPixel(0, 0, qRgb(50, 75, 100));
    display.setImage(rgb);
    display.setBrightness(100);
    if (!check(display.displayImage().pixel(0, 0) == qRgb(100, 150, 200) &&
                       rgb.pixel(0, 0) == qRgb(50, 75, 100),
               "RGB display brightness must not change the source pixel"))
        return 1;
    QImage rgbFixture(5, 1, QImage::Format_RGB32);
    const QRgb fixturePixels[] = {qRgb(1, 2, 3), qRgb(7, 29, 51),
                                  qRgb(50, 75, 100), qRgb(80, 120, 140),
                                  qRgb(240, 250, 255)};
    for (int x = 0; x < 5; ++x) rgbFixture.setPixel(x, 0, fixturePixels[x]);
    display.setImage(rgbFixture);
    display.setBrightnessContrast(65, 75);
    const QRgb expectedPillow[] = {qRgb(0, 0, 0), qRgb(0, 3, 46),
                                   qRgb(45, 93, 142), qRgb(103, 181, 220),
                                   qRgb(255, 255, 255)};
    for (int x = 0; x < 5; ++x) {
        const QRgb actual = display.displayImage().pixel(x, 0);
        if (!check(actual == expectedPillow[x],
                   "RGB display adjustment must match Pillow integer blending"))
            return 1;
    }
    display.setShapeOpacity(-10);
    if (!check(display.shapeOpacity() == 0,
               "Shape opacity must clamp to the supported range"))
        return 1;
    display.setShapeOpacity(110);
    if (!check(display.shapeOpacity() == 100,
               "Shape opacity must clamp at 100 percent"))
        return 1;

    canvas.setTool(QStringLiteral("rectangle"));
    drag(canvas, QPointF(50, 50), QPointF(150, 150));
    canvas.setTool(QStringLiteral("rotation"));
    drag(canvas, QPointF(170, 50), QPointF(240, 120));
    canvas.setTool(QStringLiteral("circle"));
    drag(canvas, QPointF(250, 50), QPointF(330, 130));
    canvas.setTool(QStringLiteral("line"));
    drag(canvas, QPointF(50, 170), QPointF(140, 220));
    canvas.setTool(QStringLiteral("point"));
    mouse(canvas, QEvent::MouseButtonPress, QPointF(180, 180));
    mouse(canvas, QEvent::MouseButtonRelease, QPointF(180, 180));
    canvas.setTool(QStringLiteral("cuboid"));
    drag(canvas, QPointF(250, 170), QPointF(330, 250));
    canvas.setTool(QStringLiteral("quadrilateral"));
    for (const QPointF& p : {QPointF(30, 260), QPointF(80, 260),
                             QPointF(90, 310), QPointF(40, 320)}) {
        mouse(canvas, QEvent::MouseButtonPress, p);
        mouse(canvas, QEvent::MouseButtonRelease, p);
    }
    canvas.setTool(QStringLiteral("polygon"));
    for (const QPointF& p :
         {QPointF(120, 260), QPointF(170, 260), QPointF(150, 310)}) {
        mouse(canvas, QEvent::MouseButtonPress, p);
        mouse(canvas, QEvent::MouseButtonRelease, p);
    }
    finish(canvas);
    canvas.setTool(QStringLiteral("linestrip"));
    for (const QPointF& p : {QPointF(210, 280), QPointF(290, 300)}) {
        mouse(canvas, QEvent::MouseButtonPress, p);
        mouse(canvas, QEvent::MouseButtonRelease, p);
    }
    finish(canvas);

    const auto& shapes = canvas.shapes();
    if (!check(shapes.size() == 9,
               "Nine geometry tools must create nine shapes"))
        return 1;
    if (!check(shapes[0].points.size() == 4,
               "Rectangle must store four corners"))
        return 1;
    if (!check(shapes[5].points.size() == 8, "Cuboid must store eight corners"))
        return 1;
    for (const AnnotationShape& shape : shapes) {
        const AnnotationShape roundTrip =
                AnnotationShape::fromJson(shape.toJson());
        if (!check(roundTrip.type == shape.type &&
                           roundTrip.points.size() == shape.points.size(),
                   "Shape JSON round-trip failed"))
            return 1;
    }
    QJsonObject source;
    source.insert(QStringLiteral("label"), QStringLiteral("person"));
    source.insert(QStringLiteral("shape_type"), QStringLiteral("rectangle"));
    source.insert(QStringLiteral("points"),
                  QJsonArray{QJsonArray{1, 2}, QJsonArray{3, 4}});
    source.insert(QStringLiteral("score"), 0.86);
    source.insert(
            QStringLiteral("attributes"),
            QJsonObject{{QStringLiteral("pose"), QStringLiteral("left")}});
    const QJsonObject restored = AnnotationShape::fromJson(source).toJson();
    if (!check(restored.value(QStringLiteral("score")).toDouble() == 0.86 &&
                       restored.value(QStringLiteral("attributes"))
                                       .toObject() ==
                               source.value(QStringLiteral("attributes"))
                                       .toObject() &&
                       restored.value(QStringLiteral("points"))
                                       .toArray()
                                       .size() == 4,
               "Legacy rectangle and custom fields must survive JSON "
               "round-trip"))
        return 1;
    source.insert(QStringLiteral("visible"), false);
    const AnnotationShape hidden = AnnotationShape::fromJson(source);
    if (!check(!hidden.visible && !hidden.toJson()
                                           .value(QStringLiteral("visible"))
                                           .toBool(true),
               "Hidden shape visibility must survive JSON round-trip"))
        return 1;

    AnnotationCanvas editor;
    editor.resize(400, 400);
    editor.setImage(QImage(100, 100, QImage::Format_RGB32));
    editor.show();
    QCoreApplication::processEvents();
    const auto screen = [](double x, double y) {
        return QPointF(12 + x * 3.76, 12 + y * 3.76);
    };
    AnnotationShape rectangle;
    rectangle.type = QStringLiteral("rectangle");
    rectangle.points = {QPointF(10, 10), QPointF(30, 10), QPointF(30, 30),
                        QPointF(10, 30)};
    editor.setShapes({rectangle});
    int edits = 0;
    int changes = 0;
    int cancellations = 0;
    QObject::connect(&editor, &AnnotationCanvas::editStarted,
                     [&edits]() { ++edits; });
    QObject::connect(&editor, &AnnotationCanvas::shapesChanged,
                     [&changes]() { ++changes; });
    QObject::connect(&editor, &AnnotationCanvas::creationCanceled,
                     [&cancellations]() { ++cancellations; });
    mouse(editor, QEvent::MouseButtonPress, screen(10, 10));
    mouse(editor, QEvent::MouseButtonRelease, screen(10, 10));
    if (!check(edits == 0 && changes == 0,
               "A click without movement must not create an undo entry"))
        return 1;
    drag(editor, screen(10, 10), screen(5, 6));
    const auto& corners = editor.shapes().first().points;
    if (!check(edits == 1 && changes == 1 && corners[0] == QPointF(5, 6) &&
                       corners[1] == QPointF(30, 6) &&
                       corners[2] == QPointF(30, 30) &&
                       corners[3] == QPointF(5, 30),
               "Rectangle corner drag must preserve four right angles"))
        return 1;

    AnnotationShape rotation;
    rotation.type = QStringLiteral("rotation");
    rotation.points = {QPointF(20, 20), QPointF(40, 20), QPointF(40, 40),
                       QPointF(20, 40)};
    editor.setShapes({rotation});
    editor.selectShape(0);
    drag(editor, screen(30, 20 - 32.0 / 3.76), screen(48.5, 30));
    const AnnotationShape turned = editor.shapes().first();
    const double angle =
            turned.toJson().value(QStringLiteral("direction")).toDouble();
    if (!check(std::abs(angle - 1.5707963267948966) < 0.03 &&
                       std::abs(turned.points[0].x() - 40) < 0.1,
               "Rotation handle must turn geometry and update direction"))
        return 1;

    AnnotationShape cuboid;
    cuboid.type = QStringLiteral("cuboid");
    cuboid.points = {QPointF(10, 10), QPointF(30, 10), QPointF(30, 30),
                     QPointF(10, 30), QPointF(15, 5),  QPointF(35, 5),
                     QPointF(35, 25), QPointF(15, 25)};
    const QJsonObject cuboidData =
            cuboid.toJson().value(QStringLiteral("cuboid3d")).toObject();
    if (!check(cuboidData.value(QStringLiteral("depth_vector")).toArray() ==
                               QJsonArray{5, -5} &&
                       cuboidData.value(QStringLiteral("vertices2d_order"))
                                       .toArray()
                                       .size() == 8,
               "Cuboid export must include upstream depth metadata"))
        return 1;

    editor.setShapes({});
    editor.setTool(QStringLiteral("polygon"));
    const int changesBeforeDraft = changes;
    mouse(editor, QEvent::MouseButtonPress, screen(10, 10));
    mouse(editor, QEvent::MouseButtonRelease, screen(10, 10));
    QKeyEvent escape(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
    QCoreApplication::sendEvent(&editor, &escape);
    if (!check(editor.shapes().isEmpty() && cancellations == 1 &&
                       changes == changesBeforeDraft,
               "Canceling a draft must not mark the document changed"))
        return 1;

    QImage magicImage(100, 100, QImage::Format_RGB32);
    magicImage.fill(Qt::black);
    for (int y = 10; y < 30; ++y)
        for (int x = 10; x < 30; ++x) magicImage.setPixelColor(x, y, Qt::red);
    magicImage.setPixelColor(90, 90, Qt::red);
    editor.setImage(magicImage);
    editor.setTool(QStringLiteral("magic_wand"));
    mouse(editor, QEvent::MouseButtonPress, screen(20, 20));
    mouse(editor, QEvent::MouseButtonRelease, screen(20, 20));
    mouse(editor, QEvent::MouseButtonPress, screen(20, 20), Qt::RightButton);
    if (!check(editor.shapes().size() == 1 &&
                       editor.shapes()[0].type == QStringLiteral("polygon") &&
                       editor.shapes()[0].points.size() == 4 &&
                       editor.shapes()[0].points.contains(QPointF(10, 10)) &&
                       editor.shapes()[0].points.contains(QPointF(29, 29)),
               "Magic wand must extract only the seed-connected region"))
        return 1;

    editor.setTool(QStringLiteral("edit"));
    editor.selectShape(0);
    const QVector<QPointF> original = editor.shapes()[0].points;
    editor.setBrushRadius(8);
    if (!check(editor.setBrushEditing(true),
               "Selected polygon must enter brush mode"))
        return 1;
    drag(editor, screen(28, 20), screen(43, 20));
    editor.brushUndo();
    editor.setBrushEditing(false);
    if (!check(editor.shapes()[0].points == original,
               "Undoing a brush stroke must restore the original polygon"))
        return 1;
    editor.setBrushEditing(true);
    drag(editor, screen(28, 20), screen(43, 20));
    editor.setBrushEditing(false);
    double rightmost = 0;
    for (const QPointF& point : editor.shapes()[0].points)
        rightmost = std::max(rightmost, point.x());
    if (!check(rightmost > 45, "Brush addition must extend the saved polygon"))
        return 1;
    const QVector<QPointF> expanded = editor.shapes()[0].points;
    editor.setBrushEditing(true);
    drag(editor, screen(43, 20), screen(43, 20), Qt::ControlModifier);
    editor.setBrushEditing(false, true);
    if (!check(editor.shapes()[0].points == expanded,
               "Canceling a brush erase must preserve saved geometry"))
        return 1;
    editor.setBrushEditing(true);
    drag(editor, screen(48, 20), screen(48, 20), Qt::ControlModifier);
    editor.setBrushEditing(false);
    double erasedRightmost = 0;
    for (const QPointF& point : editor.shapes()[0].points)
        erasedRightmost = std::max(erasedRightmost, point.x());
    if (!check(erasedRightmost < rightmost,
               "Ctrl brush stroke must erase pixels"))
        return 1;

    QImage exact(100, 100, QImage::Format_ARGB32);
    exact.fill(Qt::transparent);
    {
        QPainter painter(&exact);
        painter.fillRect(QRect(10, 10, 60, 60), QColor(0, 180, 255, 110));
        painter.setCompositionMode(QPainter::CompositionMode_Clear);
        painter.fillRect(QRect(30, 30, 20, 20), Qt::transparent);
        painter.setCompositionMode(QPainter::CompositionMode_Source);
        painter.fillRect(QRect(80, 80, 5, 5), QColor(0, 180, 255, 110));
    }
    AnnotationShape ring;
    ring.type = QStringLiteral("polygon");
    ring.label = QStringLiteral("ring");
    ring.points = {QPointF(10, 10), QPointF(70, 10), QPointF(70, 70),
                   QPointF(10, 70)};
    ring.mask = exact;
    editor.setShapes({ring});
    editor.selectShape(0);
    editor.setBrushLabel(QStringLiteral("ring"));
    if (!check(editor.setBrushEditing(true),
               "Exact mask must reopen in brush mode"))
        return 1;
    editor.setBrushErase(true);
    drag(editor, screen(20, 20), screen(20, 20));
    editor.setBrushEditing(false);
    const AnnotationShape restoredMask =
            AnnotationShape::fromJson(editor.shapes().first().toJson());
    if (!check(!restoredMask.mask.isNull() &&
                       qAlpha(restoredMask.mask.pixel(40, 40)) == 0 &&
                       qAlpha(restoredMask.mask.pixel(82, 82)) > 0 &&
                       qAlpha(restoredMask.mask.pixel(20, 20)) == 0 &&
                       qAlpha(restoredMask.mask.pixel(60, 60)) > 0,
               "Brush save must retain holes, detached islands and erased "
               "pixels"))
        return 1;
    editor.setBrushErase(false);

    editor.setShapes({ring});
    editor.selectShape(0);
    editor.setBrushRadius(3);
    editor.setBrushOperation(2);
    if (!check(editor.setBrushEditing(true),
               "Instance mask must enter invert brush mode"))
        return 1;
    mouse(editor, QEvent::MouseButtonPress, screen(20, 20));
    mouse(editor, QEvent::MouseMove, screen(30, 30));
    mouse(editor, QEvent::MouseMove, screen(35, 35));
    mouse(editor, QEvent::MouseMove, screen(40, 40));
    mouse(editor, QEvent::MouseButtonRelease, screen(40, 40));
    editor.brushUndo();
    editor.brushRedo();
    editor.setBrushEditing(false);
    if (!check(qAlpha(editor.shapes()[0].mask.pixel(20, 20)) == 0 &&
                       qAlpha(editor.shapes()[0].mask.pixel(40, 40)) > 0 &&
                       qAlpha(editor.shapes()[0].mask.pixel(60, 60)) > 0 &&
                       qAlpha(editor.shapes()[0].mask.pixel(82, 82)) > 0,
               "Invert stroke must toggle ring and hole pixels once without "
               "losing islands"))
        return 1;
    editor.setBrushOperation(0);

    editor.setShapes({ring});
    editor.selectShape(0);
    editor.setBrushLabel(QStringLiteral("ring"));
    if (!check(editor.setBrushEditing(true),
               "Brush must enter before right-click exit test"))
        return 1;
    mouse(editor, QEvent::MouseButtonPress, screen(20, 20), Qt::RightButton);
    if (!check(!editor.brushEditing(),
               "Right press must exit brush mode immediately"))
        return 1;
    mouse(editor, QEvent::MouseButtonRelease, screen(20, 20), Qt::RightButton);
    if (!check(!editor.brushEditing(),
               "Right release must keep brush mode closed"))
        return 1;

    QImage tiny(100, 100, QImage::Format_ARGB32);
    tiny.fill(Qt::transparent);
    tiny.setPixel(50, 50, qRgba(0, 180, 255, 110));
    AnnotationShape tinyShape;
    tinyShape.type = QStringLiteral("polygon");
    tinyShape.label = QStringLiteral("tiny");
    tinyShape.points = {QPointF(50, 50), QPointF(51, 50), QPointF(51, 51),
                        QPointF(50, 51)};
    tinyShape.mask = tiny;
    editor.setShapes({tinyShape});
    editor.selectShape(0);
    editor.setBrushLabel(QStringLiteral("tiny-renamed"));
    if (!check(editor.setBrushEditing(true),
               "Single-pixel mask must enter brush mode"))
        return 1;
    editor.setBrushEditing(false);
    if (!check(editor.shapes().size() == 1 &&
                       editor.shapes().first().label ==
                               QStringLiteral("tiny-renamed") &&
                       qAlpha(editor.shapes().first().mask.pixel(50, 50)) > 0,
               "Brush commit must retain a single-pixel instance"))
        return 1;

    QImage background(100, 100, QImage::Format_ARGB32_Premultiplied);
    background.fill(qRgba(0, 180, 255, 110));
    QImage foreground(100, 100, QImage::Format_ARGB32_Premultiplied);
    foreground.fill(Qt::transparent);
    {
        QPainter painter(&foreground);
        painter.fillRect(QRect(10, 10, 20, 20), QColor(0, 180, 255, 110));
        painter.setCompositionMode(QPainter::CompositionMode_Clear);
        painter.fillRect(QRect(20, 20, 4, 4), Qt::transparent);
    }
    for (int y = 0; y < background.height(); ++y)
        for (int x = 0; x < background.width(); ++x)
            if (qAlpha(foreground.pixel(x, y)) > 0)
                background.setPixel(x, y, 0);
    AnnotationShape classZero;
    classZero.type = QStringLiteral("polygon");
    classZero.label = QStringLiteral("background");
    classZero.points = {QPointF(0, 0), QPointF(100, 0), QPointF(100, 100),
                        QPointF(0, 100)};
    classZero.mask = background;
    classZero.extra.insert(QStringLiteral("class_id"), 0);
    AnnotationShape classOne;
    classOne.type = QStringLiteral("polygon");
    classOne.label = QStringLiteral("object");
    classOne.points = {QPointF(10, 10), QPointF(30, 10), QPointF(30, 30),
                       QPointF(10, 30)};
    classOne.mask = foreground;
    classOne.extra.insert(QStringLiteral("class_id"), 1);
    editor.setShapes({classZero, classOne});
    editor.selectShape(1);
    editor.setBrushLabel(QStringLiteral("object"));
    editor.setBrushClassId(1);
    editor.setBrushRadius(2);
    if (!check(editor.setBrushEditing(true),
               "Semantic mask must enter brush mode"))
        return 1;
    drag(editor, screen(40, 40), screen(40, 40));
    editor.setBrushEditing(false);
    if (!check(qAlpha(editor.shapes()[1].mask.pixel(40, 40)) > 0 &&
                       qAlpha(editor.shapes()[0].mask.pixel(40, 40)) == 0 &&
                       qAlpha(editor.shapes()[1].mask.pixel(22, 22)) == 0 &&
                       qAlpha(editor.shapes()[0].mask.pixel(22, 22)) > 0,
               "Semantic paint must clear other classes and retain holes"))
        return 1;
    editor.setBrushErase(true);
    if (!check(editor.setBrushEditing(true),
               "Semantic mask must reopen for reverse painting"))
        return 1;
    drag(editor, screen(15, 15), screen(15, 15));
    editor.setBrushEditing(false);
    if (!check(qAlpha(editor.shapes()[1].mask.pixel(15, 15)) == 0 &&
                       qAlpha(editor.shapes()[0].mask.pixel(15, 15)) > 0,
               "Semantic erase must restore the background class"))
        return 1;
    editor.setBrushErase(false);
    AnnotationShape road = classZero;
    road.label = QStringLiteral("road");
    editor.setShapes({road, classOne});
    editor.selectShape(1);
    editor.setBrushLabel(QStringLiteral("object"));
    editor.setBrushClassId(1);
    editor.setBrushErase(true);
    if (!check(editor.setBrushEditing(true),
               "Non-background class map must enter brush mode"))
        return 1;
    drag(editor, screen(15, 15), screen(15, 15));
    editor.setBrushEditing(false);
    if (!check(qAlpha(editor.shapes()[1].mask.pixel(15, 15)) == 0 &&
                       qAlpha(editor.shapes()[0].mask.pixel(15, 15)) == 0,
               "Erasing must not assume semantic class zero is background"))
        return 1;
    editor.setBrushErase(false);
    editor.setBrushLabel(QStringLiteral("replacement"));
    editor.setBrushClassId(2);
    if (!check(editor.setBrushEditing(true),
               "Semantic class must reopen in brush mode"))
        return 1;
    editor.setBrushEditing(false);
    if (!check(editor.shapes()[1].label == QStringLiteral("replacement") &&
                       editor.shapes()[1]
                                       .extra.value(QStringLiteral("class_id"))
                                       .toInt(-1) == 2,
               "Changing brush target must update semantic class ID"))
        return 1;

    editor.setShapes({});
    editor.setBrushLabel(QStringLiteral("new-class"));
    editor.setBrushClassId(7);
    editor.setBrushRadius(4);
    if (!check(editor.setBrushEditing(true),
               "Brush must start without a selected instance when a class is "
               "set"))
        return 1;
    drag(editor, screen(35, 35), screen(55, 55));
    editor.setBrushEditing(false);
    if (!check(editor.shapes().size() == 1 &&
                       editor.shapes().first().label ==
                               QStringLiteral("new-class") &&
                       editor.shapes().first()
                                       .extra.value(QStringLiteral("class_id"))
                                       .toInt(-1) == 7 &&
                       qAlpha(editor.shapes().first().mask.pixel(45, 45)) > 0,
               "Brush must create a class-aware polygon when no instance is "
               "selected"))
        return 1;
    editor.setShapes({});
    editor.setBrushLabel(QStringLiteral("cancelled"));
    editor.setBrushClassId(8);
    if (!check(editor.setBrushEditing(true),
               "New brush target must be cancellable"))
        return 1;
    drag(editor, screen(35, 35), screen(55, 55));
    editor.setBrushEditing(false, true);
    if (!check(editor.shapes().isEmpty(),
               "Cancelling a new brush target must not create an empty shape"))
        return 1;

    // Mask visibility is an independent view layer. Hiding it must change
    // only the rendered pixels; the stored alpha mask must remain intact.
    AnnotationCanvas visibility;
    visibility.resize(400, 400);
    QImage visibilityImage(100, 100, QImage::Format_RGB32);
    visibilityImage.fill(Qt::black);
    visibility.setImage(visibilityImage);
    AnnotationShape visibleMask;
    visibleMask.type = QStringLiteral("polygon");
    visibleMask.label = QStringLiteral("visible");
    visibleMask.points = {QPointF(10, 10), QPointF(30, 10), QPointF(30, 30),
                          QPointF(10, 30)};
    visibleMask.mask = QImage(100, 100, QImage::Format_ARGB32_Premultiplied);
    visibleMask.mask.fill(Qt::transparent);
    {
        QPainter painter(&visibleMask.mask);
        painter.fillRect(QRect(10, 10, 20, 20), QColor(255, 0, 0, 180));
    }
    visibility.setShapes({visibleMask});
    visibility.setShowShapes(false);
    visibility.setShowLabels(false);
    const auto renderVisibility = [&visibility]() {
        QImage frame(400, 400, QImage::Format_ARGB32);
        frame.fill(Qt::black);
        QPainter painter(&frame);
        visibility.render(&painter);
        return frame;
    };
    const QImage masksVisible = renderVisibility();
    visibility.setMaskOpacity(20);
    const QImage dimmedMasks = renderVisibility();
    if (!check(dimmedMasks.pixelColor(88, 88).red() <
                               masksVisible.pixelColor(88, 88).red() &&
                       qAlpha(visibility.shapes().first().mask.pixel(20, 20)) ==
                               180 &&
                       visibility.shapeOpacity() == 100,
               "Mask opacity must affect only the rendered mask"))
        return 1;
    visibility.setMaskOpacity(72);
    visibility.setShowMasks(false);
    const QImage masksHidden = renderVisibility();
    if (!check(masksVisible.pixelColor(88, 88).red() > 20 &&
                       masksHidden.pixelColor(88, 88).red() < 5 &&
                       qAlpha(visibility.shapes().first().mask.pixel(20, 20)) >
                               0,
               "Hiding masks must affect rendering without changing mask data"))
        return 1;
    visibility.setShowMasks(true);
    const QImage masksRestored = renderVisibility();
    if (!check(masksRestored.pixelColor(88, 88).red() > 20,
               "Re-enabling masks must restore the rendered overlay"))
        return 1;
    visibility.setShapeOpacity(0);
    const QImage transparentShapes = renderVisibility();
    visibility.setShowMasks(false);
    const QImage absentShapes = renderVisibility();
    if (!check(transparentShapes.pixel(88, 88) == absentShapes.pixel(88, 88) &&
                       qAlpha(visibility.shapes().first().mask.pixel(20, 20)) >
                               0,
               "Zero shape opacity must hide the overlay without erasing the "
               "mask"))
        return 1;
    visibility.setShowLabels(true);
    const QImage labelAtZeroOpacity = renderVisibility();
    visibility.setShapeOpacity(100);
    const QImage labelAtFullOpacity = renderVisibility();
    if (!check(labelAtZeroOpacity.copy(QRect(50, 25, 100, 25)) ==
                       labelAtFullOpacity.copy(QRect(50, 25, 100, 25)),
               "Annotation text must remain at full opacity"))
        return 1;

    AnnotationCanvas overlays;
    overlays.resize(400, 400);
    overlays.setImage(visibilityImage);
    AnnotationShape key;
    key.type = QStringLiteral("rotation");
    key.label = QStringLiteral("key");
    key.points = {QPointF(20, 30), QPointF(40, 30), QPointF(40, 50),
                  QPointF(20, 50)};
    key.extra.insert(QStringLiteral("group_id"), 1);
    key.extra.insert(QStringLiteral("description"), QStringLiteral("text"));
    key.extra.insert(
            QStringLiteral("attributes"),
            QJsonObject{{QStringLiteral("role"), QStringLiteral("key")}});
    key.extra.insert(QStringLiteral("kie_linking"),
                     QJsonArray{QJsonArray{1, 2}});
    AnnotationShape value = key;
    value.points = {QPointF(60, 30), QPointF(80, 30), QPointF(80, 50),
                    QPointF(60, 50)};
    value.extra = QJsonObject{{QStringLiteral("group_id"), 2}};
    overlays.setShapes({key, value});
    overlays.setShowShapes(false);
    overlays.setShowMasks(false);
    overlays.setShowLabels(false);
    overlays.setShowScores(false);
    overlays.setShowTexts(false);
    overlays.setShowAttributes(false);
    overlays.setShowLinking(false);
    const auto renderOverlays = [&overlays]() {
        QImage frame(400, 400, QImage::Format_ARGB32);
        frame.fill(Qt::black);
        QPainter painter(&frame);
        overlays.render(&painter);
        return frame;
    };
    const QImage plain = renderOverlays();
    overlays.setShowTexts(true);
    if (!check(renderOverlays() != plain,
               "Description visibility must change canvas pixels"))
        return 1;
    overlays.setShowTexts(false);
    overlays.setShowAttributes(true);
    if (!check(renderOverlays() != plain,
               "Attribute visibility must change canvas pixels"))
        return 1;
    overlays.setShowAttributes(false);
    overlays.setShowGroups(true);
    if (!check(renderOverlays() != plain,
               "Group visibility must change canvas pixels"))
        return 1;
    overlays.setShowGroups(false);
    overlays.setShowDegrees(true);
    if (!check(renderOverlays() != plain,
               "Rotation degree visibility must change canvas pixels"))
        return 1;
    overlays.setShowDegrees(false);
    overlays.setShowLinking(true);
    if (!check(renderOverlays() != plain,
               "KIE linking visibility must change canvas pixels"))
        return 1;
    overlays.setShowLinking(false);
    if (!check(renderOverlays() == plain &&
                       overlays.shapes().first().toJson() == key.toJson(),
               "View toggles must restore pixels without modifying sidecar "
               "data"))
        return 1;
    key.visible = false;
    overlays.setShapes({key, value});
    const QImage withoutLink = renderOverlays();
    overlays.setShowLinking(true);
    if (!check(renderOverlays() == withoutLink,
               "Hidden source shapes must not draw KIE links"))
        return 1;

    AnnotationCanvas groups;
    groups.resize(400, 400);
    groups.setImage(visibilityImage);
    groups.show();
    QCoreApplication::processEvents();
    AnnotationShape left;
    left.type = QStringLiteral("rectangle");
    left.points = {QPointF(10, 10), QPointF(20, 10), QPointF(20, 20),
                   QPointF(10, 20)};
    left.extra.insert(QStringLiteral("group_id"), 7);
    AnnotationShape right = left;
    right.points = {QPointF(40, 10), QPointF(50, 10), QPointF(50, 20),
                    QPointF(40, 20)};
    groups.setShapes({left, right});
    groups.setShowGroups(true);
    mouse(groups, QEvent::MouseButtonPress, screen(30, 15));
    if (!check(groups.selectedIndices() == QSet<int>({0, 1}),
               "Clicking between grouped shapes must select the full group"))
        return 1;
    mouse(groups, QEvent::MouseMove, screen(35, 20));
    mouse(groups, QEvent::MouseButtonRelease, screen(35, 20));
    if (!check(groups.shapes()[0].points[0] == QPointF(15, 15) &&
                       groups.shapes()[1].points[0] == QPointF(45, 15),
               "Dragging empty group bounds must move every member"))
        return 1;
    groups.setShowGroups(false);
    groups.selectShape(-1);
    mouse(groups, QEvent::MouseButtonPress, screen(35, 20));
    mouse(groups, QEvent::MouseButtonRelease, screen(35, 20));
    if (!check(groups.selectedIndices().isEmpty(),
               "Hidden group bounds must not capture empty canvas clicks"))
        return 1;

    AnnotationCanvas prompts;
    prompts.resize(400, 400);
    prompts.setImage(QImage(100, 100, QImage::Format_RGB32));
    prompts.show();
    QCoreApplication::processEvents();
    prompts.setTool(QStringLiteral("sam_point"));
    mouse(prompts, QEvent::MouseButtonPress, screen(20, 20));
    mouse(prompts, QEvent::MouseButtonPress, screen(70, 70), Qt::RightButton);
    prompts.setTool(QStringLiteral("sam_box"));
    drag(prompts, screen(10, 12), screen(60, 64));
    const AnnotationPrompt prompt = prompts.prompt();
    if (!check(prompts.shapes().isEmpty() && prompt.positive.size() == 1 &&
                       prompt.negative.size() == 1 && prompt.useBox &&
                       prompt.box == QRectF(QPointF(10, 12), QPointF(60, 64)),
               "SAM prompts must stay transient and use image coordinates"))
        return 1;
    prompts.clearPrompt();
    if (!check(prompts.prompt().isEmpty(),
               "Clearing SAM prompts must remove all marks"))
        return 1;
    prompts.setTool(QStringLiteral("sam_point"));
    mouse(prompts, QEvent::MouseButtonPress, screen(25, 25));
    prompts.setImage(QImage(100, 100, QImage::Format_RGB32));
    if (!check(prompts.prompt().isEmpty(),
               "Changing images must clear SAM prompts"))
        return 1;

    AnnotationShape second = rectangle;
    second.points = {QPointF(60, 60), QPointF(80, 60), QPointF(80, 80),
                     QPointF(60, 80)};
    editor.setImage(QImage(100, 100, QImage::Format_RGB32));
    editor.setShapes({rectangle, second});
    editor.setTool(QStringLiteral("edit"));
    mouse(editor, QEvent::MouseButtonPress, screen(20, 20));
    mouse(editor, QEvent::MouseButtonRelease, screen(20, 20));
    mouse(editor, QEvent::MouseButtonPress, screen(70, 70), Qt::LeftButton,
          Qt::ControlModifier);
    mouse(editor, QEvent::MouseButtonRelease, screen(70, 70), Qt::LeftButton,
          Qt::ControlModifier);
    if (!check(editor.selectedIndices().size() == 2 &&
                       editor.selectedIndices().contains(0) &&
                       editor.selectedIndices().contains(1),
               "Ctrl click must select both shapes"))
        return 1;
    drag(editor, screen(70, 70), screen(80, 80));
    const auto grouped = editor.shapes();
    if (!check(grouped[0].points[0] == QPointF(20, 20) &&
                       grouped[1].points[0] == QPointF(70, 70),
               "Dragging a multi-selection must move every selected shape by "
               "the same delta"))
        return 1;
    mouse(editor, QEvent::MouseButtonPress, screen(70, 70), Qt::LeftButton,
          Qt::ControlModifier);
    mouse(editor, QEvent::MouseButtonRelease, screen(70, 70), Qt::LeftButton,
          Qt::ControlModifier);
    if (!check(editor.selectedIndices().size() == 1 &&
                       editor.selectedIndices().contains(0),
               "Ctrl click must toggle a selected shape off"))
        return 1;
    drag(editor, screen(30, 30), screen(95, 95));
    const QVector<QPointF> moved = editor.shapes().first().points;
    if (!check(moved[0] == QPointF(80, 80) && moved[2] == QPointF(100, 100),
               "Dragging a shape must stop at the image boundary"))
        return 1;

    // Image-level mask compositing fixture: the source paints a binary
    // silhouette filled with the shape color at mask_opacity. Stored
    // semi-transparent alpha must not fade the fill, and holes stay
    // transparent.
    {
        QImage canvasImage(40, 40, QImage::Format_ARGB32_Premultiplied);
        canvasImage.fill(QColor(255, 255, 255));
        AnnotationCanvas renderCanvas;
        renderCanvas.setImage(canvasImage);
        renderCanvas.setShowMasks(true);
        renderCanvas.setShowShapes(false);
        renderCanvas.setMaskOpacity(255);
        AnnotationShape maskShape;
        maskShape.label = QStringLiteral("compositing");
        maskShape.mask = QImage(40, 40, QImage::Format_ARGB32_Premultiplied);
        maskShape.mask.fill(Qt::transparent);
        {
            QPainter maskPainter(&maskShape.mask);
            // Same stored color, different alpha: 50% vs 100%.
            maskPainter.fillRect(5, 5, 10, 10, QColor(255, 0, 0, 128));
            maskPainter.fillRect(20, 20, 10, 10, QColor(255, 0, 0, 255));
            maskPainter.end();
        }
        const QImage rendered =
                renderCanvas.renderVisualizationImage({maskShape});
        const QRgb semi = rendered.pixel(8, 8);
        const QRgb solid = rendered.pixel(25, 25);
        const QRgb hole = rendered.pixel(15, 15);
        if (!check(semi == solid && qRed(semi) > 0 &&
                           hole == QColor(255, 255, 255).rgba(),
                   "Mask compositing must binarize stored alpha and keep "
                   "holes transparent"))
            return 1;
    }
    {
        // RelateAnything overlay: transient, index based, never part of the
        // shapes or the sidecar, and cleared when the indices could go stale.
        AnnotationCanvas relations;
        relations.resize(400, 400);
        QImage relationImage(100, 100, QImage::Format_RGB32);
        relationImage.fill(Qt::white);
        relations.setImage(relationImage);
        AnnotationShape subject;
        subject.type = QStringLiteral("rectangle");
        subject.label = QStringLiteral("person");
        subject.points = {QPointF(10, 10), QPointF(30, 10), QPointF(30, 40),
                          QPointF(10, 40)};
        AnnotationShape object = subject;
        object.label = QStringLiteral("horse");
        object.points = {QPointF(50, 40), QPointF(90, 40), QPointF(90, 90),
                         QPointF(50, 90)};
        const auto renderRelations = [&relations]() {
            QImage frame(400, 400, QImage::Format_ARGB32);
            frame.fill(Qt::black);
            QPainter painter(&frame);
            relations.render(&painter);
            return frame;
        };
        relations.setShapes({subject, object});
        const QImage bare = renderRelations();
        const QJsonArray edges{QJsonObject{
                {QStringLiteral("subject_index"), 0},
                {QStringLiteral("object_index"), 1},
                {QStringLiteral("predicate"), QStringLiteral("riding")},
                {QStringLiteral("score"), 0.59}}};
        relations.setRelations(edges);
        const QImage drawn = renderRelations();
        if (!check(drawn != bare, "Relation edges must change canvas pixels"))
            return 1;
        relations.setHighlightedRelation(0);
        if (!check(renderRelations() != drawn,
                   "A highlighted relation must render differently"))
            return 1;
        relations.setHighlightedRelation(-1);
        // With shapes hidden the edges must disappear too: compare against
        // the same hidden-shapes frame rendered without any relations.
        relations.setShowShapes(false);
        const QImage hiddenWithEdges = renderRelations();
        relations.clearRelations();
        const QImage hiddenBare = renderRelations();
        if (!check(hiddenWithEdges == hiddenBare,
                   "Hiding shapes must hide the relation overlay"))
            return 1;
        relations.setShowShapes(true);
        relations.setRelations(edges);
        // An edge pointing at a missing shape is skipped, not drawn or fatal.
        relations.setRelations(QJsonArray{
                QJsonObject{{QStringLiteral("subject_index"), 0},
                            {QStringLiteral("object_index"), 7},
                            {QStringLiteral("predicate"), QStringLiteral("on")},
                            {QStringLiteral("score"), 0.9}}});
        if (!check(renderRelations() == bare,
                   "Edges with out-of-range shape indices must be skipped"))
            return 1;
        relations.setRelations(edges);
        relations.setShapes({subject, object});
        if (!check(renderRelations() == bare,
                   "Replacing shapes must clear the relation overlay"))
            return 1;
        relations.setRelations(edges);
        relations.setImage(relationImage);
        relations.setShapes({subject, object});
        relations.setRelations(edges);
        relations.setImage(relationImage);
        relations.setShapes({subject, object});
        if (!check(renderRelations() == bare,
                   "Loading an image must clear the relation overlay"))
            return 1;
        if (!check(relations.shapes().first().toJson() == subject.toJson(),
                   "The relation overlay must not modify shape data"))
            return 1;
    }
    return 0;
}
