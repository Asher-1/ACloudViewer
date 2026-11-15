// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QFile>
#include <QJsonDocument>
#include <QLineF>
#include <QMouseEvent>
#include <cmath>
#include <cstdio>

#include "AnnotationCanvas.h"

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

QVector<QPointF> pointsFromJson(const QJsonArray& array) {
    QVector<QPointF> points;
    for (const QJsonValue& value : array) {
        const auto pair = value.toArray();
        if (pair.size() != 2) return {};
        points.append(QPointF(pair[0].toDouble(), pair[1].toDouble()));
    }
    return points;
}

bool samePoints(const QVector<QPointF>& a, const QVector<QPointF>& b) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); ++i)
        if (QLineF(a[i], b[i]).length() > 1e-8) return false;
    return true;
}

annotationCuboid::Face faceFromName(const QString& name) {
    if (name == QStringLiteral("left")) return annotationCuboid::Face::Left;
    if (name == QStringLiteral("right")) return annotationCuboid::Face::Right;
    if (name == QStringLiteral("back")) return annotationCuboid::Face::Back;
    if (name == QStringLiteral("front")) return annotationCuboid::Face::Front;
    return annotationCuboid::Face::None;
}

bool upstreamReference(const QString& filename) {
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) return false;
    QJsonParseError error;
    const auto document = QJsonDocument::fromJson(file.readAll(), &error);
    const auto cases =
            document.object().value(QStringLiteral("cases")).toArray();
    if (error.error != QJsonParseError::NoError || cases.isEmpty())
        return false;
    int steps = 0;
    for (const QJsonValue& value : cases) {
        const auto scenario = value.toObject();
        auto points = pointsFromJson(
                scenario.value(QStringLiteral("points")).toArray());
        if (points.size() != 8) return false;
        const double minDepth =
                scenario.value(QStringLiteral("min_depth")).toDouble();
        for (const auto& operation :
             scenario.value(QStringLiteral("steps")).toArray()) {
            const auto step = operation.toObject();
            const auto pair = step.value(QStringLiteral("position")).toArray();
            if (pair.size() != 2) return false;
            const QPointF position(pair[0].toDouble(), pair[1].toDouble());
            if (step.contains(QStringLiteral("control")))
                annotationCuboid::moveControl(
                        points, step.value(QStringLiteral("control")).toInt(),
                        position, minDepth);
            else
                annotationCuboid::moveFace(
                        points,
                        faceFromName(
                                step.value(QStringLiteral("face")).toString()),
                        position, minDepth);
            if (!samePoints(
                        points,
                        pointsFromJson(step.value(QStringLiteral("expected"))
                                               .toArray()))) {
                std::fprintf(stderr, "Upstream cuboid mismatch at step %d\n",
                             steps);
                return false;
            }
            const auto controls =
                    step.value(QStringLiteral("controls")).toArray();
            const auto visible = annotationCuboid::visibleControls(points);
            if (visible.size() != controls.size()) return false;
            for (int i = 0; i < visible.size(); ++i)
                if (visible[i] != controls[i].toInt()) return false;
            ++steps;
        }
    }
    std::printf("Upstream cuboid reference: %d cases, %d steps PASS\n",
                cases.size(), steps);
    return true;
}

void mouse(AnnotationCanvas& canvas,
           QEvent::Type type,
           const QPointF& imagePoint,
           double scale = 1.0) {
    const QPointF origin((canvas.width() - 400 * scale) / 2.0,
                         (canvas.height() - 400 * scale) / 2.0);
    QMouseEvent event(
            type, origin + imagePoint * scale, Qt::LeftButton,
            type == QEvent::MouseButtonRelease ? Qt::NoButton : Qt::LeftButton,
            Qt::NoModifier);
    QCoreApplication::sendEvent(&canvas, &event);
}

void drag(AnnotationCanvas& canvas,
          const QPointF& from,
          const QPointF& to,
          double scale = 1.0) {
    mouse(canvas, QEvent::MouseButtonPress, from, scale);
    mouse(canvas, QEvent::MouseMove, to, scale);
    mouse(canvas, QEvent::MouseButtonRelease, to, scale);
}

bool interactions() {
    AnnotationCanvas canvas;
    canvas.resize(424, 424);
    QImage image(400, 400, QImage::Format_RGB32);
    image.fill(Qt::black);
    canvas.setImage(image);
    canvas.setHitEpsilon(5);
    canvas.setCrosshair(false, 2, Qt::green, 0.5);
    canvas.setShowLabels(false);
    canvas.show();
    QCoreApplication::processEvents();
    AnnotationShape cuboid;
    cuboid.type = QStringLiteral("cuboid");
    cuboid.label = QStringLiteral("car");
    cuboid.points = {QPointF(80, 100),  QPointF(200, 100), QPointF(200, 240),
                     QPointF(80, 240),  QPointF(120, 60),  QPointF(240, 60),
                     QPointF(240, 200), QPointF(120, 200)};
    cuboid.extra.insert(
            QStringLiteral("cuboid3d"),
            QJsonObject{{QStringLiteral("source"), QStringLiteral("fixture")},
                        {QStringLiteral("custom"), 42}});
    int started = 0, completed = 0;
    QVector<AnnotationShape> undo;
    QObject::connect(&canvas, &AnnotationCanvas::editStarted, [&] {
        ++started;
        undo = canvas.shapes();
    });
    QObject::connect(&canvas, &AnnotationCanvas::shapesChanged,
                     [&] { ++completed; });
    const auto reset = [&] {
        canvas.setShapes({cuboid});
        started = completed = 0;
    };
    reset();
    // Drag a front corner: the opposite corner and all four depth offsets stay
    // fixed.
    drag(canvas, QPointF(80, 100), QPointF(60, 80));
    const auto resized = canvas.shapes().first();
    if (!check(resized.points[0] == QPointF(60, 80) &&
                       resized.points[1] == QPointF(200, 80) &&
                       resized.points[2] == QPointF(200, 240) &&
                       resized.points[3] == QPointF(60, 240),
               "Front corner must resize the complete front face"))
        return false;
    for (int i = 0; i < 4; ++i)
        if (!check(resized.points[i + 4] - resized.points[i] ==
                           QPointF(40, -40),
                   "Corner drag changed depth offsets"))
            return false;
    if (!check(started == 1 && completed == 1 &&
                       undo.first().points == cuboid.points,
               "Cuboid drag must snapshot before editing and commit once"))
        return false;
    canvas.setShapes(undo);
    if (!check(canvas.shapes().first().points == cuboid.points,
               "Cuboid undo restore failed"))
        return false;

    reset();
    drag(canvas, QPointF(80, 170), QPointF(199, 190));
    if (!check(canvas.shapes().first().points[0] == QPointF(198, 100) &&
                       canvas.shapes().first().points[3] == QPointF(198, 240),
               "Edge control must constrain width and ignore tangential "
               "movement"))
        return false;

    reset();
    drag(canvas, QPointF(240, 130), QPointF(260, 190));
    const auto deepened = canvas.shapes().first();
    if (!check(deepened.points[4] == QPointF(140, 40) &&
                       deepened.points[0] == cuboid.points[0],
               "Rear center must change depth along the front-to-back ray"))
        return false;
    const auto metadata =
            deepened.toJson().value(QStringLiteral("cuboid3d")).toObject();
    if (!check(metadata.value(QStringLiteral("depth_vector")).toArray() ==
                               QJsonArray{60, -60} &&
                       metadata.value(QStringLiteral("custom")).toInt() == 42 &&
                       metadata.value(QStringLiteral("source")).toString() ==
                               QStringLiteral("fixture") &&
                       samePoints(AnnotationShape::fromJson(deepened.toJson())
                                          .points,
                                  deepened.points),
               "Depth edit must round-trip source and unknown cuboid metadata"))
        return false;

    reset();
    drag(canvas, QPointF(220, 140), QPointF(230, 150));
    if (!check(canvas.shapes().first().points[1] == QPointF(210, 110) &&
                       canvas.shapes().first().points[5] == QPointF(250, 70) &&
                       canvas.shapes().first().points[0] == cuboid.points[0],
               "Side face must move only its four vertices"))
        return false;

    reset();
    drag(canvas, QPointF(170, 80), QPointF(180, 85));
    if (!check(canvas.shapes().first().points[4] == QPointF(130, 65) &&
                       canvas.shapes().first().points[0] == cuboid.points[0],
               "Back face drag must preserve the front face"))
        return false;
    reset();
    drag(canvas, QPointF(145, 170), QPointF(155, 180));
    for (int i = 0; i < 8; ++i)
        if (!check(canvas.shapes().first().points[i] ==
                           cuboid.points[i] + QPointF(10, 10),
                   "Front face must translate the entire cuboid"))
            return false;

    reset();
    canvas.selectShape(0);
    canvas.addPointToEdge();
    canvas.removeSelectedPoint();
    if (!check(canvas.shapes().first().points == cuboid.points &&
                       completed == 0,
               "Generic vertex insertion/removal must not corrupt cuboids"))
        return false;
    cuboid.locked = true;
    reset();
    drag(canvas, QPointF(80, 100), QPointF(60, 80));
    if (!check(canvas.shapes().first().points == cuboid.points &&
                       started == 0 && completed == 0,
               "Locked cuboids must remain unchanged"))
        return false;
    cuboid.locked = false;

    reset();
    drag(canvas, QPointF(80, 100), QPointF(80, 100));
    if (!check(started == 0 && completed == 0,
               "Click without movement must not commit a cuboid edit"))
        return false;

    // Mirror the depth to verify the left visible-rear and face hit rules.
    const auto rightCuboid = cuboid;
    for (int i = 4; i < 8; ++i) cuboid.points[i].rx() -= 80;
    reset();
    drag(canvas, QPointF(40, 130), QPointF(20, 180));
    if (!check(canvas.shapes().first().points[4] == QPointF(20, 40) &&
                       canvas.shapes().first().points[0] == cuboid.points[0],
               "Left rear center must follow the same depth constraint"))
        return false;
    reset();
    drag(canvas, QPointF(60, 140), QPointF(50, 150));
    if (!check(canvas.shapes().first().points[0] == QPointF(70, 110) &&
                       canvas.shapes().first().points[1] == cuboid.points[1],
               "Left face drag must edit only its four vertices"))
        return false;
    cuboid = rightCuboid;

    reset();
    drag(canvas, QPointF(240, 60), QPointF(300, 230));
    if (!check(canvas.shapes().first().points[0].y() == 238 &&
                       canvas.shapes().first().points[4].y() == 198,
               "Rear endpoint must adjust both faces without crossing bottom "
               "edges"))
        return false;

    reset();
    // Control editing remains valid outside image bounds and at another zoom.
    canvas.setZoomFactor(2);
    drag(canvas, QPointF(80, 100), QPointF(-20, 80), 2);
    if (!check(canvas.shapes().first().points[0] == QPointF(-20, 80),
               "Cuboid control edits must use unclipped image coordinates at "
               "any zoom"))
        return false;

    reset();
    drag(canvas, QPointF(145, 170), QPointF(-55, 170), 2);
    if (!check(canvas.shapes().first().points[0] == QPointF(-120, 100),
               "Whole cuboids must remain movable beyond the image boundary"))
        return false;

    canvas.fitImage();
    reset();
    // Changing images during a drag must clear stale control/face state.
    mouse(canvas, QEvent::MouseButtonPress, QPointF(80, 100));
    canvas.setImage(image);
    canvas.setShapes({cuboid});
    mouse(canvas, QEvent::MouseMove, QPointF(60, 80));
    mouse(canvas, QEvent::MouseButtonRelease, QPointF(60, 80));
    if (!check(canvas.shapes().first().points == cuboid.points,
               "Image switch retained a stale cuboid drag"))
        return false;

    // The old eight-point polygon left an unintended diagonal across the front.
    const QImage visualization = canvas.renderVisualizationImage({cuboid});
    const auto diagonal = visualization.pixelColor(100, 150);
    const auto interior = visualization.pixelColor(110, 150);
    if (!check(diagonal == interior,
               "Cuboid rendering must not connect unrelated front/back "
               "corners"))
        return false;
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (argc == 3 && QString::fromLocal8Bit(argv[1]) ==
                             QStringLiteral("--upstream-reference"))
        return upstreamReference(QString::fromLocal8Bit(argv[2])) ? 0 : 1;
    return interactions() ? 0 : 1;
}
