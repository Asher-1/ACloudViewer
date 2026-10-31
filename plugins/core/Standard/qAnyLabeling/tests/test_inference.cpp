// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QImageReader>
#include <QJsonDocument>
#include <QSet>
#include <QThread>
#include <QTimer>
#include <cstdio>

#include "AnnotationWorker.h"
#include "aicore/backend_capi.h"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    qRegisterMetaType<AnnotationInference>("AnnotationInference");
    if (argc < 3 || argc > 7) {
        std::fprintf(stderr,
                     "usage: test_qanylabeling_inference MODEL IMAGE [DEVICE] "
                     "[TASK] [PROMPT_OR_TEXT_MODEL] "
                     "[LANDMARK_OR_OBJECT_TAGGER_MODEL]\n");
        return 2;
    }
    const QString device =
            argc >= 4 ? QString::fromLocal8Bit(argv[3]) : QStringLiteral("cpu");
    const QString task = argc >= 5 ? QString::fromLocal8Bit(argv[4])
                                   : QStringLiteral("detect");
    const QString promptMode = argc >= 6 ? QString::fromLocal8Bit(argv[5])
                                         : QStringLiteral("point");
    const QString landmarkModel =
            argc == 7 ? QString::fromLocal8Bit(argv[6]) : QString();
    const QByteArray deviceBytes = device.toUtf8();
    if (aicore_warmup_backend(deviceBytes.constData()) != 0) {
        std::fprintf(stderr, "%s backend unavailable: %s\n",
                     deviceBytes.constData(), aicore_backend_last_error());
        return 2;
    }
    QImageReader reader(QString::fromLocal8Bit(argv[2]));
    const QImage image = reader.read();
    if (image.isNull()) {
        std::fprintf(stderr, "Image unavailable\n");
        return 2;
    }
    QThread thread;
    auto* worker = new AnnotationWorker;
    worker->moveToThread(&thread);
    QObject::connect(&thread, &QThread::finished, worker,
                     &QObject::deleteLater);
    int forwards = 0;
    int exitCode = 0;
    const QString model = QString::fromLocal8Bit(argv[1]);
    const QString path = QString::fromLocal8Bit(argv[2]);
    AnnotationPrompt prompt;
    if (task == QStringLiteral("ocr") &&
        promptMode.startsWith(QStringLiteral("layout=")))
        prompt.ocrLayoutModelPath = promptMode.mid(7);
    if (task == QStringLiteral("ocr") &&
        promptMode.startsWith(QStringLiteral("cls=")))
        prompt.ocrClassifierModelPath = promptMode.mid(4);
    const bool selectedOcr =
            task == QStringLiteral("ocr") &&
            promptMode.startsWith(QStringLiteral("cls-regions="));
    if (selectedOcr) {
        prompt.ocrClassifierModelPath = promptMode.mid(12);
        prompt.ocrQuads = {{{-8.75, -4.25},
                            {175.75, -2.25},
                            {180.5, 65.75},
                            {-5.5, 62.25}},
                           {{20, 20}, {140, 25}, {145, 70}, {80, 72}, {15, 60}},
                           {{30, 90}, {210, 100}, {35, 145}}};
        for (const auto& points : prompt.ocrQuads)
            prompt.rois.append(QPolygonF(points).boundingRect());
        prompt.roiSourceIndices = {17, 3, 42};
    }
    if (task == QStringLiteral("world") && !landmarkModel.isEmpty())
        prompt.objectTaggingModelPath = landmarkModel;
    QJsonArray firstTaggedObjects;
    QJsonObject firstAttributes;
    const bool nmsCheck = task == QStringLiteral("detect") &&
                          promptMode == QStringLiteral("nms");
    int firstNmsCount = -1;
    if (nmsCheck) prompt.iouThreshold = 0.0;
    prompt.label = QStringLiteral("object");
    if (task == QStringLiteral("sam3-pvs")) {
        if (promptMode == QStringLiteral("boxes")) {
            prompt.rois = {QRectF(image.width() * 0.05, image.height() * 0.05,
                                  image.width() * 0.35, image.height() * 0.35),
                           QRectF(image.width() * 0.55, image.height() * 0.25,
                                  image.width() * 0.35, image.height() * 0.55)};
            prompt.roiSourceIndices = {3, 7};
            prompt.roiLabels = {QStringLiteral("left"),
                                QStringLiteral("right")};
        } else if (promptMode != QStringLiteral("box"))
            prompt.positive.append(
                    QPointF(image.width() / 2.0, image.height() / 2.0));
        if (promptMode == QStringLiteral("mixed"))
            prompt.negative.append(QPointF(5, 5));
        if (promptMode == QStringLiteral("box") ||
            promptMode == QStringLiteral("mixed")) {
            prompt.useBox = true;
            prompt.box = QRectF(image.width() * 0.25, image.height() * 0.25,
                                image.width() * 0.5, image.height() * 0.5);
        }
    } else if (task == QStringLiteral("sam3-pcs")) {
        prompt.text = QStringLiteral("bus");
    } else if (task == QStringLiteral("gkd")) {
        if (promptMode != QStringLiteral("visual"))
            prompt.text = QStringLiteral("nose, left eye");
        if (promptMode == QStringLiteral("visual") ||
            promptMode == QStringLiteral("multimodal")) {
            prompt.visualSupportImage = image;
            prompt.visualSupportPoints = {
                    QPointF(image.width() * 0.4, image.height() * 0.4),
                    QPointF(image.width() * 0.6, image.height() * 0.4)};
        }
        if (promptMode == QStringLiteral("multi")) {
            prompt.rois = {QRectF(0, 0, image.width(), image.height()),
                           QRectF(image.width() * 0.25, image.height() * 0.25,
                                  image.width() * 0.5, image.height() * 0.5)};
            prompt.roiGroupIds = {17, 23};
        }
    } else if (task == QStringLiteral("world") ||
               task == QStringLiteral("yoloe")) {
        if (task == QStringLiteral("yoloe") &&
            promptMode == QStringLiteral("visual")) {
            prompt.visualPrompts.append(
                    QRectF(image.width() * 0.25, image.height() * 0.25,
                           image.width() * 0.5, image.height() * 0.5));
        } else if (promptMode != QStringLiteral("pf")) {
            prompt.text = QStringLiteral("bus, person");
            prompt.textModelPath = promptMode;
        }
    }
    if (task == QStringLiteral("relate")) {
        // Person and horse boxes YOLO26s q8_0 detects on the 1280x907
        // horse.jpg sample, as fractions of the image so other sizes scale.
        prompt.rois = {
                QRectF(QPointF(image.width() * 0.3713, image.height() * 0.1503),
                       QPointF(image.width() * 0.5030,
                               image.height() * 0.6903)),
                QRectF(QPointF(image.width() * 0.1026, image.height() * 0.3481),
                       QPointF(image.width() * 0.7742,
                               image.height() * 0.9560))};
        prompt.roiSourceIndices = {2, 5};
        prompt.roiLabels = {QStringLiteral("person"), QStringLiteral("horse")};
    }
    if (task == QStringLiteral("detect") &&
        promptMode == QStringLiteral("sahi")) {
        prompt.slicedDetection = true;
        prompt.sliceWidth = 512;
        prompt.sliceHeight = 512;
        prompt.sliceOverlap = 0.2;
        prompt.sliceMergeThreshold = 0.45;
    }
    QObject::connect(
            worker, &AnnotationWorker::finished, &app,
            [&](const AnnotationInference& result) {
                const bool validDepth =
                        task == QStringLiteral("depth") &&
                        !result.depthPreview.isNull() &&
                        result.depthPreview.size() == image.size();
                if (!result.error.isEmpty() ||
                    (result.shapes.isEmpty() && !result.hasFlags &&
                     !result.hasTags && result.relations.isEmpty() &&
                     !validDepth) ||
                    result.imagePath != path || result.modelPath != model ||
                    result.task != task ||
                    result.generation != quint64(forwards + 1) ||
                    result.promptRevision != 1) {
                    std::fprintf(stderr,
                                 "Inference failed: %s; detections=%d "
                                 "relations=%d summary=%s\n",
                                 result.error.toLocal8Bit().constData(),
                                 result.shapes.size(),
                                 int(result.relations.size()),
                                 result.summary.toLocal8Bit().constData());
                    exitCode = 1;
                    app.quit();
                    return;
                }
                if (device != QStringLiteral("cpu") &&
                    !result.device.contains(device, Qt::CaseInsensitive)) {
                    std::fprintf(stderr,
                                 "Backend fallback: requested=%s resolved=%s\n",
                                 deviceBytes.constData(),
                                 result.device.toLocal8Bit().constData());
                    exitCode = 1;
                    app.quit();
                    return;
                }
                for (const AnnotationShape& shape : result.shapes) {
                    if (shape.points.isEmpty() || shape.label.isEmpty() ||
                        ((task == QStringLiteral("detect") ||
                          task == QStringLiteral("rfdetr-detect")) &&
                         (shape.points.size() != 4 || shape.points[0].x() < 0 ||
                          shape.points[0].y() < 0 ||
                          shape.points[2].x() > image.width() ||
                          shape.points[2].y() > image.height())) ||
                        (task == QStringLiteral("obb") &&
                         shape.type != QStringLiteral("rotation")) ||
                        (task == QStringLiteral("gkd") &&
                         (shape.type != QStringLiteral("point") ||
                          shape.points.size() != 1 || shape.points[0].x() < 0 ||
                          shape.points[0].y() < 0 ||
                          shape.points[0].x() >= image.width() ||
                          shape.points[0].y() >= image.height())) ||
                        (task == QStringLiteral("aliked") &&
                         (shape.type != QStringLiteral("point") ||
                          shape.points.size() != 1 || shape.points[0].x() < 0 ||
                          shape.points[0].y() < 0 ||
                          shape.points[0].x() >= image.width() ||
                          shape.points[0].y() >= image.height() ||
                          !shape.extra.contains(
                                  QStringLiteral("feature_index")))) ||
                        (task == QStringLiteral("deeplsd") &&
                         (shape.type != QStringLiteral("linestrip") ||
                          shape.points.size() != 2 ||
                          !shape.extra.contains(QStringLiteral("score")) ||
                          shape.points[0].x() < 0 || shape.points[0].y() < 0 ||
                          shape.points[1].x() < 0 || shape.points[1].y() < 0 ||
                          shape.points[0].x() >= image.width() ||
                          shape.points[0].y() >= image.height() ||
                          shape.points[1].x() >= image.width() ||
                          shape.points[1].y() >= image.height())) ||
                        ((task == QStringLiteral("segment") ||
                          task == QStringLiteral("semantic") ||
                          task == QStringLiteral("yoloe") ||
                          task == QStringLiteral("rmbg") ||
                          task.startsWith(QStringLiteral("sam3-"))) &&
                         (shape.type != QStringLiteral("polygon") ||
                          shape.points.size() < 3))) {
                        std::fprintf(stderr,
                                     "Invalid typed detection geometry\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                    if (task == QStringLiteral("gkd") &&
                        promptMode == QStringLiteral("multi") &&
                        !prompt.roiGroupIds.contains(
                                shape.extra.value(QStringLiteral("group_id"))
                                        .toInt(-1))) {
                        std::fprintf(stderr,
                                     "GKD multi-ROI keypoint lost its group\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                    if (task == QStringLiteral("semantic") &&
                        (shape.mask.size() != image.size() ||
                         shape.extra.value(QStringLiteral("class_id"))
                                         .toInt(-1) < 0)) {
                        std::fprintf(stderr,
                                     "Semantic mask or class ID is invalid\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                }
                if (task == QStringLiteral("rfdetr-segment")) {
                    bool hasPolygon = false;
                    for (const AnnotationShape& shape : result.shapes) {
                        if (shape.type != QStringLiteral("polygon")) continue;
                        hasPolygon = true;
                        if (shape.points.size() < 3) {
                            std::fprintf(
                                    stderr,
                                    "RF-DETR polygon has too few vertices\n");
                            exitCode = 1;
                            app.quit();
                            return;
                        }
                        for (const QPointF& point : shape.points) {
                            if (point.x() < 0 || point.y() < 0 ||
                                point.x() > image.width() ||
                                point.y() > image.height()) {
                                std::fprintf(stderr,
                                             "RF-DETR polygon is outside the "
                                             "image\n");
                                exitCode = 1;
                                app.quit();
                                return;
                            }
                        }
                    }
                    if (!hasPolygon) {
                        std::fprintf(
                                stderr,
                                "RF-DETR segmentation produced no polygon\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                }
                if (task == QStringLiteral("pose")) {
                    QSet<int> groups;
                    for (const AnnotationShape& shape : result.shapes)
                        if (shape.type == QStringLiteral("rectangle"))
                            groups.insert(
                                    shape.extra
                                            .value(QStringLiteral("group_id"))
                                            .toInt(-1));
                    for (const AnnotationShape& shape : result.shapes) {
                        if (shape.type == QStringLiteral("point") &&
                            !groups.contains(
                                    shape.extra
                                            .value(QStringLiteral("group_id"))
                                            .toInt(-1))) {
                            std::fprintf(stderr,
                                         "Pose keypoint has no matching box "
                                         "group\n");
                            exitCode = 1;
                            app.quit();
                            return;
                        }
                    }
                }
                if (task.startsWith(QStringLiteral("facedetect"))) {
                    QSet<int> groups;
                    int points = 0;
                    for (const AnnotationShape& shape : result.shapes)
                        if (shape.type == QStringLiteral("rectangle"))
                            groups.insert(
                                    shape.extra
                                            .value(QStringLiteral("group_id"))
                                            .toInt(-1));
                    for (const AnnotationShape& shape : result.shapes) {
                        if (shape.type != QStringLiteral("point")) continue;
                        ++points;
                        if (!groups.contains(
                                    shape.extra
                                            .value(QStringLiteral("group_id"))
                                            .toInt(-1))) {
                            std::fprintf(stderr,
                                         "Face landmark has no matching box "
                                         "group\n");
                            exitCode = 1;
                            app.quit();
                            return;
                        }
                    }
                    const int expectedPoints =
                            task == QStringLiteral("facedetect-dense")
                                    ? 5 + 106 + 68
                                    : 5;
                    if (groups.isEmpty() ||
                        points != groups.size() * expectedPoints) {
                        std::fprintf(
                                stderr,
                                "Face detection did not return the expected "
                                "landmarks per face\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                }
                if (task == QStringLiteral("attribute")) {
                    const auto shape = result.shapes.value(0);
                    const auto attributes =
                            shape.extra.value(QStringLiteral("attributes"))
                                    .toObject();
                    const bool person = shape.label == QStringLiteral("person");
                    const bool expected =
                            person ? attributes.size() == 21 &&
                                             attributes.value(QStringLiteral(
                                                     "Gender")) ==
                                                     QStringLiteral("Male") &&
                                             attributes.value(QStringLiteral(
                                                     "Direction")) ==
                                                     QStringLiteral("Back")
                                   : shape.label == QStringLiteral("vehicle") &&
                                             attributes.size() == 2 &&
                                             attributes.value(
                                                     QStringLiteral("Color")) ==
                                                     QStringLiteral("white") &&
                                             attributes.value(
                                                     QStringLiteral("Type")) ==
                                                     QStringLiteral("truck");
                    if (result.shapes.size() != 1 ||
                        shape.type != QStringLiteral("rectangle") ||
                        shape.points !=
                                QVector<QPointF>{{0, 0},
                                                 {double(image.width()), 0},
                                                 {double(image.width()),
                                                  double(image.height())},
                                                 {0, double(image.height())}} ||
                        !result.replaceShapes || result.hasFlags ||
                        result.hasTags || !expected ||
                        (forwards && attributes != firstAttributes)) {
                        std::fprintf(
                                stderr,
                                "Multi-attribute worker failed its full-image "
                                "geometry, attribute or replace contract\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                    firstAttributes = attributes;
                }
                if (task == QStringLiteral("classify") &&
                    (!result.hasFlags || result.flags.isEmpty() ||
                     result.summary.isEmpty())) {
                    std::fprintf(stderr, "Classification flags missing\n");
                    exitCode = 1;
                    app.quit();
                    return;
                }
                if (task == QStringLiteral("ocr")) {
                    if (selectedOcr &&
                        (result.sourceIndices != prompt.roiSourceIndices ||
                         result.shapes.size() != prompt.rois.size() ||
                         result.sourceRois != prompt.rois)) {
                        std::fprintf(stderr,
                                     "Selected OCR region ownership lost\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                    // Upstream PP-OCR semantics: every transcription is a
                    // "text" label (rectangle or quadrilateral) whose
                    // description carries the transcription, with a score
                    // and the rec language tag in extra, gated at the
                    // drop_score 0.5. Layout shapes (the optional
                    // PP-Structure stage) are validated separately.
                    int layoutShapeCount = 0;
                    for (const AnnotationShape& shape : result.shapes) {
                        const QString description =
                                shape.extra.value(QStringLiteral("description"))
                                        .toString();
                        if (shape.extra.contains(
                                    QStringLiteral("reading_order"))) {
                            // Layout rectangles publish the actual class name.
                            const bool layoutOk =
                                    shape.type == QStringLiteral("rectangle") &&
                                    shape.points.size() == 4 &&
                                    shape.label != QStringLiteral("layout") &&
                                    shape.extra.contains(
                                            QStringLiteral("reading_order"));
                            if (!layoutOk) {
                                std::fprintf(stderr,
                                             "OCR layout region malformed\n");
                                exitCode = 1;
                                app.quit();
                                return;
                            }
                            ++layoutShapeCount;
                            continue;
                        }
                        const bool ok =
                                shape.label == QStringLiteral("text") &&
                                (shape.type == QStringLiteral("rectangle") ||
                                 shape.type ==
                                         QStringLiteral("quadrilateral")) &&
                                shape.points.size() == 4 &&
                                (selectedOcr || !description.isEmpty()) &&
                                shape.extra.contains(QStringLiteral("score")) &&
                                shape.extra.contains(
                                        QStringLiteral("language"));
                        if (!ok) {
                            std::fprintf(
                                    stderr,
                                    "OCR annotation lost its transcription "
                                    "metadata\n");
                            exitCode = 1;
                            app.quit();
                            return;
                        }
                    }
                    if (prompt.ocrLayoutModelPath.isEmpty() &&
                        layoutShapeCount) {
                        std::fprintf(stderr,
                                     "Cached layout model silently enabled a "
                                     "stage\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                }
                if (task == QStringLiteral("sam3-pvs") &&
                    promptMode == QStringLiteral("boxes")) {
                    if (result.sourceRois.size() != prompt.rois.size() ||
                        result.sourceIndices != prompt.roiSourceIndices ||
                        result.shapes.size() < prompt.rois.size()) {
                        std::fprintf(
                                stderr,
                                "SAM3 multi-box proposal ownership lost\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                    for (const AnnotationShape& shape : result.shapes) {
                        const QJsonValue source =
                                shape.extra.value(QStringLiteral("source_roi"));
                        if (!source.isDouble() || source.toInt(-1) < 0 ||
                            source.toInt() >= prompt.roiLabels.size() ||
                            shape.label !=
                                    prompt.roiLabels.at(source.toInt())) {
                            std::fprintf(stderr,
                                         "SAM3 multi-box mask lost its source "
                                         "class\n");
                            exitCode = 1;
                            app.quit();
                            return;
                        }
                    }
                }
                if (qEnvironmentVariableIsSet("QANYLABELING_PRINT_BOXES")) {
                    for (const AnnotationShape& shape : result.shapes) {
                        if (shape.type != QStringLiteral("rectangle") ||
                            shape.points.size() != 4)
                            continue;
                        const QRectF r = QPolygonF(shape.points).boundingRect();
                        std::fprintf(stdout, "box: %s %.1f %.1f %.1f %.1f\n",
                                     shape.label.toLocal8Bit().constData(),
                                     r.left(), r.top(), r.right(), r.bottom());
                    }
                }
                if (!prompt.objectTaggingModelPath.isEmpty()) {
                    QJsonArray objects;
                    for (const auto& shape : result.shapes) {
                        if (shape.type != QStringLiteral("rectangle") ||
                            shape.points.size() != 4 ||
                            !shape.extra.contains(
                                    QStringLiteral("description")) ||
                            shape.extra.value(QStringLiteral("description"))
                                    .toString()
                                    .isEmpty() ||
                            result.hasTags) {
                            std::fprintf(stderr,
                                         "Object descriptions were not mapped "
                                         "to detector boxes\n");
                            exitCode = 1;
                            app.quit();
                            return;
                        }
                        QJsonArray points;
                        for (const auto& p : shape.points) {
                            if (p.x() != int(p.x()) || p.y() != int(p.y())) {
                                exitCode = 1;
                                app.quit();
                                return;
                            }
                            points.append(QJsonArray{p.x(), p.y()});
                        }
                        objects.append(QJsonObject{
                                {"label", shape.label},
                                {"points", points},
                                {"description",
                                 shape.extra.value(
                                         QStringLiteral("description"))}});
                    }
                    if (!forwards)
                        firstTaggedObjects = objects;
                    else if (firstTaggedObjects != objects) {
                        std::fprintf(stderr,
                                     "Object descriptions changed on repeat\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                    std::fprintf(stdout, "tagged_objects: %s\n",
                                 QJsonDocument(objects)
                                         .toJson(QJsonDocument::Compact)
                                         .constData());
                }
                if (task.startsWith(QStringLiteral("ram"))) {
                    // Image-level tags only: never object shapes, no
                    // duplicates, none blank (upstream normalize_image_tags).
                    QSet<QString> seen;
                    bool ok = result.hasTags && !result.tags.isEmpty() &&
                              result.shapes.isEmpty();
                    if (task == QStringLiteral("ram-caption"))
                        ok = ok && result.hasDescription &&
                             !result.description.trimmed().isEmpty();
                    for (const QString& tag : result.tags)
                        ok = ok && !tag.trimmed().isEmpty() &&
                             tag == tag.trimmed() && !seen.contains(tag);
                    for (const QString& tag : result.tags) seen.insert(tag);
                    if (!ok) {
                        std::fprintf(stderr,
                                     "RAM result is not clean image tags\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                    std::fprintf(stdout, "tags: %s\n",
                                 result.tags.join(QStringLiteral(", "))
                                         .toLocal8Bit()
                                         .constData());
                }
                if (task == QStringLiteral("relate")) {
                    for (const QJsonValue& value : result.relations) {
                        const QJsonObject e = value.toObject();
                        std::fprintf(
                                stdout, "relation: %s -[%s %.3f %s]-> %s\n",
                                e.value(QStringLiteral("subject"))
                                        .toString()
                                        .toLocal8Bit()
                                        .constData(),
                                e.value(QStringLiteral("predicate"))
                                        .toString()
                                        .toLocal8Bit()
                                        .constData(),
                                e.value(QStringLiteral("score")).toDouble(),
                                e.value(QStringLiteral("type"))
                                        .toString()
                                        .toLocal8Bit()
                                        .constData(),
                                e.value(QStringLiteral("object"))
                                        .toString()
                                        .toLocal8Bit()
                                        .constData());
                    }
                }
                if (task == QStringLiteral("relate")) {
                    // Edges reference the caller's canvas shape indices and
                    // never re-emit boxes.
                    bool ok = result.shapes.isEmpty() &&
                              !result.relations.isEmpty();
                    for (const QJsonValue& value : result.relations) {
                        const QJsonObject edge = value.toObject();
                        const int from =
                                edge.value(QStringLiteral("subject_index"))
                                        .toInt(-1);
                        const int to =
                                edge.value(QStringLiteral("object_index"))
                                        .toInt(-1);
                        const double score =
                                edge.value(QStringLiteral("score")).toDouble();
                        ok = ok && prompt.roiSourceIndices.contains(from) &&
                             prompt.roiSourceIndices.contains(to) &&
                             from != to &&
                             !edge.value(QStringLiteral("predicate"))
                                      .toString()
                                      .isEmpty() &&
                             score >= 0.5 && score <= 1.0;
                    }
                    if (!ok) {
                        std::fprintf(stderr, "Relation result malformed\n");
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                }
                if (nmsCheck) {
                    if (!forwards)
                        firstNmsCount = result.shapes.size();
                    else if (result.shapes.size() <= firstNmsCount) {
                        std::fprintf(stderr,
                                     "Live IoU change 0 -> 1 did not retain "
                                     "more overlapping boxes: %d -> %d\n",
                                     firstNmsCount, result.shapes.size());
                        exitCode = 1;
                        app.quit();
                        return;
                    }
                }
                ++forwards;
                std::fprintf(stdout,
                             "forward=%d task=%s annotations=%d flags=%d "
                             "device=%s\n",
                             forwards, task.toLocal8Bit().constData(),
                             result.shapes.size(), result.flags.size(),
                             result.device.toLocal8Bit().constData());
                if (forwards == 2)
                    app.quit();
                else
                    QMetaObject::invokeMethod(
                            worker,
                            [=]() {
                                AnnotationPrompt secondPrompt = prompt;
                                if (nmsCheck) secondPrompt.iouThreshold = 1.0;
                                if (!secondPrompt.text.isEmpty() &&
                                    (task == QStringLiteral("world") ||
                                     task == QStringLiteral("yoloe")))
                                    secondPrompt.text =
                                            QStringLiteral("person, bus");
                                worker->infer(
                                        model, landmarkModel, device, task,
                                        path, image, 2,
                                        task == QStringLiteral("yoloe") ? 0.02
                                        : task == QStringLiteral("gkd") ? 0.0
                                                                        : 0.25,
                                        1, secondPrompt);
                            },
                            Qt::QueuedConnection);
            });
    QTimer::singleShot(120000, &app, [&]() {
        std::fprintf(stderr, "Inference timeout\n");
        exitCode = 1;
        app.quit();
    });
    thread.start();
    const double confidence = task == QStringLiteral("yoloe") ? 0.02
                              : task == QStringLiteral("gkd") ? 0.0
                                                              : 0.25;
    QMetaObject::invokeMethod(
            worker,
            [=]() {
                worker->infer(model, landmarkModel, device, task, path, image,
                              1, confidence, 1, prompt);
            },
            Qt::QueuedConnection);
    app.exec();
    thread.quit();
    thread.wait();
    return exitCode;
}
