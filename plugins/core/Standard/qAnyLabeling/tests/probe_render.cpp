// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <cstdio>

#include "AnnotationCanvas.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (argc != 3) return 2;
    QFile input(QString::fromLocal8Bit(argv[1]));
    if (!input.open(QIODevice::ReadOnly)) return 1;
    const QJsonObject request =
            QJsonDocument::fromJson(input.readAll()).object();
    const QDir output(QString::fromLocal8Bit(argv[2]));
    if (!output.mkpath(QStringLiteral("."))) return 1;
    QJsonArray palette;
    for (int i = -256; i < 512; ++i) {
        const QColor color = annotationLabelColor(i);
        palette.append(QJsonArray{color.red(), color.green(), color.blue()});
    }
    for (const QJsonValue& value :
         request.value(QStringLiteral("cases")).toArray()) {
        const QJsonObject item = value.toObject();
        AnnotationCanvas canvas;
        QImage image(640, 480, QImage::Format_RGB32);
        image.fill(Qt::white);
        canvas.setImage(image);
        canvas.setShowLabels(
                item.value(QStringLiteral("show_labels")).toBool(true));
        canvas.setShowTexts(
                item.value(QStringLiteral("show_texts")).toBool(true));
        canvas.setShowScores(
                item.value(QStringLiteral("show_scores")).toBool(true));
        canvas.setShowMasks(
                item.value(QStringLiteral("show_masks")).toBool(true));
        QVector<AnnotationShape> shapes;
        for (const QJsonValue& shape :
             item.value(QStringLiteral("shapes")).toArray())
            shapes.append(AnnotationShape::fromJson(shape.toObject()));
        canvas.setShapes(shapes);
        const QImage result = canvas.renderVisualizationImage(shapes);
        if (!result.save(output.filePath(
                    item.value(QStringLiteral("name")).toString() +
                    QStringLiteral(".png"))))
            return 1;
    }
    QFile result(output.filePath(QStringLiteral("palette.json")));
    if (!result.open(QIODevice::WriteOnly)) return 1;
    result.write(QJsonDocument(palette).toJson());
    return 0;
}
