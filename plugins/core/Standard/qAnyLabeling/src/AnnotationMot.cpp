// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationMot.h"

#include <QFileInfo>
#include <QJsonArray>
#include <QRegularExpression>
#include <QSet>
#include <algorithm>
#include <cmath>

namespace {
int frameId(const QString& fileName) {
    const QString tail = QFileInfo(fileName).completeBaseName().section(
            QLatin1Char('_'), -1);
    const QRegularExpressionMatch match =
            QRegularExpression(QStringLiteral("\\d+")).match(tail);
    return match.hasMatch() ? match.captured().toInt() : 0;
}

QJsonObject emptyDocument(const MotImage& image) {
    return {{QStringLiteral("version"), QStringLiteral("4.0.6")},
            {QStringLiteral("flags"), QJsonObject()},
            {QStringLiteral("shapes"), QJsonArray()},
            {QStringLiteral("imagePath"), image.fileName},
            {QStringLiteral("imageData"), QJsonValue::Null},
            {QStringLiteral("imageWidth"), image.size.width()},
            {QStringLiteral("imageHeight"), image.size.height()}};
}
}  // namespace

MotImport importMot(const QByteArray& groundTruth,
                    const QStringList& classes,
                    const QVector<MotImage>& images) {
    MotImport result;
    QHash<int, MotImage> byFrame;
    for (const MotImage& image : images) {
        const int id = frameId(image.fileName);
        if (!image.size.isValid() || byFrame.contains(id)) {
            result.error =
                    QStringLiteral("Invalid or duplicate MOT frame ID %1")
                            .arg(id);
            return result;
        }
        byFrame.insert(id, image);
    }
    QHash<int, QJsonArray> byFrameShapes;
    const QStringList lines =
            QString::fromUtf8(groundTruth).split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i) {
        const QString line = lines[i].trimmed();
        if (line.isEmpty()) continue;
        // Default split behavior keeps empty parts (Qt 5.12 has no
        // Qt::KeepEmptyParts).
        const QStringList fields = line.split(QLatin1Char(','));
        if (fields.size() < 8) {
            result.error =
                    QStringLiteral("MOT line %1 needs at least eight fields")
                            .arg(i + 1);
            return result;
        }
        bool valid = false;
        const int id = fields[0].trimmed().toInt(&valid);
        if (!valid) {
            result.error = QStringLiteral("MOT line %1 has invalid frame ID")
                                   .arg(i + 1);
            return result;
        }
        if (!byFrame.contains(id)) continue;
        const int track = fields[1].trimmed().toInt(&valid);
        if (!valid) {
            result.error = QStringLiteral("MOT line %1 has invalid track ID")
                                   .arg(i + 1);
            return result;
        }
        int box[4];
        for (int j = 0; j < 4; ++j) {
            box[j] = fields[j + 2].trimmed().toInt(&valid);
            if (!valid) {
                result.error =
                        QStringLiteral(
                                "MOT line %1 has invalid box coordinates")
                                .arg(i + 1);
                return result;
            }
        }
        const int category = fields[7].trimmed().toInt(&valid);
        if (!valid || category < 0 || category >= classes.size() ||
            box[2] <= 0 || box[3] <= 0) {
            result.error =
                    QStringLiteral("MOT line %1 has invalid class or box size")
                            .arg(i + 1);
            return result;
        }
        QJsonArray shapes = byFrameShapes.value(id);
        shapes.append(QJsonObject{
                {QStringLiteral("label"), classes[category]},
                {QStringLiteral("description"), QJsonValue::Null},
                {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
                {QStringLiteral("points"),
                 QJsonArray{QJsonArray{box[0], box[1]},
                            QJsonArray{box[0] + box[2], box[1]},
                            QJsonArray{box[0] + box[2], box[1] + box[3]},
                            QJsonArray{box[0], box[1] + box[3]}}},
                {QStringLiteral("group_id"), track},
                {QStringLiteral("difficult"), false},
                {QStringLiteral("direction"), 0},
                {QStringLiteral("flags"), QJsonObject()}});
        byFrameShapes.insert(id, shapes);
        ++result.count;
    }
    for (auto it = byFrameShapes.constBegin(); it != byFrameShapes.constEnd();
         ++it) {
        QJsonObject document = emptyDocument(byFrame.value(it.key()));
        document.insert(QStringLiteral("shapes"), it.value());
        result.documents.insert(byFrame.value(it.key()).fileName, document);
    }
    return result;
}

MotConversion convertMot(const QVector<MotImage>& images,
                         const QStringList& classes,
                         const QString& outputFolder) {
    MotConversion result;
    if (images.isEmpty() || classes.isEmpty()) {
        result.error = QStringLiteral("MOT export needs images and classes");
        return result;
    }
    QSet<int> frames;
    QSize commonSize;
    QString extension;
    for (const MotImage& image : images) {
        const int id = frameId(image.fileName);
        if (frames.contains(id) || !image.size.isValid()) {
            result.error =
                    QStringLiteral("Invalid or duplicate MOT frame ID %1")
                            .arg(id);
            return result;
        }
        frames.insert(id);
        const QString suffix =
                QLatin1Char('.') + QFileInfo(image.fileName).suffix();
        if (commonSize.isValid() &&
            (commonSize != image.size || extension != suffix)) {
            result.error = QStringLiteral(
                    "MOT sequence images must have one size and extension");
            return result;
        }
        commonSize = image.size;
        extension = suffix;
        for (const QJsonValue& value :
             image.document.value(QStringLiteral("shapes")).toArray()) {
            const QJsonObject shape = value.toObject();
            if (shape.value(QStringLiteral("shape_type")).toString() !=
                QStringLiteral("rectangle"))
                continue;
            const int category = classes.indexOf(
                    shape.value(QStringLiteral("label")).toString());
            if (category < 0) continue;
            const QJsonArray points =
                    shape.value(QStringLiteral("points")).toArray();
            if (points.size() != 2 && points.size() != 4) {
                result.error = QStringLiteral("Invalid MOT rectangle in %1")
                                       .arg(image.fileName);
                return result;
            }
            const QJsonArray first = points[0].toArray();
            const QJsonArray opposite =
                    points[points.size() == 2 ? 1 : 2].toArray();
            if (first.size() != 2 || opposite.size() != 2 ||
                !first[0].isDouble() || !first[1].isDouble() ||
                !opposite[0].isDouble() || !opposite[1].isDouble() ||
                !std::isfinite(first[0].toDouble()) ||
                !std::isfinite(first[1].toDouble()) ||
                !std::isfinite(opposite[0].toDouble()) ||
                !std::isfinite(opposite[1].toDouble())) {
                result.error = QStringLiteral("Invalid MOT rectangle in %1")
                                       .arg(image.fileName);
                return result;
            }
            const int x0 = int(std::clamp(first[0].toDouble(), 0.0,
                                          double(image.size.width())));
            const int y0 = int(std::clamp(first[1].toDouble(), 0.0,
                                          double(image.size.height())));
            const int x1 = int(std::clamp(opposite[0].toDouble(), 0.0,
                                          double(image.size.width())));
            const int y1 = int(std::clamp(opposite[1].toDouble(), 0.0,
                                          double(image.size.height())));
            if (x1 <= x0 || y1 <= y0) {
                result.error = QStringLiteral("Invalid MOT box size in %1")
                                       .arg(image.fileName);
                return result;
            }
            const int group = shape.value(QStringLiteral("group_id")).toInt(-1);
            const int track = group == 0 ? -1 : group;
            result.detections +=
                    QStringLiteral("%1,-1,%2,%3,%4,%5,1,-1,-1,-1\n")
                            .arg(id)
                            .arg(x0)
                            .arg(y0)
                            .arg(x1 - x0)
                            .arg(y1 - y0)
                            .toUtf8();
            result.groundTruth +=
                    QStringLiteral("%1,%2,%3,%4,%5,%6,%7,%8,1\n")
                            .arg(id)
                            .arg(track)
                            .arg(x0)
                            .arg(y0)
                            .arg(x1 - x0)
                            .arg(y1 - y0)
                            .arg(shape.value(QStringLiteral("difficult"))
                                                 .toBool()
                                         ? 0
                                         : 1)
                            .arg(category)
                            .toUtf8();
            ++result.count;
        }
    }
    result.sequence = QStringLiteral(
                              "[Sequence]\nname = MOT\nimdir = %1\nframerate = "
                              "30\nseqlength = %2\n"
                              "imwidth = %3\nimheight = %4\nimext = %5\n")
                              .arg(outputFolder)
                              .arg(images.size())
                              .arg(commonSize.width())
                              .arg(commonSize.height())
                              .arg(extension)
                              .toUtf8();
    return result;
}

MotConversion convertMots(const QVector<MotImage>& images,
                          const QStringList& classes,
                          const QString& outputFolder) {
    MotConversion result;
    if (images.isEmpty() || classes.isEmpty()) {
        result.error = QStringLiteral("MOTS export needs images and classes");
        return result;
    }
    QSet<int> frames;
    QSize commonSize;
    QString extension;
    for (const MotImage& image : images) {
        const int id = frameId(image.fileName);
        const QString suffix =
                QLatin1Char('.') + QFileInfo(image.fileName).suffix();
        if (frames.contains(id) || !image.size.isValid()) {
            result.error =
                    QStringLiteral("Invalid or duplicate MOTS frame ID %1")
                            .arg(id);
            return result;
        }
        if (commonSize.isValid() &&
            (commonSize != image.size || extension != suffix)) {
            result.error = QStringLiteral(
                    "MOTS sequence images must have one size and extension");
            return result;
        }
        frames.insert(id);
        commonSize = image.size;
        extension = suffix;
        for (const QJsonValue& value :
             image.document.value(QStringLiteral("shapes")).toArray()) {
            const QJsonObject shape = value.toObject();
            if (shape.value(QStringLiteral("shape_type")).toString() !=
                QStringLiteral("polygon"))
                continue;
            const int category = classes.indexOf(
                    shape.value(QStringLiteral("label")).toString());
            if (category < 0) continue;
            const QJsonArray points =
                    shape.value(QStringLiteral("points")).toArray();
            if (points.size() < 3) {
                result.error = QStringLiteral("Invalid MOTS polygon in %1")
                                       .arg(image.fileName);
                return result;
            }
            QStringList coordinates;
            for (const QJsonValue& point : points) {
                const QJsonArray xy = point.toArray();
                if (xy.size() != 2 || !xy[0].isDouble() || !xy[1].isDouble() ||
                    !std::isfinite(xy[0].toDouble()) ||
                    !std::isfinite(xy[1].toDouble())) {
                    result.error = QStringLiteral("Invalid MOTS polygon in %1")
                                           .arg(image.fileName);
                    return result;
                }
                const double x = std::clamp(xy[0].toDouble(), 0.0,
                                            double(image.size.width() - 1));
                const double y = std::clamp(xy[1].toDouble(), 0.0,
                                            double(image.size.height() - 1));
                coordinates.append(QStringLiteral("[%1, %2]")
                                           .arg(QString::number(x, 'g', 15),
                                                QString::number(y, 'g', 15)));
            }
            const int group = shape.value(QStringLiteral("group_id")).toInt(-1);
            const int track = group == 0 ? -1 : group;
            result.groundTruth +=
                    QStringLiteral("%1 %2 %3 %4 %5 [%6]\n")
                            .arg(id)
                            .arg(track)
                            .arg(category)
                            .arg(image.size.height())
                            .arg(image.size.width())
                            .arg(coordinates.join(QStringLiteral(", ")))
                            .toUtf8();
            ++result.count;
        }
    }
    result.sequence = QStringLiteral(
                              "[Sequence]\nname = MOTS\nimdir = %1\nframerate "
                              "= 30\nseqlength = %2\n"
                              "imwidth = %3\nimheight = %4\nimext = %5\n")
                              .arg(outputFolder)
                              .arg(images.size())
                              .arg(commonSize.width())
                              .arg(commonSize.height())
                              .arg(extension)
                              .toUtf8();
    return result;
}
