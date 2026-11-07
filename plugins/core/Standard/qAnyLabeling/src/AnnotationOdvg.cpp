// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationOdvg.h"

#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSet>
#include <algorithm>
#include <cmath>
#include <limits>

#include "AnnotationDocument.h"

OdvgImport importOdvg(const QByteArray& jsonLines) {
    OdvgImport result;
    QSet<QString> names;
    const QList<QByteArray> lines = jsonLines.split('\n');
    for (int line = 0; line < lines.size(); ++line) {
        if (lines[line].trimmed().isEmpty()) continue;
        QJsonParseError error;
        const QJsonDocument parsed =
                QJsonDocument::fromJson(lines[line], &error);
        if (error.error != QJsonParseError::NoError || !parsed.isObject()) {
            result.error =
                    QStringLiteral("ODVG line %1: invalid JSON").arg(line + 1);
            return result;
        }
        const QJsonObject record = parsed.object();
        const QString name =
                QFileInfo(record.value(QStringLiteral("filename")).toString())
                        .fileName();
        const int width = record.value(QStringLiteral("width")).toInt();
        const int height = record.value(QStringLiteral("height")).toInt();
        if (name.isEmpty() || width <= 0 || height <= 0 ||
            !record.value(QStringLiteral("detection"))
                     .toObject()
                     .value(QStringLiteral("instances"))
                     .isArray()) {
            result.error =
                    QStringLiteral(
                            "ODVG line %1: invalid image or detection data")
                            .arg(line + 1);
            return result;
        }
        if (names.contains(name.toLower())) {
            result.error =
                    QStringLiteral("ODVG has duplicate image %1").arg(name);
            return result;
        }
        names.insert(name.toLower());
        QJsonArray shapes;
        for (const QJsonValue& value :
             record.value(QStringLiteral("detection"))
                     .toObject()
                     .value(QStringLiteral("instances"))
                     .toArray()) {
            const QJsonObject instance = value.toObject();
            const QString category =
                    instance.value(QStringLiteral("category")).toString();
            const QJsonArray bbox =
                    instance.value(QStringLiteral("bbox")).toArray();
            if (category.isEmpty() || bbox.size() != 4) {
                result.error = QStringLiteral("ODVG line %1: invalid instance")
                                       .arg(line + 1);
                return result;
            }
            double coords[4];
            for (int i = 0; i < 4; ++i) {
                coords[i] = bbox[i].toDouble(
                        std::numeric_limits<double>::quiet_NaN());
                if (!std::isfinite(coords[i])) {
                    result.error =
                            QStringLiteral(
                                    "ODVG line %1: invalid bbox coordinate")
                                    .arg(line + 1);
                    return result;
                }
            }
            if (coords[2] <= coords[0] || coords[3] <= coords[1]) {
                result.error = QStringLiteral("ODVG line %1: invalid bbox size")
                                       .arg(line + 1);
                return result;
            }
            shapes.append(QJsonObject{
                    {QStringLiteral("label"), category},
                    {QStringLiteral("description"), QJsonValue::Null},
                    {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
                    {QStringLiteral("points"),
                     QJsonArray{QJsonArray{coords[0], coords[1]},
                                QJsonArray{coords[2], coords[1]},
                                QJsonArray{coords[2], coords[3]},
                                QJsonArray{coords[0], coords[3]}}},
                    {QStringLiteral("group_id"), QJsonValue::Null},
                    {QStringLiteral("difficult"), false},
                    {QStringLiteral("direction"), 0},
                    {QStringLiteral("flags"), QJsonObject()}});
            ++result.count;
        }
        result.documents.insert(
                name, {{QStringLiteral("version"), annotationFormatVersion()},
                       {QStringLiteral("flags"), QJsonObject()},
                       {QStringLiteral("shapes"), shapes},
                       {QStringLiteral("imagePath"), name},
                       {QStringLiteral("imageData"), QJsonValue::Null},
                       {QStringLiteral("imageHeight"), height},
                       {QStringLiteral("imageWidth"), width}});
    }
    return result;
}

OdvgConversion convertOdvg(const QVector<CocoImage>& images,
                           const QStringList& classes) {
    OdvgConversion result;
    if (classes.isEmpty()) {
        result.error = QStringLiteral("ODVG export needs classes");
        return result;
    }
    QJsonObject labelMap;
    for (int i = 0; i < classes.size(); ++i)
        labelMap.insert(QString::number(i), classes[i]);
    result.labelMap = QJsonDocument(labelMap).toJson(QJsonDocument::Compact);
    for (const CocoImage& image : images) {
        const int width =
                image.document.value(QStringLiteral("imageWidth")).toInt();
        const int height =
                image.document.value(QStringLiteral("imageHeight")).toInt();
        if (image.fileName.isEmpty() || width <= 0 || height <= 0) {
            result.error =
                    QStringLiteral("ODVG image has invalid dimensions: %1")
                            .arg(image.fileName);
            return result;
        }
        QJsonArray instances;
        for (const QJsonValue& value :
             image.document.value(QStringLiteral("shapes")).toArray()) {
            const QJsonObject shape = value.toObject();
            if (shape.value(QStringLiteral("shape_type")).toString() !=
                QStringLiteral("rectangle"))
                continue;
            const QString category =
                    shape.value(QStringLiteral("label")).toString();
            const int classId = classes.indexOf(category);
            if (classId < 0) continue;
            const QJsonArray points =
                    shape.value(QStringLiteral("points")).toArray();
            if (points.size() != 2 && points.size() != 4) {
                result.error = QStringLiteral("Invalid ODVG rectangle in %1")
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
                result.error = QStringLiteral("Invalid ODVG rectangle in %1")
                                       .arg(image.fileName);
                return result;
            }
            const double x0 =
                    std::clamp(first[0].toDouble(), 0.0, double(width));
            const double y0 =
                    std::clamp(first[1].toDouble(), 0.0, double(height));
            const double x1 =
                    std::clamp(opposite[0].toDouble(), 0.0, double(width));
            const double y1 =
                    std::clamp(opposite[1].toDouble(), 0.0, double(height));
            if (x1 <= x0 || y1 <= y0) {
                result.error = QStringLiteral("Invalid ODVG bbox in %1")
                                       .arg(image.fileName);
                return result;
            }
            instances.append(QJsonObject{
                    {QStringLiteral("bbox"), QJsonArray{x0, y0, x1, y1}},
                    {QStringLiteral("label"), classId},
                    {QStringLiteral("category"), category}});
            ++result.count;
        }
        const QJsonObject record{
                {QStringLiteral("filename"), image.fileName},
                {QStringLiteral("height"), height},
                {QStringLiteral("width"), width},
                {QStringLiteral("detection"),
                 QJsonObject{{QStringLiteral("instances"), instances}}}};
        result.annotations +=
                QJsonDocument(record).toJson(QJsonDocument::Compact) + '\n';
    }
    return result;
}
