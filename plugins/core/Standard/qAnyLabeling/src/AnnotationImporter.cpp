// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationImporter.h"

#include <QtCompat.h>

#include <QBuffer>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSet>
#include <QVector>
#include <QXmlStreamReader>
#include <algorithm>
#include <cmath>
#include <limits>

#include "MaskPolygon.h"

namespace {

bool decodeCocoRle(const QJsonObject& rle,
                   const QSize& imageSize,
                   QByteArray& plane,
                   QString& error) {
    const int width = imageSize.width();
    const int height = imageSize.height();
    const qint64 pixelCount = qint64(width) * height;
    if (width <= 0 || height <= 0 || pixelCount > 100000000) {
        error = QStringLiteral("COCO RLE has invalid or oversized dimensions");
        return false;
    }
    const QJsonArray size = rle.value(QStringLiteral("size")).toArray();
    if (size.size() != 2 || size[0].toInt(-1) != height ||
        size[1].toInt(-1) != width) {
        error = QStringLiteral("COCO RLE size does not match its image");
        return false;
    }

    QVector<qint64> runs;
    const QJsonValue counts = rle.value(QStringLiteral("counts"));
    if (counts.isArray()) {
        for (const QJsonValue& count : counts.toArray()) {
            const double value = count.toDouble(-1);
            if (!std::isfinite(value) || value < 0 ||
                value != std::floor(value) || value > pixelCount) {
                error = QStringLiteral("COCO RLE has an invalid run length");
                return false;
            }
            runs.append(qint64(value));
        }
    } else if (counts.isString()) {
        const QByteArray encoded = counts.toString().toLatin1();
        for (int position = 0; position < encoded.size();) {
            qint64 value = 0;
            int shift = 0;
            int symbol = 0;
            do {
                if (position >= encoded.size() || shift > 50) {
                    error = QStringLiteral(
                            "COCO RLE has invalid compressed counts");
                    return false;
                }
                symbol = uchar(encoded[position++]) - 48;
                if (symbol < 0 || symbol > 0x3f) {
                    error = QStringLiteral(
                            "COCO RLE has invalid compressed counts");
                    return false;
                }
                value |= qint64(symbol & 0x1f) << shift;
                shift += 5;
            } while (symbol & 0x20);
            if (symbol & 0x10) value -= qint64(1) << shift;
            if (runs.size() > 2) value += runs[runs.size() - 2];
            if (value < 0 || value > pixelCount) {
                error = QStringLiteral("COCO RLE has an invalid run length");
                return false;
            }
            runs.append(value);
        }
    } else {
        error = QStringLiteral("COCO RLE counts must be an array or string");
        return false;
    }

    plane = QByteArray(int(pixelCount), '\0');
    qint64 offset = 0;
    for (int i = 0; i < runs.size(); ++i) {
        const qint64 count = runs[i];
        if (count > pixelCount - offset) {
            error = QStringLiteral("COCO RLE exceeds its image dimensions");
            return false;
        }
        if (i % 2 == 1)
            std::fill_n(plane.data() + offset, size_t(count), char(1));
        offset += count;
    }
    if (runs.isEmpty() || offset != pixelCount) {
        error = QStringLiteral("COCO RLE does not cover its image dimensions");
        return false;
    }
    return true;
}

QString encodeCocoMask(const QByteArray& plane, const QSize& size) {
    QImage mask(size, QImage::Format_ARGB32);
    if (mask.isNull()) return {};
    mask.fill(Qt::transparent);
    for (int x = 0; x < size.width(); ++x)
        for (int y = 0; y < size.height(); ++y)
            if (plane[x * size.height() + y])
                mask.setPixel(x, y, qRgba(0, 180, 255, 255));
    QByteArray png;
    QBuffer buffer(&png);
    if (!buffer.open(QIODevice::WriteOnly) || !mask.save(&buffer, "PNG"))
        return {};
    return QString::fromLatin1(png.toBase64());
}

}  // namespace

YoloImport importYolo(const QByteArray& labels,
                      const QStringList& classes,
                      YoloFormat format,
                      const QString& imageName,
                      const QSize& size,
                      const PoseConfig* poseConfig) {
    YoloImport result;
    if (format == YoloFormat::Pose) {
        if (!poseConfig || !poseConfig->error.isEmpty() ||
            poseConfig->classes.isEmpty() || !size.isValid()) {
            result.error = QStringLiteral(
                    "YOLO pose import requires valid image dimensions and pose "
                    "classes");
            return result;
        }
        QJsonArray shapes;
        int maxKeypoints = 0;
        for (const PoseClass& poseClass : poseConfig->classes)
            maxKeypoints = std::max(maxKeypoints, poseClass.keypoints.size());
        const int stride = poseConfig->hasVisible ? 3 : 2;
        const QStringList lines =
                QString::fromUtf8(labels).split(QLatin1Char('\n'));
        const QRegularExpression whitespace(QStringLiteral("\\s+"));
        for (int lineIndex = 0; lineIndex < lines.size(); ++lineIndex) {
            const QString line = lines[lineIndex].trimmed();
            if (line.isEmpty()) continue;
            const QStringList fields =
                    line.split(whitespace, QtCompat::SkipEmptyParts);
            bool valid = false;
            const int classId = fields.value(0).toInt(&valid);
            if (!valid || classId < 0 ||
                classId >= poseConfig->classes.size() ||
                fields.size() != 5 + stride * maxKeypoints) {
                result.error = QStringLiteral(
                                       "Line %1: invalid pose class or "
                                       "coordinate count")
                                       .arg(lineIndex + 1);
                return result;
            }
            QVector<double> values;
            for (int i = 1; i < fields.size(); ++i) {
                const double number = fields[i].toDouble(&valid);
                if (!valid || !std::isfinite(number)) {
                    result.error =
                            QStringLiteral("Line %1: invalid pose coordinate")
                                    .arg(lineIndex + 1);
                    return result;
                }
                values.append(number);
            }
            if (values[2] <= 0 || values[3] <= 0) {
                result.error = QStringLiteral("Line %1: invalid pose box size")
                                       .arg(lineIndex + 1);
                return result;
            }
            const int x0 = int((values[0] - values[2] / 2) * size.width());
            const int y0 = int((values[1] - values[3] / 2) * size.height());
            const int x1 = int((values[0] + values[2] / 2) * size.width());
            const int y1 = int((values[1] + values[3] / 2) * size.height());
            const int group = result.count++;
            shapes.append(QJsonObject{
                    {QStringLiteral("label"),
                     poseConfig->classes[classId].name},
                    {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
                    {QStringLiteral("flags"), QJsonObject()},
                    {QStringLiteral("points"),
                     QJsonArray{QJsonArray{x0, y0}, QJsonArray{x1, y0},
                                QJsonArray{x1, y1}, QJsonArray{x0, y1}}},
                    {QStringLiteral("group_id"), group},
                    {QStringLiteral("description"), QJsonValue::Null},
                    {QStringLiteral("difficult"), false},
                    {QStringLiteral("attributes"), QJsonObject()}});
            for (int i = 0; i < poseConfig->classes[classId].keypoints.size();
                 ++i) {
                const double x = values[4 + i * stride] * size.width();
                const double y = values[5 + i * stride] * size.height();
                const double visibility =
                        poseConfig->hasVisible ? values[6 + i * stride] : 0;
                if (visibility != std::floor(visibility) || visibility < 0 ||
                    visibility > 2) {
                    result.error =
                            QStringLiteral("Line %1: invalid pose visibility")
                                    .arg(lineIndex + 1);
                    return result;
                }
                if ((x == 0 && y == 0) ||
                    (poseConfig->hasVisible && visibility == 0))
                    continue;
                shapes.append(QJsonObject{
                        {QStringLiteral("label"),
                         poseConfig->classes[classId].keypoints[i]},
                        {QStringLiteral("shape_type"), QStringLiteral("point")},
                        {QStringLiteral("flags"), QJsonObject()},
                        {QStringLiteral("points"),
                         QJsonArray{QJsonArray{x, y}}},
                        {QStringLiteral("group_id"), group},
                        {QStringLiteral("description"), QJsonValue::Null},
                        {QStringLiteral("difficult"), visibility == 1},
                        {QStringLiteral("attributes"), QJsonObject()}});
            }
        }
        result.document = {{QStringLiteral("version"), QStringLiteral("4.0.6")},
                           {QStringLiteral("flags"), QJsonObject()},
                           {QStringLiteral("shapes"), shapes},
                           {QStringLiteral("imagePath"), imageName},
                           {QStringLiteral("imageData"), QJsonValue::Null},
                           {QStringLiteral("imageHeight"), size.height()},
                           {QStringLiteral("imageWidth"), size.width()}};
        return result;
    }
    if (size.width() <= 0 || size.height() <= 0 || classes.isEmpty()) {
        result.error = QStringLiteral("Invalid image dimensions or classes");
        return result;
    }
    QJsonArray shapes;
    const QStringList lines =
            QString::fromUtf8(labels).split(QLatin1Char('\n'));
    const QRegularExpression whitespace(QStringLiteral("\\s+"));
    for (int lineIndex = 0; lineIndex < lines.size(); ++lineIndex) {
        const QString line = lines[lineIndex].trimmed();
        if (line.isEmpty()) continue;
        const QStringList fields =
                line.split(whitespace, QtCompat::SkipEmptyParts);
        const int expected = format == YoloFormat::Boxes           ? 5
                             : format == YoloFormat::OrientedBoxes ? 9
                                                                   : 7;
        if (fields.size() < expected ||
            (format != YoloFormat::Segments && fields.size() != expected) ||
            (format == YoloFormat::Segments && fields.size() % 2 == 0)) {
            result.error = QStringLiteral("Line %1: invalid coordinate count")
                                   .arg(lineIndex + 1);
            return result;
        }
        bool valid = false;
        const int classId = fields[0].toInt(&valid);
        if (!valid || classId < 0 || classId >= classes.size()) {
            result.error = QStringLiteral("Line %1: unknown class index")
                                   .arg(lineIndex + 1);
            return result;
        }
        QVector<double> values;
        for (int i = 1; i < fields.size(); ++i) {
            const double number = fields[i].toDouble(&valid);
            if (!valid || !std::isfinite(number)) {
                result.error = QStringLiteral("Line %1: invalid coordinate")
                                       .arg(lineIndex + 1);
                return result;
            }
            values.append(number);
        }
        QJsonArray points;
        QString type;
        if (format == YoloFormat::Boxes) {
            if (values[2] <= 0 || values[3] <= 0) {
                result.error = QStringLiteral("Line %1: invalid box size")
                                       .arg(lineIndex + 1);
                return result;
            }
            const double left = (values[0] - values[2] / 2) * size.width();
            const double top = (values[1] - values[3] / 2) * size.height();
            const double right = (values[0] + values[2] / 2) * size.width();
            const double bottom = (values[1] + values[3] / 2) * size.height();
            points = {QJsonArray{left, top}, QJsonArray{right, top},
                      QJsonArray{right, bottom}, QJsonArray{left, bottom}};
            type = QStringLiteral("rectangle");
        } else {
            for (int i = 0; i < values.size(); i += 2)
                points.append(QJsonArray{values[i] * size.width(),
                                         values[i + 1] * size.height()});
            type = format == YoloFormat::Segments ? QStringLiteral("polygon")
                                                  : QStringLiteral("rotation");
        }
        QJsonObject shape{{QStringLiteral("label"), classes[classId]},
                          {QStringLiteral("shape_type"), type},
                          {QStringLiteral("flags"), QJsonObject()},
                          {QStringLiteral("points"), points},
                          {QStringLiteral("group_id"), QJsonValue::Null},
                          {QStringLiteral("description"), QJsonValue::Null},
                          {QStringLiteral("difficult"), false},
                          {QStringLiteral("attributes"), QJsonObject()}};
        if (format == YoloFormat::OrientedBoxes) {
            double radians =
                    std::atan2(values[3] - values[1], values[2] - values[0]);
            if (radians < 0) radians += 2 * std::acos(-1.0);
            shape.insert(QStringLiteral("direction"), radians);
        }
        shapes.append(shape);
    }
    result.count = shapes.size();
    result.document = {{QStringLiteral("version"), QStringLiteral("4.0.6")},
                       {QStringLiteral("flags"), QJsonObject()},
                       {QStringLiteral("shapes"), shapes},
                       {QStringLiteral("imagePath"), imageName},
                       {QStringLiteral("imageData"), QJsonValue::Null},
                       {QStringLiteral("imageHeight"), size.height()},
                       {QStringLiteral("imageWidth"), size.width()}};
    return result;
}

AnnotationImport importVoc(const QByteArray& xml,
                           const QString& imageName,
                           bool polygons) {
    AnnotationImport result;
    QXmlStreamReader reader(xml);
    QJsonArray shapes;
    int width = 0, height = 0;
    QString xmlImageName;
    if (!reader.readNextStartElement() ||
        reader.name() != QStringLiteral("annotation")) {
        result.error = QStringLiteral("VOC root must be annotation");
        return result;
    }
    while (reader.readNextStartElement()) {
        if (reader.name() == QStringLiteral("filename")) {
            xmlImageName = reader.readElementText();
        } else if (reader.name() == QStringLiteral("size")) {
            while (reader.readNextStartElement()) {
                if (reader.name() == QStringLiteral("width"))
                    width = reader.readElementText().toInt();
                else if (reader.name() == QStringLiteral("height"))
                    height = reader.readElementText().toInt();
                else
                    reader.skipCurrentElement();
            }
        } else if (reader.name() == QStringLiteral("object")) {
            QString label;
            bool difficult = false;
            QJsonObject bounds, polygon;
            while (reader.readNextStartElement()) {
                const QString field = reader.name().toString();
                if (field == QStringLiteral("name"))
                    label = reader.readElementText();
                else if (field == QStringLiteral("difficult"))
                    difficult = reader.readElementText().toInt() != 0;
                else if (field == QStringLiteral("bndbox") ||
                         field == QStringLiteral("polygon")) {
                    QJsonObject values;
                    while (reader.readNextStartElement()) {
                        bool valid = false;
                        const QString key = reader.name().toString();
                        const double number =
                                reader.readElementText().toDouble(&valid);
                        if (valid && std::isfinite(number))
                            values.insert(key, number);
                    }
                    if (field == QStringLiteral("bndbox"))
                        bounds = values;
                    else
                        polygon = values;
                } else
                    reader.skipCurrentElement();
            }
            if (label.isEmpty()) continue;
            QJsonArray points;
            QString type;
            if (polygons && !polygon.isEmpty()) {
                for (int i = 1;; ++i) {
                    const QString xKey = QStringLiteral("x%1").arg(i);
                    const QString yKey = QStringLiteral("y%1").arg(i);
                    if (!polygon.contains(xKey) || !polygon.contains(yKey))
                        break;
                    points.append(QJsonArray{polygon.value(xKey),
                                             polygon.value(yKey)});
                }
                if (points.size() >= 3) type = QStringLiteral("polygon");
            }
            if (type.isEmpty() && bounds.contains(QStringLiteral("xmin")) &&
                bounds.contains(QStringLiteral("ymin")) &&
                bounds.contains(QStringLiteral("xmax")) &&
                bounds.contains(QStringLiteral("ymax"))) {
                const double x0 =
                        bounds.value(QStringLiteral("xmin")).toDouble();
                const double y0 =
                        bounds.value(QStringLiteral("ymin")).toDouble();
                const double x1 =
                        bounds.value(QStringLiteral("xmax")).toDouble();
                const double y1 =
                        bounds.value(QStringLiteral("ymax")).toDouble();
                points = {QJsonArray{x0, y0}, QJsonArray{x1, y0},
                          QJsonArray{x1, y1}, QJsonArray{x0, y1}};
                type = QStringLiteral("rectangle");
            }
            if (type.isEmpty()) continue;
            shapes.append(
                    QJsonObject{{QStringLiteral("label"), label},
                                {QStringLiteral("description"), QString()},
                                {QStringLiteral("shape_type"), type},
                                {QStringLiteral("points"), points},
                                {QStringLiteral("group_id"), QJsonValue::Null},
                                {QStringLiteral("difficult"), difficult},
                                {QStringLiteral("flags"), QJsonObject()}});
        } else
            reader.skipCurrentElement();
    }
    if (reader.hasError() || width <= 0 || height <= 0) {
        result.error = reader.hasError()
                               ? reader.errorString()
                               : QStringLiteral("VOC is missing image size");
        return result;
    }
    if (!xmlImageName.isEmpty() &&
        (QFileInfo(xmlImageName).suffix().size() > 0 || imageName.isEmpty()))
        xmlImageName = QFileInfo(xmlImageName).fileName();
    else
        xmlImageName = imageName;
    result.count = shapes.size();
    result.document = {{QStringLiteral("version"), QStringLiteral("4.0.6")},
                       {QStringLiteral("flags"), QJsonObject()},
                       {QStringLiteral("shapes"), shapes},
                       {QStringLiteral("imagePath"), xmlImageName},
                       {QStringLiteral("imageData"), QJsonValue::Null},
                       {QStringLiteral("imageHeight"), height},
                       {QStringLiteral("imageWidth"), width}};
    return result;
}

AnnotationImport importDota(const QByteArray& text,
                            const QString& imageName,
                            const QSize& size) {
    AnnotationImport result;
    if (!size.isValid()) {
        result.error = QStringLiteral("Invalid image dimensions");
        return result;
    }
    QJsonArray shapes;
    const QRegularExpression whitespace(QStringLiteral("\\s+"));
    const QStringList lines = QString::fromUtf8(text).split(QLatin1Char('\n'));
    for (int line = 0; line < lines.size(); ++line) {
        const QStringList fields = lines[line].trimmed().split(
                whitespace, QtCompat::SkipEmptyParts);
        if (fields.isEmpty()) continue;
        if (fields.size() != 10) {
            result.error = QStringLiteral(
                                   "Line %1: DOTA needs 8 coordinates, label, "
                                   "difficult")
                                   .arg(line + 1);
            return result;
        }
        QJsonArray points;
        bool valid = true;
        for (int i = 0; i < 8; i += 2) {
            bool xValid = false, yValid = false;
            const double x = fields[i].toDouble(&xValid);
            const double y = fields[i + 1].toDouble(&yValid);
            if (!xValid || !yValid || !std::isfinite(x) || !std::isfinite(y)) {
                valid = false;
                break;
            }
            points.append(QJsonArray{x, y});
        }
        if (!valid) {
            result.error = QStringLiteral("Line %1: invalid DOTA coordinates")
                                   .arg(line + 1);
            return result;
        }
        const QString label = fields[8];
        bool difficultValid = false;
        const bool difficult = fields[9].toInt(&difficultValid) != 0;
        if (!difficultValid) {
            result.error =
                    QStringLiteral("Line %1: invalid DOTA difficult flag")
                            .arg(line + 1);
            return result;
        }
        double direction =
                std::atan2(points[1].toArray()[1].toDouble() -
                                   points[0].toArray()[1].toDouble(),
                           points[1].toArray()[0].toDouble() -
                                   points[0].toArray()[0].toDouble());
        if (direction < 0) direction += 2 * std::acos(-1.0);
        shapes.append(QJsonObject{
                {QStringLiteral("label"), label},
                {QStringLiteral("shape_type"), QStringLiteral("rotation")},
                {QStringLiteral("points"), points},
                {QStringLiteral("group_id"), QJsonValue::Null},
                {QStringLiteral("difficult"), difficult},
                {QStringLiteral("direction"), direction},
                {QStringLiteral("flags"), QJsonObject()}});
    }
    result.count = shapes.size();
    result.document = {{QStringLiteral("version"), QStringLiteral("4.0.6")},
                       {QStringLiteral("flags"), QJsonObject()},
                       {QStringLiteral("shapes"), shapes},
                       {QStringLiteral("imagePath"), imageName},
                       {QStringLiteral("imageData"), QJsonValue::Null},
                       {QStringLiteral("imageHeight"), size.height()},
                       {QStringLiteral("imageWidth"), size.width()}};
    return result;
}

CocoImport importCoco(const QByteArray& json, CocoFormat format) {
    CocoImport result;
    QJsonParseError parseError;
    const QJsonDocument parsed = QJsonDocument::fromJson(json, &parseError);
    if (parseError.error != QJsonParseError::NoError || !parsed.isObject()) {
        result.error = QStringLiteral("Invalid COCO JSON: %1")
                               .arg(parseError.errorString());
        return result;
    }
    const QJsonObject root = parsed.object();
    if (!root.value(QStringLiteral("categories")).isArray() ||
        !root.value(QStringLiteral("images")).isArray() ||
        !root.value(QStringLiteral("annotations")).isArray()) {
        result.error = QStringLiteral(
                "COCO needs categories, images, and annotations arrays");
        return result;
    }
    QHash<int, QJsonObject> categories;
    for (const QJsonValue& value :
         root.value(QStringLiteral("categories")).toArray()) {
        const QJsonObject category = value.toObject();
        const int id = category.value(QStringLiteral("id")).toInt(-1);
        if (id < 0 ||
            category.value(QStringLiteral("name")).toString().isEmpty() ||
            categories.contains(id)) {
            result.error =
                    QStringLiteral("Invalid or duplicate COCO category ID %1")
                            .arg(id);
            return result;
        }
        categories.insert(id, category);
    }
    QHash<int, QString> imageNames;
    QHash<int, QSize> imageSizes;
    QHash<int, QJsonArray> shapes;
    QSet<QString> lowerNames;
    for (const QJsonValue& value :
         root.value(QStringLiteral("images")).toArray()) {
        const QJsonObject image = value.toObject();
        const int id = image.value(QStringLiteral("id")).toInt(-1);
        const QString name =
                QFileInfo(image.value(QStringLiteral("file_name")).toString())
                        .fileName();
        const int width = image.value(QStringLiteral("width")).toInt();
        const int height = image.value(QStringLiteral("height")).toInt();
        if (id < 0 || name.isEmpty() || width <= 0 || height <= 0 ||
            imageNames.contains(id) || lowerNames.contains(name.toLower())) {
            result.error = QStringLiteral("Invalid or duplicate COCO image %1")
                                   .arg(name);
            return result;
        }
        lowerNames.insert(name.toLower());
        imageNames.insert(id, name);
        imageSizes.insert(id, QSize(width, height));
        result.documents.insert(
                name, {{QStringLiteral("version"), QStringLiteral("4.0.6")},
                       {QStringLiteral("flags"), QJsonObject()},
                       {QStringLiteral("imagePath"), name},
                       {QStringLiteral("imageData"), QJsonValue::Null},
                       {QStringLiteral("imageWidth"), width},
                       {QStringLiteral("imageHeight"), height}});
    }
    QHash<int, int> nextGroup;
    for (const QJsonValue& value :
         root.value(QStringLiteral("annotations")).toArray()) {
        const QJsonObject annotation = value.toObject();
        const int imageId =
                annotation.value(QStringLiteral("image_id")).toInt(-1);
        const int categoryId =
                annotation.value(QStringLiteral("category_id")).toInt(-1);
        if (!imageNames.contains(imageId) || !categories.contains(categoryId)) {
            result.error = QStringLiteral(
                    "COCO annotation references an unknown image or category");
            return result;
        }
        const QJsonObject category = categories.value(categoryId);
        const QString label = category.value(QStringLiteral("name")).toString();
        const bool difficult = annotation.value(QStringLiteral("ignore"))
                                       .toVariant()
                                       .toString()
                                       .toInt() != 0;
        QJsonArray imageShapes = shapes.value(imageId);
        auto appendShape = [&](const QString& shapeType,
                               const QJsonArray& points,
                               const QJsonValue& group,
                               bool pointDifficult = false,
                               const QString& exactMask = {}) {
            QJsonObject shape{{QStringLiteral("label"), label},
                              {QStringLiteral("shape_type"), shapeType},
                              {QStringLiteral("flags"), QJsonObject()},
                              {QStringLiteral("points"), points},
                              {QStringLiteral("group_id"), group},
                              {QStringLiteral("description"), QJsonValue::Null},
                              {QStringLiteral("difficult"), pointDifficult},
                              {QStringLiteral("attributes"), QJsonObject()}};
            if (!exactMask.isEmpty())
                shape.insert(QStringLiteral("acloudviewer_mask_png"),
                             exactMask);
            imageShapes.append(shape);
        };
        if (format == CocoFormat::Segments) {
            const QJsonValue segmentationValue =
                    annotation.value(QStringLiteral("segmentation"));
            auto appendRle = [&](const QJsonObject& rle) {
                QByteArray plane;
                if (!decodeCocoRle(rle, imageSizes.value(imageId), plane,
                                   result.error))
                    return false;
                const QString exactMask =
                        encodeCocoMask(plane, imageSizes.value(imageId));
                if (exactMask.isEmpty()) {
                    result.error =
                            QStringLiteral("Unable to encode COCO RLE mask");
                    return false;
                }
                const QSize size = imageSizes.value(imageId);
                QByteArray rowMajor(plane.size(), '\0');
                for (int x = 0; x < size.width(); ++x)
                    for (int y = 0; y < size.height(); ++y)
                        rowMajor[y * size.width() + x] =
                                plane[x * size.height() + y];
                const QVector<QPointF> polygon = largestMaskPolygon(
                        reinterpret_cast<const uint8_t*>(rowMajor.constData()),
                        size.width(), size.height(), size.width());
                QJsonArray points;
                for (const QPointF& point : polygon)
                    points.append(QJsonArray{point.x(), point.y()});
                appendShape(QStringLiteral("polygon"), points,
                            QJsonValue(QJsonValue::Null), difficult, exactMask);
                return true;
            };
            if (segmentationValue.isObject()) {
                if (!appendRle(segmentationValue.toObject())) return result;
                shapes.insert(imageId, imageShapes);
                continue;
            }
            const QJsonArray segmentations = segmentationValue.toArray();
            QVector<QJsonArray> polygons;
            bool hasRle = false;
            for (const QJsonValue& segment : segmentations) {
                if (segment.isObject()) {
                    if (!appendRle(segment.toObject())) return result;
                    hasRle = true;
                    continue;
                }
                const QJsonArray coordinates = segment.toArray();
                if (coordinates.size() < 6 || coordinates.size() % 2 != 0)
                    continue;
                QJsonArray points;
                QSet<QString> seen;
                for (int i = 0; i < coordinates.size(); i += 2) {
                    const double x = coordinates[i].toDouble(
                            std::numeric_limits<double>::quiet_NaN());
                    const double y = coordinates[i + 1].toDouble(
                            std::numeric_limits<double>::quiet_NaN());
                    if (!std::isfinite(x) || !std::isfinite(y)) {
                        result.error = QStringLiteral(
                                "COCO segmentation has invalid coordinates");
                        return result;
                    }
                    const QString key = QString::number(x, 'g', 17) +
                                        QLatin1Char(',') +
                                        QString::number(y, 'g', 17);
                    if (seen.contains(key)) continue;
                    seen.insert(key);
                    points.append(QJsonArray{x, y});
                }
                if (points.size() >= 3) polygons.append(points);
            }
            if (!polygons.isEmpty()) {
                const QJsonValue group =
                        polygons.size() > 1 ? QJsonValue(++nextGroup[imageId])
                                            : QJsonValue(QJsonValue::Null);
                for (const QJsonArray& points : polygons)
                    appendShape(QStringLiteral("polygon"), points, group,
                                difficult);
            }
            if (!hasRle && polygons.isEmpty()) continue;
        } else {
            const QJsonArray bbox =
                    annotation.value(QStringLiteral("bbox")).toArray();
            if (bbox.size() != 4) {
                result.error =
                        QStringLiteral("COCO annotation has no valid bbox");
                return result;
            }
            const double x =
                    bbox[0].toDouble(std::numeric_limits<double>::quiet_NaN());
            const double y =
                    bbox[1].toDouble(std::numeric_limits<double>::quiet_NaN());
            const double width =
                    bbox[2].toDouble(std::numeric_limits<double>::quiet_NaN());
            const double height =
                    bbox[3].toDouble(std::numeric_limits<double>::quiet_NaN());
            if (!std::isfinite(x) || !std::isfinite(y) ||
                !std::isfinite(width) || !std::isfinite(height) || width <= 0 ||
                height <= 0) {
                result.error = QStringLiteral(
                        "COCO annotation has invalid bbox coordinates");
                return result;
            }
            const QJsonValue group = format == CocoFormat::Pose
                                             ? QJsonValue(nextGroup[imageId]++)
                                             : QJsonValue(QJsonValue::Null);
            appendShape(QStringLiteral("rectangle"),
                        QJsonArray{QJsonArray{x, y}, QJsonArray{x + width, y},
                                   QJsonArray{x + width, y + height},
                                   QJsonArray{x, y + height}},
                        group, difficult);
            if (format == CocoFormat::Pose) {
                const QJsonArray names =
                        category.value(QStringLiteral("keypoints")).toArray();
                const QJsonArray keypoints =
                        annotation.value(QStringLiteral("keypoints")).toArray();
                const int stride = keypoints.size() == names.size() * 3 ? 3 : 2;
                if (names.isEmpty() ||
                    keypoints.size() != names.size() * stride) {
                    result.error = QStringLiteral(
                            "COCO pose keypoints do not match category names");
                    return result;
                }
                for (int i = 0; i < names.size(); ++i) {
                    const double px = keypoints[i * stride].toDouble(
                            std::numeric_limits<double>::quiet_NaN());
                    const double py = keypoints[i * stride + 1].toDouble(
                            std::numeric_limits<double>::quiet_NaN());
                    const int visibility =
                            stride == 3 ? keypoints[i * stride + 2].toInt(-1)
                                        : 0;
                    if (!std::isfinite(px) || !std::isfinite(py) ||
                        visibility < 0 || visibility > 2 ||
                        names[i].toString().isEmpty()) {
                        result.error = QStringLiteral(
                                "COCO pose has invalid keypoint data");
                        return result;
                    }
                    if (px == 0 && py == 0 && visibility == 0) continue;
                    QJsonObject point{
                            {QStringLiteral("label"), names[i].toString()},
                            {QStringLiteral("shape_type"),
                             QStringLiteral("point")},
                            {QStringLiteral("flags"), QJsonObject()},
                            {QStringLiteral("points"),
                             QJsonArray{QJsonArray{px, py}}},
                            {QStringLiteral("group_id"), group},
                            {QStringLiteral("description"), QJsonValue::Null},
                            {QStringLiteral("difficult"), visibility == 1},
                            {QStringLiteral("attributes"), QJsonObject()}};
                    imageShapes.append(point);
                }
            }
        }
        shapes.insert(imageId, imageShapes);
    }
    for (auto it = imageNames.cbegin(); it != imageNames.cend(); ++it) {
        QJsonObject document = result.documents.value(it.value());
        const QJsonArray imageShapes = shapes.value(it.key());
        document.insert(QStringLiteral("shapes"), imageShapes);
        result.count += imageShapes.size();
        result.documents.insert(it.value(), document);
    }
    return result;
}

AnnotationImport importMask(const QImage& mask,
                            const QJsonObject& colorMap,
                            const QString& imageName,
                            const QSize& imageSize) {
    AnnotationImport result;
    if (mask.isNull() || !imageSize.isValid() || mask.size() != imageSize) {
        result.error = QStringLiteral("Mask dimensions must match the image");
        return result;
    }
    const QString type = colorMap.value(QStringLiteral("type")).toString();
    const bool gray = type == QStringLiteral("grayscale");
    const QJsonObject colors =
            colorMap.value(QStringLiteral("colors")).toObject();
    if ((!gray && type != QStringLiteral("rgb")) || colors.isEmpty()) {
        result.error = QStringLiteral("Mask color map needs type and colors");
        return result;
    }
    QStringList labels;
    QHash<QRgb, int> palette;
    for (auto it = colors.constBegin(); it != colors.constEnd(); ++it) {
        QRgb color = 0;
        if (gray) {
            const int value = it.value().toInt(-1);
            if (value < 0 || value > 255 || it.value().toDouble() != value) {
                result.error =
                        QStringLiteral("Invalid grayscale mask color for %1")
                                .arg(it.key());
                return result;
            }
            color = qRgb(value, value, value);
        } else {
            const QJsonArray channels = it.value().toArray();
            if (channels.size() != 3) {
                result.error = QStringLiteral("Invalid RGB mask color for %1")
                                       .arg(it.key());
                return result;
            }
            int rgb[3];
            for (int i = 0; i < 3; ++i) {
                rgb[i] = channels[i].toInt(-1);
                if (rgb[i] < 0 || rgb[i] > 255 ||
                    channels[i].toDouble() != rgb[i]) {
                    result.error =
                            QStringLiteral("Invalid RGB mask color for %1")
                                    .arg(it.key());
                    return result;
                }
            }
            color = qRgb(rgb[0], rgb[1], rgb[2]);
        }
        if (palette.contains(color)) {
            result.error = QStringLiteral("Duplicate mask color in map");
            return result;
        }
        palette.insert(color, labels.size());
        labels.append(it.key());
    }
    const int width = mask.width(), height = mask.height();
    if (width > std::numeric_limits<int>::max() / height) {
        result.error = QStringLiteral("Mask is too large");
        return result;
    }
    QVector<QVector<int>> positions(labels.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const QRgb source = mask.pixel(x, y);
            const int intensity = qGray(source);
            const QRgb color =
                    gray ? qRgb(intensity, intensity, intensity)
                         : qRgb(qRed(source), qGreen(source), qBlue(source));
            const auto found = palette.constFind(color);
            if (found != palette.cend())
                positions[found.value()].append(y * width + x);
        }
    }
    QJsonArray shapes;
    for (int i = 0; i < labels.size(); ++i) {
        if (positions[i].isEmpty()) continue;
        QByteArray plane(width * height, '\0');
        QImage exact(width, height, QImage::Format_ARGB32);
        exact.fill(Qt::transparent);
        for (int offset : positions[i]) {
            plane[offset] = '\1';
            exact.setPixel(offset % width, offset / width,
                           qRgba(0, 180, 255, 110));
        }
        const QVector<QPointF> polygon = largestMaskPolygon(
                reinterpret_cast<const uint8_t*>(plane.constData()), width,
                height, size_t(width));
        QJsonArray points;
        for (const QPointF& point : polygon)
            points.append(QJsonArray{point.x(), point.y()});
        QByteArray png;
        QBuffer buffer(&png);
        if (!buffer.open(QIODevice::WriteOnly) || !exact.save(&buffer, "PNG")) {
            result.error = QStringLiteral("Unable to encode imported mask");
            return result;
        }
        shapes.append(QJsonObject{
                {QStringLiteral("label"), labels[i]},
                {QStringLiteral("shape_type"), QStringLiteral("polygon")},
                {QStringLiteral("flags"), QJsonObject()},
                {QStringLiteral("points"), points},
                {QStringLiteral("group_id"), QJsonValue::Null},
                {QStringLiteral("acloudviewer_mask_png"),
                 QString::fromLatin1(png.toBase64())}});
    }
    result.count = shapes.size();
    result.document = {{QStringLiteral("version"), QStringLiteral("4.0.6")},
                       {QStringLiteral("flags"), QJsonObject()},
                       {QStringLiteral("shapes"), shapes},
                       {QStringLiteral("imagePath"), imageName},
                       {QStringLiteral("imageData"), QJsonValue::Null},
                       {QStringLiteral("imageHeight"), height},
                       {QStringLiteral("imageWidth"), width}};
    return result;
}
