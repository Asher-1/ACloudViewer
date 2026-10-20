// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationExporter.h"

#include <QDate>
#include <QHash>
#include <QImage>
#include <QJsonArray>
#include <QPainter>
#include <QPointF>
#include <QVector>
#include <QXmlStreamWriter>
#include <algorithm>
#include <cmath>

#include "AnnotationDocument.h"

namespace {
QString coordinate(double value) { return QString::number(value, 'f', 6); }

QVector<QPointF> pointsFor(const QJsonObject& shape) {
    QVector<QPointF> points;
    for (const QJsonValue& value :
         shape.value(QStringLiteral("points")).toArray()) {
        const QJsonArray pair = value.toArray();
        if (pair.size() != 2 || !pair[0].isDouble() || !pair[1].isDouble())
            return {};
        const double x = pair[0].toDouble(), y = pair[1].toDouble();
        if (!std::isfinite(x) || !std::isfinite(y)) return {};
        points.append(QPointF(x, y));
    }
    return points;
}

QJsonArray boxFor(const QVector<QPointF>& points, bool pixelBounds) {
    double minX = points[0].x(), minY = points[0].y();
    double maxX = minX, maxY = minY;
    for (const QPointF& point : points) {
        minX = std::min(minX, point.x());
        minY = std::min(minY, point.y());
        maxX = std::max(maxX, point.x());
        maxY = std::max(maxY, point.y());
    }
    if (pixelBounds) {
        minX = std::floor(minX);
        minY = std::floor(minY);
        maxX = std::floor(maxX) + 1;
        maxY = std::floor(maxY) + 1;
    }
    return {minX, minY, maxX - minX, maxY - minY};
}

double rasterArea(const QVector<QVector<QPointF>>& polygons) {
    if (polygons.isEmpty()) return 0;
    QVector<QPointF> all;
    for (const auto& polygon : polygons) all += polygon;
    const QJsonArray box = boxFor(all, false);
    const int left = int(std::floor(box[0].toDouble()));
    const int top = int(std::floor(box[1].toDouble()));
    const int width = std::max(
            1, int(std::ceil(box[0].toDouble() + box[2].toDouble())) - left);
    const int height = std::max(
            1, int(std::ceil(box[1].toDouble() + box[3].toDouble())) - top);
    QImage mask(width, height, QImage::Format_Grayscale8);
    if (mask.isNull()) return 0;
    mask.fill(0);
    QPainter painter(&mask);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::white);
    for (const auto& polygon : polygons) {
        QPolygonF path;
        for (const QPointF& point : polygon)
            path.append(QPointF(std::round(point.x() - left),
                                std::round(point.y() - top)));
        painter.drawPolygon(path);
    }
    painter.end();
    double area = 0;
    for (int y = 0; y < mask.height(); ++y) {
        const uchar* row = mask.constScanLine(y);
        for (int x = 0; x < mask.width(); ++x) area += row[x] != 0;
    }
    return area;
}
}  // namespace

YoloConversion convertYolo(const QJsonObject& document,
                           const QStringList& classes,
                           YoloFormat format,
                           bool keepOutOfBounds,
                           const PoseConfig* poseConfig) {
    if (format == YoloFormat::Pose) {
        if (!poseConfig)
            return {{}, 0, QStringLiteral("Pose classes are required")};
        return convertYoloPose(document, *poseConfig);
    }
    YoloConversion output;
    const int width = document.value(QStringLiteral("imageWidth")).toInt();
    const int height = document.value(QStringLiteral("imageHeight")).toInt();
    if (width <= 0 || height <= 0) {
        output.error = QStringLiteral("Invalid image dimensions");
        return output;
    }
    for (const QJsonValue& value :
         document.value(QStringLiteral("shapes")).toArray()) {
        if (!value.isObject()) continue;
        const QJsonObject shape = value.toObject();
        const QString type =
                shape.value(QStringLiteral("shape_type")).toString();
        if ((format == YoloFormat::Boxes &&
             type != QStringLiteral("rectangle")) ||
            (format == YoloFormat::Segments &&
             type != QStringLiteral("polygon")) ||
            (format == YoloFormat::OrientedBoxes &&
             type != QStringLiteral("rotation")))
            continue;
        const int classId = classes.indexOf(
                shape.value(QStringLiteral("label")).toString());
        if (classId < 0) continue;
        QVector<QPointF> points = pointsFor(shape);
        if (format == YoloFormat::Boxes && points.size() != 2 &&
            points.size() != 4)
            continue;
        if (format == YoloFormat::Segments && points.size() < 3) continue;
        if (format == YoloFormat::OrientedBoxes && points.size() != 4) continue;
        if (format == YoloFormat::OrientedBoxes && !keepOutOfBounds &&
            std::any_of(points.cbegin(), points.cend(),
                        [width, height](const QPointF& point) {
                            return point.x() < 0 || point.y() < 0 ||
                                   point.x() > width || point.y() > height;
                        }))
            continue;
        if (format != YoloFormat::OrientedBoxes) {
            for (QPointF& point : points) {
                point.setX(std::clamp(point.x(), 0.0, double(width)));
                point.setY(std::clamp(point.y(), 0.0, double(height)));
            }
        }
        QStringList fields{QString::number(classId)};
        if (format == YoloFormat::Boxes) {
            double minX = points[0].x(), maxX = minX;
            double minY = points[0].y(), maxY = minY;
            for (const QPointF& point : points) {
                minX = std::min(minX, point.x());
                minY = std::min(minY, point.y());
                maxX = std::max(maxX, point.x());
                maxY = std::max(maxY, point.y());
            }
            if (minX == maxX || minY == maxY) continue;
            fields << coordinate((minX + maxX) / (2 * width))
                   << coordinate((minY + maxY) / (2 * height))
                   << coordinate((maxX - minX) / width)
                   << coordinate((maxY - minY) / height);
        } else {
            for (const QPointF& point : points)
                fields << coordinate(point.x() / width)
                       << coordinate(point.y() / height);
        }
        output.text += fields.join(QLatin1Char(' ')).toUtf8() + '\n';
        ++output.count;
    }
    return output;
}

YoloConversion convertDota(const QJsonObject& document) {
    YoloConversion output;
    const int width = document.value(QStringLiteral("imageWidth")).toInt();
    const int height = document.value(QStringLiteral("imageHeight")).toInt();
    if (width <= 0 || height <= 0) {
        output.error = QStringLiteral("Invalid image dimensions");
        return output;
    }
    for (const QJsonValue& value :
         document.value(QStringLiteral("shapes")).toArray()) {
        const QJsonObject shape = value.toObject();
        if (shape.value(QStringLiteral("shape_type")).toString() !=
            QStringLiteral("rotation"))
            continue;
        const QVector<QPointF> points = pointsFor(shape);
        if (points.size() != 4 ||
            shape.value(QStringLiteral("label")).toString().isEmpty() ||
            std::any_of(points.cbegin(), points.cend(),
                        [width, height](const QPointF& point) {
                            return point.x() < 0 || point.y() < 0 ||
                                   point.x() > width || point.y() > height;
                        }))
            continue;
        QStringList fields;
        for (const QPointF& point : points)
            fields << QString::number(point.x(), 'g', 15)
                   << QString::number(point.y(), 'g', 15);
        fields << shape.value(QStringLiteral("label")).toString()
               << QString::number(
                          shape.value(QStringLiteral("difficult")).toBool()
                                  ? 1
                                  : 0);
        output.text += fields.join(QLatin1Char(' ')).toUtf8() + '\n';
        ++output.count;
    }
    return output;
}

YoloConversion convertVoc(const QJsonObject& document,
                          const QString& imageName,
                          const QString& folder,
                          bool polygons) {
    YoloConversion output;
    const int width = document.value(QStringLiteral("imageWidth")).toInt();
    const int height = document.value(QStringLiteral("imageHeight")).toInt();
    if (width <= 0 || height <= 0) {
        output.error = QStringLiteral("Invalid image dimensions");
        return output;
    }
    QXmlStreamWriter xml(&output.text);
    xml.setAutoFormatting(true);
    xml.writeStartDocument();
    xml.writeStartElement(QStringLiteral("annotation"));
    xml.writeTextElement(QStringLiteral("folder"), folder);
    xml.writeTextElement(QStringLiteral("filename"), imageName);
    xml.writeStartElement(QStringLiteral("size"));
    xml.writeTextElement(QStringLiteral("width"), QString::number(width));
    xml.writeTextElement(QStringLiteral("height"), QString::number(height));
    xml.writeTextElement(QStringLiteral("depth"), QStringLiteral("3"));
    xml.writeEndElement();
    xml.writeStartElement(QStringLiteral("source"));
    xml.writeTextElement(
            QStringLiteral("database"),
            QStringLiteral("https://github.com/CVHub520/X-AnyLabeling"));
    xml.writeEndElement();
    for (const QJsonValue& value :
         document.value(QStringLiteral("shapes")).toArray()) {
        const QJsonObject shape = value.toObject();
        const QString type =
                shape.value(QStringLiteral("shape_type")).toString();
        if (type != QStringLiteral("rectangle") &&
            !(polygons && type == QStringLiteral("polygon")))
            continue;
        QVector<QPointF> points = pointsFor(shape);
        if ((type == QStringLiteral("rectangle") && points.size() != 2 &&
             points.size() != 4) ||
            (type == QStringLiteral("polygon") && points.size() < 3))
            continue;
        for (QPointF& point : points) {
            point.setX(std::clamp(point.x(), 0.0, double(width)));
            point.setY(std::clamp(point.y(), 0.0, double(height)));
        }
        double minX = points[0].x(), maxX = minX, minY = points[0].y(),
               maxY = minY;
        for (const QPointF& point : points) {
            minX = std::min(minX, point.x());
            minY = std::min(minY, point.y());
            maxX = std::max(maxX, point.x());
            maxY = std::max(maxY, point.y());
        }
        if (minX == maxX || minY == maxY) continue;
        xml.writeStartElement(QStringLiteral("object"));
        xml.writeTextElement(QStringLiteral("name"),
                             shape.value(QStringLiteral("label")).toString());
        xml.writeTextElement(QStringLiteral("pose"),
                             QStringLiteral("Unspecified"));
        xml.writeTextElement(QStringLiteral("truncated"), QStringLiteral("0"));
        xml.writeTextElement(QStringLiteral("occluded"), QStringLiteral("0"));
        xml.writeTextElement(QStringLiteral("difficult"),
                             shape.value(QStringLiteral("difficult")).toBool()
                                     ? QStringLiteral("1")
                                     : QStringLiteral("0"));
        xml.writeStartElement(QStringLiteral("bndbox"));
        xml.writeTextElement(QStringLiteral("xmin"),
                             QString::number(int(minX)));
        xml.writeTextElement(QStringLiteral("ymin"),
                             QString::number(int(minY)));
        xml.writeTextElement(QStringLiteral("xmax"),
                             QString::number(int(maxX)));
        xml.writeTextElement(QStringLiteral("ymax"),
                             QString::number(int(maxY)));
        xml.writeEndElement();
        if (type == QStringLiteral("polygon")) {
            xml.writeStartElement(QStringLiteral("polygon"));
            for (int i = 0; i < points.size(); ++i) {
                xml.writeTextElement(QStringLiteral("x%1").arg(i + 1),
                                     QString::number(points[i].x(), 'g', 15));
                xml.writeTextElement(QStringLiteral("y%1").arg(i + 1),
                                     QString::number(points[i].y(), 'g', 15));
            }
            xml.writeEndElement();
        }
        xml.writeEndElement();
        ++output.count;
    }
    xml.writeEndElement();
    xml.writeEndDocument();
    return output;
}

CocoConversion convertCoco(const QVector<CocoImage>& images,
                           const QStringList& classes,
                           bool polygons) {
    CocoConversion output;
    if (classes.isEmpty()) {
        output.error = QStringLiteral("The classes list is empty");
        return output;
    }
    QStringList categoryNames = classes;
    if (polygons) {
        if (categoryNames.first() == QStringLiteral("__ignore__"))
            categoryNames.removeFirst();
        if (categoryNames.isEmpty() ||
            categoryNames.first() != QStringLiteral("_background_"))
            categoryNames.prepend(QStringLiteral("_background_"));
    }
    QJsonArray categories, cocoImages, annotations;
    for (int i = 0; i < categoryNames.size(); ++i) {
        categories.append(
                QJsonObject{{QStringLiteral("id"), polygons ? i : i + 1},
                            {QStringLiteral("name"), categoryNames[i]},
                            {QStringLiteral("supercategory"),
                             polygons ? QJsonValue(QJsonValue::Null)
                                      : QJsonValue(QString())}});
    }
    int imageId = 0, annotationId = 0;
    for (const CocoImage& image : images) {
        const int width =
                image.document.value(QStringLiteral("imageWidth")).toInt();
        const int height =
                image.document.value(QStringLiteral("imageHeight")).toInt();
        if (width <= 0 || height <= 0) {
            output.error = QStringLiteral("Invalid image dimensions: %1")
                                   .arg(image.fileName);
            return output;
        }
        cocoImages.append(
                QJsonObject{{QStringLiteral("license"), 0},
                            {QStringLiteral("url"), QJsonValue::Null},
                            {QStringLiteral("file_name"), image.fileName},
                            {QStringLiteral("height"), height},
                            {QStringLiteral("width"), width},
                            {QStringLiteral("date_captured"), QJsonValue::Null},
                            {QStringLiteral("id"), imageId}});
        struct Group {
            QString label;
            bool difficult = false;
            QVector<QVector<QPointF>> polygons;
            QVector<QImage> masks;
        };
        QVector<Group> groups;
        QHash<QString, int> groupIndices;
        int shapeIndex = 0;
        for (const QJsonValue& value :
             image.document.value(QStringLiteral("shapes")).toArray()) {
            ++shapeIndex;
            if (!value.isObject()) continue;
            const QJsonObject shape = value.toObject();
            const QString type =
                    shape.value(QStringLiteral("shape_type")).toString();
            if (type != (polygons ? QStringLiteral("polygon")
                                  : QStringLiteral("rectangle")))
                continue;
            const QString label =
                    shape.value(QStringLiteral("label")).toString();
            const int classId = categoryNames.indexOf(label);
            if (classId < 0 ||
                (polygons && label == QStringLiteral("__ignore__")))
                continue;
            QImage exactMask;
            const QString encodedMask =
                    shape.value(QStringLiteral("acloudviewer_mask_png"))
                            .toString();
            if (polygons && !encodedMask.isEmpty()) {
                exactMask = QImage::fromData(
                        QByteArray::fromBase64(encodedMask.toLatin1()), "PNG");
                if (exactMask.size() != QSize(width, height)) {
                    output.error =
                            QStringLiteral(
                                    "Invalid instance mask dimensions: %1")
                                    .arg(image.fileName);
                    return output;
                }
            }
            QVector<QPointF> points = pointsFor(shape);
            if (points.size() < (polygons ? 3 : 2) && exactMask.isNull())
                continue;
            for (QPointF& point : points) {
                point.setX(std::clamp(point.x(), 0.0, double(width)));
                point.setY(std::clamp(point.y(), 0.0, double(height)));
            }
            if (!polygons) {
                const QJsonArray box = boxFor(points, false);
                const double area = box[2].toDouble() * box[3].toDouble();
                if (area <= 0) continue;
                annotations.append(QJsonObject{
                        {QStringLiteral("id"), annotationId++},
                        {QStringLiteral("image_id"), imageId},
                        {QStringLiteral("category_id"), classId + 1},
                        {QStringLiteral("bbox"), box},
                        {QStringLiteral("area"), area},
                        {QStringLiteral("iscrowd"), 0},
                        {QStringLiteral("ignore"),
                         int(shape.value(QStringLiteral("difficult"))
                                     .toBool())},
                        {QStringLiteral("segmentation"), QJsonArray()}});
                continue;
            }
            const QJsonValue id = shape.value(QStringLiteral("group_id"));
            const QString key =
                    label + QLatin1Char('\n') +
                    (id.isDouble()
                             ? QString::number(id.toInt())
                             : QStringLiteral("shape:%1").arg(shapeIndex));
            int index = groupIndices.value(key, -1);
            if (index < 0) {
                index = groups.size();
                groupIndices.insert(key, index);
                groups.append(
                        {label,
                         shape.value(QStringLiteral("difficult")).toBool(),
                         {},
                         {}});
            }
            groups[index].polygons.append(points);
            groups[index].masks.append(exactMask);
        }
        for (const Group& group : groups) {
            const bool hasExactMask = std::any_of(
                    group.masks.cbegin(), group.masks.cend(),
                    [](const QImage& mask) { return !mask.isNull(); });
            if (hasExactMask) {
                QImage mask(width, height, QImage::Format_ARGB32);
                if (mask.isNull()) {
                    output.error =
                            QStringLiteral("Unable to allocate COCO mask");
                    return output;
                }
                mask.fill(Qt::black);
                for (int i = 0; i < group.polygons.size(); ++i) {
                    if (!group.masks[i].isNull()) {
                        const QImage& exact = group.masks[i];
                        for (int y = 0; y < height; ++y)
                            for (int x = 0; x < width; ++x)
                                if (exact.pixelColor(x, y).alpha() > 0)
                                    mask.setPixel(x, y, qRgb(255, 255, 255));
                    } else {
                        QPainter painter(&mask);
                        painter.setPen(Qt::NoPen);
                        painter.setBrush(Qt::white);
                        painter.drawPolygon(QPolygonF(group.polygons[i]));
                    }
                }
                QJsonArray counts;
                int active = 0, runLength = 0, area = 0;
                int minX = width, minY = height, maxX = -1, maxY = -1;
                for (int x = 0; x < width; ++x) {
                    for (int y = 0; y < height; ++y) {
                        const int pixel = qRed(mask.pixel(x, y)) > 0;
                        if (pixel != active) {
                            counts.append(runLength);
                            runLength = 0;
                            active = pixel;
                        }
                        ++runLength;
                        if (pixel) {
                            ++area;
                            minX = std::min(minX, x);
                            minY = std::min(minY, y);
                            maxX = std::max(maxX, x);
                            maxY = std::max(maxY, y);
                        }
                    }
                }
                counts.append(runLength);
                if (area == 0) continue;
                annotations.append(QJsonObject{
                        {QStringLiteral("id"), annotationId++},
                        {QStringLiteral("image_id"), imageId},
                        {QStringLiteral("category_id"),
                         categoryNames.indexOf(group.label)},
                        {QStringLiteral("segmentation"),
                         QJsonObject{{QStringLiteral("size"),
                                      QJsonArray{height, width}},
                                     {QStringLiteral("counts"), counts}}},
                        {QStringLiteral("area"), area},
                        {QStringLiteral("bbox"),
                         QJsonArray{minX, minY, maxX - minX + 1,
                                    maxY - minY + 1}},
                        {QStringLiteral("iscrowd"), 0},
                        {QStringLiteral("ignore"), int(group.difficult)}});
                continue;
            }
            QVector<QPointF> all;
            QJsonArray segments;
            for (const QVector<QPointF>& polygon : group.polygons) {
                QJsonArray coordinates;
                for (const QPointF& point : polygon) {
                    all.append(point);
                    coordinates.append(point.x());
                    coordinates.append(point.y());
                }
                segments.append(coordinates);
            }
            annotations.append(QJsonObject{
                    {QStringLiteral("id"), annotationId++},
                    {QStringLiteral("image_id"), imageId},
                    {QStringLiteral("category_id"),
                     categoryNames.indexOf(group.label)},
                    {QStringLiteral("segmentation"), segments},
                    {QStringLiteral("area"), rasterArea(group.polygons)},
                    {QStringLiteral("bbox"), boxFor(all, true)},
                    {QStringLiteral("iscrowd"), 0},
                    {QStringLiteral("ignore"), int(group.difficult)}});
        }
        ++imageId;
    }
    const QDate today = QDate::currentDate();
    QJsonObject info{
            {QStringLiteral("year"), today.year()},
            {QStringLiteral("version"), annotationFormatVersion()},
            {QStringLiteral("description"),
             QStringLiteral("COCO Label Conversion")},
            {QStringLiteral("contributor"), QStringLiteral("CVHub")},
            {QStringLiteral("url"),
             QStringLiteral("https://github.com/CVHub520/X-AnyLabeling")},
            {QStringLiteral("date_created"), today.toString(Qt::ISODate)}};
    output.document = {
            {QStringLiteral("info"), info},
            {QStringLiteral("licenses"),
             QJsonArray{QJsonObject{
                     {QStringLiteral("id"), 1},
                     {QStringLiteral("url"),
                      QStringLiteral(
                              "https://www.gnu.org/licenses/gpl-3.0.html")},
                     {QStringLiteral("name"),
                      QStringLiteral(
                              "GNU GENERAL PUBLIC LICENSE Version 3")}}}},
            {QStringLiteral("categories"), categories},
            {QStringLiteral("images"), cocoImages},
            {QStringLiteral("annotations"), annotations}};
    if (polygons)
        output.document.insert(QStringLiteral("type"),
                               QStringLiteral("instances"));
    output.count = annotations.size();
    return output;
}
