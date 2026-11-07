// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationMask.h"

#include <QHash>
#include <QImage>
#include <QJsonArray>
#include <QPainter>
#include <QPolygon>
#include <QVector>
#include <algorithm>
#include <cmath>
#include <vector>

#if defined(QANYLABELING_HAS_OPENCV_MASK)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {
void fillPolygon(QImage& image, const QPolygon& polygon, const QColor& color) {
    if (polygon.size() < 3) return;
#if defined(QANYLABELING_HAS_OPENCV_MASK)
    const QRect bounds = polygon.boundingRect().intersected(image.rect());
    if (bounds.isEmpty()) return;
    std::vector<cv::Point> points;
    points.reserve(static_cast<size_t>(polygon.size()));
    for (const QPoint& point : polygon)
        points.emplace_back(point.x() - bounds.left(),
                            point.y() - bounds.top());
    cv::Mat coverage(bounds.height(), bounds.width(), CV_8UC1, cv::Scalar(0));
    const std::vector<std::vector<cv::Point>> contours{points};
    cv::fillPoly(coverage, contours, cv::Scalar(255));
    for (int y = 0; y < bounds.height(); ++y) {
        const uchar* maskRow = coverage.ptr<uchar>(y);
        QRgb* imageRow =
                reinterpret_cast<QRgb*>(image.scanLine(y + bounds.top()));
        for (int x = 0; x < bounds.width(); ++x) {
            if (maskRow[x] != 0)
                imageRow[x + bounds.left()] =
                        qRgb(color.red(), color.green(), color.blue());
        }
    }
#else
    QPainter painter(&image);
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    painter.drawPolygon(polygon);
#endif
}
}  // namespace

MaskConversion convertMask(const QJsonObject& document,
                           const QJsonObject& colorMap) {
    MaskConversion result;
    const int width = document.value(QStringLiteral("imageWidth")).toInt();
    const int height = document.value(QStringLiteral("imageHeight")).toInt();
    if (width <= 0 || height <= 0) {
        result.error = QStringLiteral("Invalid image dimensions");
        return result;
    }
    const QString type = colorMap.value(QStringLiteral("type")).toString();
    const bool gray = type == QStringLiteral("grayscale");
    if (!gray && type != QStringLiteral("rgb")) {
        result.error = QStringLiteral("Mask type must be grayscale or rgb");
        return result;
    }
    const QJsonObject colors =
            colorMap.value(QStringLiteral("colors")).toObject();
    const QJsonValue priorityValue =
            colorMap.value(QStringLiteral("label_priority"));
    if (colors.isEmpty() ||
        (!priorityValue.isUndefined() && !priorityValue.isObject())) {
        result.error = QStringLiteral("Invalid mask colors or label_priority");
        return result;
    }
    const QJsonObject priorities = priorityValue.toObject();
    QHash<QString, QColor> palette;
    for (auto it = colors.constBegin(); it != colors.constEnd(); ++it) {
        QVector<int> channels;
        if (gray) {
            if (it.value().isDouble()) channels.append(it.value().toInt(-1));
        } else if (it.value().isArray()) {
            for (const QJsonValue& channel : it.value().toArray())
                channels.append(channel.isDouble() ? channel.toInt(-1) : -1);
        }
        if (channels.size() != (gray ? 1 : 3) ||
            std::any_of(channels.cbegin(), channels.cend(),
                        [](int value) { return value < 0 || value > 255; })) {
            result.error =
                    QStringLiteral("Invalid mask color for %1").arg(it.key());
            return result;
        }
        palette.insert(it.key(),
                       gray ? QColor(channels[0], channels[0], channels[0])
                            : QColor(channels[0], channels[1], channels[2]));
    }
    for (auto it = priorities.constBegin(); it != priorities.constEnd(); ++it) {
        if (!palette.contains(it.key()) || !it.value().isDouble() ||
            it.value().toDouble() != it.value().toInt()) {
            result.error = QStringLiteral("Invalid mask priority for %1")
                                   .arg(it.key());
            return result;
        }
    }
    struct Layer {
        int priority;
        int index;
        QColor color;
        QPolygon polygon;
        QImage mask;
    };
    QVector<Layer> layers;
    int index = 0;
    for (const QJsonValue& value :
         document.value(QStringLiteral("shapes")).toArray()) {
        ++index;
        const QJsonObject shape = value.toObject();
        if (shape.value(QStringLiteral("shape_type")).toString() !=
            QStringLiteral("polygon"))
            continue;
        const QString label = shape.value(QStringLiteral("label")).toString();
        if (!palette.contains(label)) continue;
        QImage exactMask;
        const QString encoded =
                shape.value(QStringLiteral("acloudviewer_mask_png")).toString();
        if (!encoded.isEmpty()) {
            exactMask =
                    QImage::fromData(QByteArray::fromBase64(encoded.toLatin1()),
                                     "PNG")
                            .convertToFormat(QImage::Format_ARGB32);
            if (exactMask.isNull()) {
                result.error =
                        QStringLiteral("Invalid instance mask in shape %1")
                                .arg(index);
                return result;
            }
            if (exactMask.size() != QSize(width, height))
                exactMask = exactMask.scaled(QSize(width, height),
                                             Qt::IgnoreAspectRatio,
                                             Qt::FastTransformation);
        }
        QPolygon polygon;
        for (const QJsonValue& pointValue :
             shape.value(QStringLiteral("points")).toArray()) {
            const QJsonArray pair = pointValue.toArray();
            if (pair.size() != 2 || !pair[0].isDouble() ||
                !pair[1].isDouble() || !std::isfinite(pair[0].toDouble()) ||
                !std::isfinite(pair[1].toDouble())) {
                result.error =
                        QStringLiteral("Invalid polygon coordinate in shape %1")
                                .arg(index);
                return result;
            }
            polygon.append(QPoint(
                    int(std::clamp(pair[0].toDouble(), 0.0, double(width))),
                    int(std::clamp(pair[1].toDouble(), 0.0, double(height)))));
        }
        if (polygon.size() >= 3 || !exactMask.isNull())
            layers.append({priorities.value(label).toInt(), index,
                           palette.value(label), polygon, exactMask});
    }
    std::sort(layers.begin(), layers.end(), [](const Layer& a, const Layer& b) {
        return a.priority == b.priority ? a.index < b.index
                                        : a.priority < b.priority;
    });
    result.image = QImage(width, height, QImage::Format_RGB32);
    if (result.image.isNull()) {
        result.error = QStringLiteral("Unable to allocate mask image");
        return result;
    }
    result.image.fill(Qt::black);
    for (const Layer& layer : layers) {
        if (layer.mask.isNull()) {
            fillPolygon(result.image, layer.polygon, layer.color);
        } else {
            QImage overlay(layer.mask.size(), QImage::Format_ARGB32);
            for (int y = 0; y < overlay.height(); ++y) {
                const QRgb* maskRow = reinterpret_cast<const QRgb*>(
                        layer.mask.constScanLine(y));
                QRgb* overlayRow = reinterpret_cast<QRgb*>(overlay.scanLine(y));
                for (int x = 0; x < overlay.width(); ++x)
                    overlayRow[x] = qAlpha(maskRow[x])
                                            ? qRgb(layer.color.red(),
                                                   layer.color.green(),
                                                   layer.color.blue())
                                            : qRgba(0, 0, 0, 0);
            }
            QPainter painter(&result.image);
            painter.drawImage(0, 0, overlay);
        }
    }
    if (gray)
        result.image = result.image.convertToFormat(QImage::Format_Grayscale8);
    return result;
}
