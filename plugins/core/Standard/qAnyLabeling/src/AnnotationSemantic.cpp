// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationSemantic.h"

#include <QByteArray>
#include <QObject>
#include <algorithm>
#include <limits>
#include <utility>

#include "MaskPolygon.h"

SemanticConversion convertSemanticClassMap(const uint8_t* data,
                                           int width,
                                           int height,
                                           size_t rowStride,
                                           const QStringList& classNames) {
    SemanticConversion result;
    if (!data || width <= 0 || height <= 0 || rowStride < size_t(width) ||
        rowStride > std::numeric_limits<size_t>::max() / size_t(height) ||
        size_t(width) >
                size_t(std::numeric_limits<int>::max()) / size_t(height) ||
        classNames.isEmpty() || classNames.size() > 256) {
        result.error = QObject::tr("Invalid semantic class map");
        return result;
    }

    QVector<bool> present(classNames.size(), false);
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = data + size_t(y) * rowStride;
        for (int x = 0; x < width; ++x) {
            if (row[x] >= classNames.size()) {
                result.error = QObject::tr("Semantic class ID is out of range");
                return result;
            }
            present[row[x]] = true;
        }
    }

    const int pixelCount = width * height;
    for (int id = 0; id < present.size(); ++id) {
        if (!present[id]) continue;
        QByteArray binary(pixelCount, char(0));
        QImage mask(width, height, QImage::Format_ARGB32_Premultiplied);
        if (mask.isNull()) {
            result.shapes.clear();
            result.error = QObject::tr("Unable to allocate semantic mask");
            return result;
        }
        int left = width, top = height, right = -1, bottom = -1;
        // Each class is scanned separately to keep only one temporary binary
        // plane in memory; the full masks are required by the editing model.
        for (int y = 0; y < height; ++y) {
            const uint8_t* source = data + size_t(y) * rowStride;
            QRgb* target = reinterpret_cast<QRgb*>(mask.scanLine(y));
            for (int x = 0; x < width; ++x) {
                const bool selected = source[x] == id;
                binary[y * width + x] = selected ? char(255) : char(0);
                target[x] = selected ? qRgba(0, 78, 110, 110) : 0;
                if (!selected) continue;
                left = std::min(left, x);
                top = std::min(top, y);
                right = std::max(right, x);
                bottom = std::max(bottom, y);
            }
        }
        AnnotationShape shape;
        shape.type = QStringLiteral("polygon");
        shape.label = classNames[id].isEmpty() ? QObject::tr("class %1").arg(id)
                                               : classNames[id];
        shape.mask = std::move(mask);
        shape.points = largestMaskPolygon(
                reinterpret_cast<const uint8_t*>(binary.constData()), width,
                height, size_t(width));
        if (shape.points.size() < 3) {
            shape.points = {QPointF(left, top), QPointF(right + 1, top),
                            QPointF(right + 1, bottom + 1),
                            QPointF(left, bottom + 1)};
        }
        shape.extra.insert(QStringLiteral("class_id"), id);
        result.shapes.append(std::move(shape));
    }
    return result;
}
