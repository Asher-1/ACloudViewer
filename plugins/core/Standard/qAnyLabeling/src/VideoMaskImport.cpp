// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoMaskImport.h"

#include <QBuffer>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QtGlobal>
#include <algorithm>
#include <cmath>
#include <limits>

#include "MaskPolygon.h"

namespace {
QJsonObject emptyDocument(const Sam3ArchiveImage& image) {
    return {{QStringLiteral("version"), QStringLiteral("4.0.6")},
            {QStringLiteral("flags"), QJsonObject()},
            {QStringLiteral("shapes"), QJsonArray()},
            {QStringLiteral("imagePath"), QFileInfo(image.filePath).fileName()},
            {QStringLiteral("imageData"), QJsonValue::Null},
            {QStringLiteral("imageWidth"), image.size.width()},
            {QStringLiteral("imageHeight"), image.size.height()}};
}

bool maskValue(const QImage& image, int x, int y) {
    if (image.format() == QImage::Format_Grayscale8 ||
        image.format() == QImage::Format_Alpha8) {
        return image.constScanLine(y)[x] != 0;
    }
    const QRgb pixel = image.pixel(x, y);
    return image.hasAlphaChannel() ? qAlpha(pixel) != 0 : qGray(pixel) != 0;
}

bool loadMask(const QString& root,
              const QString& relative,
              const QSize& size,
              QImage* exact,
              QString* error) {
    if (relative.isEmpty() || QFileInfo(relative).isAbsolute()) {
        *error = QStringLiteral("mask path must be relative");
        return false;
    }
    const QFileInfo rootInfo(root);
    const QString rootCanonical =
            QDir::fromNativeSeparators(rootInfo.canonicalFilePath());
    const QFileInfo maskInfo(QDir(root).filePath(relative));
    const QString maskCanonical =
            QDir::fromNativeSeparators(maskInfo.canonicalFilePath());
    if (rootCanonical.isEmpty() || maskCanonical.isEmpty() ||
        !maskCanonical.startsWith(rootCanonical + QLatin1Char('/'))) {
        *error = QStringLiteral("mask path escapes archive directory");
        return false;
    }
    QImage source(maskCanonical);
    if (source.isNull() || source.size() != size) {
        *error = QStringLiteral("mask is missing or has the wrong dimensions");
        return false;
    }
    QImage result(size, QImage::Format_ARGB32);
    result.fill(qRgba(0, 0, 0, 0));
    for (int y = 0; y < size.height(); ++y) {
        QRgb* row = reinterpret_cast<QRgb*>(result.scanLine(y));
        for (int x = 0; x < size.width(); ++x)
            row[x] = maskValue(source, x, y) ? qRgba(255, 255, 255, 255)
                                             : qRgba(0, 0, 0, 0);
    }
    *exact = result;
    return true;
}

QString encodedPng(const QImage& image) {
    QByteArray bytes;
    QBuffer buffer(&bytes);
    if (!buffer.open(QIODevice::WriteOnly) || !image.save(&buffer, "PNG"))
        return {};
    return QString::fromLatin1(bytes.toBase64());
}
}  // namespace

Sam3ArchiveImport visitSam3Archive(
        const QByteArray& bytes,
        const QString& archiveDirectory,
        const QVector<Sam3ArchiveImage>& images,
        const std::function<bool(int, const QJsonObject&)>& visitor,
        const QString& labelPrefix) {
    Sam3ArchiveImport result;
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        result.error = QStringLiteral("Invalid SAM3 tracking manifest JSON");
        return result;
    }
    const QJsonObject manifest = document.object();
    if (manifest.value(QStringLiteral("format")).toString() !=
        QStringLiteral("acloudviewer_sam3_tracking_v1")) {
        result.error =
                QStringLiteral("Unsupported SAM3 tracking archive format");
        return result;
    }
    if (archiveDirectory.isEmpty() || !QDir(archiveDirectory).exists()) {
        result.error = QStringLiteral("SAM3 archive directory does not exist");
        return result;
    }
    const QJsonArray frames =
            manifest.value(QStringLiteral("frames")).toArray();
    if (frames.isEmpty()) {
        result.error =
                QStringLiteral("SAM3 tracking manifest contains no frames");
        return result;
    }
    const int archiveWidth = manifest.value(QStringLiteral("width")).toInt(-1);
    const int archiveHeight =
            manifest.value(QStringLiteral("height")).toInt(-1);
    const int totalFrames =
            manifest.value(QStringLiteral("total_frames")).toInt(-1);
    const int startFrame =
            manifest.value(QStringLiteral("start_frame")).toInt(-1);
    if (archiveWidth <= 0 || archiveHeight <= 0) {
        result.error = QStringLiteral("Invalid SAM3 tracking frame dimensions");
        return result;
    }
    if (totalFrames <= 0 || startFrame < 0 || startFrame >= totalFrames ||
        frames.size() > totalFrames - startFrame ||
        images.size() != totalFrames ||
        qint64(archiveWidth) * archiveHeight >
                std::numeric_limits<int>::max()) {
        result.error = QStringLiteral(
                "SAM3 archive needs a complete, matching video frame folder");
        return result;
    }
    for (int frameIndex = 0; frameIndex < totalFrames; ++frameIndex) {
        const Sam3ArchiveImage& image = images.at(frameIndex);
        if (image.filePath.isEmpty() || !image.size.isValid()) {
            result.error =
                    QStringLiteral("Invalid source image for SAM3 frame");
            return result;
        }
        if (image.size != QSize(archiveWidth, archiveHeight)) {
            result.error =
                    QStringLiteral(
                            "Frame %1 dimensions do not match the SAM3 archive")
                            .arg(frameIndex);
            return result;
        }
        QJsonArray shapes;
        const int archiveIndex = frameIndex - startFrame;
        if (archiveIndex >= 0 && archiveIndex < frames.size()) {
            const QJsonObject frame = frames.at(archiveIndex).toObject();
            if (frame.value(QStringLiteral("frame")).toInt(-1) != frameIndex ||
                !frame.value(QStringLiteral("instances")).isArray()) {
                result.error = QStringLiteral(
                        "Invalid SAM3 frame sequence or instances");
                return result;
            }
            for (const QJsonValue& instanceValue :
                 frame.value(QStringLiteral("instances")).toArray()) {
                const QJsonObject instance = instanceValue.toObject();
                const QJsonValue idValue = instance.value(QStringLiteral("id"));
                const bool idOk =
                        idValue.isDouble() && idValue.toDouble() >= 0.0 &&
                        idValue.toDouble() == std::floor(idValue.toDouble());
                const int id = idOk ? idValue.toInt(-1) : -1;
                const QString maskName =
                        instance.value(QStringLiteral("mask")).toString();
                if (!idOk || id < 0 || maskName.isEmpty()) {
                    result.error =
                            QStringLiteral("Invalid SAM3 instance entry");
                    return result;
                }
                QImage exactMask;
                QString maskError;
                if (!loadMask(archiveDirectory, maskName, image.size,
                              &exactMask, &maskError)) {
                    result.error = QStringLiteral("Frame %1 instance %2: %3")
                                           .arg(frameIndex)
                                           .arg(id)
                                           .arg(maskError);
                    return result;
                }
                QByteArray binary(size_t(image.size.width()) *
                                          size_t(image.size.height()),
                                  0);
                for (int y = 0; y < image.size.height(); ++y) {
                    const QRgb* row = reinterpret_cast<const QRgb*>(
                            exactMask.constScanLine(y));
                    for (int x = 0; x < image.size.width(); ++x)
                        binary[y * image.size.width() + x] =
                                qAlpha(row[x]) ? char(1) : char(0);
                }
                const auto contours = maskPolygons(
                        reinterpret_cast<const uint8_t*>(binary.constData()),
                        image.size.width(), image.size.height(),
                        size_t(image.size.width()));
                QJsonArray points;
                QVector<QPointF> outline = contours.isEmpty()
                                                   ? QVector<QPointF>()
                                                   : contours.first();
                if (outline.isEmpty()) {
                    int x0 = image.size.width(), y0 = image.size.height();
                    int x1 = -1, y1 = -1;
                    for (int y = 0; y < image.size.height(); ++y) {
                        for (int x = 0; x < image.size.width(); ++x) {
                            if (!binary[y * image.size.width() + x]) continue;
                            x0 = std::min(x0, x);
                            y0 = std::min(y0, y);
                            x1 = std::max(x1, x);
                            y1 = std::max(y1, y);
                        }
                    }
                    if (x1 < x0 || y1 < y0) continue;
                    outline = {QPointF(x0, y0), QPointF(x1 + 1, y0),
                               QPointF(x1 + 1, y1 + 1), QPointF(x0, y1 + 1)};
                }
                for (const QPointF& point : outline)
                    points.append(QJsonArray{point.x(), point.y()});
                QJsonObject shape;
                shape.insert(QStringLiteral("label"),
                             QString(labelPrefix + QString::number(id)));
                shape.insert(QStringLiteral("shape_type"),
                             QStringLiteral("polygon"));
                shape.insert(QStringLiteral("points"), points);
                shape.insert(QStringLiteral("group_id"), id);
                shape.insert(QStringLiteral("flags"), QJsonObject());
                shape.insert(QStringLiteral("description"), QString());
                shape.insert(QStringLiteral("difficult"), false);
                shape.insert(QStringLiteral("sam3_instance_id"), id);
                const QString png = encodedPng(exactMask);
                if (png.isEmpty()) {
                    result.error =
                            QStringLiteral("Cannot encode SAM3 instance mask");
                    return result;
                }
                shape.insert(QStringLiteral("acloudviewer_mask_png"), png);
                if (instance.value(QStringLiteral("score")).isDouble())
                    shape.insert(QStringLiteral("score"),
                                 instance.value(QStringLiteral("score")));
                if (instance.value(QStringLiteral("bbox")).isArray())
                    shape.insert(QStringLiteral("sam3_bbox"),
                                 instance.value(QStringLiteral("bbox")));
                shapes.append(shape);
                ++result.count;
            }
        }
        QJsonObject output;
        if (!shapes.isEmpty()) {
            output = emptyDocument(image);
            output.insert(QStringLiteral("shapes"), shapes);
            output.insert(QStringLiteral("sam3_tracking_archive"), true);
        }
        if (visitor && !visitor(frameIndex, output)) {
            result.error = QStringLiteral("Cannot process SAM3 frame %1")
                                   .arg(frameIndex);
            return result;
        }
    }
    return result;
}

Sam3ArchiveImport importSam3Archive(const QByteArray& bytes,
                                    const QString& archiveDirectory,
                                    const QVector<Sam3ArchiveImage>& images,
                                    const QString& labelPrefix) {
    QHash<QString, QJsonObject> documents;
    Sam3ArchiveImport result = visitSam3Archive(
            bytes, archiveDirectory, images,
            [&images, &documents](int index, const QJsonObject& document) {
                if (!document.isEmpty())
                    documents.insert(QFileInfo(images.at(index).filePath)
                                             .absoluteFilePath(),
                                     document);
                return true;
            },
            labelPrefix);
    if (result.error.isEmpty()) result.documents = std::move(documents);
    return result;
}
