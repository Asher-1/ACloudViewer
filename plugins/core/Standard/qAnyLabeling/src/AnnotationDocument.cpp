// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationDocument.h"

#include <QBuffer>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QObject>

AnnotationDocument loadAnnotationDocument(const QString& filename,
                                          const QString& outputDirectory,
                                          bool autoTransform) {
    AnnotationDocument result;
    const QFileInfo input(filename);
    const bool jsonInput = input.suffix().compare(QStringLiteral("json"),
                                                  Qt::CaseInsensitive) == 0;
    result.labelPath =
            jsonInput ? input.absoluteFilePath()
                      : QDir(outputDirectory.isEmpty() ? input.absolutePath()
                                                       : outputDirectory)
                                .absoluteFilePath(input.completeBaseName() +
                                                  QStringLiteral(".json"));
    result.imagePath = jsonInput ? QString() : input.absoluteFilePath();
    QByteArray embedded;
    bool hasEmbedded = false;
    if (jsonInput || QFileInfo::exists(result.labelPath)) {
        QFile labels(result.labelPath);
        if (!labels.open(QIODevice::ReadOnly)) {
            result.error = QObject::tr("Unable to read %1: %2")
                                   .arg(result.labelPath, labels.errorString());
            return result;
        }
        QJsonParseError error;
        const auto parsed = QJsonDocument::fromJson(labels.readAll(), &error);
        if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
            !parsed.object().value(QStringLiteral("shapes")).isArray() ||
            parsed.object().value(QStringLiteral("type")).toString() ==
                    QStringLiteral("video_classification")) {
            result.error = QObject::tr("Invalid image annotation: %1")
                                   .arg(result.labelPath);
            return result;
        }
        result.json = parsed.object();
        if (jsonInput) {
            const auto stored = result.json.value(QStringLiteral("imagePath"));
            if (!stored.isString() || stored.toString().isEmpty()) {
                result.error = QObject::tr("Missing imagePath in %1")
                                       .arg(result.labelPath);
                return result;
            }
            const QDir directory(QFileInfo(result.labelPath).absolutePath());
            result.imagePath = directory.absoluteFilePath(stored.toString());
            if (!QFileInfo(result.imagePath).isFile()) {
                const auto local = directory.absoluteFilePath(
                        QFileInfo(stored.toString()).fileName());
                if (QFileInfo(local).isFile()) result.imagePath = local;
            }
        }
        const auto data = result.json.value(QStringLiteral("imageData"));
        hasEmbedded = !data.isNull() && !data.isUndefined();
        if (hasEmbedded) {
            if (data.isString())
                embedded = QByteArray::fromBase64(data.toString().toLatin1());
            if (embedded.isEmpty()) {
                result.error = QObject::tr("Invalid embedded image in %1")
                                       .arg(result.labelPath);
                return result;
            }
        }
    }
    QBuffer buffer(&embedded);
    QImageReader reader;
    if (hasEmbedded) {
        buffer.open(QIODevice::ReadOnly);
        reader.setDevice(&buffer);
    } else {
        reader.setFileName(result.imagePath);
    }
    reader.setAutoTransform(autoTransform);
    result.image = reader.read();
    if (result.image.isNull()) {
        result.error = QObject::tr("Unable to decode %1: %2")
                               .arg(result.imagePath, reader.errorString());
        return result;
    }
    result.imagePath = QDir::cleanPath(result.imagePath);
    return result;
}
