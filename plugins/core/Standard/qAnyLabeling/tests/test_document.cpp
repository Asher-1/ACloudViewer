// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QBuffer>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <cstdio>

#include "AnnotationDocument.h"

namespace {
bool write(const QString& path, const QJsonObject& json) {
    QFile file(path);
    const auto bytes = QJsonDocument(json).toJson();
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}
}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc >= 2) {
        const auto loaded = loadAnnotationDocument(
                QString::fromLocal8Bit(argv[1]),
                argc >= 3 ? QString::fromLocal8Bit(argv[2]) : QString());
        const QJsonObject result{
                {QStringLiteral("ok"), loaded.error.isEmpty()},
                {QStringLiteral("path"), loaded.imagePath},
                {QStringLiteral("size"),
                 QJsonArray{loaded.image.width(), loaded.image.height()}},
                {QStringLiteral("error"), loaded.error}};
        std::puts(QJsonDocument(result)
                          .toJson(QJsonDocument::Compact)
                          .constData());
        return 0;
    }
    QTemporaryDir temporary;
    if (!temporary.isValid()) return 1;
    QDir root(temporary.path());
    if (!root.mkdir(QStringLiteral("images")) ||
        !root.mkdir(QStringLiteral("labels")))
        return 1;
    const auto imagePath = root.filePath(QStringLiteral("images/scene.png"));
    const auto labels = root.filePath(QStringLiteral("labels/scene.json"));
    QImage image(37, 19, QImage::Format_RGB32);
    image.fill(Qt::red);
    if (!image.save(imagePath)) return 1;
    QJsonObject json{{QStringLiteral("shapes"), QJsonArray()},
                     {QStringLiteral("imagePath"),
                      QStringLiteral("../images/scene.png")},
                     {QStringLiteral("imageData"), QJsonValue::Null},
                     {QStringLiteral("custom"), 42}};
    if (!write(labels, json)) return 1;
    const auto relative = loadAnnotationDocument(labels);
    if (!relative.error.isEmpty() || relative.imagePath != imagePath ||
        relative.image.size() != image.size() || relative.json != json)
        return 1;
    json.insert(QStringLiteral("imagePath"), imagePath);
    if (!write(labels, json) || !loadAnnotationDocument(labels).error.isEmpty())
        return 1;
    if (!image.save(root.filePath(QStringLiteral("labels/scene.png"))))
        return 1;
    json.insert(QStringLiteral("imagePath"),
                QStringLiteral("/old/machine/scene.png"));
    if (!write(labels, json)) return 1;
    const auto fallback = loadAnnotationDocument(labels);
    if (!fallback.error.isEmpty() ||
        fallback.imagePath != root.filePath(QStringLiteral("labels/scene.png")))
        return 1;
    json.insert(QStringLiteral("imagePath"),
                QStringLiteral("/missing/original.png"));
    if (!write(labels, json)) return 1;
    const auto overrideImage = loadAnnotationDocument(
            imagePath, root.filePath(QStringLiteral("labels")));
    if (!overrideImage.error.isEmpty() ||
        overrideImage.imagePath != imagePath ||
        overrideImage.labelPath != labels || overrideImage.json != json)
        return 1;
    // Embedded pixels have precedence even when disk dimensions differ or
    // the stored path is no longer available. Metadata is retained verbatim.
    QImage embeddedImage(11, 7, QImage::Format_RGB32);
    embeddedImage.fill(Qt::blue);
    QByteArray encoded;
    QBuffer buffer(&encoded);
    if (!buffer.open(QIODevice::WriteOnly) ||
        !embeddedImage.save(&buffer, "PNG"))
        return 1;
    json.insert(QStringLiteral("imageData"),
                QString::fromLatin1(encoded.toBase64()));
    if (!write(labels, json)) return 1;
    const auto embedded = loadAnnotationDocument(labels);
    if (!embedded.error.isEmpty() ||
        embedded.image.size() != embeddedImage.size() ||
        embedded.json.value(QStringLiteral("custom")).toInt() != 42)
        return 1;
    json.insert(QStringLiteral("imageData"), QStringLiteral("broken pixels"));
    if (!write(labels, json) || loadAnnotationDocument(labels).error.isEmpty())
        return 1;
    json.insert(QStringLiteral("imageData"), 123);
    if (!write(labels, json) || loadAnnotationDocument(labels).error.isEmpty())
        return 1;
    json.insert(QStringLiteral("shapes"), QJsonObject());
    if (!write(labels, json) || loadAnnotationDocument(labels).error.isEmpty())
        return 1;
    QFile corrupt(labels);
    if (!corrupt.open(QIODevice::WriteOnly) || corrupt.write("{") != 1)
        return 1;
    corrupt.close();
    if (loadAnnotationDocument(imagePath,
                               root.filePath(QStringLiteral("labels")))
                .error.isEmpty())
        return 1;
    std::puts(
            "Document paths, embedded pixels, overrides and invalid input: "
            "PASS");
    return 0;
}
