// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoMaskArchive.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

bool writeVideoMaskFrame(const QString& directory,
                         int frameIndex,
                         const SAM3WorkerResult& result,
                         QJsonObject* frameEntry) {
    if (frameEntry) *frameEntry = QJsonObject();
    if (!frameEntry || frameIndex < 0 || !result.valid ||
        result.instanceMasks.size() != result.instanceIds.size() ||
        result.instanceMasks.size() != result.detCount ||
        !QDir(directory).exists()) {
        return false;
    }
    QJsonArray instances;
    for (int i = 0; i < result.detCount; ++i) {
        const QImage& mask = result.instanceMasks.at(i);
        if (mask.isNull()) return false;
        const int id = result.instanceIds.at(i);
        const QString filename =
                QStringLiteral("frame_%1_instance_%2_det_%3.png")
                        .arg(frameIndex, 8, 10, QLatin1Char('0'))
                        .arg(id)
                        .arg(i, 4, 10, QLatin1Char('0'));
        QSaveFile maskFile(QDir(directory).filePath(filename));
        if (!maskFile.open(QIODevice::WriteOnly) ||
            !mask.save(&maskFile, "PNG") || !maskFile.commit()) {
            return false;
        }
        QJsonObject item{{QStringLiteral("id"), id},
                         {QStringLiteral("score"), result.scores.value(i)},
                         {QStringLiteral("mask"), filename}};
        if (i < result.boxes.size()) {
            const auto& box = result.boxes.at(i);
            item.insert(QStringLiteral("bbox"),
                        QJsonArray{box.x0, box.y0, box.x1, box.y1});
        }
        instances.append(item);
    }
    *frameEntry = QJsonObject{{QStringLiteral("frame"), frameIndex},
                              {QStringLiteral("instances"), instances}};
    return true;
}

bool writeVideoMaskManifest(const QString& directory,
                            const QJsonObject& manifest) {
    QSaveFile file(QDir(directory).filePath(QStringLiteral("manifest.json")));
    const QByteArray data =
            QJsonDocument(manifest).toJson(QJsonDocument::Indented);
    return file.open(QIODevice::WriteOnly) && file.write(data) == data.size() &&
           file.commit();
}

QString videoMaskArchiveSha256(const QString& videoPath, QString* error) {
    if (error) error->clear();
    QFile file(videoPath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = file.errorString();
        return {};
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray chunk = file.read(4 * 1024 * 1024);
        if (file.error() != QFileDevice::NoError) {
            if (error) *error = file.errorString();
            return {};
        }
        hash.addData(chunk);
    }
    return QString::fromLatin1(hash.result().toHex());
}
