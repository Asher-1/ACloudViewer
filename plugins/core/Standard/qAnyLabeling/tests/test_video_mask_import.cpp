// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QBuffer>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <cstdio>

#include "VideoMaskImport.h"

namespace {
bool writeMask(const QString& path) {
    QImage mask(8, 8, QImage::Format_Grayscale8);
    mask.fill(0);
    for (int y = 1; y < 7; ++y)
        for (int x = 1; x < 7; ++x)
            if (x != 4 || y != 4) mask.scanLine(y)[x] = 255;
    return mask.save(path, "PNG");
}
}  // namespace

int main() {
    QTemporaryDir dir;
    if (!dir.isValid() || !writeMask(dir.filePath(QStringLiteral("mask.png"))))
        return 1;
    const QJsonObject manifest{
            {QStringLiteral("format"),
             QStringLiteral("acloudviewer_sam3_tracking_v1")},
            {QStringLiteral("width"), 8},
            {QStringLiteral("height"), 8},
            {QStringLiteral("total_frames"), 1},
            {QStringLiteral("start_frame"), 0},
            {QStringLiteral("frames"),
             QJsonArray{
                     QJsonObject{{QStringLiteral("frame"), 0},
                                 {QStringLiteral("instances"),
                                  QJsonArray{QJsonObject{
                                          {QStringLiteral("id"), 9},
                                          {QStringLiteral("score"), 0.8},
                                          {QStringLiteral("mask"),
                                           QStringLiteral("mask.png")}}}}}}}};
    const Sam3ArchiveImport converted = importSam3Archive(
            QJsonDocument(manifest).toJson(), dir.path(),
            {{dir.filePath(QStringLiteral("frame_000000.png")), QSize(8, 8)}});
    if (!converted.error.isEmpty() || converted.count != 1 ||
        converted.documents.size() != 1) {
        std::fprintf(stderr, "converted error=%s count=%d docs=%d\n",
                     converted.error.toUtf8().constData(), converted.count,
                     converted.documents.size());
        return 1;
    }
    const QJsonArray shapes = converted.documents.constBegin()
                                      .value()
                                      .value(QStringLiteral("shapes"))
                                      .toArray();
    if (shapes.size() != 1 ||
        shapes.first().toObject().value(QStringLiteral("group_id")).toInt(-1) !=
                9) {
        std::fprintf(stderr, "shapes=%d group=%d\n", shapes.size(),
                     shapes.isEmpty()
                             ? -1
                             : shapes.first()
                                       .toObject()
                                       .value(QStringLiteral("group_id"))
                                       .toInt(-1));
        return 1;
    }
    const QByteArray encoded =
            shapes.first()
                    .toObject()
                    .value(QStringLiteral("acloudviewer_mask_png"))
                    .toString()
                    .toLatin1();
    const QImage exact =
            QImage::fromData(QByteArray::fromBase64(encoded), "PNG");
    if (exact.isNull() || exact.pixelColor(4, 4).alpha() != 0 ||
        exact.pixelColor(2, 2).alpha() == 0) {
        std::fprintf(stderr, "exact null=%d hole=%d fill=%d\n", exact.isNull(),
                     exact.isNull() ? -1 : exact.pixelColor(4, 4).alpha(),
                     exact.isNull() ? -1 : exact.pixelColor(2, 2).alpha());
        return 1;
    }
    const Sam3ArchiveImport invalid = importSam3Archive(
            QByteArrayLiteral("{}"), dir.path(),
            {{dir.filePath(QStringLiteral("frame_000000.png")), QSize(8, 8)}});
    if (invalid.error.isEmpty()) return 1;
    const Sam3ArchiveImport wrongSize = importSam3Archive(
            QJsonDocument(manifest).toJson(), dir.path(),
            {{dir.filePath(QStringLiteral("frame_000000.png")), QSize(7, 8)}});
    if (wrongSize.error.isEmpty()) return 1;
    QJsonObject emptyManifest = manifest;
    QJsonObject emptyFrame = emptyManifest.value(QStringLiteral("frames"))
                                     .toArray()
                                     .first()
                                     .toObject();
    emptyFrame.insert(QStringLiteral("instances"), QJsonArray());
    emptyManifest.insert(QStringLiteral("frames"), QJsonArray{emptyFrame});
    const Sam3ArchiveImport empty = importSam3Archive(
            QJsonDocument(emptyManifest).toJson(), dir.path(),
            {{dir.filePath(QStringLiteral("frame_000000.png")), QSize(8, 8)}});
    if (!empty.error.isEmpty() || empty.count != 0 ||
        !empty.documents.isEmpty())
        return 1;
    QJsonObject middle = manifest.value(QStringLiteral("frames"))
                                 .toArray()
                                 .first()
                                 .toObject();
    middle.insert(QStringLiteral("frame"), 1);
    QJsonObject last = emptyFrame;
    last.insert(QStringLiteral("frame"), 2);
    QJsonObject sequence = manifest;
    sequence.insert(QStringLiteral("total_frames"), 3);
    sequence.insert(QStringLiteral("start_frame"), 1);
    sequence.insert(QStringLiteral("frames"), QJsonArray{middle, last});
    const QVector<Sam3ArchiveImage> threeImages = {
            {dir.filePath(QStringLiteral("frame_0.png")), QSize(8, 8)},
            {dir.filePath(QStringLiteral("frame_1.png")), QSize(8, 8)},
            {dir.filePath(QStringLiteral("frame_2.png")), QSize(8, 8)}};
    QVector<int> visited;
    const Sam3ArchiveImport streamed = visitSam3Archive(
            QJsonDocument(sequence).toJson(), dir.path(), threeImages,
            [&visited](int index, const QJsonObject& document) {
                if (document.isEmpty() != (index != 1)) return false;
                visited.append(index);
                return true;
            });
    if (!streamed.error.isEmpty() || streamed.count != 1 ||
        visited != QVector<int>({0, 1, 2}))
        return 1;
    last.insert(QStringLiteral("frame"), 1);
    sequence.insert(QStringLiteral("frames"), QJsonArray{middle, last});
    const Sam3ArchiveImport badSequence = visitSam3Archive(
            QJsonDocument(sequence).toJson(), dir.path(), threeImages,
            [](int, const QJsonObject&) { return true; });
    return badSequence.error.isEmpty() ? 1 : 0;
}
