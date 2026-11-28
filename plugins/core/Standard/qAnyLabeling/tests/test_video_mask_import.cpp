// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QBuffer>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>

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

TEST(VideoMaskImport, ConvertsAndValidatesArchive) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    ASSERT_TRUE(writeMask(dir.filePath(QStringLiteral("mask.png"))));
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
    ASSERT_TRUE(converted.error.isEmpty()) << converted.error.toStdString();
    EXPECT_EQ(converted.count, 1);
    ASSERT_EQ(converted.documents.size(), 1);
    const QJsonArray shapes = converted.documents.constBegin()
                                      .value()
                                      .value(QStringLiteral("shapes"))
                                      .toArray();
    ASSERT_EQ(shapes.size(), 1);
    EXPECT_EQ(shapes.first()
                      .toObject()
                      .value(QStringLiteral("group_id"))
                      .toInt(-1),
              9);
    const QByteArray encoded =
            shapes.first()
                    .toObject()
                    .value(QStringLiteral("acloudviewer_mask_png"))
                    .toString()
                    .toLatin1();
    const QImage exact =
            QImage::fromData(QByteArray::fromBase64(encoded), "PNG");
    ASSERT_FALSE(exact.isNull());
    EXPECT_EQ(exact.pixelColor(4, 4).alpha(), 0);
    EXPECT_NE(exact.pixelColor(2, 2).alpha(), 0);
    const Sam3ArchiveImport invalid = importSam3Archive(
            QByteArrayLiteral("{}"), dir.path(),
            {{dir.filePath(QStringLiteral("frame_000000.png")), QSize(8, 8)}});
    ASSERT_FALSE(invalid.error.isEmpty());
    const Sam3ArchiveImport wrongSize = importSam3Archive(
            QJsonDocument(manifest).toJson(), dir.path(),
            {{dir.filePath(QStringLiteral("frame_000000.png")), QSize(7, 8)}});
    ASSERT_FALSE(wrongSize.error.isEmpty());
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
    if ((!empty.error.isEmpty() || empty.count != 0 ||
         !empty.documents.isEmpty())) {
        ADD_FAILURE() << "archive result mismatch";
        return;
    }
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
    if ((!streamed.error.isEmpty() || streamed.count != 1 ||
         visited != QVector<int>({0, 1, 2}))) {
        ADD_FAILURE() << "archive result mismatch";
        return;
    }
    last.insert(QStringLiteral("frame"), 1);
    sequence.insert(QStringLiteral("frames"), QJsonArray{middle, last});
    const Sam3ArchiveImport badSequence = visitSam3Archive(
            QJsonDocument(sequence).toJson(), dir.path(), threeImages,
            [](int, const QJsonObject&) { return true; });
    EXPECT_FALSE(badSequence.error.isEmpty());
}
