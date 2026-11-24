// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <cstdio>

#include "VideoMaskArchive.h"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace {
SAM3WorkerResult makeResult(QVector<int> ids) {
    SAM3WorkerResult result;
    result.valid = true;
    result.detCount = 2;
    result.instanceIds = std::move(ids);
    result.scores = {0.9f, 0.8f};
    result.boxes = {{0, 0, 4, 4}, {1, 1, 3, 3}};
    QImage ring(5, 5, QImage::Format_Grayscale8);
    ring.fill(255);
    ring.setPixel(2, 2, 0);
    QImage dot(5, 5, QImage::Format_Grayscale8);
    dot.fill(0);
    dot.setPixel(2, 2, 255);
    result.instanceMasks = {ring, dot};
    return result;
}
}  // namespace

TEST(VideoMaskArchive, ComputesSourceDigestAndReportsMissingSource) {
    QTemporaryDir folder;
    ASSERT_TRUE(folder.isValid());
    QFile source(folder.filePath(QStringLiteral("source.bin")));
    ASSERT_TRUE(source.open(QIODevice::WriteOnly));
    ASSERT_EQ(source.write("abc", 3), 3);
    source.close();

    QString error;
    EXPECT_EQ(videoMaskArchiveSha256(source.fileName(), &error),
              QStringLiteral("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb"
                             "410ff61f20015ad"));
    EXPECT_TRUE(error.isEmpty());
    EXPECT_TRUE(videoMaskArchiveSha256(
                        folder.filePath(QStringLiteral("missing")), &error)
                        .isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(VideoMaskArchive, ExportsFrameAndPreservesInstanceIdsAndMaskHole) {
    QTemporaryDir folder;
    ASSERT_TRUE(folder.isValid());
    const auto result = makeResult({7, 19});
    QJsonObject frame;
    ASSERT_TRUE(writeVideoMaskFrame(folder.path(), 42, result, &frame));
    EXPECT_EQ(frame.value(QStringLiteral("frame")).toInt(), 42);

    const QJsonArray instances =
            frame.value(QStringLiteral("instances")).toArray();
    ASSERT_EQ(instances.size(), 2);
    EXPECT_EQ(instances.at(0).toObject().value(QStringLiteral("id")).toInt(),
              7);
    EXPECT_EQ(instances.at(1).toObject().value(QStringLiteral("id")).toInt(),
              19);
    const QString mask =
            instances.at(0).toObject().value(QStringLiteral("mask")).toString();
    const QImage decoded(folder.filePath(mask));
    ASSERT_FALSE(decoded.isNull());
    EXPECT_EQ(qGray(decoded.pixel(2, 2)), 0);
    EXPECT_EQ(qGray(decoded.pixel(0, 0)), 255);
}

TEST(VideoMaskArchive, RepeatedInstanceIdsReceiveDistinctMaskFiles) {
    QTemporaryDir folder;
    ASSERT_TRUE(folder.isValid());
    QJsonObject frame;
    ASSERT_TRUE(
            writeVideoMaskFrame(folder.path(), 43, makeResult({7, 7}), &frame));
    const QJsonArray instances =
            frame.value(QStringLiteral("instances")).toArray();
    ASSERT_EQ(instances.size(), 2);
    const QString first =
            instances.at(0).toObject().value(QStringLiteral("mask")).toString();
    const QString second =
            instances.at(1).toObject().value(QStringLiteral("mask")).toString();
    EXPECT_NE(first, second);
    EXPECT_FALSE(QImage(folder.filePath(first)).isNull());
    EXPECT_FALSE(QImage(folder.filePath(second)).isNull());
}

TEST(VideoMaskArchive, RoundTripsManifestAndRejectsInvalidDirectory) {
    QTemporaryDir folder;
    ASSERT_TRUE(folder.isValid());
    QJsonObject frame;
    ASSERT_TRUE(writeVideoMaskFrame(folder.path(), 42, makeResult({7, 19}),
                                    &frame));
    const QJsonObject manifest{{QStringLiteral("complete"), false},
                               {QStringLiteral("frames"), QJsonArray{frame}}};
    ASSERT_TRUE(writeVideoMaskManifest(folder.path(), manifest));
    QFile file(folder.filePath(QStringLiteral("manifest.json")));
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_EQ(QJsonDocument::fromJson(file.readAll()).object(), manifest);

    QJsonObject rejected{{QStringLiteral("stale"), true}};
    EXPECT_FALSE(writeVideoMaskFrame(folder.filePath(QStringLiteral("missing")),
                                     43, makeResult({7, 19}), &rejected));
    EXPECT_TRUE(rejected.isEmpty());
}
