// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "VideoFrameExtraction.h"
#include "VideoMaskArchive.h"

class VideoFrameExtractionTest : public ::testing::Test {
protected:
    QTemporaryDir root;
    QString source, archive, destination, error;
    QJsonObject frame, manifest;
    QByteArray bytes;
    void SetUp() override {
        ASSERT_FALSE((!root.isValid()));
        source = root.filePath(QStringLiteral("source.avi"));
        cv::VideoWriter writer(source.toStdString(),
                               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5.0,
                               cv::Size(64, 64));
        if (!writer.isOpened()) GTEST_SKIP() << "MJPG writer unavailable";
        for (int i = 0; i < 3; ++i)
            writer.write(cv::Mat(64, 64, CV_8UC3, cv::Scalar(i * 40, 80, 200)));
        writer.release();
        archive = root.filePath(QStringLiteral("archive"));
        ASSERT_FALSE((!QDir().mkpath(archive)));
        QImage mask(64, 64, QImage::Format_Grayscale8);
        mask.fill(0);
        for (int y = 10; y < 30; ++y)
            for (int x = 10; x < 30; ++x)
                if (x != 20 || y != 20) mask.scanLine(y)[x] = 255;
        ASSERT_FALSE((!mask.save(
                QDir(archive).filePath(QStringLiteral("mask.png")), "PNG")));
        QJsonObject instance;
        instance.insert(QStringLiteral("id"), 7);
        instance.insert(QStringLiteral("mask"), QStringLiteral("mask.png"));

        frame.insert(QStringLiteral("frame"), 1);
        frame.insert(QStringLiteral("instances"), QJsonArray{instance});

        manifest.insert(QStringLiteral("format"),
                        QStringLiteral("acloudviewer_sam3_tracking_v1"));
        manifest.insert(QStringLiteral("source"), source);
        manifest.insert(QStringLiteral("width"), 64);
        manifest.insert(QStringLiteral("height"), 64);
        manifest.insert(QStringLiteral("total_frames"), 3);
        manifest.insert(QStringLiteral("start_frame"), 1);
        manifest.insert(QStringLiteral("frames"), QJsonArray{frame});
        QString digestError;
        const QString sourceDigest =
                videoMaskArchiveSha256(source, &digestError);
        ASSERT_FALSE((sourceDigest.isEmpty() || !digestError.isEmpty()))
                << "source digest failed: %s\n";
        manifest.insert(QStringLiteral("source_sha256"), sourceDigest);
        bytes = QJsonDocument(manifest).toJson();
        destination = root.filePath(QStringLiteral("frames"));
    }
};

TEST_F(VideoFrameExtractionTest, ExtractsFramesAndPreservesExactMaskInstance) {
    ASSERT_FALSE((!extractSam3FrameWorkspace(bytes, archive, source,
                                             destination, {}, &error)))
            << "extract failed: %s\n";
    for (int i = 0; i < 3; ++i) {
        const QString name =
                QStringLiteral("frame_%1").arg(i, 8, 10, QLatin1Char('0'));
        const QImage image(
                QDir(destination).filePath(name + QStringLiteral(".png")));
        ASSERT_FALSE((image.size() != QSize(64, 64)));
        const bool annotated = QFile::exists(
                QDir(destination).filePath(name + QStringLiteral(".json")));
        ASSERT_FALSE((annotated != (i == 1)));
    }
    QFile annotation(
            QDir(destination).filePath(QStringLiteral("frame_00000001.json")));
    ASSERT_FALSE((!annotation.open(QIODevice::ReadOnly)));
    const QJsonArray shapes = QJsonDocument::fromJson(annotation.readAll())
                                      .object()
                                      .value(QStringLiteral("shapes"))
                                      .toArray();
    ASSERT_FALSE((shapes.size() != 1));
    const QImage exact = QImage::fromData(
            QByteArray::fromBase64(
                    shapes.first()
                            .toObject()
                            .value(QStringLiteral("acloudviewer_mask_png"))
                            .toString()
                            .toLatin1()),
            "PNG");
    ASSERT_FALSE((exact.isNull() ||
                  shapes.first().toObject()
                                  .value(QStringLiteral("group_id"))
                                  .toInt() != 7 ||
                  exact.pixelColor(20, 20).alpha() != 0 ||
                  exact.pixelColor(15, 15).alpha() != 255));
}

TEST_F(VideoFrameExtractionTest, CancellationRemovesUncommittedWorkspace) {
    const QString cancelled = root.filePath(QStringLiteral("cancelled"));
    ASSERT_FALSE((extractSam3FrameWorkspace(
                          bytes, archive, source, cancelled,
                          [](int index, int) { return index < 1; }, &error) ||
                  QFileInfo::exists(cancelled)));
}

TEST_F(VideoFrameExtractionTest, EmptyInstancesProduceOnlyImages) {
    QJsonObject emptyFrame = frame;
    emptyFrame.insert(QStringLiteral("instances"), QJsonArray());
    QJsonObject emptyManifest = manifest;
    emptyManifest.insert(QStringLiteral("frames"), QJsonArray{emptyFrame});
    const QString emptyWorkspace = root.filePath(QStringLiteral("empty"));
    ASSERT_FALSE((
            !extractSam3FrameWorkspace(QJsonDocument(emptyManifest).toJson(),
                                       archive, source, emptyWorkspace, {},
                                       &error) ||
            !QFileInfo::exists(
                    QDir(emptyWorkspace)
                            .filePath(QStringLiteral("frame_00000000.png"))) ||
            QFileInfo::exists(
                    QDir(emptyWorkspace)
                            .filePath(QStringLiteral("frame_00000001.json")))));
}

TEST_F(VideoFrameExtractionTest, DuplicateFrameRecordsRejectWholeWorkspace) {
    QJsonObject lateFrame = frame;
    lateFrame.insert(QStringLiteral("instances"), QJsonArray());
    lateFrame.insert(QStringLiteral("frame"), 1);
    QJsonObject lateManifest = manifest;
    lateManifest.insert(QStringLiteral("frames"), QJsonArray{frame, lateFrame});
    const QString malformed = root.filePath(QStringLiteral("malformed"));
    ASSERT_FALSE((
            extractSam3FrameWorkspace(QJsonDocument(lateManifest).toJson(),
                                      archive, source, malformed, {}, &error) ||
            QFileInfo::exists(malformed)));
}

TEST_F(VideoFrameExtractionTest, ReplacedSourceContentRejectsArchivedMasks) {
    // A replacement with the same dimensions and frame count must not receive
    // the archived masks. Rewriting the source path keeps the path/metadata
    // checks valid and exercises the content binding specifically.
    cv::VideoWriter replacement(source.toStdString(),
                                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                5.0, cv::Size(64, 64));
    if (!replacement.isOpened()) {
        std::fprintf(stderr, "replacement video encoder unavailable\n");
        GTEST_SKIP() << "Replacement video encoder unavailable";
    }
    for (int i = 0; i < 3; ++i)
        replacement.write(
                cv::Mat(64, 64, CV_8UC3, cv::Scalar(200 - i * 30, 10, 10)));
    replacement.release();
    const QString replaced = root.filePath(QStringLiteral("replaced"));
    ASSERT_FALSE(
            (extractSam3FrameWorkspace(bytes, archive, source, replaced, {},
                                       &error) ||
             QFileInfo::exists(replaced) ||
             !error.contains(QStringLiteral("content"), Qt::CaseInsensitive)))
            << "replacement not rejected: %s\n";
}

TEST_F(VideoFrameExtractionTest, MalformedDigestRejectsWholeWorkspace) {
    QJsonObject malformedDigest = manifest;
    malformedDigest.insert(QStringLiteral("source_sha256"),
                           QStringLiteral("bad"));
    const QString malformedDigestDir =
            root.filePath(QStringLiteral("malformed_digest"));
    ASSERT_FALSE(
            (extractSam3FrameWorkspace(QJsonDocument(malformedDigest).toJson(),
                                       archive, source, malformedDigestDir, {},
                                       &error) ||
             QFileInfo::exists(malformedDigestDir) ||
             !error.contains(QStringLiteral("SHA-256"), Qt::CaseInsensitive)))
            << "malformed digest not rejected: %s\n";
}

TEST_F(VideoFrameExtractionTest, LegacyManifestRetainsStructuralCompatibility) {
    // Legacy manifests without a digest stay readable when the source still
    // satisfies their structural checks.
    QJsonObject legacy = manifest;
    legacy.remove(QStringLiteral("source_sha256"));
    const QString legacyDir = root.filePath(QStringLiteral("legacy"));
    ASSERT_FALSE(
            (!extractSam3FrameWorkspace(QJsonDocument(legacy).toJson(), archive,
                                        source, legacyDir, {}, &error) ||
             !QFileInfo::exists(QDir(legacyDir).filePath(
                     QStringLiteral("frame_00000000.png")))))
            << "legacy manifest rejected: %s\n";
}

TEST_F(VideoFrameExtractionTest, FrameCountMismatchLeavesNoWorkspace) {
    manifest.insert(QStringLiteral("total_frames"), 4);
    const QString mismatch = root.filePath(QStringLiteral("mismatch"));
    EXPECT_FALSE(extractSam3FrameWorkspace(QJsonDocument(manifest).toJson(),
                                           archive, source, mismatch, {},
                                           &error));
    EXPECT_FALSE(QFileInfo::exists(mismatch));
}
