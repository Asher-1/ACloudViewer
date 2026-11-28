// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QFile>
#include <QJsonDocument>
#include <QProcess>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <opencv2/videoio.hpp>

#include "VideoDatasetExporter.h"
#include "VideoSidecar.h"

class VideoDatasetExporterTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(temp.isValid());
        video = temp.filePath(QStringLiteral("source.avi"));
        cv::VideoWriter writer(video.toStdString(),
                               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                               10.0, cv::Size(32, 24));
        ASSERT_TRUE(writer.isOpened());
        for (int i = 0; i < 20; ++i)
            writer.write(cv::Mat(24, 32, CV_8UC3, cv::Scalar(i, i + 1, i + 2)));
        writer.release();
        sidecar.video = video;
        sidecar.fps = 10.0;
        sidecar.durationMs = 2000;
        sidecar.width = 32;
        sidecar.height = 24;
        sidecar.labels =
                QStringList{QStringLiteral("zebra"), QStringLiteral("car")};
        sidecar.segments = {{QStringLiteral("first"), QStringLiteral("zebra"),
                             0, 1000, 0, 10, QStringLiteral("first")},
                            {QStringLiteral("second"), QStringLiteral("car"),
                             1000, 2000, 10, 20, QStringLiteral("second")}};
        ASSERT_TRUE(saveVideoSidecar(video, sidecar, &error))
                << error.toStdString();
    }
    QJsonObject metadata(const QString& output) {
        QFile file(output + QStringLiteral("/metadata.json"));
        if (!file.open(QIODevice::ReadOnly)) {
            ADD_FAILURE() << "metadata cannot be opened";
            return {};
        }
        return QJsonDocument::fromJson(file.readAll()).object();
    }
    void expectDecodable(const QString& path) {
        cv::VideoCapture capture(path.toStdString());
        cv::Mat frame;
        ASSERT_TRUE(capture.isOpened());
        ASSERT_TRUE(capture.read(frame));
        EXPECT_FALSE(frame.empty());
    }
    QTemporaryDir temp;
    QString video;
    QString error;
    VideoSidecar sidecar;
};

TEST_F(VideoDatasetExporterTest, ExportsClipsFramesProgressAndArchive) {
    const auto output = temp.filePath(QStringLiteral("dataset"));
    QString archive;
    int previous = -1, totalProgress = -1;
    bool valid = true, sawFrame = false;
    ASSERT_TRUE(exportVideoDataset(
            {video, output, true, true, 5, true},
            [&](int current, int total) {
                if (totalProgress < 0) totalProgress = total;
                valid &= total == totalProgress && current >= previous &&
                         current <= total;
                sawFrame |=
                        current > 0 && current < total && current % 1000 != 0;
                previous = current;
                return true;
            },
            &error, &archive))
            << error.toStdString();
    EXPECT_TRUE(valid);
    EXPECT_TRUE(sawFrame);
    EXPECT_EQ(previous, totalProgress);
    const QStringList expected = {
            output + "/videos/zebra/source_first.mp4",
            output + "/videos/car/source_second.mp4",
            output + "/rawframes/zebra/source_first/img_00001.jpg",
            output + "/rawframes/zebra/source_first/img_00005.jpg",
            output + "/rawframes/car/source_second/img_00005.jpg",
            output + "/label_map.txt",
            output + "/metadata.json"};
    for (const auto& path : expected)
        EXPECT_TRUE(QFile::exists(path)) << path.toStdString();
    const auto json = metadata(output);
    EXPECT_EQ(json.value("num_classes").toInt(), 2);
    EXPECT_FALSE(json.value("re_encode").toBool());
    EXPECT_TRUE(json.contains("stream_copy_fallbacks"));
    EXPECT_TRUE(json.contains("ffmpeg_reencode_fallbacks"));
    EXPECT_EQ(json.value("counts").toObject().value("zebra").toInt(), 1);
    if (!QStandardPaths::findExecutable("ffmpeg").isEmpty()) {
        EXPECT_EQ(json.value("stream_copy_fallbacks").toInt(), 0);
        expectDecodable(expected.first());
    }
    QFile zip(archive);
    ASSERT_FALSE(archive.isEmpty());
    ASSERT_TRUE(zip.open(QIODevice::ReadOnly));
    EXPECT_EQ(zip.read(4), QByteArray("PK\x03\x04", 4));
    zip.close();
    const auto unzip = QStandardPaths::findExecutable("unzip");
    if (!unzip.isEmpty())
        EXPECT_EQ(QProcess::execute(unzip, {"-tqq", archive}), 0);
}

TEST_F(VideoDatasetExporterTest, ExportsUnsavedUnicodeLabel) {
    auto unsaved = sidecar;
    unsaved.segments[0].label = QStringLiteral("\u884c\u4eba");
    const auto output = temp.filePath("live-dataset");
    ASSERT_TRUE(exportVideoDataset({video, output, false, true, 0, false}, {},
                                   &error, nullptr, &unsaved));
    EXPECT_TRUE(QFile::exists(
            output +
            QStringLiteral(
                    "/rawframes/\u884c\u4eba/source_first/img_00001.jpg")));
}

TEST_F(VideoDatasetExporterTest, RecordsStreamCopyAndReencodeFallbacks) {
    auto one = sidecar;
    one.segments.removeLast();
    const QByteArray path = qgetenv("PATH");
    qputenv("PATH", QByteArray());
    const auto output = temp.filePath("fallback");
    const bool copied =
            exportVideoDataset({video, output, true, false, 0, false, false},
                               {}, &error, nullptr, &one);
    qputenv("PATH", path);
    ASSERT_TRUE(copied) << error.toStdString();
    EXPECT_EQ(metadata(output).value("stream_copy_fallbacks").toInt(), 1);
    qputenv("PATH", QByteArray());
    const auto fallback = temp.filePath("reencode-fallback");
    const bool reencoded =
            exportVideoDataset({video, fallback, true, false, 0, false, true},
                               {}, &error, nullptr, &one);
    qputenv("PATH", path);
    ASSERT_TRUE(reencoded) << error.toStdString();
    EXPECT_EQ(metadata(fallback).value("ffmpeg_reencode_fallbacks").toInt(), 1);
}

TEST_F(VideoDatasetExporterTest, ExplicitReencodeProducesDecodableClip) {
    auto one = sidecar;
    one.segments.removeLast();
    const auto output = temp.filePath("reencoded");
    ASSERT_TRUE(exportVideoDataset({video, output, true, false, 0, false, true},
                                   {}, &error, nullptr, &one));
    const auto json = metadata(output);
    EXPECT_TRUE(json.value("re_encode").toBool());
    EXPECT_EQ(json.value("stream_copy_fallbacks").toInt(), 0);
    EXPECT_TRUE(json.contains("ffmpeg_reencode_fallbacks"));
    expectDecodable(output + "/videos/zebra/source_first.mp4");
}

TEST_F(VideoDatasetExporterTest, CancellationPublishesNoDatasetOrArchive) {
    const auto output = temp.filePath("cancelled");
    EXPECT_FALSE(exportVideoDataset(
            {video, output, false, true, 0, true},
            [](int, int) { return false; }, &error));
    EXPECT_FALSE(QFile::exists(output));
    EXPECT_FALSE(QFile::exists(output + ".zip"));
}

TEST_F(VideoDatasetExporterTest, RejectsCollidingLabelsAndSegmentNames) {
    auto collision = sidecar;
    collision.segments[1].label = QStringLiteral("ZEBRA");
    const auto output = temp.filePath("collision");
    EXPECT_FALSE(exportVideoDataset({video, output, false, true, 0, false}, {},
                                    &error, nullptr, &collision));
    EXPECT_FALSE(QFile::exists(output));
    collision.segments[1].label = QStringLiteral("zebra");
    collision.segments[1].id = QStringLiteral("first");
    EXPECT_FALSE(exportVideoDataset({video, output, false, true, 0, false}, {},
                                    &error, nullptr, &collision));
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
