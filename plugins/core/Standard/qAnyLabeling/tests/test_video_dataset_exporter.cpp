// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <cstdio>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "VideoDatasetExporter.h"
#include "VideoSidecar.h"

namespace {
bool check(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}
}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir temp;
    if (!check(temp.isValid(), "temporary folder unavailable")) return 1;
    const QString video = temp.filePath(QStringLiteral("source.avi"));
    cv::VideoWriter writer(video.toStdString(),
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0,
                           cv::Size(32, 24));
    if (!check(writer.isOpened(), "video writer unavailable")) return 1;
    for (int i = 0; i < 20; ++i)
        writer.write(cv::Mat(24, 32, CV_8UC3, cv::Scalar(i, i + 1, i + 2)));
    writer.release();

    VideoSidecar sidecar;
    sidecar.video = video;
    sidecar.fps = 10.0;
    sidecar.durationMs = 2000;
    sidecar.width = 32;
    sidecar.height = 24;
    sidecar.labels =
            QStringList{QStringLiteral("zebra"), QStringLiteral("car")};
    sidecar.segments = {{QStringLiteral("first"), QStringLiteral("zebra"), 0,
                         1000, 0, 10, QStringLiteral("first")},
                        {QStringLiteral("second"), QStringLiteral("car"), 1000,
                         2000, 10, 20, QStringLiteral("second")}};
    QString error;
    if (!check(saveVideoSidecar(video, sidecar, &error),
               qPrintable(
                       QStringLiteral("sidecar save failed: %1").arg(error))))
        return 1;

    const QString output = temp.filePath(QStringLiteral("dataset"));
    QString archive;
    int previousProgress = -1;
    int progressTotal = -1;
    bool progressValid = true;
    bool sawFrameProgress = false;
    if (!check(exportVideoDataset(
                       {video, output, true, true, 5, true},
                       [&](int current, int total) {
                           if (progressTotal < 0) progressTotal = total;
                           progressValid &= total == progressTotal &&
                                            current >= previousProgress &&
                                            current <= total;
                           sawFrameProgress |= current > 0 && current < total &&
                                               current % 1000 != 0;
                           previousProgress = current;
                           return true;
                       },
                       &error, &archive),
               qPrintable(
                       QStringLiteral("dataset export failed: %1").arg(error))))
        return 1;
    if (!check(progressValid && sawFrameProgress &&
                       previousProgress == progressTotal,
               "dataset frame progress is incomplete or non-monotonic"))
        return 1;
    const QStringList expected = {
            output + QStringLiteral("/videos/zebra/source_first.mp4"),
            output + QStringLiteral("/videos/car/source_second.mp4"),
            output + QStringLiteral(
                             "/rawframes/zebra/source_first/img_00001.jpg"),
            output + QStringLiteral(
                             "/rawframes/zebra/source_first/img_00005.jpg"),
            output + QStringLiteral(
                             "/rawframes/car/source_second/img_00005.jpg"),
            output + QStringLiteral("/label_map.txt"),
            output + QStringLiteral("/metadata.json")};
    for (const QString& path : expected)
        if (!check(QFile::exists(path),
                   qPrintable(QStringLiteral("missing %1").arg(path))))
            return 1;
    QFile metadata(output + QStringLiteral("/metadata.json"));
    if (!check(metadata.open(QIODevice::ReadOnly), "metadata cannot be opened"))
        return 1;
    const QJsonObject json =
            QJsonDocument::fromJson(metadata.readAll()).object();
    if (!check(json.value(QStringLiteral("num_classes")).toInt() == 2,
               "metadata class count mismatch") ||
        !check(!json.value(QStringLiteral("re_encode")).toBool(),
               "default clip mode must use stream copy") ||
        !check(json.contains(QStringLiteral("stream_copy_fallbacks")),
               "metadata is missing the stream-copy fallback count") ||
        !check(json.contains(QStringLiteral("ffmpeg_reencode_fallbacks")),
               "metadata is missing the FFmpeg re-encode fallback count") ||
        !check(json.value(QStringLiteral("counts"))
                               .toObject()
                               .value(QStringLiteral("zebra"))
                               .toInt() == 1,
               "metadata segment count mismatch"))
        return 1;
    if (!QStandardPaths::findExecutable(QStringLiteral("ffmpeg")).isEmpty()) {
        if (!check(json.value(QStringLiteral("stream_copy_fallbacks"))
                                   .toInt() == 0,
                   "FFmpeg stream copy unexpectedly fell back"))
            return 1;
        cv::VideoCapture copied(expected.first().toStdString());
        cv::Mat firstFrame;
        if (!check(copied.isOpened() && copied.read(firstFrame) &&
                           !firstFrame.empty(),
                   "stream-copied clip is not decodable"))
            return 1;
    }
    QFile zip(archive);
    if (!check(!archive.isEmpty() && zip.open(QIODevice::ReadOnly),
               "ZIP archive cannot be opened"))
        return 1;
    if (!check(zip.read(4) == QByteArray("PK\x03\x04", 4),
               "ZIP local header mismatch"))
        return 1;
    zip.close();
    const QString unzip =
            QStandardPaths::findExecutable(QStringLiteral("unzip"));
    if (!unzip.isEmpty() &&
        !check(QProcess::execute(unzip, {QStringLiteral("-tqq"), archive}) == 0,
               "ZIP integrity check failed"))
        return 1;

    VideoSidecar unsaved = sidecar;
    unsaved.segments[0].label = QStringLiteral("行人");
    const QString liveOutput = temp.filePath(QStringLiteral("live-dataset"));
    if (!check(exportVideoDataset({video, liveOutput, false, true, 0, false},
                                  {}, &error, nullptr, &unsaved),
               "current sidecar export failed") ||
        !check(QFile::exists(
                       liveOutput +
                       QStringLiteral(
                               "/rawframes/行人/source_first/img_00001.jpg")),
               "unsaved Unicode label was not exported"))
        return 1;

    VideoSidecar oneSegment = sidecar;
    oneSegment.segments.removeLast();
    const QByteArray originalPath = qgetenv("PATH");
    qputenv("PATH", QByteArray());
    const QString fallbackOutput = temp.filePath(QStringLiteral("fallback"));
    const bool fallbackOk = exportVideoDataset(
            {video, fallbackOutput, true, false, 0, false, false}, {}, &error,
            nullptr, &oneSegment);
    qputenv("PATH", originalPath);
    if (!check(fallbackOk, "stream-copy fallback export failed")) return 1;
    QFile fallbackMetadata(fallbackOutput + QStringLiteral("/metadata.json"));
    if (!check(fallbackMetadata.open(QIODevice::ReadOnly),
               "fallback metadata cannot be opened") ||
        !check(QJsonDocument::fromJson(fallbackMetadata.readAll())
                               .object()
                               .value(QStringLiteral("stream_copy_fallbacks"))
                               .toInt() == 1,
               "stream-copy fallback was not recorded"))
        return 1;

    const QString reEncodedOutput = temp.filePath(QStringLiteral("reencoded"));
    if (!check(exportVideoDataset(
                       {video, reEncodedOutput, true, false, 0, false, true},
                       {}, &error, nullptr, &oneSegment),
               "explicit re-encode export failed"))
        return 1;
    QFile reEncodedMetadata(reEncodedOutput + QStringLiteral("/metadata.json"));
    if (!check(reEncodedMetadata.open(QIODevice::ReadOnly),
               "re-encode metadata cannot be opened"))
        return 1;
    const QJsonObject reEncoded =
            QJsonDocument::fromJson(reEncodedMetadata.readAll()).object();
    if (!check(reEncoded.value(QStringLiteral("re_encode")).toBool() &&
                       reEncoded.value(QStringLiteral("stream_copy_fallbacks"))
                                       .toInt() == 0 &&
                       reEncoded.contains(
                               QStringLiteral("ffmpeg_reencode_fallbacks")),
               "explicit re-encode metadata mismatch"))
        return 1;
    const QString encodedPath =
            reEncodedOutput + QStringLiteral("/videos/zebra/source_first.mp4");
    cv::VideoCapture encoded(encodedPath.toStdString());
    cv::Mat encodedFrame;
    if (!check(encoded.isOpened() && encoded.read(encodedFrame) &&
                       !encodedFrame.empty(),
               "re-encoded clip is not decodable"))
        return 1;

    qputenv("PATH", QByteArray());
    const QString reencodeFallbackOutput =
            temp.filePath(QStringLiteral("reencode-fallback"));
    const bool reencodeFallbackOk = exportVideoDataset(
            {video, reencodeFallbackOutput, true, false, 0, false, true}, {},
            &error, nullptr, &oneSegment);
    qputenv("PATH", originalPath);
    if (!check(reencodeFallbackOk, "FFmpeg re-encode fallback export failed"))
        return 1;
    QFile reencodeFallbackMetadata(reencodeFallbackOutput +
                                   QStringLiteral("/metadata.json"));
    if (!check(reencodeFallbackMetadata.open(QIODevice::ReadOnly),
               "FFmpeg fallback metadata cannot be opened") ||
        !check(QJsonDocument::fromJson(reencodeFallbackMetadata.readAll())
                               .object()
                               .value(QStringLiteral(
                                       "ffmpeg_reencode_fallbacks"))
                               .toInt() == 1,
               "FFmpeg re-encode fallback was not recorded"))
        return 1;

    const QString cancelledOutput = temp.filePath(QStringLiteral("cancelled"));
    if (!check(!exportVideoDataset(
                       {video, cancelledOutput, false, true, 0, true},
                       [](int, int) { return false; }, &error),
               "cancelled export unexpectedly succeeded") ||
        !check(!QFile::exists(cancelledOutput) &&
                       !QFile::exists(cancelledOutput + QStringLiteral(".zip")),
               "cancelled export left a published artifact"))
        return 1;

    VideoSidecar collision = sidecar;
    collision.segments[1].label = QStringLiteral("ZEBRA");
    const QString collisionOutput = temp.filePath(QStringLiteral("collision"));
    if (!check(!exportVideoDataset(
                       {video, collisionOutput, false, true, 0, false}, {},
                       &error, nullptr, &collision),
               "colliding labels were accepted") ||
        !check(!QFile::exists(collisionOutput),
               "colliding labels left a published dataset"))
        return 1;

    collision.segments[1].label = QStringLiteral("zebra");
    collision.segments[1].id = QStringLiteral("first");
    if (!check(!exportVideoDataset(
                       {video, collisionOutput, false, true, 0, false}, {},
                       &error, nullptr, &collision),
               "colliding segment names were accepted"))
        return 1;
    return 0;
}
