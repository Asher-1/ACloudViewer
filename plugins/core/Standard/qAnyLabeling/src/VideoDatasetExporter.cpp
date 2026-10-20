// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoDatasetExporter.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QTemporaryDir>
#include <QUuid>
#include <algorithm>
#include <cmath>
#include <limits>

#include "AnnotationArchive.h"
#include "VideoSegmentExporter.h"
#include "VideoSidecar.h"

namespace {
QString safeToken(QString value) {
    value.replace(QRegularExpression(QStringLiteral("[^\\p{L}\\p{N}._-]+")),
                  QStringLiteral("_"));
    value = value.trimmed();
    return value.isEmpty() || value == QStringLiteral(".") ||
                           value == QStringLiteral("..")
                   ? QStringLiteral("unknown")
                   : value;
}

bool writeText(const QString& path, const QByteArray& bytes) {
    QSaveFile file(path);
    return file.open(QIODevice::WriteOnly) &&
           file.write(bytes) == bytes.size() && file.commit();
}

}  // namespace

bool exportVideoDataset(const VideoDatasetExportOptions& options,
                        const std::function<bool(int, int)>& progress,
                        QString* error,
                        QString* zipPath,
                        const VideoSidecar* currentSidecar) {
    const auto fail = [error](const QString& message) {
        if (error) *error = message;
        return false;
    };
    if (zipPath) zipPath->clear();
    const QFileInfo video(options.videoPath);
    const QFileInfo destination(options.outputDirectory);
    if (!video.isFile() || destination.fileName().isEmpty() ||
        (!options.includeVideo && !options.includeRawFrames) ||
        options.rawFrameFps < 0)
        return fail(QStringLiteral("Invalid video dataset export options"));
    if (destination.exists() ||
        (options.zipOutput &&
         QFileInfo::exists(options.outputDirectory + QStringLiteral(".zip"))))
        return fail(QStringLiteral("Dataset destination already exists"));
    if (!destination.absoluteDir().exists())
        return fail(QStringLiteral("Dataset parent folder does not exist"));
    VideoSidecar sidecar;
    QString localError;
    if (currentSidecar)
        sidecar = *currentSidecar;
    else if (!loadVideoSidecar(options.videoPath, &sidecar, &localError))
        return fail(localError);
    if (!std::isfinite(sidecar.fps) || sidecar.fps <= 0)
        return fail(QStringLiteral("Video sidecar has no valid frame rate"));
    QVector<VideoSegment> segments;
    QStringList labels;
    QSet<QString> safeLabels;
    QSet<QString> outputNames;
    for (const VideoSegment& segment : sidecar.segments) {
        if (segment.label.trimmed().isEmpty() ||
            segment.endMs <= segment.startMs)
            continue;
        segments.append(segment);
        if (!labels.contains(segment.label)) {
            const QString normalized = safeToken(segment.label).toCaseFolded();
            if (safeLabels.contains(normalized))
                return fail(
                        QStringLiteral("Labels have colliding folder names: %1")
                                .arg(segment.label));
            safeLabels.insert(normalized);
            labels.append(segment.label);
        }
        const QString stem =
                safeToken(video.completeBaseName()) + QLatin1Char('_') +
                safeToken(segment.id.isEmpty()
                                  ? QString::number(segments.size())
                                  : segment.id);
        const QString outputName = safeToken(segment.label).toCaseFolded() +
                                   QLatin1Char('/') + stem.toCaseFolded();
        if (outputNames.contains(outputName))
            return fail(
                    QStringLiteral("Segments have colliding output names: %1")
                            .arg(segment.id));
        outputNames.insert(outputName);
    }
    if (segments.isEmpty())
        return fail(QStringLiteral("No labeled video segments to export"));
    std::sort(labels.begin(), labels.end());
    QTemporaryDir staging(destination.absoluteDir().filePath(
            QStringLiteral(".anylabeling-export-XXXXXX")));
    if (!staging.isValid())
        return fail(QStringLiteral("Cannot create staging folder"));
    const QString dataset =
            QDir(staging.path()).filePath(destination.fileName());
    if (!QDir().mkpath(dataset))
        return fail(QStringLiteral("Cannot create dataset folder"));

    QJsonObject counts;
    int completed = 0;
    int copyFallbacks = 0;
    int ffmpegFallbacks = 0;
    constexpr int kProgressUnitsPerExport = 1000;
    const int exportsPerSegment =
            int(options.includeVideo) + int(options.includeRawFrames);
    if (segments.size() > std::numeric_limits<int>::max() /
                                  (kProgressUnitsPerExport * exportsPerSegment))
        return fail(QStringLiteral("Too many segments to export"));
    const int total =
            segments.size() * exportsPerSegment * kProgressUnitsPerExport;
    for (int index = 0; index < segments.size(); ++index) {
        const VideoSegment& segment = segments[index];
        const QString base =
                safeToken(video.completeBaseName()) + QLatin1Char('_') +
                safeToken(segment.id.isEmpty() ? QString::number(index + 1)
                                               : segment.id);
        const QString label = safeToken(segment.label);
        counts.insert(segment.label, counts.value(segment.label).toInt() + 1);
        for (int kind = 0; kind < 2; ++kind) {
            if ((kind == 0 && !options.includeVideo) ||
                (kind == 1 && !options.includeRawFrames))
                continue;
            if (progress && !progress(completed, total))
                return fail(QStringLiteral("Export cancelled"));
            const QString output = QDir(dataset).filePath(
                    (kind == 0 ? QStringLiteral("videos/")
                               : QStringLiteral("rawframes/")) +
                    label + QLatin1Char('/') + base +
                    (kind == 0 ? QStringLiteral(".mp4") : QString()));
            VideoSegmentExportRequest request{
                    options.videoPath,
                    output,
                    segment.startMs,
                    segment.endMs,
                    sidecar.fps,
                    kind == 0 ? VideoSegmentExportFormat::Video
                              : VideoSegmentExportFormat::RawFrames,
                    kind == 0 ? 0 : options.rawFrameFps};
            request.reEncodeVideo = options.reEncodeVideo;
            bool usedCopyFallback = false;
            bool usedFfmpegFallback = false;
            if (!exportVideoSegment(
                        request,
                        [&](int frame, int frames) {
                            const int withinExport =
                                    frames > 0 ? int(qint64(frame) *
                                                     kProgressUnitsPerExport /
                                                     frames)
                                               : 0;
                            return !progress ||
                                   progress(completed + withinExport, total);
                        },
                        &localError, &usedCopyFallback, &usedFfmpegFallback))
                return fail(localError);
            if (usedCopyFallback) ++copyFallbacks;
            if (usedFfmpegFallback) ++ffmpegFallbacks;
            completed += kProgressUnitsPerExport;
        }
    }
    if (progress && !progress(total, total))
        return fail(QStringLiteral("Export cancelled"));
    QByteArray labelMap;
    QJsonArray labelArray;
    for (const QString& label : labels) {
        labelMap += label.toUtf8() + '\n';
        labelArray.append(label);
    }
    if (!writeText(QDir(dataset).filePath(QStringLiteral("label_map.txt")),
                   labelMap))
        return fail(QStringLiteral("Cannot write label_map.txt"));
    const QJsonObject metadata{
            {QStringLiteral("video"), video.fileName()},
            {QStringLiteral("include_video"), options.includeVideo},
            {QStringLiteral("include_rawframes"), options.includeRawFrames},
            {QStringLiteral("re_encode"), options.reEncodeVideo},
            {QStringLiteral("stream_copy_fallbacks"), copyFallbacks},
            {QStringLiteral("ffmpeg_reencode_fallbacks"), ffmpegFallbacks},
            {QStringLiteral("rawframe_fps"), options.rawFrameFps},
            {QStringLiteral("counts"), counts},
            {QStringLiteral("num_classes"), labels.size()},
            {QStringLiteral("label_map"), labelArray},
            {QStringLiteral("exported_at"),
             double(QDateTime::currentSecsSinceEpoch())}};
    if (!writeText(QDir(dataset).filePath(QStringLiteral("metadata.json")),
                   QJsonDocument(metadata).toJson(QJsonDocument::Indented)))
        return fail(QStringLiteral("Cannot write metadata.json"));

    QString stagedZip;
    if (options.zipOutput) {
        stagedZip = QDir(staging.path())
                            .filePath(destination.fileName() +
                                      QStringLiteral(".zip"));
        if (!writeZipDirectory(dataset, stagedZip, &localError))
            return fail(localError);
    }
    QString finalZip;
    if (options.zipOutput) {
        finalZip = options.outputDirectory + QStringLiteral(".zip");
        if (!QFile::rename(stagedZip, finalZip))
            return fail(QStringLiteral("Cannot finalize dataset ZIP"));
    }
    if (!QDir().rename(dataset, options.outputDirectory)) {
        if (!finalZip.isEmpty()) QFile::remove(finalZip);
        return fail(QStringLiteral("Cannot finalize dataset folder"));
    }
    if (zipPath) *zipPath = finalZip;
    if (error) error->clear();
    return true;
}
