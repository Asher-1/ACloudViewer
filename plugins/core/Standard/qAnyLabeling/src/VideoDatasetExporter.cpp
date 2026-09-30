// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoDatasetExporter.h"

#include <zlib.h>

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
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

#include "VideoSegmentExporter.h"
#include "VideoSidecar.h"

namespace {
struct ZipEntry {
    QByteArray name;
    quint32 offset = 0;
    quint32 size = 0;
    quint32 compressed = 0;
    quint32 crc = 0;
};

bool write16(QFile& file, quint16 value) {
    const char bytes[] = {char(value), char(value >> 8)};
    return file.write(bytes, 2) == 2;
}

bool write32(QFile& file, quint32 value) {
    return write16(file, quint16(value)) && write16(file, quint16(value >> 16));
}

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

bool writeZip(const QString& directory,
              const QString& zipPath,
              QString* error) {
    QFile zip(zipPath);
    if (!zip.open(QIODevice::WriteOnly | QIODevice::NewOnly)) {
        if (error) *error = zip.errorString();
        return false;
    }
    QVector<ZipEntry> entries;
    QDirIterator iterator(directory, QDir::Files | QDir::NoSymLinks,
                          QDirIterator::Subdirectories);
    const QDir root(directory);
    while (iterator.hasNext()) {
        const QString path = iterator.next();
        QFile input(path);
        const qint64 size = QFileInfo(path).size();
        if (size > std::numeric_limits<quint32>::max() ||
            zip.pos() > std::numeric_limits<quint32>::max() ||
            entries.size() >= std::numeric_limits<quint16>::max() ||
            !input.open(QIODevice::ReadOnly)) {
            if (error)
                *error = QStringLiteral(
                                 "ZIP64 is required or a file cannot be read: "
                                 "%1")
                                 .arg(path);
            zip.close();
            QFile::remove(zipPath);
            return false;
        }
        ZipEntry entry;
        entry.name = (QFileInfo(directory).fileName() + QLatin1Char('/') +
                      root.relativeFilePath(path).replace(QLatin1Char('\\'),
                                                          QLatin1Char('/')))
                             .toUtf8();
        if (entry.name.size() > std::numeric_limits<quint16>::max()) break;
        entry.offset = quint32(zip.pos());
        entry.size = quint32(size);
        // Bit 3 allows CRC and sizes to be written after a streaming deflate.
        if (!write32(zip, 0x04034b50) || !write16(zip, 20) ||
            !write16(zip, 0x0808) || !write16(zip, 8) || !write16(zip, 0) ||
            !write16(zip, 0) || !write32(zip, 0) || !write32(zip, 0) ||
            !write32(zip, 0) || !write16(zip, quint16(entry.name.size())) ||
            !write16(zip, 0) || zip.write(entry.name) != entry.name.size())
            break;

        z_stream stream{};
        if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS,
                         8, Z_DEFAULT_STRATEGY) != Z_OK)
            break;
        bool success = true;
        int result = Z_OK;
        char output[65536];
        while (success && result != Z_STREAM_END) {
            const QByteArray chunk = input.read(65536);
            if (chunk.isEmpty() && !input.atEnd()) {
                success = false;
                break;
            }
            entry.crc = crc32(entry.crc,
                              reinterpret_cast<const Bytef*>(chunk.constData()),
                              uInt(chunk.size()));
            stream.next_in = reinterpret_cast<Bytef*>(
                    const_cast<char*>(chunk.constData()));
            stream.avail_in = uInt(chunk.size());
            const int flush = input.atEnd() ? Z_FINISH : Z_NO_FLUSH;
            do {
                stream.next_out = reinterpret_cast<Bytef*>(output);
                stream.avail_out = sizeof(output);
                result = deflate(&stream, flush);
                const qint64 produced = sizeof(output) - stream.avail_out;
                if (result == Z_STREAM_ERROR ||
                    zip.write(output, produced) != produced) {
                    success = false;
                    break;
                }
            } while (stream.avail_in > 0 || stream.avail_out == 0 ||
                     (flush == Z_FINISH && result != Z_STREAM_END));
        }
        const uLong totalIn = stream.total_in;
        const uLong totalOut = stream.total_out;
        deflateEnd(&stream);
        if (!success || totalIn != quint64(size) ||
            totalOut > std::numeric_limits<quint32>::max() ||
            zip.pos() > std::numeric_limits<quint32>::max() ||
            !write32(zip, 0x08074b50) || !write32(zip, entry.crc) ||
            !write32(zip, quint32(totalOut)) || !write32(zip, entry.size))
            break;
        entry.compressed = quint32(totalOut);
        entries.append(entry);
    }
    // A premature loop exit leaves fewer entries than files in the directory.
    QDirIterator countIterator(directory, QDir::Files | QDir::NoSymLinks,
                               QDirIterator::Subdirectories);
    int fileCount = 0;
    while (countIterator.hasNext()) {
        countIterator.next();
        ++fileCount;
    }
    bool success = entries.size() == fileCount && fileCount > 0;
    const qint64 centralOffset = zip.pos();
    for (const ZipEntry& entry : entries) {
        success = success && write32(zip, 0x02014b50) && write16(zip, 20) &&
                  write16(zip, 20) && write16(zip, 0x0808) && write16(zip, 8) &&
                  write16(zip, 0) && write16(zip, 0) &&
                  write32(zip, entry.crc) && write32(zip, entry.compressed) &&
                  write32(zip, entry.size) &&
                  write16(zip, quint16(entry.name.size())) && write16(zip, 0) &&
                  write16(zip, 0) && write16(zip, 0) && write16(zip, 0) &&
                  write32(zip, 0) && write32(zip, entry.offset) &&
                  zip.write(entry.name) == entry.name.size();
    }
    const qint64 centralSize = zip.pos() - centralOffset;
    success = success && centralOffset <= std::numeric_limits<quint32>::max() &&
              centralSize <= std::numeric_limits<quint32>::max() &&
              write32(zip, 0x06054b50) && write16(zip, 0) && write16(zip, 0) &&
              write16(zip, quint16(entries.size())) &&
              write16(zip, quint16(entries.size())) &&
              write32(zip, quint32(centralSize)) &&
              write32(zip, quint32(centralOffset)) && write16(zip, 0);
    if (!zip.flush()) success = false;
    zip.close();
    if (!success) {
        QFile::remove(zipPath);
        if (error) *error = QStringLiteral("Could not create ZIP archive");
    }
    return success;
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
        if (!writeZip(dataset, stagedZip, &localError)) return fail(localError);
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
