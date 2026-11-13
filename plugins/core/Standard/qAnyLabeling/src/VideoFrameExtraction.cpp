// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoFrameExtraction.h"

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSaveFile>
#include <QTemporaryDir>
#include <exception>
#include <limits>

#include "VideoMaskArchive.h"
#include "VideoMaskImport.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include "OpenCVFrameSource.h"
#include "VideoPlaybackWidget.h"
#endif

namespace {
bool fail(QString* error, const QString& message) {
    if (error) *error = message;
    return false;
}
}  // namespace

bool extractSam3FrameWorkspace(const QByteArray& bytes,
                               const QString& archiveDirectory,
                               const QString& videoPath,
                               const QString& outputDirectory,
                               const std::function<bool(int, int)>& progress,
                               QString* error) {
#ifndef HAS_OPENCV_FACE_CAPTURE
    Q_UNUSED(bytes)
    Q_UNUSED(archiveDirectory)
    Q_UNUSED(videoPath)
    Q_UNUSED(outputDirectory)
    Q_UNUSED(progress)
    return fail(error, QStringLiteral("OpenCV video decoding is unavailable"));
#else
    const QJsonDocument parsed = QJsonDocument::fromJson(bytes);
    if (!parsed.isObject())
        return fail(error, QStringLiteral("Invalid SAM3 tracking manifest"));
    const QJsonObject manifest = parsed.object();
    const int count = manifest.value(QStringLiteral("total_frames")).toInt(-1);
    const QSize size(manifest.value(QStringLiteral("width")).toInt(-1),
                     manifest.value(QStringLiteral("height")).toInt(-1));
    if (count <= 0 || count > 1000000 || !size.isValid() ||
        qint64(size.width()) * size.height() > std::numeric_limits<int>::max())
        return fail(
                error,
                QStringLiteral("Invalid SAM3 video dimensions or frame count"));
    const QFileInfo video(videoPath);
    const QString canonicalVideo = video.canonicalFilePath();
    const QString originalPath =
            manifest.value(QStringLiteral("source")).toString();
    const QFileInfo original(originalPath);
    if (canonicalVideo.isEmpty() || !video.isFile())
        return fail(error, QStringLiteral("Source video does not exist"));
    if (originalPath.isEmpty())
        return fail(error,
                    QStringLiteral("SAM3 archive has no source video path"));
    if (original.exists() && original.canonicalFilePath() != canonicalVideo)
        return fail(error,
                    QStringLiteral(
                            "Selected video differs from the archive source"));
    const QString expectedDigest =
            manifest.value(QStringLiteral("source_sha256"))
                    .toString()
                    .trimmed();
    if (manifest.contains(QStringLiteral("source_sha256"))) {
        static const QRegularExpression sha256Pattern(
                QStringLiteral("^[0-9a-fA-F]{64}$"));
        if (!sha256Pattern.match(expectedDigest).hasMatch())
            return fail(error,
                        QStringLiteral("Invalid SAM3 source video SHA-256"));
        QString digestError;
        const QString actualDigest =
                videoMaskArchiveSha256(canonicalVideo, &digestError);
        if (actualDigest.isEmpty() && !digestError.isEmpty())
            return fail(error, QStringLiteral("Cannot hash source video: %1")
                                       .arg(digestError));
        if (actualDigest.compare(expectedDigest, Qt::CaseInsensitive) != 0)
            return fail(error,
                        QStringLiteral("Source video content does not match "
                                       "the SAM3 archive"));
    }
    if (outputDirectory.isEmpty() || QFileInfo::exists(outputDirectory))
        return fail(
                error,
                QStringLiteral(
                        "Frame output directory already exists or is invalid"));

    OpenCVFrameSource source;
    try {
        if (!source.openVideo(canonicalVideo.toStdString()))
            return fail(error, QStringLiteral("Cannot open source video"));
        if (source.frameCount() != count || source.width() != size.width() ||
            source.height() != size.height())
            return fail(error, QStringLiteral("Video frame count or dimensions "
                                              "differ from the archive"));
    } catch (const std::exception& ex) {
        return fail(error, QString::fromLocal8Bit(ex.what()));
    }

    const QFileInfo output(outputDirectory);
    if (!QDir().mkpath(output.absolutePath()))
        return fail(error,
                    QStringLiteral("Cannot create parent frame directory"));
    QTemporaryDir temporary(output.absoluteDir().filePath(
            QStringLiteral(".%1.XXXXXX").arg(output.fileName())));
    if (!temporary.isValid())
        return fail(error,
                    QStringLiteral("Cannot create temporary frame directory"));

    QVector<Sam3ArchiveImage> images;
    images.reserve(count);
    for (int index = 0; index < count; ++index) {
        images.append({QDir(temporary.path())
                               .filePath(QStringLiteral("frame_%1.png")
                                                 .arg(index, 8, 10,
                                                      QLatin1Char('0'))),
                       size});
    }
    QString writeError;
    try {
        const Sam3ArchiveImport imported = visitSam3Archive(
                bytes, archiveDirectory, images,
                [&](int index, const QJsonObject& document) {
                    if (progress && !progress(index, count)) {
                        writeError =
                                QStringLiteral("Frame extraction cancelled");
                        return false;
                    }
                    cv::Mat frame;
                    if (source.read(frame, nullptr) !=
                                IFrameSource::ReadResult::Ok ||
                        frame.empty()) {
                        writeError = QStringLiteral("Cannot decode frame %1")
                                             .arg(index);
                        return false;
                    }
                    const QImage image =
                            VideoPlaybackWidget::cvMatToQImage(frame);
                    if (image.isNull() || image.size() != size ||
                        !image.save(images.at(index).filePath, "PNG")) {
                        writeError = QStringLiteral("Cannot write frame %1")
                                             .arg(index);
                        return false;
                    }
                    if (document.isEmpty()) return true;
                    const QString sidecar =
                            QDir(temporary.path())
                                    .filePath(
                                            QFileInfo(images.at(index).filePath)
                                                    .completeBaseName() +
                                            QStringLiteral(".json"));
                    QSaveFile file(sidecar);
                    const QByteArray data = QJsonDocument(document).toJson();
                    if (!file.open(QIODevice::WriteOnly) ||
                        file.write(data) != data.size() || !file.commit()) {
                        writeError = QStringLiteral(
                                             "Cannot write frame %1 annotation")
                                             .arg(index);
                        return false;
                    }
                    return true;
                });
        if (!imported.error.isEmpty())
            return fail(error,
                        writeError.isEmpty() ? imported.error : writeError);
    } catch (const std::exception& ex) {
        return fail(error, QString::fromLocal8Bit(ex.what()));
    }
    if (progress && !progress(count, count))
        return fail(error, QStringLiteral("Frame extraction cancelled"));
    if (QFileInfo::exists(outputDirectory) ||
        !QDir().rename(temporary.path(), outputDirectory))
        return fail(error, QStringLiteral("Cannot publish frame directory"));
    temporary.setAutoRemove(false);
    if (error) error->clear();
    return true;
#endif
}
