// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoSegmentExporter.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QSaveFile>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QUuid>
#include <algorithm>
#include <cmath>
#include <memory>

#include "QtCompat.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

#include "VideoSidecar.h"
#ifdef HAS_OPENCV_FACE_CAPTURE
#include "OpenCVFrameSource.h"
#endif

namespace {
enum class FfmpegResult { Succeeded, Failed, Cancelled };

FfmpegResult runFfmpegVideoSegment(
        const VideoSegmentExportRequest& request,
        const QString& temporaryVideo,
        const std::function<bool(int, int)>& progress) {
    const QString ffmpeg =
            QStandardPaths::findExecutable(QStringLiteral("ffmpeg"));
    if (ffmpeg.isEmpty()) return FfmpegResult::Failed;
    QProcess process;
    process.setProcessChannelMode(QProcess::MergedChannels);
    QStringList arguments{
            QStringLiteral("-nostdin"),
            QStringLiteral("-hide_banner"),
            QStringLiteral("-loglevel"),
            QStringLiteral("error"),
            QStringLiteral("-y"),
            QStringLiteral("-ss"),
            QString::number(request.startMs / 1000.0, 'f', 3),
            QStringLiteral("-i"),
            request.videoPath,
            QStringLiteral("-t"),
            QString::number((request.endMs - request.startMs) / 1000.0, 'f', 3),
            QStringLiteral("-map"),
            QStringLiteral("0:v:0")};
    if (request.reEncodeVideo)
        arguments << QStringLiteral("-c:v") << QStringLiteral("libx264")
                  << QStringLiteral("-preset") << QStringLiteral("veryfast")
                  << QStringLiteral("-crf") << QStringLiteral("23")
                  << QStringLiteral("-an");
    else
        arguments << QStringLiteral("-c") << QStringLiteral("copy")
                  << QStringLiteral("-an")
                  << QStringLiteral("-avoid_negative_ts")
                  << QStringLiteral("make_zero");
    arguments << temporaryVideo;
    process.start(ffmpeg, arguments);
    if (!process.waitForStarted(5000)) return FfmpegResult::Failed;
    while (process.state() != QProcess::NotRunning) {
        process.waitForFinished(50);
        process.readAll();
        if (progress && !progress(0, 1)) {
            process.kill();
            process.waitForFinished();
            QFile::remove(temporaryVideo);
            return FfmpegResult::Cancelled;
        }
    }
    const bool succeeded = process.exitStatus() == QProcess::NormalExit &&
                           process.exitCode() == 0 &&
                           QFileInfo(temporaryVideo).size() > 0;
    if (!succeeded) QFile::remove(temporaryVideo);
    return succeeded ? FfmpegResult::Succeeded : FfmpegResult::Failed;
}

bool finalizeVideo(const QString& temporaryVideo,
                   const QString& outputPath,
                   QString* error) {
    if (!QFileInfo::exists(outputPath)) {
        if (QFile::rename(temporaryVideo, outputPath)) return true;
    } else {
        QFile source(temporaryVideo);
        QSaveFile destination(outputPath);
        if (source.open(QIODevice::ReadOnly) &&
            destination.open(QIODevice::WriteOnly)) {
            bool copied = true;
            while (!source.atEnd()) {
                const QByteArray chunk = source.read(1024 * 1024);
                if (chunk.isEmpty() ||
                    destination.write(chunk) != chunk.size()) {
                    copied = false;
                    break;
                }
            }
            if (copied && destination.commit()) {
                source.close();
                QFile::remove(temporaryVideo);
                return true;
            }
        }
    }
    QFile::remove(temporaryVideo);
    if (error)
        *error = QStringLiteral("Cannot finalize video clip: %1")
                         .arg(outputPath);
    return false;
}
}  // namespace

bool exportVisualizationVideo(
        const QString& outputPath,
        int fps,
        int frameCount,
        const std::function<QImage(int, QString*)>& renderFrame,
        const std::function<bool(int, int)>& progress,
        QString* error) {
    if (error) error->clear();
    const auto fail = [error](const QString& message) {
        if (error) *error = message;
        return false;
    };
#ifndef HAS_OPENCV_FACE_CAPTURE
    Q_UNUSED(outputPath)
    Q_UNUSED(fps)
    Q_UNUSED(frameCount)
    Q_UNUSED(renderFrame)
    Q_UNUSED(progress)
    return fail(
            QStringLiteral("OpenCV video export is unavailable in this build"));
#else
    if (QFileInfo(outputPath)
                        .suffix()
                        .compare(QStringLiteral("mp4"), Qt::CaseInsensitive) !=
                0 ||
        fps < 1 || fps > 120 || frameCount <= 0 || !renderFrame)
        return fail(
                QStringLiteral("Invalid MP4 destination, FPS or frame count"));
    const auto parent = QFileInfo(outputPath).absoluteDir();
    if (!parent.exists())
        return fail(QStringLiteral("Output folder does not exist"));
    QTemporaryDir temporary(
            parent.filePath(QStringLiteral(".qany-video-XXXXXX")));
    if (!temporary.isValid())
        return fail(QStringLiteral("Cannot create temporary video"));
    const QString encoded =
            temporary.filePath(QStringLiteral("visualization.mp4"));
    cv::VideoWriter writer;
    QSize size;
    try {
        for (int index = 0; index < frameCount; ++index) {
            if (progress && !progress(index, frameCount))
                return fail(QStringLiteral("Export cancelled"));
            QString renderError;
            const QImage image =
                    renderFrame(index, &renderError)
                            .convertToFormat(QImage::Format_RGB888);
            if (image.isNull())
                return fail(
                        renderError.isEmpty()
                                ? QStringLiteral("Unable to render frame %1")
                                          .arg(index + 1)
                                : renderError);
            if (index == 0) {
                size = image.size();
                // MP4 encoders truncate odd dimensions. Reject instead of
                // silently dropping the final row or column of annotations.
                if (size.width() % 2 || size.height() % 2)
                    return fail(QStringLiteral(
                            "MP4 frames require even width and height"));
                writer.open(encoded.toStdString(),
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                            cv::Size(size.width(), size.height()));
                if (!writer.isOpened())
                    return fail(QStringLiteral(
                            "Current environment cannot write MP4 files"));
            } else if (image.size() != size) {
                return fail(QStringLiteral(
                        "All video frames must have the same dimensions"));
            }
            cv::Mat rgb(image.height(), image.width(), CV_8UC3,
                        const_cast<uchar*>(image.constBits()),
                        image.bytesPerLine());
            cv::Mat bgr;
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            writer.write(bgr);
        }
        writer.release();
        if (progress && !progress(frameCount, frameCount))
            return fail(QStringLiteral("Export cancelled"));
        if (QFileInfo(encoded).size() <= 0)
            return fail(QStringLiteral("Video encoder produced no output"));
        return finalizeVideo(encoded, outputPath, error);
    } catch (const std::exception& exception) {
        return fail(QString::fromLocal8Bit(exception.what()));
    }
#endif
}

bool exportVideoSegment(const VideoSegmentExportRequest& request,
                        const std::function<bool(int, int)>& progress,
                        QString* error,
                        bool* usedStreamCopyFallback,
                        bool* usedFfmpegFallback) {
    if (usedStreamCopyFallback) *usedStreamCopyFallback = false;
    if (usedFfmpegFallback) *usedFfmpegFallback = false;
    const auto fail = [error](const QString& message) {
        if (error) *error = message;
        return false;
    };
#ifndef HAS_OPENCV_FACE_CAPTURE
    Q_UNUSED(request)
    Q_UNUSED(progress)
    return fail(
            QStringLiteral("OpenCV video export is unavailable in this build"));
#else
    if (request.videoPath.isEmpty() || request.outputPath.isEmpty() ||
        !std::isfinite(request.fps) || request.fps <= 0 ||
        request.endMs <= request.startMs)
        return fail(
                QStringLiteral("Invalid video export range or destination"));
    if (request.format == VideoSegmentExportFormat::RawFrames &&
        QFileInfo::exists(request.outputPath))
        return fail(QStringLiteral("Output already exists: %1")
                            .arg(request.outputPath));
    const QFileInfo outputInfo(request.outputPath);
    if (!QDir().mkpath(outputInfo.absolutePath()))
        return fail(QStringLiteral("Cannot create output folder: %1")
                            .arg(outputInfo.absolutePath()));

    QString temporaryVideo;
    const auto removeTemporaryVideo = [&temporaryVideo]() {
        if (!temporaryVideo.isEmpty()) QFile::remove(temporaryVideo);
    };
    QString extension;
    if (request.format == VideoSegmentExportFormat::Video) {
        extension = outputInfo.suffix().toLower();
        if (extension != QStringLiteral("avi") &&
            extension != QStringLiteral("mp4"))
            return fail(QStringLiteral("Export video must be .avi or .mp4"));
        temporaryVideo = outputInfo.absoluteDir().filePath(
                QStringLiteral(".%1.%2.%3")
                        .arg(outputInfo.completeBaseName(),
                             qtCompatUuidWithoutBraces(QUuid::createUuid()),
                             extension));
        if (progress && !progress(0, 1))
            return fail(QStringLiteral("Export cancelled"));
        const FfmpegResult encoded =
                runFfmpegVideoSegment(request, temporaryVideo, progress);
        if (encoded == FfmpegResult::Cancelled)
            return fail(QStringLiteral("Export cancelled"));
        if (encoded == FfmpegResult::Succeeded) {
            if (progress && !progress(1, 1)) {
                removeTemporaryVideo();
                return fail(QStringLiteral("Export cancelled"));
            }
            if (!finalizeVideo(temporaryVideo, request.outputPath, error))
                return false;
            if (error) error->clear();
            return true;
        }
        if (request.reEncodeVideo) {
            if (usedFfmpegFallback) *usedFfmpegFallback = true;
        } else if (usedStreamCopyFallback) {
            *usedStreamCopyFallback = true;
        }
    }

    OpenCVFrameSource source;
    if (!source.openVideo(request.videoPath.toStdString()))
        return fail(QStringLiteral("Cannot open source video: %1")
                            .arg(request.videoPath));
    const int totalFrames = std::max(0, int(source.frameCount()));
    const int startFrame = videoFrameAt(request.startMs, request.fps);
    const int endFrame = std::min(
            totalFrames,
            std::max(startFrame + 1, videoFrameAt(request.endMs, request.fps)));
    if (startFrame >= totalFrames || endFrame <= startFrame)
        return fail(QStringLiteral("Selected range has no decodable frames"));
    if (!source.seekToFrame(startFrame))
        return fail(QStringLiteral("Cannot seek to the selected start frame"));

    std::unique_ptr<QTemporaryDir> temporaryFrames;
    QString workDir;
    cv::VideoWriter writer;
    if (request.format == VideoSegmentExportFormat::RawFrames) {
        temporaryFrames.reset(
                new QTemporaryDir(outputInfo.absoluteDir().filePath(
                        QStringLiteral(".%1.XXXXXX")
                                .arg(outputInfo.fileName()))));
        if (!temporaryFrames->isValid())
            return fail(QStringLiteral("Cannot create temporary frame folder"));
        workDir = temporaryFrames->path();
    } else {
        const int codec = extension == QStringLiteral("avi")
                                  ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                                  : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        const int width = source.width();
        const int height = source.height();
        if (width <= 0 || height <= 0)
            return fail(QStringLiteral("Source video has invalid dimensions"));
        writer.open(temporaryVideo.toStdString(), codec, request.fps,
                    cv::Size(width, height));
        if (!writer.isOpened())
            return fail(QStringLiteral("Video encoder unavailable for .%1")
                                .arg(extension));
    }

    const int frameCount = endFrame - startFrame;
    const double outputFps =
            request.rawFrameFps > 0 ? request.rawFrameFps : request.fps;
    const double desiredFrames =
            std::ceil(frameCount * outputFps / request.fps);
    if (request.format == VideoSegmentExportFormat::RawFrames &&
        (!std::isfinite(outputFps) || outputFps <= 0 ||
         !std::isfinite(desiredFrames) || desiredFrames > 1000000))
        return fail(
                QStringLiteral("Invalid raw frame rate or excessive output"));
    const int outputFrames =
            request.format == VideoSegmentExportFormat::RawFrames
                    ? std::max(1, int(desiredFrames))
                    : frameCount;
    int writtenFrames = 0;
    try {
        for (int index = 0; index < frameCount; ++index) {
            if (progress && !progress(index, frameCount)) {
                writer.release();
                removeTemporaryVideo();
                return fail(QStringLiteral("Export cancelled"));
            }
            cv::Mat frame;
            if (source.read(frame, nullptr) != IFrameSource::ReadResult::Ok ||
                frame.empty()) {
                writer.release();
                removeTemporaryVideo();
                return fail(QStringLiteral("Failed to read source frame %1")
                                    .arg(startFrame + index));
            }
            if (request.format == VideoSegmentExportFormat::RawFrames) {
                while (writtenFrames < outputFrames &&
                       std::min(frameCount - 1,
                                int(std::floor(writtenFrames * request.fps /
                                               outputFps))) == index) {
                    const QString path = QDir(workDir).filePath(
                            QStringLiteral("img_%1.jpg")
                                    .arg(writtenFrames + 1, 5, 10,
                                         QLatin1Char('0')));
                    ++writtenFrames;
                    if (!cv::imwrite(path.toStdString(), frame))
                        return fail(QStringLiteral("Failed to write frame %1")
                                            .arg(writtenFrames));
                }
            } else {
                writer.write(frame);
            }
        }
    } catch (const cv::Exception& exception) {
        writer.release();
        removeTemporaryVideo();
        return fail(QString::fromUtf8(exception.what()));
    }
    writer.release();
    if (progress && !progress(frameCount, frameCount)) {
        removeTemporaryVideo();
        return fail(QStringLiteral("Export cancelled"));
    }
    if (request.format == VideoSegmentExportFormat::RawFrames) {
        if (!QDir().rename(workDir, request.outputPath))
            return fail(QStringLiteral("Cannot finalize frame folder: %1")
                                .arg(request.outputPath));
    } else {
        if (!finalizeVideo(temporaryVideo, request.outputPath, error))
            return false;
    }
    if (error) error->clear();
    return true;
#endif
}
