// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QString>
#include <functional>

enum class VideoSegmentExportFormat { Video, RawFrames };

struct VideoSegmentExportRequest {
    QString videoPath;
    QString outputPath;
    qint64 startMs = 0;
    qint64 endMs = 0;
    double fps = 0;
    VideoSegmentExportFormat format = VideoSegmentExportFormat::Video;
    int rawFrameFps = 0;  // 0 = write every decoded source frame
    bool reEncodeVideo = false;
};

// The interval is [startFrame, endFrame). The callback returns false to cancel.
bool exportVideoSegment(const VideoSegmentExportRequest& request,
                        const std::function<bool(int, int)>& progress,
                        QString* error,
                        bool* usedStreamCopyFallback = nullptr,
                        bool* usedFfmpegFallback = nullptr);

// Stream one rendered frame at a time. Failure/cancellation preserves an
// existing destination; working memory is O(width * height), not frame count.
bool exportVisualizationVideo(
        const QString& outputPath,
        int fps,
        int frameCount,
        const std::function<QImage(int, QString*)>& renderFrame,
        const std::function<bool(int, int)>& progress,
        QString* error);
