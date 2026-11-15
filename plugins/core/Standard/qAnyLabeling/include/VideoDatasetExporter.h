// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>
#include <functional>

struct VideoSidecar;

struct VideoDatasetExportOptions {
    QString videoPath;
    QString outputDirectory;
    bool includeVideo = true;
    bool includeRawFrames = false;
    int rawFrameFps = 0;  // 0 = source FPS
    bool zipOutput = true;
    bool reEncodeVideo =
            false;  // false = stream copy, then re-encode on failure
};

// Exports every valid sidecar segment into a label-organized dataset. The
// callback returns false to cancel and receives completed/total progress units
// (1000 per exported clip or frame sequence).
bool exportVideoDataset(const VideoDatasetExportOptions& options,
                        const std::function<bool(int, int)>& progress,
                        QString* error,
                        QString* zipPath = nullptr,
                        const VideoSidecar* currentSidecar = nullptr);
