// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QString>
#include <functional>

// Extract every source frame and its SAM3 sidecar into a new directory. The
// directory is published only after decoding and annotation writes succeed.
bool extractSam3FrameWorkspace(const QByteArray& manifest,
                               const QString& archiveDirectory,
                               const QString& videoPath,
                               const QString& outputDirectory,
                               const std::function<bool(int, int)>& progress,
                               QString* error);
