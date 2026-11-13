// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QString>

// Stream a directory to a new UTF-8/deflate ZIP, using 64 KiB chunks.
// Existing files are preserved; callers stage before replacing a destination.
// ZIP32 limits are checked and an incomplete archive is removed on failure.
bool writeZipDirectory(const QString& directory,
                       const QString& zipPath,
                       QString* error,
                       bool includeRootDirectory = true);
