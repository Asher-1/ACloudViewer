// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QJsonObject>
#include <QString>

#include "SAM3Types.h"

// Frame files are committed atomically before their manifest entry is added.
bool writeVideoMaskFrame(const QString& directory,
                         int frameIndex,
                         const SAM3WorkerResult& result,
                         QJsonObject* frameEntry);
bool writeVideoMaskManifest(const QString& directory,
                            const QJsonObject& manifest);

// Returns the lowercase SHA-256 digest of a source video. The archive
// importer uses this only when a manifest carries source_sha256, preserving
// compatibility with version-1 manifests created before content binding.
QString videoMaskArchiveSha256(const QString& videoPath,
                               QString* error = nullptr);
