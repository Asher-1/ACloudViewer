// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QJsonObject>
#include <QString>

// Default for newly created sidecars at the pinned X-AnyLabeling baseline.
// The normal editor preserves explicit version fields in existing sidecars.
inline QString annotationFormatVersion() { return QStringLiteral("4.1.3"); }

struct AnnotationDocument {
    QImage image;
    QJsonObject json;
    QString imagePath;
    QString labelPath;
    QString error;
};

// Resolve and decode everything before the caller replaces its current state.
// An image input overrides stale imagePath metadata; a JSON input resolves
// relative to that JSON, then tries the local basename. Embedded pixels win.
AnnotationDocument loadAnnotationDocument(const QString& filename,
                                          const QString& outputDirectory = {},
                                          bool autoTransform = true);
