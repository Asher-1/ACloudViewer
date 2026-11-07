// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QHash>
#include <QJsonObject>
#include <QSize>
#include <QString>
#include <QVector>
#include <functional>

struct Sam3ArchiveImage {
    QString filePath;
    QSize size;
};

struct Sam3ArchiveImport {
    QHash<QString, QJsonObject> documents;
    QString error;
    int count = 0;
};

// Convert the qSAM3 video archive into ordinary image sidecars. The exact
// binary mask is preserved in acloudviewer_mask_png; polygon points are only
// the editable outline used by the canvas.
Sam3ArchiveImport importSam3Archive(
        const QByteArray& manifest,
        const QString& archiveDirectory,
        const QVector<Sam3ArchiveImage>& images,
        const QString& labelPrefix = QStringLiteral("sam3_instance_"));

// Visit frames in video order. Empty documents indicate frames without masks.
Sam3ArchiveImport visitSam3Archive(
        const QByteArray& manifest,
        const QString& archiveDirectory,
        const QVector<Sam3ArchiveImage>& images,
        const std::function<bool(int, const QJsonObject&)>& visitor,
        const QString& labelPrefix = QStringLiteral("sam3_instance_"));
