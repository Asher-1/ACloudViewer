// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QFile>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <cstdio>

#include "VideoSidecar.h"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir folder;
    if (!folder.isValid()) return 1;
    const QString video = folder.filePath(QStringLiteral("clip.mp4"));
    const QString sidecar = videoSidecarPath(video);
    VideoSidecar data;
    data.fps = 29.97;
    data.durationMs = 4000;
    data.width = 1920;
    data.height = 1080;
    data.video = QStringLiteral("clip.mp4");
    data.labels.append(QStringLiteral("running"));
    data.labelColors.insert(QStringLiteral("running"),
                            QStringLiteral("#EF476F"));
    data.segments.append(
            {QStringLiteral("s1234567890"), QStringLiteral("running"), 1000,
             2500, videoFrameAt(1000, data.fps), videoFrameAt(2500, data.fps),
             QStringLiteral("fast")});
    QString error;
    if (!saveVideoSidecar(video, data, &error)) {
        std::fprintf(stderr, "initial save failed: %s\n", qPrintable(error));
        return 1;
    }
    VideoSidecar loaded;
    if (!loadVideoSidecar(video, &loaded, &error) ||
        loaded.toJson() != data.toJson() || videoFrameAt(1000, 29.97) != 30) {
        std::fprintf(stderr, "video sidecar round-trip failed: %s\n",
                     qPrintable(error));
        return 1;
    }
    QFile legacy(sidecar);
    if (!legacy.open(QIODevice::WriteOnly | QIODevice::Truncate) ||
        legacy.write("{\"video\":\"clip.mp4\",\"segments\":[]}") < 0)
        return 1;
    legacy.close();
    if (!loadVideoSidecar(video, &loaded, &error) ||
        loaded.video != QStringLiteral("clip.mp4") ||
        !loaded.segments.isEmpty() || !loaded.labels.isEmpty() ||
        !saveVideoSidecar(video, loaded, &error)) {
        std::fprintf(stderr, "legacy video sidecar failed: %s\n",
                     qPrintable(error));
        return 1;
    }
    QFile foreign(sidecar);
    if (!foreign.open(QIODevice::WriteOnly | QIODevice::Truncate) ||
        foreign.write("{\"version\": \"5.0.1\", \"shapes\": []}") < 0)
        return 1;
    foreign.close();
    if (saveVideoSidecar(video, data, &error)) {
        std::fprintf(stderr, "image annotation sidecar overwritten\n");
        return 1;
    }
    if (!foreign.open(QIODevice::ReadOnly) ||
        !foreign.readAll().contains("shapes"))
        return 1;
    return 0;
}
