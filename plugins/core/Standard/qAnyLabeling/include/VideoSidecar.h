// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QJsonObject>
#include <QStringList>
#include <QVector>

struct VideoSegment {
    QString id;
    QString label;
    qint64 startMs = 0;
    qint64 endMs = 0;
    int startFrame = 0;
    int endFrame = 0;
    QString description;
};

struct VideoSidecar {
    QString video;
    double fps = 0;
    qint64 durationMs = 0;
    int width = 0;
    int height = 0;
    QStringList labels;
    QJsonObject labelColors;
    QVector<VideoSegment> segments;

    QJsonObject toJson() const;
    static bool fromJson(const QJsonObject& json,
                         VideoSidecar* out,
                         QString* error);
};

QString videoSidecarPath(const QString& videoPath);
bool loadVideoSidecar(const QString& videoPath,
                      VideoSidecar* out,
                      QString* error);
bool saveVideoSidecar(const QString& videoPath,
                      const VideoSidecar& sidecar,
                      QString* error);
int videoFrameAt(qint64 ms, double fps);
