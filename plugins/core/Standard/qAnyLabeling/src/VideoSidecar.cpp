// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoSidecar.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr auto kType = "video_classification";
constexpr auto kVersion = "1.0.0";

bool integerField(const QJsonObject& object, const QString& key) {
    const QJsonValue value = object.value(key);
    return value.isUndefined() ||
           (value.isDouble() && std::isfinite(value.toDouble()) &&
            std::trunc(value.toDouble()) == value.toDouble());
}
}  // namespace

int videoFrameAt(qint64 ms, double fps) {
    if (!std::isfinite(fps) || fps <= 0) return 0;
    return int(std::clamp(std::round(double(ms) * fps / 1000.0), 0.0,
                          double(std::numeric_limits<int>::max())));
}

QString videoSidecarPath(const QString& videoPath) {
    if (videoPath.isEmpty()) return {};
    const QFileInfo info(videoPath);
    return info.dir().filePath(info.completeBaseName() +
                               QStringLiteral(".json"));
}

QJsonObject VideoSidecar::toJson() const {
    QJsonArray labelArray;
    for (const QString& label : labels) labelArray.append(label);
    QJsonArray segmentArray;
    for (const VideoSegment& segment : segments) {
        segmentArray.append(QJsonObject{
                {QStringLiteral("id"), segment.id},
                {QStringLiteral("label"), segment.label},
                {QStringLiteral("start_ms"), double(segment.startMs)},
                {QStringLiteral("end_ms"), double(segment.endMs)},
                {QStringLiteral("start_frame"), segment.startFrame},
                {QStringLiteral("end_frame"), segment.endFrame},
                {QStringLiteral("description"), segment.description}});
    }
    return {{QStringLiteral("version"), QString::fromLatin1(kVersion)},
            {QStringLiteral("type"), QString::fromLatin1(kType)},
            {QStringLiteral("video"), video},
            {QStringLiteral("fps"), fps},
            {QStringLiteral("duration_ms"), double(durationMs)},
            {QStringLiteral("width"), width},
            {QStringLiteral("height"), height},
            {QStringLiteral("labels"), labelArray},
            {QStringLiteral("label_colors"), labelColors},
            {QStringLiteral("segments"), segmentArray}};
}

bool VideoSidecar::fromJson(const QJsonObject& json,
                            VideoSidecar* out,
                            QString* error) {
    const auto fail = [error](const QString& message) {
        if (error) *error = message;
        return false;
    };
    if (!out) return fail(QStringLiteral("No sidecar destination"));
    const QJsonValue type = json.value(QStringLiteral("type"));
    if ((!type.isUndefined() && type.toString() != QLatin1String(kType)) ||
        json.contains(QStringLiteral("shapes")) ||
        (!json.contains(QStringLiteral("segments")) &&
         !json.contains(QStringLiteral("video")) &&
         !json.contains(QStringLiteral("label_colors"))))
        return fail(QStringLiteral(
                "This JSON belongs to another annotation workflow"));
    const QJsonValue version = json.value(QStringLiteral("version"));
    if (!version.isUndefined() && version.toString() != QLatin1String(kVersion))
        return fail(QStringLiteral("Unsupported video sidecar version"));
    if ((!json.value(QStringLiteral("labels")).isUndefined() &&
         !json.value(QStringLiteral("labels")).isArray()) ||
        (!json.value(QStringLiteral("label_colors")).isUndefined() &&
         !json.value(QStringLiteral("label_colors")).isObject()) ||
        (!json.value(QStringLiteral("segments")).isUndefined() &&
         !json.value(QStringLiteral("segments")).isArray()))
        return fail(QStringLiteral("Invalid video sidecar collections"));
    for (const QString& key :
         {QStringLiteral("duration_ms"), QStringLiteral("width"),
          QStringLiteral("height")})
        if (!integerField(json, key))
            return fail(QStringLiteral("Invalid %1").arg(key));

    VideoSidecar result;
    result.video = json.value(QStringLiteral("video")).toString();
    result.fps = json.value(QStringLiteral("fps")).toDouble();
    result.durationMs =
            qint64(json.value(QStringLiteral("duration_ms")).toDouble());
    result.width = json.value(QStringLiteral("width")).toInt();
    result.height = json.value(QStringLiteral("height")).toInt();
    if (!std::isfinite(result.fps) || result.fps < 0 || result.durationMs < 0 ||
        result.width < 0 || result.height < 0)
        return fail(QStringLiteral("Invalid video metadata"));
    for (const QJsonValue& value :
         json.value(QStringLiteral("labels")).toArray()) {
        if (!value.isString()) return fail(QStringLiteral("Invalid label"));
        result.labels.append(value.toString());
    }
    result.labelColors = json.value(QStringLiteral("label_colors")).toObject();
    for (const QJsonValue& value : result.labelColors)
        if (!value.isString())
            return fail(QStringLiteral("Invalid label color"));
    for (const QJsonValue& value :
         json.value(QStringLiteral("segments")).toArray()) {
        if (!value.isObject()) return fail(QStringLiteral("Invalid segment"));
        const QJsonObject object = value.toObject();
        for (const QString& key :
             {QStringLiteral("start_ms"), QStringLiteral("end_ms"),
              QStringLiteral("start_frame"), QStringLiteral("end_frame")})
            if (!integerField(object, key))
                return fail(QStringLiteral("Invalid segment %1").arg(key));
        VideoSegment segment;
        segment.id = object.value(QStringLiteral("id")).toString();
        segment.label = object.value(QStringLiteral("label")).toString();
        segment.startMs =
                qint64(object.value(QStringLiteral("start_ms")).toDouble());
        segment.endMs =
                qint64(object.value(QStringLiteral("end_ms")).toDouble());
        segment.startFrame =
                object.value(QStringLiteral("start_frame")).toInt();
        segment.endFrame = object.value(QStringLiteral("end_frame")).toInt();
        segment.description =
                object.value(QStringLiteral("description")).toString();
        if (segment.startMs < 0 || segment.endMs < segment.startMs ||
            segment.startFrame < 0 || segment.endFrame < segment.startFrame)
            return fail(QStringLiteral("Invalid segment range"));
        result.segments.append(segment);
    }
    *out = result;
    return true;
}

bool loadVideoSidecar(const QString& videoPath,
                      VideoSidecar* out,
                      QString* error) {
    const QString path = videoSidecarPath(videoPath);
    if (!QFileInfo::exists(path)) {
        if (out) *out = VideoSidecar{};
        return true;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = file.errorString();
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document =
            QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error)
            *error = QStringLiteral("Invalid JSON: %1")
                             .arg(parseError.errorString());
        return false;
    }
    return VideoSidecar::fromJson(document.object(), out, error);
}

bool saveVideoSidecar(const QString& videoPath,
                      const VideoSidecar& sidecar,
                      QString* error) {
    const QString path = videoSidecarPath(videoPath);
    VideoSidecar existing;
    if (!loadVideoSidecar(videoPath, &existing, error)) return false;
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        if (error) *error = file.errorString();
        return false;
    }
    const QByteArray payload =
            QJsonDocument(sidecar.toJson()).toJson(QJsonDocument::Indented);
    if (file.write(payload) != payload.size() || !file.commit()) {
        if (error) *error = file.errorString();
        return false;
    }
    return true;
}
