// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QJsonObject>
#include <QStringList>
#include <QVector>

enum class YoloFormat { Boxes, Segments, OrientedBoxes, Pose };

struct PoseClass {
    QString name;
    QStringList keypoints;
};

struct PoseConfig {
    QVector<PoseClass> classes;
    bool hasVisible = true;
    QString error;
};

PoseConfig parsePoseConfig(const QByteArray& yaml);

struct YoloConversion {
    QByteArray text;
    int count = 0;
    QString error;
};

YoloConversion convertYolo(const QJsonObject& document,
                           const QStringList& classes,
                           YoloFormat format,
                           bool keepOutOfBounds = true,
                           const PoseConfig* poseConfig = nullptr);
YoloConversion convertYoloPose(const QJsonObject& document,
                               const PoseConfig& config);
YoloConversion convertDota(const QJsonObject& document);
YoloConversion convertVoc(const QJsonObject& document,
                          const QString& imageName,
                          const QString& folder,
                          bool polygons);

struct CocoImage {
    QString fileName;
    QJsonObject document;
};

struct CocoConversion {
    QJsonObject document;
    int count = 0;
    QString error;
};

CocoConversion convertCoco(const QVector<CocoImage>& images,
                           const QStringList& classes,
                           bool polygons);
CocoConversion convertCocoPose(const QVector<CocoImage>& images,
                               const PoseConfig& config);
