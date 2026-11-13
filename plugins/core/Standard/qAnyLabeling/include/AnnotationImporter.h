// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QHash>
#include <QImage>
#include <QJsonObject>
#include <QSize>
#include <QVector>

#include "AnnotationExporter.h"

struct YoloImport {
    QJsonObject document;
    QString error;
    int count = 0;
};

YoloImport importYolo(const QByteArray& labels,
                      const QStringList& classes,
                      YoloFormat format,
                      const QString& imageName,
                      const QSize& size,
                      const PoseConfig* poseConfig = nullptr);

struct AnnotationImport {
    QJsonObject document;
    QString error;
    int count = 0;
};

AnnotationImport importVoc(const QByteArray& xml,
                           const QString& imageName,
                           bool polygons);
AnnotationImport importDota(const QByteArray& text,
                            const QString& imageName,
                            const QSize& size);

enum class CocoFormat { Boxes, Segments, Pose };

struct CocoImport {
    QHash<QString, QJsonObject> documents;
    QString error;
    int count = 0;
};

CocoImport importCoco(const QByteArray& json, CocoFormat format);
AnnotationImport importMask(const QImage& mask,
                            const QJsonObject& colorMap,
                            const QString& imageName,
                            const QSize& imageSize);
