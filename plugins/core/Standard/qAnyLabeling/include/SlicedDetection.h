// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QRect>
#include <QRectF>
#include <QSize>
#include <QString>
#include <QVector>

struct SlicedDetectionBox {
    QRectF bounds;
    float score = 0;
    int classId = -1;
    QString label;
};

QVector<QRect> detectionSlices(const QSize& imageSize,
                               const QSize& sliceSize,
                               double overlap);

QVector<SlicedDetectionBox> mergeSlicedDetections(
        const QVector<SlicedDetectionBox>& detections, double iosThreshold);
