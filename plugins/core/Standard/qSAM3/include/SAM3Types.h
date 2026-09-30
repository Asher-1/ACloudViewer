// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Shared SAM3 image/video result and prompt data types.
#pragma once

#include <aicore/sam3_capi.h>

#include <QImage>
#include <QString>
#include <QVector>
#include <vector>

struct SAM3WorkerResult {
    bool valid = false;
    int detCount = 0;
    QVector<aicore_sam3_box> boxes;
    QVector<float> scores;
    QVector<float> ious;
    QVector<int> instanceIds;
    QVector<QImage> instanceMasks;
    QImage maskComposite;
    aicore_sam3_timings timings{};
    QString errorMsg;
};

Q_DECLARE_METATYPE(SAM3WorkerResult)

struct SAM3Prompt {
    QVector<aicore_sam3_point> posPoints;
    QVector<aicore_sam3_point> negPoints;
    std::vector<aicore_sam3_box> posExemplars;
    std::vector<aicore_sam3_box> negExemplars;
    aicore_sam3_box pvsBox{};
    bool usePvsBox = false;
    bool multimask = false;
    char text[256] = {};
    float scoreThreshold = 0.5f;
    float nmsThreshold = 0.1f;
};
