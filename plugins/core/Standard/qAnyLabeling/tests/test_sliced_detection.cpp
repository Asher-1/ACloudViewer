// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QSize>
#include <cstdio>

#include "SlicedDetection.h"

namespace {
bool check(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}
}  // namespace

int main() {
    const QVector<QRect> slices =
            detectionSlices(QSize(1024, 768), QSize(512, 512), 0.2);
    if (!check(slices.size() == 6, "SAHI slice count is not source-compatible"))
        return 1;
    if (!check(slices.front() == QRect(0, 0, 512, 512),
               "First slice geometry is incorrect"))
        return 1;
    if (!check(slices.back().right() == 1023 && slices.back().bottom() == 767,
               "Edge slice does not cover the source image"))
        return 1;

    const QVector<SlicedDetectionBox> input = {
            {QRectF(10, 10, 100, 100), 0.9f, 2, QStringLiteral("car")},
            {QRectF(20, 20, 100, 100), 0.7f, 2, QStringLiteral("car")},
            {QRectF(20, 20, 100, 100), 0.99f, 3, QStringLiteral("person")}};
    const QVector<SlicedDetectionBox> output =
            mergeSlicedDetections(input, 0.5);
    if (!check(output.size() == 2,
               "Same-class overlapping tiles were not merged"))
        return 1;
    int carIndex = -1;
    int personIndex = -1;
    for (int i = 0; i < output.size(); ++i) {
        if (output[i].classId == 2) carIndex = i;
        if (output[i].classId == 3) personIndex = i;
    }
    if (!check(carIndex >= 0 &&
                       output[carIndex].bounds == QRectF(10, 10, 110, 110),
               "Merged box is not the union of the tile boxes"))
        return 1;
    if (!check(personIndex >= 0 && personIndex != carIndex,
               "Class-specific merge crossed categories"))
        return 1;
    return 0;
}
