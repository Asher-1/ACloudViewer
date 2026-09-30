// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SlicedDetection.h"

#include <algorithm>
#include <numeric>

namespace {
double intersectionOverSmaller(const QRectF& a, const QRectF& b) {
    const double smaller =
            std::min(a.width() * a.height(), b.width() * b.height());
    if (smaller <= 0) return 0.0;
    const QRectF intersection = a.intersected(b);
    return std::max(0.0, intersection.width()) *
           std::max(0.0, intersection.height()) / smaller;
}
}  // namespace

QVector<QRect> detectionSlices(const QSize& imageSize,
                               const QSize& sliceSize,
                               double overlap) {
    QVector<QRect> slices;
    if (imageSize.isEmpty() || sliceSize.isEmpty() || overlap < 0.0 ||
        overlap >= 1.0)
        return slices;
    const int xStep =
            std::max(1, sliceSize.width() - int(overlap * sliceSize.width()));
    const int yStep =
            std::max(1, sliceSize.height() - int(overlap * sliceSize.height()));
    for (int y = 0; y < imageSize.height(); y += yStep) {
        for (int x = 0; x < imageSize.width(); x += xStep) {
            const int right =
                    std::min(imageSize.width(), x + sliceSize.width());
            const int bottom =
                    std::min(imageSize.height(), y + sliceSize.height());
            const int left = std::max(0, right - sliceSize.width());
            const int top = std::max(0, bottom - sliceSize.height());
            const QRect slice(left, top, right - left, bottom - top);
            if (!slices.contains(slice)) slices.append(slice);
        }
    }
    return slices;
}

QVector<SlicedDetectionBox> mergeSlicedDetections(
        const QVector<SlicedDetectionBox>& detections, double iosThreshold) {
    QVector<SlicedDetectionBox> merged;
    if (iosThreshold <= 0.0 || iosThreshold > 1.0) return detections;
    QVector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        // SAHI batched_greedy_nmm iterates sorted category ids first, then
        // applies confidence ordering inside each category.
        if (detections[a].classId != detections[b].classId)
            return detections[a].classId < detections[b].classId;
        return detections[a].score > detections[b].score;
    });

    // SAHI GREEDYNMM compares each winner with all remaining same-class boxes.
    // O(n^2) is inherent to this direct matching stage; per-tile detector NMS
    // already bounds the input count.
    while (!order.isEmpty()) {
        const int winner = order.takeFirst();
        SlicedDetectionBox combined = detections[winner];
        QVector<int> unmatched;
        for (int index : order) {
            const SlicedDetectionBox& candidate = detections[index];
            if (candidate.classId != combined.classId ||
                intersectionOverSmaller(detections[winner].bounds,
                                        candidate.bounds) < iosThreshold) {
                unmatched.append(index);
                continue;
            }
            if (intersectionOverSmaller(combined.bounds, candidate.bounds) >
                iosThreshold)
                combined.bounds = combined.bounds.united(candidate.bounds);
        }
        order = std::move(unmatched);
        merged.append(std::move(combined));
    }
    return merged;
}
