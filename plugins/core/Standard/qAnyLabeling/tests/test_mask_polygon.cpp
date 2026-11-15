// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <vector>

#include "MaskPolygon.h"

int main() {
    constexpr int width = 12, height = 10, stride = 16;
    std::vector<uint8_t> mask(stride * height);
    for (int y = 1; y < 8; ++y)
        for (int x = 1; x < 9; ++x) mask[y * stride + x] = 1;
    for (int y = 3; y < 6; ++y)
        for (int x = 3; x < 7; ++x) mask[y * stride + x] = 0;
    mask[0] = 1;
    for (int y = 0; y < 2; ++y)
        for (int x = 10; x < 12; ++x) mask[y * stride + x] = 1;
    const QVector<QPointF> polygon =
            largestMaskPolygon(mask.data(), width, height, stride);
    if (polygon.size() != 4 || !polygon.contains(QPointF(1, 1)) ||
        !polygon.contains(QPointF(8, 7))) {
        std::fprintf(stderr, "Largest external mask contour failed\n");
        return 1;
    }
    if (maskPolygons(mask.data(), width, height, stride).size() != 2) {
        std::fprintf(stderr, "Multiple external mask contours were lost\n");
        return 1;
    }
    if (!largestMaskPolygon(nullptr, width, height, stride).isEmpty() ||
        !largestMaskPolygon(mask.data(), width, height, 2).isEmpty()) {
        std::fprintf(stderr, "Invalid mask view must fail cleanly\n");
        return 1;
    }
    return 0;
}
