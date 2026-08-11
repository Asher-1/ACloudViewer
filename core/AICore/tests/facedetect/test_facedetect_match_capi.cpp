// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cmath>
#include <cstdio>

#include "aicore/facedetect_capi.h"

int main() {
    const float inv = 1.0f / std::sqrt(2.0f);
    const float queries[] = {1.f, 0.f, 0.f, inv, inv, 0.f};
    const float gallery[] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    float distances[6] = {};
    if (aicore_facedetect_cosine_distance_matrix(queries, 2, gallery, 3, 3,
                                                 distances) != 0) {
        std::fprintf(stderr, "distance matrix call failed\n");
        return 1;
    }
    if (std::abs(distances[0]) > 1e-6f ||
        std::abs(distances[1] - 1.f) > 1e-6f ||
        std::abs(distances[3] - (1.f - inv)) > 1e-6f ||
        std::abs(distances[4] - (1.f - inv)) > 1e-6f) {
        std::fprintf(stderr, "unexpected cosine distances\n");
        return 2;
    }
    if (aicore_facedetect_cosine_distance_matrix(nullptr, 1, gallery, 3, 3,
                                                 distances) == 0) {
        std::fprintf(stderr, "invalid input was accepted\n");
        return 3;
    }
    return 0;
}
