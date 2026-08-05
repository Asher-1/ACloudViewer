// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <vector>

namespace lightglue {

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float scale = 1.0f;
    float orientation = 0.0f;
};

struct Features {
    std::vector<Keypoint> keypoints;
    std::vector<float> descriptors;
    int32_t descriptor_dim = 0;
    int32_t image_width = 0;
    int32_t image_height = 0;
};

}  // namespace lightglue
