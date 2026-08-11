// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>

#include "compute_mode.hpp"

int main() {
    if (aicore::depth::gpu_mode()) {
        std::fprintf(stderr, "gpu_mode should default to false\n");
        return 1;
    }
    aicore::depth::set_gpu_mode(true);
    if (!aicore::depth::gpu_mode()) {
        std::fprintf(stderr, "gpu_mode not set\n");
        return 1;
    }
    aicore::depth::set_gpu_mode(false);
    if (aicore::depth::gpu_mode()) {
        std::fprintf(stderr, "gpu_mode not cleared\n");
        return 1;
    }
    return 0;
}
