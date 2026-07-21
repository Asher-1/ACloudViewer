// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/aicore.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore::depth::ImageDepth::isAvailable());
    AICORE_CHECK(aicore_depth_abi_version() >= 4);
    AICORE_CHECK(aicore_gaussian_abi_version() >= 1);
    std::fprintf(stderr, "aicore umbrella + ImageDepth ok\n");
    return failures == 0 ? 0 : 1;
}
