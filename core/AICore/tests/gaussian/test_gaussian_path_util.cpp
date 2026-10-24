// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>

#include "common/model_cache.hpp"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    const std::string dir = aicore::gaussian_model_cache_dir();
    AICORE_CHECK(!dir.empty());
    AICORE_CHECK(dir.find("freesplatter_models") != std::string::npos);
    std::fprintf(stderr, "gaussian_model_cache_dir: %s\n", dir.c_str());
    return failures == 0 ? 0 : 1;
}
