// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

void NotImplemented() {
    std::cout << "\033[0;32m" << "[          ] " << "\033[0;0m";
    std::cout << "\033[0;31m" << "Not implemented." << "\033[0;0m" << std::endl;

    GTEST_NONFATAL_FAILURE_("Not implemented");
}

}  // namespace tests
}  // namespace cloudViewer
