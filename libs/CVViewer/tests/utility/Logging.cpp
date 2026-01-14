// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/utility/Logging.h"

#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(Logging, LogError) {
    EXPECT_THROW(utility::LogError("Example exception message."),
                 std::runtime_error);
}

TEST(Logging, LogInfo) {
    utility::LogInfo("{}", "Example shape print {1, 2, 3}.");
}

}  // namespace tests
}  // namespace cloudViewer
