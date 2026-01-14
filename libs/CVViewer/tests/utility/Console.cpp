// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/utility/Console.h"

#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(Logger, LogError) {
    EXPECT_THROW(utility::LogError("Example exception message."),
                 std::runtime_error);
}

TEST(Logger, LogInfo) {
    utility::LogInfo("{}", "Example shape print {1, 2, 3}.");
}

TEST(Console, DISABLED_SetVerbosityLevel) { NotImplemented(); }

TEST(Console, DISABLED_GetVerbosityLevel) { NotImplemented(); }

TEST(Console, DISABLED_PrintWarning) { NotImplemented(); }

TEST(Console, DISABLED_PrintInfo) { NotImplemented(); }

TEST(Console, DISABLED_PrintDebug) { NotImplemented(); }

TEST(Console, DISABLED_PrintAlways) { NotImplemented(); }

TEST(Console, DISABLED_ResetConsoleProgress) { NotImplemented(); }

TEST(Console, DISABLED_AdvanceConsoleProgress) { NotImplemented(); }

TEST(Console, DISABLED_GetCurrentTimeStamp) { NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsString) { NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsInt) { NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsDouble) { NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsEigenVectorXd) { NotImplemented(); }

TEST(Console, DISABLED_ProgramOptionExists) { NotImplemented(); }

TEST(Console, DISABLED_ProgramOptionExistsAny) { NotImplemented(); }

}  // namespace tests
}  // namespace cloudViewer
