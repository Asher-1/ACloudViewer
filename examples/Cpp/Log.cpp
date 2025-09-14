// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>

#include "CloudViewer.h"

int main(int argc, char **argv) {
    using namespace cloudViewer;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    utility::LogDebug("This Debug message should be visible, {} {:.2f}",
                      "format:", 0.42001);
    utility::LogInfo("This Info message should be visible, {} {:.2f}",
                     "format:", 0.42001);
    utility::LogWarning("This Warning message should be visible, {} {:.2f}",
                        "format:", 0.42001);

    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);

    utility::LogDebug("This Debug message should NOT be visible, {} {:.2f}",
                      "format:", 0.42001);
    utility::LogInfo("This Info message should be visible, {} {:.2f}",
                     "format:", 0.42001);
    utility::LogWarning("This Warning message should be visible, {} {:.2f}",
                        "format:", 0.42001);

    try {
        utility::LogError("This Error exception is catched");
    } catch (const std::exception &e) {
        utility::LogInfo("Catched exception msg: {}", e.what());
    }
    utility::LogInfo("This Info message shall print in regular color");
    utility::LogError("This Error message terminates the program");
    utility::LogError("This Error message should NOT be visible");

    return 0;
}
