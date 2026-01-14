// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>

// #include "cloudViewer/utility/CPUInfo.h"  // TODO: Fix missing headers
// #include "cloudViewer/utility/CompilerInfo.h"  // TODO: Fix missing headers
// #include "cloudViewer/utility/ISAInfo.h"  // TODO: Fix missing headers
#include <Logging.h>

#ifdef BUILD_CUDA_MODULE
#include "cloudViewer/core/CUDAUtils.h"
// #include "cloudViewer/core/CUDAState.h"  // TODO: Fix missing header
#endif

#include "tests/Tests.h"

#ifdef BUILD_CUDA_MODULE
/// Returns true if --disable_p2p flag is used.
bool ShallDisableP2P(int argc, char** argv) {
    bool shall_disable_p2p = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--disable_p2p") == 0) {
            shall_disable_p2p = true;
            break;
        }
    }
    return shall_disable_p2p;
}
#endif

int main(int argc, char** argv) {
    using namespace cloudViewer;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    // utility::CompilerInfo::GetInstance().Print();  // TODO: Fix missing
    // headers utility::CPUInfo::GetInstance().Print();  // TODO: Fix missing
    // headers

    // Print ISA info if available
    // if (utility::ISAInfo::GetInstance().IsSupported()) {  // TODO: Fix
    // missing headers
    //     utility::ISAInfo::GetInstance().Print();
    // }

#ifdef BUILD_CUDA_MODULE
    // if (ShallDisableP2P(argc, argv)) {  // TODO: Fix CUDAState API
    //     core::CUDAState::GetInstance().ForceDisableP2PForTesting();
    //     utility::LogInfo("P2P device transfer has been disabled.");
    // }
#endif

    testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}
