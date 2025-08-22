// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#ifdef BUILD_CUDA_MODULE
#include "core/CUDAState.cuh"
#endif

#include "utility/Console.h"
#include "tests/UnitTest.h"

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
#ifdef BUILD_CUDA_MODULE
    if (ShallDisableP2P(argc, argv)) {
        std::shared_ptr<cloudViewer::core::CUDAState> cuda_state =
                cloudViewer::core::CUDAState::GetInstance();
        cuda_state->ForceDisableP2PForTesting();
        cloudViewer::utility::LogInfo("P2P device transfer has been disabled.");
    }
#endif
    testing::InitGoogleTest(&argc, argv);
    cloudViewer::utility::SetVerbosityLevel(cloudViewer::utility::VerbosityLevel::Debug);
    return RUN_ALL_TESTS();
}
