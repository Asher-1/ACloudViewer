// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifdef BUILD_CUDA_MODULE

#include "cloudViewer/core/CUDAUtils.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace cloudViewer {
namespace tests {

TEST(CUDAState, InitState) {
    if (!IsCUDAAvailable()) {
        GTEST_SKIP() << "CUDA not available. Set BUILD_CUDA_MODULE=ON to "
                        "enable CUDA support.";
    }
    const int device_count = core::cuda::DeviceCount();
    const core::CUDAState& cuda_state = core::CUDAState::GetInstance();
    utility::LogInfo("Number of CUDA devices: {}", device_count);
    for (int i = 0; i < device_count; ++i) {
        for (int j = 0; j < device_count; ++j) {
            utility::LogInfo("P2PEnabled {}->{}: {}", i, j,
                             cuda_state.IsP2PEnabled(i, j));
        }
    }
}

}  // namespace tests
}  // namespace cloudViewer

#endif
