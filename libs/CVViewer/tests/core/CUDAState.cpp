// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifdef BUILD_CUDA_MODULE

#include "core/CUDAState.cuh"

#include "tests/UnitTest.h"

namespace cloudViewer {
namespace tests {

TEST(CUDAState, InitState) {
    std::shared_ptr<core::CUDAState> cuda_state =
            core::CUDAState::GetInstance();
    utility::LogInfo("Number of CUDA devices: {}", cuda_state->GetNumDevices());
    for (int i = 0; i < cuda_state->GetNumDevices(); ++i) {
        for (int j = 0; j < cuda_state->GetNumDevices(); ++j) {
            utility::LogInfo("P2PEnabled {}->{}: {}", i, j,
                             cuda_state->GetP2PEnabled()[i][j]);
        }
    }
}

}  // namespace tests
}  // namespace cloudViewer

#endif
