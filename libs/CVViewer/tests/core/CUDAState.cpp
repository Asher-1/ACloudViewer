// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
