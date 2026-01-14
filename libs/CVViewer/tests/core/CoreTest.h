// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "core/Device.h"
#include "core/SizeVector.h"
#include "tests/Tests.h"

#ifdef BUILD_CUDA_MODULE
#include "core/CUDAState.cuh"
#endif

namespace cloudViewer {
namespace tests {

class PermuteDevices : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases() {
#ifdef BUILD_CUDA_MODULE
        std::shared_ptr<core::CUDAState> cuda_state =
                core::CUDAState::GetInstance();
        if (cuda_state->GetNumDevices() >= 1) {
            return {
                    core::Device("CPU:0"),
                    core::Device("CUDA:0"),
            };
        } else {
            return {
                    core::Device("CPU:0"),
            };
        }
#else
        return {
                core::Device("CPU:0"),
        };
#endif
    }
};

class PermuteDevicePairs
    : public testing::TestWithParam<std::pair<core::Device, core::Device>> {
public:
    static std::vector<std::pair<core::Device, core::Device>> TestCases() {
#ifdef BUILD_CUDA_MODULE
        std::shared_ptr<core::CUDAState> cuda_state =
                core::CUDAState::GetInstance();
        if (cuda_state->GetNumDevices() > 1) {
            // To test multiple CUDA devices, we only need to test CUDA 0 and 1.
            return {
                    {core::Device("CPU:0"), core::Device("CPU:0")},    // 0
                    {core::Device("CPU:0"), core::Device("CUDA:0")},   // 1
                    {core::Device("CPU:0"), core::Device("CUDA:1")},   // 2
                    {core::Device("CUDA:0"), core::Device("CPU:0")},   // 3
                    {core::Device("CUDA:0"), core::Device("CUDA:0")},  // 4
                    {core::Device("CUDA:0"), core::Device("CUDA:1")},  // 5
                    {core::Device("CUDA:1"), core::Device("CPU:0")},   // 6
                    {core::Device("CUDA:1"), core::Device("CUDA:0")},  // 7
                    {core::Device("CUDA:1"), core::Device("CUDA:1")},  // 8
            };
        } else if (cuda_state->GetNumDevices() == 1) {
            return {
                    {core::Device("CPU:0"), core::Device("CPU:0")},
                    {core::Device("CPU:0"), core::Device("CUDA:0")},
                    {core::Device("CUDA:0"), core::Device("CPU:0")},
                    {core::Device("CUDA:0"), core::Device("CUDA:0")},
            };
        } else {
            return {
                    {core::Device("CPU:0"), core::Device("CPU:0")},
            };
        }
#else
        return {
                {core::Device("CPU:0"), core::Device("CPU:0")},
        };
#endif
    }
};

class PermuteSizesDefaultStrides
    : public testing::TestWithParam<
              std::pair<core::SizeVector, core::SizeVector>> {
public:
    static std::vector<std::pair<core::SizeVector, core::SizeVector>>
    TestCases() {
        return {
                {{}, {}},
                {{0}, {1}},
                {{0, 0}, {1, 1}},
                {{0, 1}, {1, 1}},
                {{1, 0}, {1, 1}},
                {{1}, {1}},
                {{1, 2}, {2, 1}},
                {{1, 2, 3}, {6, 3, 1}},
                {{4, 3, 2}, {6, 2, 1}},
                {{2, 0, 3}, {3, 3, 1}},
        };
    }
};

class TensorSizes : public testing::TestWithParam<int64_t> {
public:
    static std::vector<int64_t> TestCases() {
        std::vector<int64_t> tensor_sizes{
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
        // clang-format off
        std::vector<int64_t> large_sizes{
                (1 << 6 ) - 1, (1 << 6 ), (1 << 6 ) + 1,
                (1 << 10) - 6, (1 << 10), (1 << 10) + 6,
                (1 << 15) - 7, (1 << 15), (1 << 15) + 7,
                (1 << 20) - 1, (1 << 20), (1 << 20) + 1,
                (1 << 25) - 2, (1 << 25), (1 << 25) + 2, // ~128MB for float32
        };
        // clang-format on
        tensor_sizes.insert(tensor_sizes.end(), large_sizes.begin(),
                            large_sizes.end());
        return tensor_sizes;
    }
};

}  // namespace tests
}  // namespace cloudViewer
