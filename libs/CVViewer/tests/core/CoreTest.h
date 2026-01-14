// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "cloudViewer/core/Device.h"
#include "cloudViewer/core/Dtype.h"
#include "cloudViewer/core/SizeVector.h"
#include "tests/Tests.h"

#ifdef BUILD_CUDA_MODULE
#include "cloudViewer/core/CUDAUtils.h"
#endif

namespace cloudViewer {
namespace tests {

class PermuteDtypesWithBool : public testing::TestWithParam<core::Dtype> {
public:
    static std::vector<core::Dtype> TestCases();
};

class PermuteDevices : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases();
};

/// Permute one device for each device type, in {CPU, CUDA, SYCL}.
class PermuteDevicesWithSYCL : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases();
};

class PermuteDevicePairs
    : public testing::TestWithParam<std::pair<core::Device, core::Device>> {
public:
    static std::vector<std::pair<core::Device, core::Device>> TestCases();
};

/// Permute device pairs, in {CPU, CUDA, SYCL}.
class PermuteDevicePairsWithSYCL
    : public testing::TestWithParam<std::pair<core::Device, core::Device>> {
public:
    static std::vector<std::pair<core::Device, core::Device>> TestCases();
};

class PermuteSizesDefaultStrides
    : public testing::TestWithParam<
              std::pair<core::SizeVector, core::SizeVector>> {
public:
    static std::vector<std::pair<core::SizeVector, core::SizeVector>>
    TestCases();
};

class TensorSizes : public testing::TestWithParam<int64_t> {
public:
    static std::vector<int64_t> TestCases();
};

}  // namespace tests
}  // namespace cloudViewer
