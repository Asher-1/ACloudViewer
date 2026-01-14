// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/MemoryManager.h"

#include <vector>

#include "core/Blob.h"
#include "core/Device.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace cloudViewer {
namespace tests {

class MemoryManagerPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(MemoryManager,
                         MemoryManagerPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class MemoryManagerPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        MemoryManager,
        MemoryManagerPermuteDevicePairs,
        testing::ValuesIn(MemoryManagerPermuteDevicePairs::TestCases()));

TEST_P(MemoryManagerPermuteDevices, MallocFree) {
    core::Device device = GetParam();

    void* ptr = core::MemoryManager::Malloc(10, device);
    core::MemoryManager::Free(ptr, device);
}

TEST_P(MemoryManagerPermuteDevicePairs, Memcpy) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    char dst_vals[6] = "xxxxx";
    char src_vals[6] = "hello";
    size_t num_bytes = strlen(src_vals) + 1;

    void* dst_ptr = core::MemoryManager::Malloc(num_bytes, dst_device);
    void* src_ptr = core::MemoryManager::Malloc(num_bytes, src_device);
    core::MemoryManager::MemcpyFromHost(src_ptr, src_device, (void*)src_vals,
                                        num_bytes);

    core::MemoryManager::Memcpy(dst_ptr, dst_device, src_ptr, src_device,
                                num_bytes);
    core::MemoryManager::MemcpyToHost((void*)dst_vals, dst_ptr, dst_device,
                                      num_bytes);
    ASSERT_STREQ(dst_vals, src_vals);

    core::MemoryManager::Free(dst_ptr, dst_device);
    core::MemoryManager::Free(src_ptr, src_device);
}

}  // namespace tests
}  // namespace cloudViewer
