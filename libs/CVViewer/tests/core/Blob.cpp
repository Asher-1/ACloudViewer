// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/Blob.h"

#include "cloudViewer/core/Device.h"
#include "cloudViewer/core/MemoryManager.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace cloudViewer {
namespace tests {

class BlobPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Blob,
                         BlobPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(BlobPermuteDevices, BlobConstructor) {
    core::Device device = GetParam();

    core::Blob b(10, core::Device(device));
}

TEST_P(BlobPermuteDevices, BlobConstructorWithExternalMemory) {
    core::Device device = GetParam();

    void* data_ptr = core::MemoryManager::Malloc(8, device);
    bool deleter_called = false;

    auto deleter = [&device, &deleter_called, data_ptr](void* dummy) -> void {
        core::MemoryManager::Free(data_ptr, device);
        deleter_called = true;
    };

    {
        core::Blob b(device, data_ptr, deleter);
        EXPECT_EQ(b.GetDataPtr(), data_ptr);
        EXPECT_FALSE(deleter_called);
    }
    EXPECT_TRUE(deleter_called);
}

}  // namespace tests
}  // namespace cloudViewer
