// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/SYCLUtils.h"

#include <vector>

#include "cloudViewer/core/MemoryManager.h"
#include "cloudViewer/utility/Helper.h"
#include "cloudViewer/utility/Timer.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace cloudViewer {
namespace tests {

TEST(SYCLUtils, SYCLDemo) { core::sy::SYCLDemo(); }

TEST(SYCLUtils, PrintAllSYCLDevices) {
    core::sy::PrintSYCLDevices(/*print_all=*/true);
}

TEST(SYCLUtils, PrintSYCLDevices) {
    core::sy::PrintSYCLDevices(/*print_all=*/false);
}

TEST(SYCLUtils, SYCLUnifiedSharedMemory) {
    if (!core::sy::IsAvailable()) {
        return;
    }

    size_t byte_size = sizeof(int) * 4;
    int* host_ptr = static_cast<int*>(malloc(byte_size));
    for (int i = 0; i < 4; i++) {
        host_ptr[i] = i;
    }
    core::Device host_device;

#ifdef ENABLE_SYCL_UNIFIED_SHARED_MEMORY
    utility::LogInfo("SYCLMemoryModel: unified shared memory");
    // Can host access SYCL GPU's memory directly? Yes.
    core::Device sycl_device("SYCL:0");
    int* sycl_ptr = static_cast<int*>(
            core::MemoryManager::Malloc(byte_size, sycl_device));
    core::MemoryManager::Memcpy(sycl_ptr, sycl_device, host_ptr, host_device,
                                byte_size);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(sycl_ptr[i], i);
    }
#else
    utility::LogInfo("SYCLMemoryModel: device memory");
    // Can host access SYCL GPU's memory directly? No.
    core::Device sycl_device("SYCL:0");
    int* sycl_ptr = static_cast<int*>(
            core::MemoryManager::Malloc(byte_size, sycl_device));
    core::MemoryManager::Memcpy(sycl_ptr, sycl_device, host_ptr, host_device,
                                byte_size);
    for (int i = 0; i < 4; i++) {
        // EXPECT_EQ(sycl_ptr[i], i); // This will segfault.
    }
#endif

    free(host_ptr);
    core::MemoryManager::Free(sycl_ptr, sycl_device);
}

}  // namespace tests
}  // namespace cloudViewer
