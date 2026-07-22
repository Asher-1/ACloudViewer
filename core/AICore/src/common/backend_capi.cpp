// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/backend_capi.h"

#include "ggml_common/ggml_backend_utils.hpp"

#include <cstring>

#if (defined(GGML_USE_CUDA) || defined(GGML_CUDA)) && !defined(GGML_BACKEND_DL)
#include <cuda_runtime.h>
#endif

namespace {

#if defined(__APPLE__)
static const aicore_device_info kDevices[] = {
    {"auto",  "Auto (Metal \xe2\x86\x92 CUDA \xe2\x86\x92 CPU)", 0},
    {"metal", "GPU (Metal)",                                      1},
    {"cuda",  "GPU (CUDA)",                                       0},
    {"cpu",   "CPU",                                              0},
};
static const char* kAutoOrder = "Metal \xe2\x86\x92 CUDA \xe2\x86\x92 CPU";
#else
static const aicore_device_info kDevices[] = {
    {"auto",   "Auto (CUDA \xe2\x86\x92 OpenCL \xe2\x86\x92 CPU)", 0},
    {"cuda",   "GPU (CUDA)",                                        0},
    {"opencl", "GPU (OpenCL)",                                      0},
    {"cpu",    "CPU",                                                0},
};
static const char* kAutoOrder = "CUDA \xe2\x86\x92 OpenCL \xe2\x86\x92 CPU";
#endif

static constexpr int kDeviceCount =
        static_cast<int>(sizeof(kDevices) / sizeof(kDevices[0]));

}  // namespace

int AICORE_CAPI aicore_device_count(void) { return kDeviceCount; }

const aicore_device_info* AICORE_CAPI aicore_device_at(int index) {
    if (index < 0 || index >= kDeviceCount) return nullptr;
    return &kDevices[index];
}

const char* AICORE_CAPI aicore_auto_device_order(void) { return kAutoOrder; }

int AICORE_CAPI aicore_warmup_backend(const char* device) {
    (void)device;
    ggml_common::load_backends_once();
#if (defined(GGML_USE_CUDA) || defined(GGML_CUDA)) && !defined(GGML_BACKEND_DL)
    cudaGetLastError();
#endif
    return 0;
}

int AICORE_CAPI aicore_is_gpu_device(const char* device) {
    if (!device || device[0] == '\0') return 0;
    if (std::strcmp(device, "cpu") == 0) return 0;
    return 1;
}
