// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 Asher-1.github.io
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

#include "core/CUDAUtils.h"

#include "Macro.h"

#include <Logging.h>

#ifdef BUILD_CUDA_MODULE
#include "core/MemoryManager.h"
#endif

namespace cloudViewer {
namespace core {
namespace cuda {


int DeviceCount() {
#ifdef BUILD_CUDA_MODULE
    try {
        int num_devices;
        CLOUDVIEWER_CUDA_CHECK(cudaGetDeviceCount(&num_devices));
        return num_devices;
    }
    // This function is also used to detect CUDA support in our Python code.
    // Thus, catch any errors if no GPU is available.
    catch (const std::runtime_error&) {
        return 0;
    }
#else
    return 0;
#endif
}

bool IsAvailable() { return cuda::DeviceCount() > 0; }

void ReleaseCache() {
#ifdef BUILD_CUDA_MODULE
#ifdef ENABLE_CACHED_CUDA_MANAGER
    // Release cache from all devices. Since only memory from MemoryManagerCUDA
    // is cached at the moment, this works as expected. In the future, the logic
    // could become more fine-grained.
    MemoryManagerCached::ReleaseCache();
#else
    utility::LogWarning(
            "Built without cached CUDA memory manager, cuda::ReleaseCache() "
            "has no effect.");
#endif

#else
    utility::LogWarning("Built without CUDA module, cuda::ReleaseCache().");
#endif
}

void Synchronize() {
#ifdef BUILD_CUDA_MODULE
    for (int i = 0; i < DeviceCount(); ++i) {
        Synchronize(Device(Device::DeviceType::CUDA, i));
    }
#endif
}

void Synchronize(const Device& device) {
#ifdef BUILD_CUDA_MODULE
    if (device.IsCUDA()) {
        CUDAScopedDevice scoped_device(device);
        CLOUDVIEWER_CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
}

void AssertCUDADeviceAvailable(int device_id) {
#ifdef BUILD_CUDA_MODULE
    int num_devices = cuda::DeviceCount();
    if (num_devices == 0) {
        utility::LogError(
                "Invalid device 'CUDA:{}'. -DBUILD_CUDA_MODULE=ON, but no "
                "CUDA device available.",
                device_id);
    } else if (num_devices == 1 && device_id != 0) {
        utility::LogError(
                "Invalid CUDA Device 'CUDA:{}'. Device ID expected to "
                "be 0, but got {}.",
                device_id, device_id);
    } else if (device_id < 0 || device_id >= num_devices) {
        utility::LogError(
                "Invalid CUDA Device 'CUDA:{}'. Device ID expected to "
                "be between 0 to {}, but got {}.",
                device_id, num_devices - 1, device_id);
    }
#else
    utility::LogError(
            "-DBUILD_CUDA_MODULE=OFF. Please build with -DBUILD_CUDA_MODULE=ON "
            "to use CUDA device.");
#endif
}

void AssertCUDADeviceAvailable(const Device& device) {
    if (device.IsCUDA()) {
        AssertCUDADeviceAvailable(device.GetID());
    } else {
        utility::LogError(
                "Expected device-type to be CUDA, but got device '{}'",
                device.ToString());
    }
}

bool SupportsMemoryPools(const Device& device) {
#if defined(BUILD_CUDA_MODULE) && (CUDART_VERSION >= 11020)
    if (device.IsCUDA()) {
        int driverVersion = 0;
        int deviceSupportsMemoryPools = 0;
        CLOUDVIEWER_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
        if (driverVersion >=
            11020) {  // avoid invalid value error in cudaDeviceGetAttribute
            CLOUDVIEWER_CUDA_CHECK(cudaDeviceGetAttribute(
                    &deviceSupportsMemoryPools, cudaDevAttrMemoryPoolsSupported,
                    device.GetID()));
        }
        return !!deviceSupportsMemoryPools;
    } else {
        return false;
    }
#else
    return false;
#endif
}

#ifdef BUILD_CUDA_MODULE
int GetDevice() {
    int device;
    CLOUDVIEWER_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

static void SetDevice(int device_id) {
    AssertCUDADeviceAvailable(device_id);
    CLOUDVIEWER_CUDA_CHECK(cudaSetDevice(device_id));
}

class CUDAStream {
public:
    static CUDAStream& GetInstance() {
        // The global stream state is given per thread like CUDA's internal
        // device state.
        static thread_local CUDAStream instance;
        return instance;
    }

    cudaStream_t Get() { return stream_; }
    void Set(cudaStream_t stream) { stream_ = stream; }

    static cudaStream_t Default() { return static_cast<cudaStream_t>(0); }

private:
    CUDAStream() = default;
    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;

    cudaStream_t stream_ = Default();
};

cudaStream_t GetStream() { return CUDAStream::GetInstance().Get(); }

static void SetStream(cudaStream_t stream) {
    CUDAStream::GetInstance().Set(stream);
}

cudaStream_t GetDefaultStream() { return CUDAStream::Default(); }

#endif

}  // namespace cuda

#ifdef BUILD_CUDA_MODULE

CUDAScopedDevice::CUDAScopedDevice(int device_id)
    : prev_device_id_(cuda::GetDevice()) {
    cuda::SetDevice(device_id);
}

CUDAScopedDevice::CUDAScopedDevice(const Device& device)
    : CUDAScopedDevice(device.GetID()) {
    cuda::AssertCUDADeviceAvailable(device);
}

CUDAScopedDevice::~CUDAScopedDevice() { cuda::SetDevice(prev_device_id_); }

CUDAScopedStream::CreateNewStreamTag
        CUDAScopedStream::CreateNewStream = {};

CUDAScopedStream::CUDAScopedStream(const CreateNewStreamTag&)
    : prev_stream_(cuda::GetStream()), owns_new_stream_(true) {
    CLOUDVIEWER_CUDA_CHECK(cudaStreamCreate(&new_stream_));
    cuda::SetStream(new_stream_);
}

CUDAScopedStream::CUDAScopedStream(cudaStream_t stream)
    : prev_stream_(cuda::GetStream()),
      new_stream_(stream),
      owns_new_stream_(false) {
    cuda::SetStream(stream);
}

CUDAScopedStream::~CUDAScopedStream() {
    if (owns_new_stream_) {
        CLOUDVIEWER_CUDA_CHECK(cudaStreamDestroy(new_stream_));
    }
    cuda::SetStream(prev_stream_);
}

CUDAState& CUDAState::GetInstance() {
    static CUDAState instance;
    return instance;
}

bool CUDAState::IsP2PEnabled(int src_id, int tar_id) const {
    cuda::AssertCUDADeviceAvailable(src_id);
    cuda::AssertCUDADeviceAvailable(tar_id);
    return p2p_enabled_[src_id][tar_id];
}

bool CUDAState::IsP2PEnabled(const Device& src, const Device& tar) const {
    cuda::AssertCUDADeviceAvailable(src);
    cuda::AssertCUDADeviceAvailable(tar);
    return p2p_enabled_[src.GetID()][tar.GetID()];
}

void CUDAState::ForceDisableP2PForTesting() {
    for (int src_id = 0; src_id < cuda::DeviceCount(); ++src_id) {
        for (int tar_id = 0; tar_id < cuda::DeviceCount(); ++tar_id) {
            if (src_id != tar_id && p2p_enabled_[src_id][tar_id]) {
                p2p_enabled_[src_id][tar_id] = false;
            }
        }
    }
}

CUDAState::CUDAState() {
    // Check and enable all possible peer to peer access.
    p2p_enabled_ = std::vector<std::vector<bool>>(
            cuda::DeviceCount(), std::vector<bool>(cuda::DeviceCount(), false));

    for (int src_id = 0; src_id < cuda::DeviceCount(); ++src_id) {
        for (int tar_id = 0; tar_id < cuda::DeviceCount(); ++tar_id) {
            if (src_id == tar_id) {
                p2p_enabled_[src_id][tar_id] = true;
            } else {
                CUDAScopedDevice scoped_device(src_id);

                // Check access.
                int can_access = 0;
                CLOUDVIEWER_CUDA_CHECK(
                        cudaDeviceCanAccessPeer(&can_access, src_id, tar_id));
                // Enable access.
                if (can_access) {
                    p2p_enabled_[src_id][tar_id] = true;
                    cudaError_t err = cudaDeviceEnablePeerAccess(tar_id, 0);
                    if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        // Ignore error since P2P is already enabled.
                        cudaGetLastError();
                    } else {
                        CLOUDVIEWER_CUDA_CHECK(err);
                    }
                } else {
                    p2p_enabled_[src_id][tar_id] = false;
                }
            }
        }
    }
}

int GetCUDACurrentDeviceTextureAlignment() {
    int value;
    CLOUDVIEWER_CUDA_CHECK(cudaDeviceGetAttribute(
            &value, cudaDevAttrTextureAlignment, cuda::GetDevice()));
    return value;
}

int GetCUDACurrentWarpSize() {
    int value;
    CLOUDVIEWER_CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize,
                                             cuda::GetDevice()));
    return value;
}

size_t GetCUDACurrentTotalMemSize() {
    size_t free;
    size_t total;
    CLOUDVIEWER_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total;
}

#endif

}  // namespace core
}  // namespace cloudViewer

#ifdef BUILD_CUDA_MODULE
namespace cloudViewer {
namespace core {

void __CLOUDVIEWER_CUDA_CHECK(cudaError_t err,
                              const char* file,
                              const int line) {
    if (err != cudaSuccess) {
        utility::LogError("{}:{} CUDA runtime error: {}", file, line,
                          cudaGetErrorString(err));
    }
}

void __CLOUDVIEWER_GET_LAST_CUDA_ERROR(const char* message,
                                       const char* file,
                                       const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        utility::LogError("{}:{} {}: CLOUDVIEWER_GET_LAST_CUDA_ERROR(): {}", file,
                          line, message, cudaGetErrorString(err));
    }
}

}  // namespace core
}  // namespace cloudViewer

#endif

// C interface to provide un-mangled function to Python ctypes
extern "C" CLOUDVIEWER_DLL_EXPORT int cloudViewer_core_cuda_device_count() {
    return cloudViewer::core::cuda::DeviceCount();
}
