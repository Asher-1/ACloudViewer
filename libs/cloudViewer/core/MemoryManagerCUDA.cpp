// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cuda.h>
#include <cuda_runtime.h>

#include "cloudViewer/core/CUDAUtils.h"
#include "cloudViewer/core/MemoryManager.h"

namespace cloudViewer {
namespace core {

void* MemoryManagerCUDA::Malloc(size_t byte_size, const Device& device) {
    CUDAScopedDevice scoped_device(device);

    void* ptr;
    if (device.IsCUDA()) {
#if CUDART_VERSION >= 11020
        if (cuda::SupportsMemoryPools(device)) {
            CLOUDVIEWER_CUDA_CHECK(cudaMallocAsync(static_cast<void**>(&ptr),
                                              byte_size, cuda::GetStream()));
        } else {
            CLOUDVIEWER_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));
        }
#else
        CLOUDVIEWER_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));
#endif
    } else {
        utility::LogError("Internal error: Unimplemented device {}.",
                          device.ToString());
    }
    return ptr;
}

void MemoryManagerCUDA::Free(void* ptr, const Device& device) {
    CUDAScopedDevice scoped_device(device);

    if (device.IsCUDA()) {
        if (ptr && IsCUDAPointer(ptr, device)) {
#if CUDART_VERSION >= 11020
            if (cuda::SupportsMemoryPools(device)) {
                CLOUDVIEWER_CUDA_CHECK(cudaFreeAsync(ptr, cuda::GetStream()));
            } else {
                CLOUDVIEWER_CUDA_CHECK(cudaFree(ptr));
            }
#else
            CLOUDVIEWER_CUDA_CHECK(cudaFree(ptr));
#endif
        }
    } else {
        utility::LogError("Internal error: Unimplemented device {}.",
                          device.ToString());
    }
}

void MemoryManagerCUDA::Memcpy(void* dst_ptr,
                               const Device& dst_device,
                               const void* src_ptr,
                               const Device& src_device,
                               size_t num_bytes) {
    if (dst_device.IsCUDA() && src_device.IsCPU()) {
        if (!IsCUDAPointer(dst_ptr, dst_device)) {
            utility::LogError("dst_ptr is not a CUDA pointer.");
        }
        CUDAScopedDevice scoped_device(dst_device);
        CLOUDVIEWER_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, num_bytes,
                                          cudaMemcpyHostToDevice,
                                          cuda::GetStream()));
    } else if (dst_device.IsCPU() && src_device.IsCUDA()) {
        if (!IsCUDAPointer(src_ptr, src_device)) {
            utility::LogError("src_ptr is not a CUDA pointer.");
        }
        CUDAScopedDevice scoped_device(src_device);
        CLOUDVIEWER_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, num_bytes,
                                          cudaMemcpyDeviceToHost,
                                          cuda::GetStream()));
    } else if (dst_device.IsCUDA() && src_device.IsCUDA()) {
        if (!IsCUDAPointer(dst_ptr, dst_device)) {
            utility::LogError("dst_ptr is not a CUDA pointer.");
        }
        if (!IsCUDAPointer(src_ptr, src_device)) {
            utility::LogError("src_ptr is not a CUDA pointer.");
        }

        if (dst_device == src_device) {
            CUDAScopedDevice scoped_device(src_device);
            CLOUDVIEWER_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, num_bytes,
                                              cudaMemcpyDeviceToDevice,
                                              cuda::GetStream()));
        } else if (CUDAState::GetInstance().IsP2PEnabled(src_device.GetID(),
                                                         dst_device.GetID())) {
            CLOUDVIEWER_CUDA_CHECK(cudaMemcpyPeerAsync(
                    dst_ptr, dst_device.GetID(), src_ptr, src_device.GetID(),
                    num_bytes, cuda::GetStream()));
        } else {
            void* cpu_buf = MemoryManager::Malloc(num_bytes, Device("CPU:0"));
            {
                CUDAScopedDevice scoped_device(src_device);
                CLOUDVIEWER_CUDA_CHECK(cudaMemcpyAsync(cpu_buf, src_ptr, num_bytes,
                                                  cudaMemcpyDeviceToHost,
                                                  cuda::GetStream()));
            }
            {
                CUDAScopedDevice scoped_device(dst_device);
                CLOUDVIEWER_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, cpu_buf, num_bytes,
                                                  cudaMemcpyHostToDevice,
                                                  cuda::GetStream()));
            }
            MemoryManager::Free(cpu_buf, Device("CPU:0"));
        }
    } else {
        utility::LogError("Wrong cudaMemcpyKind.");
    }
}

bool MemoryManagerCUDA::IsCUDAPointer(const void* ptr, const Device& device) {
    CUDAScopedDevice scoped_device(device);

    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    return attributes.devicePointer != nullptr ? true : false;
}

}  // namespace core
}  // namespace cloudViewer
