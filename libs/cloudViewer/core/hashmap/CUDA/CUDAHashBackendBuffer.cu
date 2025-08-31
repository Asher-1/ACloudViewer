// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "cloudViewer/core/CUDAUtils.h"
#include "cloudViewer/core/hashmap/HashBackendBuffer.h"

namespace cloudViewer {
namespace core {
void CUDAResetHeap(Tensor &heap) {
    uint32_t *heap_ptr = heap.GetDataPtr<uint32_t>();
    thrust::sequence(thrust::device, heap_ptr, heap_ptr + heap.GetLength(), 0);
    CLOUDVIEWER_CUDA_CHECK(cudaGetLastError());
}
}  // namespace core
}  // namespace cloudViewer
