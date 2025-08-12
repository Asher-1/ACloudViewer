// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/kernel/IndexGetSet.h"

#include "cloudViewer/core/Dtype.h"
#include "cloudViewer/core/MemoryManager.h"
#include "cloudViewer/core/SizeVector.h"
#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/core/kernel/UnaryEW.h"
#include <Logging.h>

namespace cloudViewer {
namespace core {
namespace kernel {

void IndexGet(const Tensor& src,
              Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const SizeVector& indexed_shape,
              const SizeVector& indexed_strides) {
    // index_tensors has been preprocessed to be on the same device as src,
    // however, dst may be in a different device.
    if (dst.GetDevice() != src.GetDevice()) {
        Tensor dst_same_device(dst.GetShape(), dst.GetDtype(), src.GetDevice());
        IndexGet(src, dst_same_device, index_tensors, indexed_shape,
                 indexed_strides);
        dst.CopyFrom(dst_same_device);
        return;
    }

    if (src.IsCPU()) {
        IndexGetCPU(src, dst, index_tensors, indexed_shape, indexed_strides);
    } else if (src.IsSYCL()) {
#ifdef BUILD_SYCL_MODULE
        IndexGetSYCL(src, dst, index_tensors, indexed_shape, indexed_strides);
#endif
    } else if (src.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        IndexGetCUDA(src, dst, index_tensors, indexed_shape, indexed_strides);
#endif
    } else {
        utility::LogError("IndexGet: Unimplemented device");
    }
}

void IndexSet(const Tensor& src,
              Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const SizeVector& indexed_shape,
              const SizeVector& indexed_strides) {
    // index_tensors has been preprocessed to be on the same device as dst,
    // however, src may be on a different device.
    Tensor src_same_device = src.To(dst.GetDevice());

    if (dst.IsCPU()) {
        IndexSetCPU(src_same_device, dst, index_tensors, indexed_shape,
                    indexed_strides);
    } else if (dst.IsSYCL()) {
#ifdef BUILD_SYCL_MODULE
        IndexSetSYCL(src_same_device, dst, index_tensors, indexed_shape,
                     indexed_strides);
#endif
    } else if (dst.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        IndexSetCUDA(src_same_device, dst, index_tensors, indexed_shape,
                     indexed_strides);
#endif
    } else {
        utility::LogError("IndexSet: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
