// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                        -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "core/AdvancedIndexing.h"
#include "core/CUDAUtils.h"
#include "core/Dispatch.h"
#include "core/Indexer.h"
#include "core/ParallelFor.h"
#include "core/Tensor.h"
#include "core/kernel/IndexGetSet.h"

namespace cloudViewer {
namespace core {
namespace kernel {

template <typename func_t>
void LaunchAdvancedIndexerKernel(const Device& device,
                                 const AdvancedIndexer& indexer,
                                 const func_t& element_kernel) {
    CLOUDVIEWER_ASSERT_HOST_DEVICE_LAMBDA(func_t);
    auto element_func = [=] CLOUDVIEWER_HOST_DEVICE(int64_t i) {
        element_kernel(indexer.GetInputPtr(i), indexer.GetOutputPtr(i));
    };
    ParallelFor(device, indexer.NumWorkloads(), element_func);
    CLOUDVIEWER_GET_LAST_CUDA_ERROR("LaunchAdvancedIndexerKernel failed.");
}

template <typename scalar_t>
static CLOUDVIEWER_HOST_DEVICE void CUDACopyElementKernel(const void* src,
                                                     void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(src);
}

static CLOUDVIEWER_HOST_DEVICE void CUDACopyObjectElementKernel(
        const void* src, void* dst, int64_t object_byte_size) {
    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);
    for (int i = 0; i < object_byte_size; ++i) {
        dst_bytes[i] = src_bytes[i];
    }
}

void IndexGetCUDA(const Tensor& src,
                  Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const SizeVector& indexed_shape,
                  const SizeVector& indexed_strides) {
    Dtype dtype = src.GetDtype();
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::GET);

    if (dtype.IsObject()) {
        int64_t object_byte_size = dtype.ByteSize();
        LaunchAdvancedIndexerKernel(
                src.GetDevice(), ai,
                [=] CLOUDVIEWER_HOST_DEVICE(const void* src, void* dst) {
                    CUDACopyObjectElementKernel(src, dst, object_byte_size);
                });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
            LaunchAdvancedIndexerKernel(
                    src.GetDevice(), ai,
                    // Need to wrap as extended CUDA lambda function
                    [] CLOUDVIEWER_HOST_DEVICE(const void* src, void* dst) {
                        CUDACopyElementKernel<scalar_t>(src, dst);
                    });
        });
    }
}

void IndexSetCUDA(const Tensor& src,
                  Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const SizeVector& indexed_shape,
                  const SizeVector& indexed_strides) {
    Dtype dtype = src.GetDtype();
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::SET);

    if (dtype.IsObject()) {
        int64_t object_byte_size = dtype.ByteSize();
        LaunchAdvancedIndexerKernel(
                src.GetDevice(), ai,
                [=] CLOUDVIEWER_HOST_DEVICE(const void* src, void* dst) {
                    CUDACopyObjectElementKernel(src, dst, object_byte_size);
                });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
            LaunchAdvancedIndexerKernel(
                    src.GetDevice(), ai,
                    // Need to wrap as extended CUDA lambda function
                    [] CLOUDVIEWER_HOST_DEVICE(const void* src, void* dst) {
                        CUDACopyElementKernel<scalar_t>(src, dst);
                    });
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
