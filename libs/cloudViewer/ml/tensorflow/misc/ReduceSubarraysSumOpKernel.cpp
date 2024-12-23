// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ReduceSubarraysSumOpKernel.h"

#include "ml/impl/misc/ReduceSubarraysSum.h"

using namespace cloudViewer::ml::impl;
using namespace reduce_subarrays_sum_opkernel;
using namespace tensorflow;

template <class T>
class ReduceSubarraysSumOpKernelCPU : public ReduceSubarraysSumOpKernel {
public:
    explicit ReduceSubarraysSumOpKernelCPU(OpKernelConstruction* construction)
        : ReduceSubarraysSumOpKernel(construction) {}

    void Kernel(OpKernelContext* context,
                const tensorflow::Tensor& values,
                const tensorflow::Tensor& row_splits,
                tensorflow::Tensor& sums) {
        ReduceSubarraysSumCPU(
                values.flat<T>().data(), values.shape().dim_size(0),
                (int64_t*)row_splits.flat<int64>().data(),
                row_splits.shape().dim_size(0) - 1, sums.flat<T>().data());
    }
};

#define REG_KB(type)                                            \
    REGISTER_KERNEL_BUILDER(Name("CloudViewerReduceSubarraysSum")    \
                                    .Device(DEVICE_CPU)         \
                                    .TypeConstraint<type>("T"), \
                            ReduceSubarraysSumOpKernelCPU<type>);
REG_KB(int32_t)
REG_KB(int64)
REG_KB(float)
REG_KB(double)
#undef REG_KB
