// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/Dispatch.h"
#include "core/ParallelFor.h"
#include "core/Tensor.h"
#include "core/kernel/Arange.h"

namespace cloudViewer {
namespace core {
namespace kernel {

void ArangeCUDA(const Tensor& start,
                const Tensor& stop,
                const Tensor& step,
                Tensor& dst) {
    Dtype dtype = start.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sstart = start.Item<scalar_t>();
        scalar_t sstep = step.Item<scalar_t>();
        scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
        int64_t n = dst.GetLength();
        ParallelFor(start.GetDevice(), n,
                    [=] CLOUDVIEWER_HOST_DEVICE(int64_t workload_idx) {
                        dst_ptr[workload_idx] =
                                sstart +
                                static_cast<scalar_t>(sstep * workload_idx);
                    });
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
