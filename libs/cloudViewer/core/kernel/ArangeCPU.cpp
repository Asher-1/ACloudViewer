// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/Dispatch.h"
#include "cloudViewer/core/ParallelFor.h"
#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/core/kernel/Arange.h"

namespace cloudViewer {
namespace core {
namespace kernel {

void ArangeCPU(const Tensor& start,
               const Tensor& stop,
               const Tensor& step,
               Tensor& dst) {
    Dtype dtype = start.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sstart = start.Item<scalar_t>();
        scalar_t sstep = step.Item<scalar_t>();
        scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
        int64_t n = dst.GetLength();
        ParallelFor(start.GetDevice(), n, [&](int64_t workload_idx) {
            dst_ptr[workload_idx] =
                    sstart + static_cast<scalar_t>(sstep * workload_idx);
        });
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
