// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Logging.h>

#include "cloudViewer/core/Tensor.h"

namespace cloudViewer {
namespace core {
namespace kernel {

void IndexAdd_(int64_t dim,
               const Tensor& index,
               const Tensor& src,
               Tensor& dst);

void IndexAddCPU_(int64_t dim,
                  const Tensor& index,
                  const Tensor& src,
                  Tensor& dst);

#ifdef BUILD_SYCL_MODULE
void IndexAddSYCL_(int64_t dim,
                   const Tensor& index,
                   const Tensor& src,
                   Tensor& dst);
#endif

#ifdef BUILD_CUDA_MODULE
void IndexAddCUDA_(int64_t dim,
                   const Tensor& index,
                   const Tensor& src,
                   Tensor& dst);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
