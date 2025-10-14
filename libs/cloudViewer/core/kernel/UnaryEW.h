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

enum class UnaryEWOpCode {
    Sqrt,
    Sin,
    Cos,
    Neg,
    Exp,
    Abs,
    IsNan,
    IsInf,
    IsFinite,
    Floor,
    Ceil,
    Round,
    Trunc,
    LogicalNot
};

void UnaryEW(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code);
void UnaryEWCPU(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code);

#ifdef BUILD_SYCL_MODULE
void UnaryEWSYCL(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code);
#endif

#ifdef BUILD_CUDA_MODULE
void UnaryEWCUDA(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code);
#endif

// Copy is separated from other unary ops since it supports cross-device copy
// and dtype casting.
void Copy(const Tensor& src, Tensor& dst);

void CopyCPU(const Tensor& src, Tensor& dst);

#ifdef BUILD_CUDA_MODULE
void CopyCUDA(const Tensor& src, Tensor& dst);
#endif

#ifdef BUILD_SYCL_MODULE
void CopySYCL(const Tensor& src, Tensor& dst);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
