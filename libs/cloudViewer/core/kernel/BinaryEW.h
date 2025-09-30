// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <unordered_set>

#include "cloudViewer/core/Tensor.h"
#include <Helper.h>
#include <Logging.h>

namespace cloudViewer {
namespace core {
namespace kernel {

enum class BinaryEWOpCode {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
};

extern const std::unordered_set<BinaryEWOpCode, utility::hash_enum_class>
        s_boolean_binary_ew_op_codes;

void BinaryEW(const Tensor& lhs,
              const Tensor& rhs,
              Tensor& dst,
              BinaryEWOpCode op_code);

void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 Tensor& dst,
                 BinaryEWOpCode op_code);

#ifdef BUILD_SYCL_MODULE
void BinaryEWSYCL(const Tensor& lhs,
                  const Tensor& rhs,
                  Tensor& dst,
                  BinaryEWOpCode op_code);
#endif

#ifdef BUILD_CUDA_MODULE
void BinaryEWCUDA(const Tensor& lhs,
                  const Tensor& rhs,
                  Tensor& dst,
                  BinaryEWOpCode op_code);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
