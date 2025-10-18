// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Helper.h>
#include <Logging.h>

#include <unordered_set>

#include "cloudViewer/core/SizeVector.h"
#include "cloudViewer/core/Tensor.h"

namespace cloudViewer {
namespace core {
namespace kernel {

enum class ReductionOpCode {
    Sum,
    Prod,
    Min,
    Max,
    ArgMin,
    ArgMax,
    All,
    Any,
};

static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_regular_reduce_ops = {
                ReductionOpCode::Sum,
                ReductionOpCode::Prod,
                ReductionOpCode::Min,
                ReductionOpCode::Max,
};
static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_arg_reduce_ops = {
                ReductionOpCode::ArgMin,
                ReductionOpCode::ArgMax,
};
static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_boolean_reduce_ops = {
                ReductionOpCode::All,
                ReductionOpCode::Any,
};

void Reduction(const Tensor& src,
               Tensor& dst,
               const SizeVector& dims,
               bool keepdim,
               ReductionOpCode op_code);

void ReductionCPU(const Tensor& src,
                  Tensor& dst,
                  const SizeVector& dims,
                  bool keepdim,
                  ReductionOpCode op_code);

#ifdef BUILD_SYCL_MODULE
void ReductionSYCL(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keepdim,
                   ReductionOpCode op_code);
#endif

#ifdef BUILD_CUDA_MODULE
void ReductionCUDA(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keepdim,
                   ReductionOpCode op_code);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace cloudViewer
