// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include "torch/script.h"

template <class T>
torch::Tensor ReduceSubarraysSumCPU(const torch::Tensor& values,
                                    const torch::Tensor& row_splits);

#ifdef BUILD_CUDA_MODULE
template <class T>
torch::Tensor ReduceSubarraysSumCUDA(const torch::Tensor& values,
                                     const torch::Tensor& row_splits);
#endif
