// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
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

#pragma once

#include "core/Tensor.h"
#include "core/linalg/Tri.h"

namespace cloudViewer {
namespace core {

void TriuCPU(const Tensor& A, Tensor& output, const int diagonal = 0);

void TrilCPU(const Tensor& A, Tensor& output, const int diagonal = 0);

void TriulCPU(const Tensor& A,
              Tensor& upper,
              Tensor& lower,
              const int diagonal = 0);

#ifdef BUILD_CUDA_MODULE
void TriuCUDA(const Tensor& A, Tensor& output, const int diagonal = 0);

void TrilCUDA(const Tensor& A, Tensor& output, const int diagonal = 0);

void TriulCUDA(const Tensor& A,
               Tensor& upper,
               Tensor& lower,
               const int diagonal = 0);
#endif
}  // namespace core
}  // namespace cloudViewer
