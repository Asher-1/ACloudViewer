// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include "cloudViewer/core/CUDAUtils.h"
#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/utility/Logging.h"

namespace cloudViewer {
namespace core {

void MatmulAB(benchmark::State& state, const Device& device) {
    Tensor A = Tensor::Ones({10000, 4}, core::Float32, device);
    Tensor B = Tensor::Ones({4, 10000}, core::Float32, device);

    Tensor output = A.Matmul(B);
    for (auto _ : state) {
        output = A.Matmul(B);
        core::cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(MatmulAB, CPU, Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(MatmulAB, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace core
}  // namespace cloudViewer
