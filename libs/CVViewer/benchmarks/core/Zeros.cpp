// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include "core/CUDAUtils.h"
#include "core/Tensor.h"

namespace cloudViewer {
namespace core {

void Zeros(benchmark::State& state, const Device& device) {
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};

    Tensor warm_up = Tensor::Zeros(shape, core::Float32, device);
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = Tensor::Zeros(shape, core::Float32, device);
        cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(Zeros, CPU, Device("CPU:0"))->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Zeros, CUDA, Device("CUDA:0"))->Unit(benchmark::kMillisecond);
#endif

}  // namespace core
}  // namespace cloudViewer
