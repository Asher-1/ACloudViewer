// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include "cloudViewer/core/AdvancedIndexing.h"
#include "cloudViewer/core/CUDAUtils.h"
#include "cloudViewer/core/Dtype.h"
#include "cloudViewer/core/MemoryManager.h"
#include "cloudViewer/core/SizeVector.h"
#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/core/kernel/Kernel.h"

namespace cloudViewer {
namespace core {

void Reduction(benchmark::State& state, const Device& device) {
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};
    Tensor src(shape, core::Int64, device);
    Tensor warm_up = src.Sum({1});
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = src.Sum({1});
        cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(Reduction, CPU, Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Reduction, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace core
}  // namespace cloudViewer
