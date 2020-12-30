// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                       -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#include "t/geometry/PointCloud.h"

#include <benchmark/benchmark.h>

namespace cloudViewer {
namespace t {
namespace geometry {

void FromLegacyPointCloud(benchmark::State& state, const core::Device& device) {
    ccPointCloud legacy_pcd;
    unsigned int num_points = 1000000;  // 1M
    legacy_pcd.reserveThePointsTable(num_points);
    legacy_pcd.reserveTheRGBTable();
    legacy_pcd.resize(num_points);

    // Warm up.
    t::geometry::PointCloud pcd = t::geometry::PointCloud::FromLegacyPointCloud(
            legacy_pcd, core::Dtype::Float32, device);
    (void)pcd;

    for (auto _ : state) {
        t::geometry::PointCloud pcd =
                t::geometry::PointCloud::FromLegacyPointCloud(
                        legacy_pcd, core::Dtype::Float32, device);
    }
}

void ToLegacyPointCloud(benchmark::State& state, const core::Device& device) {
    int64_t num_points = 1000000;  // 1M
    PointCloud pcd(device);
    pcd.SetPoints(core::Tensor({num_points, 3}, core::Dtype::Float32, device));
    pcd.SetPointColors(
            core::Tensor({num_points, 3}, core::Dtype::Float32, device));

    // Warm up.
    ccPointCloud legacy_pcd = pcd.ToLegacyPointCloud();
    (void)legacy_pcd;

    for (auto _ : state) {
        ccPointCloud legacy_pcd = pcd.ToLegacyPointCloud();
    }
}

BENCHMARK_CAPTURE(FromLegacyPointCloud, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ToLegacyPointCloud, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(FromLegacyPointCloud, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(ToLegacyPointCloud, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
