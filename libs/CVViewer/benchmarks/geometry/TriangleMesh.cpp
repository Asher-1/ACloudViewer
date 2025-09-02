// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include "cloudViewer/data/Dataset.h"
#include "cloudViewer/io/PointCloudIO.h"

namespace cloudViewer {
namespace pipelines {
namespace registration {

// This pcd does not contains non-finite points.
// TODO: Change this to pcd with non-finite points.
static void BenchmarkCreateFromPointCloudBallPivoting(
        benchmark::State& state, const bool remove_non_finite_points) {
    data::PCDPointCloud sample_pcd;
    auto pcd = io::CreatePointCloudFromFile(sample_pcd.GetPath());

    if (remove_non_finite_points) {
        pcd->RemoveNonFinitePoints();
    }

    std::vector<double> distance = pcd->ComputeNearestNeighborDistance();
    size_t n = distance.size();
    double dist_average = 0.0;
    if (n != 0) {
        dist_average =
                std::accumulate(distance.begin(), distance.end(), 0.0) / n;
    }
    double radius = 1.5 * dist_average;
    std::vector<double> radii = {radius, radius * 1};
    std::shared_ptr<ccMesh> mesh;

    mesh = ccMesh::CreateFromPointCloudBallPivoting(*pcd, radii);

    for (auto _ : state) {
        mesh = ccMesh::CreateFromPointCloudBallPivoting(*pcd, radii);
    }
}

BENCHMARK_CAPTURE(BenchmarkCreateFromPointCloudBallPivoting,
                  Without Non Finite Points,
                  /*remove_non_finite_points*/ true)
        ->Unit(benchmark::kMillisecond);

// TODO: Add BENCHMARK for case `With Non Finite Points`.

}  // namespace registration
}  // namespace pipelines
}  // namespace cloudViewer
