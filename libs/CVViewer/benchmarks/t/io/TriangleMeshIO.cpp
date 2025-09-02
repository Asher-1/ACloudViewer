// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/io/TriangleMeshIO.h"

#include <benchmark/benchmark.h>

#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/data/Dataset.h"
#include "cloudViewer/t/io/TriangleMeshIO.h"

namespace cloudViewer {
namespace t {
namespace geometry {

data::KnotMesh knot_data;

void IOReadLegacyTriangleMesh(benchmark::State& state,
                              const std::string& input_file_path) {
    ccMesh mesh;
    mesh.createInternalCloud();
    cloudViewer::io::ReadTriangleMesh(input_file_path, mesh);

    for (auto _ : state) {
        cloudViewer::io::ReadTriangleMesh(input_file_path, mesh);
    }
}

void IOReadTensorTriangleMesh(benchmark::State& state,
                              const std::string& input_file_path) {
    t::geometry::TriangleMesh mesh;
    t::io::ReadTriangleMesh(input_file_path, mesh);

    for (auto _ : state) {
        t::io::ReadTriangleMesh(input_file_path, mesh);
    }
}

BENCHMARK_CAPTURE(IOReadLegacyTriangleMesh, CPU, knot_data.GetPath())
        ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(IOReadTensorTriangleMesh, CPU, knot_data.GetPath())
        ->Unit(benchmark::kMillisecond);

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
