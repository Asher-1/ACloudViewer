// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include <ecvMesh.h>
#include "cloudViewer/data/Dataset.h"
#include "cloudViewer/io/TriangleMeshIO.h"

namespace cloudViewer {
namespace benchmarks {

class SamplePointsFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) {
        data::KnotMesh knot_data;
        trimesh = io::CreateMeshFromFile(knot_data.GetPath());
    }

    void TearDown(const benchmark::State& state) {
        // empty
    }
    std::shared_ptr<ccMesh> trimesh;
};

BENCHMARK_DEFINE_F(SamplePointsFixture, Poisson)(benchmark::State& state) {
    for (auto _ : state) {
        trimesh->SamplePointsPoissonDisk(state.range(0));
    }
}

BENCHMARK_REGISTER_F(SamplePointsFixture, Poisson)->Args({123})->Args({1000});

BENCHMARK_DEFINE_F(SamplePointsFixture, Uniform)(benchmark::State& state) {
    for (auto _ : state) {
        trimesh->SamplePointsUniformly(state.range(0));
    }
}

BENCHMARK_REGISTER_F(SamplePointsFixture, Uniform)->Args({123})->Args({1000});

}  // namespace benchmarks
}  // namespace cloudViewer
