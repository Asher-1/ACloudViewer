// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#include <benchmark/benchmark.h>

#include <ecvMesh.h>
#include <TriangleMeshIO.h>

namespace cloudViewer {
namespace benchmarks {

class SamplePointsFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) {
        trimesh = cloudViewer::io::CreateMeshFromFile(TEST_DATA_DIR "/knot.ply");
    }

    void TearDown(const benchmark::State& state) {
        // empty
    }
    std::shared_ptr<ccMesh> trimesh;
};

BENCHMARK_DEFINE_F(SamplePointsFixture, Poisson)(benchmark::State& state) {
    for (auto _ : state) {
        trimesh->samplePointsPoissonDisk(state.range(0));
    }
}

BENCHMARK_REGISTER_F(SamplePointsFixture, Poisson)->Args({123})->Args({1000});

BENCHMARK_DEFINE_F(SamplePointsFixture, Uniform)(benchmark::State& state) {
    for (auto _ : state) {
        trimesh->samplePointsUniformly(state.range(0));
    }
}

BENCHMARK_REGISTER_F(SamplePointsFixture, Uniform)->Args({123})->Args({1000});

}  // namespace benchmarks
}  // namespace cloudViewer
