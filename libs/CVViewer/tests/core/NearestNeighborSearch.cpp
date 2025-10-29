// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/nns/NearestNeighborSearch.h"

#include <Helper.h>
#include <ecvPointCloud.h>

#include <cmath>
#include <limits>

#include "core/Dtype.h"
#include "core/SizeVector.h"
#include "tests/UnitTest.h"

namespace cloudViewer {
namespace tests {

TEST(NearestNeighborSearch, KnnSearch) {
    // Set up nns.
    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighborSearch nns(ref);
    nns.KnnIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64);
    std::pair<core::Tensor, core::Tensor> result;
    core::Tensor indices;
    core::Tensor distances;

    // If k <= 0.
    EXPECT_THROW(nns.KnnSearch(query, -1), std::runtime_error);
    EXPECT_THROW(nns.KnnSearch(query, 0), std::runtime_error);

    // If k == 3.
    result = nns.KnnSearch(query, 3);
    indices = result.first;
    distances = result.second;
    ExpectEQ(indices.ToFlatVector<int64_t>(), std::vector<int64_t>({1, 4, 9}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938, 0.0108912}));

    // If k > size.result.
    result = nns.KnnSearch(query, 12);
    indices = result.first;
    distances = result.second;
    ExpectEQ(indices.ToFlatVector<int64_t>(),
             std::vector<int64_t>({1, 4, 9, 0, 3, 2, 5, 7, 6, 8}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938, 0.0108912, 0.0138322,
                                  0.015048, 0.018695, 0.0199108, 0.0286952,
                                  0.0362638, 0.0411266}));

    // Multiple points.
    query = core::Tensor(std::vector<double>({0.064705, 0.043921, 0.087843,
                                              0.064705, 0.043921, 0.087843}),
                         {2, 3}, core::Dtype::Float64);
    result = nns.KnnSearch(query, 3);
    indices = result.first;
    distances = result.second;
    ExpectEQ(indices.ToFlatVector<int64_t>(),
             std::vector<int64_t>({1, 4, 9, 1, 4, 9}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938, 0.0108912, 0.00626358,
                                  0.00747938, 0.0108912}));
}

TEST(NearestNeighborSearch, FixedRadiusSearch) {
    // Set up nns.
    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighborSearch nns(ref);
    nns.FixedRadiusIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64);

    // If radius <= 0.
    EXPECT_THROW(nns.FixedRadiusSearch(query, -1.0), std::runtime_error);
    EXPECT_THROW(nns.FixedRadiusSearch(query, 0.0), std::runtime_error);

    // If radius == 0.1.
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            nns.FixedRadiusSearch(query, 0.1);
    core::Tensor indices = std::get<0>(result);
    core::Tensor distances = std::get<1>(result);

    ExpectEQ(indices.ToFlatVector<int64_t>(), std::vector<int64_t>({1, 4}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938}));
}

TEST(NearestNeighborSearch, MultiRadiusSearch) {
    // Set up nns.
    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighborSearch nns(ref);
    nns.MultiRadiusIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843,
                                            0.064705, 0.043921, 0.087843}),
                       {2, 3}, core::Dtype::Float64);
    core::Tensor radius;

    // If radius <= 0.
    radius = core::Tensor(std::vector<double>({1.0, 0.0}), {2},
                          core::Dtype::Float64);
    EXPECT_THROW(nns.MultiRadiusSearch(query, radius), std::runtime_error);
    EXPECT_THROW(nns.MultiRadiusSearch(query, radius), std::runtime_error);

    // If radius == 0.1.
    radius = core::Tensor(std::vector<double>({0.1, 0.1}), {2},
                          core::Dtype::Float64);
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            nns.MultiRadiusSearch(query, radius);
    core::Tensor indices = std::get<0>(result);
    core::Tensor distances = std::get<1>(result);

    ExpectEQ(indices.ToFlatVector<int64_t>(),
             std::vector<int64_t>({1, 4, 1, 4}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>(
                     {0.00626358, 0.00747938, 0.00626358, 0.00747938}));
}

TEST(NearestNeighborSearch, HybridSearch) {
    // Set up nns.
    int size = 10;

    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighborSearch nns(ref);
    nns.HybridIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64);

    std::pair<core::Tensor, core::Tensor> result =
            nns.HybridSearch(query, 0.1, 1);

    core::Tensor indices = result.first;
    core::Tensor distainces = result.second;
    ExpectEQ(indices.ToFlatVector<int64_t>(), std::vector<int64_t>({1}));
    ExpectEQ(distainces.ToFlatVector<double>(),
             std::vector<double>({0.00626358}));
}

}  // namespace tests
}  // namespace cloudViewer
