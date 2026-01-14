// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/nns/NanoFlannIndex.h"

#include <Helper.h>
#include <ecvPointCloud.h>

#include <cmath>
#include <limits>

#include "core/Dtype.h"
#include "core/SizeVector.h"
#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(NanoFlannIndex, SearchKnn) {
    // set up index
    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NanoFlannIndex index(ref);

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64);

    // if k is smaller or equal to 0
    EXPECT_THROW(index.SearchKnn(query, -1), std::runtime_error);
    EXPECT_THROW(index.SearchKnn(query, 0), std::runtime_error);

    // if k == 3
    core::Tensor indices;
    core::Tensor distances;
    std::tie(indices, distances) = index.SearchKnn(query, 3);

    ExpectEQ(indices.ToFlatVector<int64_t>(), std::vector<int64_t>({1, 4, 9}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938, 0.0108912}));
    EXPECT_EQ(indices.GetShape(), core::SizeVector({1, 3}));
    EXPECT_EQ(distances.GetShape(), core::SizeVector({1, 3}));

    // if k > size
    std::tie(indices, distances) = index.SearchKnn(query, 12);

    ExpectEQ(indices.ToFlatVector<int64_t>(),
             std::vector<int64_t>({1, 4, 9, 0, 3, 2, 5, 7, 6, 8}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938, 0.0108912, 0.0138322,
                                  0.015048, 0.018695, 0.0199108, 0.0286952,
                                  0.0362638, 0.0411266}));
    EXPECT_EQ(indices.GetShape(), core::SizeVector({1, 10}));
    EXPECT_EQ(distances.GetShape(), core::SizeVector({1, 10}));
}

TEST(NanoFlannIndex, SearchRadius) {
    std::vector<int> ref_indices = {1, 4};
    std::vector<double> ref_distance = {0.00626358, 0.00747938};

    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NanoFlannIndex index(ref);

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64);

    // if radius <= 0
    EXPECT_THROW(index.SearchRadius(query, -1.0), std::runtime_error);
    EXPECT_THROW(index.SearchRadius(query, 0.0), std::runtime_error);

    // if radius == 0.1
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            index.SearchRadius(query, 0.1);
    core::Tensor indices = std::get<0>(result).To(core::Dtype::Int32);
    core::Tensor distances = std::get<1>(result);
    ExpectEQ(indices.ToFlatVector<int>(), std::vector<int>({1, 4}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938}));
}

}  // namespace tests
}  // namespace cloudViewer
