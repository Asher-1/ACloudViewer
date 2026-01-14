// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "t/geometry/TensorListMap.h"

#include <vector>

#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace cloudViewer {
namespace tests {

TEST(TensorListMap, Constructor_GetPrimaryKey) {
    t::geometry::TensorListMap tm("points");
    EXPECT_EQ(tm.GetPrimaryKey(), "points");
}

TEST(TensorListMap, Assign) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm("points");
    tm["points"] = core::TensorList({3}, dtype, device);
    tm["dummy"] = core::TensorList({3}, dtype, device);
    EXPECT_TRUE(tm.Contains("points"));
    EXPECT_TRUE(tm.Contains("dummy"));

    std::unordered_map<std::string, core::TensorList> replacement{
            {"points", core::TensorList::FromTensor(
                               core::Tensor::Ones({5, 3}, dtype, device))},
            {"colors", core::TensorList::FromTensor(
                               core::Tensor::Ones({5, 3}, dtype, device))},
    };
    tm.Assign(replacement);
    EXPECT_TRUE(tm.Contains("points"));
    EXPECT_TRUE(tm.Contains("colors"));
    EXPECT_FALSE(tm.Contains("dummy"));

    // Underlying memory are the same.
    EXPECT_TRUE(
            tm["points"].AsTensor().IsSame(replacement["points"].AsTensor()));
    EXPECT_TRUE(
            tm["colors"].AsTensor().IsSame(replacement["colors"].AsTensor()));
}

TEST(TensorListMap, SynchronizedPushBack) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm(
            "points",
            {{"points", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))},
             {"colors", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))}});
    EXPECT_EQ(tm["points"].GetSize(), 5);
    EXPECT_EQ(tm["colors"].GetSize(), 5);

    // Good.
    core::Tensor a_point = core::Tensor::Ones({3}, dtype, device);
    core::Tensor a_color = core::Tensor::Ones({3}, dtype, device);
    tm.SynchronizedPushBack({{"points", a_point}, {"colors", a_color}});
    EXPECT_EQ(tm["points"].GetSize(), 6);
    EXPECT_EQ(tm["colors"].GetSize(), 6);
    EXPECT_TRUE(tm["points"][5].AllClose(a_point));
    EXPECT_FALSE(tm["points"][5].IsSame(a_point));  // PushBack copies memory.
    EXPECT_TRUE(tm["colors"][5].AllClose(a_color));
    EXPECT_FALSE(tm["colors"][5].IsSame(a_color));  // PushBack copies memory.

    // Missing key.
    EXPECT_ANY_THROW(tm.SynchronizedPushBack({{"colors", a_color}}));

    // Unexpected key.
    EXPECT_ANY_THROW(tm.SynchronizedPushBack({{"points", a_point},
                                              {"colors", a_color},
                                              {"more_colors", a_color}}));

    // Wrong dtype.
    EXPECT_ANY_THROW(tm.SynchronizedPushBack(
            {{"points", core::Tensor::Ones({3}, core::Dtype::Float64, device)},
             {"colors", a_color}}));

    // Wrong shape.
    EXPECT_ANY_THROW(tm.SynchronizedPushBack(
            {{"points", core::Tensor::Ones({5}, core::Dtype::Float32, device)},
             {"colors", a_color}}));
}

TEST(TensorListMap, IsSizeSynchronized) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm(
            "points",
            {{"points", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))},
             {"colors", core::TensorList::FromTensor(
                                core::Tensor::Ones({4, 3}, dtype, device))}});
    EXPECT_FALSE(tm.IsSizeSynchronized());

    tm["colors"].PushBack(core::Tensor::Ones({3}, dtype, device));
    EXPECT_TRUE(tm.IsSizeSynchronized());
}

TEST(TensorListMap, AssertSizeSynchronized) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm(
            "points",
            {{"points", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))},
             {"colors", core::TensorList::FromTensor(
                                core::Tensor::Ones({4, 3}, dtype, device))}});
    EXPECT_ANY_THROW(tm.AssertSizeSynchronized());

    tm["colors"].PushBack(core::Tensor::Ones({3}, dtype, device));
    EXPECT_NO_THROW(tm.AssertSizeSynchronized());
}

TEST(TensorListMap, Contains) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm(
            "points",
            {{"points", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))},
             {"colors", core::TensorList::FromTensor(
                                core::Tensor::Ones({4, 3}, dtype, device))}});
    EXPECT_TRUE(tm.Contains("points"));
    EXPECT_TRUE(tm.Contains("colors"));
    EXPECT_FALSE(tm.Contains("normals"));
}

}  // namespace tests
}  // namespace cloudViewer
