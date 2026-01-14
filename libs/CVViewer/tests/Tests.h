// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "CloudViewer.h"
#include "cloudViewer/Macro.h"
#include "cloudViewer/data/Dataset.h"
#include "test_utility/Compare.h"
#include "test_utility/Print.h"
#include "test_utility/Rand.h"
#include "test_utility/Raw.h"
#include "test_utility/Sort.h"

namespace cloudViewer {
namespace tests {

// Eigen Zero()
const Eigen::Vector2d Zero2d = Eigen::Vector2d::Zero();
const Eigen::Vector3d Zero3d = Eigen::Vector3d::Zero();
const Eigen::Matrix<double, 6, 1> Zero6d = Eigen::Matrix<double, 6, 1>::Zero();
const Eigen::Vector2i Zero2i = Eigen::Vector2i::Zero();

// Mechanism for reporting unit tests for which there is no implementation yet.
void NotImplemented();

#define AllCloseOrShow(Arr1, Arr2, rtol, atol)                               \
    EXPECT_TRUE(Arr1.AllClose(Arr2, rtol, atol)) << fmt::format(             \
            "Tensors are not close wrt (relative, absolute) tolerance ({}, " \
            "{}). Max error: {}\n{}\n{}",                                    \
            rtol, atol,                                                      \
            (Arr1 - Arr2)                                                    \
                    .Abs()                                                   \
                    .Flatten()                                               \
                    .Max({0})                                                \
                    .To(core::Float32)                                       \
                    .Item<float>(),                                          \
            Arr1.ToString(), Arr2.ToString());

}  // namespace tests
}  // namespace cloudViewer
