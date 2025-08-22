// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/Device.h"

#include "tests/UnitTest.h"

namespace cloudViewer {
namespace tests {

TEST(Device, DefaultConstructor) {
    core::Device ctx;
    EXPECT_EQ(ctx.GetType(), core::Device::DeviceType::CPU);
    EXPECT_EQ(ctx.GetID(), 0);
}

TEST(Device, CPUMustBeID0) {
    EXPECT_EQ(core::Device(core::Device::DeviceType::CPU, 0).GetID(), 0);
    EXPECT_THROW(core::Device(core::Device::DeviceType::CPU, 1),
                 std::runtime_error);
}

TEST(Device, SpecifiedConstructor) {
    core::Device ctx(core::Device::DeviceType::CUDA, 1);
    EXPECT_EQ(ctx.GetType(), core::Device::DeviceType::CUDA);
    EXPECT_EQ(ctx.GetID(), 1);
}

TEST(Device, StringConstructor) {
    core::Device ctx("CUDA:1");
    EXPECT_EQ(ctx.GetType(), core::Device::DeviceType::CUDA);
    EXPECT_EQ(ctx.GetID(), 1);
}

TEST(Device, StringConstructorLower) {
    core::Device ctx("cuda:1");
    EXPECT_EQ(ctx.GetType(), core::Device::DeviceType::CUDA);
    EXPECT_EQ(ctx.GetID(), 1);
}

}  // namespace tests
}  // namespace cloudViewer
