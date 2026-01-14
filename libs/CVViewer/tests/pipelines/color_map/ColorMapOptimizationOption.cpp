// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tests/Tests.h"

// #include "pipelines/color_map/ColorMapOptimizationOption.h"

namespace cloudViewer {
namespace tests {

/* TODO
As the pipelines::color_map::ColorMapOptimization subcomponents go back into
hiding several lines of code had to commented out. Do not remove these lines,
they may become useful again after a decision has been made about the way to
make these subcomponents visible to UnitTest.
*/

TEST(ColorMapOptimizationOption, DISABLED_Constructor) {
    // cloudViewer::ColorMapOptimizationOption option;

    // EXPECT_FALSE(option.non_rigid_camera_coordinate_);

    // EXPECT_EQ(16, option.number_of_vertical_anchors_);
    // EXPECT_EQ(3, option.half_dilation_kernel_size_for_discontinuity_map_);

    // EXPECT_NEAR(0.316, option.non_rigid_anchor_point_weight_,
    // tests::THRESHOLD_1E_6); EXPECT_NEAR(300, option.maximum_iteration_,
    // tests::THRESHOLD_1E_6); EXPECT_NEAR(2.5,
    // option.maximum_allowable_depth_, tests::THRESHOLD_1E_6);
    // EXPECT_NEAR(0.03, option.depth_threshold_for_visibility_check_,
    // tests::THRESHOLD_1E_6); EXPECT_NEAR(0.1,
    // option.depth_threshold_for_discontinuity_check_,
    // tests::THRESHOLD_1E_6);
}

}  // namespace tests
}  // namespace cloudViewer
