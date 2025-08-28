// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

namespace cloudViewer {
namespace pipelines {
namespace odometry {

/// \class OdometryOption
///
/// Class that defines Odometry options.
class OdometryOption {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param iteration_number_per_pyramid_level Number of iterations per level
    /// of pyramid.
    /// \param max_depth_diff Maximum depth difference to be considered as
    /// correspondence.
    /// \param min_depth Minimum depth below which pixel values
    /// are ignored.
    /// \param max_depth Maximum depth above which pixel values are
    /// ignored.
    OdometryOption(
            const std::vector<int> &iteration_number_per_pyramid_level =
                    {20, 10,
                     5} /* {smaller image size to original image size} */,
            double max_depth_diff = 0.03,
            double min_depth = 0.0,
            double max_depth = 4.0)
        : iteration_number_per_pyramid_level_(
                  iteration_number_per_pyramid_level),
          max_depth_diff_(max_depth_diff),
          min_depth_(min_depth),
          max_depth_(max_depth) {}
    ~OdometryOption() {}

public:
    /// Iteration number per image pyramid level, typically larger image in the
    /// pyramid have lower interation number to reduce computation time.
    std::vector<int> iteration_number_per_pyramid_level_;
    /// Maximum depth difference to be considered as correspondence. In depth
    /// image domain, if two aligned pixels have a depth difference less than
    /// specified value, they are considered as a correspondence. Larger value
    /// induce more aggressive search, but it is prone to unstable result.
    double max_depth_diff_;
    /// Pixels that has larger than specified depth values are ignored.
    double min_depth_;
    /// Pixels that has larger than specified depth values are ignored.
    double max_depth_;
};

}  // namespace odometry
}  // namespace pipelines
}  // namespace cloudViewer
