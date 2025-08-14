// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/core/Tensor.h"

namespace cloudViewer {
namespace t {
namespace geometry {
namespace kernel {
namespace pcapartition {

/// Partition the point cloud by recursively doing PCA.
/// \param points Points tensor with shape (N,3).
/// \param max_points The maximum allowed number of points in a partition.
/// \return The number of partitions and an int32 tensor with the partition id
/// for each point. The output tensor uses always the CPU device.
std::tuple<int, core::Tensor> PCAPartition(core::Tensor& points,
                                           int max_points);

}  // namespace pcapartition
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
