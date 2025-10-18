// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdint>
#include <nanoflann.hpp>
#include <set>

#include "ml/contrib/Cloud.h"

namespace cloudViewer {
namespace ml {
namespace contrib {

/// TOOD: This is a temporary function for 3DML repositiory use. In the future,
/// the native CloudViewer Python API should be improved and used.
///
/// Nearest neighbours within a given radius.
/// For each query point, finds a set of neighbor indices whose
/// distance is less than given radius.
/// Modifies the neighbors_indices inplace.
void ordered_neighbors(std::vector<PointXYZ>& queries,
                       std::vector<PointXYZ>& supports,
                       std::vector<int>& neighbors_indices,
                       float radius);

/// TOOD: This is a temporary function for 3DML repositiory use. In the future,
/// the native CloudViewer Python API should be improved and used.
///
/// Nearest neighbours withing a radius with batching.
/// queries and supports are sliced with their respective batch elements.
/// Uses nanoflann to build a KDTree and find neighbors.
void batch_nanoflann_neighbors(std::vector<PointXYZ>& queries,
                               std::vector<PointXYZ>& supports,
                               std::vector<int>& q_batches,
                               std::vector<int>& s_batches,
                               std::vector<int>& neighbors_indices,
                               float radius);
}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
