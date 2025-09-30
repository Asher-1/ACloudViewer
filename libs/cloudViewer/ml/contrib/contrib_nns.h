// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/nns/NearestNeighborSearch.h"

namespace cloudViewer {
namespace ml {
namespace contrib {

/// TOOD: This is a temory wrapper for 3DML repositiory use. In the future, the
/// native CloudViewer Python API should be improved and used.
///
/// \param query_points Tensor of shape {n_query_points, d}, dtype Float32.
/// \param dataset_points Tensor of shape {n_dataset_points, d}, dtype Float32.
/// \param knn Int.
/// \return Tensor of shape (n_query_points, knn), dtype Int32.
const core::Tensor KnnSearch(const core::Tensor& query_points,
                             const core::Tensor& dataset_points,
                             int knn);

/// TOOD: This is a temory wrapper for 3DML repositiory use. In the future, the
/// native CloudViewer Python API should be improved and used.
///
/// \param query_points Tensor of shape {n_query_points, d}, dtype Float32.
/// \param dataset_points Tensor of shape {n_dataset_points, d}, dtype Float32.
/// \param query_batches Tensor of shape {n_batches,}, dtype Int32. It is
/// required that sum(query_batches) == n_query_points.
/// \param dataset_batches Tensor of shape {n_batches,}, dtype Int32. It is
/// required that that sum(dataset_batches) == n_dataset_points.
/// \param radius The radius to search.
/// \return Tensor of shape {n_query_points, max_neighbor}, dtype Int32, where
/// max_neighbor is the maximum number neighbor of neighbors for all query
/// points. For query points with less than max_neighbor neighbors, the neighbor
/// index will be padded by -1.
const core::Tensor RadiusSearch(const core::Tensor& query_points,
                                const core::Tensor& dataset_points,
                                const core::Tensor& query_batches,
                                const core::Tensor& dataset_batches,
                                double radius);
}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
