// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_ESTIMATORS_EUCLIDEAN_TRANSFORM_H_
#define COLMAP_SRC_ESTIMATORS_EUCLIDEAN_TRANSFORM_H_

#include "base/similarity_transform.h"
#include "util/alignment.h"

namespace colmap {

// N-D Euclidean transform estimator from corresponding point pairs in the
// source and destination coordinate systems.
//
// This algorithm is based on the following paper:
//
//      S. Umeyama. Least-Squares Estimation of Transformation Parameters
//      Between Two Point Patterns. IEEE Transactions on Pattern Analysis and
//      Machine Intelligence, Volume 13 Issue 4, Page 376-380, 1991.
//      http://www.stanford.edu/class/cs273/refs/umeyama.pdf
//
// and uses the Eigen implementation.
template <int kDim>
using EuclideanTransformEstimator = SimilarityTransformEstimator<kDim, false>;

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_EUCLIDEAN_TRANSFORM_H_
