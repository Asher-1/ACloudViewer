// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "feature/types.h"

namespace colmap {

// Convert feature keypoints to vector of points.
std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
        const FeatureKeypoints& keypoints);

// L2-normalize feature descriptor, where each row represents one feature.
// Upstream parity (d3ccaf35 feature/utils.h): the descriptor matrices are
// RowMajor — a ColMajor parameter would implicitly transpose the caller's
// RowMajor data and change the row-reduction (lpNorm) summation order, which
// tips uchar descriptor elements by ±1 downstream.
FeatureDescriptorsFloat L2NormalizeFeatureDescriptors(
        const FeatureDescriptorsFloat& descriptors);

// L1-Root-normalize feature descriptors, where each row represents one feature.
// See "Three things everyone should know to improve object retrieval",
// Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
FeatureDescriptorsFloat L1RootNormalizeFeatureDescriptors(
        const FeatureDescriptorsFloat& descriptors);

// Convert normalized floating point feature descriptor to unsigned byte
// representation by linear scaling from range [0, 0.5] to [0, 255]. Truncation
// to a maximum value of 0.5 is used to avoid precision loss and follows the
// common practice of representing SIFT vectors.
FeatureDescriptors FeatureDescriptorsToUnsignedByte(
        const FeatureDescriptorsFloat& descriptors);

// Extract the descriptors corresponding to the largest-scale features.
void ExtractTopScaleFeatures(FeatureKeypoints* keypoints,
                             FeatureDescriptors* descriptors,
                             const size_t num_features);

}  // namespace colmap
