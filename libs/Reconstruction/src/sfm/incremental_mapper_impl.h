// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

// Upstream parity (d3ccaf35 sfm/incremental_mapper_impl.h): the mapper's
// selection heuristics live in a stateless algorithm class whose methods
// receive the mapper state explicitly. The fork moves its legacy member
// helpers here unchanged (W3-3); the upstream InitInfo orchestration for
// FindInitialImagePair / EstimateInitialTwoViewGeometry stays in
// IncrementalMapper because it touches the mapper's book-keeping members.

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "scene/reconstruction.h"
#include "sfm/incremental_mapper.h"
#include "util/types.h"

namespace colmap {

class CorrespondenceGraph;

// Algorithm class for the incremental mapper to make it easier to extend.
class IncrementalMapperImpl {
public:
    // Find seed images for incremental reconstruction. Suitable seed images
    // have a large number of correspondences and have camera calibration
    // priors. The returned list is ordered such that the most suitable images
    // are in the front.
    static std::vector<image_t> FindFirstInitialImage(
            const IncrementalMapper::Options& options,
            const CorrespondenceGraph& correspondence_graph,
            const Reconstruction& reconstruction,
            const std::unordered_map<image_t, size_t>& init_num_reg_trials,
            const std::unordered_map<image_t, size_t>& num_registrations);

    // For a given first seed image, find other images that are connected to
    // the first image. Suitable second images have a large number of
    // correspondences to the first image and have camera calibration priors.
    static std::vector<image_t> FindSecondInitialImage(
            const IncrementalMapper::Options& options,
            image_t image_id1,
            const CorrespondenceGraph& correspondence_graph,
            const Reconstruction& reconstruction,
            const std::unordered_map<image_t, size_t>& num_registrations);

    // Implement IncrementalMapper::FindNextImages.
    static std::vector<image_t> FindNextImages(
            const IncrementalMapper::Options& options,
            const Reconstruction& reconstruction,
            const std::unordered_map<image_t, size_t>& num_reg_trials,
            const std::unordered_set<image_t>& filtered_images);

    // Implement IncrementalMapper::FindLocalBundle.
    static std::vector<image_t> FindLocalBundle(
            const IncrementalMapper::Options& options,
            image_t image_id,
            const Reconstruction& reconstruction);
};

}  // namespace colmap
