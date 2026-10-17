// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

// Upstream parity (d3ccaf35 sfm/incremental_mapper_impl.h): the mapper's
// selection heuristics and the initial-pair orchestration live in a stateless
// algorithm class whose methods receive the mapper state explicitly.

#include <optional>
#include <vector>

#include "geometry/pose.h"
#include "scene/camera.h"
#include "scene/database_cache.h"
#include "scene/reconstruction.h"
#include "sfm/incremental_mapper.h"
#include "sfm/observation_manager.h"
#include "util/hash_containers.h"
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
            const FlatHashMap<image_t, size_t>& init_num_reg_trials,
            const FlatHashMap<image_t, size_t>& num_registrations);

    // For a given first seed image, find other images that are connected to
    // the first image. Suitable second images have a large number of
    // correspondences to the first image and have camera calibration priors.
    static std::vector<image_t> FindSecondInitialImage(
            const IncrementalMapper::Options& options,
            image_t image_id1,
            const CorrespondenceGraph& correspondence_graph,
            const Reconstruction& reconstruction,
            const FlatHashMap<image_t, size_t>& num_registrations);

    // Result of selecting and/or estimating the initial image pair. On
    // success, `camera1`/`camera2` carry the intrinsics estimated for the
    // chosen pair by two-view solvers that recover them (e.g. the
    // shared-focal solver, which sets both to the same camera), and are
    // std::nullopt otherwise.
    struct InitInfo {
        image_t image_id1 = kInvalidImageId;
        image_t image_id2 = kInvalidImageId;
        Rigid3d cam2_from_cam1;
        std::optional<Camera> camera1;
        std::optional<Camera> camera2;
    };

    // Implement IncrementalMapper::FindInitialImagePair. Returns the selected
    // pair, or std::nullopt if no suitable pair was found. `image_id1`/
    // `image_id2` optionally constrain the search to a specific first and/or
    // second image (kInvalidImageId leaves the respective image
    // unconstrained).
    //
    // Fork decision (第四十六轮手术边界): the candidate scan stays serial.
    // Upstream fans the first-seed candidates out over a ThreadPool, which is
    // a legal nondeterminism axis that the within-distribution gate does not
    // require; the serial scan is what gives the fork its A/A reproducibility.
    static std::optional<InitInfo> FindInitialImagePair(
            const IncrementalMapper::Options& options,
            const DatabaseCache& database_cache,
            const Reconstruction& reconstruction,
            const FlatHashMap<image_t, size_t>& init_num_reg_trials,
            const FlatHashMap<image_t, size_t>& num_registrations,
            FlatHashSet<image_pair_t>& init_image_pairs,
            image_t image_id1,
            image_t image_id2);

    // Implement IncrementalMapper::FindNextImages (upstream frame-aware
    // form: visibility ranks come from the ObservationManager and filtering
    // happens at frame granularity).
    static std::vector<image_t> FindNextImages(
            const IncrementalMapper::Options& options,
            const ObservationManager& obs_manager,
            const FlatHashSet<frame_t>& filtered_frames,
            FlatHashMap<image_t, size_t>& num_reg_trials,
            bool structure_less = false);

    // Implement IncrementalMapper::FindLocalBundle.
    static std::vector<image_t> FindLocalBundle(
            const IncrementalMapper::Options& options,
            image_t image_id,
            const Reconstruction& reconstruction);

    // Implement IncrementalMapper::EstimateInitialTwoViewGeometry. Returns
    // the estimated two-view geometry, or std::nullopt if the pair is
    // unsuitable for initialization.
    static std::optional<InitInfo> EstimateInitialTwoViewGeometry(
            const IncrementalMapper::Options& options,
            const DatabaseCache& database_cache,
            image_t image_id1,
            image_t image_id2);
};

}  // namespace colmap
