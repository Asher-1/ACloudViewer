// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Upstream parity (d3ccaf35 feature/matcher.h): the new-style feature
// matching abstraction as a thin shell over the fork engines. The factory
// dispatches SIFT (in-memory CPU matching) and LoMa (ggml AICore); the
// ALIKED branch throws until its AICore task lands.

#include "estimators/two_view_geometry.h"
#include "feature/extractor.h"
#include "feature/learned_matchers.h"
#include "feature/sift.h"
#include "feature/types.h"
#include "geometry/pose_prior.h"
#include "scene/camera.h"

#include <memory>
#include <string>

namespace colmap {

struct FeatureMatchingOptions {
    explicit FeatureMatchingOptions(
            FeatureMatcherType type = FeatureMatcherType::SIFT_BRUTEFORCE);

    // Upstream parity (FeatureMatchingTypeOptions): all type-specific option
    // groups are allocated and deep-copied, so callers may switch `type`
    // freely and mutate any option group.
    FeatureMatchingOptions(const FeatureMatchingOptions& other);
    FeatureMatchingOptions& operator=(const FeatureMatchingOptions& other);
    FeatureMatchingOptions(FeatureMatchingOptions&& other) = default;
    FeatureMatchingOptions& operator=(FeatureMatchingOptions&& other) =
            default;

    FeatureMatcherType type = FeatureMatcherType::SIFT_BRUTEFORCE;

    // Number of threads for feature matching and geometric verification.
    int num_threads = -1;

    // Whether to use the GPU for feature matching.
#ifdef CUDA_ENABLED
    bool use_gpu = true;
#else
    bool use_gpu = false;
#endif

    // Index of the GPU used for feature matching.
    std::string gpu_index = "-1";

    // Maximum number of matches.
    int max_num_matches = 32768;

    // Whether to perform guided matching.
    bool guided_matching = false;

    // Skips the geometric verification stage and forwards matches unchanged.
    bool skip_geometric_verification = false;

    // Whether to perform rig-based geometric verification (fork: not
    // supported, kept for the upstream option layout).
    bool rig_verification = false;

    // Whether to skip matching images within the same frame.
    bool skip_image_pairs_in_same_frame = false;

    // Type-specific option sets (same flattening note as
    // FeatureExtractionOptions).
    std::shared_ptr<SiftMatchingOptions> sift;
    std::shared_ptr<LomaMatchingOptions> loma;
    std::shared_ptr<AlikedMatchingOptions> aliked;

    // Whether the selected matcher requires OpenGL.
    bool RequiresOpenGL() const;

    bool Check() const;
};

class FeatureMatcher {
public:
    virtual ~FeatureMatcher() = default;

    struct Image {
        // Unique identifier for the image. Allows a matcher to cache some
        // computations per image in consecutive calls to matching.
        image_t image_id = kInvalidImageId;
        std::shared_ptr<const Camera> camera;
        // Value semantics: the fork's feature types are plain Eigen/STL
        // containers with transparent pybind casters.
        std::vector<FeatureKeypoint> keypoints;
        FeatureDescriptorsFloat descriptors;
        const PosePrior* pose_prior = nullptr;
    };

    static std::unique_ptr<FeatureMatcher> Create(
            const FeatureMatchingOptions& options);

    virtual void Match(const Image& image1,
                       const Image& image2,
                       FeatureMatches* matches) = 0;

    virtual void MatchGuided(double max_error,
                             const Image& image1,
                             const Image& image2,
                             TwoViewGeometry* two_view_geometry) = 0;
};

}  // namespace colmap
