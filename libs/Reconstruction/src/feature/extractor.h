// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Upstream parity (d3ccaf35 feature/extractor.h): the new-style feature
// extraction abstraction as a thin shell over the fork engines. The factory
// dispatches SIFT (CPU) and LoMa (ggml AICore); the ALIKED branch throws
// until its AICore task lands.

#include <memory>
#include <string>

#include "feature/aliked.h"
#include "feature/loma.h"
#include "feature/sift.h"
#include "feature/types.h"
#include "sensor/bitmap.h"

namespace colmap {

struct FeatureExtractionOptions {
    explicit FeatureExtractionOptions(
            FeatureExtractorType type = FeatureExtractorType::SIFT);

    // Upstream parity (FeatureExtractionTypeOptions): all type-specific
    // option groups are allocated and deep-copied, so callers may switch
    // `type` freely and mutate any option group.
    FeatureExtractionOptions(const FeatureExtractionOptions& other);
    FeatureExtractionOptions& operator=(const FeatureExtractionOptions& other);
    FeatureExtractionOptions(FeatureExtractionOptions&& other) = default;
    FeatureExtractionOptions& operator=(FeatureExtractionOptions&& other) =
            default;

    FeatureExtractorType type = FeatureExtractorType::SIFT;

    // Maximum image size, otherwise image will be down-scaled.
    int max_image_size = -1;

    // Number of threads for feature extraction.
    int num_threads = -1;

    // Whether to use the GPU for feature extraction.
#ifdef CUDA_ENABLED
    bool use_gpu = true;
#else
    bool use_gpu = false;
#endif

    // Index of the GPU used for feature extraction.
    std::string gpu_index = "-1";

    // Type-specific option sets (upstream carries these as shared_ptr
    // members in FeatureExtractionTypeOptions; flattened here with the same
    // copy semantics via the accessor methods).
    std::shared_ptr<SiftExtractionOptions> sift;
    std::shared_ptr<LomaExtractionOptions> loma;
    std::shared_ptr<AlikedExtractionOptions> aliked;

    // Whether the selected extractor requires RGB (or grayscale) images.
    bool RequiresRGB() const;

    // Whether the selected extractor requires OpenGL.
    bool RequiresOpenGL() const;

    // Returns the effective maximum image size. If max_image_size is
    // non-positive, the appropriate size is selected automatically based
    // on the extractor type.
    int EffMaxImageSize() const;

    bool Check() const;
};

class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    static std::unique_ptr<FeatureExtractor> Create(
            const FeatureExtractionOptions& options);

    virtual bool Extract(const Bitmap& bitmap,
                         FeatureKeypoints* keypoints,
                         FeatureDescriptorsFloat* descriptors) = 0;
};

}  // namespace colmap
