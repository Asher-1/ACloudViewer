// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "feature/types.h"
#include "sensor/bitmap.h"

namespace colmap {

struct SiftExtractionOptions;
struct AlikedExtractionOptions;
struct LomaExtractionOptions;

struct FeatureExtractionTypeOptions {
    explicit FeatureExtractionTypeOptions();

    std::shared_ptr<SiftExtractionOptions> sift;
    std::shared_ptr<AlikedExtractionOptions> aliked;
    std::shared_ptr<LomaExtractionOptions> loma;

    FeatureExtractionTypeOptions(const FeatureExtractionTypeOptions& other);
    FeatureExtractionTypeOptions& operator=(
            const FeatureExtractionTypeOptions& other);
    FeatureExtractionTypeOptions(FeatureExtractionTypeOptions&& other) =
            default;
    FeatureExtractionTypeOptions& operator=(
            FeatureExtractionTypeOptions&& other) = default;
};

struct FeatureExtractionOptions : public FeatureExtractionTypeOptions {
    explicit FeatureExtractionOptions(
            FeatureExtractorType type = FeatureExtractorType::SIFT);

    FeatureExtractorType type = FeatureExtractorType::SIFT;

    // Maximum image size, otherwise image will be down-scaled.
    // If max_image_size is non-positive, the appropriate size is selected
    // automatically based on the extractor type.
    int max_image_size = -1;

    // Number of threads for feature extraction.
    int num_threads = -1;

    // Whether to use the GPU for feature extraction.
#ifdef GPU_ENABLED
    bool use_gpu = true;
#else
    bool use_gpu = false;
#endif

    // Index of the GPU used for feature extraction. For multi-GPU extraction,
    // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
    std::string gpu_index = "-1";

    // Whether the selected extractor requires RGB (or grayscale) images.
    bool RequiresRGB() const;

    // Whether the selected extractor requires OpenGL.
    bool RequiresOpenGL() const;

    // Returns the effective maximum image size. If max_image_size is set to -1,
    // the appropriate size is selected automatically based on the extractor
    // type.
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
                         FeatureDescriptors* descriptors) = 0;
};

}  // namespace colmap
