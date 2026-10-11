// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Upstream parity (feature/loma.h): the LoMa extractor / matcher contracts.
// Upstream executes through onnxruntime and resolves models from
// feature/resources.h URIs; the fork executes through the AICore ggml LoMa
// task and resolves its default models from the shared AICore LoMa catalog.
//
// Reference: https://github.com/davnords/LoMa

#include <memory>
#include <string>

#include "aicore/loma_capi.h"
#include "feature/extractor.h"
#include "feature/learned_matchers.h"
#include "feature/matcher.h"

namespace colmap {

struct LomaExtractionOptions {
    // Number of keypoints to detect. Recommended is to use either 2048 or
    // 4096.
    int max_num_features = 2048;

    // The DaD detector does not filter in the same way as ALIKED. Thus, we
    // set min_score = 0.0.
    double min_score = 0.0;

    // Detector (DaD) is shared across all LoMa variants. Empty resolves to the
    // shared-catalog default at construction time; caller-provided paths are
    // never downloaded or replaced.
    std::string detector_model_path;

    // Descriptor for FeatureExtractorType::LOMA_B (dedode_g, also used by the
    // LOMA_B/LOMA_R/LOMA_L/LOMA_G matcher variants).
    std::string descriptor_model_path;

    // Descriptor for FeatureExtractorType::LOMA_B128 (dedode_b, VGG-only).
    std::string descriptor_b128_model_path;

    // ggml engine placement (fork extension; upstream uses use_gpu/gpu_index).
    std::string device = "auto";

    bool Check() const;
};

std::unique_ptr<FeatureExtractor> CreateLomaFeatureExtractor(
        const FeatureExtractionOptions& options);

struct LomaMatchingOptions {
    // Per-variant model paths. Empty resolves to the shared-catalog default.
    struct Variant {
        std::string model_path;
    };

    // Matching filter, matches LG.
    double min_score = 0.1;

    // One entry per FeatureMatcherType::LOMA_* dedicated-matcher variant.
    Variant b;
    Variant b128;
    Variant r;
    Variant l;
    Variant g;

    // Brute-force matching options (FeatureMatcherType::LOMA_BRUTEFORCE);
    // descriptor-dimension agnostic cosine-similarity matching.
    BruteForceGGMLMatchingOptions brute_force = []() {
        BruteForceGGMLMatchingOptions options;
        options.min_cossim = 0.85;
        options.max_ratio = 1.0;
        options.cross_check = true;
        return options;
    }();

    // ggml engine placement (fork extension).
    std::string device = "auto";

    bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateLomaFeatureMatcher(
        const FeatureMatchingOptions& options);

// Shared-catalog default path for a LoMa model role/variant, or "" when
// unpublished.
std::string DefaultLomaModelPath(aicore_loma_model_role role);
std::string DefaultLomaModelPath(aicore_loma_model_variant variant);

// The descriptor model variant for a LoMa extractor type and the matcher model
// variant for a LoMa matcher type.
aicore_loma_model_variant LomaDescriptorVariant(FeatureExtractorType type);
aicore_loma_model_variant LomaMatcherVariant(FeatureMatcherType type);

// Fill in catalog-default paths (no download) for the models the given
// extractor/matcher type consumes.
void ResolveDefaultLomaModelPaths(FeatureExtractorType type,
                                  LomaExtractionOptions* options);
void ResolveDefaultLomaModelPaths(FeatureMatcherType type,
                                  LomaMatchingOptions* options);

// Catalog-default paths are provisioned through the shared AICore cache with
// SHA-256 verification; non-default caller-provided paths are never
// downloaded or replaced.
bool ProvisionDefaultLomaModels(FeatureExtractorType type,
                                LomaExtractionOptions* options);
bool ProvisionDefaultLomaModels(FeatureMatcherType type,
                                LomaMatchingOptions* options);

}  // namespace colmap
