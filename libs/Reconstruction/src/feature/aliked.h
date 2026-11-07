// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>

#include "feature/extractor.h"
#include "feature/learned_matchers.h"
#include "feature/lightglue.h"
#include "feature/matcher.h"

namespace colmap {

// Upstream parity (feature/aliked.h): the ALIKED option contracts and
// factories. Upstream executes through onnxruntime; the fork executes through
// the AICore ggml task family (tasks/aliked + tasks/lightglue). Model defaults
// resolve through the shared aicore model catalog (AICORE_MODEL_FAMILY_ALIKED
// / LIGHTGLUE) and download into the AICore shared cache with pinned SHA-256
// verification instead of upstream's resources.h URIs.

struct AlikedExtractionOptions {
    // Maximum number of features to detect, keeping higher-score features.
    int max_num_features = 2048;

    // Minimum score threshold for keypoint detection.
    double min_score = 0.2;

    // The path to the ALIKED extractor GGUF models. Empty resolves to the
    // shared-catalog default at construction time; caller-provided paths are
    // never downloaded or replaced.
    std::string n16rot_model_path;
    std::string n32_model_path;

    // ggml engine placement (fork extension; upstream uses use_gpu/gpu_index).
    std::string device = "auto";

    bool Check() const;
};

std::unique_ptr<FeatureExtractor> CreateAlikedFeatureExtractor(
        const FeatureExtractionOptions& options);

struct AlikedMatchingOptions {
    // Brute-force matching options.
    BruteForceGGMLMatchingOptions brute_force = []() {
        BruteForceGGMLMatchingOptions options;
        options.min_cossim = 0.85;
        options.max_ratio = 1.0;
        options.cross_check = true;
        return options;
    }();

    // LightGlue matching options (ALIKED geometry).
    LightGlueGGMLMatchingOptions lightglue = []() {
        LightGlueGGMLMatchingOptions options;
        options.min_score = 0.1;
        return options;
    }();

    bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateAlikedFeatureMatcher(
        const FeatureMatchingOptions& options);

// Shared-catalog default path for the extractor role ("n16rot"/"n32"), or ""
// when unpublished. The optional quant selects "f32"/"f16"/"q8_0"; empty
// picks the catalog default (first row).
std::string DefaultAlikedExtractorModelPath(const char* role,
                                            const char* quant = "");

// Fills in the catalog-default extractor paths (no download). The N32 model
// is not published in the shared catalog yet, so its default resolves to "".
void ResolveDefaultAlikedModelPaths(AlikedExtractionOptions* options);

// Catalog-default extractor paths are provisioned through the shared AICore
// cache with SHA-256 verification; caller-provided paths are left untouched.
bool ProvisionDefaultAlikedModels(AlikedExtractionOptions* options);

}  // namespace colmap
