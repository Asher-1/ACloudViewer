// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>

#include "feature/types.h"
#include "sensor/bitmap.h"

namespace colmap {

// Forward declarations: the factories are declared here for upstream parity
// but the full option-shell types live in feature/extractor.h and
// feature/matcher.h, which include this header for the option structs.
struct FeatureExtractionOptions;
struct FeatureMatchingOptions;
class FeatureExtractor;
class FeatureMatcher;

// Upstream parity (d3ccaf35 feature/aliked.h): the ALIKED option contracts.
// Upstream executes through onnxruntime; the fork executes through the AICore
// ggml task family (tasks/aliked + tasks/lightglue). Model defaults resolve
// through the shared aicore model catalog (AICORE_MODEL_FAMILY_ALIKED /
// LIGHTGLUE) and download into the AICore shared cache with pinned SHA-256
// verification, mirroring the LoMa wiring (feature/loma.cc).
//
// Upstream's nested BruteForceONNXMatchingOptions / LightGlueONNXMatchingOptions
// groups are flattened onto AlikedMatchingOptions (learned_matchers.h
// adjudication carried by feature/matcher_test.cc). The brute-force matcher
// itself is pure Eigen cosine-similarity matching on the fork - semantically
// identical to upstream's brute-force ONNX network (mutual NN + min_cossim +
// ratio test + cross check) without a model round trip.

struct AlikedExtractionOptions {
    // Maximum number of features to detect, keeping higher-score features.
    int max_num_features = 2048;

    // Minimum score threshold for keypoint detection.
    double min_score = 0.2;

    // The path to the ALIKED extractor GGUF model. Empty resolves to the
    // shared-catalog default at Check()/extraction time; caller-provided
    // paths are never downloaded or replaced.
    std::string n16rot_model_path;
    std::string n32_model_path;

    // ggml engine placement (fork extension; upstream uses use_gpu/gpu_index).
    std::string device = "auto";

    bool Check() const;
};

struct AlikedMatchingOptions {
    // Brute-force matching (upstream: BruteForceONNXMatchingOptions).
    // Minimum cosine similarity for a match to be considered valid.
    double min_cossim = 0.85;

    // Maximum ratio for the second-best/best similarity test; 1 disables it.
    double max_ratio = 1.0;

    // Enable cross-checking (mutual nearest neighbor).
    bool cross_check = true;

    // LightGlue matching (upstream: LightGlueONNXMatchingOptions).
    // Minimum accepted match score (post-model filtering).
    double min_score = 0.1;

    // Path to the LightGlue GGUF matcher model (ALIKED geometry). Empty
    // resolves to the shared-catalog default.
    std::string lightglue_model_path;

    // ggml engine placement (fork extension).
    std::string device = "auto";

    bool Check() const;
};

// Shared-catalog default path for the extractor role ("n16rot"/"n32") and
// the ALIKED-geometry LightGlue matcher, or "" when unpublished. The
// optional quant selects "f32"/"f16"/"q8_0"; empty picks the catalog
// default (first row, F16).
std::string DefaultAlikedExtractorModelPath(const char* role,
                                            const char* quant = "");
std::string DefaultAlikedLightGlueModelPath(const char* quant = "");

// Fill in the catalog-default paths (no download); mirroring
// ResolveDefaultLomaModelPaths. The N32 extractor model is not published
// upstream yet, so its default resolves to "".
void ResolveDefaultAlikedModelPaths(AlikedExtractionOptions* options);
void ResolveDefaultAlikedModelPaths(AlikedMatchingOptions* options);

// Catalog-default paths are provisioned through the shared AICore cache with
// SHA-256 verification; non-default caller-provided paths are never
// downloaded or replaced.
bool ProvisionDefaultAlikedModels(AlikedExtractionOptions* options);
bool ProvisionDefaultAlikedModels(AlikedMatchingOptions* options);

// Stateless entry points (fork parity of the upstream engine functions).
bool ExtractAlikedFeatures(const AlikedExtractionOptions& options,
                           const Bitmap& bitmap,
                           FeatureKeypoints* keypoints,
                           FeatureDescriptorsFloat* descriptors);

// Brute-force cosine-similarity matching (options.min_cossim/max_ratio/
// cross_check). The descriptors are matched as-is; callers pass
// unit-length or raw rows alike because cosine similarity normalizes.
bool MatchAlikedFeaturesBruteForce(
        const AlikedMatchingOptions& options,
        const FeatureKeypoints& keypoints1,
        const FeatureDescriptorsFloat& descriptors1,
        const FeatureKeypoints& keypoints2,
        const FeatureDescriptorsFloat& descriptors2,
        FeatureMatches* matches);

// LightGlue matching through the AICore ggml task (ALIKED geometry).
bool MatchAlikedFeaturesLightGlue(
        const AlikedMatchingOptions& options,
        const FeatureKeypoints& keypoints1,
        const FeatureDescriptorsFloat& descriptors1,
        int image_width1,
        int image_height1,
        const FeatureKeypoints& keypoints2,
        const FeatureDescriptorsFloat& descriptors2,
        int image_width2,
        int image_height2,
        FeatureMatches* matches);

// Upstream parity factories; the ALIKED branch of
// FeatureExtractor/FeatureMatcher::Create dispatches to these.
std::unique_ptr<FeatureExtractor> CreateAlikedFeatureExtractor(
        const FeatureExtractionOptions& options);

std::unique_ptr<FeatureMatcher> CreateAlikedFeatureMatcher(
        const FeatureMatchingOptions& options);

}  // namespace colmap
