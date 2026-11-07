// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Upstream parity (feature/lightglue.h): the LightGlue matcher contract.
// Upstream executes through onnxruntime (LightGlueONNXMatchingOptions /
// CreateLightGlueONNXFeatureMatcher); the fork executes through the AICore
// ggml LightGlue task and resolves its default models from the shared AICore
// catalog (AICORE_MODEL_FAMILY_LIGHTGLUE).

#include <memory>
#include <string>

#include "feature/matcher.h"

namespace colmap {

// AICore catalog matcher kinds (aicore_lightglue_options_set_matcher_type).
enum class LightGlueFeatureKind : int {
    SIFT = 1,
    ALIKED = 2,
};

struct LightGlueGGMLMatchingOptions {
    // Minimum match score threshold (post-model filtering).
    double min_score = 0.1;

    // Path to the LightGlue GGUF model. Empty resolves to the shared-catalog
    // default for the matcher's feature kind at construction time; caller
    // provided paths are never downloaded or replaced.
    std::string model_path;

    // ggml engine placement (fork extension; upstream uses use_gpu/gpu_index).
    std::string device = "auto";

    bool Check() const;
};

// Shared-catalog default path for the LightGlue model of the given feature
// kind, or "" when unpublished. The optional quant selects "f32"/"f16"/"q8_0";
// empty picks the catalog default (first row).
std::string DefaultLightGlueModelPath(LightGlueFeatureKind kind,
                                      const char* quant = "");

// Fills in the catalog-default model path (no download).
void ResolveDefaultLightGlueModelPath(LightGlueFeatureKind kind,
                                      LightGlueGGMLMatchingOptions* options);

// Catalog-default paths are provisioned through the shared AICore cache with
// SHA-256 verification; non-default caller-provided paths are left untouched.
bool ProvisionDefaultLightGlueModel(LightGlueFeatureKind kind,
                                    LightGlueGGMLMatchingOptions* options);

// Creates a LightGlue matcher for SIFT_LIGHTGLUE or ALIKED_LIGHTGLUE. The
// matcher owns one ggml session for its whole lifetime. SIFT descriptors
// arrive as uint8 rows and are L2-normalized to recover the unit-norm RootSIFT
// vectors the model was trained on; ALIKED descriptors arrive as packed
// float32 bytes (see FeatureDescriptors reinterpretation in feature/types.h).
std::unique_ptr<FeatureMatcher> CreateLightGlueGGMLFeatureMatcher(
        const FeatureMatchingOptions& options,
        const LightGlueGGMLMatchingOptions& lightglue_options);

}  // namespace colmap
