// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>

#include "feature/matcher.h"

namespace colmap {

// Upstream parity (feature/onnx_matchers.h, renamed here to
// learned_matchers.h because the fork has no onnxruntime): the generic
// brute-force matcher option contract shared by the learned descriptors
// (ALIKED, LoMa). Upstream runs a small ONNX graph; the fork evaluates the
// identical semantics (mutual nearest neighbour on cosine similarity +
// min_cossim threshold + Lowe ratio test + cross check) directly with Eigen,
// so no model round trip is needed and model_path is accepted for option
// compatibility only.
struct BruteForceGGMLMatchingOptions {
    // The minimum cosine similarity for a match to be considered valid.
    double min_cossim = 0.85;

    // Maximum ratio for Lowe's ratio test (best / second-best distance);
    // values >= 1 disable the test.
    double max_ratio = 1.0;

    // Enable cross-checking (mutual nearest neighbor).
    bool cross_check = true;

    // Unused by the Eigen evaluator; kept for option-surface parity.
    std::string model_path;

    bool Check() const;
};

// Descriptor-dimension agnostic brute-force matcher over float32 descriptors
// that travel through FeatureDescriptors as packed bytes (feature/types.h).
// When `normalize_descriptors` is set, rows are L2-normalized before the
// similarity is evaluated; extractors that already emit unit-norm rows pass
// false (ALIKED), others pass true (LoMa).
std::unique_ptr<FeatureMatcher> CreateBruteForceGGMLFeatureMatcher(
        const FeatureMatchingOptions& options,
        const BruteForceGGMLMatchingOptions& brute_force_options,
        bool normalize_descriptors);

}  // namespace colmap
