// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace colmap {

// Upstream parity (d3ccaf35 feature/onnx_matchers.h, renamed here to
// learned_matchers.h because the fork has no onnxruntime): the generic
// matcher option contracts. The fork records the data contracts only - the
// execution backend is the AICore ggml task family (see feature/aliked.h).
struct BruteForceGGMLMatchingOptions {
    // The minimum cosine similarity for a match to be considered valid.
    double min_cossim = 0.85;

    // Maximum ratio for Lowe's ratio test (second-best / best distance).
    double max_ratio = 1.0;

    // Enable cross-checking (mutual nearest neighbor).
    bool cross_check = true;

    // The path to the model file for the brute-force matcher.
    std::string model_path;

    bool Check() const { return min_cossim >= 0.0 && max_ratio > 0.0; }
};

struct LightGlueGGMLMatchingOptions {
    // Minimum match score threshold (post-model filtering).
    double min_score = 0.1;

    // Path to the LightGlue model file.
    std::string model_path;

    bool Check() const { return min_score >= 0.0; }
};

}  // namespace colmap
