// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <vector>

#include "util/logging.h"

namespace colmap {

// Abstract base class for sampling methods.
class Sampler {
public:
    Sampler() {};
    explicit Sampler(const size_t num_samples);

    // Initialize the sampler, before calling the `Sample` method.
    virtual void Initialize(const size_t total_num_samples) = 0;

    // Maximum number of unique samples that can be generated.
    virtual size_t MaxNumSamples() = 0;

    // Sample `num_samples` elements from all samples.
    virtual std::vector<size_t> Sample() = 0;

    // Sample elements from `X` into `X_rand`.
    //
    // Note that `X.size()` should equal `num_total_samples` and `X_rand.size()`
    // should equal `num_samples`.
    template <typename X_t>
    void SampleX(const X_t& X, X_t* X_rand);

    // Sample elements from `X` and `Y` into `X_rand` and `Y_rand`.
    //
    // Note that `X.size()` should equal `num_total_samples` and `X_rand.size()`
    // should equal `num_samples`. The same applies for `Y` and `Y_rand`.
    template <typename X_t, typename Y_t>
    void SampleXY(const X_t& X, const Y_t& Y, X_t* X_rand, Y_t* Y_rand);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename X_t>
void Sampler::SampleX(const X_t& X, X_t* X_rand) {
    const auto sample_idxs = Sample();
    for (size_t i = 0; i < X_rand->size(); ++i) {
        (*X_rand)[i] = X[sample_idxs[i]];
    }
}

template <typename X_t, typename Y_t>
void Sampler::SampleXY(const X_t& X, const Y_t& Y, X_t* X_rand, Y_t* Y_rand) {
    CHECK_EQ(X.size(), Y.size());
    CHECK_EQ(X_rand->size(), Y_rand->size());
    const auto sample_idxs = Sample();
    for (size_t i = 0; i < X_rand->size(); ++i) {
        (*X_rand)[i] = X[sample_idxs[i]];
        (*Y_rand)[i] = Y[sample_idxs[i]];
    }
}

}  // namespace colmap
