// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "optim/sampler.h"

namespace colmap {

// Random sampler for RANSAC-based methods.
//
// Note that a separate sampler should be instantiated per thread.
class RandomSampler : public Sampler {
public:
    explicit RandomSampler(const size_t num_samples);

    void Initialize(const size_t total_num_samples) override;

    size_t MaxNumSamples() override;

    std::vector<size_t> Sample() override;

private:
    const size_t num_samples_;
    std::vector<size_t> sample_idxs_;
};

}  // namespace colmap
