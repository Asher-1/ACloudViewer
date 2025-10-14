// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_OPTIM_COMBINATION_SAMPLER_H_
#define COLMAP_SRC_OPTIM_COMBINATION_SAMPLER_H_

#include "optim/sampler.h"

namespace colmap {

// Random sampler for RANSAC-based methods that generates unique samples.
//
// Note that a separate sampler should be instantiated per thread and it assumes
// that the input data is shuffled in advance.
class CombinationSampler : public Sampler {
public:
    explicit CombinationSampler(const size_t num_samples);

    void Initialize(const size_t total_num_samples) override;

    size_t MaxNumSamples() override;

    std::vector<size_t> Sample() override;

private:
    const size_t num_samples_;
    std::vector<size_t> total_sample_idxs_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_COMBINATION_SAMPLER_H_
