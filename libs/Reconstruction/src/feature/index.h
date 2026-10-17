// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "feature/types.h"
#include "util/types.h"

namespace colmap {

class FeatureDescriptorIndex {
public:
    enum class Type {
        DEFAULT = 1,
        FAISS = 1,
    };

    virtual ~FeatureDescriptorIndex() = default;

    static std::unique_ptr<FeatureDescriptorIndex> Create(
            Type type = Type::DEFAULT, int num_threads = 1);

    // Upstream parity (a2bd15c2 feature/index.h): the index is typed; Build
    // records the descriptor type and Search rejects a mismatching query.
    virtual void Build(const FeatureDescriptorsFloat& descriptors) = 0;

    virtual void Search(int num_neighbors,
                        const FeatureDescriptorsFloat& query_descriptors,
                        Eigen::RowMajorMatrixXi& indices,
                        Eigen::RowMajorMatrixXf& l2_dists) const = 0;
};

}  // namespace colmap
