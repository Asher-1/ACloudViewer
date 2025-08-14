// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/nns/NNSIndex.h"

namespace cloudViewer {
namespace core {
namespace nns {

int NNSIndex::GetDimension() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<int>(shape[1]);
}

size_t NNSIndex::GetDatasetSize() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<size_t>(shape[0]);
}

Dtype NNSIndex::GetDtype() const { return dataset_points_.GetDtype(); }

Device NNSIndex::GetDevice() const { return dataset_points_.GetDevice(); }

Dtype NNSIndex::GetIndexDtype() const { return index_dtype_; }

}  // namespace nns
}  // namespace core
}  // namespace cloudViewer
