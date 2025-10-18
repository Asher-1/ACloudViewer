// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

namespace cloudViewer {
namespace ml {
namespace contrib {

#ifdef BUILD_CUDA_MODULE

void roipool3dLauncher(int batch_size,
                       int pts_num,
                       int boxes_num,
                       int feature_in_len,
                       int sampled_pts_num,
                       const float *xyz,
                       const float *boxes3d,
                       const float *pts_feature,
                       float *pooled_features,
                       int *pooled_empty_flag);
#endif

}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
