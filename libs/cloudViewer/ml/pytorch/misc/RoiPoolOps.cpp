// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ml/contrib/RoiPoolKernel.h"
#include "ml/pytorch/TorchHelper.h"

#ifdef BUILD_CUDA_MODULE
std::tuple<torch::Tensor, torch::Tensor> roi_pool(
        torch::Tensor xyz,
        torch::Tensor boxes3d,
        torch::Tensor pts_feature,
        const int64_t sampled_pts_num) {
    int batch_size = xyz.size(0);
    int pts_num = xyz.size(1);
    int boxes_num = boxes3d.size(1);
    int feature_in_len = pts_feature.size(2);

    auto device = xyz.device();
    torch::Tensor features = torch::zeros(
            {batch_size, boxes_num, sampled_pts_num, 3 + feature_in_len},
            torch::dtype(ToTorchDtype<float>()).device(device));

    torch::Tensor empty_flag =
            torch::zeros({batch_size, boxes_num},
                         torch::dtype(ToTorchDtype<int>()).device(device));

    const float *xyz_data = xyz.data_ptr<float>();
    const float *boxes3d_data = boxes3d.data_ptr<float>();
    const float *pts_feature_data = pts_feature.data_ptr<float>();
    float *pooled_features_data = features.data_ptr<float>();
    int *pooled_empty_flag_data = empty_flag.data_ptr<int>();

    cloudViewer::ml::contrib::roipool3dLauncher(
            batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
            xyz_data, boxes3d_data, pts_feature_data, pooled_features_data,
            pooled_empty_flag_data);

    return std::tuple<torch::Tensor, torch::Tensor>(features, empty_flag);
}

static auto registry = torch::RegisterOperators(
        "cloudViewer::roi_pool(Tensor xyz, Tensor boxes3d,"
        "Tensor pts_feature, int sampled_pts_num)"
        " -> (Tensor features, Tensor flags)",
        &roi_pool);
#endif
