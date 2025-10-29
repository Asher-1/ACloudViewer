// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "ml/pytorch/TorchHelper.h"
#include "ml/pytorch/pointnet/BallQueryKernel.h"
#include "torch/script.h"

#ifdef BUILD_CUDA_MODULE
torch::Tensor ball_query(torch::Tensor xyz,
                         torch::Tensor center,
                         double radius,
                         const int64_t nsample) {
    int batch_size = xyz.size(0);
    int pts_num = xyz.size(1);
    int ball_num = center.size(1);

    auto device = xyz.device();
    torch::Tensor out =
            torch::zeros({batch_size, ball_num, nsample},
                         torch::dtype(ToTorchDtype<int>()).device(device));

    const float *center_data = center.data_ptr<float>();
    const float *xyz_data = xyz.data_ptr<float>();
    int *idx = out.data_ptr<int>();

    ball_query_launcher(batch_size, pts_num, ball_num, radius, nsample,
                        center_data, xyz_data, idx);
    return out;
}

static auto registry = torch::RegisterOperators(
        "cloudViewer::ball_query(Tensor xyz, Tensor center,"
        "float radius, int nsample)"
        " -> Tensor out",
        &ball_query);
#endif
