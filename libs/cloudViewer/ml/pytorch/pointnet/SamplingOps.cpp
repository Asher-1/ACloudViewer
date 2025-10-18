// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "ml/pytorch/TorchHelper.h"
#include "ml/pytorch/pointnet/SamplingKernel.h"
#include "torch/script.h"

#ifdef BUILD_CUDA_MODULE

torch::Tensor furthest_point_sampling(torch::Tensor points,
                                      const int64_t sample_size) {
    int batch_size = points.size(0);
    int pts_size = points.size(1);

    auto device = points.device();
    torch::Tensor out =
            torch::zeros({batch_size, sample_size},
                         torch::dtype(ToTorchDtype<int>()).device(device));
    torch::Tensor temp =
            torch::full({batch_size, pts_size}, 1e10,
                        torch::dtype(ToTorchDtype<float>()).device(device));

    const float *points_data = points.data_ptr<float>();
    float *temp_data = temp.data_ptr<float>();
    int *out_data = out.data_ptr<int>();

    furthest_point_sampling_launcher(batch_size, pts_size, sample_size,
                                     points_data, temp_data, out_data);

    return out;
}

static auto registry_fp = torch::RegisterOperators(
        "cloudViewer::furthest_point_sampling(Tensor points, int sample_siz)"
        " -> Tensor out",
        &furthest_point_sampling);
#endif
