// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "ml/contrib/Nms.h"
#include "ml/pytorch/TorchHelper.h"
#include "torch/script.h"

torch::Tensor Nms(torch::Tensor boxes,
                  torch::Tensor scores,
                  double nms_overlap_thresh) {
    boxes = boxes.contiguous();
    CHECK_TYPE(boxes, kFloat);
    CHECK_TYPE(scores, kFloat);

    if (boxes.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        std::vector<int64_t> keep_indices =
                cloudViewer::ml::contrib::NmsCUDAKernel(
                        boxes.data_ptr<float>(), scores.data_ptr<float>(),
                        boxes.size(0), nms_overlap_thresh);
        return torch::from_blob(keep_indices.data(),
                                {static_cast<int64_t>(keep_indices.size())},
                                torch::TensorOptions().dtype(torch::kLong))
                .to(boxes.device());
#else
        TORCH_CHECK(false, "Nms was not compiled with CUDA support")

#endif
    } else {
        std::vector<int64_t> keep_indices =
                cloudViewer::ml::contrib::NmsCPUKernel(
                        boxes.data_ptr<float>(), scores.data_ptr<float>(),
                        boxes.size(0), nms_overlap_thresh);
        return torch::from_blob(keep_indices.data(),
                                {static_cast<int64_t>(keep_indices.size())},
                                torch::TensorOptions().dtype(torch::kLong))
                .clone();
    }
}

static auto registry = torch::RegisterOperators(
        "cloudViewer::nms(Tensor boxes, Tensor scores, float "
        "nms_overlap_thresh) -> "
        "Tensor keep_indices",
        &Nms);
