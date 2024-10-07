// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ml/Helper.h"
#include "ml/contrib/Nms.h"
#include "ml/tensorflow/misc/NmsOpKernel.h"

using namespace nms_opkernel;
using namespace tensorflow;

class NmsOpKernelCUDA : public NmsOpKernel {
public:
    explicit NmsOpKernelCUDA(OpKernelConstruction* construction)
        : NmsOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& boxes,
                const tensorflow::Tensor& scores) {
        std::vector<int64_t> keep_indices = cloudViewer::ml::contrib::NmsCUDAKernel(
                boxes.flat<float>().data(), scores.flat<float>().data(),
                boxes.dim_size(0), this->nms_overlap_thresh);

        OutputAllocator output_allocator(context);
        int64_t* ret_keep_indices = nullptr;
        output_allocator.AllocKeepIndices(&ret_keep_indices,
                                          keep_indices.size());
        OPEN3D_CUDA_CHECK(cudaMemcpy(ret_keep_indices, keep_indices.data(),
                                     keep_indices.size() * sizeof(int64_t),
                                     cudaMemcpyHostToDevice));
    }
};

#define REG_KB(type)                                                        \
    REGISTER_KERNEL_BUILDER(                                                \
            Name("CloudViewerNms").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
            NmsOpKernelCUDA);
REG_KB(float)
#undef REG_KB
