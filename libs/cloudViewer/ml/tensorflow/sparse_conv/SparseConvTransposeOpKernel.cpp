// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "SparseConvTransposeOpKernel.h"

#include "ml/impl/sparse_conv/SparseConvTranspose.h"

using namespace cloudViewer;
using namespace cloudViewer::ml::impl;
using namespace tensorflow;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
class SparseConvTransposeOpKernelCPU
    : public SparseConvTransposeOpKernel<TIndex> {
public:
    explicit SparseConvTransposeOpKernelCPU(OpKernelConstruction* construction)
        : SparseConvTransposeOpKernel<TIndex>(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& filter,
                const tensorflow::Tensor& out_importance,
                const tensorflow::Tensor& inp_features,
                const tensorflow::Tensor& inp_neighbors_importance_sum,
                const tensorflow::Tensor& inp_neighbors_row_splits,
                const tensorflow::Tensor& neighbors_index,
                const tensorflow::Tensor& neighbors_kernel_index,
                const tensorflow::Tensor& neighbors_importance,
                const tensorflow::Tensor& neighbors_row_splits,
                const std::vector<int>& filter_dims,
                const bool point_importances,
                const bool has_neighbors_importances,
                tensorflow::Tensor& out_features) {
        SparseConvTransposeComputeFeaturesCPU<TFeat, TOut, TIndex,
                                              TKernelIndex>(
                out_features.flat<TOut>().data(), filter_dims,
                filter.flat<TFeat>().data(),
                neighbors_row_splits.shape().dim_size(0) - 1,
                point_importances ? out_importance.flat<TFeat>().data()
                                  : nullptr,
                inp_features.shape().dim_size(0),
                inp_features.flat<TFeat>().data(),
                has_neighbors_importances
                        ? inp_neighbors_importance_sum.flat<TFeat>().data()
                        : nullptr,
                (int64_t*)inp_neighbors_row_splits.flat<int64>().data(),
                (TIndex*)neighbors_index.flat<TIndex>().data(),
                (TKernelIndex*)neighbors_kernel_index.flat<TKernelIndex>()
                        .data(),
                has_neighbors_importances
                        ? neighbors_importance.flat<TFeat>().data()
                        : nullptr,
                (int64_t*)neighbors_row_splits.flat<int64>().data(),
                this->normalize);
    }
};

#define REG_KB(feattype, outtype, indextype, kernelindextype)            \
    REGISTER_KERNEL_BUILDER(                                             \
            Name("CloudviewerSparseConvTranspose")                       \
                    .Device(DEVICE_CPU)                                  \
                    .TypeConstraint<feattype>("TFeat")                   \
                    .TypeConstraint<outtype>("output_type")              \
                    .TypeConstraint<indextype>("TIndex")                 \
                    .TypeConstraint<kernelindextype>("TKernelIndex"),    \
            SparseConvTransposeOpKernelCPU<feattype, outtype, indextype, \
                                           kernelindextype>);
REG_KB(float, float, int32, int16_t)
REG_KB(float, float, int32, uint8_t)
REG_KB(bfloat16, float, int32, int16_t)
REG_KB(bfloat16, float, int32, uint8_t)
REG_KB(bfloat16, bfloat16, int32, int16_t)
REG_KB(bfloat16, bfloat16, int32, uint8_t)
#undef REG_KB
