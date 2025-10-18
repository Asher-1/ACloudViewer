// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#define EIGEN_USE_GPU
#include "BuildSpatialHashTableOpKernel.h"
#include "cloudViewer/core/CUDAUtils.h"
#include "core/nns/FixedRadiusSearchImpl.cuh"

using namespace cloudViewer;
using namespace tensorflow;

template <class T>
class BuildSpatialHashTableOpKernelCUDA : public BuildSpatialHashTableOpKernel {
public:
    explicit BuildSpatialHashTableOpKernelCUDA(
            OpKernelConstruction* construction)
        : BuildSpatialHashTableOpKernel(construction) {
        texture_alignment =
                cloudViewer::core::GetCUDACurrentDeviceTextureAlignment();
    }

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& points,
                const tensorflow::Tensor& radius,
                const tensorflow::Tensor& points_row_splits,
                const std::vector<uint32_t>& hash_table_splits,
                tensorflow::Tensor& hash_table_index,
                tensorflow::Tensor& hash_table_cell_splits) {
        auto device = context->eigen_gpu_device();

        void* temp_ptr = nullptr;
        size_t temp_size = 0;

        // determine temp_size
        cloudViewer::core::nns::impl::BuildSpatialHashTableCUDA(
                device.stream(), temp_ptr, temp_size, texture_alignment,
                points.shape().dim_size(0), points.flat<T>().data(),
                radius.scalar<T>()(), points_row_splits.shape().dim_size(0),
                (int64_t*)points_row_splits.flat<int64>().data(),
                hash_table_splits.data(),
                hash_table_cell_splits.shape().dim_size(0),
                hash_table_cell_splits.flat<uint32_t>().data(),
                hash_table_index.flat<uint32_t>().data());

        Tensor temp_tensor;
        TensorShape temp_shape({ssize_t(temp_size)});
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<uint8_t>::v(),
                                              temp_shape, &temp_tensor));
        temp_ptr = temp_tensor.flat<uint8_t>().data();

        // actually build the table
        cloudViewer::core::nns::impl::BuildSpatialHashTableCUDA(
                device.stream(), temp_ptr, temp_size, texture_alignment,
                points.shape().dim_size(0), points.flat<T>().data(),
                radius.scalar<T>()(), points_row_splits.shape().dim_size(0),
                (int64_t*)points_row_splits.flat<int64>().data(),
                hash_table_splits.data(),
                hash_table_cell_splits.shape().dim_size(0),
                hash_table_cell_splits.flat<uint32_t>().data(),
                hash_table_index.flat<uint32_t>().data());
    }

private:
    int texture_alignment;
};

#define REG_KB(type)                                                       \
    REGISTER_KERNEL_BUILDER(Name("CloudViewerBuildSpatialHashTable")       \
                                    .Device(DEVICE_GPU)                    \
                                    .TypeConstraint<type>("T")             \
                                    .HostMemory("radius")                  \
                                    .HostMemory("points_row_splits")       \
                                    .HostMemory("hash_table_splits")       \
                                    .HostMemory("hash_table_size_factor"), \
                            BuildSpatialHashTableOpKernelCUDA<type>);
REG_KB(float)
#undef REG_KB
