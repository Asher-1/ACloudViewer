
#include "BuildSpatialHashTableOpKernel.h"

#include "core/nns/FixedRadiusSearchImpl.h"

using namespace tensorflow;

template <class T>
class BuildSpatialHashTableOpKernelCPU : public BuildSpatialHashTableOpKernel {
public:
    explicit BuildSpatialHashTableOpKernelCPU(
            OpKernelConstruction* construction)
        : BuildSpatialHashTableOpKernel(construction) {}

    void Kernel(tensorflow::OpKernelContext* context,
                const tensorflow::Tensor& points,
                const tensorflow::Tensor& radius,
                const tensorflow::Tensor& points_row_splits,
                const std::vector<uint32_t>& hash_table_splits,
                tensorflow::Tensor& hash_table_index,
                tensorflow::Tensor& hash_table_cell_splits) {
        cloudViewer::core::nns::impl::BuildSpatialHashTableCPU(
                points.shape().dim_size(0), points.flat<T>().data(),
                radius.scalar<T>()(), points_row_splits.shape().dim_size(0),
                (int64_t*)points_row_splits.flat<int64>().data(),
                hash_table_splits.data(),
                hash_table_cell_splits.shape().dim_size(0),
                hash_table_cell_splits.flat<uint32_t>().data(),
                hash_table_index.flat<uint32_t>().data());
    }
};

#define REG_KB(type)                                            \
    REGISTER_KERNEL_BUILDER(Name("CloudViewerBuildSpatialHashTable") \
                                    .Device(DEVICE_CPU)         \
                                    .TypeConstraint<type>("T"), \
                            BuildSpatialHashTableOpKernelCPU<type>);
REG_KB(float)
REG_KB(double)
#undef REG_KB
