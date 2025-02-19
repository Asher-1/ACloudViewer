// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 asher-1.github.io
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
//

#include "core/nns/FixedRadiusSearchImpl.h"
#include "core/nns/NeighborSearchCommon.h"
#include "ml/pytorch/TorchHelper.h"
#include "ml/pytorch/misc/NeighborSearchAllocator.h"
#include "torch/script.h"

using namespace cloudViewer::core::nns;

template <class T, class TIndex>
void FixedRadiusSearchCPU(const torch::Tensor& points,
                          const torch::Tensor& queries,
                          double radius,
                          const torch::Tensor& points_row_splits,
                          const torch::Tensor& queries_row_splits,
                          const torch::Tensor& hash_table_splits,
                          const torch::Tensor& hash_table_index,
                          const torch::Tensor& hash_table_cell_splits,
                          const Metric metric,
                          const bool ignore_query_point,
                          const bool return_distances,
                          torch::Tensor& neighbors_index,
                          torch::Tensor& neighbors_row_splits,
                          torch::Tensor& neighbors_distance) {
    NeighborSearchAllocator<T, TIndex> output_allocator(
            points.device().type(), points.device().index());

    impl::FixedRadiusSearchCPU<T, TIndex>(
            neighbors_row_splits.data_ptr<int64_t>(), points.size(0),
            points.data_ptr<T>(), queries.size(0), queries.data_ptr<T>(),
            T(radius), points_row_splits.size(0),
            points_row_splits.data_ptr<int64_t>(), queries_row_splits.size(0),
            queries_row_splits.data_ptr<int64_t>(),
            (uint32_t*)hash_table_splits.data_ptr<int32_t>(),
            hash_table_cell_splits.size(0),
            (uint32_t*)hash_table_cell_splits.data_ptr<int32_t>(),
            (uint32_t*)hash_table_index.data_ptr<int32_t>(), metric,
            ignore_query_point, return_distances, output_allocator);

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

#define INSTANTIATE(T, TIndex)                                                \
    template void FixedRadiusSearchCPU<T, TIndex>(                            \
            const torch::Tensor& points, const torch::Tensor& queries,        \
            double radius, const torch::Tensor& points_row_splits,            \
            const torch::Tensor& queries_row_splits,                          \
            const torch::Tensor& hash_table_splits,                           \
            const torch::Tensor& hash_table_index,                            \
            const torch::Tensor& hash_table_cell_splits, const Metric metric, \
            const bool ignore_query_point, const bool return_distances,       \
            torch::Tensor& neighbors_index,                                   \
            torch::Tensor& neighbors_row_splits,                              \
            torch::Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)
