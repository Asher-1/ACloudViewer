// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#pragma once

#include <vector>

#include "core/Tensor.h"
#ifdef WITH_FAISS
#include "core/nns/FaissIndex.h"
#endif
#include "core/nns/NanoFlannIndex.h"

namespace cloudViewer {
namespace core {
namespace nns {

/// \class NearestNeighborSearch
///
/// \brief A Class for nearest neighbor search.
class NearestNeighborSearch {
public:
    /// Constructor.
    ///
    /// \param dataset_points Dataset points for constructing search index. Must
    /// be 2D, with shape {n, d}.
    NearestNeighborSearch(const Tensor &dataset_points)
        : dataset_points_(dataset_points){}

    ~NearestNeighborSearch();
    NearestNeighborSearch(const NearestNeighborSearch &) = delete;
    NearestNeighborSearch &operator=(const NearestNeighborSearch &) = delete;

public:
    /// Set index for knn search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool KnnIndex();

    /// Set index for multi-radius search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool MultiRadiusIndex();

    /// Set index for fixed-radius search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool FixedRadiusIndex();

    /// Set index for hybrid search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool HybridIndex();

    /// Perform knn search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param knn Number of neighbors to search per query point.
    /// \return Pair of Tensors, (indices, distances):
    /// - indices: Tensor of shape {n, knn}, with dtype Int64.
    /// - distainces: Tensor of shape {n, knn}, same dtype with query_points.
    std::pair<Tensor, Tensor> KnnSearch(const Tensor &query_points, int knn);

    /// Perform fixed radius search. All query points share the same radius.
    ///
    /// \param query_points Data points for querying. Must be 2D, with shape {n,
    /// d}.
    /// \param radius Radius.
    /// \return Tuple of Tensors, (indices, distances, num_neighbors):
    /// - indicecs: Tensor of shape {total_number_of_neighbors,}, with dtype
    /// Int64.
    /// - distances: Tensor of shape {total_number_of_neighbors,}, same dtype
    /// with query_points.
    /// - num_neighbors: Tensor of shape {n,}, with dtype Int64.
    std::tuple<Tensor, Tensor, Tensor> FixedRadiusSearch(
            const Tensor &query_points, double radius);

    /// Perform multi-radius search. Each query point has an independent radius.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param radii Radii of query points. Each query point has one radius.
    /// Must be 1D, with shape {n,}.
    /// \return Tuple of Tensors, (indices,distances, num_neighbors):
    /// - indicecs: Tensor of shape {total_number_of_neighbors,}, with dtype
    /// Int64.
    /// - distances: Tensor of shape {total_number_of_neighbors,}, same dtype
    /// with query_points.
    /// - num_neighbors: Tensor of shape {n,}, with dtype Int64.
    std::tuple<Tensor, Tensor, Tensor> MultiRadiusSearch(
            const Tensor &query_points, const Tensor &radii);

    /// Perform hybrid search.
    ///
    /// \param query_points Data points for querying. Must be 2D, with shape {n,
    /// d}.
    /// \param radius Radius.
    /// \param max_knn Maximum number of neighbor to search per query.
    /// \return Pair of Tensors, (indices, distances):
    /// - indices: Tensor of shape {n, knn}, with dtype Int64.
    /// - distainces: Tensor of shape {n, knn}, with same dtype with
    /// query_points.
    std::pair<Tensor, Tensor> HybridSearch(const Tensor &query_points,
                                           double radius,
                                           int max_knn);

private:
    bool SetIndex();

    /// Assert a Tensor is not CUDA tensoer. This will be removed in the future.
    void AssertNotCUDA(const Tensor &t) const;

protected:
    std::unique_ptr<NanoFlannIndex> nanoflann_index_;
#ifdef WITH_FAISS
    std::unique_ptr<FaissIndex> faiss_index_;
#endif
    const Tensor dataset_points_;
};

}  // namespace nns
}  // namespace core
}  // namespace cloudViewer
