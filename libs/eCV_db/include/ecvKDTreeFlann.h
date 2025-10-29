// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>
#include <nanoflann.hpp>
#include <vector>

#include "ecvFeature.h"
#include "ecvHObject.h"
#include "ecvKDTreeSearchParam.h"

/// @cond
namespace nanoflann {
struct metric_L2;
template <class MatrixType, int DIM, class Distance, bool row_major>
struct KDTreeEigenMatrixAdaptor;
}  // namespace nanoflann
/// @endcond

namespace cloudViewer {
namespace geometry {

/// \class KDTreeFlann
///
/// \brief KDTree with FLANN for nearest neighbor search.
class ECV_DB_LIB_API KDTreeFlann {
public:
    /// \brief Default Constructor.
    KDTreeFlann();
    /// \brief Parameterized Constructor.
    ///
    /// \param data Provides set of data points for KDTree construction.
    KDTreeFlann(const Eigen::MatrixXd &data);
    /// \brief Parameterized Constructor.
    ///
    /// \param geometry Provides geometry from which KDTree is constructed.
    KDTreeFlann(const ccHObject &geometry);
    /// \brief Parameterized Constructor.
    ///
    /// \param feature Provides a set of features from which the KDTree is
    /// constructed.
    KDTreeFlann(const utility::Feature &feature);
    ~KDTreeFlann();
    KDTreeFlann(const KDTreeFlann &) = delete;
    KDTreeFlann &operator=(const KDTreeFlann &) = delete;

public:
    /// Sets the data for the KDTree from a matrix.
    ///
    /// \param data Data points for KDTree Construction.
    bool SetMatrixData(const Eigen::MatrixXd &data);
    /// Sets the data for the KDTree from geometry.
    ///
    /// \param geometry Geometry for KDTree Construction.
    bool SetGeometry(const ccHObject &geometry);
    /// Sets the data for the KDTree from the feature data.
    ///
    /// \param feature Set of features for KDTree construction.
    bool SetFeature(const utility::Feature &feature);

    template <typename T>
    int Search(const T &query,
               const KDTreeSearchParam &param,
               std::vector<int> &indices,
               std::vector<double> &distance2) const {
        switch (param.GetSearchType()) {
            case KDTreeSearchParam::SearchType::Knn:
                return SearchKNN(query,
                                 ((const KDTreeSearchParamKNN &)param).knn_,
                                 indices, distance2);
            case KDTreeSearchParam::SearchType::Radius:
                return SearchRadius(
                        query, ((const KDTreeSearchParamRadius &)param).radius_,
                        indices, distance2);
            case KDTreeSearchParam::SearchType::Hybrid:
                return SearchHybrid(
                        query, ((const KDTreeSearchParamHybrid &)param).radius_,
                        ((const KDTreeSearchParamHybrid &)param).max_nn_,
                        indices, distance2);
            default:
                return -1;
        }
        return -1;
    }

    template <typename T>
    int Query(const std::vector<T> &queries,
              const KDTreeSearchParam &param,
              std::vector<std::vector<int>> &indices,
              std::vector<std::vector<double>> &distance2) const {
        // precompute all neighbours with given queries
        indices.resize(queries.size());
        distance2.resize(queries.size());
        int flag = 1;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int idx = 0; idx < int(queries.size()); ++idx) {
            int k = Search(queries[idx], param, indices[idx], distance2[idx]);
            if (k < 0) {
                flag = -1;
            }
        }

        if (flag < 0) {
            utility::LogWarning("[Query] some queries failed!");
        }

        return flag;
    }

    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  std::vector<int> &indices,
                  std::vector<double> &distance2) const {
        // This is optimized code for heavily repeated search.
        // Other flann::Index::knnSearch() implementations lose performance due
        // to memory allocation/deallocation.
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_ || knn < 0) {
            return -1;
        }
        indices.resize(knn);
        distance2.resize(knn);
        std::vector<Eigen::Index> indices_eigen(knn);
        int k = nanoflann_index_->index_->knnSearch(
                query.data(), knn, indices_eigen.data(), distance2.data());
        indices.resize(k);
        distance2.resize(k);
        std::copy_n(indices_eigen.begin(), k, indices.begin());
        return k;
    }

    template <typename T>
    int SearchRadius(const T &query,
                     double radius,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const {
        // This is optimized code for heavily repeated search.
        // Since max_nn is not given, we let flann to do its own memory
        // management. Other flann::Index::radiusSearch() implementations lose
        // performance due to memory management and CPU caching.
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_) {
            return -1;
        }
        std::vector<nanoflann::ResultItem<Eigen::Index, double>> indices_dists;
        int k = nanoflann_index_->index_->radiusSearch(
                query.data(), radius * radius, indices_dists,
                nanoflann::SearchParameters(0.0));
        indices.resize(k);
        distance2.resize(k);
        for (int i = 0; i < k; ++i) {
            indices[i] = indices_dists[i].first;
            distance2[i] = indices_dists[i].second;
        }
        return k;
    }

    template <typename T>
    int SearchHybrid(const T &query,
                     double radius,
                     int max_nn,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const {
        // This is optimized code for heavily repeated search.
        // It is also the recommended setting for search.
        // Other flann::Index::radiusSearch() implementations lose performance
        // due to memory allocation/deallocation.
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_ || max_nn < 0) {
            return -1;
        }
        distance2.resize(max_nn);
        std::vector<Eigen::Index> indices_eigen(max_nn);
        int k = nanoflann_index_->index_->knnSearch(
                query.data(), max_nn, indices_eigen.data(), distance2.data());
        k = std::distance(
                distance2.begin(),
                std::lower_bound(distance2.begin(), distance2.begin() + k,
                                 radius * radius));
        indices.resize(k);
        distance2.resize(k);
        std::copy_n(indices_eigen.begin(), k, indices.begin());
        return k;
    }

private:
    /// \brief Sets the KDTree data from the data provided by the other methods.
    ///
    /// Internal method that sets all the members of KDTree by data provided by
    /// features, geometry, etc.
    bool SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data);

protected:
    using KDTree_t = nanoflann::KDTreeEigenMatrixAdaptor<
            Eigen::Map<const Eigen::MatrixXd>,
            -1,
            nanoflann::metric_L2,
            false>;

    std::vector<double> data_;
    std::unique_ptr<KDTree_t> nanoflann_index_;
    std::unique_ptr<Eigen::Map<const Eigen::MatrixXd>> data_interface_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

}  // namespace geometry
}  // namespace cloudViewer
