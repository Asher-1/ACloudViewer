// ----------------------------------------------------------------------------
// -                        CVLib: www.erow.cn                            -
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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)
#endif

// CV_CORE_LIB
#include <Console.h>

// FLANN_LIB
#include <flann/flann.hpp>

// LOCAL
#include "ecvMesh.h"
#include "ecvHalfEdgeMesh.h"
#include "ecvKDTreeFlann.h"
#include "ecvPointCloud.h"
#include "ecvHObjectCaster.h"

namespace cloudViewer {
namespace geometry {

KDTreeFlann::KDTreeFlann(size_t leaf_size/* = 15*/,
						 bool reorder/* = true*/)
	: leaf_size_(leaf_size)
	, reorder_(reorder)
{}

KDTreeFlann::KDTreeFlann(const Eigen::MatrixXd &data,
						 size_t leaf_size/* = 15*/,
						 bool reorder/* = true*/)
	: leaf_size_(leaf_size)
	, reorder_(reorder)
{
	SetMatrixData(data); 
}

KDTreeFlann::KDTreeFlann(const ccHObject &geometry, 
						 size_t leaf_size/* = 15*/,
						 bool reorder/* = true*/) 
	: leaf_size_(leaf_size)
	, reorder_(reorder)
{
	SetGeometry(geometry);
}

KDTreeFlann::KDTreeFlann(const utility::Feature &feature,
						 size_t leaf_size/* = 15*/,
						 bool reorder/* = true*/)
	: leaf_size_(leaf_size)
	, reorder_(reorder)
{
    SetFeature(feature);
}

KDTreeFlann::~KDTreeFlann() {}

bool KDTreeFlann::SetMatrixData(const Eigen::MatrixXd &data) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            data.data(), data.rows(), data.cols()));
}

bool KDTreeFlann::SetGeometry(const ccHObject &geometry, bool use_eigen/* = true*/) {
    use_eigen_ = use_eigen;

    switch (geometry.getClassID()) {
	case CV_TYPES::POINT_CLOUD:
	{
		const ccPointCloud& cloud = static_cast<const ccPointCloud &>(geometry);
        if (use_eigen) {
            std::vector<Eigen::Vector3d> points = cloud.getEigenPoints();
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                (const double *)points.data(), 3, points.size()));
        } else {
            return SetRawData(cloud.getPoints());
        }
	}
    case CV_TYPES::MESH:
	{
		const ccMesh& mesh = static_cast<const ccMesh &>(geometry);
        if (use_eigen) {
            std::vector<Eigen::Vector3d> points = mesh.getEigenVertices();
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                    (const double *)points.data(), 3, points.size()));
        } else {
            return SetRawData(mesh.getVertices());
        }

	}
    case CV_TYPES::HALF_EDGE_MESH: {
        use_eigen_ = true; // only support eigen
        const ecvHalfEdgeMesh &mesh = static_cast<const ecvHalfEdgeMesh &>(geometry);
        return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                (const double *)mesh.vertices_.data(), 3, mesh.vertices_.size()));
    }
    case CV_TYPES::IMAGE:
	case CV_TYPES::HIERARCHY_OBJECT:
    default:
        CVLib::utility::LogWarning("[KDTreeFlann::SetGeometry] Unsupported Geometry type.");
        return false;
    }
}

bool KDTreeFlann::SetFeature(const utility::Feature &feature) {
    return SetMatrixData(feature.data_);
}

template <typename T>
int KDTreeFlann::Search(const T &query,
                        const KDTreeSearchParam &param,
                        std::vector<int> &indices,
                        std::vector<double> &distance2) const {
    switch (param.GetSearchType()) {
        case KDTreeSearchParam::SearchType::Knn:
            return SearchKNN(query, ((const KDTreeSearchParamKNN &)param).knn_,
                             indices, distance2);
        case KDTreeSearchParam::SearchType::Radius:
            return SearchRadius(
                    query, ((const KDTreeSearchParamRadius &)param).radius_,
                    indices, distance2);
        case KDTreeSearchParam::SearchType::Hybrid:
            return SearchHybrid(
                    query, ((const KDTreeSearchParamHybrid &)param).radius_,
                    ((const KDTreeSearchParamHybrid &)param).max_nn_, indices,
                    distance2);
        default:
            return -1;
    }
    return -1;
}

template <typename T>
int KDTreeFlann::Query(const std::vector<T> &queries,
	const KDTreeSearchParam &param,
	std::vector < std::vector<int> > & indices,
	std::vector < std::vector<double> > & distance2) const {

	// precompute all neighbours with given queries
	indices.resize(queries.size());
	distance2.resize(queries.size());
	int flag = 1;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int idx = 0; idx < int(queries.size()); ++idx) {
		int k = Search(queries[idx], param, indices[idx], distance2[idx]);
		if (k < 0)
		{
			flag = -1;
		}
	}

	if (flag < 0)
	{
		CVLib::utility::LogWarning("[KDTreeFlann::Query] some queries failed!");
	}

	return flag;
}


template <typename T>
int KDTreeFlann::SearchKNN(const T &query,
                           int knn,
                           std::vector<int> &indices,
                           std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Other flann::Index::knnSearch() implementations lose performance due to
    // memory allocation/deallocation.

    std::size_t KNN = static_cast<std::size_t>(knn);

    if (!dataf_.empty()) // fast-float
    {
        if (dataset_size_ <= 0 || knn < 0) {
            return -1;
        }

        flann::Matrix<PointCoordinateType> query_flann((PointCoordinateType *)query.data(), 1, dimension_);
        indices.resize(KNN);
        flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, KNN);
        std::vector<PointCoordinateType> tempDis(KNN);
        flann::Matrix<PointCoordinateType> dists_flann(tempDis.data(), query_flann.rows, KNN);
        int k = flann_indexf_->knnSearch(query_flann, indices_flann, dists_flann,
                                        KNN, flann::SearchParams(-1, 0.0));
        indices.resize(k);
        tempDis.resize(k);
        distance2 = std::vector<double>(tempDis.begin(), tempDis.end());
        return k;

    }
    else // Eigen(double)
    {
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_ || knn < 0) {
            return -1;
        }
        flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
        indices.resize(KNN);
        distance2.resize(KNN);
        flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, KNN);
        flann::Matrix<double> dists_flann(distance2.data(), query_flann.rows, KNN);
        int k = flann_index_->knnSearch(query_flann, indices_flann, dists_flann,
                                        KNN, flann::SearchParams(-1, 0.0));
        indices.resize(k);
        distance2.resize(k);
        return k;
    }
}

template <typename T>
int KDTreeFlann::SearchRadius(const T &query,
                              double radius,
                              std::vector<int> &indices,
                              std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Since max_nn is not given, we let flann to do its own memory management.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory management and CPU caching.

    if (!dataf_.empty()) // fast-float
    {
        if (dataset_size_ <= 0) {
            return -1;
        }

        flann::Matrix<PointCoordinateType> query_flann((PointCoordinateType *)query.data(), 1, dimension_);
        flann::SearchParams param(-1, 0.0);
        param.max_neighbors = -1;
        std::vector<std::vector<int>> indices_vec(1);
        std::vector<std::vector<PointCoordinateType>> dists_vec(1);
        int k = flann_indexf_->radiusSearch(query_flann, indices_vec, dists_vec,
                                           float(radius * radius), param);
        indices = indices_vec[0];
        distance2 = std::vector<double>(dists_vec[0].begin(), dists_vec[0].end());
        return k;
    }
    else // Eigen(double)
    {
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_) {
            return -1;
        }

        flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
        flann::SearchParams param(-1, 0.0);
        param.max_neighbors = -1;
        std::vector<std::vector<int>> indices_vec(1);
        std::vector<std::vector<double>> dists_vec(1);
        int k = flann_index_->radiusSearch(query_flann, indices_vec, dists_vec,
                                           float(radius * radius), param);
        indices = indices_vec[0];
        distance2 = dists_vec[0];
        return k;
    }

}

template <typename T>
int KDTreeFlann::SearchHybrid(const T &query,
                              double radius,
                              int max_nn,
                              std::vector<int> &indices,
                              std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // It is also the recommended setting for search.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory allocation/deallocation.

    std::size_t KNN = static_cast<std::size_t>(max_nn);

    if (!dataf_.empty()) // fast-float
    {
        if (dataset_size_ <= 0 || dataset_size_ <= 0 || max_nn < 0) {
            return -1;
        }

        flann::Matrix<PointCoordinateType> query_flann((PointCoordinateType *)query.data(), 1, dimension_);
        flann::SearchParams param(-1, 0.0);
        param.max_neighbors = max_nn;
        indices.resize(KNN);
        flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, KNN);
        std::vector<PointCoordinateType> tempDis(KNN);
        flann::Matrix<PointCoordinateType> dists_flann(tempDis.data(), query_flann.rows, KNN);
        int k = flann_indexf_->radiusSearch(query_flann, indices_flann, dists_flann,
                                           float(radius * radius), param);
        indices.resize(k);
        tempDis.resize(k);
        distance2 = std::vector<double>(tempDis.begin(), tempDis.end());
        return k;
    }
    else // Eigen(double)
    {
        if (data_.empty() || dataset_size_ <= 0 ||
            size_t(query.rows()) != dimension_ || max_nn < 0) {
            return -1;
        }
        flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
        flann::SearchParams param(-1, 0.0);
        param.max_neighbors = max_nn;
        indices.resize(KNN);
        distance2.resize(KNN);
        flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, KNN);
        flann::Matrix<double> dists_flann(distance2.data(), query_flann.rows, KNN);
        int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                           float(radius * radius), param);
        indices.resize(k);
        distance2.resize(k);
        return k;
    }

}

bool KDTreeFlann::SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
    dimension_ = data.rows();
    dataset_size_ = data.cols();
    if (dimension_ == 0 || dataset_size_ == 0) {
		CVLib::utility::LogWarning("[KDTreeFlann::SetRawData] Failed due to no data.");
        return false;
    }
    data_.resize(dataset_size_ * dimension_);
    memcpy(data_.data(), data.data(),
           dataset_size_ * dimension_ * sizeof(double));
    flann_dataset_.reset(new flann::Matrix<double>((double *)data_.data(),
                                                   dataset_size_, dimension_));
    flann_index_.reset(new flann::Index<flann::L2<double>>(
            *flann_dataset_, flann::KDTreeSingleIndexParams(leaf_size_, reorder_)));
    flann_index_->buildIndex();
    return true;
}

bool KDTreeFlann::SetRawData(const std::vector<CCVector3> &data)
{
    std::vector<const CCVector3*> temp;
    for (std::size_t i = 0; i < data.size(); ++i) {
        temp.push_back(&(data.at(i)));
    }
    return SetRawData(temp);
}

bool KDTreeFlann::SetRawData(const std::vector<const CCVector3 *> &data)
{
    dimension_ = 3;
    dataset_size_ = data.size();
    if (dimension_ == 0 || dataset_size_ == 0) {
        CVLib::utility::LogWarning("[KDTreeFlann::SetRawData] Failed due to no data.");
        return false;
    }
    dataf_.resize(dataset_size_ * dimension_);
    memcpy(dataf_.data(), *data.data(),
           dataset_size_ * dimension_ * sizeof(PointCoordinateType));
    flann_datasetf_.reset(new flann::Matrix<PointCoordinateType>((PointCoordinateType *)dataf_.data(),
                                                   dataset_size_, dimension_));
    flann_indexf_.reset(new flann::Index<flann::L2<PointCoordinateType>>(
            *flann_datasetf_, flann::KDTreeSingleIndexParams(leaf_size_, reorder_)));
    flann_indexf_->buildIndex();
    return true;
}

template int KDTreeFlann::Search<CCVector3>(
        const CCVector3 &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::Query<CCVector3>(
        const std::vector<CCVector3> &queries,
        const KDTreeSearchParam &param,
        std::vector < std::vector<int> > &indices,
        std::vector < std::vector<double> > &distance2) const;
template int KDTreeFlann::SearchKNN<CCVector3>(
        const CCVector3 &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchRadius<CCVector3>(
        const CCVector3 &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchHybrid<CCVector3>(
        const CCVector3 &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTreeFlann::Search<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::Query<Eigen::Vector3d>(
		const std::vector<Eigen::Vector3d> &queries,
		const KDTreeSearchParam &param,
		std::vector < std::vector<int> > &indices,
		std::vector < std::vector<double> > &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTreeFlann::Search<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::Query<Eigen::VectorXd>(
		const std::vector<Eigen::VectorXd> &queries,
		const KDTreeSearchParam &param,
		std::vector < std::vector<int> > &indices,
		std::vector < std::vector<double> > &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

}  // namespace geometry
}  // namespace cloudViewer

#ifdef _MSC_VER
#pragma warning(pop)
#endif
