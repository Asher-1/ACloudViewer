// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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

#include "ecvKDTreeFlann.h"

// CV_CORE_LIB
#include <Logging.h>
#include <nanoflann.hpp>


// LOCAL
#include "ecvHObjectCaster.h"
#include "ecvHalfEdgeMesh.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"

namespace cloudViewer {
namespace geometry {

KDTreeFlann::KDTreeFlann() {}

KDTreeFlann::KDTreeFlann(const Eigen::MatrixXd &data) { SetMatrixData(data); }

KDTreeFlann::KDTreeFlann(const ccHObject &geometry) { SetGeometry(geometry); }

KDTreeFlann::KDTreeFlann(const utility::Feature &feature) {
    SetFeature(feature);
}

KDTreeFlann::~KDTreeFlann() {}

bool KDTreeFlann::SetMatrixData(const Eigen::MatrixXd &data) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            data.data(), data.rows(), data.cols()));
}

bool KDTreeFlann::SetGeometry(const ccHObject &geometry) {
    switch (geometry.getClassID()) {
        case CV_TYPES::POINT_CLOUD: {
            const auto &cloud = dynamic_cast<const ccPointCloud &>(geometry);
            std::vector<Eigen::Vector3d> points = cloud.getEigenPoints();
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                    (const double *)points.data(), 3, points.size()));
        }
        case CV_TYPES::MESH: {
            const auto &mesh = dynamic_cast<const ccMesh &>(geometry);
            std::vector<Eigen::Vector3d> points = mesh.getEigenVertices();
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                    (const double *)points.data(), 3, points.size()));
        }
        case CV_TYPES::HALF_EDGE_MESH: {
            const auto &mesh = dynamic_cast<const ecvHalfEdgeMesh &>(geometry);
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                    (const double *)mesh.vertices_.data(), 3,
                    mesh.vertices_.size()));
        }
        case CV_TYPES::IMAGE:
        case CV_TYPES::HIERARCHY_OBJECT:
        default:
            utility::LogWarning(
                    "[KDTreeFlann::SetGeometry] Unsupported Geometry type.");
            return false;
    }
}

bool KDTreeFlann::SetFeature(const utility::Feature &feature) {
    return SetMatrixData(feature.data_);
}

bool KDTreeFlann::SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
    dimension_ = data.rows();
    dataset_size_ = data.cols();
    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning("[KDTreeFlann::SetRawData] Failed due to no data.");
        return false;
    }
    data_.resize(dataset_size_ * dimension_);
    memcpy(data_.data(), data.data(),
           dataset_size_ * dimension_ * sizeof(double));
    data_interface_.reset(new Eigen::Map<const Eigen::MatrixXd>((const double *)data_.data(), dimension_, dataset_size_));
    nanoflann_index_.reset(
            new KDTree_t(dimension_, std::cref(*data_interface_), 15));
    nanoflann_index_->index->buildIndex();
    return true;
}

template int KDTreeFlann::Search<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::Query<Eigen::Vector3d>(
        const std::vector<Eigen::Vector3d> &queries,
        const KDTreeSearchParam &param,
        std::vector<std::vector<int>> &indices,
        std::vector<std::vector<double>> &distance2) const;
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
        std::vector<std::vector<int>> &indices,
        std::vector<std::vector<double>> &distance2) const;
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
