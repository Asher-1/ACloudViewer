// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_FEATURE_HEADER
#define ECV_FEATURE_HEADER

#include "eCV_db.h"

// LOCAL
#include "ecvKDTreeSearchParam.h"

// EIGEN
#include <Eigen.h>

class ccPointCloud;
namespace cloudViewer {
namespace utility {

/// \class Feature
///
/// \brief Class to store featrues for registration.
class ECV_DB_LIB_API Feature {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    /// Resize feature data buffer to `dim x n`.
    ///
    /// \param dim Feature dimension per point.
    /// \param n Number of points.
    void Resize(int dim, int n) {
        data_.resize(dim, n);
        data_.setZero();
    }
    /// Returns feature dimensions per point.
    size_t Dimension() const { return data_.rows(); }
    /// Returns number of points.
    size_t Num() const { return data_.cols(); }

public:
    /// Data buffer storing features.
    Eigen::MatrixXd data_;
};

/// Function to compute FPFH feature for a point cloud.
///
/// \param input The Input point cloud.
/// \param search_param KDTree KNN search parameter.
std::shared_ptr<Feature> ECV_DB_LIB_API ComputeFPFHFeature(
        const ccPointCloud &input,
        const geometry::KDTreeSearchParam &search_param =
		geometry::KDTreeSearchParamKNN());

}  // namespace utility
}  // namespace cloudViewer

#endif // ECV_FEATURE_HEADER
