// ----------------------------------------------------------------------------
// -                        CVLib: www.CVLib.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.CVLib.org
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

#ifndef CV_KDTREE_FLANN_HEADER
#define CV_KDTREE_FLANN_HEADER

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "ecvHObject.h"
#include "ecvFeature.h"
#include "ecvKDTreeSearchParam.h"

namespace flann {
template <typename T>
class Matrix;
template <typename T>
struct L2;
template <typename T>
class Index;
}  // namespace flann

namespace cloudViewer {
namespace geometry {

/// \class KDTreeFlann
///
/// \brief KDTree with FLANN for nearest neighbor search.
class ECV_DB_LIB_API KDTreeFlann {
public:
    /// \brief Default Constructor.
	/// \param leaf_size positive integer (default = 15), Number of points at which to switch to brute-force. Changing
	/// leaf_size will not affect the results of a query, but can
	///	significantly impact the speed of a query and the memory required
	///	to store the constructed tree.
	/// \param reorder
	KDTreeFlann(size_t leaf_size = 15, bool reorder = true);
    /// \brief Parameterized Constructor.
    ///
    /// \param data Provides set of data points for KDTree construction.
	/// \param leaf_size positive integer (default = 15), Number of points at which to switch to brute-force. Changing
	/// leaf_size will not affect the results of a query, but can
	///	significantly impact the speed of a query and the memory required
	///	to store the constructed tree.
	/// \param reorder
    KDTreeFlann(const Eigen::MatrixXd &data,
		size_t leaf_size = 15, bool reorder = true);
    /// \brief Parameterized Constructor.
    ///
    /// \param geometry Provides geometry from which KDTree is constructed.
	/// \param leaf_size positive integer (default = 15), Number of points at which to switch to brute-force. Changing
	/// leaf_size will not affect the results of a query, but can
	///	significantly impact the speed of a query and the memory required
	///	to store the constructed tree.
	/// \param reorder
    KDTreeFlann(const ccHObject &geometry, 
		size_t leaf_size = 15, bool reorder = true);
    /// \brief Parameterized Constructor.
    ///
    /// \param feature Provides a set of features from which the KDTree is constructed.
	/// \param leaf_size positive integer (default = 15), Number of points at which to switch to brute-force. Changing
	/// leaf_size will not affect the results of a query, but can
	///	significantly impact the speed of a query and the memory required
	///	to store the constructed tree.
	/// \param reorder
    KDTreeFlann(const utility::Feature &feature, 
		size_t leaf_size = 15, bool reorder = true);
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
               std::vector<double> &distance2) const;

	template <typename T>
	int Query(const std::vector<T> &queries,
		const KDTreeSearchParam &param,
		std::vector<std::vector<int>> &indices,
		std::vector<std::vector<double>> &distance2) const;

    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  std::vector<int> &indices,
                  std::vector<double> &distance2) const;

    template <typename T>
    int SearchRadius(const T &query,
                     double radius,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const;

    template <typename T>
    int SearchHybrid(const T &query,
                     double radius,
                     int max_nn,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const;

private:
    /// \brief Sets the KDTree data from the data provided by the other methods.
    ///
    /// Internal method that sets all the members of KDTree by data provided by
    /// features, geometry, etc.
    bool SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data);

public:
	std::vector<double> data_;
	size_t dimension_ = 0;
	size_t dataset_size_ = 0;
	size_t leaf_size_ = 15;
	bool reorder_ = true;

protected:
    std::unique_ptr<flann::Matrix<double>> flann_dataset_;
    std::unique_ptr<flann::Index<flann::L2<double>>> flann_index_;
};

}  // namespace geometry
}  // namespace cloudViewer

#endif // CV_KDTREE_FLANN_HEADER