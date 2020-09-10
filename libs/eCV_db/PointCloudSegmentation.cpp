// ----------------------------------------------------------------------------
// -                        Open3D: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.cloudViewer.org
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

#include "ecvPointCloud.h"
#include "ecvMesh.h"

#include <Eigen/Dense>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <unordered_set>

#include <Console.h>

namespace cloudViewer {
namespace geometry {

/// \class RANSACResult
///
/// \brief Stores the current best result in the RANSAC algorithm.
class RANSACResult {
public:
    RANSACResult() : fitness_(0), inlier_rmse_(0) {}
    ~RANSACResult() {}

public:
    double fitness_;
    double inlier_rmse_;
};

// Calculates the number of inliers given a list of points and a plane model,
// and the total distance between the inliers and the plane. These numbers are
// then used to evaluate how well the plane model fits the given points.
RANSACResult EvaluateRANSACBasedOnDistance(
	const std::vector<CCVector3> &points,
	const Eigen::Vector4d plane_model,
	std::vector<size_t> &inliers,
	double distance_threshold,
	double error) {
	RANSACResult result;

	for (size_t idx = 0; idx < points.size(); ++idx) {
		Eigen::Vector4d point(points[idx](0), points[idx](1), points[idx](2),
			1);
		double distance = std::abs(plane_model.dot(point));

		if (distance < distance_threshold) {
			error += distance;
			inliers.emplace_back(idx);
		}
	}

	size_t inlier_num = inliers.size();
	if (inlier_num == 0) {
		result.fitness_ = 0;
		result.inlier_rmse_ = 0;
	}
	else {
		result.fitness_ = (double)inlier_num / (double)points.size();
		result.inlier_rmse_ = error / std::sqrt((double)inlier_num);
	}
	return result;
}

RANSACResult EvaluateRANSACBasedOnDistance(
        const std::vector<Eigen::Vector3d> &points,
        const Eigen::Vector4d plane_model,
        std::vector<size_t> &inliers,
        double distance_threshold,
        double error) {
	return EvaluateRANSACBasedOnDistance(
		CCVector3::fromArrayContainer(points),
		plane_model, inliers, distance_threshold, error);
}


// Find the plane such that the summed squared distance from the
// plane to all points is minimized.
//
// Reference:
// https://www.ilikebigbits.com/2015_03_04_plane_from_points.html

Eigen::Vector4d GetPlaneFromPoints(
	const std::vector<CCVector3> &points,
	const std::vector<size_t> &inliers) {
	CCVector3 centroid(0, 0, 0);
	for (size_t idx : inliers) {
		centroid += points[idx];
	}
	centroid /= PointCoordinateType(inliers.size());

	PointCoordinateType xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

	for (size_t idx : inliers) {
		CCVector3 r = points[idx] - centroid;
		xx += r(0) * r(0);
		xy += r(0) * r(1);
		xz += r(0) * r(2);
		yy += r(1) * r(1);
		yz += r(1) * r(2);
		zz += r(2) * r(2);
	}

	PointCoordinateType det_x = yy * zz - yz * yz;
	PointCoordinateType det_y = xx * zz - xz * xz;
	PointCoordinateType det_z = xx * yy - xy * xy;

	CCVector3 abc;
	if (det_x > det_y && det_x > det_z) {
		abc = CCVector3(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
	}
	else if (det_y > det_z) {
		abc = CCVector3(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
	}
	else {
		abc = CCVector3(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
	}

	PointCoordinateType norm = abc.norm();
	// Return invalid plane if the points don't span a plane.
	if (norm == 0) {
		return Eigen::Vector4d(0, 0, 0, 0);
	}
	abc /= abc.norm();
	double d = -abc.dot(centroid);
	return Eigen::Vector4d(abc(0), abc(1), abc(2), d);
}

Eigen::Vector4d GetPlaneFromPoints(const std::vector<Eigen::Vector3d> &points,
                                   const std::vector<size_t> &inliers) {
	return GetPlaneFromPoints(CCVector3::fromArrayContainer(points), inliers);
}


}  // namespace geometry
}  // namespace cloudViewer

using namespace cloudViewer::geometry;
std::tuple<Eigen::Vector4d, std::vector<size_t>>
ccPointCloud::segmentPlane(
	const double distance_threshold /* = 0.01 */,
	const int ransac_n /* = 3 */,
	const int num_iterations /* = 100 */) const {
	RANSACResult result;
	double error = 0;

	// Initialize the plane model ax + by + cz + d = 0.
	Eigen::Vector4d plane_model = Eigen::Vector4d(0, 0, 0, 0);
	// Initialize the best plane model.
	Eigen::Vector4d best_plane_model = Eigen::Vector4d(0, 0, 0, 0);

	// Initialize consensus set.
	std::vector<size_t> inliers;

	size_t num_points = size();
	std::vector<size_t> indices(num_points);
	std::iota(std::begin(indices), std::end(indices), 0);

	std::random_device rd;
	std::mt19937 rng(rd());

	// Return if ransac_n is less than the required plane model parameters.
	if (ransac_n < 3) {
		CVLib::utility::LogError(
			"ransac_n should be set to higher than or equal to 3.");
		return std::make_tuple(best_plane_model, inliers);
	}
	if (num_points < size_t(ransac_n)) {
		CVLib::utility::LogError("There must be at least 'ransac_n' points.");
		return std::make_tuple(best_plane_model, inliers);
	}

	for (int itr = 0; itr < num_iterations; itr++) {
		for (int i = 0; i < ransac_n; ++i) {
			std::swap(indices[i], indices[rng() % num_points]);
		}
		inliers.clear();
		for (int idx = 0; idx < ransac_n; ++idx) {
			inliers.emplace_back(indices[idx]);
		}

		// Fit model to num_model_parameters randomly selected points among the
		// inliers.
		plane_model = ccMesh::ComputeTrianglePlane(
			getEigenPoint(inliers[0]), getEigenPoint(inliers[1]), getEigenPoint(inliers[2]));
		if (plane_model.isZero(0)) {
			continue;
		}

		error = 0;
		inliers.clear();
		auto this_result = EvaluateRANSACBasedOnDistance(
			getPoints(), plane_model, inliers, distance_threshold, error);
		if (this_result.fitness_ > result.fitness_ ||
			(this_result.fitness_ == result.fitness_ &&
				this_result.inlier_rmse_ < result.inlier_rmse_)) {
			result = this_result;
			best_plane_model = plane_model;
		}
	}

	// Find the final inliers using best_plane_model.
	inliers.clear();
	for (size_t idx = 0; idx < size(); ++idx) {
		Eigen::Vector4d point(m_points[idx](0), m_points[idx](1), m_points[idx](2),
			1);
		double distance = std::abs(best_plane_model.dot(point));

		if (distance < distance_threshold) {
			inliers.emplace_back(idx);
		}
	}

	// Improve best_plane_model using the final inliers.
	best_plane_model = GetPlaneFromPoints(getPoints(), inliers);

	CVLib::utility::LogDebug("RANSAC | Inliers: {:d}, Fitness: {:e}, RMSE: {:e}",
		inliers.size(), result.fitness_, result.inlier_rmse_);
	return std::make_tuple(best_plane_model, inliers);
}

std::shared_ptr<ccPointCloud>
ccPointCloud::Crop(const ccBBox &bbox) const
{
	if (!bbox.isValid()) {
		CVLog::Warning(

			"[CropPointCloud::Crop] ccBBox either has zeros "
			"size, or has wrong bounds.");
	}
	return selectByIndex(bbox.getPointIndicesWithinBoundingBox(m_points));
}

std::shared_ptr<ccPointCloud>
ccPointCloud::Crop(const ecvOrientedBBox &bbox) const
{
	if (bbox.isEmpty()) {
		CVLog::Warning(
			"[CropPointCloud::Crop] ecvOrientedBBox either has zeros "
			"size, or has wrong bounds.");
		return nullptr;
	}
	return selectByIndex(bbox.getPointIndicesWithinBoundingBox(m_points));
}
