// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
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

#include "ecvPointCloud.h"
#include "Image.h"
#include "ecvMesh.h"
#include "ecvTetraMesh.h"
#include "RGBDImage.h"
#include "VoxelGrid.h"
#include "ecvQhull.h"
#include "ecvKDTreeFlann.h"
#include "ecvScalarField.h"
#include "camera/PinholeCameraIntrinsic.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <queue>
#include <tuple>
#include <limits>
#include <Console.h>

using namespace CVLib;

namespace cloudViewer {

namespace {
using namespace geometry;

int CountValidDepthPixels(const Image& depth, int stride) {
	int num_valid_pixels = 0;
	for (int i = 0; i < depth.height_; i += stride) {
		for (int j = 0; j < depth.width_; j += stride) {
			const float* p = depth.PointerAt<float>(j, i);
			if (*p > 0) num_valid_pixels += 1;
		}
	}
	return num_valid_pixels;
}

std::shared_ptr<ccPointCloud> CreatePointCloudFromFloatDepthImage(
	const Image& depth,
	const camera::PinholeCameraIntrinsic& intrinsic,
	const Eigen::Matrix4d& extrinsic,
	int stride,
	bool project_valid_depth_only) {
	auto pointcloud = std::make_shared<ccPointCloud>();
	Eigen::Matrix4d camera_pose = extrinsic.inverse();
	auto focal_length = intrinsic.GetFocalLength();
	auto principal_point = intrinsic.GetPrincipalPoint();
	int num_valid_pixels;
	if (!project_valid_depth_only) {
		num_valid_pixels = int(depth.height_ / stride) * int(depth.width_ / stride);
	}
	else {
		num_valid_pixels = CountValidDepthPixels(depth, stride);
	}
	pointcloud->resize(num_valid_pixels);
	int cnt = 0;
	for (int i = 0; i < depth.height_; i += stride) {
		for (int j = 0; j < depth.width_; j += stride) {
			const float* p = depth.PointerAt<float>(j, i);
			if (*p > 0) {
				double z = (double)(*p);
				double x = (j - principal_point.first) * z / focal_length.first;
				double y =
					(i - principal_point.second) * z / focal_length.second;
				Eigen::Vector4d point = camera_pose * Eigen::Vector4d(x, y, z, 1.0);

				pointcloud->setEigenPoint(static_cast<size_t>(cnt++), point.block<3, 1>(0, 0));
			}
			else if (!project_valid_depth_only) {
				double z = std::numeric_limits<float>::quiet_NaN();
				double x = std::numeric_limits<float>::quiet_NaN();
				double y = std::numeric_limits<float>::quiet_NaN();
				pointcloud->setEigenPoint(static_cast<size_t>(cnt++), Eigen::Vector3d(x, y, z));
			}
		}
	}
	return pointcloud;
}

template <typename TC, int NC>
std::shared_ptr<ccPointCloud> CreatePointCloudFromRGBDImageT(
	const RGBDImage& image,
	const camera::PinholeCameraIntrinsic& intrinsic,
	const Eigen::Matrix4d& extrinsic,
	bool project_valid_depth_only) {
	auto pointcloud = std::make_shared<ccPointCloud>();
	Eigen::Matrix4d camera_pose = extrinsic.inverse();
	auto focal_length = intrinsic.GetFocalLength();
	auto principal_point = intrinsic.GetPrincipalPoint();
	double scale = (sizeof(TC) == 1) ? 255.0 : 1.0;
	int num_valid_pixels;
	if (!project_valid_depth_only) {
		num_valid_pixels = image.depth_.height_ * image.depth_.width_;
	}
	else {
		num_valid_pixels = CountValidDepthPixels(image.depth_, 1);
	}
	pointcloud->resize(num_valid_pixels);
	pointcloud->resizeTheRGBTable();
	int cnt = 0;
	for (int i = 0; i < image.depth_.height_; i++) {
		float* p = (float*)(image.depth_.data_.data() +
			i * image.depth_.BytesPerLine());
		TC* pc = (TC*)(image.color_.data_.data() +
			i * image.color_.BytesPerLine());
		for (int j = 0; j < image.depth_.width_; j++, p++, pc += NC) {
			if (*p > 0) {
				double z = (double)(*p);
				double x = (j - principal_point.first) * z / focal_length.first;
				double y = (i - principal_point.second) * z / focal_length.second;
				Eigen::Vector4d point = camera_pose * Eigen::Vector4d(x, y, z, 1.0);
				pointcloud->setEigenPoint(static_cast<size_t>(cnt), point.block<3, 1>(0, 0));
				pointcloud->setEigenColor(static_cast<size_t>(cnt++),
					Eigen::Vector3d(pc[0], pc[(NC - 1) / 2], pc[NC - 1]) / scale);
			}
			else if (!project_valid_depth_only) {
				double z = std::numeric_limits<float>::quiet_NaN();
				double x = std::numeric_limits<float>::quiet_NaN();
				double y = std::numeric_limits<float>::quiet_NaN();
				pointcloud->setEigenPoint(static_cast<size_t>(cnt), Eigen::Vector3d(x, y, z));
				pointcloud->setEigenColor(static_cast<size_t>(cnt++),
					Eigen::Vector3d(
						std::numeric_limits<TC>::quiet_NaN(),
						std::numeric_limits<TC>::quiet_NaN(),
						std::numeric_limits<TC>::quiet_NaN()));
			}
		}
	}
	return pointcloud;
}

// Disjoint set data structure to find cycles in graphs
class DisjointSet {
public:
    DisjointSet(size_t size) : parent_(size), size_(size) {
        for (size_t idx = 0; idx < size; idx++) {
            parent_[idx] = idx;
            size_[idx] = 0;
        }
    }

    // find representative element for given x
    // using path compression
    size_t Find(size_t x) {
        if (x != parent_[x]) {
            parent_[x] = Find(parent_[x]);
        }
        return parent_[x];
    }

    // combine two sets using size of sets
    void Union(size_t x, size_t y) {
        x = Find(x);
        y = Find(y);
        if (x != y) {
            if (size_[x] < size_[y]) {
                size_[y] += size_[x];
                parent_[x] = y;
            } else {
                size_[x] += size_[y];
                parent_[y] = x;
            }
        }
    }

private:
    std::vector<size_t> parent_;
    std::vector<size_t> size_;
};

struct WeightedEdge {
    WeightedEdge(size_t v0, size_t v1, double weight)
        : v0_(v0), v1_(v1), weight_(weight) {}
    size_t v0_;
    size_t v1_;
    double weight_;
};

// Minimum Spanning Tree algorithm (Kruskal's algorithm)
std::vector<WeightedEdge> Kruskal(
        std::vector<WeightedEdge> &edges, size_t n_vertices) {
    std::sort(edges.begin(), edges.end(),
                [](WeightedEdge &e0, WeightedEdge &e1) {
                    return e0.weight_ < e1.weight_;
                });
    DisjointSet disjoint_set(n_vertices);
    std::vector<WeightedEdge> mst;
    for (size_t eidx = 0; eidx < edges.size(); ++eidx) {
        size_t set0 = disjoint_set.Find(edges[eidx].v0_);
        size_t set1 = disjoint_set.Find(edges[eidx].v1_);
        if (set0 != set1) {
            mst.push_back(edges[eidx]);
            disjoint_set.Union(set0, set1);
        }
    }
    return mst;
}

}  // unnamed namespace
}  // namespace cloudViewer

std::shared_ptr<ccPointCloud> ccPointCloud::CreateFromDepthImage(
	const cloudViewer::geometry::Image& depth,
	const cloudViewer::camera::PinholeCameraIntrinsic& intrinsic,
	const Eigen::Matrix4d& extrinsic/* = Eigen::Matrix4d::Identity()*/,
	double depth_scale/* = 1000.0*/,
	double depth_trunc/* = 1000.0*/,
	int stride/* = 1*/,
	bool project_valid_depth_only/* = true*/) {
	if (depth.num_of_channels_ == 1) {
		if (depth.bytes_per_channel_ == 2) {
			auto float_depth =
				depth.ConvertDepthToFloatImage(depth_scale, depth_trunc);
			return cloudViewer::CreatePointCloudFromFloatDepthImage(
				*float_depth, intrinsic, extrinsic, stride,
				project_valid_depth_only);
		}
		else if (depth.bytes_per_channel_ == 4) {
			return cloudViewer::CreatePointCloudFromFloatDepthImage(
				depth, intrinsic, extrinsic, stride,
				project_valid_depth_only);
		}
	}
	CVLib::utility::LogError(
		"[CreatePointCloudFromDepthImage] Unsupported image format.");
	return std::make_shared<ccPointCloud>();
}

std::shared_ptr<ccPointCloud> ccPointCloud::CreateFromRGBDImage(
	const cloudViewer::geometry::RGBDImage& image,
	const cloudViewer::camera::PinholeCameraIntrinsic& intrinsic,
	const Eigen::Matrix4d& extrinsic/* = Eigen::Matrix4d::Identity()*/,
	bool project_valid_depth_only/* = true*/) {
    if (image.depth_.num_of_channels_ == 1 &&
        image.depth_.bytes_per_channel_ == 4) {
        if (image.color_.bytes_per_channel_ == 1 &&
            image.color_.num_of_channels_ == 3) {
            return cloudViewer::CreatePointCloudFromRGBDImageT<uint8_t, 3>(
                    image, intrinsic, extrinsic, project_valid_depth_only);
        } else if (image.color_.bytes_per_channel_ == 1 &&
                   image.color_.num_of_channels_ == 4) {
            return cloudViewer::CreatePointCloudFromRGBDImageT<uint8_t, 4>(
                    image, intrinsic, extrinsic, project_valid_depth_only);
        } else if (image.color_.bytes_per_channel_ == 4 &&
                   image.color_.num_of_channels_ == 1) {
            return cloudViewer::CreatePointCloudFromRGBDImageT<float, 1>(
                    image, intrinsic, extrinsic, project_valid_depth_only);
        }
    }
	CVLib::utility::LogError(
		"[CreatePointCloudFromRGBDImage] Unsupported image format.");
	return std::make_shared<ccPointCloud>();
}

std::shared_ptr<ccPointCloud> ccPointCloud::createFromVoxelGrid(
	const cloudViewer::geometry::VoxelGrid &voxel_grid) {
	auto output = std::make_shared<ccPointCloud>();
	output->resize(static_cast<unsigned int>(voxel_grid.voxels_.size()));
	bool has_colors = voxel_grid.HasColors();
	if (has_colors) {
		output->resizeTheRGBTable();
	}
	size_t vidx = 0;
	for (auto &it : voxel_grid.voxels_) {
		const cloudViewer::geometry::Voxel voxel = it.second;
		output->setPoint(vidx,
			voxel_grid.GetVoxelCenterCoordinate(voxel.grid_index_));
		if (has_colors) {
			output->setPointColor(static_cast<unsigned int>(vidx),
				ecvColor::Rgb::FromEigen(voxel.color_));
		}
		vidx++;
	}
	return output;
}

ccPointCloud& ccPointCloud::normalizeNormals()
{
	if (hasNormals())
	{
		for (size_t i = 0; i < m_normals->size(); ++i) {
			ccNormalVectors::GetNormalPtr(m_normals->getValue(i)).normalize();
		}
	}
	return *this;
}

std::vector<double>
ccPointCloud::computePointCloudDistance(const ccPointCloud &target)
{
	std::vector<double> distances(size());
	cloudViewer::geometry::KDTreeFlann kdtree;
	kdtree.SetGeometry(target);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < static_cast<int>(size()); i++) {
		std::vector<int> indices(1);
		std::vector<double> dists(1);
		if (kdtree.SearchKNN(CCVector3d::fromArray(*getPoint(static_cast<unsigned int>(i))), 1, indices, dists) == 0) {
			utility::LogDebug(
				"[ccPointCloud::computePointCloudDistance] Found a point without neighbors.");
			distances[i] = 0.0;
		}
		else {
			distances[i] = std::sqrt(dists[0]);
		}
	}
	return distances;
}

ccPointCloud& ccPointCloud::removeNonFinitePoints(bool remove_nan, bool remove_infinite)
{

	bool has_normal = hasNormals();
	bool has_color = hasColors();
	bool has_fwf = hasFWF();
	//scalar fields
	unsigned sfCount = getNumberOfScalarFields();

	size_t old_point_num = m_points.size();
	size_t k = 0;                                 // new index
	for (size_t i = 0; i < old_point_num; i++) {  // old index

		bool is_nan = remove_nan &&
			(std::isnan(m_points[i][0]) ||
				std::isnan(m_points[i][1]) ||
				std::isnan(m_points[i][2]));

		bool is_infinite = remove_infinite &&
			(std::isinf(m_points[i][0]) ||
				std::isinf(m_points[i][1]) ||
				std::isinf(m_points[i][2]));

		if (!is_nan && !is_infinite) {
			m_points[k] = m_points[i];
			if (has_normal) m_points[k] = m_points[i];
			if (has_color) m_points[k] = m_points[i];
			if (has_fwf)
			{
				if (fwfDescriptors().contains(m_fwfWaveforms[k].descriptorID()))
				{
					// remove invalid descriptors
					fwfDescriptors().remove(m_fwfWaveforms[k].descriptorID());
				}
				m_fwfWaveforms[k] = m_fwfWaveforms[i];
			}


			if (sfCount != 0)
			{
				bool error = false;
				for (unsigned j = 0; j < sfCount; ++j)
				{
					ccScalarField* currentScalarField = static_cast<ccScalarField*>(getScalarField(j));

					if (currentScalarField)
					{
						currentScalarField->setValue(j, currentScalarField->getValue(i));
					}
					else
					{
						error = true;
						utility::LogWarning("[removeNonFinitePoints] Not enough memory to copy scalar field!");
					}
				}

				if (error)
				{
					deleteAllScalarFields();
				}
			}

			k++;
		}
	}

	if (!resize(static_cast<unsigned int>(k)))
	{
		utility::LogError("point cloud resize error!!!");
	}

	sfCount = getNumberOfScalarFields();
	if (sfCount != 0)
	{
		for (unsigned j = 0; j < sfCount; ++j)
		{
			ccScalarField* currentScalarField = static_cast<ccScalarField*>(getScalarField(j));
			currentScalarField->resizeSafe(k);
			currentScalarField->computeMinAndMax();
		}

		//we display the same scalar field as the source (if we managed to copy it!)
		if (getCurrentDisplayedScalarField())
		{
			int sfIdx = getScalarFieldIndexByName(getCurrentDisplayedScalarField()->getName());
			if (sfIdx >= 0)
				setCurrentDisplayedScalarField(sfIdx);
			else
				setCurrentDisplayedScalarField(static_cast<int>(sfCount) - 1);
		}

	}

	CVLog::Print(
		"[RemoveNonFinitePoints] {:d} nan points have been removed.",
		(int)(old_point_num - k));
	utility::LogDebug("[RemoveNonFinitePoints] {:d} nan points have been removed.",
		(int)(old_point_num - k));

	return *this;
}

// helper classes for VoxelDownSample and VoxelDownSampleAndTrace
namespace {
	class AccumulatedPoint {
	public:
		AccumulatedPoint()
			: num_of_points_(0),
			point_(0.0, 0.0, 0.0),
			normal_(0.0, 0.0, 0.0),
			color_(0.0, 0.0, 0.0) {}

	public:
		void AddPoint(const ccPointCloud &cloud, unsigned int index) {

			point_ += cloud.getEigenPoint(index);
			if (cloud.hasNormals()) {
				if (!std::isnan(cloud.getPointNormal(index)[0]) &&
					!std::isnan(cloud.getPointNormal(index)[1]) &&
					!std::isnan(cloud.getPointNormal(index)[2])) {
					normal_ += cloud.getEigenNormal(index);
				}
			}
			if (cloud.hasColors()) {
				color_ += cloud.getEigenColor(index);
			}
			num_of_points_++;
		}

		Eigen::Vector3d GetAveragePoint() const {
			return point_ / double(num_of_points_);
		}

		Eigen::Vector3d GetAverageNormal() const { return normal_.normalized(); }

		Eigen::Vector3d GetAverageColor() const {
			return color_ / double(num_of_points_);
		}

	public:
		int num_of_points_;
		Eigen::Vector3d point_;
		Eigen::Vector3d normal_;
		Eigen::Vector3d color_;
	};

	class point_cubic_id {
	public:
		size_t point_id;
		int cubic_id;
	};

	class AccumulatedPointForTrace : public AccumulatedPoint {
	public:
		void AddPoint(const ccPointCloud &cloud,
			size_t index, int cubic_index,
			bool approximate_class)
		{
			point_ += cloud.getEigenPoint(index);
			if (cloud.hasNormals()) {
				if (!std::isnan(cloud.getEigenNormal(index)(0)) &&
					!std::isnan(cloud.getEigenNormal(index)(1)) &&
					!std::isnan(cloud.getEigenNormal(index)(2))) {
					normal_ += cloud.getEigenNormal(index);
				}
			}
			if (cloud.hasColors()) {
				Eigen::Vector3d fcolor = cloud.getEigenColor(index);

				if (approximate_class) {
					auto got = classes.find(int(fcolor[0]));
					if (got == classes.end())
						classes[int(fcolor[0])] = 1;
					else
						classes[int(fcolor[0])] += 1;
				}
				else {
					color_ += fcolor;
				}
			}

			point_cubic_id new_id;
			new_id.point_id = index;
			new_id.cubic_id = cubic_index;
			original_id.push_back(new_id);
			num_of_points_++;
		}

		Eigen::Vector3d GetMaxClass() {
			int max_class = -1;
			int max_count = -1;
			for (auto it = classes.begin(); it != classes.end(); it++) {
				if (it->second > max_count) {
					max_count = it->second;
					max_class = it->first;
				}
			}
			return Eigen::Vector3d(max_class, max_class, max_class);
		}

		std::vector<point_cubic_id> GetOriginalID() { return original_id; }

	private:
		// original point cloud id in higher resolution + its cubic id
		std::vector<point_cubic_id> original_id;
		std::unordered_map<int, int> classes;
	};
}  // namespace

std::shared_ptr<ccPointCloud>
ccPointCloud::voxelDownSample(double voxel_size)
{
	auto output = std::make_shared<ccPointCloud>("pointCloud");
	//visibility
	output->setVisible(isVisible());
	output->setEnabled(isEnabled());

	//other parameters
	output->importParametersFrom(this);

	if (voxel_size <= 0.0) {
		utility::LogError("[ccPointCloud::voxelDownSample] voxel_size <= 0.");
	}

	Eigen::Vector3d voxel_size3 =
		Eigen::Vector3d(voxel_size, voxel_size, voxel_size);

	Eigen::Vector3d voxel_min_bound = getMinBound() - voxel_size3 * 0.5;
	Eigen::Vector3d voxel_max_bound = getMaxBound() + voxel_size3 * 0.5;
	if (voxel_size * std::numeric_limits<int>::max() <
		(voxel_max_bound - voxel_min_bound).maxCoeff()) {
		utility::LogError("[ccPointCloud::voxelDownSample] voxel_size is too small.");
	}
	std::unordered_map<Eigen::Vector3i, AccumulatedPoint,
		CVLib::utility::hash_eigen::hash<Eigen::Vector3i>> voxelindex_to_accpoint;

	Eigen::Vector3d ref_coord;
	Eigen::Vector3i voxel_index;
	for (int i = 0; i < (int)size(); i++) {
		Eigen::Vector3d p = getEigenPoint(i); // must reserve a temp variable
		ref_coord = (p - voxel_min_bound) / voxel_size;
		voxel_index <<	int(floor(ref_coord(0))), 
						int(floor(ref_coord(1))),
						int(floor(ref_coord(2)));
		voxelindex_to_accpoint[voxel_index].AddPoint(*this, i);
	}

	bool has_normals = hasNormals();
	bool has_colors = hasColors();

	if (!output->reserveThePointsTable(
		static_cast<unsigned int>(voxelindex_to_accpoint.size())))
	{
		utility::LogError(
			"[ccPointCloud::voxelDownSample] Not enough memory to duplicate cloud!");
		return nullptr;
	}

	// RGB colors
	if (has_colors)
	{
		if (output->reserveTheRGBTable())
		{
			output->showColors(colorsShown());
		}
		else
		{
			utility::LogWarning("[ccPointCloud::voxelDownSample] Not enough memory to copy RGB colors!");
			has_colors = false;
		}
	}

	// Normals
	if (has_normals)
	{
		if (output->reserveTheNormsTable())
		{
			output->showNormals(normalsShown());
		}
		else
		{
			utility::LogWarning("[ccPointCloud::voxelDownSample] Not enough memory to copy normals!");
			has_normals = false;
		}
	}

	int j = 0;
	for (auto accpoint : voxelindex_to_accpoint) {

		//import points
		output->addPoint(accpoint.second.GetAveragePoint());

		// import colors
		if (has_colors)
		{
			output->addEigenColor(accpoint.second.GetAverageColor());
		}

		// import normals
		if (has_normals)
		{
			output->addEigenNorm(accpoint.second.GetAverageNormal());
		}
	}

	utility::LogDebug(
		"ccPointCloud down sampled from {:d} points to {:d} points.",
		(int)size(), (int)output->size());
	return output;
}

std::tuple<std::shared_ptr<ccPointCloud>, 
		   Eigen::MatrixXi,
		   std::vector<std::vector<int>>>
ccPointCloud::voxelDownSampleAndTrace(double voxel_size,
	const Eigen::Vector3d &min_bound,
	const Eigen::Vector3d &max_bound,
	bool approximate_class) const
{
	if (voxel_size <= 0.0) {
		utility::LogError("[voxelDownSampleAndTrace] voxel_size <= 0.");
	}

	// Note: this is different from VoxelDownSample.
	// It is for fixing coordinate for multiscale voxel space
	auto voxel_min_bound = min_bound;
	auto voxel_max_bound = max_bound;
	if (voxel_size * std::numeric_limits<int>::max() <
		(voxel_max_bound - voxel_min_bound).maxCoeff()) {
		utility::LogError("[voxelDownSampleAndTrace] voxel_size is too small.");
	}

	Eigen::MatrixXi cubic_id;
	auto output = std::make_shared<ccPointCloud>("pointCloud");
	//visibility
	output->setVisible(isVisible());
	output->setEnabled(isEnabled());

	//other parameters
	output->importParametersFrom(this);

	std::unordered_map<Eigen::Vector3i, AccumulatedPointForTrace,
		CVLib::utility::hash_eigen::hash<Eigen::Vector3i>> voxelindex_to_accpoint;

	int cid_temp[3] = { 1, 2, 4 };
	for (size_t i = 0; i < this->size(); i++) {
		Eigen::Vector3d p = getEigenPoint(i); // must reserve a temp variable
		auto ref_coord = (p - voxel_min_bound) / voxel_size;
		auto voxel_index = Eigen::Vector3i(int(floor(ref_coord(0))),
										   int(floor(ref_coord(1))),
										   int(floor(ref_coord(2))));
		int cid = 0;
		for (int c = 0; c < 3; c++) {
			if ((ref_coord(c) - voxel_index(c)) >= 0.5) {
				cid += cid_temp[c];
			}
		}
		voxelindex_to_accpoint[voxel_index].AddPoint(*this, i, cid, approximate_class);
	}

	bool has_normals = hasNormals();
	bool has_colors = hasColors();
	int cnt = 0;
	cubic_id.resize(voxelindex_to_accpoint.size(), 8);
	cubic_id.setConstant(-1);
	std::vector<std::vector<int>> original_indices(voxelindex_to_accpoint.size());

	if (!output->reserveThePointsTable(
		static_cast<unsigned int>(voxelindex_to_accpoint.size())))
	{
		utility::LogError("[ccPointCloud::voxelDownSampleAndTrace] Not enough memory to duplicate cloud!");
	}

	// RGB colors
	if (has_colors)
	{
		if (output->reserveTheRGBTable())
		{
			output->showColors(colorsShown());
		}
		else
		{
			utility::LogWarning("[ccPointCloud::voxelDownSampleAndTrace] Not enough memory to copy RGB colors!");
			has_colors = false;
		}
	}

	// Normals
	if (has_normals)
	{
		if (output->reserveTheNormsTable())
		{
			output->showNormals(normalsShown());
		}
		else
		{
			utility::LogWarning("[ccPointCloud::voxelDownSampleAndTrace] Not enough memory to copy normals!");
			has_normals = false;
		}
	}

	for (auto accpoint : voxelindex_to_accpoint) {
		//import points
		output->addEigenPoint(accpoint.second.GetAveragePoint());

		// import colors
		if (has_colors)
		{
			if (approximate_class) {
				output->addEigenColor(accpoint.second.GetMaxClass());
			}
			else {
				output->addEigenColor(accpoint.second.GetAverageColor());
			}
		}

		// import normals
		if (has_normals)
		{
			output->addEigenNorm(accpoint.second.GetAverageNormal());
		}

		auto original_id = accpoint.second.GetOriginalID();
		for (int i = 0; i < (int)original_id.size(); i++) {
			size_t pid = original_id[i].point_id;
			int cid = original_id[i].cubic_id;
			cubic_id(cnt, cid) = int(pid);
			original_indices[cnt].push_back(int(pid));
		}
		cnt++;
	}

	utility::LogDebug(
		"ccPointCloud down sampled from {:d} points to {:d} points.",
		(int)size(), (int)output->size());

	return std::make_tuple(output, cubic_id, original_indices);
}

std::shared_ptr<ccPointCloud>
ccPointCloud::uniformDownSample(size_t every_k_points) const
{
	if (every_k_points == 0) {
		utility::LogWarning("[ccPointCloud::uniformDownSample] Illegal sample rate.");
		return nullptr;
	}

	std::vector<size_t> indices;
	for (size_t i = 0; i < size(); i += every_k_points) {
		indices.push_back(i);
	}

	return selectByIndex(indices);
}


std::tuple<std::shared_ptr<ccPointCloud>, std::vector<size_t>>
ccPointCloud::removeRadiusOutliers(size_t nb_points, double search_radius) const
{
	if (nb_points < 1 || search_radius <= 0) {
		utility::LogWarning(
			"[RemoveRadiusOutliers] Illegal input parameters,"
			"number of points and radius must be positive");
		return std::make_tuple(std::make_shared<ccPointCloud>("pointCloud"),
			std::vector<size_t>());
	}
	cloudViewer::geometry::KDTreeFlann kdtree;
	kdtree.SetGeometry(*this);
	std::vector<bool> mask = std::vector<bool>(size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < int(size()); i++) {
		std::vector<int> tmp_indices;
		std::vector<double> dist;
		size_t nb_neighbors = kdtree.SearchRadius(
			CCVector3d::fromArray(*getPoint(static_cast<unsigned int>(i))),
			search_radius,
			tmp_indices, dist);
		mask[i] = (nb_neighbors > nb_points);
	}
	std::vector<size_t> indices;
	for (size_t i = 0; i < mask.size(); i++) {
		if (mask[i]) {
			indices.push_back(i);
		}
	}
	return std::make_tuple(selectByIndex(indices), indices);
}

std::tuple<std::shared_ptr<ccPointCloud>, std::vector<size_t>>
ccPointCloud::removeStatisticalOutliers(size_t nb_neighbors,
	double std_ratio) const {
	if (nb_neighbors < 1 || std_ratio <= 0) {
		utility::LogWarning(
			"[RemoveStatisticalOutliers] Illegal input parameters, number "
			"of neighbors and standard deviation ratio must be positive");
		return std::make_tuple(std::make_shared<ccPointCloud>("pointCloud"),
			std::vector<size_t>());
	}

	if (size() == 0) {
		return std::make_tuple(std::make_shared<ccPointCloud>("pointCloud"),
			std::vector<size_t>());
	}

	cloudViewer::geometry::KDTreeFlann kdtree;
	kdtree.SetGeometry(*this);
	std::vector<double> avg_distances = std::vector<double>(size());
	std::vector<size_t> indices;
	size_t valid_distances = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < int(size()); i++) {
		std::vector<int> tmp_indices;
		std::vector<double> dist;
		kdtree.SearchKNN(
			CCVector3d::fromArray(*getPoint(static_cast<unsigned int>(i))),
			int(nb_neighbors), tmp_indices, dist);
		double mean = -1.0;
		if (dist.size() > 0u) {
			valid_distances++;
			std::for_each(dist.begin(), dist.end(),
				[](double &d) { d = std::sqrt(d); });
			mean = std::accumulate(dist.begin(), dist.end(), 0.0) / dist.size();
		}
		avg_distances[i] = mean;
	}
	if (valid_distances == 0) {
		return std::make_tuple(std::make_shared<ccPointCloud>("pointCloud"),
			std::vector<size_t>());
	}
	double cloud_mean = std::accumulate(
		avg_distances.begin(), avg_distances.end(), 0.0,
		[](double const &x, double const &y) { return y > 0 ? x + y : x; });
	cloud_mean /= valid_distances;
	double sq_sum = std::inner_product(
		avg_distances.begin(), avg_distances.end(), avg_distances.begin(),
		0.0, [](double const &x, double const &y) { return x + y; },
		[cloud_mean](double const &x, double const &y) {
		return x > 0 ? (x - cloud_mean) * (y - cloud_mean) : 0;
	});
	// Bessel's correction
	double std_dev = std::sqrt(sq_sum / (valid_distances - 1));
	double distance_threshold = cloud_mean + std_ratio * std_dev;
	for (size_t i = 0; i < avg_distances.size(); i++) {
		if (avg_distances[i] > 0 && avg_distances[i] < distance_threshold) {
			indices.push_back(i);
		}
	}
	return std::make_tuple(selectByIndex(indices), indices);
}


namespace cloudViewer {

	namespace {
		using namespace geometry;

		Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d &A, double eval0) {
			Eigen::Vector3d row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
			Eigen::Vector3d row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
			Eigen::Vector3d row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
			Eigen::Vector3d r0xr1 = row0.cross(row1);
			Eigen::Vector3d r0xr2 = row0.cross(row2);
			Eigen::Vector3d r1xr2 = row1.cross(row2);
			double d0 = r0xr1.dot(r0xr1);
			double d1 = r0xr2.dot(r0xr2);
			double d2 = r1xr2.dot(r1xr2);

			double dmax = d0;
			int imax = 0;
			if (d1 > dmax) {
				dmax = d1;
				imax = 1;
			}
			if (d2 > dmax) {
				imax = 2;
			}

			if (imax == 0) {
				return r0xr1 / std::sqrt(d0);
			}
			else if (imax == 1) {
				return r0xr2 / std::sqrt(d1);
			}
			else {
				return r1xr2 / std::sqrt(d2);
			}
		}

		Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d &A,
			const Eigen::Vector3d &evec0,
			double eval1) {
			Eigen::Vector3d U, V;
			if (std::abs(evec0(0)) > std::abs(evec0(1))) {
				double inv_length =
					1 / std::sqrt(evec0(0) * evec0(0) + evec0(2) * evec0(2));
				U << -evec0(2) * inv_length, 0, evec0(0) * inv_length;
			}
			else {
				double inv_length =
					1 / std::sqrt(evec0(1) * evec0(1) + evec0(2) * evec0(2));
				U << 0, evec0(2) * inv_length, -evec0(1) * inv_length;
			}
			V = evec0.cross(U);

			Eigen::Vector3d AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
				A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
				A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

			Eigen::Vector3d AV = { A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
								  A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
								  A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2) };

			double m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
			double m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
			double m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

			double absM00 = std::abs(m00);
			double absM01 = std::abs(m01);
			double absM11 = std::abs(m11);
			double max_abs_comp;
			if (absM00 >= absM11) {
				max_abs_comp = std::max(absM00, absM01);
				if (max_abs_comp > 0) {
					if (absM00 >= absM01) {
						m01 /= m00;
						m00 = 1 / std::sqrt(1 + m01 * m01);
						m01 *= m00;
					}
					else {
						m00 /= m01;
						m01 = 1 / std::sqrt(1 + m00 * m00);
						m00 *= m01;
					}
					return m01 * U - m00 * V;
				}
				else {
					return U;
				}
			}
			else {
				max_abs_comp = std::max(absM11, absM01);
				if (max_abs_comp > 0) {
					if (absM11 >= absM01) {
						m01 /= m11;
						m11 = 1 / std::sqrt(1 + m01 * m01);
						m01 *= m11;
					}
					else {
						m11 /= m01;
						m01 = 1 / std::sqrt(1 + m11 * m11);
						m11 *= m01;
					}
					return m11 * U - m01 * V;
				}
				else {
					return U;
				}
			}
		}

		Eigen::Vector3d FastEigen3x3(Eigen::Matrix3d &A) {
			// Previous version based on:
			// https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
			// Current version based on
			// https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
			// which handles edge cases like points on a plane

			double max_coeff = A.maxCoeff();
			if (max_coeff == 0) {
				return Eigen::Vector3d::Zero();
			}
			A /= max_coeff;

			double norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
			if (norm > 0) {
				Eigen::Vector3d eval;
				Eigen::Vector3d evec0;
				Eigen::Vector3d evec1;
				Eigen::Vector3d evec2;

				double q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

				double b00 = A(0, 0) - q;
				double b11 = A(1, 1) - q;
				double b22 = A(2, 2) - q;

				double p =
					std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

				double c00 = b11 * b22 - A(1, 2) * A(1, 2);
				double c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
				double c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
				double det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

				double half_det = det * 0.5;
				half_det = std::min(std::max(half_det, -1.0), 1.0);

				double angle = std::acos(half_det) / (double)3;
				double const two_thirds_pi = 2.09439510239319549;
				double beta2 = std::cos(angle) * 2;
				double beta0 = std::cos(angle + two_thirds_pi) * 2;
				double beta1 = -(beta0 + beta2);

				eval(0) = q + p * beta0;
				eval(1) = q + p * beta1;
				eval(2) = q + p * beta2;

				if (half_det >= 0) {
					evec2 = ComputeEigenvector0(A, eval(2));
					if (eval(2) < eval(0) && eval(2) < eval(1)) {
						A *= max_coeff;
						return evec2;
					}
					evec1 = ComputeEigenvector1(A, evec2, eval(1));
					A *= max_coeff;
					if (eval(1) < eval(0) && eval(1) < eval(2)) {
						return evec1;
					}
					evec0 = evec1.cross(evec2);
					return evec0;
				}
				else {
					evec0 = ComputeEigenvector0(A, eval(0));
					if (eval(0) < eval(1) && eval(0) < eval(2)) {
						A *= max_coeff;
						return evec0;
					}
					evec1 = ComputeEigenvector1(A, evec0, eval(1));
					A *= max_coeff;
					if (eval(1) < eval(0) && eval(1) < eval(2)) {
						return evec1;
					}
					evec2 = evec0.cross(evec1);
					return evec2;
				}
			}
			else {
				A *= max_coeff;
				if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) {
					return Eigen::Vector3d(1, 0, 0);
				}
				else if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) {
					return Eigen::Vector3d(0, 1, 0);
				}
				else {
					return Eigen::Vector3d(0, 0, 1);
				}
			}
		}

		Eigen::Vector3d ComputeNormal(const ccPointCloud &cloud,
			const std::vector<int> &indices,
			bool fast_normal_computation) {
			if (indices.size() == 0) {
				return Eigen::Vector3d::Zero();
			}
			Eigen::Matrix3d covariance;
			Eigen::Matrix<double, 9, 1> cumulants;
			cumulants.setZero();
			for (size_t i = 0; i < indices.size(); i++) {
				const Eigen::Vector3d &point = cloud.getEigenPoint(indices[i]);
				cumulants(0) += point(0);
				cumulants(1) += point(1);
				cumulants(2) += point(2);
				cumulants(3) += point(0) * point(0);
				cumulants(4) += point(0) * point(1);
				cumulants(5) += point(0) * point(2);
				cumulants(6) += point(1) * point(1);
				cumulants(7) += point(1) * point(2);
				cumulants(8) += point(2) * point(2);
			}
			cumulants /= (double)indices.size();
			covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
			covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
			covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
			covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
			covariance(1, 0) = covariance(0, 1);
			covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
			covariance(2, 0) = covariance(0, 2);
			covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
			covariance(2, 1) = covariance(1, 2);

			if (fast_normal_computation) {
				return FastEigen3x3(covariance);
			}
			else {
				Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
				solver.compute(covariance, Eigen::ComputeEigenvectors);
				return solver.eigenvectors().col(0);
			}
		}

	}  // unnamed namespace
}

bool ccPointCloud::estimateNormals(
	const cloudViewer::geometry::KDTreeSearchParam &search_param /* = KDTreeSearchParamKNN()*/,
	bool fast_normal_computation /* = true */) {
	bool has_normal = hasNormals();
	if (!hasNormals()) {
		resizeTheNormsTable();
	}

	cloudViewer::geometry::KDTreeFlann kdtree;
	kdtree.SetGeometry(*this);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)size(); i++) {
		std::vector<int> indices;
		std::vector<double> distance2;
		Eigen::Vector3d normal;
		if (kdtree.Search(getEigenPoint(static_cast<size_t>(i)), search_param, indices, distance2) >= 3) {
			normal = cloudViewer::ComputeNormal(*this, indices, fast_normal_computation);
			if (normal.norm() == 0.0) {
				if (has_normal) {
					normal = CCVector3d::fromArray(getPointNormal(static_cast<unsigned int>(i)));
				}
				else {
					normal = Eigen::Vector3d(0.0, 0.0, 1.0);
				}
			}
			if (has_normal && normal.dot(CCVector3d::fromArray(getPointNormal(static_cast<unsigned int>(i)))) < 0.0) {
				normal *= -1.0;
			}
			setPointNormal(static_cast<unsigned int>(i), normal);
		}
		else {
			setPointNormal(static_cast<unsigned int>(i), CCVector3(0.0f, 0.0f, 1.0f));
		}
	}

	return true;
}

bool ccPointCloud::orientNormalsToAlignWithDirection(
	const Eigen::Vector3d &orientation_reference
/* = Eigen::Vector3d(0.0, 0.0, 1.0)*/) {
    if (!hasNormals()) {
		CVLib::utility::LogWarning(
			"[OrientNormalsToAlignWithDirection] No normals in the "
			"ccPointCloud. Call EstimateNormals() first.");
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)this->size(); i++) {
        auto &normal = getPointNormalPtr(static_cast<size_t>(i));
        if (normal.norm() == 0.0f) {
            normal = CCVector3::fromArray(orientation_reference);
		}
        else if (normal.dot(CCVector3::fromArray(orientation_reference)) < 0.0f) {
            normal *= -1.0f;
		}
	}
	return true;
}

bool ccPointCloud::orientNormalsTowardsCameraLocation(
	const Eigen::Vector3d &camera_location /* = Eigen::Vector3d::Zero()*/) {
	if (hasNormals() == false) {
		CVLib::utility::LogWarning(
			"[OrientNormalsTowardsCameraLocation] No normals in the "
			"ccPointCloud. Call EstimateNormals() first.");
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)this->size(); i++) {
		Eigen::Vector3d orientation_reference =
			camera_location - getEigenPoint(static_cast<size_t>(i));
        auto &normal = getPointNormalPtr(static_cast<size_t>(i));
        if (normal.norm() == 0.0f) {
			normal = orientation_reference;
            if (normal.norm() == 0.0f) {
                normal = CCVector3(0.0f, 0.0f, 1.0f);
			}
			else {
				normal.normalize();
			}
		}
        else if (normal.dot(orientation_reference) < 0.0f) {
            normal *= -1.0f;
		}
	}
	return true;
}

void ccPointCloud::orientNormalsConsistentTangentPlane(size_t k) {
    if (!hasNormals()) {
        utility::LogError(
                "[orientNormalsConsistentTangentPlane] No normals in the "
                "ccPointCloud. Call estimateNormals() first.");
    }

    // Create Riemannian graph (Euclidian MST + kNN)
    // Euclidian MST is subgraph of Delaunay triangulation
    std::shared_ptr<cloudViewer::geometry::TetraMesh> delaunay_mesh;
    std::vector<size_t> pt_map;
    std::tie(delaunay_mesh, pt_map) = cloudViewer::geometry::TetraMesh::CreateFromPointCloud(*this);
    std::vector<cloudViewer::WeightedEdge> delaunay_graph;
    std::unordered_set<size_t> graph_edges;
    auto EdgeIndex = [&](size_t v0, size_t v1) -> size_t {
        return std::min(v0, v1) * this->size() + std::max(v0, v1);
    };
    auto AddEdgeToDelaunayGraph = [&](size_t v0, size_t v1) {
        v0 = pt_map[v0];
        v1 = pt_map[v1];
        size_t edge = EdgeIndex(v0, v1);
        if (graph_edges.count(edge) == 0) {
            double dist = (getEigenPoint(v0) - getEigenPoint(v1)).squaredNorm();
            delaunay_graph.push_back(cloudViewer::WeightedEdge(v0, v1, dist));
            graph_edges.insert(edge);
        }
    };
    for (const Eigen::Vector4i &tetra : delaunay_mesh->tetras_) {
        AddEdgeToDelaunayGraph(tetra[0], tetra[1]);
        AddEdgeToDelaunayGraph(tetra[0], tetra[2]);
        AddEdgeToDelaunayGraph(tetra[0], tetra[3]);
        AddEdgeToDelaunayGraph(tetra[1], tetra[2]);
        AddEdgeToDelaunayGraph(tetra[1], tetra[3]);
        AddEdgeToDelaunayGraph(tetra[2], tetra[3]);
    }

    std::vector<cloudViewer::WeightedEdge> mst =
            cloudViewer::Kruskal(delaunay_graph, this->size());

    auto NormalWeight = [&](size_t v0, size_t v1) -> double {
        return 1.0 - std::abs(getEigenNormal(v0).dot(getEigenNormal(v1)));
    };
    for (auto &edge : mst) {
        edge.weight_ = NormalWeight(edge.v0_, edge.v1_);
    }

    // Add k nearest neighbors to Riemannian graph
    cloudViewer::geometry::KDTreeFlann kdtree(*this);
    for (size_t v0 = 0; v0 < this->size(); ++v0) {
        std::vector<int> neighbors;
        std::vector<double> dists2;
        kdtree.SearchKNN(getEigenPoint(v0), int(k), neighbors, dists2);
        for (size_t vidx1 = 0; vidx1 < neighbors.size(); ++vidx1) {
            size_t v1 = size_t(neighbors[vidx1]);
            if (v0 == v1) {
                continue;
            }
            size_t edge = EdgeIndex(v0, v1);
            if (graph_edges.count(edge) == 0) {
                double weight = NormalWeight(v0, v1);
                mst.push_back(cloudViewer::WeightedEdge(v0, v1, weight));
                graph_edges.insert(edge);
            }
        }
    }

    // extract MST from Riemannian graph
    mst = cloudViewer::Kruskal(mst, this->size());

    // convert list of edges to graph
    std::vector<std::unordered_set<size_t>> mst_graph(this->size());
    for (const auto &edge : mst) {
        size_t v0 = edge.v0_;
        size_t v1 = edge.v1_;
        mst_graph[v0].insert(v1);
        mst_graph[v1].insert(v0);
    }

    // find start node for tree traversal
    // init with node that maximizes z
    double max_z = std::numeric_limits<double>::lowest();
    size_t v0;
    for (size_t vidx = 0; vidx < this->size(); ++vidx) {
        const Eigen::Vector3d &v = getEigenPoint(vidx);
        if (v(2) > max_z) {
            max_z = v(2);
            v0 = vidx;
        }
    }

    // traverse MST and orient normals consistently
    std::queue<size_t> traversal_queue;
    std::vector<bool> visited(this->size(), false);
    traversal_queue.push(v0);
    auto TestAndOrientNormal = [&](const CCVector3 &n0, CCVector3 &n1) {
        if (n0.dot(n1) < 0) {
            n1 *= -1;
        }
    };
    TestAndOrientNormal(CCVector3(0.0f, 0.0f, 1.0f), getPointNormalPtr(v0));
    while (!traversal_queue.empty()) {
        v0 = traversal_queue.front();
        traversal_queue.pop();
        visited[v0] = true;
        for (size_t v1 : mst_graph[v0]) {
            if (!visited[v1]) {
                traversal_queue.push(v1);
                TestAndOrientNormal(getPointNormalPtr(v0), getPointNormalPtr(v1));
            }
        }
    }
}

std::vector<double> ccPointCloud::computeMahalanobisDistance() const
{
	std::vector<double> mahalanobis(size());
	Eigen::Vector3d mean;
	Eigen::Matrix3d covariance;
	std::tie(mean, covariance) = computeMeanAndCovariance();

	Eigen::Matrix3d cov_inv = covariance.inverse();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)size(); i++) {
		Eigen::Vector3d p = CCVector3d::fromArray(*getPoint(static_cast<unsigned int>(i))) - mean;
		mahalanobis[i] = std::sqrt(p.transpose() * cov_inv * p);
	}
	return mahalanobis;
}

std::vector<double> ccPointCloud::computeNearestNeighborDistance() const
{
	std::vector<double> nn_dis(size());
	cloudViewer::geometry::KDTreeFlann kdtree(*this);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)size(); i++) {
		std::vector<int> indices(2);
		std::vector<double> dists(2);
		if (kdtree.SearchKNN(
			CCVector3d::fromArray(*getPoint(static_cast<unsigned int>(i))),
			2, indices, dists) <= 1)
		{
			utility::LogDebug(
				"[ComputePointCloudNearestNeighborDistance] Found a point without neighbors.");
			nn_dis[i] = 0.0;
		}
		else {
			nn_dis[i] = std::sqrt(dists[1]);
		}
	}
	return nn_dis;
}

double ccPointCloud::computeResolution() const
{
	std::vector<double> nn_dis = computeNearestNeighborDistance();
	return std::accumulate(std::begin(nn_dis), std::end(nn_dis), 0.0) / nn_dis.size();
}

std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
ccPointCloud::computeConvexHull() const
{
	return cloudViewer::utility::Qhull::ComputeConvexHull(m_points);
}

std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
ccPointCloud::hiddenPointRemoval(const Eigen::Vector3d &camera_location, const double radius) const
{
	if (radius <= 0) {
		utility::LogError(
			"[ccPointCloud::hiddenPointRemoval] radius must be larger than zero.");
		return std::make_tuple(std::make_shared<ccMesh>(nullptr),
			std::vector<size_t>());
	}

	// perform spherical projection
	std::vector<CCVector3> spherical_projection;
	for (unsigned int pidx = 0; pidx < size(); ++pidx) {
		CCVector3 projected_point = *getPoint(pidx) - camera_location;
		double norm = projected_point.norm();
		spherical_projection.push_back(
			projected_point + 2 * (radius - norm) * projected_point / norm);
	}

	// add origin
	size_t origin_pidx = spherical_projection.size();
	spherical_projection.push_back(CCVector3(0, 0, 0));

	// calculate convex hull of spherical projection
	std::shared_ptr<ccMesh> visible_mesh;
	std::vector<size_t> pt_map;
	std::tie(visible_mesh, pt_map) =
		cloudViewer::utility::Qhull::ComputeConvexHull(spherical_projection);

	// reassign original points to mesh
	size_t origin_vidx = pt_map.size();
	for (size_t vidx = 0; vidx < pt_map.size(); vidx++) {
		size_t pidx = pt_map[vidx];

		if (pidx < size())
		{
			visible_mesh->setVertice(vidx, getEigenPoint(pidx));
		}
		
		if (pidx == origin_pidx) {
			origin_vidx = vidx;
			visible_mesh->setVertice(vidx, camera_location);
		}
	}

	// erase origin if part of mesh
	if (origin_vidx < visible_mesh->getVerticeSize()) {
		visible_mesh->getAssociatedCloud()->removePoints(origin_vidx);
		pt_map.erase(pt_map.begin() + origin_vidx);
		for (size_t tidx = visible_mesh->size(); tidx-- > 0;) {
			
			auto& tsi = visible_mesh->getTrianglesPtr()->getValue(tidx);
			if (tsi.i1 == (unsigned int)origin_vidx ||
				tsi.i2 == (unsigned int)origin_vidx ||
				tsi.i3 == (unsigned int)origin_vidx)
			{
				visible_mesh->removeTriangles(tidx);
			}
			else {
				if (tsi.i1 > (int)origin_vidx)
					tsi.i1 -= 1;
				if (tsi.i2 > (int)origin_vidx)
					tsi.i2 -= 1;
				if (tsi.i3 > (int)origin_vidx)
					tsi.i3 -= 1;
			}
		}
	}
	return std::make_tuple(visible_mesh, pt_map);
}

ccPointCloud& ccPointCloud::paintUniformColor(const Eigen::Vector3d & color)
{
	setRGBColor(ecvColor::Rgb::FromEigen(color));
	return (*this);
}
