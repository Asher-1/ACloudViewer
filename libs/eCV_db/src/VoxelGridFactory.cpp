// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
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

#include <numeric>
#include <unordered_map>

#include <Console.h>
#include <Helper.h>
#include <IntersectionTest.h>

#include "ecvPointCloud.h"
#include "ecvMesh.h"
#include "VoxelGrid.h"

namespace cloudViewer {
namespace geometry {
	using namespace cloudViewer;

std::shared_ptr<VoxelGrid> VoxelGrid::CreateDense(const Eigen::Vector3d &origin,
                                                  const Eigen::Vector3d &color,
                                                  double voxel_size,
                                                  double width,
                                                  double height,
                                                  double depth) {
    auto output = cloudViewer::make_shared<VoxelGrid>();
    int num_w = int(std::round(width / voxel_size));
    int num_h = int(std::round(height / voxel_size));
    int num_d = int(std::round(depth / voxel_size));
    output->origin_ = origin;
    output->voxel_size_ = voxel_size;
    for (int widx = 0; widx < num_w; widx++) {
        for (int hidx = 0; hidx < num_h; hidx++) {
            for (int didx = 0; didx < num_d; didx++) {
                Eigen::Vector3i grid_index(widx, hidx, didx);
                output->AddVoxel(geometry::Voxel(grid_index, color));
            }
        }
    }
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromPointCloudWithinBounds(
        const ccPointCloud &input,
        double voxel_size,
        const Eigen::Vector3d &min_bound,
        const Eigen::Vector3d &max_bound) {
    auto output = cloudViewer::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("[VoxelGridFromPointCloud] voxel_size <= 0.");
    }

    if (voxel_size * std::numeric_limits<int>::max() <
        (max_bound - min_bound).maxCoeff()) {
        utility::LogError("[VoxelGridFromPointCloud] voxel_size is too small.");
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = min_bound;
    std::unordered_map<Eigen::Vector3i, AvgColorVoxel,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    bool has_colors = input.hasColors();
    for (size_t i = 0; i < input.size(); i++) {
		Eigen::Vector3d p = input.getEigenPoint(i); // must reserve a temp variable
        ref_coord = (p - min_bound) / voxel_size;
        voxel_index <<	int(floor(ref_coord(0))), 
						int(floor(ref_coord(1))),
						int(floor(ref_coord(2)));
        if (has_colors) {
            voxelindex_to_accpoint[voxel_index].Add(
				voxel_index, input.getEigenColor(i));
        } else {
            voxelindex_to_accpoint[voxel_index].Add(voxel_index);
        }
    }
    for (auto accpoint : voxelindex_to_accpoint) {
        const Eigen::Vector3i &grid_index = accpoint.second.GetVoxelIndex();
        const Eigen::Vector3d &color =
                has_colors ? accpoint.second.GetAverageColor()
                           : Eigen::Vector3d(0, 0, 0);
        output->AddVoxel(geometry::Voxel(grid_index, color));
    }
    utility::LogDebug(
            "Pointcloud is voxelized from {:d} points to {:d} voxels.",
            (int)input.size(), (int)output->voxels_.size());
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromPointCloud(
        const ccPointCloud &input, double voxel_size) {
    Eigen::Vector3d voxel_size3(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d min_bound = input.getMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d max_bound = input.getMaxBound() + voxel_size3 * 0.5;
    return CreateFromPointCloudWithinBounds(input, voxel_size, min_bound, max_bound);
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromTriangleMeshWithinBounds(
        const ccMesh &input,
        double voxel_size,
        const Eigen::Vector3d &min_bound,
        const Eigen::Vector3d &max_bound) {
    auto output = cloudViewer::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("[CreateFromTriangleMesh] voxel_size <= 0.");
    }

    if (voxel_size * std::numeric_limits<int>::max() <
        (max_bound - min_bound).maxCoeff()) {
        utility::LogError("[CreateFromTriangleMesh] voxel_size is too small.");
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = min_bound;

    Eigen::Vector3d grid_size = max_bound - min_bound;
    int num_w = int(std::round(grid_size(0) / voxel_size));
    int num_h = int(std::round(grid_size(1) / voxel_size));
    int num_d = int(std::round(grid_size(2) / voxel_size));
    const Eigen::Vector3d box_half_size(voxel_size / 2, voxel_size / 2,
                                        voxel_size / 2);
    for (int widx = 0; widx < num_w; widx++) {
        for (int hidx = 0; hidx < num_h; hidx++) {
            for (int didx = 0; didx < num_d; didx++) {
                const Eigen::Vector3d box_center =
                        min_bound + Eigen::Vector3d(widx, hidx, didx) * voxel_size;
				unsigned int triNum = input.size();
				for (unsigned int i = 0; i < triNum; ++i) {
					Eigen::Vector3d v0, v1, v2;
					input.getTriangleVertices(i, v0.data(), v1.data(), v2.data());
                    if (utility::IntersectionTest::TriangleAABB(
                                box_center, box_half_size, v0, v1, v2)) {
                        Eigen::Vector3i grid_index(widx, hidx, didx);
                        output->AddVoxel(geometry::Voxel(grid_index));
                        break;
                    }
                }
            }
        }
    }

    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromTriangleMesh(
        const ccMesh &input, double voxel_size) {
    Eigen::Vector3d voxel_size3(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d min_bound = input.getMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d max_bound = input.getMaxBound() + voxel_size3 * 0.5;
    return CreateFromTriangleMeshWithinBounds(input, voxel_size, min_bound, max_bound);
}

}  // namespace geometry
}  // namespace cloudViewer
