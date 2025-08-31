// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Helper.h>
#include <IntersectionTest.h>
#include <Logging.h>

#include <numeric>
#include <unordered_map>

#include "VoxelGrid.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"

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
        const Eigen::Vector3d &max_bound,
        VoxelGrid::VoxelPoolingMode pooling_mode) {
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
                       utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    bool has_colors = input.hasColors();
    for (size_t i = 0; i < input.size(); i++) {
        Eigen::Vector3d p =
                input.getEigenPoint(i);  // must reserve a temp variable
        ref_coord = (p - min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
                int(floor(ref_coord(2)));
        if (has_colors) {
            voxelindex_to_accpoint[voxel_index].Add(voxel_index,
                                                    input.getEigenColor(i));
        } else {
            voxelindex_to_accpoint[voxel_index].Add(voxel_index);
        }
    }
    for (auto accpoint : voxelindex_to_accpoint) {
        const Eigen::Vector3i &grid_index = accpoint.second.GetVoxelIndex();
        const Eigen::Vector3d &color =
                has_colors ? (pooling_mode == VoxelPoolingMode::AVG
                                      ? accpoint.second.GetAverageColor()
                              : pooling_mode == VoxelPoolingMode::MIN
                                      ? accpoint.second.GetMinColor()
                              : pooling_mode == VoxelPoolingMode::MAX
                                      ? accpoint.second.GetMaxColor()
                              : pooling_mode == VoxelPoolingMode::SUM
                                      ? accpoint.second.GetSumColor()
                                      : Eigen::Vector3d::Zero())
                           : Eigen::Vector3d::Zero();
        output->AddVoxel(geometry::Voxel(grid_index, color));
    }
    utility::LogDebug(
            "Pointcloud is voxelized from {:d} points to {:d} voxels.",
            (int)input.size(), (int)output->voxels_.size());
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromPointCloud(
        const ccPointCloud &input,
        double voxel_size,
        VoxelGrid::VoxelPoolingMode pooling_mode) {
    Eigen::Vector3d voxel_size3(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d min_bound = input.getMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d max_bound = input.getMaxBound() + voxel_size3 * 0.5;
    return CreateFromPointCloudWithinBounds(input, voxel_size, min_bound,
                                            max_bound, pooling_mode);
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromTriangleMeshWithinBounds(
        const ccMesh &input,
        double voxel_size,
        const Eigen::Vector3d &min_bound,
        const Eigen::Vector3d &max_bound) {
    auto output = cloudViewer::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("voxel_size <= 0.");
    }

    if (voxel_size * std::numeric_limits<int>::max() <
        (max_bound - min_bound).maxCoeff()) {
        utility::LogError("voxel_size is too small.");
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = min_bound;

    const Eigen::Vector3d box_half_size(voxel_size / 2, voxel_size / 2,
                                        voxel_size / 2);
    unsigned int triNum = input.size();
    for (unsigned int i = 0; i < triNum; ++i) {
        Eigen::Vector3d v0, v1, v2;
        input.getTriangleVertices(i, v0.data(), v1.data(), v2.data());
        double minx, miny, minz, maxx, maxy, maxz;
        int num_w, num_h, num_d, inix, iniy, iniz;
        minx = std::min(v0[0], std::min(v1[0], v2[0]));
        miny = std::min(v0[1], std::min(v1[1], v2[1]));
        minz = std::min(v0[2], std::min(v1[2], v2[2]));
        maxx = std::max(v0[0], std::max(v1[0], v2[0]));
        maxy = std::max(v0[1], std::max(v1[1], v2[1]));
        maxz = std::max(v0[2], std::max(v1[2], v2[2]));
        inix = static_cast<int>(std::floor((minx - min_bound[0]) / voxel_size));
        iniy = static_cast<int>(std::floor((miny - min_bound[1]) / voxel_size));
        iniz = static_cast<int>(std::floor((minz - min_bound[2]) / voxel_size));
        num_w = static_cast<int>((std::round((maxx - minx) / voxel_size))) + 2;
        num_h = static_cast<int>((std::round((maxy - miny) / voxel_size))) + 2;
        num_d = static_cast<int>((std::round((maxz - minz) / voxel_size))) + 2;
        for (int widx = inix; widx < inix + num_w; widx++) {
            for (int hidx = iniy; hidx < iniy + num_h; hidx++) {
                for (int didx = iniz; didx < iniz + num_d; didx++) {
                    const Eigen::Vector3d box_center =
                            min_bound + box_half_size +
                            Eigen::Vector3d(widx, hidx, didx) * voxel_size;
                    if (utility::IntersectionTest::TriangleAABB(
                                box_center, box_half_size, v0, v1, v2)) {
                        Eigen::Vector3i grid_index(widx, hidx, didx);
                        output->AddVoxel(geometry::Voxel(grid_index));
                        // Don't `break` here, since a triangle can span
                        // across multiple voxels.
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
    return CreateFromTriangleMeshWithinBounds(input, voxel_size, min_bound,
                                              max_bound);
}

}  // namespace geometry
}  // namespace cloudViewer
