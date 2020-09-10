// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#include <Image.h>
#include <VoxelGrid.h>

#include "Camera/PinholeCameraParameters.h"

namespace cloudViewer {
namespace geometry {
	using namespace CVLib;
	VoxelGrid &VoxelGrid::CarveDepthMap(
		const Image &depth_map,
		const camera::PinholeCameraParameters &camera_parameter,
		bool keep_voxels_outside_image) {
		if (depth_map.height_ != camera_parameter.intrinsic_.height_ ||
			depth_map.width_ != camera_parameter.intrinsic_.width_) {
			utility::LogError(
				"[VoxelGrid] provided depth_map dimensions are not compatible "
				"with the provided camera_parameters");
		}

		auto rot = camera_parameter.extrinsic_.block<3, 3>(0, 0);
		auto trans = camera_parameter.extrinsic_.block<3, 1>(0, 3);
		auto intrinsic = camera_parameter.intrinsic_.intrinsic_matrix_;

		// get for each voxel if it projects to a valid pixel and check if the voxel
		// depth is behind the depth of the depth map at the projected pixel.
		for (auto it = voxels_.begin(); it != voxels_.end();) {
			bool carve = true;
			const geometry::Voxel &voxel = it->second;
			auto pts = GetVoxelBoundingPoints(voxel.grid_index_);
			for (auto &x : pts) {
				auto x_trans = rot * x + trans;
				auto uvz = intrinsic * x_trans;
				double z = uvz(2);
				double u = uvz(0) / z;
				double v = uvz(1) / z;
				double d;
				bool within_boundary;
				std::tie(within_boundary, d) = depth_map.FloatValueAt(u, v);
				if ((!within_boundary && keep_voxels_outside_image) ||
					(within_boundary && d > 0 && z >= d)) {
					carve = false;
					break;
				}
			}
			if (carve)
				it = voxels_.erase(it);
			else
				it++;
		}
		return *this;
	}

	VoxelGrid &VoxelGrid::CarveSilhouette(
		const Image &silhouette_mask,
		const camera::PinholeCameraParameters &camera_parameter,
		bool keep_voxels_outside_image) {
		if (silhouette_mask.height_ != camera_parameter.intrinsic_.height_ ||
			silhouette_mask.width_ != camera_parameter.intrinsic_.width_) {
			utility::LogError(
				"[VoxelGrid] provided silhouette_mask dimensions are not "
				"compatible with the provided camera_parameters");
		}

		auto rot = camera_parameter.extrinsic_.block<3, 3>(0, 0);
		auto trans = camera_parameter.extrinsic_.block<3, 1>(0, 3);
		auto intrinsic = camera_parameter.intrinsic_.intrinsic_matrix_;

		// get for each voxel if it projects to a valid pixel and check if the pixel
		// is set (>0).
		for (auto it = voxels_.begin(); it != voxels_.end();) {
			bool carve = true;
			const geometry::Voxel &voxel = it->second;
			auto pts = GetVoxelBoundingPoints(voxel.grid_index_);
			for (auto &x : pts) {
				auto x_trans = rot * x + trans;
				auto uvz = intrinsic * x_trans;
				double z = uvz(2);
				double u = uvz(0) / z;
				double v = uvz(1) / z;
				double d;
				bool within_boundary;
				std::tie(within_boundary, d) = silhouette_mask.FloatValueAt(u, v);
				if ((!within_boundary && keep_voxels_outside_image) ||
					(within_boundary && d > 0)) {
					carve = false;
					break;
				}
			}
			if (carve)
				it = voxels_.erase(it);
			else
				it++;
		}
		return *this;
	}

}  // namespace geometry
}  // namespace cloudViewer
